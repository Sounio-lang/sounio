#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

source_file=scripts/research/cs6_plucker_cocycle_probe.cpp
leaf_verifier=scripts/research/cs6_plucker_cocycle_verify.py
runner=scripts/research/cs6_plucker_cocycle_run.py
coordinate_manifest=scripts/research/cs6_plucker_cocycle_coordinates_v1.tsv
retained_verifier=scripts/research/cs6_plucker_cocycle_retained_verify.py
retained=scripts/research/receipts/cs6_plucker_cocycle_retained_53_v1

manifest_value() {
  local key=$1
  awk -F= -v key="$key" '$1 == key {print $2}' "$retained/run-manifest.txt"
}

python3 -m py_compile "$leaf_verifier" "$runner" "$retained_verifier"
python3 "$retained_verifier" "$retained"

test "$(sha256sum "$source_file" | awk '{print $1}')" = "$(manifest_value SOURCE_SHA256)"
test "$(sha256sum "$leaf_verifier" | awk '{print $1}')" = "$(manifest_value VERIFIER_SHA256)"
test "$(sha256sum "$runner" | awk '{print $1}')" = "$(manifest_value RUNNER_SHA256)"
test "$(sha256sum "$coordinate_manifest" | awk '{print $1}')" = "$(manifest_value COORDINATE_MANIFEST_SHA256)"

sample_id=$(awk -F '\t' 'NR > 1 && $8 == "true" {print $1; exit}' "$retained/leaves.tsv")
test -n "$sample_id"
sample_input="$retained/inputs/$sample_id.txt"
sample_receipt="$retained/receipts/$sample_id.txt"
sample_challenge=$(awk -F '\t' -v id="$sample_id" '$1 == id {print $18}' "$retained/leaves.tsv")
source_sha=$(manifest_value SOURCE_SHA256)

mutation_dir=$(mktemp -d)
trap 'rm -rf "$mutation_dir"' EXIT
cp "$sample_receipt" "$mutation_dir/chart-mutated.txt"
python3 - "$mutation_dir/chart-mutated.txt" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
raw = path.read_text(encoding="ascii")
pattern = r"(HOMOGENEOUS_EVENT1_RAY0 ELIGIBLE=true CHART=)(X|Y|PLUS|MINUS)"
match = re.search(pattern, raw)
if match is None:
    raise SystemExit("sample has no eligible event-1 chart")
replacement = "Y" if match.group(2) != "Y" else "X"
path.write_text(raw[: match.start(2)] + replacement + raw[match.end(2) :], encoding="ascii")
PY
if python3 "$leaf_verifier" "$mutation_dir/chart-mutated.txt" \
  --source-sha "$source_sha" --input "$sample_input" \
  --challenge "$sample_challenge" --require-probe \
  >"$mutation_dir/chart.stdout" 2>"$mutation_dir/chart.stderr"; then
  echo "chart mutation escaped the leaf verifier" >&2
  exit 1
fi

cp -a "$retained" "$mutation_dir/retained-mutated"
printf '\n' >>"$mutation_dir/retained-mutated/summary.txt"
if python3 "$retained_verifier" "$mutation_dir/retained-mutated" \
  >"$mutation_dir/retained.stdout" 2>"$mutation_dir/retained.stderr"; then
  echo "retained-file mutation escaped the files index" >&2
  exit 1
fi

for semantic_mutation in root-status leaf-method extra-file path-symlink; do
  semantic_dir="$mutation_dir/retained-$semantic_mutation"
  cp -a "$retained" "$semantic_dir"
  python3 - "$semantic_dir" "$semantic_mutation" <<'PY'
from pathlib import Path
import hashlib
import sys

root = Path(sys.argv[1])
mutation = sys.argv[2]
leaves_path = root / "leaves.tsv"
lines = leaves_path.read_text(encoding="ascii").splitlines()
header = lines[0].split("\t")
column = {name: index for index, name in enumerate(header)}
if mutation == "extra-file":
    (root / "unexpected.txt").write_text("unexpected\n", encoding="ascii")
elif mutation == "path-symlink":
    (root / "unexpected-tree").symlink_to("/etc", target_is_directory=True)
else:
    row_index = 1 if mutation == "root-status" else 2
    row = lines[row_index].split("\t")
    if mutation == "root-status":
        row[column["STATUS"]] = "COMPUTATION_UNRESOLVED_TIMEOUT"
    else:
        row[column["METHOD"]] = "AFFINE"
    lines[row_index] = "\t".join(row)
    leaves_path.write_text("\n".join(lines) + "\n", encoding="ascii")

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

run_manifest = root / "run-manifest.txt"
run_lines = run_manifest.read_text(encoding="ascii").splitlines()
if mutation != "extra-file":
    run_lines = [
        f"LEAVES_TSV_SHA256={digest(leaves_path)}"
        if line.startswith("LEAVES_TSV_SHA256=") else line
        for line in run_lines
    ]
    run_manifest.write_text("\n".join(run_lines) + "\n", encoding="ascii")

excluded = {"files.sha256", "retained-manifest.txt"}
files = sorted(
    path for path in root.rglob("*")
    if path.is_file() and path.relative_to(root).as_posix() not in excluded
)
index = root / "files.sha256"
index.write_text(
    "".join(f"{digest(path)}  {path.relative_to(root).as_posix()}\n" for path in files),
    encoding="ascii",
)
retained_manifest = root / "retained-manifest.txt"
retained_lines = retained_manifest.read_text(encoding="ascii").splitlines()
replacements = {
    "FILES_INDEX_SHA256": digest(index),
    "FILE_COUNT": str(len(files)),
    "RUN_MANIFEST_SHA256": digest(run_manifest),
}
retained_lines = [
    f"{key}={replacements.get(key, value)}"
    for key, value in (line.split("=", 1) for line in retained_lines)
]
retained_manifest.write_text("\n".join(retained_lines) + "\n", encoding="ascii")
PY
  if python3 "$retained_verifier" "$semantic_dir" \
    >"$mutation_dir/$semantic_mutation.stdout" \
    2>"$mutation_dir/$semantic_mutation.stderr"; then
    echo "coordinated $semantic_mutation mutation escaped retained verification" >&2
    exit 1
  fi
done

if [[ ${CS6_PLUCKER_COCYCLE_REPLAY:-0} == 1 ]]; then
  capd_config=${CS6_CAPD_CONFIG:-/tmp/capd-build/bin/capd-config}
  test -x "$capd_config"
  test "$($capd_config --modversion)" = 5.3.0
  cxx=${CXX:-g++}
  fresh_source_sha=$(sha256sum "$source_file" | awk '{print $1}')
  binary="$mutation_dir/worker"
  # shellcheck disable=SC2046
  "$cxx" -std=c++17 $($capd_config --cflags) -O0 \
    -DCS6_WORKER_SOURCE_SHA256=\"$fresh_source_sha\" \
    "$source_file" -o "$binary" $($capd_config --libs) \
    >"$mutation_dir/compile.stdout" 2>"$mutation_dir/compile.stderr"
  read -r u_depth u_index s_depth s_index < <(
    awk -F '\t' -v id="$sample_id" '$1 == id {print $2, $3, $4, $5}' "$retained/leaves.tsv"
  )
  input_sha=$(sha256sum "$sample_input" | awk '{print $1}')
  "$binary" "$u_depth" "$u_index" "$s_depth" "$s_index" \
    "$input_sha" "$sample_challenge" >"$mutation_dir/fresh-receipt.txt"
  python3 "$leaf_verifier" "$mutation_dir/fresh-receipt.txt" \
    --source-sha "$fresh_source_sha" --input "$sample_input" \
    --challenge "$sample_challenge" --self-test-mutations --require-probe \
    >"$mutation_dir/fresh-verification.txt"
  grep -qx 'MUTATION_TESTS=76' "$mutation_dir/fresh-verification.txt"
  grep -qx 'MUTATIONS_REJECTED=76' "$mutation_dir/fresh-verification.txt"
  grep -qx 'PROBE_PASS=true' "$mutation_dir/fresh-verification.txt"
fi

echo "cs6 plucker cocycle gate: PASS"
echo "coordinate_count=53"
echo "probe_valid_count=52"
echo "homogeneous_certified_count=0"
echo "promotion_eligible=false"
