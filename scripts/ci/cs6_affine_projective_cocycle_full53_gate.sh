#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

source_file=scripts/research/cs6_affine_projective_cocycle_full53_probe.cpp
leaf_verifier=scripts/research/cs6_affine_projective_cocycle_full53_verify.py
runner=scripts/research/cs6_affine_projective_cocycle_full53_run.py
coordinate_manifest=scripts/research/cs6_affine_projective_cocycle_full53_coordinates_v1.tsv
retained_verifier=scripts/research/cs6_affine_projective_cocycle_full53_retained_verify.py
retained=scripts/research/receipts/cs6_affine_projective_cocycle_full53_retained_53_v1
parent=scripts/research/receipts/cs6_plucker_cocycle_retained_53_v1
predeclaration_report=docs/research/cs6_affine_projective_cocycle_2026-08-01.md

predeclared_commit=9dcf1fca964d7e54e1109f9210689809666b2a54
predeclaration_report_sha=74d8f596c9258eab49775192e8b244e032dfb5e3481a23804075c66ff316618c
contract_frozen_commit=58905019754cf66f077a0db228f1a99a4a7612eb
frozen_manifest_sha=61b2b0649983a332b5abb530443a3ff14a19e62514ef9b1d3175d8e9a6bbfd9c
parent_run_manifest_sha=21f0ca224fc79d1cfc815f203100e091bf27dbb070cb3a2c69f6056b7a6fee27
parent_files_index_sha=740ae39070ff90604048c37d25e2f0de954be9ea55e020a529e094699a82802d
parent_coordinates_sha=6169dd7705ca4f01180f65d13d620320845684f0a9fc28411c881cfae7e4f2d8
parent_leaves_sha=98c639a616ac640f1983209530f8fe30d769a0e4b0136665201c9bc57320e37f
parent_affine_obligation_sha=9f36931e672aba1b9735c45eef286fbca97da67b911c64daa0b3b8c8affecd6c
root_id=U00-0000000000_S00-0000000000
replay_xy_id=U08-0000000064_S08-0000000192
replay_minus_e2_positive_id=U13-0000005120_S14-0000010240

fail() {
  echo "cs6 full53 gate error: $*" >&2
  exit 1
}

digest() {
  sha256sum "$1" | awk '{print $1}'
}

manifest_value() {
  local key=$1
  awk -F= -v key="$key" '$1 == key {print $2}' "$retained/run-manifest.txt"
}

retained_value() {
  local key=$1
  awk -F= -v key="$key" '$1 == key {print $2}' "$retained/retained-manifest.txt"
}

require_equal() {
  local actual=$1
  local expected=$2
  local label=$3
  [[ $actual == "$expected" ]] || fail "$label mismatch"
}

for required in \
  "$source_file" "$leaf_verifier" "$runner" "$coordinate_manifest" \
  "$retained_verifier" "$retained" "$parent"; do
  [[ -e $required ]] || fail "missing required path: $required"
done

python3 -m py_compile "$leaf_verifier" "$runner" "$retained_verifier"
python3 "$retained_verifier" "$retained"

require_equal "$(digest "$source_file")" "$(manifest_value SOURCE_SHA256)" \
  "worker source hash"
require_equal "$(digest "$leaf_verifier")" "$(manifest_value VERIFIER_SHA256)" \
  "leaf verifier hash"
require_equal "$(digest "$runner")" "$(manifest_value RUNNER_SHA256)" \
  "runner hash"
require_equal "$(digest "$coordinate_manifest")" \
  "$(manifest_value COORDINATE_MANIFEST_SHA256)" "coordinate manifest hash"
require_equal "$(digest "$retained_verifier")" \
  "$(retained_value RETAINED_VERIFIER_SHA256)" "retained verifier hash"

require_equal "$(digest "$coordinate_manifest")" "$frozen_manifest_sha" \
  "frozen coordinate manifest"
require_equal "$(manifest_value CONTRACT_FROZEN_IN_COMMIT)" \
  "$contract_frozen_commit" "contract freeze commit"
require_equal "$(manifest_value FROZEN_MANIFEST_SHA256)" \
  "$frozen_manifest_sha" "run-manifest frozen contract hash"
require_equal \
  "$(git show "${contract_frozen_commit}:${coordinate_manifest}" | sha256sum | awk '{print $1}')" \
  "$frozen_manifest_sha" "contract manifest replay"

require_equal "$(manifest_value PREDECLARED_IN_COMMIT)" "$predeclared_commit" \
  "predeclaration commit"
require_equal "$(manifest_value PREDECLARATION_REPORT_SHA256)" \
  "$predeclaration_report_sha" "predeclaration report anchor"
require_equal \
  "$(git show "${predeclared_commit}:${predeclaration_report}" | sha256sum | awk '{print $1}')" \
  "$predeclaration_report_sha" "predeclaration report replay"
require_equal "$(manifest_value PARENT_RUN_MANIFEST_SHA256)" \
  "$parent_run_manifest_sha" "parent run-manifest anchor"
require_equal "$(manifest_value PARENT_FILES_INDEX_SHA256)" \
  "$parent_files_index_sha" "parent files index anchor"
require_equal "$(manifest_value PARENT_COORDINATES_SHA256)" \
  "$parent_coordinates_sha" "parent coordinates anchor"
require_equal "$(manifest_value PARENT_LEAVES_SHA256)" \
  "$parent_leaves_sha" "parent leaves anchor"
require_equal "$(manifest_value PARENT_AFFINE_OBLIGATION_SHA256)" \
  "$parent_affine_obligation_sha" "parent affine obligation anchor"
require_equal "$(digest "$parent/run-manifest.txt")" "$parent_run_manifest_sha" \
  "parent run-manifest source"
require_equal "$(digest "$parent/files.sha256")" "$parent_files_index_sha" \
  "parent files index source"
require_equal "$(digest "$parent/coordinates.tsv")" "$parent_coordinates_sha" \
  "parent coordinates source"
require_equal "$(digest "$parent/leaves.tsv")" "$parent_leaves_sha" \
  "parent leaves source"

# Re-run the runner's independent manifest construction checks against the
# anchored parent receipts instead of trusting the retained TSV by itself.
python3 - "$runner" "$coordinate_manifest" "$repo_root" \
  "$parent_affine_obligation_sha" <<'PY'
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

runner_path = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
repo = Path(sys.argv[3]).resolve()
expected_affine_sha = sys.argv[4]
spec = importlib.util.spec_from_file_location("cs6_full53_gate_runner", runner_path)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load full53 runner")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
leaves = module.parse_coordinate_manifest(manifest_path, repo)
root_id = "U00-0000000000_S00-0000000000"
paired = [leaf for leaf in leaves if leaf.identity != root_id]
affine_ids = sorted(leaf.identity for leaf in paired if leaf.parent_affine_pass)
affine_raw = ("\n".join(affine_ids) + "\n").encode("ascii")
if len(leaves) != 53 or len(paired) != 52 or len(affine_ids) != 28:
    raise SystemExit("full53 source replay population mismatch")
if hashlib.sha256(affine_raw).hexdigest() != expected_affine_sha:
    raise SystemExit("full53 source replay affine obligation mismatch")
if leaves[0].identity != root_id:
    raise SystemExit("full53 source replay root mismatch")
PY

require_equal "$(manifest_value LEAF_COUNT)" "53" "retained leaf count"
require_equal "$(manifest_value PAIRED_VALID_COUNT)" "52" \
  "retained paired-valid count"
require_equal "$(manifest_value ROOT_INTERVAL_DOMAIN_CLASS_MATCH)" "true" \
  "retained root failure class"
require_equal "$(manifest_value NEW_UNRESOLVED_COUNT)" "0" \
  "retained new-unresolved count"
require_equal "$(manifest_value PARENT_AFFINE_OBLIGATION_COUNT)" "28" \
  "retained parent-affine obligation count"
require_equal "$(manifest_value PARENT_AFFINE_LOSS_COUNT)" "0" \
  "retained parent-affine loss count"
require_equal "$(manifest_value MUTATION_AUDITED_LEAF_COUNT)" "52" \
  "retained mutation-audited leaf count"
require_equal "$(manifest_value MUTATION_SUITE_SIZE_PER_LEAF)" "112" \
  "retained mutation suite size"
require_equal "$(manifest_value MUTATION_TESTS)" "5824" \
  "retained mutation test count"
require_equal "$(manifest_value MUTATIONS_REJECTED)" "5824" \
  "retained mutation rejection count"
require_equal "$(manifest_value H_APG_CS6_FULL53_SUPPORTED)" "true" \
  "retained full53 support result"
require_equal "$(manifest_value PROMOTION_ELIGIBLE)" "false" \
  "run-manifest promotion boundary"
require_equal "$(retained_value PROMOTION_ELIGIBLE)" "false" \
  "retained-manifest promotion boundary"
require_equal "$(awk -F= '$1 == "PROMOTION_ELIGIBLE" {print $2}' "$retained/summary.txt")" \
  "false" "summary promotion boundary"

mutation_dir=$(mktemp -d)
trap 'rm -rf "$mutation_dir"' EXIT

rehash_mutation() {
  local root=$1
  python3 - "$root" <<'PY'
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

root = Path(sys.argv[1])

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_kv(path: Path) -> list[tuple[str, str]]:
    lines = path.read_text(encoding="ascii").splitlines()
    return [tuple(line.split("=", 1)) for line in lines]

def write_kv(path: Path, fields: list[tuple[str, str]]) -> None:
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in fields),
        encoding="ascii",
    )

run_manifest = root / "run-manifest.txt"
run_fields = read_kv(run_manifest)
run_values = dict(run_fields)
source_bindings = {
    "worker-source.cpp": "SOURCE_SHA256",
    "leaf-verifier.py": "VERIFIER_SHA256",
    "runner.py": "RUNNER_SHA256",
    "coordinates.tsv": "COORDINATE_MANIFEST_SHA256",
}
for path in root.iterdir():
    if path.is_symlink() or not path.is_file():
        continue
    if path.name in {"files.sha256", "retained-manifest.txt", "run-manifest.txt"}:
        continue
    artifact_key = path.name.upper().replace("-", "_").replace(".", "_") + "_SHA256"
    if artifact_key in run_values:
        run_values[artifact_key] = digest(path)
    source_key = source_bindings.get(path.name)
    if source_key is not None:
        run_values[source_key] = digest(path)
write_kv(run_manifest, [(key, run_values[key]) for key, _ in run_fields])

excluded = {"files.sha256", "retained-manifest.txt"}
files = sorted(
    path
    for path in root.rglob("*")
    if not path.is_symlink()
    and path.is_file()
    and path.relative_to(root).as_posix() not in excluded
)
index = root / "files.sha256"
index.write_text(
    "".join(
        f"{digest(path)}  {path.relative_to(root).as_posix()}\n"
        for path in files
    ),
    encoding="ascii",
)
retained_manifest = root / "retained-manifest.txt"
retained_fields = read_kv(retained_manifest)
replacements = {
    "FILES_INDEX_SHA256": digest(index),
    "FILE_COUNT": str(len(files)),
    "RUN_MANIFEST_SHA256": digest(run_manifest),
}
write_kv(
    retained_manifest,
    [(key, replacements.get(key, value)) for key, value in retained_fields],
)
PY
}

for semantic_mutation in \
  chart sign chart-challenge-rebinding parent-affine-obligation leaf-method \
  root-class mutation-aggregate extra-file path-symlink; do
  semantic_dir="$mutation_dir/retained-$semantic_mutation"
  cp -a "$retained" "$semantic_dir"
  python3 - "$semantic_dir" "$semantic_mutation" "$root_id" <<'PY'
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
mutation = sys.argv[2]
root_id = sys.argv[3]

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_table(path: Path) -> tuple[list[str], list[list[str]]]:
    lines = path.read_text(encoding="ascii").splitlines()
    return lines[0].split("\t"), [line.split("\t") for line in lines[1:]]

def write_table(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.write_text(
        "\t".join(header) + "\n" + "".join("\t".join(row) + "\n" for row in rows),
        encoding="ascii",
    )

def update_kv(path: Path, updates: dict[str, str]) -> None:
    fields = [line.split("=", 1) for line in path.read_text(encoding="ascii").splitlines()]
    path.write_text(
        "".join(f"{key}={updates.get(key, value)}\n" for key, value in fields),
        encoding="ascii",
    )

def alternate_chart(chart: str) -> str:
    return "Y" if chart != "Y" else "X"

leaves_path = root / "leaves.tsv"
leaf_header, leaf_rows = read_table(leaves_path)
leaf_column = {name: index for index, name in enumerate(leaf_header)}
paired_index = next(index for index, row in enumerate(leaf_rows) if row[0] != root_id)

if mutation == "extra-file":
    (root / "unexpected.txt").write_text("unexpected\n", encoding="ascii")
elif mutation == "path-symlink":
    (root / "unexpected-tree").symlink_to("/etc", target_is_directory=True)
elif mutation == "chart":
    column = leaf_column["E1_R0_CHART"]
    leaf_rows[paired_index][column] = alternate_chart(leaf_rows[paired_index][column])
    write_table(leaves_path, leaf_header, leaf_rows)
elif mutation == "sign":
    column = leaf_column["E1_R0_SIGN"]
    leaf_rows[paired_index][column] = str(-int(leaf_rows[paired_index][column]))
    write_table(leaves_path, leaf_header, leaf_rows)
elif mutation == "leaf-method":
    column = leaf_column["METHOD"]
    leaf_rows[paired_index][column] = (
        "NONE" if leaf_rows[paired_index][column] != "NONE" else "AFFINE"
    )
    write_table(leaves_path, leaf_header, leaf_rows)
elif mutation == "parent-affine-obligation":
    affine_index = next(
        index
        for index, row in enumerate(leaf_rows)
        if row[leaf_column["PARENT_AFFINE_PASS"]] == "true"
    )
    leaf_rows[affine_index][leaf_column["PARENT_AFFINE_PASS"]] = "false"
    leaf_rows[affine_index][leaf_column["PARENT_STATUS"]] = "PROBE_VALID_UNCERTIFIED"
    write_table(leaves_path, leaf_header, leaf_rows)
elif mutation == "root-class":
    root_index = next(index for index, row in enumerate(leaf_rows) if row[0] == root_id)
    leaf_rows[root_index][leaf_column["STATUS"]] = "COMPUTATION_UNRESOLVED_TIMEOUT"
    stderr_path = root / "stderr" / f"{root_id}.txt"
    stderr_path.write_text("probe timeout\n", encoding="ascii")
    leaf_rows[root_index][leaf_column["STDERR_SHA256"]] = digest(stderr_path)
    write_table(leaves_path, leaf_header, leaf_rows)
elif mutation == "mutation-aggregate":
    audit_path = root / "mutation-audits.tsv"
    audit_header, audit_rows = read_table(audit_path)
    audit_column = {name: index for index, name in enumerate(audit_header)}
    rejected_column = audit_column["MUTATIONS_REJECTED"]
    audit_rows[-1][rejected_column] = str(int(audit_rows[-1][rejected_column]) - 1)
    write_table(audit_path, audit_header, audit_rows)
    total_tests = sum(int(row[audit_column["MUTATION_TESTS"]]) for row in audit_rows)
    total_rejected = sum(int(row[rejected_column]) for row in audit_rows)
    updates = {
        "MUTATION_TESTS": str(total_tests),
        "MUTATIONS_REJECTED": str(total_rejected),
    }
    update_kv(root / "summary.txt", updates)
    update_kv(root / "run-manifest.txt", updates)
elif mutation == "chart-challenge-rebinding":
    coordinates_path = root / "coordinates.tsv"
    coordinate_lines = coordinates_path.read_text(encoding="ascii").splitlines()
    coordinate_header_index = next(
        index for index, line in enumerate(coordinate_lines) if line.startswith("LEAF_ID\t")
    )
    coordinate_header = coordinate_lines[coordinate_header_index].split("\t")
    coordinate_rows = [line.split("\t") for line in coordinate_lines[coordinate_header_index + 1 :]]
    coordinate_column = {name: index for index, name in enumerate(coordinate_header)}
    identity = leaf_rows[paired_index][leaf_column["LEAF_ID"]]
    coordinate_index = next(index for index, row in enumerate(coordinate_rows) if row[0] == identity)
    new_chart = alternate_chart(coordinate_rows[coordinate_index][coordinate_column["E1_R0_CHART"]])
    coordinate_rows[coordinate_index][coordinate_column["E1_R0_CHART"]] = new_chart
    coordinate_body = "".join("\t".join(row) + "\n" for row in coordinate_rows)
    coordinates_path.write_text(
        "\n".join(coordinate_lines[: coordinate_header_index + 1]) + "\n" + coordinate_body,
        encoding="ascii",
    )
    manifest_sha = digest(coordinates_path)
    coordinate_row = coordinate_rows[coordinate_index]
    chart_sign_fields: list[str] = []
    for prefix in ("E1_R0", "E1_R1", "E2_R0", "E2_R1"):
        chart_sign_fields.extend(
            (
                coordinate_row[coordinate_column[f"{prefix}_CHART"]],
                coordinate_row[coordinate_column[f"{prefix}_SIGN"]],
            )
        )
    run_values = dict(
        line.split("=", 1)
        for line in (root / "run-manifest.txt").read_text(encoding="ascii").splitlines()
    )
    preimage = (
        b"sounio.cs6.affine-projective-cocycle-full53-leaf-challenge.v1\0"
        + bytes.fromhex(run_values["ROOT_CHALLENGE"])
        + b"\0"
        + identity.encode("ascii")
        + b"\0"
        + bytes.fromhex(coordinate_row[coordinate_column["PARENT_INPUT_SHA256"]])
        + b"\0"
        + bytes.fromhex(coordinate_row[coordinate_column["PARENT_RECEIPT_SHA256"]])
        + b"\0"
        + ("\t".join(chart_sign_fields) + "\n").encode("ascii")
        + b"\0"
        + bytes.fromhex(manifest_sha)
    )
    challenge = hashlib.sha256(preimage).hexdigest()
    leaf_rows[paired_index][leaf_column["E1_R0_CHART"]] = new_chart
    leaf_rows[paired_index][leaf_column["LEAF_CHALLENGE"]] = challenge

    receipt_path = root / "receipts" / f"{identity}.txt"
    receipt = receipt_path.read_text(encoding="ascii")
    receipt = re.sub(
        r"(?m)^COORDINATE_MANIFEST_SHA256=.*$",
        f"COORDINATE_MANIFEST_SHA256={manifest_sha}",
        receipt,
        count=1,
    )
    receipt = re.sub(
        r"(?m)^RUN_CHALLENGE=.*$", f"RUN_CHALLENGE={challenge}", receipt, count=1
    )
    receipt = re.sub(
        r"(?m)^(APG_EVENT1_RAY0 .*? CHART=)(X|Y|PLUS|MINUS)",
        rf"\g<1>{new_chart}",
        receipt,
        count=1,
    )
    receipt_path.write_text(receipt, encoding="ascii")
    receipt_sha = digest(receipt_path)
    leaf_rows[paired_index][leaf_column["RECEIPT_SHA256"]] = receipt_sha

    verification_path = root / "verifications" / f"{identity}.txt"
    verification = verification_path.read_text(encoding="ascii")
    for key, value in (
        ("RECEIPT_SHA256", receipt_sha),
        ("COORDINATE_MANIFEST_SHA256", manifest_sha),
        ("LEAF_CHALLENGE", challenge),
    ):
        verification = re.sub(
            rf"(?m)^{key}=.*$", f"{key}={value}", verification, count=1
        )
    verification_path.write_text(verification, encoding="ascii")
    verification_sha = digest(verification_path)
    leaf_rows[paired_index][leaf_column["VERIFICATION_SHA256"]] = verification_sha
    write_table(leaves_path, leaf_header, leaf_rows)

    audit_path = root / "mutation-audits.tsv"
    audit_header, audit_rows = read_table(audit_path)
    audit_column = {name: index for index, name in enumerate(audit_header)}
    for row in audit_rows:
        if row[audit_column["LEAF_ID"]] == identity:
            row[audit_column["RECEIPT_SHA256"]] = receipt_sha
            row[audit_column["VERIFICATION_SHA256"]] = verification_sha
            break
    write_table(audit_path, audit_header, audit_rows)
else:
    raise SystemExit(f"unknown semantic mutation: {mutation}")
PY
  rehash_mutation "$semantic_dir"
  if python3 "$retained_verifier" "$semantic_dir" \
    >"$mutation_dir/$semantic_mutation.stdout" \
    2>"$mutation_dir/$semantic_mutation.stderr"; then
    fail "coordinated $semantic_mutation mutation escaped retained verification"
  fi
done

fresh_replay=skipped
if [[ ${CS6_AFFINE_PROJECTIVE_COCYCLE_FULL53_REPLAY:-0} == 1 ]]; then
  capd_config=${CS6_CAPD_CONFIG:-/tmp/capd-build/bin/capd-config}
  [[ -x $capd_config ]] || fail "capd-config is not executable: $capd_config"
  require_equal "$($capd_config --modversion)" "5.3.0" "CAPD version"
  cxx=${CXX:-g++}
  command -v "$cxx" >/dev/null || fail "C++ compiler is unavailable: $cxx"
  fresh_source_sha=$(digest "$source_file")
  binary="$mutation_dir/full53-worker"
  # shellcheck disable=SC2046
  "$cxx" -std=c++17 $($capd_config --cflags) -O0 \
    "-DCS6_WORKER_SOURCE_SHA256=\"$fresh_source_sha\"" \
    "$source_file" -o "$binary" $($capd_config --libs) \
    >"$mutation_dir/compile.stdout" 2>"$mutation_dir/compile.stderr"
  root_challenge=$(manifest_value ROOT_CHALLENGE)
  manifest_sha=$(digest "$coordinate_manifest")

  python3 - "$coordinate_manifest" "$replay_xy_id" \
    "$replay_minus_e2_positive_id" <<'PY'
from pathlib import Path
import sys

lines = Path(sys.argv[1]).read_text(encoding="ascii").splitlines()
header_index = next(index for index, line in enumerate(lines) if line.startswith("LEAF_ID\t"))
header = lines[header_index].split("\t")
rows = {row[0]: dict(zip(header, row)) for row in (line.split("\t") for line in lines[header_index + 1 :])}
selected = [rows[identity] for identity in sys.argv[2:]]
charts = {
    row[key]
    for row in selected
    for key in ("E1_R0_CHART", "E1_R1_CHART", "E2_R0_CHART", "E2_R1_CHART")
}
if charts != {"X", "Y", "PLUS", "MINUS"}:
    raise SystemExit("fresh replay representatives do not cover all four charts")
if not any(row[key] == "1" for row in selected for key in ("E2_R0_SIGN", "E2_R1_SIGN")):
    raise SystemExit("fresh replay representatives lack an E2 positive pivot")
PY

  for identity in "$replay_xy_id" "$replay_minus_e2_positive_id"; do
    row=$(awk -F '\t' -v id="$identity" '$1 == id {print; exit}' "$coordinate_manifest")
    [[ -n $row ]] || fail "fresh replay leaf absent from manifest: $identity"
    IFS=$'\t' read -r _ u_depth u_index s_depth s_index input_sha _ \
      parent_receipt_sha e1_r0_chart e1_r0_sign e1_r1_chart e1_r1_sign \
      e2_r0_chart e2_r0_sign e2_r1_chart e2_r1_sign <<<"$row"
    input_path="$parent/inputs/$identity.txt"
    parent_receipt="$parent/receipts/$identity.txt"
    challenge=$(python3 - "$leaf_verifier" "$coordinate_manifest" \
      "$input_path" "$root_challenge" <<'PY'
import importlib.util
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("cs6_full53_gate_verifier", Path(sys.argv[1]))
if spec is None or spec.loader is None:
    raise SystemExit("cannot load full53 leaf verifier")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
leaf_input = module.read_leaf_input(Path(sys.argv[3]))
contract = module.read_full53_contract(Path(sys.argv[2]), leaf_input)
print(module.full53_leaf_challenge(sys.argv[4], contract))
PY
    )
    receipt="$mutation_dir/$identity.receipt.txt"
    stderr="$mutation_dir/$identity.stderr.txt"
    "$binary" "$u_depth" "$u_index" "$s_depth" "$s_index" \
      "$input_sha" "$parent_receipt_sha" \
      "$e1_r0_chart" "$e1_r0_sign" "$e1_r1_chart" "$e1_r1_sign" \
      "$e2_r0_chart" "$e2_r0_sign" "$e2_r1_chart" "$e2_r1_sign" \
      "$manifest_sha" "$challenge" >"$receipt" 2>"$stderr"
    [[ ! -s $stderr ]] || fail "fresh worker emitted stderr: $identity"
    python3 "$leaf_verifier" "$receipt" --source-sha "$fresh_source_sha" \
      --input "$input_path" --coordinate-manifest "$coordinate_manifest" \
      --parent-receipt "$parent_receipt" --root-challenge "$root_challenge" \
      --require-probe \
      >"$mutation_dir/$identity.verification.txt"
    grep -qx 'PROBE_PASS=true' "$mutation_dir/$identity.verification.txt"
    grep -qx 'APG_COMPUTATION_VALID=true' "$mutation_dir/$identity.verification.txt"
    cmp -s "$mutation_dir/$identity.verification.txt" \
      "$retained/verifications/$identity.txt" || \
      fail "fresh verifier output differs from retained replay: $identity"
  done

  root_row=$(awk -F '\t' -v id="$root_id" '$1 == id {print; exit}' "$coordinate_manifest")
  [[ -n $root_row ]] || fail "root leaf absent from manifest"
  IFS=$'\t' read -r _ root_u_depth root_u_index root_s_depth root_s_index \
    root_input_sha _ root_parent_receipt_sha root_e1_r0_chart root_e1_r0_sign \
    root_e1_r1_chart root_e1_r1_sign root_e2_r0_chart root_e2_r0_sign \
    root_e2_r1_chart root_e2_r1_sign <<<"$root_row"
  root_input="$parent/inputs/$root_id.txt"
  root_parent_receipt="$parent/receipts/$root_id.txt"
  root_challenge_leaf=$(python3 - "$runner" "$coordinate_manifest" "$repo_root" \
    "$root_id" "$root_challenge" <<'PY'
import importlib.util
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("cs6_full53_gate_root_runner", Path(sys.argv[1]))
if spec is None or spec.loader is None:
    raise SystemExit("cannot load full53 runner")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
leaves = module.parse_coordinate_manifest(Path(sys.argv[2]), Path(sys.argv[3]))
leaf = next(item for item in leaves if item.identity == sys.argv[4])
print(module.leaf_challenge(sys.argv[5], module.FROZEN_MANIFEST_SHA256, leaf))
PY
  )
  root_receipt="$mutation_dir/$root_id.receipt.txt"
  root_stderr="$mutation_dir/$root_id.stderr.txt"
  if "$binary" "$root_u_depth" "$root_u_index" "$root_s_depth" "$root_s_index" \
    "$root_input_sha" "$root_parent_receipt_sha" \
    "$root_e1_r0_chart" "$root_e1_r0_sign" \
    "$root_e1_r1_chart" "$root_e1_r1_sign" \
    "$root_e2_r0_chart" "$root_e2_r0_sign" \
    "$root_e2_r1_chart" "$root_e2_r1_sign" \
    "$manifest_sha" "$root_challenge_leaf" >"$root_receipt" 2>"$root_stderr"; then
    fail "fresh U00 replay unexpectedly computed"
  else
    root_rc=$?
  fi
  require_equal "$root_rc" "1" "fresh U00 worker status"
  [[ ! -s $root_receipt ]] || fail "fresh U00 replay emitted a partial receipt"
  python3 - "$root_stderr" <<'PY'
from pathlib import Path
import math
import sys

raw = Path(sys.argv[1]).read_bytes()
try:
    lines = raw.decode("ascii").splitlines()
except UnicodeError as error:
    raise SystemExit("fresh U00 stderr is not ASCII") from error
prefix = "probe error: Interval error: Division by 0 in operator/(Interval, Interval)"
if len(lines) != 2 or lines[0] != prefix or not raw.endswith(b"\n"):
    raise SystemExit("fresh U00 failure class drifted")
if not lines[1].startswith("   left=") or "  right=" not in lines[1]:
    raise SystemExit("fresh U00 interval-domain detail drifted")
left, right = lines[1].removeprefix("   left=").split("  right=", 1)
if not math.isfinite(float(left)) or not math.isfinite(float(right)):
    raise SystemExit("fresh U00 interval-domain bounds are nonfinite")
PY
  if python3 "$leaf_verifier" "$root_receipt" --source-sha "$fresh_source_sha" \
    --input "$root_input" --coordinate-manifest "$coordinate_manifest" \
    --parent-receipt "$root_parent_receipt" --root-challenge "$root_challenge" \
    >"$mutation_dir/$root_id.verification.stdout" \
    2>"$mutation_dir/$root_id.verification.stderr"; then
    fail "full53 leaf verifier accepted the unresolved root"
  fi
  grep -qx \
    'verification error: the unresolved full53 root is outside the paired leaf verifier' \
    "$mutation_dir/$root_id.verification.stderr"
  fresh_replay=2/52+root-class
fi

echo "cs6 affine-projective cocycle full53 retained-integrity gate: PASS"
echo "fresh_replay=$fresh_replay"
echo "coordinate_count=53"
echo "paired_valid_count=52"
echo "parent_affine_obligation_count=28"
echo "parent_affine_loss_count=0"
echo "h_apg_cs6_full53_supported=true"
echo "promotion_eligible=false"
