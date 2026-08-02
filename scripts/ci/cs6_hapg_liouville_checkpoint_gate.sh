#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

contract=scripts/research/cs6_hapg_liouville_checkpoint_contract_v1.txt
coordinates=scripts/research/cs6_hapg_liouville_checkpoint_coordinates_v1.tsv
source_file=scripts/research/cs6_hapg_liouville_checkpoint_probe.cpp
verifier=scripts/research/cs6_hapg_liouville_checkpoint_verify.py
runner=scripts/research/cs6_hapg_liouville_checkpoint_run.py
retained=scripts/research/cs6_hapg_liouville_checkpoint_retained_verify.py
slurm_job=scripts/research/cs6_hapg_liouville_checkpoint_slurm_job.sh

python3 -B - "$contract" "$coordinates" "$source_file" <<'PY'
from pathlib import Path
import csv
import hashlib
import sys

contract_path, coordinate_path, source_path = map(Path, sys.argv[1:])

def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"V7-A.1 gate error: {message}")

def canonical(path: Path) -> bytes:
    raw = path.read_bytes()
    check(raw.endswith(b"\n"), f"missing final LF: {path}")
    check(b"\r" not in raw and b"\0" not in raw and raw.isascii(), f"noncanonical bytes: {path}")
    return raw

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def kv(path: Path) -> dict[str, str]:
    fields = {}
    for line in canonical(path).decode("ascii").splitlines():
        check(line.count("=") == 1, f"malformed KV line: {path}")
        key, value = line.split("=", 1)
        check(bool(key and value and key not in fields), f"duplicate KV field: {path}")
        fields[key] = value
    return fields

check(
    digest(contract_path) == "3afc0475847ad8054234a2ddfa108b768cfd81991d0be71fc21c991f363631ce",
    "frozen contract digest drift",
)
check(
    digest(coordinate_path) == "527afc7c205fcf09b15a0bff91df6935f19ed2b7e7926895916ac5da33a992a7",
    "coordinate manifest digest drift",
)
contract = kv(contract_path)
exact_contract = {
    "SCHEMA": "sounio.cs6.hapg-liouville-checkpoint-contract.v1",
    "CONTRACT_STATE": "PRE_RESULT_FROZEN",
    "BASE_COMMIT": "74d25167a123c2d6eaba2281eb0a919cd7a33f80",
    "PARENT_V7_JOB": "8480",
    "PARENT_V7_RESULTS_SHA256": "cf61025f8b3441045827432a6a3dd830a19aeb43535893c613825fff973b8b3b",
    "COORDINATE_MANIFEST_SHA256": "527afc7c205fcf09b15a0bff91df6935f19ed2b7e7926895916ac5da33a992a7",
    "ROOT_CHALLENGE": "ad536f25d02182c8b2add12ef1a7e8a8a18b4eb0d12e68535ea94ddb3eff0bdf",
    "CELL_COUNT": "3",
    "CARRIER_COUNT": "3",
    "CARRIERS": "C0HOTripletonSet,C0HORect2Set,C0Rect2Set",
    "MAXIMUM_EVALUATIONS": "9",
    "EXACT_ATTEMPT_MATRIX": "3_CONTIGUOUS_V7A_TARGET_CELLS_TIMES_3_CARRIERS",
    "PHYSICAL_FLOW_DIMENSION": "3",
    "AUXILIARY_LIOUVILLE_STATE": "true",
    "RETURN_MAP_DIMENSION": "2",
    "CAPD_VERSION": "5.3.0",
    "INTERVAL_BACKEND": "FILIB",
    "OPTIMIZATION_LEVEL": "O0",
    "PROBE_BOUNDARY": "ONLY_LIOUVILLE_TWO_RETURN_THEN_SERIALIZE_AND_EXIT",
    "C1_EXECUTED": "false",
    "C2_EXECUTED": "false",
    "DOWNSTREAM_SECTION_RESIDENT_EXECUTED": "false",
    "CHECKPOINT_OUTPUT_POLICY": "EMPTY_STDOUT_UNLESS_COMPLETE",
    "V7A_RETROACTIVE_REINTERPRETATION_ALLOWED": "false",
    "V7A_MISSING_HPG_RECEIPT_SUPPLIED": "false",
    "C1_C2_DETERMINANT_COMPATIBILITY_EVALUATED": "false",
    "FULL_HPG_PIPELINE_EVALUATED": "false",
    "V7_B_ELIGIBILITY": "false",
    "V7_B_WINNER": "NONE",
    "PROMOTION_ELIGIBLE": "false",
    "OPEN_PROBLEM_SOLVED": "false",
    "FPGA_EXECUTION": "false",
}
for key, value in exact_contract.items():
    check(contract.get(key) == value, f"frozen contract mismatch: {key}")

lines = canonical(coordinate_path).decode("ascii").splitlines()
header = (
    "ORDINAL\tCHECKPOINT_ROLE\tPARENT_V7_ORDINAL\tPARENT_V7_ATTEMPTS\t"
    "NODE_ID\tU_DEPTH\tU_INDEX\tS_DEPTH\tS_INDEX\tPARENT_INPUT_SHA256\t"
    "PARENT_ALT_INITIAL_HULL_SHA256\tPARENT_HO_RECT2_DET\tPARENT_RECT2_DET"
)
check(header in lines, "coordinate table header missing")
rows = list(csv.DictReader(lines[lines.index(header):], delimiter="\t"))
check(len(rows) == 3, "coordinate row count differs")
check([row["PARENT_V7_ORDINAL"] for row in rows] == ["22", "23", "24"], "parent ordinal drift")
check([row["PARENT_V7_ATTEMPTS"] for row in rows] == ["64,65,66", "67,68,69", "70,71,72"], "parent attempts drift")
check(len({row["NODE_ID"] for row in rows}) == 3, "coordinate nodes are not unique")

domain = b"sounio.cs6.hapg-liouville-checkpoint-root.v1\0"
root = hashlib.sha256(
    domain
    + b"2026-08-02\0"
    + b"74d25167a123c2d6eaba2281eb0a919cd7a33f80\0"
    + bytes(contract["COORDINATE_MANIFEST_SHA256"], "ascii")
).hexdigest()
check(root == contract["ROOT_CHALLENGE"], "root challenge does not recompute")

source = canonical(source_path).decode("ascii")
forbidden = (
    "C1Rect2Set",
    "C2Rect2Set",
    "SectionResidentMap",
    "resident_return",
    "crossSectionInOneStep",
    "IC2OdeSolver",
    "IC2PoincareMap",
    "kVectorField",
)
for token in forbidden:
    check(token not in source, f"forbidden downstream token present: {token}")
check(source.count("result.image = map(set, result.time, kReturnCount);") == 1, "two-return map call drift")
check(source.count("const LiouvilleData liouville = liouville_two_return(input, carrier);") == 1, "main Liouville call drift")
check(source.count("IPoincareMap map(") == 1 and source.count("map(") == 2, "unexpected Poincare-map call count")
check("std::ostringstream receipt;" in source, "receipt is not buffered until completion")
check(source.index("const LiouvilleData liouville = liouville_two_return(input, carrier);") < source.index("std::ostringstream receipt;"), "receipt starts before Liouville completion")
print("STATIC_ASSERTIONS=61")
PY

pycache=$(mktemp -d)
build_dir=$(mktemp -d)
cleanup() {
  rm -rf "$pycache" "$build_dir"
}
trap cleanup EXIT
PYTHONPYCACHEPREFIX=$pycache python3 -m py_compile "$verifier" "$runner" "$retained"
bash -n "$slurm_job"

capd_config=${CS6_CAPD_CONFIG:-/tmp/capd-build/bin/capd-config}
if [[ ! -x $capd_config ]]; then
  echo "FULL_SMOKE=skipped_no_capd_config"
  echo "STATIC_GATE_PASS=true"
  exit 0
fi

compiler=${CXX:-/usr/bin/x86_64-linux-gnu-g++-13}
[[ -x $compiler ]] || {
  echo "V7-A.1 gate error: compiler is unavailable" >&2
  exit 1
}
source_sha=$(sha256sum "$source_file" | awk '{print $1}')
cflags=$($capd_config --cflags)
libs=$($capd_config --libs)
read -r -a cflag_args <<< "$cflags"
read -r -a lib_args <<< "$libs"
printf '%s\n' "$cflags" > "$build_dir/capd-cflags.txt"
printf '%s\n' "$libs" > "$build_dir/capd-libs.txt"
printf '5.3.0\n' > "$build_dir/capd-version.txt"
cp "$source_file" "$build_dir/worker-source.cpp"
printf '%s\n' "$source_sha" > "$build_dir/worker-source.sha256"
compile_command=(
  "$compiler" -std=c++17 "${cflag_args[@]}" -O0
  "-DCS6_WORKER_SOURCE_SHA256=\"$source_sha\"" "$source_file"
  -MD -MF "$build_dir/dependencies.d" -o "$build_dir/worker-binary"
  "${lib_args[@]}"
)
printf '%q ' "${compile_command[@]}" > "$build_dir/compile-command.txt"
printf '\n' >> "$build_dir/compile-command.txt"

set +e
"${compile_command[@]}" \
  > "$build_dir/compile-stdout.txt" 2> "$build_dir/compile-stderr.txt"
compile_rc=$?
set -e
[[ $compile_rc -eq 0 ]] || {
  cat "$build_dir/compile-stderr.txt" >&2
  exit "$compile_rc"
}
second_compile_command=(
  "$compiler" -std=c++17 "${cflag_args[@]}" -O0
  "-DCS6_WORKER_SOURCE_SHA256=\"$source_sha\"" "$source_file"
  -MD -MF "$build_dir/dependencies.second.d" -o "$build_dir/worker-binary.second"
  "${lib_args[@]}"
)
"${second_compile_command[@]}" \
  >/dev/null 2>"$build_dir/compile-second.stderr.txt"
cmp "$build_dir/worker-binary" "$build_dir/worker-binary.second"
chmod 700 "$build_dir/worker-binary"
sha256sum "$build_dir/worker-binary" | awk '{print $1}' > "$build_dir/worker-binary.sha256"
$compiler --version > "$build_dir/compiler-version.txt"
sha256sum "$build_dir/dependencies.d" > "$build_dir/dependencies.sha256"
ldd "$build_dir/worker-binary" > "$build_dir/runtime-linkage.txt"
sha256sum "$build_dir/runtime-linkage.txt" > "$build_dir/runtime-libraries.sha256"

echo "DETERMINISTIC_BUILD_SMOKE=passed"
if [[ -n $(git status --porcelain=v1 --untracked-files=all) ]]; then
  echo "FULL_SMOKE=skipped_dirty_repo"
  echo "STATIC_GATE_PASS=true"
  exit 0
fi

prebuilt_members=(
  capd-cflags.txt capd-libs.txt capd-version.txt compile-command.txt
  compile-stderr.txt compile-stdout.txt compiler-version.txt dependencies.sha256
  runtime-libraries.sha256 runtime-linkage.txt worker-source.cpp
  worker-source.sha256 worker-binary worker-binary.sha256
)
tar --format=ustar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  -C "$build_dir" -cf "$build_dir/prebuilt.tar" "${prebuilt_members[@]}"
repo_archive_members=(
  scripts/research/cs6_hapg_liouville_checkpoint_contract_v1.txt
  scripts/research/cs6_hapg_liouville_checkpoint_coordinates_v1.tsv
  scripts/research/cs6_hapg_liouville_checkpoint_probe.cpp
  scripts/research/cs6_hapg_liouville_checkpoint_verify.py
  scripts/research/cs6_hapg_liouville_checkpoint_run.py
  scripts/research/cs6_hapg_liouville_checkpoint_retained_verify.py
  scripts/research/cs6_hapg_liouville_checkpoint_slurm_job.sh
  scripts/research/cs6_plucker_cocycle_verify.py
)
git archive --format=tar -o "$build_dir/repo-source.tar" HEAD \
  "${repo_archive_members[@]}"
[[ $(git get-tar-commit-id < "$build_dir/repo-source.tar") == $(git rev-parse HEAD) ]]

node=$(hostname -s)
fake_scontrol="$build_dir/scontrol"
cat > "$fake_scontrol" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
case ${1:-} in
  --version)
    echo 'synthetic-scontrol 1.0'
    ;;
  -o)
    [[ ${2:-} == show && ${3:-} == job && ${4:-} == "$SLURM_JOB_ID" ]]
    printf 'JobId=%s JobName=cs6-v7a1-checkpoint UserId=gate(%s) Account=synthetic QOS=synthetic JobState=RUNNING Partition=synthetic-gate TimeLimit=00:20:00 NodeList=%s BatchHost=%s NumNodes=1 NumCPUs=9 NumTasks=1 CPUs/Task=9 MinMemoryNode=8G OverSubscribe=NO Requeue=0 Restarts=0 Command=%s\n' \
      "$SLURM_JOB_ID" "$(id -u)" "$SLURM_JOB_NODELIST" "$SLURM_JOB_NODELIST" \
      "$V7A1_SYNTHETIC_COMMAND"
    ;;
  show)
    [[ ${2:-} == hostnames && ${3:-} == "$SLURM_JOB_NODELIST" ]]
    printf '%s\n' "$SLURM_JOB_NODELIST"
    ;;
  *)
    exit 64
    ;;
esac
SH
chmod 700 "$fake_scontrol"

head=$(git rev-parse HEAD)
job_sha=$(sha256sum "$slurm_job" | awk '{print $1}')
repo_archive_sha=$(sha256sum "$build_dir/repo-source.tar" | awk '{print $1}')
prebuilt_sha=$(sha256sum "$build_dir/prebuilt.tar" | awk '{print $1}')
config="$build_dir/config.txt"
cat > "$config" <<EOF
SCHEMA=sounio.cs6.hapg-liouville-checkpoint-slurm-config.v2
EXECUTION_CLASS=SYNTHETIC_GATE
REPO_ARCHIVE=$build_dir/repo-source.tar
REPO_ARCHIVE_SHA256=$repo_archive_sha
GIT_HEAD=$head
PREBUILT_ARCHIVE=$build_dir/prebuilt.tar
PREBUILT_ARCHIVE_SHA256=$prebuilt_sha
OUTPUT_ARCHIVE=$build_dir/result.tar
ROOT_CHALLENGE=ad536f25d02182c8b2add12ef1a7e8a8a18b4eb0d12e68535ea94ddb3eff0bdf
FROZEN_CONTRACT_SHA256=3afc0475847ad8054234a2ddfa108b768cfd81991d0be71fc21c991f363631ce
COORDINATE_MANIFEST_SHA256=527afc7c205fcf09b15a0bff91df6935f19ed2b7e7926895916ac5da33a992a7
JOB_SCRIPT_SHA256=$job_sha
SUBMITTED_JOB_SCRIPT=$repo_root/$slurm_job
JOBS=9
TIMEOUT_SECONDS=300
SLURM_PARTITION=synthetic-gate
SLURM_ACCOUNT=synthetic
SLURM_QOS=synthetic
SLURM_NODELIST=$node
SLURM_NODES=1
SLURM_NTASKS=1
SLURM_CPUS_PER_TASK=9
SLURM_JOB_NAME=cs6-v7a1-checkpoint
SLURM_TIME_LIMIT=00:20:00
SLURM_MIN_MEMORY_NODE=8G
SLURM_EXCLUSIVE=NODE
EOF

env \
  CS6_V7A1_SYNTHETIC_GATE=1 \
  CS6_V7A1_SCONTROL_BIN="$fake_scontrol" \
  V7A1_SYNTHETIC_COMMAND="$repo_root/$slurm_job" \
  SLURM_JOB_ID=900001 \
  SLURM_JOB_NODELIST="$node" \
  SLURMD_NODENAME="$node" \
  SLURM_JOB_NUM_NODES=1 \
  SLURM_NTASKS=1 \
  SLURM_CPUS_ON_NODE=9 \
  SLURM_CPUS_PER_TASK=9 \
  SLURM_RESTART_COUNT=0 \
  SLURM_EXPORT_ENV=NIL \
  bash "$slurm_job" "$config" > "$build_dir/wrapper.stdout.txt" \
  2> "$build_dir/wrapper.stderr.txt"

mkdir "$build_dir/published"
tar -C "$build_dir/published" -xf "$build_dir/result.tar"
cat "$build_dir/published/transport/runner.stdout.txt"
cat "$build_dir/published/transport/retained-audit.stdout.txt"
grep -Fxq 'AUDIT_PASS=true' "$build_dir/published/transport/retained-audit.stdout.txt"
grep -Fxq 'ATTEMPTS_RECONSTRUCTED=9' "$build_dir/published/transport/retained-audit.stdout.txt"
grep -Fxq 'WORKER_REPLAYS=9' "$build_dir/published/transport/retained-audit.stdout.txt"
grep -Fxq 'V7_B_WINNER=NONE' "$build_dir/published/transport/retained-audit.stdout.txt"
python3 -B "$retained" --repo "$repo_root" --allow-synthetic-gate \
  "$build_dir/published/result" > "$build_dir/independent-retained.stdout.txt"
grep -Fxq 'AUDIT_PASS=true' "$build_dir/independent-retained.stdout.txt"
grep -Fxq 'WORKER_REPLAYS=9' "$build_dir/independent-retained.stdout.txt"
echo "TRANSPORT_SMOKE=passed"
echo "FULL_SMOKE=passed"
echo "V7A1_GATE_PASS=true"
