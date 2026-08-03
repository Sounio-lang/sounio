#!/bin/bash
#SBATCH --job-name=cs6-v7b-t23-arb
#SBATCH --partition=gpu-orangefs
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --nodelist=gpuorangefs-multi-r740-proxmox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --export=ALL

set -euo pipefail

fail() {
  echo "V7-B target-23 Arb validated Slurm error: $*" >&2
  exit 1
}

[[ $# -eq 2 ]] || fail "usage: $0 CONFIG EXPECTED_CONFIG_SHA256"
config_source=$1
expected_config_sha=$2
[[ $expected_config_sha =~ ^[0-9a-f]{64}$ ]] || fail "invalid config digest"
[[ $config_source == /tmp/cs6-v7b-t23-arb-stage-* && -f $config_source && ! -L $config_source ]] || \
  fail "config must be a regular worker-local staging file"
[[ ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ ]] || fail "missing Slurm job id"
[[ ${SLURM_JOB_NAME:-} == cs6-v7b-t23-arb ]] || fail "job name drifted"
[[ ${SLURM_CPUS_PER_TASK:-} == 4 ]] || fail "CPU request drifted"

work=$(mktemp -d "/tmp/cs6-v7b-t23-arb-${SLURM_JOB_ID}.XXXXXXXX")
cleanup() {
  chmod -R u+w "$work" 2>/dev/null || true
  rm -rf -- "$work"
}
trap cleanup EXIT

config="$work/config.txt"
cp --no-preserve=mode,ownership,timestamps "$config_source" "$config"
[[ $(sha256sum "$config" | awk '{print $1}') == "$expected_config_sha" ]] || fail "config digest mismatch"

declare -A cfg=()
while IFS= read -r line || [[ -n $line ]]; do
  [[ -n $line && $line != *$'\r'* && $line == *=* ]] || fail "malformed config"
  [[ ${line//[^=]/} == = ]] || fail "config rows require one equals sign"
  key=${line%%=*}; value=${line#*=}
  [[ $key =~ ^[A-Z0-9_]+$ && -n $value && -z ${cfg[$key]+present} ]] || fail "unsafe or duplicate config field"
  cfg[$key]=$value
done < "$config"

required=(
  SCHEMA PAYLOAD_ARCHIVE PAYLOAD_SHA256 PYTHON_FLINT_WHEEL PYTHON_FLINT_WHEEL_SHA256
  EXPECTED_GIT_HEAD EXPECTED_CONTRACT_SHA256 EXPECTED_CAPD_RESULTS_SHA256
  EXPECTED_WORKER_SHA256 EXPECTED_VERIFIER_SHA256 EXPECTED_JOB_SCRIPT_SHA256
  EXPECTED_NODE OUTPUT_ARCHIVE RETURN_HOST RETURN_PORT WORKER_TIMEOUT_SECONDS
)
[[ ${#cfg[@]} -eq ${#required[@]} ]] || fail "config field count mismatch"
for key in "${required[@]}"; do [[ -n ${cfg[$key]+present} ]] || fail "missing config field: $key"; done
[[ ${cfg[SCHEMA]} == sounio.cs6.v7b-target23-arb-validated-minmargin-slurm-config.v1 ]] || fail "config schema mismatch"
for key in PAYLOAD_SHA256 PYTHON_FLINT_WHEEL_SHA256 EXPECTED_CONTRACT_SHA256 EXPECTED_CAPD_RESULTS_SHA256 EXPECTED_WORKER_SHA256 EXPECTED_VERIFIER_SHA256 EXPECTED_JOB_SCRIPT_SHA256; do
  [[ ${cfg[$key]} =~ ^[0-9a-f]{64}$ ]] || fail "invalid $key"
done
[[ ${cfg[EXPECTED_GIT_HEAD]} =~ ^[0-9a-f]{40}$ ]] || fail "invalid Git identity"
[[ ${cfg[EXPECTED_NODE]} == gpuorangefs-multi-r740-proxmox ]] || fail "unsupported node"
[[ ${SLURM_JOB_NODELIST:-} == ${cfg[EXPECTED_NODE]} ]] || fail "node drifted"
[[ ${cfg[WORKER_TIMEOUT_SECONDS]} == 300 ]] || fail "worker timeout drifted"
[[ ${cfg[RETURN_HOST]} =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || fail "invalid return host"
[[ ${cfg[RETURN_PORT]} =~ ^[1-9][0-9]{3,4}$ && ${cfg[RETURN_PORT]} -le 65535 ]] || fail "invalid return port"
for key in PAYLOAD_ARCHIVE PYTHON_FLINT_WHEEL OUTPUT_ARCHIVE; do
  [[ ${cfg[$key]} == /tmp/cs6-v7b-t23-arb-stage-* ]] || fail "$key escapes worker-local staging"
done
[[ -f ${cfg[PAYLOAD_ARCHIVE]} && ! -L ${cfg[PAYLOAD_ARCHIVE]} ]] || fail "payload missing"
[[ -f ${cfg[PYTHON_FLINT_WHEEL]} && ! -L ${cfg[PYTHON_FLINT_WHEEL]} ]] || fail "wheel missing"
[[ $(sha256sum "${cfg[PAYLOAD_ARCHIVE]}" | awk '{print $1}') == ${cfg[PAYLOAD_SHA256]} ]] || fail "payload digest mismatch"
[[ $(sha256sum "${cfg[PYTHON_FLINT_WHEEL]}" | awk '{print $1}') == ${cfg[PYTHON_FLINT_WHEEL_SHA256]} ]] || fail "wheel digest mismatch"
[[ $(sha256sum "$0" | awk '{print $1}') == ${cfg[EXPECTED_JOB_SCRIPT_SHA256]} ]] || fail "job script digest mismatch"

tar -xf "${cfg[PAYLOAD_ARCHIVE]}" -C "$work"
repo="$work/repo"
[[ -d $repo && $(cat "$work/git-head.txt") == ${cfg[EXPECTED_GIT_HEAD]} ]] || fail "payload layout or Git identity invalid"
contract="$repo/scripts/research/cs6_v7b_target23_arb_validated_minmargin_contract_v1.txt"
capd_results="$repo/scripts/research/receipts/cs6_v7b_target23_decimal_center_replay_v1/results.tsv"
worker="$repo/scripts/research/cs6_v7b_target23_arb_validated_minmargin_worker.py"
verifier="$repo/scripts/research/cs6_v7b_target23_arb_validated_minmargin_verify.py"
[[ $(sha256sum "$contract" | awk '{print $1}') == ${cfg[EXPECTED_CONTRACT_SHA256]} ]] || fail "contract drifted"
[[ $(sha256sum "$capd_results" | awk '{print $1}') == ${cfg[EXPECTED_CAPD_RESULTS_SHA256]} ]] || fail "CAPD comparison source drifted"
[[ $(sha256sum "$worker" | awk '{print $1}') == ${cfg[EXPECTED_WORKER_SHA256]} ]] || fail "worker drifted"
[[ $(sha256sum "$verifier" | awk '{print $1}') == ${cfg[EXPECTED_VERIFIER_SHA256]} ]] || fail "verifier drifted"

deps="$work/deps"
mkdir -p "$deps"
wheel="$work/python_flint-0.8.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
cp --no-preserve=mode,ownership,timestamps "${cfg[PYTHON_FLINT_WHEEL]}" "$wheel"
[[ $(sha256sum "$wheel" | awk '{print $1}') == ${cfg[PYTHON_FLINT_WHEEL_SHA256]} ]] || fail "private wheel digest mismatch"
PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install --no-index --no-deps \
  --target "$deps" "$wheel" > "$work/pip-install.txt" 2>&1
python_exec=$(command -v python3)
flint_version=$(PYTHONPATH="$deps" "$python_exec" -c 'import flint; print(flint.__version__)')
[[ $flint_version == 0.8.0 ]] || fail "installed python-flint version drifted"
extension=$(find "$deps/flint" -maxdepth 1 -type f -name 'pyflint*.so' -print -quit)
[[ -n $extension && -f $extension ]] || fail "python-flint extension missing"
extension_sha=$(sha256sum "$extension" | awk '{print $1}')

result="$work/result"
mkdir -p "$result/provenance"
cp "$config" "$result/provenance/config.txt"
cp "$work/pip-install.txt" "$result/provenance/pip-install.txt"
printf '%s\n' "${cfg[EXPECTED_GIT_HEAD]}" > "$result/provenance/git-head.txt"
printf 'SLURM_JOB_ID=%s\nSLURM_JOB_NODELIST=%s\nSLURM_CPUS_PER_TASK=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$SLURM_CPUS_PER_TASK" > "$result/provenance/slurm-context.txt"
printf 'PYTHON_VERSION=%s\nPYTHON_EXECUTABLE=%s\n' \
  "$($python_exec -c 'import platform; print(platform.python_version())')" "$python_exec" \
  > "$result/provenance/python-runtime.txt"
printf 'PYTHON_FLINT_WHEEL_SHA256=%s\nPYTHON_FLINT_VERSION=%s\nFLINT_EXTENSION_SHA256=%s\n' \
  "${cfg[PYTHON_FLINT_WHEEL_SHA256]}" "$flint_version" "$extension_sha" \
  > "$result/provenance/dependency-attestation.txt"
ldd "$extension" > "$result/provenance/flint-extension-linkage.txt"

cd "$repo"
read -r challenge binding < <(PYTHONDONTWRITEBYTECODE=1 python3 -B - <<PY
from pathlib import Path
from scripts.research.cs6_v7b_target23_arb_validated_minmargin_verify import expected_bindings
print(*expected_bindings(Path('.'), '${cfg[EXPECTED_GIT_HEAD]}', '${cfg[PYTHON_FLINT_WHEEL_SHA256]}'))
PY
)
printf '%s -B %s %s %s\n' "$python_exec" "$worker" "$challenge" "$binding" > "$result/worker-command.txt"
set +e
timeout "${cfg[WORKER_TIMEOUT_SECONDS]}" env PYTHONPATH="$deps" "$python_exec" -B "$worker" \
  "$challenge" "$binding" > "$result/worker.stdout.txt" 2> "$result/worker.stderr.txt"
worker_rc=$?
set -e
stdout_sha=$(sha256sum "$result/worker.stdout.txt" | awk '{print $1}')
stderr_sha=$(sha256sum "$result/worker.stderr.txt" | awk '{print $1}')
printf 'SCHEMA=sounio.cs6.v7b-target23-arb-validated-minmargin-execution.v1\nPRE_EXECUTION_GIT_COMMIT=%s\nWORKER_RC=%s\nSTDOUT_SHA256=%s\nSTDERR_SHA256=%s\nRUN_COMPLETE=%s\nINDEPENDENT_VALIDATED_CENTER_ORBIT_CERTIFICATE=false\nLEAF_WIDE_CERTIFICATE=false\nINDEPENDENT_FULL_LEAF_INTERVAL_ENGINE=false\nGLOBAL_HPG_CERTIFICATE=false\nV7_B_ELIGIBILITY=false\nPROMOTION_ELIGIBLE=false\nOPEN_PROBLEM_SOLVED=false\nNOVELTY_OR_PRIORITY_CLAIMED=false\nFPGA_EXECUTION=false\n' \
  "${cfg[EXPECTED_GIT_HEAD]}" "$worker_rc" "$stdout_sha" "$stderr_sha" \
  "$([[ $worker_rc == 0 ]] && echo true || echo false)" > "$result/execution-summary.txt"
[[ $worker_rc == 0 ]] || fail "validated worker failed with rc=$worker_rc"

PYTHONDONTWRITEBYTECODE=1 python3 -B "$verifier" "$result" \
  --source-commit "${cfg[EXPECTED_GIT_HEAD]}" \
  --wheel-sha256 "${cfg[PYTHON_FLINT_WHEEL_SHA256]}" | tee "$result/verification.txt"
scontrol -o show job "$SLURM_JOB_ID" > "$result/provenance/slurm-control-plane.txt"

archive_tmp="$work/result.tar"
tar -cf "$archive_tmp" -C "$result" .
archive_sha=$(sha256sum "$archive_tmp" | awk '{print $1}')
output=${cfg[OUTPUT_ARCHIVE]}
cp "$archive_tmp" "$output.tmp-$SLURM_JOB_ID"
mv "$output.tmp-$SLURM_JOB_ID" "$output"
printf '%s  %s\n' "$archive_sha" "$(basename "$output")" > "$output.sha256.tmp-$SLURM_JOB_ID"
mv "$output.sha256.tmp-$SLURM_JOB_ID" "$output.sha256"
printf 'RESULT_ARCHIVE=%s\nRESULT_SHA256=%s\n' "$output" "$archive_sha"

archive_bytes=$(stat -c '%s' "$output")
timeout 120 bash -c '
  set -euo pipefail
  exec 3<>"/dev/tcp/$1/$2"
  printf "SOUNIO_CS6_V7B_ARB_RESULT_V1 %s %s\n" "$4" "$5" >&3
  cat "$3" >&3
  exec 3>&-
' _ "${cfg[RETURN_HOST]}" "${cfg[RETURN_PORT]}" "$output" "$archive_bytes" "$archive_sha" || fail "result return failed"
