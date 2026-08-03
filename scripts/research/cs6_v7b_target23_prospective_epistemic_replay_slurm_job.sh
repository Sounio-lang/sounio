#!/usr/bin/env bash
#SBATCH --job-name=cs6-v7b-t23-prosp
#SBATCH --partition=gpu-orangefs
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --nodelist=gpuorangefs-r770-proxmox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --export=NIL

set -euo pipefail

fail() {
  echo "V7-B target-23 prospective replay Slurm error: $*" >&2
  exit 1
}

[[ $# -eq 2 ]] || fail "usage: $0 CONFIG EXPECTED_CONFIG_SHA256"
config_source=$1
expected_config_sha=$2
[[ $expected_config_sha =~ ^[0-9a-f]{64}$ ]] || fail "invalid config digest"
[[ $config_source == /tmp/cs6-v7b-t23-prosp-stage-* && -f $config_source && ! -L $config_source ]] || \
  fail "config must be a regular worker-local staging file"
[[ ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ ]] || fail "missing Slurm job id"
[[ ${SLURM_JOB_NAME:-} == cs6-v7b-t23-prosp ]] || fail "job name drifted"
[[ ${SLURM_CPUS_PER_TASK:-} == 32 ]] || fail "CPU request drifted"

work=$(mktemp -d "/tmp/cs6-v7b-t23-prosp-${SLURM_JOB_ID}.XXXXXXXX")
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
  key=${line%%=*}
  value=${line#*=}
  [[ $key =~ ^[A-Z0-9_]+$ && -n $value && -z ${cfg[$key]+present} ]] || fail "unsafe or duplicate config field"
  cfg[$key]=$value
done < "$config"

required=(
  SCHEMA PAYLOAD_ARCHIVE PAYLOAD_SHA256 EXPECTED_GIT_HEAD EXPECTED_CONTRACT_SHA256
  EXPECTED_MANIFEST_SHA256 EXPECTED_JOB_SCRIPT_SHA256 EXPECTED_WORKER_SOURCE_SHA256
  EXPECTED_WORKER_SHA256 PRE_EXECUTION_COMMIT_EPOCH WORKER_BUILD_EPOCH EXPECTED_NODE
  OUTPUT_ARCHIVE RETURN_HOST RETURN_PORT JOBS TIMEOUT_SECONDS
)
[[ ${#cfg[@]} -eq ${#required[@]} ]] || fail "config field count mismatch"
for key in "${required[@]}"; do [[ -n ${cfg[$key]+present} ]] || fail "missing config field: $key"; done
[[ ${cfg[SCHEMA]} == sounio.cs6.v7b-target23-prospective-replay-slurm-config.v1 ]] || fail "config schema mismatch"
for key in PAYLOAD_SHA256 EXPECTED_CONTRACT_SHA256 EXPECTED_MANIFEST_SHA256 EXPECTED_JOB_SCRIPT_SHA256 EXPECTED_WORKER_SOURCE_SHA256 EXPECTED_WORKER_SHA256; do
  [[ ${cfg[$key]} =~ ^[0-9a-f]{64}$ ]] || fail "invalid $key"
done
[[ ${cfg[EXPECTED_GIT_HEAD]} =~ ^[0-9a-f]{40}$ ]] || fail "invalid Git identity"
[[ ${cfg[PRE_EXECUTION_COMMIT_EPOCH]} =~ ^[1-9][0-9]*$ && ${cfg[WORKER_BUILD_EPOCH]} =~ ^[1-9][0-9]*$ ]] || fail "invalid build chronology"
(( cfg[WORKER_BUILD_EPOCH] > cfg[PRE_EXECUTION_COMMIT_EPOCH] )) || fail "worker was not built after frozen commit"
[[ ${cfg[JOBS]} == 32 && ${cfg[TIMEOUT_SECONDS]} == 120 ]] || fail "runtime options drifted"
[[ ${cfg[EXPECTED_NODE]} == gpuorangefs-r770-proxmox || ${cfg[EXPECTED_NODE]} == gpuorangefs-multi-r740-proxmox ]] || fail "unsupported node"
[[ ${SLURM_JOB_NODELIST:-} == ${cfg[EXPECTED_NODE]} ]] || fail "node drifted"
[[ ${cfg[RETURN_HOST]} =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || fail "invalid return host"
[[ ${cfg[RETURN_PORT]} =~ ^[1-9][0-9]{3,4}$ && ${cfg[RETURN_PORT]} -le 65535 ]] || fail "invalid return port"
for key in PAYLOAD_ARCHIVE OUTPUT_ARCHIVE; do
  [[ ${cfg[$key]} == /tmp/cs6-v7b-t23-prosp-stage-* ]] || fail "$key escapes worker-local staging"
done
[[ -f ${cfg[PAYLOAD_ARCHIVE]} && ! -L ${cfg[PAYLOAD_ARCHIVE]} ]] || fail "payload missing"
[[ $(sha256sum "${cfg[PAYLOAD_ARCHIVE]}" | awk '{print $1}') == ${cfg[PAYLOAD_SHA256]} ]] || fail "payload digest mismatch"
[[ $(sha256sum "$0" | awk '{print $1}') == ${cfg[EXPECTED_JOB_SCRIPT_SHA256]} ]] || fail "job script digest mismatch"

tar -xf "${cfg[PAYLOAD_ARCHIVE]}" -C "$work"
repo="$work/repo"
worker="$work/worker-binary"
[[ -d $repo && -x $worker ]] || fail "payload layout invalid"
[[ $(cat "$work/git-head.txt") == ${cfg[EXPECTED_GIT_HEAD]} ]] || fail "Git head mismatch"
[[ $(sha256sum "$worker" | awk '{print $1}') == ${cfg[EXPECTED_WORKER_SHA256]} ]] || fail "worker digest mismatch"
worker_source="$repo/scripts/research/cs6_hapg_liouville_carrier_ablation_probe.cpp"
contract="$repo/scripts/research/cs6_v7b_target23_prospective_epistemic_replay_contract_v1.txt"
manifest="$repo/scripts/research/cs6_v7b_target23_prospective_epistemic_replay_coordinates_v1.tsv"
[[ $(sha256sum "$worker_source" | awk '{print $1}') == ${cfg[EXPECTED_WORKER_SOURCE_SHA256]} ]] || fail "worker source drifted"
[[ $(sha256sum "$contract" | awk '{print $1}') == ${cfg[EXPECTED_CONTRACT_SHA256]} ]] || fail "contract drifted"
[[ $(sha256sum "$manifest" | awk '{print $1}') == ${cfg[EXPECTED_MANIFEST_SHA256]} ]] || fail "manifest drifted"

result="$work/result"
mkdir -p "$result/provenance"
cp "$config" "$result/provenance/config.txt"
printf '%s\n' "${cfg[EXPECTED_GIT_HEAD]}" > "$result/provenance/git-head.txt"
printf 'SLURM_JOB_ID=%s\nSLURM_JOB_NODELIST=%s\nSLURM_CPUS_PER_TASK=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$SLURM_CPUS_PER_TASK" > "$result/provenance/slurm-context.txt"
printf 'WORKER_SOURCE_SHA256=%s\nWORKER_BINARY_SHA256=%s\nPRE_EXECUTION_COMMIT_EPOCH=%s\nWORKER_BUILD_EPOCH=%s\nBUILD_AFTER_PRE_EXECUTION_COMMIT=true\n' \
  "${cfg[EXPECTED_WORKER_SOURCE_SHA256]}" "${cfg[EXPECTED_WORKER_SHA256]}" \
  "${cfg[PRE_EXECUTION_COMMIT_EPOCH]}" "${cfg[WORKER_BUILD_EPOCH]}" \
  > "$result/provenance/fresh-build-attestation.txt"
ldd "$worker" > "$result/provenance/worker-linkage.txt"

cd "$repo"
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/research/cs6_v7b_target23_prospective_epistemic_replay_run.py \
  --out-dir "$result" --source-commit "${cfg[EXPECTED_GIT_HEAD]}" --binary "$worker" \
  --jobs "${cfg[JOBS]}" --timeout "${cfg[TIMEOUT_SECONDS]}" | tee "$result/run.stdout.txt"
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/research/cs6_v7b_target23_prospective_epistemic_replay_verify.py \
  "$result" --source-commit "${cfg[EXPECTED_GIT_HEAD]}" | tee "$result/verification.txt"
scontrol -o show job "$SLURM_JOB_ID" > "$result/provenance/slurm-control-plane.txt"

archive_tmp="$work/result.tar"
tar -cf "$archive_tmp" -C "$result" .
archive_sha=$(sha256sum "$archive_tmp" | awk '{print $1}')
output=${cfg[OUTPUT_ARCHIVE]}
output_dir=$(dirname "$output")
[[ -d $output_dir && ! -L $output_dir ]] || fail "output directory missing"
cp "$archive_tmp" "$output.tmp-$SLURM_JOB_ID"
mv "$output.tmp-$SLURM_JOB_ID" "$output"
printf '%s  %s\n' "$archive_sha" "$(basename "$output")" > "$output.sha256.tmp-$SLURM_JOB_ID"
mv "$output.sha256.tmp-$SLURM_JOB_ID" "$output.sha256"
printf 'RESULT_ARCHIVE=%s\nRESULT_SHA256=%s\n' "$output" "$archive_sha"

archive_bytes=$(stat -c '%s' "$output")
timeout 120 bash -c '
  set -euo pipefail
  exec 3<>"/dev/tcp/$1/$2"
  printf "SOUNIO_CS6_V7B_PROSPECTIVE_RESULT_V1 %s %s\\n" "$4" "$5" >&3
  cat "$3" >&3
  exec 3>&-
' _ "${cfg[RETURN_HOST]}" "${cfg[RETURN_PORT]}" "$output" "$archive_bytes" "$archive_sha" || fail "result return failed"
