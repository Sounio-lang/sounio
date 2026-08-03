#!/usr/bin/env bash
#SBATCH --job-name=cs6-v7b-t23-cover
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
  echo "V7-B target-23 cover Slurm error: $*" >&2
  exit 1
}

[[ $# -eq 2 ]] || fail "usage: $0 CONFIG EXPECTED_CONFIG_SHA256"
config_source=$1
expected_config_sha=$2
[[ $expected_config_sha =~ ^[0-9a-f]{64}$ ]] || fail "invalid config digest"
[[ $config_source == /orangefs/training/* && -f $config_source && ! -L $config_source ]] || \
  fail "config must be a regular OrangeFS file"
[[ ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ ]] || fail "missing Slurm job id"
[[ ${SLURM_JOB_NAME:-} == cs6-v7b-t23-cover ]] || fail "job name drifted"
[[ ${SLURM_JOB_NODELIST:-} == gpuorangefs-r770-proxmox ]] || fail "node drifted"
[[ ${SLURM_CPUS_PER_TASK:-} == 32 ]] || fail "CPU request drifted"

work=$(mktemp -d "/tmp/cs6-v7b-t23-cover-${SLURM_JOB_ID}.XXXXXXXX")
cleanup() {
  chmod -R u+w "$work" 2>/dev/null || true
  rm -rf -- "$work"
}
trap cleanup EXIT

config="$work/config.txt"
cp --no-preserve=mode,ownership,timestamps "$config_source" "$config"
[[ $(sha256sum "$config" | awk '{print $1}') == "$expected_config_sha" ]] || \
  fail "config digest mismatch"

declare -A cfg=()
while IFS= read -r line || [[ -n $line ]]; do
  [[ -n $line && $line != *$'\r'* && $line == *=* ]] || fail "malformed config"
  [[ ${line//[^=]/} == = ]] || fail "config rows require one equals sign"
  key=${line%%=*}
  value=${line#*=}
  [[ $key =~ ^[A-Z0-9_]+$ && -n $value && -z ${cfg[$key]+present} ]] || \
    fail "unsafe, empty, or duplicate config field"
  cfg[$key]=$value
done < "$config"

required=(
  SCHEMA PAYLOAD_ARCHIVE PAYLOAD_SHA256 EXPECTED_GIT_HEAD EXPECTED_CONTRACT_SHA256
  EXPECTED_JOB_SCRIPT_SHA256 EXPECTED_WORKER_SHA256 OUTPUT_ARCHIVE JOBS TIMEOUT_SECONDS
)
[[ ${#cfg[@]} -eq ${#required[@]} ]] || fail "config field count mismatch"
for key in "${required[@]}"; do
  [[ -n ${cfg[$key]+present} ]] || fail "missing config field: $key"
done
[[ ${cfg[SCHEMA]} == sounio.cs6.v7b-target23-depth4-cover-slurm-config.v1 ]] || \
  fail "config schema mismatch"
for key in PAYLOAD_SHA256 EXPECTED_GIT_HEAD EXPECTED_CONTRACT_SHA256 EXPECTED_JOB_SCRIPT_SHA256 EXPECTED_WORKER_SHA256; do
  if [[ $key == EXPECTED_GIT_HEAD ]]; then
    [[ ${cfg[$key]} =~ ^[0-9a-f]{40}$ ]] || fail "invalid $key"
  else
    [[ ${cfg[$key]} =~ ^[0-9a-f]{64}$ ]] || fail "invalid $key"
  fi
done
[[ ${cfg[JOBS]} == 32 && ${cfg[TIMEOUT_SECONDS]} == 120 ]] || fail "runtime options drifted"
for key in PAYLOAD_ARCHIVE OUTPUT_ARCHIVE; do
  [[ ${cfg[$key]} == /orangefs/training/* ]] || fail "$key escapes OrangeFS"
done
[[ -f ${cfg[PAYLOAD_ARCHIVE]} && ! -L ${cfg[PAYLOAD_ARCHIVE]} ]] || fail "payload missing"
[[ $(sha256sum "${cfg[PAYLOAD_ARCHIVE]}" | awk '{print $1}') == ${cfg[PAYLOAD_SHA256]} ]] || \
  fail "payload digest mismatch"
executed_job_sha=$(sha256sum "$0" | awk '{print $1}')
[[ $executed_job_sha == ${cfg[EXPECTED_JOB_SCRIPT_SHA256]} ]] || fail "job script digest mismatch"

tar -xf "${cfg[PAYLOAD_ARCHIVE]}" -C "$work"
repo="$work/repo"
worker="$work/worker-binary"
[[ -d $repo && -x $worker ]] || fail "payload layout invalid"
[[ $(cat "$work/git-head.txt") == ${cfg[EXPECTED_GIT_HEAD]} ]] || fail "git head mismatch"
[[ $(sha256sum "$worker" | awk '{print $1}') == ${cfg[EXPECTED_WORKER_SHA256]} ]] || \
  fail "worker digest mismatch"
contract="$repo/scripts/research/cs6_v7b_target23_depth4_cover_contract_v1.txt"
[[ $(sha256sum "$contract" | awk '{print $1}') == ${cfg[EXPECTED_CONTRACT_SHA256]} ]] || \
  fail "contract digest mismatch"

result="$work/result"
mkdir -p "$result/provenance"
cd "$repo"
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/research/cs6_v7b_target23_depth4_cover_run.py \
  --out-dir "$result" \
  --binary "$worker" \
  --jobs "${cfg[JOBS]}" \
  --timeout "${cfg[TIMEOUT_SECONDS]}" | tee "$result/run.stdout.txt"
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/research/cs6_v7b_target23_depth4_cover_verify.py \
  "$result" | tee "$result/verification.txt"

scontrol -o show job "$SLURM_JOB_ID" > "$result/provenance/slurm-control-plane.txt"
cp "$config" "$result/provenance/config.txt"
printf '%s\n' "${cfg[EXPECTED_GIT_HEAD]}" > "$result/provenance/git-head.txt"
printf '%s  worker-binary\n' "${cfg[EXPECTED_WORKER_SHA256]}" > "$result/provenance/worker.sha256"
printf 'SLURM_JOB_ID=%s\nSLURM_JOB_NODELIST=%s\nSLURM_CPUS_PER_TASK=%s\n' \
  "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$SLURM_CPUS_PER_TASK" \
  > "$result/provenance/slurm-context.txt"

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
