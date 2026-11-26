#!/usr/bin/env bash
#SBATCH --job-name=granite-collect-receipt
#SBATCH --no-requeue
#SBATCH --comment=lane:llm-granite
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --gres=none
#SBATCH --time=00:15:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/granite-collect-%j.log
set -euo pipefail
stage=/orangefs/training/llm-granite-n10
rtx=${1:?RTX audit job id}
l4=${5:-}
ref_job=${l4:-$rtx}
a5=${2:?A5000 audit job id}
joblist=${3:?comma-separated dependency job ids}
head_c=${4:?pinned audit commit}
out="$stage/cluster-receipt-$SLURM_JOB_ID.txt"
{
  printf 'Audit source commit: %s\nCollector job: %s\n' "$head_c" "$SLURM_JOB_ID"
  date -u
  sacct -j "$joblist" --format=JobID,JobName%120,State,Elapsed,MaxRSS,ExitCode,NodeList -P
  sinfo -N -o '%N %T %G'
  model="$stage/granite-3.1-8b-instruct"
  if [[ -s "$model/SHA256SUMS" ]]; then
    printf '\nModel revision and staged manifest (stage job verified bytes):\n'
    cat "$model/MODEL_REVISION" "$model/SHA256SUMS"
    sha256sum "$model/SHA256SUMS"
  fi
  for j in "$l4" "$rtx" "$a5"; do
    [[ -n "$j" ]] || continue
    art="$stage/audit-$j"
    printf '\nAudit job %s artifacts:\n' "$j"
    if [[ -s "$art/SHA256SUMS" ]]; then
      cat "$art/SHA256SUMS"
      if sha256sum -c "$art/SHA256SUMS"; then printf 'artifact_hash_check=PASS\n'; else printf 'artifact_hash_check=FAIL\n'; fi
    else printf 'No completed audit manifest\n'; fi
    if [[ -s "$art/partial.sha256" ]]; then cat "$art/partial.sha256"; fi
    if [[ -s "$art/exit-code" ]]; then printf 'exit-code='; cat "$art/exit-code"; fi
    if [[ -s "$art/cmp.out" ]]; then cat "$art/cmp.out"; fi
  done
  if [[ -s "$stage/audit-$ref_job/cuda.tsv" ]]; then
    printf '\nREF_TSV for Mac lane: %s\n' "$stage/audit-$ref_job/cuda.tsv"
    sha256sum "$stage/audit-$ref_job/cuda.tsv"
  else printf '\nREF_TSV unavailable; no complete reference run.\n'; fi
  printf '\nA divergent tuple is not a localized causal diagnosis. No fix is applied by this collector.\n'
} > "$out.tmp"
mv "$out.tmp" "$out"
sha256sum "$out" > "$out.sha256"
exit 0
__CA_BUNDLE__
