#!/usr/bin/env bash
#SBATCH --job-name=granite-l4-preflight
#SBATCH --no-requeue
#SBATCH --partition=gpu-orangefs
#SBATCH --nodelist=gpuorangefs-r770-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --gres=none
#SBATCH --time=00:30:00
#SBATCH --chdir=/tmp
#SBATCH --comment=lane:llm-granite
#SBATCH --output=/tmp/granite-l4-preflight-%j.log
set -euo pipefail
stage=/orangefs/training/llm-granite-n10
model="$stage/granite-3.1-8b-instruct"
work=/tmp/granite-l4-preflight-$SLURM_JOB_ID
mkdir -p "$work"
rec(){ srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | tr '\n\t' '  ' | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
trap 'rc=$?; rec "l4-preflight-exit-$rc"; if [[ -w "$stage" ]]; then mkdir -p "$stage/preflight-$SLURM_JOB_ID"; cp "$work/"*.log "$stage/preflight-$SLURM_JOB_ID/" 2>/dev/null || true; fi' EXIT
rec "node-$(hostname)"
[[ -r "$model/SHA256SUMS" && -r "$model/MODEL_REVISION" ]]
[[ $(cat "$model/MODEL_REVISION") == 4009206d5fc95d2e65a7b7633e159d6e97e25d35 ]]
[[ $(sha256sum "$model/SHA256SUMS" | cut -d' ' -f1) == 10edfe81703521bea67e7a285388590c6b740a639665411c76157d82a83609e6 ]]
rec "model-manifest-visible-$(sha256sum "$model/SHA256SUMS" | cut -d' ' -f1)"
(cd "$model" && sha256sum -c SHA256SUMS) > "$work/weights.log" 2>&1
rec L4_WEIGHT_SHA256SUMS_PASS
# Only recover the exact registered file; never synthesize a replacement corpus.
shared="$stage/v2_480_prompts.txt"
if [[ -s "$shared" ]]; then
  [[ $(grep -cve '^[[:space:]]*$' "$shared") -eq 480 ]]
  rec "prompts-sha-$(sha256sum "$shared" | cut -d' ' -f1)"
else
  find /tmp /scratch -maxdepth 3 -type f -name v2_480_prompts.txt 2>/dev/null | sort -u > "$work/prompt-candidates.log" || true
  candidate=""
  digest=""
  while IFS= read -r file; do
    [[ -r "$file" ]] || continue
    if [[ $(grep -cve '^[[:space:]]*$' "$file") -eq 480 ]]; then
      got=$(sha256sum "$file" | cut -d' ' -f1)
      if [[ -n "$digest" && "$digest" != "$got" ]]; then rec L4_PROMPT_COPY_CONFLICT; exit 2; fi
      candidate="$file"; digest="$got"
    fi
  done < "$work/prompt-candidates.log"
  if [[ -n "$candidate" && -w "$stage" ]]; then
    cp "$candidate" "$shared.tmp-$SLURM_JOB_ID"
    [[ $(sha256sum "$shared.tmp-$SLURM_JOB_ID" | cut -d' ' -f1) == "$digest" ]]
    if [[ ! -e "$shared" ]]; then mv -n "$shared.tmp-$SLURM_JOB_ID" "$shared"; fi
    [[ $(sha256sum "$shared" | cut -d' ' -f1) == "$digest" ]]
    rec "prompts-sha-$digest"
  else rec L4_ORIGINAL_PROMPTS_NOT_FOUND; fi
fi
exit 0
__CA_BUNDLE__
