#!/usr/bin/env bash
#SBATCH --job-name=granite-stage-monitor
#SBATCH --no-requeue
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --gres=none
#SBATCH --time=00:05:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/granite-stage-monitor-%j.log
set -uo pipefail
rec() { srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | tr '\n\t' '  ' | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
stage=/orangefs/training/llm-granite-n10
model="$stage/granite-3.1-8b-instruct"
if [[ -d "$model" ]]; then
  rec "model-bytes-$(du -sb "$model" | cut -f1)"
  if [[ -d "$model/.git/lfs" ]]; then rec "lfs-bytes-$(du -sb "$model/.git/lfs" | cut -f1)"; fi
  for f in "$model"/*.safetensors; do [[ ! -f "$f" ]] || rec "$(basename "$f")-bytes-$(stat -c '%s' "$f")"; done
  for f in "$model"/tokenizer.json "$model"/tokenizer_config.json "$model"/config.json; do [[ ! -f "$f" ]] || rec "$(basename "$f")-bytes-$(stat -c '%s' "$f")"; done
  if [[ -s "$model/SHA256SUMS" ]]; then rec "SHA256SUMS-$(sha256sum "$model/SHA256SUMS" | cut -d' ' -f1)"; while IFS= read -r line; do rec "$line"; done < "$model/SHA256SUMS"; fi
fi
for f in "$stage"/stage-*.log; do
  [[ -f "$f" ]] || continue
  grep -E 'git-lfs/[0-9]|GLIBC_[0-9]|Cloning into|fatal: |Permission denied|No space left|[.]safetensors: OK|MODEL_REVISION: OK' "$f" | sed -E 's@https?://[^ ]+@[URL]@g' | head -10 | while IFS= read -r line; do rec "$line"; done
done
if [[ "${1:-}" == search ]]; then
  find /orangefs/training -maxdepth 6 -type f \( -iname '*480*prompt*' -o -iname '*prompt*480*' -o -iname '*61440*prompt*' -o -iname '*granite*ids*' \) 2>/dev/null | head -20 | while IFS= read -r p; do rec "input-$p"; done
fi
exit 0
__CA_BUNDLE__
