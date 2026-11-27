#!/usr/bin/env bash
#SBATCH --job-name=granite-binary-recover
#SBATCH --partition=gpu-orangefs
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=none
#SBATCH --time=00:10:00
#SBATCH --chdir=/tmp
#SBATCH --comment=lane:llm-granite
#SBATCH --output=/tmp/granite-binary-recover-%j.log
set -euo pipefail
stage=/orangefs/training/llm-granite-n10
rec(){ srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
for j in "$@"; do
  dst=$stage/provenance-$j
  mkdir -p "$dst"
  for from in "$stage/audit-$j" "$stage/l4-ref-$j" "/tmp/llm-granite-parity-$j"; do
    [[ -d "$from" ]] || continue
    for f in audit madaros build-audit.log build-madaros.log inputs.sha256 run.log cuda.tsv cuda.partial.tsv exit-code exit-hang-status.log; do
      [[ -f "$from/$f" ]] || continue
      if [[ -e "$dst/$f" ]]; then cmp -s "$from/$f" "$dst/$f" || rec "artifact-conflict-$j-$f"; else cp "$from/$f" "$dst/$f"; fi
    done
  done
  for f in audit madaros build-audit.log build-madaros.log inputs.sha256 run.log cuda.tsv cuda.partial.tsv exit-code; do
    if [[ -f "$dst/$f" ]]; then
      rec "$j-$f-$(sha256sum "$dst/$f" | cut -d' ' -f1)"
      if [[ "$f" == cuda.tsv || "$f" == cuda.partial.tsv ]]; then rec "$j-$f-lines-$(wc -l < "$dst/$f")"; fi
    else rec "$j-$f-unavailable"; fi
  done
  if [[ -f "$dst/run.log" ]]; then tail -12 "$dst/run.log" | while IFS= read -r l; do rec "$l"; done; fi
  if [[ -f "$dst/inputs.sha256" ]]; then while read -r l; do rec "$l"; done < "$dst/inputs.sha256"; fi
  (cd "$dst" && find . -maxdepth 1 -type f ! -name 'SHA256SUMS*' -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS)
done
exit 0
__CA_BUNDLE__
