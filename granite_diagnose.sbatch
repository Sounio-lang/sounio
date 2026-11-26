#!/usr/bin/env bash
#SBATCH --job-name=granite-diagnose
#SBATCH --no-requeue
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=00:05:00
#SBATCH --gres=none
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/granite-diagnose-%j.log
set -uo pipefail
rec() { srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | tr '\n\t' '  ' | cut -c1-110)" true </dev/null >/dev/null 2>&1 || true; }
for p in /orangefs /orangefs/training /mnt/models /mnt/experiments /scratch; do
  if [[ -d "$p" ]]; then rec "exists-$p-writable-$([[ -w "$p" ]] && echo yes || echo no)"; else rec "absent-$p"; fi
done
for f in /tmp/granite-tail-29222/check*.log /tmp/granite-stage-29219.log; do
  if [[ -f "$f" ]]; then
    rec "$f"
    grep -iE 'error|fail|denied|not found|unavailable' "$f" | head -12 | while IFS= read -r line; do rec "$line"; done
    tail -5 "$f" | while IFS= read -r line; do rec "$line"; done
  fi
done
exit 0
__CA_BUNDLE__
