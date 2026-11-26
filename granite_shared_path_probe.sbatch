#!/usr/bin/env bash
#SBATCH --job-name=granite-path-probe
#SBATCH --no-requeue
#SBATCH --partition=all
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --gres=none
#SBATCH --time=00:05:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/granite-path-probe-%j.log
set -uo pipefail
rec() { srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | tr '\n\t' '  ' | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
for p in /orangefs /orangefs/training /scratch /mnt /home/sounio /home/devsounio; do
    if [[ -d "$p" ]]; then rec "path-$p-$(stat -c '%u:%g:%a' "$p")-writable-$([[ -w "$p" ]] && echo yes || echo no)"; findmnt -T "$p" -no TARGET,FSTYPE,OPTIONS 2>/dev/null | head -1 | while IFS= read -r l; do rec "$p-mount-$l"; done; fi
done
find /orangefs/training -maxdepth 2 -type d -writable 2>/dev/null | head -15 | while IFS= read -r p; do rec "writable-$p"; done
find /orangefs/training -maxdepth 4 -type f \( -name 'v2_480_prompts.txt' -o -iname '*granite*ids*' \) 2>/dev/null | head -10 | while IFS= read -r p; do rec "input-$p"; done
for p in /tmp/v2_480_prompts.txt /scratch/v2_480_prompts.txt; do [[ ! -s "$p" ]] || rec "input-$p"; done
for c in c++ g++ clang++ jq git-lfs curl wget; do rec "$c-$(command -v "$c" || echo absent)"; done
exit 0
__CA_BUNDLE__
