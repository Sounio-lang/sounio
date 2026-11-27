#!/usr/bin/env bash
#SBATCH --job-name=granite-cli-probe
#SBATCH --comment=lane:llm-granite
#SBATCH --no-requeue
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --gres=none
#SBATCH --time=00:05:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/granite-cli-probe-%j.log
set -uo pipefail
rec() { srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | tr '\n\t' '  ' | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
audit=/tmp/granite-runtime-check-29243/llm_v2_61440_audit
[[ -x "$audit" ]] || exit 1
SOUNIO_AUDIT_BACKEND=cuda "$audit" --arch granite --prompts 1 > /tmp/granite-cli-probe-$SLURM_JOB_ID.log1 2>&1
rc=$?
rec "registered-shape-refusal-rc-$rc"
grep -E 'AUDIT_|GRANITE_' /tmp/granite-cli-probe-$SLURM_JOB_ID.log1 | head -5 | while IFS= read -r l; do rec "$l"; done
SOUNIO_AUDIT_BACKEND=cuda "$audit" --arch unknown > /tmp/granite-cli-probe-$SLURM_JOB_ID.log2 2>&1
rc=$?
rec "unknown-arch-refusal-rc-$rc"
grep -E 'AUDIT_|GRANITE_' /tmp/granite-cli-probe-$SLURM_JOB_ID.log2 | head -5 | while IFS= read -r l; do rec "$l"; done
exit 0
__CA_BUNDLE__
