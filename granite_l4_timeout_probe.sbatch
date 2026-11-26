#!/usr/bin/env bash
#SBATCH --job-name=granite-l4-timeout-probe
#SBATCH --partition=gpu-orangefs
#SBATCH --nodelist=gpuorangefs-r770-proxmox
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --comment=lane:llm-granite
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/granite-l4-timeout-probe-%j.log
set -euo pipefail
stage=/orangefs/training/llm-granite-n10
art=$stage/timeout-probe-$SLURM_JOB_ID
mkdir -p "$art"
rec(){ srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
for p in "$stage/ops-29516/cuda.log" /tmp/granite-tail-29516/cuda.log /tmp/granite-tail-29516.log; do
    if [[ -f "$p" ]]; then
        rec "prior-log-$(basename "$p")-bytes-$(stat -c %s "$p")"
        cp "$p" "$art/$(basename "$p")"
        tail -20 "$p" | while IFS= read -r l; do [[ -z "$l" ]] || rec "$l"; done
    else rec "prior-log-unavailable-$(basename "$p")"; fi
done
if timeout 30s nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader > "$art/gpu.txt" 2>&1; then
    while IFS= read -r l; do rec "GPU-$l"; done < "$art/gpu.txt"
else rec NVIDIA_SMI_FAILED; exit 1; fi
exit 0
__CA_BUNDLE__
