#!/usr/bin/env bash
#SBATCH -p gpu-orangefs
#SBATCH -A plruntime
#SBATCH --qos=burst
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -J san-large-gpt
#SBATCH --time=08:00:00
#SBATCH --mem=64G

set -euo pipefail

REPO="/workspace/sounio"
OUT_DIR="${REPO}/artifacts/san_large/gpu_runs"
mkdir -p "${OUT_DIR}"

echo "=== SAN Large Architecture — GPT decoder on corpus ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pip3 install torch --break-system-packages --quiet 2>/dev/null || true
python3 -c "import torch; print('PyTorch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
echo ""

cd "${REPO}"

export SAN_LARGE_ONLY=gpt
export SAN_LARGE_THREADS=16
export SAN_LARGE_TAU_GPT=0.168
export SAN_LARGE_DELTA_GPT=0.31
export SAN_LARGE_DETACH_AUX=1

python3 scripts/research/suffering_aware_large_architecture.py 2>&1 | \
  tee "${OUT_DIR}/gpt_${SLURM_JOB_ID}.log"

echo ""
echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "DONE"
