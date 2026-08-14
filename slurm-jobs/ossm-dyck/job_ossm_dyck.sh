#!/usr/bin/env bash
#SBATCH -p gpu-orangefs
#SBATCH -A plruntime
#SBATCH --qos=burst
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -J ossm-dyck-gpu
#SBATCH --time=04:00:00
#SBATCH --mem=64G

set -euo pipefail

REPO="/workspace/sounio"
OUT_DIR="${REPO}/artifacts/ossm_dyck"
mkdir -p "${OUT_DIR}"

echo "=== OSSM Dyck Scaling Experiment ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# Ensure PyTorch with CUDA is available
pip3 install torch --break-system-packages --quiet 2>/dev/null || true
python3 -c "import torch; print('PyTorch', torch.__version__, 'CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
echo ""

cd "${REPO}"

# Dyck-1: single bracket type, lengths 32 to 1024
echo "=== Dyck-1 (lengths 32..1024) ==="
python3 scripts/research/ossm_dyck_scaling.py \
  --lengths 32 64 128 256 512 1024 \
  --n-types 1 \
  --epochs 100 \
  --train-size 4096 \
  --test-size 1024 \
  --gpu \
  --seed 20260806 \
  2>&1 | tee "${OUT_DIR}/dyck1_${SLURM_JOB_ID}.log"

echo ""
echo "=== Dyck-2 (lengths 32..512) ==="
python3 scripts/research/ossm_dyck_scaling.py \
  --lengths 32 64 128 256 512 \
  --n-types 2 \
  --epochs 100 \
  --train-size 4096 \
  --test-size 1024 \
  --gpu \
  --seed 20260806 \
  2>&1 | tee "${OUT_DIR}/dyck2_${SLURM_JOB_ID}.log"

cp scripts/research/ossm_dyck_results.json "${OUT_DIR}/ossm_dyck_results_${SLURM_JOB_ID}.json" 2>/dev/null || true

echo ""
echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "DONE"
