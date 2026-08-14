#!/usr/bin/env bash
#SBATCH -p gpu-orangefs
#SBATCH -A plruntime
#SBATCH --qos=burst
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -J ossm-length-gen
#SBATCH --time=04:00:00
#SBATCH --mem=64G

set -euo pipefail

REPO="/workspace/sounio"
OUT_DIR="${REPO}/artifacts/ossm_dyck"
mkdir -p "${OUT_DIR}"

echo "=== OSSM Length Generalization ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pip3 install torch --break-system-packages --quiet 2>/dev/null || true
python3 -c "import torch; print('PyTorch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
echo ""

cd "${REPO}"

# Dyck-1: train at L=256, test at L=256, 512, 1024, 4096
echo "=== Dyck-1: train L=256 → test L=256,512,1024,4096 ==="
python3 scripts/research/ossm_dyck_scaling.py \
  --generalize \
  --train-len 256 \
  --test-lens 256 512 1024 4096 \
  --n-types 1 \
  --epochs 100 \
  --train-size 4096 \
  --test-size 1024 \
  --gpu \
  --seed 20260806 \
  2>&1 | tee "${OUT_DIR}/len_gen_dyck1_${SLURM_JOB_ID}.log"

echo ""
echo "=== Dyck-2: train L=256 → test L=256,512,1024,2048 ==="
python3 scripts/research/ossm_dyck_scaling.py \
  --generalize \
  --train-len 256 \
  --test-lens 256 512 1024 2048 \
  --n-types 2 \
  --epochs 100 \
  --train-size 4096 \
  --test-size 1024 \
  --gpu \
  --seed 20260806 \
  2>&1 | tee "${OUT_DIR}/len_gen_dyck2_${SLURM_JOB_ID}.log"

cp scripts/research/ossm_length_generalization_results.json \
   "${OUT_DIR}/len_gen_results_${SLURM_JOB_ID}.json" 2>/dev/null || true

echo ""
echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "DONE"
