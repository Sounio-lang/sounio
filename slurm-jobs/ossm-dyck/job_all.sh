#!/usr/bin/env bash
#SBATCH -p gpu-orangefs
#SBATCH -A plruntime
#SBATCH --qos=burst
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -J ossm-dyck
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH -o /workspace/sounio/artifacts/ossm_dyck/%x_%j.out

set -euo pipefail

REPO="/workspace/sounio"
mkdir -p "${REPO}/artifacts/ossm_dyck"

echo "=== OSSM Dyck Scaling ==="
echo "Job: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

# Try to install torch if not present (silent, fast)
python3 -c "import torch" 2>/dev/null || pip3 install torch --break-system-packages --quiet 2>/dev/null || true

python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cd "${REPO}"

echo "=== Dyck-1 (L=32..1024) ==="
python3 scripts/research/ossm_dyck_scaling.py \
  --lengths 32 64 128 256 512 1024 \
  --n-types 1 --epochs 100 --train-size 4096 --test-size 1024 --gpu --seed 20260806

echo "=== Dyck-2 (L=32..512) ==="
python3 scripts/research/ossm_dyck_scaling.py \
  --lengths 32 64 128 256 512 \
  --n-types 2 --epochs 100 --train-size 4096 --test-size 1024 --gpu --seed 20260806

echo "=== Length Generalization ==="
python3 scripts/research/ossm_dyck_scaling.py \
  --generalize --train-len 256 --test-lens 256 512 1024 4096 \
  --n-types 1 --epochs 100 --train-size 4096 --test-size 1024 --gpu --seed 20260806

echo "DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
