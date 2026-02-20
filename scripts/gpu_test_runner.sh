#!/usr/bin/env bash
# gpu_test_runner.sh — run epistemic GEMM profile on remote GPU node
#
# Usage:
#   GPU_HOST=gpu-appliance-l4  bash scripts/gpu_test_runner.sh
#   GPU_HOST=r740              bash scripts/gpu_test_runner.sh
#   GPU_HOST=5860              bash scripts/gpu_test_runner.sh
#
# Outputs:
#   [souc-gpu] epistemic_gemm NxNxN  ... X.X GFLOPS  bound=...  eff=...%  dbuf=yes
#
# Environment overrides:
#   GPU_HOST        — hostname/IP of GPU node  (required)
#   REMOTE_DIR      — path on GPU host         (default: ~/work/sounio)
#   LOCAL_DIR       — local repo root          (default: auto-detected)
#   GEMM_M/N/K      — matrix dimensions        (default: 4096)
#   SM_MAJOR        — PTX SM version major      (default: auto from nvidia-smi)
#   SKIP_BUILD      — set to 1 to skip cargo build (reuse previous binary)

set -euo pipefail

GPU_HOST="${GPU_HOST:?set GPU_HOST to gpu-appliance-l4, r740, or 5860}"
REMOTE_DIR="${REMOTE_DIR:-~/work/sounio}"
LOCAL_DIR="${LOCAL_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || echo "$HOME/work/sounio")}"
GEMM_M="${GEMM_M:-4096}"
GEMM_N="${GEMM_N:-4096}"
GEMM_K="${GEMM_K:-4096}"
SKIP_BUILD="${SKIP_BUILD:-0}"

echo "======================================================================="
echo "  Sounio GPU Test Runner"
echo "  Host  : $GPU_HOST"
echo "  Local : $LOCAL_DIR"
echo "  Remote: $REMOTE_DIR"
echo "  GEMM  : ${GEMM_M}×${GEMM_N}×${GEMM_K}"
echo "======================================================================="

# ── 1. Sync repo (exclude build artefacts) ──────────────────────────────────
echo ""
echo "[1] Syncing repo to $GPU_HOST:$REMOTE_DIR ..."
rsync -az --delete \
  --exclude 'target/' \
  --exclude '.git/' \
  --exclude 'artifacts/' \
  --exclude '.continue/' \
  "${LOCAL_DIR}/" "${GPU_HOST}:${REMOTE_DIR}/"
echo "    sync done."

# ── 2. Remote build + profile ────────────────────────────────────────────────
echo ""
echo "[2] Running on $GPU_HOST ..."
ssh -t "$GPU_HOST" bash -s <<REMOTE_SCRIPT
set -euo pipefail
cd ${REMOTE_DIR}

echo ""
echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || \
  { echo "ERROR: nvidia-smi not found. Is CUDA installed?"; exit 1; }

SM=\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
echo "Compute capability: \$SM (SM_MAJOR=\${SM:0:1})"

if [ "${SKIP_BUILD}" != "1" ]; then
  echo ""
  echo "=== Building with --features gpu ==="
  cargo build --release --features gpu -p souc 2>&1 | tail -5
fi

echo ""
echo "=== Epistemic GEMM profile (${GEMM_M}×${GEMM_N}×${GEMM_K}) ==="

# Run the GPU unit tests — the roofline hook fires on every GEMM launch
# and prints: [souc-gpu] epistemic_gemm M×K×N  ... GFLOPS  bound=...  eff=...%
SOUNIO_SELFHOST_DRIVER_REQUIRE_OUTPUT=0 \
SOUNIO_GPU_GEMM_M=${GEMM_M} \
SOUNIO_GPU_GEMM_N=${GEMM_N} \
SOUNIO_GPU_GEMM_K=${GEMM_K} \
  cargo test --release --features gpu -p souc \
    --lib "epistemic_gemm" -- --nocapture 2>&1 | \
    grep -E "\[souc-gpu\]|GFLOPS|test result|FAILED" || true

echo ""
echo "=== Epistemic GEMM benchmark via souc run ==="
SOUNIO_SELFHOST_DRIVER_REQUIRE_OUTPUT=0 \
  ./target/release/souc run benchmarks/cl44_vs_octonion.sio 2>/dev/null | \
  grep -E "Octonion|GFLOPS|BENCHMARK"

echo ""
echo "=== fMRI Equivariant demo (CPU validation) ==="
SOUNIO_SELFHOST_DRIVER_REQUIRE_OUTPUT=0 \
  ./target/release/souc run experiments/02_fmri_equivariant/fmri_equivariant.sio 2>/dev/null | \
  grep -E "PASS|FAIL|Experiment 02 complete|r="

REMOTE_SCRIPT

echo ""
echo "======================================================================="
echo "  Done. Check [souc-gpu] lines above for GFLOPS measurement."
echo "  Target: >= 500 GFLOPS sustained on L4/A5000/4000Ada"
echo "======================================================================="
