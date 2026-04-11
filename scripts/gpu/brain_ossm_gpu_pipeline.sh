#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Brain O-SSM GPU Pipeline — Deploy and run on L4 (10.100.100.215)
# ═══════════════════════════════════════════════════════════════════════════
#
# Prerequisites:
#   - SSH key access to demetrios@10.100.100.215
#   - souc binary at ~/work/sounio/bin/souc on remote
#   - CUDA driver on remote (for GPU kernel dispatch)
#
# Usage:
#   ./scripts/gpu/brain_ossm_gpu_pipeline.sh [--dry-run]
#
# What this does:
#   1. Deploys brain_ossm_gpu.sio + fractal_g2_ossm_v3.sio to remote
#   2. Compiles to native ELF on remote (uses remote souc)
#   3. Runs the full benchmark (10 seeds × O-SSM + H-SSM on probe + brain task)
#   4. Retrieves results to local artifacts/
#
# For ABIDE-scale (future):
#   - Download ABIDE data to remote ~/work/sounio/data/abide/
#   - Run brain_ossm_abide.sio (reads NIfTI, extracts ROI time series)
#   - GPU kernel dispatch for parallel oct_mul across subjects

set -euo pipefail

GPU_HOST="${GPU_HOST:-10.100.100.215}"
GPU_USER="${GPU_USER:-demetrios}"
REMOTE_DIR="${REMOTE_DIR:-~/work/sounio}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no)
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/artifacts/omega/brain_ossm"
DRY_RUN="${1:-}"

mkdir -p "$LOG_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  Brain O-SSM GPU Pipeline"
echo "═══════════════════════════════════════════════════════════════"
echo "  Host: ${GPU_USER}@${GPU_HOST}:${REMOTE_DIR}"
echo "  Log:  ${LOG_DIR}"
echo ""

# ─── Step 1: Connectivity check ───
echo "[1/5] Testing SSH connectivity..."
if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "  [DRY-RUN] Skipping SSH test"
else
    if ! ssh "${SSH_OPTS[@]}" "${GPU_USER}@${GPU_HOST}" "echo ok" > /dev/null 2>&1; then
        echo "  FAIL: Cannot reach ${GPU_HOST}. Check SSH keys and network."
        echo "  To test locally: souc compile examples/fractal_g2_ossm_v3.sio -o /tmp/fg2.elf && /tmp/fg2.elf"
        exit 1
    fi
    echo "  OK: SSH connected"
fi

# ─── Step 2: Deploy files ───
DEPLOY_FILES=(
    "examples/fractal_g2_ossm_v3.sio"
    "examples/brain_ossm_classifier.sio"
    "examples/brain_associator_demo.sio"
    "examples/hssm_native_algebra.sio"
    "examples/associativity_probe_benchmark.sio"
    "examples/triple_product_ossm_benchmark.sio"
    "examples/multihead_unit_oct_benchmark.sio"
    "bin/souc"
)

echo "[2/5] Deploying ${#DEPLOY_FILES[@]} files to remote..."
if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "  [DRY-RUN] Would deploy:"
    for f in "${DEPLOY_FILES[@]}"; do echo "    $f"; done
else
    tar -cf - -C "$ROOT_DIR" "${DEPLOY_FILES[@]}" 2>/dev/null | \
        ssh "${SSH_OPTS[@]}" "${GPU_USER}@${GPU_HOST}" \
            "mkdir -p ${REMOTE_DIR} && tar -xf - -C ${REMOTE_DIR}" 2>/dev/null
    echo "  OK: Files deployed"
fi

# ─── Step 3: Compile on remote ───
echo "[3/5] Compiling benchmarks on remote..."
COMPILE_TARGETS=(
    "fractal_g2_ossm_v3.sio:fg2v3.elf"
    "brain_ossm_classifier.sio:brain.elf"
    "hssm_native_algebra.sio:native_alg.elf"
)

if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "  [DRY-RUN] Would compile ${#COMPILE_TARGETS[@]} targets"
else
    for target in "${COMPILE_TARGETS[@]}"; do
        src="${target%%:*}"
        elf="${target##*:}"
        echo "  Compiling ${src} → ${elf}..."
        ssh "${SSH_OPTS[@]}" "${GPU_USER}@${GPU_HOST}" \
            "cd ${REMOTE_DIR} && ./bin/souc compile examples/${src} -o /tmp/${elf}" 2>/dev/null
    done
    echo "  OK: All targets compiled"
fi

# ─── Step 4: Run benchmarks ───
echo "[4/5] Running benchmarks on remote..."
BENCHMARKS=(
    "/tmp/fg2v3.elf:fractal_g2_v3_results.txt"
    "/tmp/brain.elf:brain_classifier_results.txt"
    "/tmp/native_alg.elf:native_algebra_results.txt"
)

if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "  [DRY-RUN] Would run ${#BENCHMARKS[@]} benchmarks"
else
    for bench in "${BENCHMARKS[@]}"; do
        elf="${bench%%:*}"
        out="${bench##*:}"
        echo "  Running ${elf}..."
        ssh "${SSH_OPTS[@]}" "${GPU_USER}@${GPU_HOST}" \
            "cd ${REMOTE_DIR} && timeout 300 ${elf}" > "${LOG_DIR}/${out}" 2>&1 || true
        echo "  → ${LOG_DIR}/${out}"
    done
    echo "  OK: All benchmarks complete"
fi

# ─── Step 5: Retrieve results ───
echo "[5/5] Results saved to ${LOG_DIR}/"
if [[ "$DRY_RUN" != "--dry-run" ]]; then
    echo ""
    echo "═══ Fractal-G2 v3 Results ═══"
    cat "${LOG_DIR}/fractal_g2_v3_results.txt" 2>/dev/null | grep -E "PROBE|LISTOPS|Gap|Overall|NonAssoc" || true
    echo ""
    echo "═══ Brain Classifier Results ═══"
    cat "${LOG_DIR}/brain_classifier_results.txt" 2>/dev/null | grep -E "O-SSM|H-SSM|Accuracy" || true
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Pipeline complete. Results in ${LOG_DIR}/"
echo ""
echo "  For ABIDE-scale (next step):"
echo "    1. Download ABIDE to remote: ${REMOTE_DIR}/data/abide/"
echo "    2. Write brain_ossm_abide.sio (NIfTI reader + ROI extraction)"
echo "    3. GPU kernel dispatch via examples/gpu.sio pattern"
echo "    4. Rerun this script"
echo "═══════════════════════════════════════════════════════════════"
