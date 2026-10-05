#!/usr/bin/env bash
# gpu_test_runner.sh — run epistemic GEMM profile on remote GPU node
#
# Usage:
#   GPU_HOST=gpu-appliance-l4  bash scripts/gpu/gpu_test_runner.sh
#   GPU_HOST=r740              bash scripts/gpu/gpu_test_runner.sh
#   GPU_HOST=5860              bash scripts/gpu/gpu_test_runner.sh
#
# Outputs:
#   [souc-gpu] epistemic_gemm NxNxN  ... X.X GFLOPS  bound=...  eff=...%  dbuf=yes
#
# Environment overrides:
#   GPU_HOST        — logical name: gpu-appliance-l4, r740, or 5860  (required)
#   GPU_HOST_IP     — override resolved IP (default: auto per GPU_HOST)
#   GPU_USER        — SSH username           (default: demetrios)
#   GPU_KEY         — SSH private key path   (default: ~/.ssh/id_ed25519)
#   GPU_PASS        — keyboard-interactive password for 2FA nodes (optional)
#   REMOTE_DIR      — path on GPU host       (default: ~/work/sounio)
#   REMOTE_CARGO_BIN— prepend remote cargo bin dir (default: ~/.cargo/bin)
#   LOCAL_DIR       — local repo root        (default: auto-detected)
#   GEMM_M/N/K      — matrix dimensions      (default: 4096)
#   SM_MAJOR        — PTX SM version major   (default: auto from nvidia-smi)
#   SKIP_BUILD      — set to 1 to skip cargo build (reuse previous binary)
#   CONTRACT_PATH   — independence benchmark contract path
#   GPU_PROFILE_TIMEOUT_SECS — timeout for cargo-test profile stage (default: 900)
#   GPU_RUN_TIMEOUT_SECS     — timeout for souc run stages (default: 300)
#   GPU_DIRECT_DISPATCH_FALLBACK — run direct dispatch on profile failure/timeout (default: 1)
#   GPU_DIRECT_ITERS         — iterations for direct dispatch fallback (default: 8)
#   GPU_DIRECT_PTX_FILE      — optional PTX file for direct dispatch fallback

set -euo pipefail

GPU_HOST="${GPU_HOST:?set GPU_HOST to gpu-appliance-l4, r740, or 5860}"
REMOTE_DIR="${REMOTE_DIR:-~/work/sounio}"
LOCAL_DIR="${LOCAL_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || echo "$HOME/work/sounio")}"
GEMM_M="${GEMM_M:-4096}"
GEMM_N="${GEMM_N:-4096}"
GEMM_K="${GEMM_K:-4096}"
SKIP_BUILD="${SKIP_BUILD:-0}"
GPU_PASS="${GPU_PASS:-}"   # keyboard-interactive password (2FA); set if needed
CONTRACT_PATH="${CONTRACT_PATH:-benchmarks/independence/contract.v1.json}"
REMOTE_CARGO_BIN="${REMOTE_CARGO_BIN:-\$HOME/.cargo/bin}"
GPU_PROFILE_TIMEOUT_SECS="${GPU_PROFILE_TIMEOUT_SECS:-900}"
GPU_RUN_TIMEOUT_SECS="${GPU_RUN_TIMEOUT_SECS:-300}"
GPU_DIRECT_DISPATCH_FALLBACK="${GPU_DIRECT_DISPATCH_FALLBACK:-1}"
GPU_DIRECT_ITERS="${GPU_DIRECT_ITERS:-8}"
GPU_DIRECT_PTX_FILE="${GPU_DIRECT_PTX_FILE:-}"

# Resolve GPU_HOST to IP for direct 100G path
case "$GPU_HOST" in
  gpu-appliance-l4) GPU_HOST_IP="${GPU_HOST_IP:-10.100.100.215}" ;;
  r740)             GPU_HOST_IP="${GPU_HOST_IP:-r740}" ;;
  5860)             GPU_HOST_IP="${GPU_HOST_IP:-5860}" ;;
  *)                GPU_HOST_IP="${GPU_HOST_IP:-$GPU_HOST}" ;;
esac
GPU_USER="${GPU_USER:-demetrios}"
GPU_KEY="${GPU_KEY:-$HOME/.ssh/id_ed25519}"

# SSH helper: uses sshpass if GPU_PASS is set, otherwise plain ssh
SSH_CMD="ssh -i $GPU_KEY -o StrictHostKeyChecking=no -o PreferredAuthentications=publickey,keyboard-interactive"
RSYNC_SSH="$SSH_CMD"
if [ -n "$GPU_PASS" ]; then
  export SSHPASS="$GPU_PASS"
  SSH_CMD="sshpass -e ssh -i $GPU_KEY -o StrictHostKeyChecking=no -o PreferredAuthentications=publickey,keyboard-interactive"
  RSYNC_SSH="sshpass -e ssh -i $GPU_KEY -o StrictHostKeyChecking=no -o PreferredAuthentications=publickey,keyboard-interactive"
fi

echo "======================================================================="
echo "  Sounio GPU Test Runner"
echo "  Host  : $GPU_HOST"
echo "  Local : $LOCAL_DIR"
echo "  Remote: $REMOTE_DIR"
echo "  GEMM  : ${GEMM_M}×${GEMM_N}×${GEMM_K}"
echo "======================================================================="

if [ -f "$LOCAL_DIR/$CONTRACT_PATH" ]; then
  echo "[contract] validating $CONTRACT_PATH ..."
  python3 - "$LOCAL_DIR/$CONTRACT_PATH" <<'PY'
import json
import pathlib
import sys

obj = json.loads(pathlib.Path(sys.argv[1]).read_text())
schema = obj.get("schema")
if schema not in {"sounio.independence.contract.v1", "sounio.independence.contract.v2"}:
    raise SystemExit(f"invalid contract schema: {obj.get('schema')}")
acc = obj.get("epistemic_power_accumulator_bound")
if not isinstance(acc, dict):
    raise SystemExit("missing epistemic_power_accumulator_bound")
if int(acc.get("delta_32_max_exclusive", 0)) != 96:
    raise SystemExit(
        "invalid delta_32_max_exclusive; expected 96 "
        f"(got {acc.get('delta_32_max_exclusive')!r})"
    )
formula = str(acc.get("delta_32_formula", "")).replace(" ", "")
if formula != "4*(F%8)+2*(P%16)+(Q%32)":
    raise SystemExit(f"invalid delta_32_formula: {formula!r}")
print(
    f"[contract] ok schema={schema} "
    f"threshold={obj['performance_gate']['threshold']} "
    f"delta_32_max_exclusive={acc['delta_32_max_exclusive']}"
)
PY
fi

# ── 1. Sync repo (exclude build artefacts) ──────────────────────────────────
echo ""
echo "[1] Syncing repo to ${GPU_USER}@${GPU_HOST_IP}:$REMOTE_DIR ..."
rsync -az --delete \
  -e "$RSYNC_SSH" \
  --exclude '/target/' \
  --exclude '/.git/' \
  --exclude '/artifacts/' \
  --exclude '/.continue/' \
  "${LOCAL_DIR}/" "${GPU_USER}@${GPU_HOST_IP}:${REMOTE_DIR}/"
echo "    sync done."

# ── 2. Remote build + profile ────────────────────────────────────────────────
echo ""
echo "[2] Running on ${GPU_USER}@${GPU_HOST_IP} ..."
$SSH_CMD "${GPU_USER}@${GPU_HOST_IP}" bash -s <<REMOTE_SCRIPT
set -euo pipefail

if [ -d "${REMOTE_CARGO_BIN}" ]; then
  export PATH="${REMOTE_CARGO_BIN}:\$PATH"
fi

run_with_timeout() {
  local timeout_secs="\$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "\${timeout_secs}s" "\$@"
  else
    "\$@"
  fi
}

echo ""
echo "=== Rust toolchain ==="
echo "cargo: \$(command -v cargo || echo 'not found')"
echo "rustc: \$(command -v rustc || echo 'not found')"
cargo --version || true
rustc --version || true

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
profile_log="\$(mktemp /tmp/sounio_gpu_profile.XXXXXX.log)"
direct_log="\$(mktemp /tmp/sounio_gpu_direct_dispatch.XXXXXX.log)"
cleanup_logs() {
  rm -f "\$profile_log" "\$direct_log"
}
trap cleanup_logs EXIT

profile_ok=0
set +e
run_with_timeout "${GPU_PROFILE_TIMEOUT_SECS}" env \
  SOUNIO_GPU_GEMM_M=${GEMM_M} \
  SOUNIO_GPU_GEMM_N=${GEMM_N} \
  SOUNIO_GPU_GEMM_K=${GEMM_K} \
  cargo test --release --features gpu -p souc \
    --lib "epistemic_gemm" -- --nocapture >"\$profile_log" 2>&1
profile_rc=\$?
set -e

grep -E "\[souc-gpu\]|GFLOPS|test result|FAILED" "\$profile_log" || true

if [ "\$profile_rc" -eq 0 ]; then
  profile_ok=1
  echo "GPU_PROFILE_STATUS=pass mode=cargo_test timeout_s=${GPU_PROFILE_TIMEOUT_SECS}"
else
  if [ "\$profile_rc" -eq 124 ] || [ "\$profile_rc" -eq 137 ]; then
    echo "warning: epistemic_gemm profile timed out after ${GPU_PROFILE_TIMEOUT_SECS}s"
  else
    echo "warning: epistemic_gemm profile exited rc=\$profile_rc"
  fi
fi

if [ "\$profile_ok" -ne 1 ] && [ "${GPU_DIRECT_DISPATCH_FALLBACK}" = "1" ]; then
  echo ""
  echo "=== Direct CUDA dispatch fallback (${GEMM_M}×${GEMM_N}×${GEMM_K}) ==="
  set +e
  if [ -n "${GPU_DIRECT_PTX_FILE}" ] && [ -f "${GPU_DIRECT_PTX_FILE}" ]; then
    run_with_timeout "${GPU_PROFILE_TIMEOUT_SECS}" env \
      GEMM_M=${GEMM_M} \
      GEMM_N=${GEMM_N} \
      GEMM_K=${GEMM_K} \
      GEMM_ITERS=${GPU_DIRECT_ITERS} \
      GEMM_PTX_FILE="${GPU_DIRECT_PTX_FILE}" \
      python3 scripts/gpu/cuda_gemm_dispatch.py >"\$direct_log" 2>&1
  else
    run_with_timeout "${GPU_PROFILE_TIMEOUT_SECS}" env \
      GEMM_M=${GEMM_M} \
      GEMM_N=${GEMM_N} \
      GEMM_K=${GEMM_K} \
      GEMM_ITERS=${GPU_DIRECT_ITERS} \
      python3 scripts/gpu/cuda_gemm_dispatch.py >"\$direct_log" 2>&1
  fi
  direct_rc=\$?
  set -e
  cat "\$direct_log"

  if [ "\$direct_rc" -eq 0 ]; then
    profile_ok=1
    echo "GPU_PROFILE_STATUS=pass mode=direct_dispatch timeout_s=${GPU_PROFILE_TIMEOUT_SECS}"
  else
    echo "warning: direct dispatch fallback failed rc=\$direct_rc"
  fi
fi

if [ "\$profile_ok" -ne 1 ]; then
  echo "GPU_PROFILE_STATUS=fail"
fi

echo ""
echo "=== Epistemic GEMM benchmark via souc run ==="
run_with_timeout "${GPU_RUN_TIMEOUT_SECS}" ./target/release/souc run benchmarks/cl44_vs_octonion.sio \
  2>/dev/null | grep -E "Octonion|GFLOPS|BENCHMARK" || true

echo ""
echo "=== fMRI Equivariant demo (CPU validation) ==="
run_with_timeout "${GPU_RUN_TIMEOUT_SECS}" ./target/release/souc run experiments/02_fmri_equivariant/fmri_equivariant.sio \
  2>/dev/null | grep -E "PASS|FAIL|Experiment 02 complete|r=" || true

REMOTE_SCRIPT

# ── cuBLAS baseline (optional) ────────────────────────────────────────────
BASELINE="${BASELINE:-0}"
if [ "$BASELINE" = "1" ]; then
  echo ""
  echo "=== cuBLAS SGEMM baseline ==="
  $SSH_CMD "${GPU_USER}@${GPU_HOST_IP}" <<BASELINE_SCRIPT
export PATH="${REMOTE_CARGO_BIN}:\$PATH"
cd ${REMOTE_DIR}
GEMM_DIMS="${GEMM_M}" GEMM_ITERS="${GPU_DIRECT_ITERS}" \
  python3 scripts/gpu/cuda_cublas_baseline.py 2>&1
BASELINE_SCRIPT
  echo "GPU_BASELINE_STATUS=done"
fi

echo ""
echo "======================================================================="
echo "  Done. Check [souc-gpu] lines above for GFLOPS measurement."
echo "  Target: >= 500 GFLOPS sustained on L4/A5000/4000Ada"
if [ "$BASELINE" = "1" ]; then
  echo "  cuBLAS baseline also ran — check [souc-gpu] cublas_sgemm lines."
fi
echo "======================================================================="
