#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

FIXTURE="${1:-$ROOT_DIR/tests/run-pass/gpu_vec_add.sio}"

if ! ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1'; then
  echo "SKIP: libcuda.so.1 not found; native CUDA smoke requires NVIDIA driver runtime"
  exit 0
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
OUT_FILE="$TMP_DIR/native_cuda_smoke.out"

"$SOUC_BIN" run "$FIXTURE" >"$OUT_FILE" 2>&1
cat "$OUT_FILE"

if grep -q 'GPU unavailable:' "$OUT_FILE"; then
  echo "FAIL: fallback path was used during native CUDA smoke"
  exit 1
fi

if ! grep -q 'PASS: GPU vec_add' "$OUT_FILE"; then
  echo "FAIL: expected PASS: GPU vec_add"
  exit 1
fi

echo "CUDA_SMOKE_OK"
