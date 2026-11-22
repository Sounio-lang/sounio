#!/usr/bin/env bash
# Build PTX (sm_89) and optional lab host binary for Q5_K MoE F32 mmid.
set -euo pipefail
ROOT="${SOUNIO_ROOT:-/workspace/sounio}"
SRC="$ROOT/stdlib/hetero/kernels/mmid_q5k_dp4a_f32.cu"
OUT_DIR="$ROOT/stdlib/hetero/kernels"
ARCH="${CUDA_ARCH:-sm_89}"
NVCC="${NVCC:-nvcc}"

if [[ ! -f "$SRC" ]]; then
  echo "missing $SRC" >&2
  exit 1
fi

"$NVCC" -O3 -ptx -arch="$ARCH" -o "$OUT_DIR/mmid_q5k_dp4a_f32.ptx" "$SRC"
echo "wrote $OUT_DIR/mmid_q5k_dp4a_f32.ptx"

if [[ "${BUILD_LAB:-1}" == "1" ]]; then
  "$NVCC" -O3 -arch="$ARCH" -DMMID_LAB_HOST -o "$OUT_DIR/mmid_q5k_dp4a_f32_lab" "$SRC"
  echo "wrote $OUT_DIR/mmid_q5k_dp4a_f32_lab"
fi
