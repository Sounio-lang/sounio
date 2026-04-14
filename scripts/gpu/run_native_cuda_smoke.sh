#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC_BIN="${SOUC_BIN:-$ROOT_DIR/bin/souc}"

if [[ ! -x "$SOUC_BIN" ]]; then
  echo "FAIL: souc binary not found at $SOUC_BIN" >&2
  exit 1
fi

FIXTURE="${1:-$ROOT_DIR/tests/run-pass/gpu_vec_add.sio}"
STRICT="${SOUNIO_CUDA_SMOKE_STRICT:-0}"

find_cuda_driver_lib() {
  local candidate=""
  if [[ -n "${SOUNIO_CUDA_DRIVER_LIB:-}" && -e "${SOUNIO_CUDA_DRIVER_LIB}" ]]; then
    printf '%s\n' "${SOUNIO_CUDA_DRIVER_LIB}"
    return 0
  fi
  if [[ -n "${CUDA_HOME:-}" && -e "${CUDA_HOME}/compat/libcuda.so.1" ]]; then
    printf '%s\n' "${CUDA_HOME}/compat/libcuda.so.1"
    return 0
  fi
  if [[ -n "${CUDA_PATH:-}" && -e "${CUDA_PATH}/compat/libcuda.so.1" ]]; then
    printf '%s\n' "${CUDA_PATH}/compat/libcuda.so.1"
    return 0
  fi
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    local dir
    IFS=: read -r -a _cuda_ld_paths <<< "${LD_LIBRARY_PATH}"
    for dir in "${_cuda_ld_paths[@]}"; do
      if [[ -n "$dir" && -e "$dir/libcuda.so.1" ]]; then
        printf '%s\n' "$dir/libcuda.so.1"
        return 0
      fi
    done
  fi
  for candidate in \
    /lib/x86_64-linux-gnu/libcuda.so.1 \
    /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    /usr/lib64/libcuda.so.1 \
    /usr/lib/wsl/lib/libcuda.so.1 \
    /run/opengl-driver/lib/libcuda.so.1 \
    /usr/local/cuda/compat/libcuda.so.1
  do
    if [[ -e "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  if command -v ldconfig >/dev/null 2>&1; then
    candidate="$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/ {print $NF; exit}')"
    if [[ -n "$candidate" && -e "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi
  return 1
}

CUDA_DRIVER_LIB="$(find_cuda_driver_lib || true)"
if [[ -z "$CUDA_DRIVER_LIB" ]]; then
  if ls /dev/nvidia* >/dev/null 2>&1; then
    echo "INFO: NVIDIA device nodes are present:"
    ls -l /dev/nvidia* 2>/dev/null || true
  fi
  if grep -i nvidia /proc/modules >/dev/null 2>&1; then
    echo "INFO: NVIDIA kernel modules are loaded:"
    grep -i nvidia /proc/modules || true
  fi
  if [[ -d /proc/driver/nvidia ]]; then
    echo "INFO: /proc/driver/nvidia is present"
    ls -l /proc/driver/nvidia 2>/dev/null || true
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "INFO: nvidia-smi is present"
  else
    echo "INFO: nvidia-smi is not present in PATH"
  fi
  if [[ "$STRICT" == "1" ]]; then
    echo "FAIL: libcuda.so.1 not found; strict native CUDA smoke requires visible NVIDIA driver runtime"
    exit 1
  fi
  echo "SKIP: libcuda.so.1 not found; native CUDA smoke requires NVIDIA driver runtime"
  exit 0
fi

export SOUNIO_CUDA_DRIVER_LIB="$CUDA_DRIVER_LIB"
export LD_LIBRARY_PATH="$(dirname "$CUDA_DRIVER_LIB")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
OUT_FILE="$TMP_DIR/native_cuda_smoke.out"

"$SOUC_BIN" run --gpu-runtime "$FIXTURE" >"$OUT_FILE" 2>&1
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
