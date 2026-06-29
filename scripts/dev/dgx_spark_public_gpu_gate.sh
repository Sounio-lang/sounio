#!/usr/bin/env bash
# Validate public GPU PTX on the DGX Spark development host.
#
# This gate keeps the local workspace as the compiler/source authority and uses
# the DGX Spark only as the CUDA toolchain/runtime authority. It intentionally
# does not manage passwords. Use an SSH key or an existing ControlMaster socket.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-./bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

DGX_SPARK_HOST="${DGX_SPARK_HOST:-192.168.3.43}"
DGX_SPARK_USER="${DGX_SPARK_USER:-demetrios}"
DGX_SPARK_TARGET="${DGX_SPARK_TARGET:-${DGX_SPARK_USER}@${DGX_SPARK_HOST}}"
DGX_SPARK_REMOTE_DIR="${DGX_SPARK_REMOTE_DIR:-/tmp/sounio-dgx-spark-public-gpu}"
DGX_SPARK_PTXAS="${DGX_SPARK_PTXAS:-/usr/local/cuda-13.0/bin/ptxas}"
DGX_SPARK_NVCC="${DGX_SPARK_NVCC:-/usr/local/cuda-13.0/bin/nvcc}"
DGX_SPARK_ARCH="${DGX_SPARK_ARCH:-sm_121}"
DGX_SPARK_RUNTIME="${DGX_SPARK_RUNTIME:-1}"
DGX_SPARK_JSON="${DGX_SPARK_JSON:-$ROOT_DIR/artifacts/gpu/dgx_spark_public_gpu_gate.v1.json}"

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o StrictHostKeyChecking=accept-new
)

if [ -n "${DGX_SPARK_SSH_CONTROL_PATH:-}" ]; then
  SSH_OPTS+=(-o "ControlPath=${DGX_SPARK_SSH_CONTROL_PATH}")
fi

case "$DGX_SPARK_REMOTE_DIR" in
  *[!a-zA-Z0-9_./-]*|'')
    echo "dgx_spark_public_gpu_gate: FAIL invalid DGX_SPARK_REMOTE_DIR=$DGX_SPARK_REMOTE_DIR" >&2
    exit 2
    ;;
esac

if [ ! -x "$SOUC" ]; then
  echo "dgx_spark_public_gpu_gate: FAIL souc not executable: $SOUC" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

LOCAL_REPORT="$TMP_DIR/report.json"
mkdir -p "$(dirname "$DGX_SPARK_JSON")"

write_json() {
  local status="$1"
  local reason="$2"
  local remote_hostname="${3:-}"
  local remote_uname="${4:-}"
  local ptxas_version="${5:-}"
  local nvcc_version="${6:-}"
  local runtime_output="${7:-}"
  python3 - "$LOCAL_REPORT" "$status" "$reason" "$DGX_SPARK_TARGET" "$DGX_SPARK_ARCH" \
    "$remote_hostname" "$remote_uname" "$ptxas_version" "$nvcc_version" "$runtime_output" <<'PY'
import json
import pathlib
import sys

out, status, reason, target, arch, hostname, uname, ptxas_version, nvcc_version, runtime_output = sys.argv[1:]
payload = {
    "schema": "sounio.dgx-spark-public-gpu-gate.v1",
    "status": status,
    "reason": reason,
    "target": target,
    "arch": arch,
    "remote": {
        "hostname": hostname,
        "uname_m": uname,
        "ptxas_version": ptxas_version,
        "nvcc_version": nvcc_version,
    },
    "local": {
        "souc": "bin/souc",
        "stdlib": "stdlib",
        "sources": [
            "tests/run-pass/gpu_vec_add_e2e.sio",
            "tests/run-pass/gpu_launch_vec_slices.sio",
        ],
    },
    "runtime_output": runtime_output,
    "boundaries": [
        "local_workspace_is_compiler_authority",
        "dgx_spark_is_cuda_toolchain_and_runtime_authority",
        "validates_public_gpu_f64_ptx_for_selected_kernels",
        "does_not_store_or_manage_ssh_passwords",
        "does_not_claim_general_gpu_backend_correctness",
    ],
}
pathlib.Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  cp "$LOCAL_REPORT" "$DGX_SPARK_JSON"
}

run_ssh() {
  ssh "${SSH_OPTS[@]}" "$DGX_SPARK_TARGET" "$@"
}

run_scp_to_remote() {
  scp "${SSH_OPTS[@]}" "$1" "$DGX_SPARK_TARGET:$2"
}

echo "=== DGX Spark Public GPU Gate ==="
echo "SOUC:               $SOUC"
echo "SOUNIO_STDLIB_PATH: $SOUNIO_STDLIB_PATH"
echo "target:             $DGX_SPARK_TARGET"
echo "arch:               $DGX_SPARK_ARCH"
echo "remote dir:         $DGX_SPARK_REMOTE_DIR"
echo ""

echo "=== Local public GPU structural gate ==="
bash tests/gpu/gate_public_gpu_cfg_build.sh

VEC_PTX="$TMP_DIR/gpu_vec_add_e2e.ptx"
SLICES_PTX="$TMP_DIR/gpu_launch_vec_slices.ptx"

timeout 30 "$SOUC" build tests/run-pass/gpu_vec_add_e2e.sio --backend gpu -o "$VEC_PTX"
timeout 30 "$SOUC" build tests/run-pass/gpu_launch_vec_slices.sio --backend gpu -o "$SLICES_PTX"

if [ ! -s "$VEC_PTX" ] || [ ! -s "$SLICES_PTX" ]; then
  write_json "fail" "local_ptx_missing"
  echo "dgx_spark_public_gpu_gate: FAIL local PTX output missing" >&2
  exit 1
fi

echo ""
echo "=== Remote CUDA toolchain probe ==="
REMOTE_PROBE="$(
  run_ssh "set -euo pipefail
hostname
uname -m
test -x '$DGX_SPARK_PTXAS'
'$DGX_SPARK_PTXAS' --version | tr '\n' ' '
echo
if [ '$DGX_SPARK_RUNTIME' = '1' ]; then
  test -x '$DGX_SPARK_NVCC'
  '$DGX_SPARK_NVCC' --version | tail -n 1
fi"
)" || {
  write_json "fail" "ssh_or_cuda_toolchain_unreachable"
  echo "dgx_spark_public_gpu_gate: FAIL cannot reach Spark CUDA toolchain over BatchMode SSH" >&2
  echo "hint: establish an SSH key or ControlMaster and set DGX_SPARK_SSH_CONTROL_PATH" >&2
  exit 1
}

REMOTE_HOSTNAME="$(printf '%s\n' "$REMOTE_PROBE" | sed -n '1p')"
REMOTE_UNAME="$(printf '%s\n' "$REMOTE_PROBE" | sed -n '2p')"
PTXAS_VERSION="$(printf '%s\n' "$REMOTE_PROBE" | sed -n '3p')"
NVCC_VERSION="$(printf '%s\n' "$REMOTE_PROBE" | sed -n '4p')"

printf 'remote hostname: %s\n' "$REMOTE_HOSTNAME"
printf 'remote uname -m: %s\n' "$REMOTE_UNAME"
printf 'ptxas:           %s\n' "$PTXAS_VERSION"
if [ "$DGX_SPARK_RUNTIME" = "1" ]; then
  printf 'nvcc:            %s\n' "$NVCC_VERSION"
fi

run_ssh "rm -rf '$DGX_SPARK_REMOTE_DIR' && mkdir -p '$DGX_SPARK_REMOTE_DIR'"
run_scp_to_remote "$VEC_PTX" "$DGX_SPARK_REMOTE_DIR/gpu_vec_add_e2e.ptx"
run_scp_to_remote "$SLICES_PTX" "$DGX_SPARK_REMOTE_DIR/gpu_launch_vec_slices.ptx"

echo ""
echo "=== Remote ptxas validation ==="
run_ssh "set -euo pipefail
cd '$DGX_SPARK_REMOTE_DIR'
'$DGX_SPARK_PTXAS' -arch='$DGX_SPARK_ARCH' gpu_vec_add_e2e.ptx -o gpu_vec_add_e2e.cubin
'$DGX_SPARK_PTXAS' -arch='$DGX_SPARK_ARCH' gpu_launch_vec_slices.ptx -o gpu_launch_vec_slices.cubin
test -s gpu_vec_add_e2e.cubin
test -s gpu_launch_vec_slices.cubin
wc -c gpu_vec_add_e2e.cubin gpu_launch_vec_slices.cubin"

RUNTIME_OUTPUT=""
if [ "$DGX_SPARK_RUNTIME" = "1" ]; then
  HARNESS="$TMP_DIR/run_vec_add.cu"
  cat > "$HARNESS" <<'EOF'
#include <cuda.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do { \
  CUresult status__ = (call); \
  if (status__ != CUDA_SUCCESS) { \
    const char *name__ = nullptr; \
    cuGetErrorName(status__, &name__); \
    std::fprintf(stderr, "CUDA failure %s at %s:%d\n", name__ ? name__ : "unknown", __FILE__, __LINE__); \
    return 1; \
  } \
} while (0)

int main() {
  CHECK(cuInit(0));
  CUdevice dev;
  CHECK(cuDeviceGet(&dev, 0));
  char name[256] = {0};
  int major = 0;
  int minor = 0;
  CHECK(cuDeviceGetName(name, sizeof(name), dev));
  CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  CUcontext ctx;
  CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CHECK(cuCtxSetCurrent(ctx));

  CUmodule mod;
  CHECK(cuModuleLoad(&mod, "gpu_vec_add_e2e.cubin"));
  CUfunction fn;
  CHECK(cuModuleGetFunction(&fn, mod, "vec_add"));

  const long long n = 4;
  const double a[4] = {1.0, 2.5, -4.0, 9.0};
  const double b[4] = {3.0, 0.5, 7.0, 1.125};
  const double expected[4] = {4.0, 3.0, 3.0, 10.125};
  double c[4] = {0.0, 0.0, 0.0, 0.0};

  CUdeviceptr d_a = 0;
  CUdeviceptr d_b = 0;
  CUdeviceptr d_c = 0;
  CHECK(cuMemAlloc(&d_a, sizeof(a)));
  CHECK(cuMemAlloc(&d_b, sizeof(b)));
  CHECK(cuMemAlloc(&d_c, sizeof(c)));
  CHECK(cuMemcpyHtoD(d_a, a, sizeof(a)));
  CHECK(cuMemcpyHtoD(d_b, b, sizeof(b)));
  CHECK(cuMemsetD8(d_c, 0, sizeof(c)));

  void *args[] = {
    const_cast<long long *>(&n),
    &d_a,
    &d_b,
    &d_c,
  };
  CHECK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr));
  CHECK(cuCtxSynchronize());
  CHECK(cuMemcpyDtoH(c, d_c, sizeof(c)));

  for (int i = 0; i < 4; ++i) {
    if (std::fabs(c[i] - expected[i]) > 1e-9) {
      std::fprintf(stderr, "mismatch[%d]: got %.17g expected %.17g\n", i, c[i], expected[i]);
      return 1;
    }
  }

  CHECK(cuMemFree(d_a));
  CHECK(cuMemFree(d_b));
  CHECK(cuMemFree(d_c));
  CHECK(cuModuleUnload(mod));
  CHECK(cuDevicePrimaryCtxRelease(dev));

  std::printf("PASS vec_add f64 on %s cc %d.%d: %.17g %.17g %.17g %.17g\n",
              name, major, minor, c[0], c[1], c[2], c[3]);
  return 0;
}
EOF
  run_scp_to_remote "$HARNESS" "$DGX_SPARK_REMOTE_DIR/run_vec_add.cu"
  echo ""
  echo "=== Remote CUDA Driver API runtime ==="
  RUNTIME_OUTPUT="$(
    run_ssh "set -euo pipefail
cd '$DGX_SPARK_REMOTE_DIR'
'$DGX_SPARK_NVCC' -std=c++17 run_vec_add.cu -lcuda -o run_vec_add
./run_vec_add"
  )"
  printf '%s\n' "$RUNTIME_OUTPUT"
fi

write_json "pass" "dgx_spark_public_gpu_validated" "$REMOTE_HOSTNAME" "$REMOTE_UNAME" "$PTXAS_VERSION" "$NVCC_VERSION" "$RUNTIME_OUTPUT"

echo ""
echo "dgx_spark_public_gpu_gate: PASS report=$(python3 - "$ROOT_DIR" "$DGX_SPARK_JSON" <<'PY'
import os
import sys
print(os.path.relpath(sys.argv[2], sys.argv[1]))
PY
)"
