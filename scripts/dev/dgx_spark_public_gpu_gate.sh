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
DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER="${DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER:-0}"
DGX_SPARK_JSON="${DGX_SPARK_JSON:-$ROOT_DIR/artifacts/gpu/dgx_spark_public_gpu_gate.v1.json}"
DGX_SPARK_PACKAGE_ONLY="${DGX_SPARK_PACKAGE_ONLY:-0}"
DGX_SPARK_PACKAGE_DIR="${DGX_SPARK_PACKAGE_DIR:-$ROOT_DIR/artifacts/gpu/dgx_spark_public_gpu_package}"
DGX_SPARK_PUBLIC_KERNELS="${DGX_SPARK_PUBLIC_KERNELS:-1}"

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
  local gpu_knowledge_runtime_output="${8:-}"
  local gpu_knowledge_marker_status="${9:-disabled}"
  local package_dir="${10:-}"
  local package_manifest="${11:-}"
  python3 - "$LOCAL_REPORT" "$status" "$reason" "$DGX_SPARK_TARGET" "$DGX_SPARK_ARCH" \
    "$remote_hostname" "$remote_uname" "$ptxas_version" "$nvcc_version" "$runtime_output" \
    "$DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER" "$gpu_knowledge_runtime_output" "$gpu_knowledge_marker_status" \
    "$DGX_SPARK_PACKAGE_ONLY" "$package_dir" "$DGX_SPARK_PUBLIC_KERNELS" "$package_manifest" <<'PY'
import json
import pathlib
import sys

out, status, reason, target, arch, hostname, uname, ptxas_version, nvcc_version, runtime_output, marker_enabled, marker_runtime_output, marker_status, package_only, package_dir, public_kernels, package_manifest = sys.argv[1:]
local_sources = []
if public_kernels == "1":
    local_sources.extend([
        "tests/run-pass/gpu_vec_add_e2e.sio",
        "tests/run-pass/gpu_launch_vec_slices.sio",
    ])
if marker_enabled == "1":
    local_sources.extend([
        "scripts/dev/gpu_knowledge_vec4_ptxas_probe.sh",
        "artifacts/gpu/knowledge_vecmat_evidence_audit/ptxas_probe/gpu_knowledge_vec4_aggregate_marker.ptx",
        "artifacts/gpu/knowledge_vecmat_evidence_audit/ptxas_probe/gpu_knowledge_vec4_aggregate_marker_cuda_runner.cu",
    ])
boundaries = [
    "local_workspace_is_compiler_authority",
    "does_not_store_or_manage_ssh_passwords",
    "does_not_claim_general_gpu_backend_correctness",
]
if package_only == "1":
    boundaries.extend([
        "package_only_no_remote_ssh",
        "package_only_does_not_claim_dgx_toolchain_or_runtime",
    ])
else:
    boundaries.append("dgx_spark_is_cuda_toolchain_and_runtime_authority")
if public_kernels == "1":
    boundaries.append("validates_public_gpu_f64_ptx_for_selected_kernels")
if marker_enabled == "1":
    boundaries.extend([
        "gpu_knowledge_vec4_marker_is_opt_in",
        "gpu_knowledge_vec4_marker_runtime_claim_requires_runtime_output",
        "does_not_claim_automatic_backend_pack_unpack",
        "does_not_claim_imported_runtime_fixture",
    ])
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
        "sources": local_sources,
    },
    "runtime_output": runtime_output,
    "package_only": package_only == "1",
    "package_dir": package_dir,
    "package_manifest": package_manifest,
    "public_kernels_enabled": public_kernels == "1",
    "gpu_knowledge_vec4_marker": {
        "enabled": marker_enabled == "1",
        "status": marker_status,
        "kernel": "gpu_knowledge_vec4_aggregate_marker",
        "copyback_offsets_bytes": [0, 32, 64, 96],
        "expected_value_lanes": [1.0, 2.0, 3.0, 4.0],
        "runtime_output": marker_runtime_output,
    },
    "boundaries": boundaries,
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

write_package_manifest() {
  local package_dir="$1"
  local manifest="$package_dir/gpu_knowledge_vec4_package_manifest.v1.json"
  python3 - "$ROOT_DIR" "$package_dir" "$manifest" "$DGX_SPARK_ARCH" <<'PY'
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

root = pathlib.Path(sys.argv[1])
package_dir = pathlib.Path(sys.argv[2])
manifest = pathlib.Path(sys.argv[3])
arch = sys.argv[4]

def rel(path):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)

def entry(name):
    path = package_dir / name
    data = path.read_bytes()
    return {
        "path": rel(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }

payload = {
    "schema": "sounio.gpu-knowledge-vec4-dgx-package.v1",
    "status": "pass",
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "arch": arch,
    "package_dir": rel(package_dir),
    "files": {
        "ptx": entry("gpu_knowledge_vec4_aggregate_marker.ptx"),
        "runtime_harness": entry("gpu_knowledge_vec4_aggregate_marker_cuda_runner.cu"),
        "local_ptxas_cubin": entry("gpu_knowledge_vec4_aggregate_marker.local-ptxas.cubin"),
    },
    "runtime_launch_contract": {
        "kernel": "gpu_knowledge_vec4_aggregate_marker",
        "params": ["out_ptr"],
        "copyback_offsets_bytes": [0, 32, 64, 96],
        "expected_value_lanes": [1.0, 2.0, 3.0, 4.0],
        "status": "local_package_only_not_remote_not_launched",
    },
    "boundaries": [
        "local_ptxas_package_proof",
        "package_only_no_remote_ssh",
        "does_not_claim_dgx_toolchain_or_runtime",
        "does_not_claim_cuda_device_runtime_execution",
    ],
}
manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

echo "=== DGX Spark Public GPU Gate ==="
echo "SOUC:               $SOUC"
echo "SOUNIO_STDLIB_PATH: $SOUNIO_STDLIB_PATH"
echo "target:             $DGX_SPARK_TARGET"
echo "arch:               $DGX_SPARK_ARCH"
echo "remote dir:         $DGX_SPARK_REMOTE_DIR"
echo "knowledge marker:   $DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER"
echo "package only:       $DGX_SPARK_PACKAGE_ONLY"
echo "public kernels:     $DGX_SPARK_PUBLIC_KERNELS"
echo ""

VEC_PTX="$TMP_DIR/gpu_vec_add_e2e.ptx"
SLICES_PTX="$TMP_DIR/gpu_launch_vec_slices.ptx"
GK_PROBE_DIR="$TMP_DIR/gpu_knowledge_vec4_probe"
GK_PTX="$GK_PROBE_DIR/gpu_knowledge_vec4_aggregate_marker.ptx"
GK_RUNNER="$GK_PROBE_DIR/gpu_knowledge_vec4_aggregate_marker_cuda_runner.cu"

if [ "$DGX_SPARK_PUBLIC_KERNELS" = "1" ]; then
  echo "=== Local public GPU structural gate ==="
  bash tests/gpu/gate_public_gpu_cfg_build.sh

  timeout 30 "$SOUC" build tests/run-pass/gpu_vec_add_e2e.sio --backend gpu -o "$VEC_PTX"
  timeout 30 "$SOUC" build tests/run-pass/gpu_launch_vec_slices.sio --backend gpu -o "$SLICES_PTX"
fi

if [ "$DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER" = "1" ]; then
  SOUNIO_GPU_KNOWLEDGE_PTXAS_PROBE_DIR="$GK_PROBE_DIR" \
    scripts/dev/gpu_knowledge_vec4_ptxas_probe.sh >/dev/null
  if [ ! -s "$GK_PTX" ] || [ ! -s "$GK_RUNNER" ]; then
    write_json "fail" "gpu_knowledge_vec4_marker_missing"
    echo "dgx_spark_public_gpu_gate: FAIL GPU Knowledge Vec4 marker artifact missing" >&2
    exit 1
  fi
fi

if [ "$DGX_SPARK_PUBLIC_KERNELS" = "1" ] && { [ ! -s "$VEC_PTX" ] || [ ! -s "$SLICES_PTX" ]; }; then
  write_json "fail" "local_ptx_missing"
  echo "dgx_spark_public_gpu_gate: FAIL local PTX output missing" >&2
  exit 1
fi

if [ "$DGX_SPARK_PACKAGE_ONLY" = "1" ]; then
  rm -rf "$DGX_SPARK_PACKAGE_DIR"
  mkdir -p "$DGX_SPARK_PACKAGE_DIR"
  if [ "$DGX_SPARK_PUBLIC_KERNELS" = "1" ]; then
    cp "$VEC_PTX" "$DGX_SPARK_PACKAGE_DIR/gpu_vec_add_e2e.ptx"
    cp "$SLICES_PTX" "$DGX_SPARK_PACKAGE_DIR/gpu_launch_vec_slices.ptx"
  fi
  GPU_KNOWLEDGE_MARKER_STATUS="disabled"
  if [ "$DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER" = "1" ]; then
    cp "$GK_PTX" "$DGX_SPARK_PACKAGE_DIR/gpu_knowledge_vec4_aggregate_marker.ptx"
    cp "$GK_RUNNER" "$DGX_SPARK_PACKAGE_DIR/gpu_knowledge_vec4_aggregate_marker_cuda_runner.cu"
    if [ -s "$GK_PROBE_DIR/gpu_knowledge_vec4_aggregate_marker.cubin" ]; then
      cp "$GK_PROBE_DIR/gpu_knowledge_vec4_aggregate_marker.cubin" "$DGX_SPARK_PACKAGE_DIR/gpu_knowledge_vec4_aggregate_marker.local-ptxas.cubin"
    fi
    write_package_manifest "$DGX_SPARK_PACKAGE_DIR"
    GPU_KNOWLEDGE_MARKER_STATUS="local_ptxas_only_not_remote_not_launched"
  fi
  if [ "$DGX_SPARK_PUBLIC_KERNELS" != "1" ] && [ "$DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER" != "1" ]; then
    write_json "fail" "package_only_no_artifacts_requested" "" "" "" "" "" "" "$GPU_KNOWLEDGE_MARKER_STATUS" "$DGX_SPARK_PACKAGE_DIR"
    echo "dgx_spark_public_gpu_gate: FAIL package-only requested with no public kernels and no marker" >&2
    exit 1
  fi
  PACKAGE_MANIFEST=""
  if [ "$DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER" = "1" ]; then
    PACKAGE_MANIFEST="$DGX_SPARK_PACKAGE_DIR/gpu_knowledge_vec4_package_manifest.v1.json"
  fi
  write_json "pass" "dgx_spark_package_only_prepared" "" "" "" "" "" "" "$GPU_KNOWLEDGE_MARKER_STATUS" "$DGX_SPARK_PACKAGE_DIR" "$PACKAGE_MANIFEST"
  echo "dgx_spark_public_gpu_gate: PASS package_only dir=${DGX_SPARK_PACKAGE_DIR#$ROOT_DIR/} report=${DGX_SPARK_JSON#$ROOT_DIR/}"
  exit 0
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
if [ "$DGX_SPARK_PUBLIC_KERNELS" = "1" ]; then
  run_scp_to_remote "$VEC_PTX" "$DGX_SPARK_REMOTE_DIR/gpu_vec_add_e2e.ptx"
  run_scp_to_remote "$SLICES_PTX" "$DGX_SPARK_REMOTE_DIR/gpu_launch_vec_slices.ptx"
fi
if [ "$DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER" = "1" ]; then
  run_scp_to_remote "$GK_PTX" "$DGX_SPARK_REMOTE_DIR/gpu_knowledge_vec4_aggregate_marker.ptx"
  run_scp_to_remote "$GK_RUNNER" "$DGX_SPARK_REMOTE_DIR/gpu_knowledge_vec4_aggregate_marker_cuda_runner.cu"
fi

echo ""
echo "=== Remote ptxas validation ==="
run_ssh "set -euo pipefail
cd '$DGX_SPARK_REMOTE_DIR'
if [ '$DGX_SPARK_PUBLIC_KERNELS' = '1' ]; then
  '$DGX_SPARK_PTXAS' -arch='$DGX_SPARK_ARCH' gpu_vec_add_e2e.ptx -o gpu_vec_add_e2e.cubin
  '$DGX_SPARK_PTXAS' -arch='$DGX_SPARK_ARCH' gpu_launch_vec_slices.ptx -o gpu_launch_vec_slices.cubin
  test -s gpu_vec_add_e2e.cubin
  test -s gpu_launch_vec_slices.cubin
  wc -c gpu_vec_add_e2e.cubin gpu_launch_vec_slices.cubin
fi
if [ '$DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER' = '1' ]; then
  '$DGX_SPARK_PTXAS' -arch='$DGX_SPARK_ARCH' gpu_knowledge_vec4_aggregate_marker.ptx -o gpu_knowledge_vec4_aggregate_marker.cubin
  test -s gpu_knowledge_vec4_aggregate_marker.cubin
fi
if [ '$DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER' = '1' ]; then
  wc -c gpu_knowledge_vec4_aggregate_marker.cubin
fi"

RUNTIME_OUTPUT=""
GPU_KNOWLEDGE_RUNTIME_OUTPUT=""
GPU_KNOWLEDGE_MARKER_STATUS="disabled"
if [ "$DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER" = "1" ]; then
  GPU_KNOWLEDGE_MARKER_STATUS="ptxas_only_not_launched"
fi
if [ "$DGX_SPARK_RUNTIME" = "1" ] && [ "$DGX_SPARK_PUBLIC_KERNELS" = "1" ]; then
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
  if [ "$DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER" = "1" ]; then
    echo ""
    echo "=== Remote GPU Knowledge Vec4 marker runtime ==="
    GPU_KNOWLEDGE_RUNTIME_OUTPUT="$(
      run_ssh "set -euo pipefail
cd '$DGX_SPARK_REMOTE_DIR'
'$DGX_SPARK_NVCC' -std=c++17 gpu_knowledge_vec4_aggregate_marker_cuda_runner.cu -lcuda -o run_gpu_knowledge_vec4_marker
./run_gpu_knowledge_vec4_marker gpu_knowledge_vec4_aggregate_marker.cubin"
    )"
    printf '%s\n' "$GPU_KNOWLEDGE_RUNTIME_OUTPUT"
    if printf '%s\n' "$GPU_KNOWLEDGE_RUNTIME_OUTPUT" | grep -q 'PASS gpu_knowledge_vec4_aggregate_marker'; then
      GPU_KNOWLEDGE_MARKER_STATUS="runtime_pass"
    else
      GPU_KNOWLEDGE_MARKER_STATUS="runtime_output_missing_pass_marker"
      write_json "fail" "gpu_knowledge_vec4_marker_runtime_output_missing_pass" "$REMOTE_HOSTNAME" "$REMOTE_UNAME" "$PTXAS_VERSION" "$NVCC_VERSION" "$RUNTIME_OUTPUT" "$GPU_KNOWLEDGE_RUNTIME_OUTPUT" "$GPU_KNOWLEDGE_MARKER_STATUS"
      echo "dgx_spark_public_gpu_gate: FAIL GPU Knowledge Vec4 marker runtime output missing PASS marker" >&2
      exit 1
    fi
  fi
elif [ "$DGX_SPARK_RUNTIME" = "1" ] && [ "$DGX_SPARK_GPU_KNOWLEDGE_VEC4_MARKER" = "1" ]; then
  echo ""
  echo "=== Remote GPU Knowledge Vec4 marker runtime ==="
  GPU_KNOWLEDGE_RUNTIME_OUTPUT="$(
    run_ssh "set -euo pipefail
cd '$DGX_SPARK_REMOTE_DIR'
'$DGX_SPARK_NVCC' -std=c++17 gpu_knowledge_vec4_aggregate_marker_cuda_runner.cu -lcuda -o run_gpu_knowledge_vec4_marker
./run_gpu_knowledge_vec4_marker gpu_knowledge_vec4_aggregate_marker.cubin"
  )"
  printf '%s\n' "$GPU_KNOWLEDGE_RUNTIME_OUTPUT"
  if printf '%s\n' "$GPU_KNOWLEDGE_RUNTIME_OUTPUT" | grep -q 'PASS gpu_knowledge_vec4_aggregate_marker'; then
    GPU_KNOWLEDGE_MARKER_STATUS="runtime_pass"
  else
    GPU_KNOWLEDGE_MARKER_STATUS="runtime_output_missing_pass_marker"
    write_json "fail" "gpu_knowledge_vec4_marker_runtime_output_missing_pass" "$REMOTE_HOSTNAME" "$REMOTE_UNAME" "$PTXAS_VERSION" "$NVCC_VERSION" "$RUNTIME_OUTPUT" "$GPU_KNOWLEDGE_RUNTIME_OUTPUT" "$GPU_KNOWLEDGE_MARKER_STATUS"
    echo "dgx_spark_public_gpu_gate: FAIL GPU Knowledge Vec4 marker runtime output missing PASS marker" >&2
    exit 1
  fi
fi

write_json "pass" "dgx_spark_public_gpu_validated" "$REMOTE_HOSTNAME" "$REMOTE_UNAME" "$PTXAS_VERSION" "$NVCC_VERSION" "$RUNTIME_OUTPUT" "$GPU_KNOWLEDGE_RUNTIME_OUTPUT" "$GPU_KNOWLEDGE_MARKER_STATUS"

echo ""
echo "dgx_spark_public_gpu_gate: PASS report=$(python3 - "$ROOT_DIR" "$DGX_SPARK_JSON" <<'PY'
import os
import sys
print(os.path.relpath(sys.argv[2], sys.argv[1]))
PY
)"
