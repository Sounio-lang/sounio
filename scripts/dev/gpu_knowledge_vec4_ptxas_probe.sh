#!/usr/bin/env bash
# Emit and assemble a launchable GPU Knowledge Vec4 marker artifact.
#
# This is a toolchain/contract probe only: it validates that ptxas accepts the
# PTX and emits a CUDA Driver API harness that can launch/copy back the marker
# lanes later. It does not claim CUDA device runtime execution.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_GPU_KNOWLEDGE_PTXAS_PROBE_DIR:-$ROOT_DIR/artifacts/gpu/knowledge_vecmat_evidence_audit/ptxas_probe}"
OUT_JSON="$OUT_DIR/gpu_knowledge_vec4_ptxas_probe.v1.json"
OUT_PTX="$OUT_DIR/gpu_knowledge_vec4_aggregate_marker.ptx"
OUT_CUBIN="$OUT_DIR/gpu_knowledge_vec4_aggregate_marker.cubin"
OUT_RUNNER="$OUT_DIR/gpu_knowledge_vec4_aggregate_marker_cuda_runner.cu"
ARCH="${SOUNIO_GPU_KNOWLEDGE_PTXAS_ARCH:-sm_80}"

mkdir -p "$OUT_DIR"

find_ptxas() {
  if [ -n "${SOUNIO_GPU_KNOWLEDGE_PTXAS_BIN:-}" ] && [ -x "$SOUNIO_GPU_KNOWLEDGE_PTXAS_BIN" ]; then
    printf '%s\n' "$SOUNIO_GPU_KNOWLEDGE_PTXAS_BIN"
    return 0
  fi
  if command -v ptxas >/dev/null 2>&1; then
    command -v ptxas
    return 0
  fi
  local candidate
  for candidate in \
    /usr/local/cuda/bin/ptxas \
    /usr/local/cuda-13.0/bin/ptxas \
    /workspace/.home/openvscode-server/.agents/claude-2/.local/lib/python3.12/site-packages/torch/bin/ptxas \
    /workspace/.home/openvscode-server/.agents/codex-1/.cache/uv/archive-v0/F6BRiDncsYqX5vDBPCSlZ/torch/bin/ptxas
  do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

rel_path() {
  local path="$1"
  if [[ "$path" == "$ROOT_DIR/"* ]]; then
    printf '%s' "${path#$ROOT_DIR/}"
  else
    printf '%s' "$path"
  fi
}

sha_file() {
  sha256sum "$1" | awk '{print $1}'
}

size_file() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

emit_ptx() {
  cat > "$OUT_PTX" <<'PTX'
.version 7.0
.target sm_70
.address_size 64

// SOUNIO_AGG_RUNTIME_CONTRACT kernel=gpu_knowledge_vec4_aggregate_marker param=out_ptr copyback_offsets=0,32,64,96 expected_values=1,2,3,4
.visible .entry gpu_knowledge_vec4_aggregate_marker(
    .param .u64 out_ptr
)
{
    .reg .b64 %rd<2>;
    .reg .f64 %fd<5>;

    ld.param.u64 %rd1, [out_ptr];
    mov.f64 %fd1, 0d3ff0000000000000;
    mov.f64 %fd2, 0d4000000000000000;
    mov.f64 %fd3, 0d4008000000000000;
    mov.f64 %fd4, 0d4010000000000000;
    st.global.f64 [%rd1+0], %fd1;
    st.global.f64 [%rd1+32], %fd2;
    st.global.f64 [%rd1+64], %fd3;
    st.global.f64 [%rd1+96], %fd4;
    ret;
}
PTX
}

emit_runtime_harness() {
  cat > "$OUT_RUNNER" <<'CU'
#include <cuda.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(call) do { \
  CUresult status__ = (call); \
  if (status__ != CUDA_SUCCESS) { \
    const char *name__ = nullptr; \
    cuGetErrorName(status__, &name__); \
    std::fprintf(stderr, "CUDA failure %s at %s:%d\n", name__ ? name__ : "unknown", __FILE__, __LINE__); \
    return 1; \
  } \
} while (0)

int main(int argc, char **argv) {
  const char *cubin_path = argc > 1 ? argv[1] : "gpu_knowledge_vec4_aggregate_marker.cubin";

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
  CHECK(cuModuleLoad(&mod, cubin_path));
  CUfunction fn;
  CHECK(cuModuleGetFunction(&fn, mod, "gpu_knowledge_vec4_aggregate_marker"));

  double host[16];
  std::memset(host, 0, sizeof(host));
  CUdeviceptr out = 0;
  CHECK(cuMemAlloc(&out, sizeof(host)));
  CHECK(cuMemsetD8(out, 0, sizeof(host)));

  void *args[] = {&out};
  CHECK(cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
  CHECK(cuCtxSynchronize());
  CHECK(cuMemcpyDtoH(host, out, sizeof(host)));

  const int lanes[4] = {0, 4, 8, 12};
  const double expected[4] = {1.0, 2.0, 3.0, 4.0};
  for (int i = 0; i < 4; ++i) {
    const double got = host[lanes[i]];
    if (std::fabs(got - expected[i]) > 1e-12) {
      std::fprintf(stderr, "mismatch offset=%d got %.17g expected %.17g\n", lanes[i] * 8, got, expected[i]);
      return 1;
    }
  }

  CHECK(cuMemFree(out));
  CHECK(cuModuleUnload(mod));
  CHECK(cuDevicePrimaryCtxRelease(dev));

  std::printf("PASS gpu_knowledge_vec4_aggregate_marker on %s cc %d.%d copyback offsets=0,32,64,96 values=1,2,3,4\n",
              name, major, minor);
  return 0;
}
CU
}

write_json() {
  local status="$1"
  local reason="$2"
  local ptxas_bin="${3:-}"
  local ptxas_version="${4:-}"
  local ptxas_output="${5:-}"
  python3 - "$ROOT_DIR" "$OUT_JSON" "$status" "$reason" "$ARCH" "$ptxas_bin" "$ptxas_version" "$ptxas_output" "$OUT_PTX" "$OUT_CUBIN" "$OUT_RUNNER" <<'PY'
import json
import hashlib
import pathlib
import sys
from datetime import datetime, timezone

root = pathlib.Path(sys.argv[1])
out_json = pathlib.Path(sys.argv[2])
status, reason, arch, ptxas_bin, ptxas_version, ptxas_output = sys.argv[3:9]
ptx, cubin, runner = map(pathlib.Path, sys.argv[9:12])

def rel(path):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)

def entry(path):
    if not path.exists():
        return {"path": rel(path), "present": False}
    data = path.read_bytes()
    return {
        "path": rel(path),
        "present": True,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }

payload = {
    "schema": "sounio.gpu-knowledge-vec4-ptxas-probe.v1",
    "status": status,
    "reason": reason,
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "arch": arch,
    "ptxas": {
        "path": ptxas_bin,
        "version": ptxas_version,
        "output": ptxas_output,
    },
    "artifacts": {
        "ptx": entry(ptx),
        "cubin": entry(cubin),
        "runtime_harness": entry(runner),
    },
    "runtime_launch_contract": {
        "kernel": "gpu_knowledge_vec4_aggregate_marker",
        "params": ["out_ptr"],
        "copyback_offsets_bytes": [0, 32, 64, 96],
        "expected_value_lanes": [1.0, 2.0, 3.0, 4.0],
        "harness": rel(runner),
        "status": "ptxas_only_not_launched",
    },
    "boundaries": [
        "ptxas_toolchain_assembly_proof",
        "cuda_driver_api_harness_shape_proof",
        "does_not_claim_cuda_device_runtime_execution",
        "does_not_claim_automatic_backend_pack_unpack",
        "does_not_claim_imported_runtime_fixture",
    ],
}
out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

emit_ptx
emit_runtime_harness

PTXAS_BIN="$(find_ptxas || true)"
if [ -z "$PTXAS_BIN" ]; then
  write_json "blocked" "ptxas_not_found"
  echo "gpu_knowledge_vec4_ptxas_probe: BLOCKED ptxas_not_found report=$(rel_path "$OUT_JSON")"
  exit 0
fi

PTXAS_VERSION="$("$PTXAS_BIN" --version 2>&1 | tr '\n' ' ')"
PTXAS_LOG="$OUT_DIR/ptxas.log"
if "$PTXAS_BIN" -arch="$ARCH" "$OUT_PTX" -o "$OUT_CUBIN" >"$PTXAS_LOG" 2>&1; then
  PTXAS_OUTPUT="$(cat "$PTXAS_LOG")"
  write_json "pass" "ptxas_assembled_launchable_marker" "$PTXAS_BIN" "$PTXAS_VERSION" "$PTXAS_OUTPUT"
  echo "gpu_knowledge_vec4_ptxas_probe: PASS report=$(rel_path "$OUT_JSON")"
else
  PTXAS_OUTPUT="$(cat "$PTXAS_LOG")"
  write_json "fail" "ptxas_failed" "$PTXAS_BIN" "$PTXAS_VERSION" "$PTXAS_OUTPUT"
  echo "gpu_knowledge_vec4_ptxas_probe: FAIL ptxas_failed report=$(rel_path "$OUT_JSON")" >&2
  exit 1
fi
