#!/usr/bin/env bash
# Launch the GPU Knowledge Vec4 marker PTX on a Slurm GPU node.
#
# This is a runtime proof, not a compiler proof. It uses the CUDA Driver API via
# dlopen so the compute node does not need nvcc or CUDA headers. The local PTX
# remains the artifact under test.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_GPU_KNOWLEDGE_SLURM_RUNTIME_DIR:-$ROOT_DIR/artifacts/gpu/knowledge_vecmat_evidence_audit/slurm_runtime_probe}"
OUT_JSON="$OUT_DIR/gpu_knowledge_vec4_slurm_runtime_probe.v1.json"
OUT_STDOUT="$OUT_DIR/slurm_runtime.stdout"
OUT_STDERR="$OUT_DIR/slurm_runtime.stderr"
REMOTE_SCRIPT="$OUT_DIR/slurm_runtime_remote.sh"
LOCAL_RUNNER_C="$OUT_DIR/slurm_runtime_runner.c"
LOCAL_RUNNER_BIN="$OUT_DIR/slurm_runtime_runner"
PTX="${SOUNIO_GPU_KNOWLEDGE_SLURM_RUNTIME_PTX:-$ROOT_DIR/artifacts/gpu/dgx_spark_public_gpu_package/gpu_knowledge_vec4_aggregate_marker.ptx}"
SLURM_PARTITION="${SOUNIO_GPU_KNOWLEDGE_SLURM_PARTITION:-gpu-orangefs}"
SLURM_GRES="${SOUNIO_GPU_KNOWLEDGE_SLURM_GRES:-gpu:1}"
SLURM_TIME="${SOUNIO_GPU_KNOWLEDGE_SLURM_TIME:-00:03:00}"

mkdir -p "$OUT_DIR"

write_json() {
  local status="$1"
  local reason="$2"
  local exit_code="$3"
  python3 - "$ROOT_DIR" "$OUT_JSON" "$status" "$reason" "$exit_code" "$PTX" "$OUT_STDOUT" "$OUT_STDERR" "$SLURM_PARTITION" "$SLURM_GRES" "$SLURM_TIME" <<'PY'
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

root = pathlib.Path(sys.argv[1])
out_json = pathlib.Path(sys.argv[2])
status, reason, exit_code = sys.argv[3], sys.argv[4], int(sys.argv[5])
ptx, stdout_path, stderr_path = map(pathlib.Path, sys.argv[6:9])
partition, gres, time_limit = sys.argv[9:12]

def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)

def file_entry(path: pathlib.Path) -> dict:
    if not path.exists() or not path.is_file():
        return {"path": rel(path), "present": False}
    data = path.read_bytes()
    return {
        "path": rel(path),
        "present": True,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }

stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
payload = {
    "schema": "sounio.gpu-knowledge-vec4-slurm-runtime-probe.v1",
    "status": status,
    "reason": reason,
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "slurm": {
        "partition": partition,
        "gres": gres,
        "time_limit": time_limit,
        "exit_code": exit_code,
    },
    "runtime_launch_contract": {
        "kernel": "gpu_knowledge_vec4_aggregate_marker",
        "params": ["out_ptr"],
        "copyback_offsets_bytes": [0, 32, 64, 96],
        "expected_value_lanes": [1.0, 2.0, 3.0, 4.0],
        "status": "cuda_device_runtime_pass" if status == "pass" else "missing_or_unproved",
    },
    "artifacts": {
        "ptx": file_entry(ptx),
        "stdout": file_entry(stdout_path),
        "stderr": file_entry(stderr_path),
    },
    "stdout_tail": stdout[-4000:],
    "stderr_tail": stderr[-4000:],
    "boundaries": [
        "slurm_gpu_runtime_probe",
        "does_not_claim_dgx_spark_runtime",
        "does_not_claim_automatic_backend_pack_unpack",
        "does_not_claim_imported_runtime_fixture",
    ],
}
out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

if [ ! -s "$PTX" ]; then
  : >"$OUT_STDOUT"
  printf 'missing PTX: %s\n' "$PTX" >"$OUT_STDERR"
  write_json "blocked" "ptx_missing" 1
  echo "gpu_knowledge_vec4_slurm_runtime_probe: BLOCKED ptx_missing report=${OUT_JSON#$ROOT_DIR/}"
  exit 0
fi

cat >"$REMOTE_SCRIPT" <<'REMOTE'
set -euo pipefail

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work"

printf '%s' "$PTX_B64" | base64 -d > marker.ptx

cat > runner.c <<'C'
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76

typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDeviceGet_t)(CUdevice *, int);
typedef CUresult (*cuDeviceGetName_t)(char *, int, CUdevice);
typedef CUresult (*cuDeviceGetAttribute_t)(int *, int, CUdevice);
typedef CUresult (*cuDevicePrimaryCtxRetain_t)(CUcontext *, CUdevice);
typedef CUresult (*cuCtxSetCurrent_t)(CUcontext);
typedef CUresult (*cuModuleLoadDataEx_t)(CUmodule *, const void *, unsigned int, void *, void *);
typedef CUresult (*cuModuleGetFunction_t)(CUfunction *, CUmodule, const char *);
typedef CUresult (*cuMemAlloc_t)(CUdeviceptr *, size_t);
typedef CUresult (*cuMemsetD8_t)(CUdeviceptr, unsigned char, size_t);
typedef CUresult (*cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, void *, void **, void **);
typedef CUresult (*cuCtxSynchronize_t)(void);
typedef CUresult (*cuMemcpyDtoH_t)(void *, CUdeviceptr, size_t);
typedef CUresult (*cuMemFree_t)(CUdeviceptr);
typedef CUresult (*cuModuleUnload_t)(CUmodule);
typedef CUresult (*cuDevicePrimaryCtxRelease_t)(CUdevice);
typedef CUresult (*cuGetErrorName_t)(CUresult, const char **);

static cuGetErrorName_t p_cuGetErrorName;

static void *must_sym(void *lib, const char *name) {
  void *sym = dlsym(lib, name);
  if (!sym) {
    fprintf(stderr, "missing CUDA symbol %s\n", name);
    exit(1);
  }
  return sym;
}

static int check(CUresult status, const char *call, int line) {
  if (status == CUDA_SUCCESS) return 0;
  const char *name = NULL;
  if (p_cuGetErrorName) p_cuGetErrorName(status, &name);
  fprintf(stderr, "CUDA failure %s at %s:%d\n", name ? name : "unknown", call, line);
  return 1;
}

#define CHECK(call) do { if (check((call), #call, __LINE__)) return 1; } while (0)

static char *read_file(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    perror(path);
    exit(1);
  }
  if (fseek(f, 0, SEEK_END) != 0) exit(1);
  long n = ftell(f);
  if (n < 0) exit(1);
  rewind(f);
  char *buf = (char *)calloc((size_t)n + 1, 1);
  if (!buf) exit(1);
  if (fread(buf, 1, (size_t)n, f) != (size_t)n) exit(1);
  fclose(f);
  return buf;
}

int main(int argc, char **argv) {
  const char *ptx_path = argc > 1 ? argv[1] : "marker.ptx";
  void *cuda = dlopen("libcuda.so.1", RTLD_NOW);
  if (!cuda) {
    fprintf(stderr, "failed to open libcuda.so.1: %s\n", dlerror());
    return 1;
  }

  cuInit_t cuInit = (cuInit_t)must_sym(cuda, "cuInit");
  cuDeviceGet_t cuDeviceGet = (cuDeviceGet_t)must_sym(cuda, "cuDeviceGet");
  cuDeviceGetName_t cuDeviceGetName = (cuDeviceGetName_t)must_sym(cuda, "cuDeviceGetName");
  cuDeviceGetAttribute_t cuDeviceGetAttribute = (cuDeviceGetAttribute_t)must_sym(cuda, "cuDeviceGetAttribute");
  cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain = (cuDevicePrimaryCtxRetain_t)must_sym(cuda, "cuDevicePrimaryCtxRetain");
  cuCtxSetCurrent_t cuCtxSetCurrent = (cuCtxSetCurrent_t)must_sym(cuda, "cuCtxSetCurrent");
  cuModuleLoadDataEx_t cuModuleLoadDataEx = (cuModuleLoadDataEx_t)must_sym(cuda, "cuModuleLoadDataEx");
  cuModuleGetFunction_t cuModuleGetFunction = (cuModuleGetFunction_t)must_sym(cuda, "cuModuleGetFunction");
  cuMemAlloc_t cuMemAlloc = (cuMemAlloc_t)must_sym(cuda, "cuMemAlloc_v2");
  cuMemsetD8_t cuMemsetD8 = (cuMemsetD8_t)must_sym(cuda, "cuMemsetD8_v2");
  cuLaunchKernel_t cuLaunchKernel = (cuLaunchKernel_t)must_sym(cuda, "cuLaunchKernel");
  cuCtxSynchronize_t cuCtxSynchronize = (cuCtxSynchronize_t)must_sym(cuda, "cuCtxSynchronize");
  cuMemcpyDtoH_t cuMemcpyDtoH = (cuMemcpyDtoH_t)must_sym(cuda, "cuMemcpyDtoH_v2");
  cuMemFree_t cuMemFree = (cuMemFree_t)must_sym(cuda, "cuMemFree_v2");
  cuModuleUnload_t cuModuleUnload = (cuModuleUnload_t)must_sym(cuda, "cuModuleUnload");
  cuDevicePrimaryCtxRelease_t cuDevicePrimaryCtxRelease = (cuDevicePrimaryCtxRelease_t)must_sym(cuda, "cuDevicePrimaryCtxRelease");
  p_cuGetErrorName = (cuGetErrorName_t)must_sym(cuda, "cuGetErrorName");

  char *ptx = read_file(ptx_path);

  CHECK(cuInit(0));
  CUdevice dev = 0;
  CHECK(cuDeviceGet(&dev, 0));
  char name[256] = {0};
  int major = 0;
  int minor = 0;
  CHECK(cuDeviceGetName(name, sizeof(name), dev));
  CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));

  CUcontext ctx = NULL;
  CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CHECK(cuCtxSetCurrent(ctx));

  CUmodule mod = NULL;
  CHECK(cuModuleLoadDataEx(&mod, ptx, 0, NULL, NULL));
  CUfunction fn = NULL;
  CHECK(cuModuleGetFunction(&fn, mod, "gpu_knowledge_vec4_aggregate_marker"));

  double host[16];
  memset(host, 0, sizeof(host));
  CUdeviceptr out = 0;
  CHECK(cuMemAlloc(&out, sizeof(host)));
  CHECK(cuMemsetD8(out, 0, sizeof(host)));

  void *args[] = {&out};
  CHECK(cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL));
  CHECK(cuCtxSynchronize());
  CHECK(cuMemcpyDtoH(host, out, sizeof(host)));

  const int lanes[4] = {0, 4, 8, 12};
  const double expected[4] = {1.0, 2.0, 3.0, 4.0};
  for (int i = 0; i < 4; ++i) {
    const double got = host[lanes[i]];
    if (fabs(got - expected[i]) > 1e-12) {
      fprintf(stderr, "mismatch offset=%d got %.17g expected %.17g\n", lanes[i] * 8, got, expected[i]);
      return 1;
    }
  }

  CHECK(cuMemFree(out));
  CHECK(cuModuleUnload(mod));
  CHECK(cuDevicePrimaryCtxRelease(dev));
  free(ptx);
  dlclose(cuda);

  printf("PASS gpu_knowledge_vec4_aggregate_marker on %s cc %d.%d copyback offsets=0,32,64,96 values=1,2,3,4\n", name, major, minor);
  return 0;
}
C

if command -v cc >/dev/null 2>&1; then
  cc -O2 runner.c -ldl -lm -o runner
elif [ -n "${RUNNER_B64:-}" ]; then
  printf '%s' "$RUNNER_B64" | base64 -d > runner
  chmod +x runner
else
  echo "cc not found and no precompiled runner was provided" >&2
  exit 127
fi
./runner marker.ptx
REMOTE

awk '
  /^cat > runner\.c <<'\''C'\''$/ { emit=1; next }
  emit && /^C$/ { emit=0; next }
  emit { print }
' "$REMOTE_SCRIPT" >"$LOCAL_RUNNER_C"

RUNNER_B64=""
if command -v cc >/dev/null 2>&1; then
  cc -O2 "$LOCAL_RUNNER_C" -ldl -lm -o "$LOCAL_RUNNER_BIN"
  RUNNER_B64="$(base64 -w0 "$LOCAL_RUNNER_BIN")"
fi

PTX_B64="$(base64 -w0 "$PTX")"
export PTX_B64
export RUNNER_B64

set +e
srun \
  --partition="$SLURM_PARTITION" \
  --gres="$SLURM_GRES" \
  --nodes=1 \
  --ntasks=1 \
  --time="$SLURM_TIME" \
  --chdir=/tmp \
  bash -s <"$REMOTE_SCRIPT" >"$OUT_STDOUT" 2>"$OUT_STDERR"
exit_code=$?
set -e

if [ "$exit_code" -eq 0 ] && grep -q "PASS gpu_knowledge_vec4_aggregate_marker" "$OUT_STDOUT"; then
  write_json "pass" "slurm_cuda_device_runtime_pass" "$exit_code"
  echo "gpu_knowledge_vec4_slurm_runtime_probe: PASS report=${OUT_JSON#$ROOT_DIR/}"
  exit 0
fi

write_json "blocked" "slurm_cuda_device_runtime_failed" "$exit_code"
echo "gpu_knowledge_vec4_slurm_runtime_probe: BLOCKED slurm_cuda_device_runtime_failed report=${OUT_JSON#$ROOT_DIR/}" >&2
exit 0
