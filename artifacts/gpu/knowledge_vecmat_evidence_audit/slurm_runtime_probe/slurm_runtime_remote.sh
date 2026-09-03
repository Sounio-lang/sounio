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
