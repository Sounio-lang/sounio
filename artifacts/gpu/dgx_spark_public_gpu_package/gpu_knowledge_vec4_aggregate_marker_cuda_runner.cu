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
