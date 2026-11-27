// Test-only CUDA Driver API double. No GPU or CUDA toolkit needed.
#include <cstddef>
#include <cstdio>
#include <cstdint>
namespace { int allocations = 0, frees = 0; }
extern "C" {
int cuInit(unsigned) { return 0; }
int cuDeviceGet(int* d, int) { *d = 0; return 0; }
int cuDeviceGetAttribute(int* value, int attr, int) { *value = attr == 75 ? 8 : 9; return 0; }
int cuDeviceGetName(char* name, int len, int) { if (len) name[0] = '\0'; return 0; }
int cuDriverGetVersion(int* version) { *version = 13020; return 0; }
int cuCtxCreate_v2(void** ctx, unsigned, int) { *ctx = reinterpret_cast<void*>(1); return 0; }
int cuCtxDestroy_v2(void*) { std::printf("FAKE_CTX_DESTROY allocations=%d frees=%d\n", allocations, frees); std::fflush(stdout); return 0; }
int cuCtxSynchronize() { return 0; }
int cuModuleLoadData(void** mod, const void*) { *mod = reinterpret_cast<void*>(2); return 0; }
int cuModuleLoadDataEx(void** mod, const void*, unsigned, void*, void*) { *mod = reinterpret_cast<void*>(2); return 0; }
int cuModuleUnload(void*) { return 0; }
int cuModuleGetFunction(void** fn, void*, const char*) { *fn = reinterpret_cast<void*>(3); return 0; }
int cuModuleGetGlobal_v2(std::uint64_t* ptr, std::size_t* bytes, void*, const char*) { *ptr = 4; *bytes = 8; return 0; }
int cuMemAlloc_v2(std::uint64_t* ptr, std::size_t bytes) {
    ++allocations;
    std::printf("FAKE_ALLOC_CALL=%d bytes=%zu\n", allocations, bytes); std::fflush(stdout);
    if (allocations == 2) { *ptr = 0; return 2; }
    *ptr = 65536 + static_cast<std::uint64_t>(allocations) * 4096;
    return 0;
}
int cuMemFree_v2(std::uint64_t) { ++frees; return 0; }
int cuMemcpyHtoD_v2(std::uint64_t, const void*, std::size_t) { return 0; }
int cuMemcpyDtoH_v2(void*, std::uint64_t, std::size_t) { return 0; }
int cuLaunchKernel(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*, void*, void*) { return 0; }
}
