// Admission-only CUDA Driver API probe. No inference or kernel changes.
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <string_view>

namespace {
constexpr std::size_t required = 16341738616ULL + 3801088ULL + 2 * 83886080ULL + 262144ULL;
static_assert(required == 16513574008ULL);
constexpr int admission(std::size_t free_bytes) { return free_bytes < required ? 42 : 0; }
template<class T> T symbol(void* lib, const char* name) {
    return reinterpret_cast<T>(dlsym(lib, name));
}
int failed(const char* stage, int rc) {
    std::cerr << "VRAM_PREFLIGHT_FAIL stage=" << stage << " CUresult=" << rc << '\n';
    return 1;
}
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--selftest") {
        if (admission(required - 1) != 42 || admission(required) != 0 || admission(required + 1) != 0) return 1;
        std::cout << "VRAM_ADMISSION_SELFTEST_PASS\n";
        return 0;
    }
    if (argc != 2) return failed("args", -1);
    const std::string_view smarg(argv[1]);
    int expected = 0;
    auto parsed = std::from_chars(smarg.data(), smarg.data() + smarg.size(), expected);
    if (parsed.ec != std::errc{} || parsed.ptr != smarg.data() + smarg.size() || (expected != 75 && expected != 86 && expected != 89)) return failed("expected-sm", -1);
    const char* path = std::getenv("SOUNIO_CUDA_DRIVER_LIB");
    void* lib = dlopen(path && *path ? path : "libcuda.so.1", RTLD_LAZY);
    if (!lib) return failed("dlopen", -1);
    using Init = int(*)(unsigned);
    using Device = int(*)(int*, int);
    using Attr = int(*)(int*, int, int);
    using Create = int(*)(void**, unsigned, int);
    using Destroy = int(*)(void*);
    using Info = int(*)(std::size_t*, std::size_t*);
    auto init = symbol<Init>(lib,"cuInit");
    auto device = symbol<Device>(lib,"cuDeviceGet");
    auto attr = symbol<Attr>(lib,"cuDeviceGetAttribute");
    auto create = symbol<Create>(lib,"cuCtxCreate_v2");
    auto destroy = symbol<Destroy>(lib,"cuCtxDestroy_v2");
    auto info = symbol<Info>(lib,"cuMemGetInfo_v2");
    if (!init || !device || !attr || !create || !destroy || !info) return failed("symbols", -1);
    int rc = init(0); if (rc) return failed("cuInit", rc);
    int dev = 0, major = 0, minor = 0;
    rc = device(&dev, 0); if (rc) return failed("cuDeviceGet", rc);
    rc = attr(&major,75,dev); if (rc) return failed("sm-major", rc);
    rc = attr(&minor,76,dev); if (rc) return failed("sm-minor", rc);
    const int sm = major * 10 + minor;
    if (sm != expected) return failed("unexpected-sm", sm);
    void* ctx = nullptr;
    rc = create(&ctx,0,dev); if (rc) return failed("cuCtxCreate_v2",rc);
    std::size_t free_bytes = 0, total_bytes = 0;
    rc = info(&free_bytes,&total_bytes);
    std::cout << "VRAM_PREFLIGHT sm=" << sm << " free_bytes=" << free_bytes << " total_bytes=" << total_bytes
              << " required_bytes=" << required << " CUresult=" << rc << '\n' << std::flush;
    const int destroy_rc = destroy(ctx);
    dlclose(lib);
    if (rc) return failed("cuMemGetInfo_v2", rc);
    if (destroy_rc) return failed("cuCtxDestroy_v2",destroy_rc);
    const int result = admission(free_bytes);
    std::cout << (result ? "VRAM_INSUFFICIENT" : "VRAM_ADMISSION_PASS") << '\n';
    return result;
}
