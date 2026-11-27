// Native ABI admission control for the LLM session bridge (macOS).
#include <array>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "%s\n", dlerror()); return 1; }
    constexpr std::array names{
        "sounio_metal_session_create", "sounio_metal_session_destroy",
        "sounio_metal_buffer_nocopy", "sounio_metal_buffer_alloc",
        "sounio_metal_buffer_contents", "sounio_metal_buffer_free",
        "sounio_metal_compile_msl", "sounio_metal_cmd_begin",
        "sounio_metal_cmd_cancel", "sounio_metal_cmd_set_buffer",
        "sounio_metal_cmd_set_u64", "sounio_metal_cmd_dispatch_1d",
        "sounio_metal_cmd_commit_sync"
    };
    bool ok = true;
    for (const char* name : names) {
        if (!dlsym(h, name)) { std::fprintf(stderr, "missing %s\n", name); ok = false; }
    }
    if (ok) {
        auto create = reinterpret_cast<int64_t(*)()>(dlsym(h, names[0]));
        auto destroy = reinterpret_cast<int64_t(*)(int64_t)>(dlsym(h, names[1]));
        const auto session = create();
        ok = session > 0 && destroy(session) == 0;
    }
    dlclose(h);
    std::puts(ok ? "METAL_SESSION_ABI_PASS" : "METAL_SESSION_ABI_FAIL");
    return ok ? 0 : 1;
}
