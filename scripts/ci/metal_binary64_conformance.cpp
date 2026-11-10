#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <vector>
struct Arg { uint64_t value, kind; };
struct Array {
    std::vector<uint64_t> words;
    uint64_t* header;
    explicit Array(size_t count) : words(count + 4), header(words.data()) {
        words[1] = count * 8;
        words[2] = count;
    }
    uint64_t& operator[](size_t i) { return words[i + 4]; }
    uint64_t ref() { return reinterpret_cast<uint64_t>(&header); }
};
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::fprintf(stderr, "%s\n", dlerror()); return 1; }
    using Launch = int64_t (*)(const char*, const char*, const Arg*, uint64_t, uint64_t);
    auto launch = reinterpret_cast<Launch>(dlsym(library, "sounio_metal_launch"));
    if (!launch) return 1;
    constexpr size_t n = 8192;
    Array a(n), b(n), out(5 * n);
    constexpr uint64_t edges[] = {
        0, 0x8000000000000000, 1, 0x8000000000000001,
        0x000fffffffffffff, 0x0010000000000000, 0x3ff0000000000000,
        0xbff0000000000000, 0x3ff0000000000001, 0x7fefffffffffffff,
        0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000001
    };
    uint64_t state = 0x4d414441524f535f;
    auto random = [&] { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state; };
    for (size_t i = 0; i < n; ++i) {
        a[i] = i < std::size(edges) * std::size(edges) ? edges[i / std::size(edges)] : random();
        b[i] = i < std::size(edges) * std::size(edges) ? edges[i % std::size(edges)] : random();
    }
    constexpr char source[] = R"(
using namespace metal;
kernel void sounio_binary64_conformance(constant ulong& n [[buffer(0)]],
 device const ulong* a [[buffer(1)]], device const ulong* b [[buffer(2)]],
 device ulong* out [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
 if (tid >= n) return;
 out[tid] = __softfloat64_fadd(a[tid], b[tid], 0u);
 out[n+tid] = __softfloat64_fsub(a[tid], b[tid], 0u);
 out[2*n+tid] = __softfloat64_fmul(a[tid], b[tid], 0u);
 out[3*n+tid] = __softfloat64_fdiv(a[tid], b[tid], 0u);
 out[4*n+tid] = __softfloat64_fsqrt(a[tid], 0u);
})";
    const Arg args[] = {{n, 0}, {a.ref(), 2}, {b.ref(), 2}, {out.ref(), 3}};
    if (launch(source, "sounio_binary64_conformance", args, 4, n)) return 1;
    size_t failures = 0, checked = 0;
    for (size_t i = 0; i < n; ++i) {
        const double x = std::bit_cast<double>(a[i]), y = std::bit_cast<double>(b[i]);
        const double expected[] = {x + y, x - y, x * y, x / y, std::sqrt(x)};
        for (size_t op = 0; op < 5; ++op) {
            const auto got = out[op * n + i], want = std::bit_cast<uint64_t>(expected[op]);
            ++checked;
            if (got != want && !(std::isnan(expected[op]) && std::isnan(std::bit_cast<double>(got)))) {
                if (failures < 12) std::fprintf(stderr,
                    "FAIL i=%zu op=%zu a=%016llx b=%016llx got=%016llx want=%016llx\n",
                    i, op, static_cast<unsigned long long>(a[i]), static_cast<unsigned long long>(b[i]),
                    static_cast<unsigned long long>(got), static_cast<unsigned long long>(want));
                ++failures;
            }
        }
    }
    std::printf("metal_binary64 checked=%zu failures=%zu oracle=C++23 policy=packed_RNE\n", checked, failures);
    return failures ? 1 : 0;
}
