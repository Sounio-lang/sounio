#include <array>
#include <cstdint>
#include <cstdio>

extern "C" void pireus_xor_avx512(const double*, const double*, double*);
extern "C" const std::int8_t pireus_expected_sign[256];

int main() {
    std::array<double, 16> a{};
    std::array<double, 16> b{};
    std::array<double, 16> out{};
    int failures = 0;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            a.fill(0.0);
            b.fill(0.0);
            out.fill(7.0);
            a[i] = 1.0;
            b[j] = 1.0;
            pireus_xor_avx512(a.data(), b.data(), out.data());
            const int destination = i ^ j;
            const double expected = static_cast<double>(pireus_expected_sign[i * 16 + j]);
            for (int d = 0; d < 16; ++d) {
                const double want = d == destination ? expected : 0.0;
                if (out[d] != want) {
                    if (failures < 8) {
                        std::printf("mismatch i=%d j=%d d=%d got=%.17g want=%.17g\n",
                                    i, j, d, out[d], want);
                    }
                    ++failures;
                }
            }
        }
    }
    std::printf("schema=pireus-xeon-avx512-material-result-v1\n");
    std::printf("producer_language=C++\nproducer_role=MATERIAL_PARITY\n");
    std::printf("basis_pairs=256 component_checks=4096 failures=%d\n", failures);
    std::printf("result=%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
