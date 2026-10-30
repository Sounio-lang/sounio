// C++23 oracle for tests/stdlib/math/test_libm_trig.sio.
// Prints glibc sin/cos at the test's points as exact IEEE-754 bit patterns
// (decimal i64, the form f64_to_bits returns) so the test can hold libm_sin /
// libm_cos to within 1 ulp.
// Build: g++ -std=c++23 -O2 libm_trig_glibc.cpp && ./a.out
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
int main() {
    const double xs[] = {
        0.0, 1.0e-8, 0.5, 0.7853981633974483, 1.0, 1.5707963267948966, 2.0, 3.0,
        3.141592653589793, 4.71238898038469, 5.0, 6.283185307179586, 10.0,
        314.1592653589793, 1000.0, 12345.678, 1000000.0, -1.0, -3.141592653589793,
        -1000.0, 1.6e6, -1.6e6,
    };
    for (double x : xs) {
        std::printf("%.17g sin=%lld cos=%lld\n", x,
                    static_cast<long long>(std::bit_cast<std::int64_t>(std::sin(x))),
                    static_cast<long long>(std::bit_cast<std::int64_t>(std::cos(x))));
    }
}
