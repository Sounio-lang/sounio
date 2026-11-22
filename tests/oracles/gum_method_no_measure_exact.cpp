// Oracle for tests/run-pass/gum_method_no_measure_exact.sio.
// Rig::quad(a) = k a^2 with k = 3 and a = 1.5 measured with u = 0.1:
// var = (2 k a u)^2 = 0.81, second-order mean = k a^2 + 1/2 (2 k) u^2.
// A method that neither measures nor asks for an uncertainty, and one whose
// body asks for the variance of a constant, called with a certain argument.
// Build: g++ -std=c++23 -O2 gum_method_no_measure_exact.cpp && ./a.out
#include <cstdio>
int main() {
    const double k = 3.0, a = 1.5, u = 0.1;
    const double d = 2.0 * k * a;           // d(k a^2)/da
    std::printf("quad_var %.17g\n", d * d * u * u);
    std::printf("quad_mean %.17g\n", k * a * a + 0.5 * (2.0 * k) * u * u);
    std::printf("quad_sq_var %.17g\n", (2.0 * (k * a * a) * d) * (2.0 * (k * a * a) * d) * u * u);
    std::printf("certain_var %.17g\n", 0.0);
    return 0;
}
