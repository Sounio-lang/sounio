// Oracle for tests/run-pass/gum_fo_hof_drawing_fnptr_exact.sio:
// r = a0 * m with a0 = 2 certain and m = 2 measured with u = 0.3, so
// Var(r) = a0^2 u^2.
// Build: g++ -std=c++23 -O2 gum_fo_hof_drawing_exact.cpp && ./a.out
#include <cstdio>
int main() {
    const double a0 = 2.0, u = 0.3;
    std::printf("var %.17g\n", a0 * a0 * u * u);
    return 0;
}
