// Oracle for tests/run-pass/gum_so_32_channels.sio: 32 measured quantities
// x_i = i + 1.5 (i = 0..31), each with standard uncertainty u = 0.1, and
//   r = x_0 x_1 + sum_{i=2..31} x_i^2.
// Closed forms:  dr/dx_0 = x_1, dr/dx_1 = x_0, dr/dx_i = 2 x_i;
//   H_01 = 1, H_ii = 2 (i >= 2), H_00 = H_11 = 0;
//   Var  = u^2 (x_1^2 + x_0^2 + sum_{i>=2} 4 x_i^2);
//   E2   = r + 1/2 u^2 sum_i H_ii.
// Build: g++ -std=c++23 -O2 gum_so_32_channels.cpp && ./a.out
#include <cstdio>
int main() {
    const int n = 32;
    const double u = 0.1;
    double x[n];
    for (int i = 0; i < n; ++i) x[i] = i + 1.5;
    double r = x[0] * x[1], var = u * u * (x[1] * x[1] + x[0] * x[0]), h = 0.0;
    for (int i = 2; i < n; ++i) { r += x[i] * x[i]; var += u * u * 4.0 * x[i] * x[i]; h += 2.0; }
    std::printf("variance %.17g\nhessian01 1\nmean %.17g\n", var, r + 0.5 * u * u * h);
    return 0;
}
