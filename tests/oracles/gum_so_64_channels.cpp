// Oracle for tests/run-pass/gum_so_64_channels.sio: 64 measured quantities
// x_i = i + 1.5 (i = 0..63), each with standard uncertainty u = 0.1, and
//   r = x_0 x_1 + sum_{i=2..63} x_i^2 + x_62 x_63.
// The last term puts a cross second derivative in the last column of the
// 2080-slot Hessian triangle (slot 63*64/2 + 62 = 2078).
// Closed forms:  dr/dx_0 = x_1, dr/dx_1 = x_0, dr/dx_i = 2 x_i (2 <= i <= 61),
//   dr/dx_62 = 2 x_62 + x_63, dr/dx_63 = 2 x_63 + x_62;
//   H_01 = H_62,63 = 1, H_ii = 2 (i >= 2), H_00 = H_11 = 0;
//   Var  = u^2 sum_i (dr/dx_i)^2;
//   E2   = r + 1/2 u^2 sum_i H_ii.
// Build: g++ -std=c++23 -O2 gum_so_64_channels.cpp && ./a.out
#include <cstdio>
int main() {
    const int n = 64;
    const double u = 0.1;
    double x[n], g[n];
    for (int i = 0; i < n; ++i) x[i] = i + 1.5;
    double r = x[0] * x[1], h = 0.0;
    g[0] = x[1];
    g[1] = x[0];
    for (int i = 2; i < n; ++i) { r += x[i] * x[i]; g[i] = 2.0 * x[i]; h += 2.0; }
    r += x[62] * x[63];
    g[62] += x[63];
    g[63] += x[62];
    double var = 0.0;
    for (int i = 0; i < n; ++i) var += u * u * g[i] * g[i];
    std::printf("variance %.17g\nhessian01 1\nhessian_62_63 1\nhessian_diag 2\nmean %.17g\n", var, r + 0.5 * u * u * h);
    return 0;
}
