// Oracle for tests/run-pass/gum_fo_tangent_clone_hof.sio.
// First-order GUM (JCGM 100:2008 eq. 10/13) and the second-order Hessian,
// both from analytic derivatives of the composed function (the higher-order
// call is just the composition it names).
// Build: g++ -std=c++23 -O2 gum_fo_tangent_clone_hof.cpp && ./a.out
#include <cmath>
#include <cstdio>

int main() {
    const double x = 5.0, ux = 0.3, z = 2.0, uz = 0.1;
    const double vx = ux * ux, vz = uz * uz;
    auto out = [](const char* n, double v) { std::printf("%s %.17g\n", n, v); };

    // dbl(y) = 2y, sq(y) = y^2, expo(y) = exp(-y/10)
    const double e5 = std::exp(-x / 10.0);
    // apply1(f, x) = f(x)
    out("hof_dbl", 4.0 * vx);
    out("hof_sq", (2.0 * x) * (2.0 * x) * vx);
    const double dexpo = -0.1 * e5;
    out("hof_expo", dexpo * dexpo * vx);
    // apply2(f, g, x) = f(g(x))
    out("two_dbl_sq", (4.0 * x) * (4.0 * x) * vx);            // 2 x^2
    out("two_sq_dbl", (8.0 * x) * (8.0 * x) * vx);            // 4 x^2
    const double g = std::exp(-x * x / 10.0);                  // expo(sq(x))
    const double dg = -(x / 5.0) * g;
    out("two_expo_sq", dg * dg * vx);
    // applyn(f, x, n): f applied n times
    out("loop_dbl3", (8.0 * 8.0) * vx);                        // 8 x
    out("loop_sq2", (4.0 * x * x * x) * (4.0 * x * x * x) * vx);  // x^4
    // forward(f, x) = apply1(f, x) * x
    out("fwd_dbl", (4.0 * x) * (4.0 * x) * vx);                // 2 x^2
    out("fwd_sq", (3.0 * x * x) * (3.0 * x * x) * vx);         // x^3
    // compose_fwd(f, g, x) = apply2(g, f, x) = g(f(x)); f = dbl, g = sq
    out("fwd2_dbl_sq", (8.0 * x) * (8.0 * x) * vx);            // (2x)^2
    // let h = dbl; apply1(h, x)
    out("alias_dbl", 4.0 * vx);
    // the call stays correlated with its own input
    out("cancel_dbl", 0.0);
    out("cancel_sq", 0.0);
    // apply_bin(f, a, b) = f(a, b), f = a * b
    out("bin_mul", z * z * vx + x * x * vz);
    out("bin_mul_same", (2.0 * x) * (2.0 * x) * vx);
    // apply1(dbl, x) + apply1(sq, x): two clones of one HOF at two call sites
    const double dsum = 2.0 + 2.0 * x;
    out("two_sites", dsum * dsum * vx);
    // covariance_of(apply1(dbl, x), x) = 2 vx
    out("cov_dbl_x", 2.0 * vx);
    // a function-valued parameter forwarded through two levels
    out("fwd3_sq", (3.0 * x * x) * (3.0 * x * x) * vx);

    // second order: Hessian over the measurements (x, z)
    out("h_sq", 2.0);
    out("h_expo", 0.01 * e5);
    // expo(sq(x)) = exp(-x^2/10): g'' = (-1/5 + x^2/25) exp(-x^2/10)
    out("h_expo_sq", (-1.0 / 5.0 + x * x / 25.0) * g);
    // mixed(a, b) = a exp(-b/10), through apply_bin with (x, z)
    const double ez = std::exp(-z / 10.0);
    out("h_bin_xx", 0.0);
    out("h_bin_xz", -0.1 * ez);
    out("h_bin_zz", x * 0.01 * ez);
    return 0;
}
