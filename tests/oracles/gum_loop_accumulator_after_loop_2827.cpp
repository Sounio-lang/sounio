// Analytic oracle for tests/run-pass/gum_loop_accumulator_after_loop_2827.sio (#2827).
// Build: g++ -std=c++23 -O2 gum_loop_accumulator_after_loop_2827.cpp && ./a.out
// First-order GUM for acc = sum_i w_i * y_i with y_i ~ (., u_i) independent and
// w_i = 1/u_i^2 exact: var(acc) = sum_i w_i^2 u_i^2 = sum_i 1/u_i^2 = W, and
// mu = acc / W gives var(mu) = 1/W; var(0.5 * acc) = W / 4.
// The last three lines are the stale_init / between / straight cases.
#include <cstdio>
int main() {
    const long double u[3] = {0.1L, 0.2L, 0.15L};
    long double W = 0.0L;
    for (long double x : u) W += 1.0L / (x * x);
    std::printf("acc_var=%.17Lg\nmu_var=%.17Lg\nhalf_var=%.17Lg\n", W, 1.0L / W, W / 4.0L);
    // stale_init: a freshly declared 0.0 carries no uncertainty.
    std::printf("stale_init=%.17Lg\n", 0.0L);
    // between: acc = sum_{i<4} 0.5 * m with ONE m (sigma 0.1): d acc / d m = 2.
    const long double s_m = 4 * 0.5L;
    std::printf("between=%.17Lg\n", s_m * s_m * 0.1L * 0.1L);
    // straight: (x + m2) - m2 with x exact: d/d m2 = 0.
    std::printf("straight=%.17Lg\n", 0.0L);
}
