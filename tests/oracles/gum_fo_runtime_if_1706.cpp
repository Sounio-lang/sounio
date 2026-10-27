// Analytic oracle for tests/run-pass/gum_fo_runtime_if_1706.sio (#1706, KL-11).
// Build: g++ -std=c++23 -O2 gum_fo_runtime_if_1706.cpp && ./a.out
// First-order GUM, var(y) = sum_k (dy/dx_k)^2 sigma_k^2, with the derivative taken
// along the branch the program actually executes. Forward-mode dual numbers.
#include <cstdio>
struct D { long double v, d; };
static D mul(D a, D b) { return {a.v * b.v, a.d * b.v + a.v * b.d}; }
static D scal(long double c, D a) { return {c * a.v, c * a.d}; }
// pick(n, x) = n < 0.5 ? x : 2x  (runtime branch on an exact n)
static D pick(long double n, D x) { return n < 0.5L ? x : scal(2.0L, x); }
// pw(x, n) = x^n by recursion, pw(x, 0) = 1
static D pw(D x, int n) { return n <= 0 ? D{1.0L, 0.0L} : mul(x, pw(x, n - 1)); }
// even/odd mutual recursion: each level multiplies by 1.1
static D odd_m(D x, int n);
static D even_m(D x, int n) { return n < 1 ? x : odd_m(scal(1.1L, x), n - 1); }
static D odd_m(D x, int n) { return n < 1 ? x : even_m(scal(1.1L, x), n - 1); }
int main() {
    const long double sa = 0.05L;               // a = 2.0 +- 0.05
    D a{2.0L, 1.0L};
    auto var = [&](D y) { return y.d * y.d * sa * sa; };
    std::printf("inline_then=%.17Lg\n", var(pick(-2.0L, a)));   // then arm: x
    std::printf("inline_else=%.17Lg\n", var(pick(2.0L, a)));    // else arm: 2x
    std::printf("helper_then=%.17Lg\n", var(pick(0.0L, a)));
    std::printf("helper_else=%.17Lg\n", var(pick(1.0L, a)));
    std::printf("pw3=%.17Lg\n", var(pw(a, 3)));
    std::printf("pw5=%.17Lg\n", var(pw(a, 5)));
    std::printf("mut4=%.17Lg\n", var(even_m(a, 4)));
}
