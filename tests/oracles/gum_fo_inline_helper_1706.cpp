// C++23 oracle for tests/run-pass/gum_fo_inline_helper_1706.sio (#1706).
// Forward-mode first-order sensitivities (dual numbers), then the GUM
// first-order variance  var(f) = sum_k (df/dx_k)^2 sigma_k^2.
//   g++ -std=c++23 -O2 gum_fo_inline_helper_1706.cpp && ./a.out
#include <cstdio>

struct D { double v; double d[3]; };   // value + sensitivities to up to 3 inputs

static D lit(double v) { return {v, {0, 0, 0}}; }
static D seed(double v, int k) { D r = lit(v); r.d[k] = 1.0; return r; }
static D add(D a, D b) { D r{a.v + b.v, {}}; for (int k = 0; k < 3; ++k) r.d[k] = a.d[k] + b.d[k]; return r; }
static D sub(D a, D b) { D r{a.v - b.v, {}}; for (int k = 0; k < 3; ++k) r.d[k] = a.d[k] - b.d[k]; return r; }
static D mul(D a, D b) { D r{a.v * b.v, {}}; for (int k = 0; k < 3; ++k) r.d[k] = a.d[k] * b.v + a.v * b.d[k]; return r; }
static double var(D a, const double s[3]) { double t = 0; for (int k = 0; k < 3; ++k) t += a.d[k] * a.d[k] * s[k] * s[k]; return t; }

static D rhs(D y, D k) { return sub(lit(0.0), mul(k, y)); }
static D stage(D x, D dx) { return add(x, dx); }
static D step_chain(D v, D dx) {
    D a1 = stage(v, dx), a2 = stage(a1, dx);
    D a3 = mul(a2, lit(1.001)), a4 = sub(a3, dx), a5 = add(a4, v), a6 = mul(a5, lit(0.999));
    return sub(a6, v);
}

int main() {
    {   // Euler: y0 = 100 +- 2, k = 0.1 +- 0.01
        D y = seed(100.0, 0), k = seed(0.1, 1);
        for (int i = 0; i < 10; ++i) y = add(y, mul(lit(0.1), rhs(y, k)));
        const double s[3] = {2.0, 0.01, 0};
        std::printf("euler_var=%.6f\n", var(y, s));
    }
    {   // chain: kc = 1 +- 0.01, 40 steps of step_chain(acc, 0.01 kc)
        D kc = seed(1.0, 0), acc = kc;
        for (int i = 0; i < 40; ++i) acc = step_chain(acc, mul(kc, lit(0.01)));
        const double s[3] = {0.01, 0, 0};
        std::printf("chain_var=%.6f\n", var(acc, s));
    }
    {   // add3 of three independent measurements
        D r = add(add(seed(100.0, 0), seed(50.0, 1)), seed(10.0, 2));
        const double s[3] = {2.0, 1.0, 3.0};
        std::printf("add3_var=%.6f\n", var(r, s));
    }
}
