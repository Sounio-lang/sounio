// C++23 second-order jets (value, gradient, Hessian), as in gum_so_hyperdual_clones.cpp.
// Build: g++ -std=c++23 -O2 gum_so_struct_return_internal.cpp && ./a.out
#include <cmath>
#include <cstdio>

template <int N>
struct Jet {
    double v = 0.0;
    double g[N] = {};
    double h[N][N] = {};
    static Jet cst(double c) { Jet r; r.v = c; return r; }
    static Jet var(int i, double c) { Jet r; r.v = c; r.g[i] = 1.0; return r; }
};
template <int N> Jet<N> operator+(const Jet<N>& a, const Jet<N>& b) {
    Jet<N> r; r.v = a.v + b.v;
    for (int i = 0; i < N; ++i) { r.g[i] = a.g[i] + b.g[i]; for (int j = 0; j < N; ++j) r.h[i][j] = a.h[i][j] + b.h[i][j]; }
    return r;
}
template <int N> Jet<N> operator-(const Jet<N>& a, const Jet<N>& b) {
    Jet<N> r; r.v = a.v - b.v;
    for (int i = 0; i < N; ++i) { r.g[i] = a.g[i] - b.g[i]; for (int j = 0; j < N; ++j) r.h[i][j] = a.h[i][j] - b.h[i][j]; }
    return r;
}
template <int N> Jet<N> operator*(const Jet<N>& a, const Jet<N>& b) {
    Jet<N> r; r.v = a.v * b.v;
    for (int i = 0; i < N; ++i) {
        r.g[i] = a.g[i] * b.v + a.v * b.g[i];
        for (int j = 0; j < N; ++j)
            r.h[i][j] = a.h[i][j] * b.v + a.g[i] * b.g[j] + a.g[j] * b.g[i] + a.v * b.h[i][j];
    }
    return r;
}
template <int N> Jet<N> operator*(double c, const Jet<N>& b) { return Jet<N>::cst(c) * b; }
template <int N> Jet<N> operator/(const Jet<N>& a, const Jet<N>& b) {
    // a * (1/b); u = 1/b: u' = -b'/b^2, u'' = -b''/b^2 + 2 b' b'/b^3
    Jet<N> u; u.v = 1.0 / b.v;
    for (int i = 0; i < N; ++i) {
        u.g[i] = -b.g[i] / (b.v * b.v);
        for (int j = 0; j < N; ++j)
            u.h[i][j] = -b.h[i][j] / (b.v * b.v) + 2.0 * b.g[i] * b.g[j] / (b.v * b.v * b.v);
    }
    return a * u;
}
template <int N> Jet<N> jexp(const Jet<N>& a) {
    Jet<N> r; const double e = std::exp(a.v); r.v = e;
    for (int i = 0; i < N; ++i) {
        r.g[i] = e * a.g[i];
        for (int j = 0; j < N; ++j) r.h[i][j] = e * (a.h[i][j] + a.g[i] * a.g[j]);
    }
    return r;
}


// Oracle for tests/run-pass/gum_so_struct_return_internal.sio.
//
// A struct-returning callee with two measurements made inside it, correlated
// (rho = 0.6), feeding two output fields, called with an argument measured by
// the caller. Channels (a, m1, m2); u = (0.02, 0.05, 0.1).
//   p = eps * eps (a loop in the Sounio body); r.u = p m1 m2; r.w = m1 m1 + eps m2.
// Var = sum_ij J_i J_j Cov_ij, E2 = f + 1/2 sum_ij H_ij Cov_ij over all channels.
int main() {
    using J3 = Jet<3>;
    const double u[3] = {0.02, 0.05, 0.1};
    double cov[3][3] = {};
    for (int i = 0; i < 3; ++i) cov[i][i] = u[i] * u[i];
    cov[1][2] = cov[2][1] = 0.6 * u[1] * u[2];
    auto var_of = [&](const J3& r) { double v = 0; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) v += r.g[i] * r.g[j] * cov[i][j]; return v; };
    auto mean2 = [&](const J3& r) { double v = r.v; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) v += 0.5 * r.h[i][j] * cov[i][j]; return v; };
    const J3 a = J3::var(0, 1.2), m1 = J3::var(1, 0.5), m2 = J3::var(2, 1.5);
    const J3 p = a * a;
    const J3 ru = p * m1 * m2;
    const J3 rw = m1 * m1 + a * m2;
    std::printf("var_u %.17g\n", var_of(ru));
    std::printf("var_w %.17g\n", var_of(rw));
    std::printf("var_diff %.17g\n", var_of(ru - rw));
    std::printf("hess_u_00 %.17g\n", ru.h[0][0]);
    std::printf("hess_w_01 %.17g\n", rw.h[0][1]);
    std::printf("mean_u %.17g\n", mean2(ru));
    std::printf("mean_w %.17g\n", mean2(rw));
    std::printf("mean_uw %.17g\n", mean2(ru * rw));
    // the same with no correlation declared (what dropping the correlate reads)
    double c2[3][3] = {};
    for (int i = 0; i < 3; ++i) c2[i][i] = u[i] * u[i];
    double v0 = 0; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) v0 += (ru.g[i] - rw.g[i]) * (ru.g[j] - rw.g[j]) * c2[i][j];
    std::printf("var_diff_independent %.17g\n", v0);
    return 0;
}
