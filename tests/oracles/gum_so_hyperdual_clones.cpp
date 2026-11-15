// Oracle for the second-order checks of
// tests/run-pass/gum_fo_tangent_clone_second_order_on.sio.
//
// Second-order GUM needs the Hessian H = d2 f / d p2 of the measurand over
// the measured inputs p (JCGM 101 / the E2 mean f(mu) + 1/2 sum_k H_kk s_k^2).
// Here H is computed by forward-mode second-order automatic differentiation
// ("jets": value, gradient and Hessian carried through every operation of the
// same algorithm the Sounio program runs, loops and early returns included),
// not by closed forms, so it is independent of the compiler's chain rule.
// Closed forms are printed beside the jets where one exists.
//
// Build: g++ -std=c++23 -O2 gum_so_hyperdual_clones.cpp && ./a.out
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

// The Sounio programs, written once over the number type.
template <class T> T nsqrt(const T& x) {          // six Newton steps from y = x
    T y = x;
    for (int i = 0; i < 6; ++i) y = T::cst(0.5) * (y + x / y);
    return y;
}
template <class T> T boltz_sum(const T& e) {      // sum_{i=1..4} exp(-e i / 100)
    T s = T::cst(0.0);
    for (int i = 1; i <= 4; ++i) s = s + jexp(T::cst(-i / 100.0) * e);
    return s;
}
template <class T> T shell(const T& a, const T& b) {  // sum_{i=1..3} a exp(-b i / 10)
    T s = T::cst(0.0);
    for (int i = 1; i <= 3; ++i) s = s + a * jexp(T::cst(-i / 10.0) * b);
    return s;
}
template <class T> T pw(const T& x, int n) { return n == 0 ? T::cst(1.0) : x * pw(x, n - 1); }
template <class T> T mymax(const T& a, const T& b) { if (a.v > b.v) return a; return b; }
template <class T> T branchy(const T& x) {       // var reassigned under a branch
    T y = x;
    if (x.v > 10.0) y = y * y; else y = y * y * y;
    return y;
}
template <class T> T partition(const T& b, const T& d) {  // loop with an early return
    T z = T::cst(0.0);
    for (int n = 0; n <= 5; ++n) {
        const T t = jexp(T::cst(-double(n)) * b * d);
        z = z + t;
        if (t.v < 1e-3) return z;
    }
    return z;
}
template <class T> T outer(const T& x) {         // a loop calling a looping helper
    T s = T::cst(0.0);
    for (int j = 1; j <= 3; ++j) s = s + boltz_sum(T::cst(double(j)) * x);
    return s;
}

int main() {
    using J1 = Jet<1>;
    using J2 = Jet<2>;
    // nsqrt at x = 20, hessian_of(nsqrt(x), 0, 0) and hessian_of(s, 0, 0)
    {
        const J1 r = nsqrt(J1::var(0, 20.0));
        std::printf("newton_h %.17g\n", r.h[0][0]);
        // limit of the iteration: sqrt, d2/dx2 = -1/4 x^{-3/2}
        std::printf("newton_h_sqrt_closed_form %.17g\n", -0.25 * std::pow(20.0, -1.5));
    }
    // boltz_sum at e = 34.2 (+- 1): Hessian and E2 mean offset
    {
        const J1 r = boltz_sum(J1::var(0, 34.2));
        double h = 0.0;
        for (int i = 1; i <= 4; ++i) h += (i / 100.0) * (i / 100.0) * std::exp(-34.2 * i / 100.0);
        std::printf("boltz_h %.17g\n", r.h[0][0]);
        std::printf("boltz_h_closed_form %.17g\n", h);
        std::printf("boltz_mean %.17g\n", r.v + 0.5 * r.h[0][0] * 1.0);
        // boltz_sum(e) + boltz_sum(e): the same channel twice
        const J1 e = J1::var(0, 34.2);
        std::printf("boltz_twice_h %.17g\n", (boltz_sum(e) + boltz_sum(e)).h[0][0]);
    }
    // shell(a, b), a = 2.0 +- .1, b = 1.5 +- .2: mixed partial through a loop
    {
        const J2 r = shell(J2::var(0, 2.0), J2::var(1, 1.5));
        double hab = 0.0;
        for (int i = 1; i <= 3; ++i) hab += -(i / 10.0) * std::exp(-1.5 * i / 10.0);
        std::printf("shell_hab %.17g\n", r.h[0][1]);
        std::printf("shell_hab_closed_form %.17g\n", hab);
        std::printf("shell_haa %.17g\n", r.h[0][0]);
        std::printf("shell_hbb %.17g\n", r.h[1][1]);
    }
    // hessian_of(nsqrt(x) - x*x): the call stays correlated with its input
    {
        const J1 x = J1::var(0, 20.0);
        const J1 r = nsqrt(x) - x * x;
        std::printf("corr_h %.17g\n", r.h[0][0]);
    }
    // shell(x*y, y): arguments that are expressions of the caller's channels
    {
        const J2 x = J2::var(0, 1.2), y = J2::var(1, 0.8);
        const J2 r = shell(x * y, y);
        std::printf("compose_h00 %.17g\n", r.h[0][0]);
        std::printf("compose_h01 %.17g\n", r.h[0][1]);
        std::printf("compose_h11 %.17g\n", r.h[1][1]);
    }
    // recursion: pw(x, 5) = x^5, d2/dx2 = 20 x^3 = 160 at x = 2
    {
        const J1 r = pw(J1::var(0, 2.0), 5);
        std::printf("recur_h %.17g\n", r.h[0][0]);
        std::printf("recur_h_closed_form %.17g\n", 20.0 * 8.0);
    }
    // early return: mymax(x*x, 3) at x = 2 returns x*x, d2/dx2 = 2
    {
        const J1 x = J1::var(0, 2.0);
        const J1 r = mymax(x * x, J1::cst(3.0));
        std::printf("early_h %.17g\n", r.h[0][0]);
    }
    // var reassigned under a branch: x = 12 takes the x*x arm, x = 3 the x^3 arm
    {
        std::printf("branch_then_h %.17g\n", branchy(J1::var(0, 12.0)).h[0][0]);
        std::printf("branch_else_h %.17g\n", branchy(J1::var(0, 3.0)).h[0][0]);
    }
    // partition(beta, dE), early return, beta = 0.8, dE = 1.3
    {
        const J2 r = partition(J2::var(0, 0.8), J2::var(1, 1.3));
        std::printf("partition_hbb %.17g\n", r.h[0][0]);
        std::printf("partition_hbd %.17g\n", r.h[0][1]);
        std::printf("partition_hdd %.17g\n", r.h[1][1]);
    }
    // outer(x) = sum_{j=1..3} boltz_sum(j x) at x = 11.4: a helper inside a helper
    {
        const J1 r = outer(J1::var(0, 11.4));
        std::printf("nested_h %.17g\n", r.h[0][0]);
    }
    return 0;
}
