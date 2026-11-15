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

// Wide calls: more than 8 tangents, all measured by the caller.
template <class T> T nine(const T* a) {          // sum of products / exp / quotient
    T s = a[0];
    s = s * a[1];
    s = s + a[2] * a[3] * a[4];
    s = s + jexp(T::cst(0.05) * a[5] * a[6]);
    s = s + a[7] / (a[8] + T::cst(10.0));
    return s;
}
template <class T> T twelve(const T* a) {
    T s = a[0];
    s = s * a[1];
    s = s + a[2] * a[3];
    s = s + jexp(T::cst(0.1) * a[4] * a[5]);
    s = s + a[6] * a[7] * a[8];
    s = s + a[9] / (a[10] + T::cst(10.0));
    s = s + a[11] * a[11] * a[0];
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
    // nine tangents, all measured: values 1.5 .. 2.3 step 0.1
    {
        using J9 = Jet<9>;
        J9 a[9];
        for (int i = 0; i < 9; ++i) a[i] = J9::var(i, 1.5 + 0.1 * i);
        const J9 r = nine(a);
        std::printf("nine_h01 %.17g\n", r.h[0][1]);
        std::printf("nine_h24 %.17g\n", r.h[2][4]);
        std::printf("nine_h56 %.17g\n", r.h[5][6]);
        std::printf("nine_h55 %.17g\n", r.h[5][5]);
        std::printf("nine_h78 %.17g\n", r.h[7][8]);
        std::printf("nine_h88 %.17g\n", r.h[8][8]);
    }
    // twelve tangents, all measured: values 1.2 .. 2.3 step 0.1
    {
        using J12 = Jet<12>;
        J12 a[12];
        for (int i = 0; i < 12; ++i) a[i] = J12::var(i, 1.2 + 0.1 * i);
        const J12 r = twelve(a);
        std::printf("twelve_h01 %.17g\n", r.h[0][1]);
        std::printf("twelve_h45 %.17g\n", r.h[4][5]);
        std::printf("twelve_h44 %.17g\n", r.h[4][4]);
        std::printf("twelve_h678 %.17g\n", r.h[6][8]);
        std::printf("twelve_h910 %.17g\n", r.h[9][10]);
        std::printf("twelve_h1010 %.17g\n", r.h[10][10]);
        std::printf("twelve_h110 %.17g\n", r.h[11][0]);
        std::printf("twelve_h1111 %.17g\n", r.h[11][11]);
        std::printf("twelve_mean %.17g\n", r.v + 0.5 * (r.h[0][0] + r.h[1][1] + r.h[4][4] + r.h[5][5] + r.h[10][10] + r.h[11][11]) * 0.01);
    }
    // IO callee whose call appears only inside the query: y = x*x at x = 3, u = 0.1.
    {
        using J1 = Jet<1>;
        const J1 x = J1::var(0, 3.0);
        const J1 y = x * x;
        std::printf("io_query_hess %.17g\n", y.h[0][0]);
        std::printf("io_query_mean %.17g\n", y.v + 0.5 * y.h[0][0] * 0.1 * 0.1);
    }
    // A measurement made inside the callee: channels (a, m), a measured by the
    // caller (u_a = 0.1), m measured inside the callee (u_m = 0.3). The second-order
    // mean is f + 1/2 (H_aa u_a^2 + H_mm u_m^2).
    {
        using J2 = Jet<2>;
        const double ua = 0.1, um = 0.3;
        auto mean2 = [&](const J2& r) { return r.v + 0.5 * (r.h[0][0] * ua * ua + r.h[1][1] * um * um); };
        const J2 a = J2::var(0, 1.5);
        const J2 m = J2::var(1, 2.0);
        // drawn1(a) = exp(0.2 a) m^2
        const J2 r1 = jexp(J2::cst(0.2) * a) * m * m;
        std::printf("draw1_hess_aa %.17g\n", r1.h[0][0]);
        std::printf("draw1_mean %.17g\n", mean2(r1));
        const J2 r1sq = r1 * r1;
        std::printf("draw1_sq_mean %.17g\n", mean2(r1sq));
        // drawn2(a) = a + (m - 2)^2: stationary in m, so the internal variance is 0
        // and the whole internal contribution is the curvature term
        const J2 md = m - J2::cst(2.0);
        const J2 r2 = a + md * md;
        std::printf("draw2_hess_aa %.17g\n", r2.h[0][0]);
        std::printf("draw2_mean %.17g\n", mean2(r2));
        // outer_draw(a) = drawn1(a) * a: a callee that calls a drawing callee
        const J2 r3 = r1 * a;
        std::printf("draw3_hess_aa %.17g\n", r3.h[0][0]);
        std::printf("draw3_mean %.17g\n", mean2(r3));
    }
    // A `var` declared before the first measure() (#3547): a loop-carried update,
    // a nested loop and a branch all go through it. Channels (a, b), u_a = 0.1,
    // u_b = 0.2.
    {
        using J2 = Jet<2>;
        const double ua = 0.1, ub = 0.2;
        auto var_of = [&](const J2& r) { return r.g[0] * r.g[0] * ua * ua + r.g[1] * r.g[1] * ub * ub; };
        auto mean2 = [&](const J2& r) { return r.v + 0.5 * (r.h[0][0] * ua * ua + r.h[1][1] * ub * ub); };
        const J2 a = J2::var(0, 2.0), b = J2::var(1, 3.0);
        // s = 0; 3 x { s += a b }; s *= b
        J2 s = J2::cst(0.0);
        for (int i = 0; i < 3; ++i) s = s + a * b;
        s = s * b;
        std::printf("late_loop_var %.17g\n", var_of(s));
        std::printf("late_loop_h01 %.17g\n", s.h[0][1]);
        std::printf("late_loop_h11 %.17g\n", s.h[1][1]);
        std::printf("late_loop_mean %.17g\n", mean2(s));
        // 2 x 3 nested loops: s += a b, t *= a; r = s + t
        J2 s2 = J2::cst(0.0), t2 = J2::cst(1.0);
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 3; ++j) { s2 = s2 + a * b; t2 = t2 * a; }
        const J2 r2 = s2 + t2;
        std::printf("late_nested_var %.17g\n", var_of(r2));
        std::printf("late_nested_h01 %.17g\n", r2.h[0][1]);
        std::printf("late_nested_h00 %.17g\n", r2.h[0][0]);
        std::printf("late_nested_mean %.17g\n", mean2(r2));
        // a > 1: s = a b b (else a + b); then 2 x { s *= a }
        J2 s3 = a * b * b;
        for (int i = 0; i < 2; ++i) s3 = s3 * a;
        std::printf("late_branch_var %.17g\n", var_of(s3));
        std::printf("late_branch_h01 %.17g\n", s3.h[0][1]);
        std::printf("late_branch_mean %.17g\n", mean2(s3));
    }
    return 0;
}
