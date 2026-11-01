// C++23 oracle for the physics tests that came in with e876dbf66 but whose
// modules never existed until stdlib/physics/{mechanics,phonon_quantity,
// pbpk_phonon_q}.sio were written:
//
//   tests/stdlib/physics/test_mechanics.sio
//   tests/stdlib/physics/test_phonon_quantity.sio
//   tests/stdlib/physics/test_phonon_quantity_parity.sio
//   tests/stdlib/physics/test_pbpk_phonon_q.sio
//   tests/stdlib/physics/test_phonon_quantity_gum.sio
//
// It recomputes every number those tests compare and the margins behind each
// inequality, with an independent implementation of the sinusoid lattice
// (upwind advection, axial dispersion with zero-flux outlet, tissue exchange,
// zonated metabolism; RK4; clamp at 0), in ng/mL (legacy) and kg/m^3 (SI).
// It also shows the two defects of stdlib/physics/phonon.sio fixed or
// documented alongside:
//   - its Bhaskara sine is not odd: -1.5 at -pi/2;
//   - its variance channel overflows (inf) within the 10-cycle parity run.
//
// Build: g++ -std=c++23 -O2 phonon_quantity.cpp && ./a.out
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

constexpr int N = 64;
constexpr double PI = 3.141592653589793;

struct P {
    double D, dx, krel, cl, fu, kp, cp, cha, f, ap, aha, phi, ct, v, z;
};
static P rapa_ngml() { return {1e-9, 3.9e-6, 0.1, 0.003, 0.075, 5.0, 10.0, 8.0, 1.2, 0.05, 0.30, 0.785, 2.0, 3e-4, 1.0}; }
static P scale_conc(P p, double s) { p.cp *= s; p.cha *= s; p.ct *= s; return p; }

static double bhaskara(double x) {
    double a = x;
    while (a > PI) a -= 2 * PI;
    while (a < -PI) a += 2 * PI;
    const double num = 16 * a * (PI - a), den = 5 * PI * PI - 4 * a * (PI - a);
    return den == 0 ? 0 : num / den;
}

template <bool kBhaskara>
static double c_in(double t, const P& p) {
    const double w = 2 * PI * p.f;
    auto s = [](double x) { return kBhaskara ? bhaskara(x) : std::sin(x); };
    return 0.75 * p.cp * (1 + p.ap * s(w * t)) + 0.25 * p.cha * (1 + p.aha * s(w * t + p.phi));
}

using V = std::array<double, N>;

template <bool kB>
static void rhs(double t, const V& y, const P& p, V& out) {
    const double adv = p.v > 0 ? p.v / p.dx : 0, dif = p.D / (p.dx * p.dx), cin = c_in<kB>(t, p);
    for (int i = 0; i < N; ++i) {
        const double ci = y[i], up = i == 0 ? cin : y[i - 1];
        double d = -adv * (ci - up);
        if (i == 0) d += dif * (cin - 2 * ci + y[1]);
        else if (i == N - 1) d += dif * (y[i - 1] - ci);
        else d += dif * (y[i - 1] - 2 * ci + y[i + 1]);
        double zf = 1.0;
        if (p.z > 1.01) zf = 2 / (p.z + 1) * (1 + (p.z - 1) * double(i) / (N - 1));
        d += -p.krel * (ci - p.ct / p.kp) - p.cl * zf * p.fu * ci;
        out[i] = d;
    }
}

template <bool kB>
static void rk4(double t, V& y, const P& p, double h) {
    V k1, k2, k3, k4, tmp;
    rhs<kB>(t, y, p, k1);
    for (int i = 0; i < N; ++i) tmp[i] = y[i] + 0.5 * h * k1[i];
    rhs<kB>(t + 0.5 * h, tmp, p, k2);
    for (int i = 0; i < N; ++i) tmp[i] = y[i] + 0.5 * h * k2[i];
    rhs<kB>(t + 0.5 * h, tmp, p, k3);
    for (int i = 0; i < N; ++i) tmp[i] = y[i] + h * k3[i];
    rhs<kB>(t + h, tmp, p, k4);
    for (int i = 0; i < N; ++i) {
        y[i] += h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        if (y[i] < 0) y[i] = 0;
    }
}

static double cfl(const P& p) {
    const double dd = p.D > 0 ? 0.4 * p.dx * p.dx / p.D : 1e-4, da = p.v > 0 ? 0.8 * p.dx / p.v : 1e-4;
    return std::min({dd, da, 1e-4});
}

template <bool kB>
static long simulate(V& y, const P& p, double t0, double t1, double dt) {
    double t = t0;
    long steps = 0;
    while (t < t1) {
        const double h = t + dt > t1 ? t1 - t : dt;
        rk4<kB>(t, y, p, h);
        t += h;
        ++steps;
    }
    return steps;
}

// One cardiac cycle from t = 0; returns {plain mean of samples, time average}.
template <bool kB>
static std::array<double, 2> liver_cycle(V& y, const P& p) {
    const double T = 1 / p.f, dt = cfl(p);
    double t = 0, sum = 0, tw = 0;
    long n = 0;
    while (t < T) {
        const double h = t + dt > T ? T - t : dt;
        rk4<kB>(t, y, p, h);
        sum += y[N - 1];
        tw += h * y[N - 1];
        ++n;
        t += h;
    }
    return {sum / n, tw / T};
}

static double max_abs_diff_si(const V& legacy_ngml, const V& si) {
    double m = 0;
    for (int i = 0; i < N; ++i) m = std::max(m, std::fabs(si[i] - legacy_ngml[i] * 1e-6));
    return m;
}

// The legacy variance channel of phonon.sio (dVar/dt = sum of a^2 Var).
static double legacy_variance_outlet(double t_end) {
    const P p = rapa_ngml();
    V y{}, var{};
    const double dt = 1e-4, adv = p.v / p.dx, dif = p.D / (p.dx * p.dx);
    double t = 0;
    // Forward Euler is enough to show the blow-up: the growth rate of the
    // channel is sum a^2 ~ 1e4 s^-1.
    while (t < t_end) {
        V dv;
        const double vin = std::pow(c_in<true>(t, p) * 0.1, 2);
        for (int i = 0; i < N; ++i) {
            const double vu = i == 0 ? vin : var[i - 1];
            double vd;
            if (i == 0) vd = dif * dif * (vin + 4 * var[i] + var[1]);
            else if (i == N - 1) vd = dif * dif * (var[i - 1] + var[i]);
            else vd = dif * dif * (var[i - 1] + 4 * var[i] + var[i + 1]);
            const double clz = p.cl * p.fu;
            dv[i] = adv * adv * (var[i] + vu) + vd + p.krel * p.krel * var[i] + clz * clz * var[i];
        }
        for (int i = 0; i < N; ++i) var[i] += dt * dv[i];
        t += dt;
        if (!std::isfinite(var[N - 1])) return var[N - 1];
    }
    (void)y;
    return var[N - 1];
}

int main() {
    // ---- mechanics ----
    const double G = 6.67430e-11, uG = 0.00015e-11;
    std::printf("KE(1 kg, 10 m/s)      %.17g J\n", 0.5 * 1 * 100.0);
    std::printf("F = k x (100, 0.5)    %.17g N\n", 100 * 0.5);
    std::printf("W = F d (10, 5)       %.17g J\n", 10.0 * 5);
    std::printf("P = W / t (100, 2)    %.17g W\n", 100.0 / 2);
    const double fg = G * 1.0 * 5.972e24 / (6.371e6 * 6.371e6);
    std::printf("F_grav earth surface  %.17g N  (|F - 9.82| = %.3g, test needs < 0.5)\n", fg, std::fabs(fg - 9.82));
    std::printf("  u from G alone      %.3g N\n", fg * uG / G);
    const double ke = 0.5 * 2 * 9, uke = std::hypot(0.5 * 9 * 0.1, 2 * 3 * 0.1);
    std::printf("KE(2+-0.1, 3+-0.1)    %.17g J  u = %.17g\n", ke, uke);
    std::printf("KE(2+-0.01, 4+-0.02)  %.17g J  u = %.17g\n", 0.5 * 2 * 16, std::hypot(0.5 * 16 * 0.01, 2 * 4 * 0.02));

    // ---- phonon_quantity ----
    const P si = scale_conc(rapa_ngml(), 1e-6);
    std::printf("c_portal %.17g c_ha %.17g c_tissue %.17g kg/m^3\n", si.cp, si.cha, si.ct);
    std::printf("cfl_dt  %.17g s (diffusion %.6g, advection %.6g)\n", cfl(si), 0.4 * si.dx * si.dx / si.D, 0.8 * si.dx / si.v);
    {
        // pulsatile_input_q(0): value and uncertainty with 10% u on c_p, c_ha
        const double gp = 0.75 * (1 + si.ap * std::sin(0.0)), gh = 0.25 * (1 + si.aha * std::sin(si.phi));
        std::printf("c_in(0) %.17g  u %.17g\n", gp * si.cp + gh * si.cha, std::hypot(gp * 1e-6, gh * 8e-7));
    }
    {
        // uniform steady state: v = k = cl = 0, a = 0, c_ha = c_portal
        P q = si;
        q.krel = 0; q.cl = 0; q.kp = 1; q.cha = q.cp; q.ap = 0; q.aha = 0; q.ct = q.cp; q.v = 0;
        V y; y.fill(1e-5);
        const long steps = simulate<false>(y, q, 0, 0.01, cfl(q));
        double dev = 0;
        for (double c : y) dev = std::max(dev, std::fabs(c - 1e-5));
        std::printf("steady: steps %ld  max |C - 1e-5| %.3g (test needs < 1e-9)\n", steps, dev);
    }

    // ---- parity (test_phonon_quantity_parity): legacy ng/mL x 1e-6 vs SI ----
    struct Sc { double cycles, c0, ct; };
    for (Sc sc : {Sc{10, 0, 2}, Sc{5, 5, 2}, Sc{20, 0, 0}}) {
        P lg = rapa_ngml();
        lg.ct = sc.ct;
        const P q = scale_conc(lg, 1e-6);
        const double t_end = sc.cycles / 1.2;
        V a; a.fill(sc.c0);
        V b; b.fill(sc.c0 * 1e-6);
        V c; c.fill(sc.c0);
        simulate<false>(a, lg, 0, t_end, 1e-4);
        simulate<false>(b, q, 0, t_end, cfl(q));
        simulate<true>(c, lg, 0, t_end, 1e-4);
        std::printf("parity %2.0f cycles c0=%g ct=%g: max|SI - legacy*1e-6| %.3g (tol 1e-9); outlet %.6g kg/m^3;"
                    " old Bhaskara legacy vs sine model %.3g\n",
                    sc.cycles, sc.c0, sc.ct, max_abs_diff_si(a, b), b[N - 1], max_abs_diff_si(c, b));
    }

    // ---- pbpk_phonon_q ----
    {
        V y{};
        const auto r = liver_cycle<false>(y, si);
        std::printf("liver cycle: time-avg outflow %.17g, sample mean %.17g, diff %.3g (tol 1e-9)\n", r[1], r[0], r[1] - r[0]);
        std::printf("  after one cycle: C[63] %.6g (> 0)  C[0] %.6g (> 0.5e-5)\n", y[N - 1], y[0]);
        V yl{};
        const auto rl = liver_cycle<false>(yl, rapa_ngml());
        std::printf("  legacy ng/mL sample mean x 1e-6 %.17g vs SI time-avg: diff %.3g\n", rl[0] * 1e-6, r[1] - rl[0] * 1e-6);
    }
    {
        P q = si;
        q.ct = 2e-6;  // tissue = c_liver
        V y{};
        const auto r = liver_cycle<false>(y, q);
        std::printf("correction: phonon %.6g - well-stirred %.6g = %.6g (0 < |x| < 1e-5)\n", r[1], 2e-6 / 5, r[1] - 2e-6 / 5);
    }
    {
        // test_phonon_quantity_gum: outlet/inlet sensitivities after one cycle.
        auto run = [](double cp, double c0) {
            P q = scale_conc(rapa_ngml(), 1e-6);
            q.cp = cp;
            V y; y.fill(c0);
            simulate<false>(y, q, 0, 1 / 1.2, cfl(q));
            return y;
        };
        const V a = run(1e-5 + 1e-7, 0), b2 = run(1e-5 - 1e-7, 0);
        const V c = run(1e-5, 5e-6 + 1e-7), d = run(1e-5, 5e-6 - 1e-7);
        std::printf("gum: dC63/dc_portal %.10g  dC0/dc_portal %.10g  dC63/dc0 %.10g\n",
                    (a[N - 1] - b2[N - 1]) / 2e-7, (a[0] - b2[0]) / 2e-7, (c[N - 1] - d[N - 1]) / 2e-7);
    }
    std::printf("Bhaskara sin(-pi/2) = %.6g (sin = -1)\n", bhaskara(-PI / 2));
    std::printf("legacy variance channel at outlet after 10 cycles: %g\n", legacy_variance_outlet(10 / 1.2));
    return 0;
}
