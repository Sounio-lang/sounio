// Oracle for tests/run-pass/gum_fo_tangent_clone_struct_returns2.sio.
// First-order GUM (JCGM 100:2008 eq. 13) and the second-order quantities of the
// same jets, with analytic derivatives (a finite-difference cross-check of every
// analytic derivative runs first and aborts on disagreement).
//
// Part A, a site built from two measured inputs a = 0.8 +- 0.01, b = 1.3 +- 0.02:
//   S(b) = sum_{i=1..3} 1 / (i + b^2)
//   s = a S
//   pos.x = a b,  pos.y = s,  pos.z = b - a,  e = s^2 + a
//   Site { pos: Vec3 { x, y, z }, e }
//   energy(site) = H (e pos.x + pos.z) + pos.y^2,  H = sum_{i=1..3} 1/i
// Part B, the LJ shell sum of the first witness (eps = 0.997 +- 0.02,
// r = 1.12 +- 0.01): E = sum 4 eps (q^12 - q^6), F = -dE/dr, q = 1/(i r).
// Part C, a measurement inside the call feeding two output fields:
//   m = 0.5 +- 0.05 drawn inside; u = eps * m, w = m (eps = 0.997 +- 0.02).
// Build: g++ -std=c++23 -O2 gum_fo_tangent_clone_struct_returns2.cpp && ./a.out
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>

using F2 = std::function<double(double, double)>;
struct J2 {
    double v, d0, d1, h00, h01, h11;
};

static J2 fd(const F2& f, double x, double y) {
    const double h = 1e-4;
    auto d4 = [&](auto g) {  // 4th-order central difference in a lambda of t
        return (-g(2 * h) + 8 * g(h) - 8 * g(-h) + g(-2 * h)) / (12 * h);
    };
    J2 r{};
    r.v = f(x, y);
    r.d0 = d4([&](double t) { return f(x + t, y); });
    r.d1 = d4([&](double t) { return f(x, y + t); });
    auto dd = [&](auto g) { return (-g(2 * h) + 16 * g(h) - 30 * g(0) + 16 * g(-h) - g(-2 * h)) / (12 * h * h); };
    r.h00 = dd([&](double t) { return f(x + t, y); });
    r.h11 = dd([&](double t) { return f(x, y + t); });
    auto mix = [&](double s, double t) { return f(x + s, y + t); };
    r.h01 = (mix(h, h) - mix(h, -h) - mix(-h, h) + mix(-h, -h)) / (4 * h * h);
    return r;
}
static void check(const char* n, const J2& an, const F2& f, double x, double y) {
    const J2 n2 = fd(f, x, y);
    auto bad = [](double a, double b) { return std::fabs(a - b) > 2e-6 * (1.0 + std::fabs(a)); };
    if (bad(an.d0, n2.d0) || bad(an.d1, n2.d1) || bad(an.h00, n2.h00) || bad(an.h01, n2.h01) || bad(an.h11, n2.h11)) {
        std::fprintf(stderr, "oracle self-check FAILED for %s\n", n);
        std::exit(1);
    }
}
static double var2(const J2& j, double u0, double u1) { return j.d0 * j.d0 * u0 * u0 + j.d1 * j.d1 * u1 * u1; }
static J2 lin(const J2& a, double ca, const J2& b, double cb) {
    return J2{ca * a.v + cb * b.v, ca * a.d0 + cb * b.d0, ca * a.d1 + cb * b.d1, ca * a.h00 + cb * b.h00, ca * a.h01 + cb * b.h01, ca * a.h11 + cb * b.h11};
}
static J2 mul(const J2& a, const J2& b) {
    return J2{a.v * b.v, a.d0 * b.v + a.v * b.d0, a.d1 * b.v + a.v * b.d1,
              a.h00 * b.v + 2 * a.d0 * b.d0 + a.v * b.h00, a.h01 * b.v + a.d0 * b.d1 + a.d1 * b.d0 + a.v * b.h01,
              a.h11 * b.v + 2 * a.d1 * b.d1 + a.v * b.h11};
}

struct Site { J2 x, y, z, e; };
static Site site(double a, double b) {
    double S = 0, T = 0, U = 0;
    for (int i = 1; i <= 3; ++i) { const double t = 1.0 / (i + b * b); S += t; T += t * t; U += t * t * t; }
    const double Sb = -2 * b * T, Sbb = -2 * T + 8 * b * b * U;
    J2 s{a * S, S, a * Sb, 0.0, Sb, a * Sbb};
    Site r;
    r.x = J2{a * b, b, a, 0, 1, 0};
    r.y = s;
    r.z = J2{b - a, -1, 1, 0, 0, 0};
    r.e = lin(mul(s, s), 1.0, J2{a, 1, 0, 0, 0, 0}, 1.0);
    return r;
}
static J2 energy(const Site& s) {
    const double H = 1.0 + 0.5 + 1.0 / 3.0;
    return lin(lin(mul(s.e, s.x), H, s.z, H), 1.0, mul(s.y, s.y), 1.0);
}
static double site_value(double a, double b, int which) {
    const Site s = site(a, b);
    if (which == 0) return s.x.v;
    if (which == 1) return s.y.v;
    if (which == 2) return s.z.v;
    if (which == 3) return s.e.v;
    return energy(s).v;
}

// LJ shell sum over (eps, r)
static void lj(double eps, double r, J2& E, J2& Fo) {
    E = J2{}; Fo = J2{};
    for (int i = 1; i <= 3; ++i) {
        const double q = 1.0 / (i * r);
        const double q6 = std::pow(q, 6), q12 = q6 * q6;
        E.v += 4 * eps * (q12 - q6);
        E.d0 += 4 * (q12 - q6);
        E.d1 += 4 * eps * (-12 * q12 + 6 * q6) / r;
        E.h01 += 4 * (-12 * q12 + 6 * q6) / r;
        E.h11 += 4 * eps * (156 * q12 - 42 * q6) / (r * r);
        Fo.v += 4 * eps * (12 * q12 - 6 * q6) / r;
        Fo.d0 += 4 * (12 * q12 - 6 * q6) / r;
        Fo.d1 += 4 * eps * (-156 * q12 + 42 * q6) / (r * r);
        Fo.h01 += 4 * (-156 * q12 + 42 * q6) / (r * r);
        Fo.h11 += 4 * eps * (2184 * q12 - 336 * q6) / (r * r * r);
    }
}
static double lj_value(double eps, double r, int which) {
    J2 E, Fo; lj(eps, r, E, Fo);
    return which == 0 ? E.v : Fo.v;
}

int main() {
    const double a = 0.8, ua = 0.01, b = 1.3, ub = 0.02;
    const Site s = site(a, b);
    for (int w = 0; w < 5; ++w) {
        const J2 j = w == 0 ? s.x : w == 1 ? s.y : w == 2 ? s.z : w == 3 ? s.e : energy(s);
        check("site", j, [&](double p, double q) { return site_value(p, q, w); }, a, b);
    }
    const double eps = 0.997, ue = 0.02, r = 1.12, ur = 0.01;
    J2 E, Fo; lj(eps, r, E, Fo);
    check("lj_E", E, [&](double p, double q) { return lj_value(p, q, 0); }, eps, r);
    check("lj_F", Fo, [&](double p, double q) { return lj_value(p, q, 1); }, eps, r);

    // Part A, first order
    std::printf("nested_x_var       %.17g\n", var2(s.x, ua, ub));
    std::printf("nested_y_var       %.17g\n", var2(s.y, ua, ub));
    std::printf("nested_z_var       %.17g\n", var2(s.z, ua, ub));
    std::printf("nested_e_var       %.17g\n", var2(s.e, ua, ub));
    std::printf("nested_x_minus_y_var %.17g\n", var2(lin(s.x, 1.0, s.y, -1.0), ua, ub));
    // e - y^2 = a
    std::printf("nested_e_minus_y2_var %.17g\n", ua * ua);
    // direct (no let): same values
    std::printf("direct_e_var       %.17g\n", var2(s.e, ua, ub));
    std::printf("direct_x_var       %.17g\n", var2(s.x, ua, ub));
    std::printf("direct_y_plus_z_var %.17g\n", var2(lin(s.y, 1.0, s.z, 1.0), ua, ub));
    std::printf("direct_e_minus_let_e_var %.17g\n", 0.0);
    // g(f(x))
    const J2 en = energy(s);
    std::printf("composed_var       %.17g\n", var2(en, ua, ub));
    // Part A, second order (channel 0 = a, channel 1 = b)
    std::printf("so_e_00            %.17g\n", s.e.h00);
    std::printf("so_e_01            %.17g\n", s.e.h01);
    std::printf("so_e_11            %.17g\n", s.e.h11);
    std::printf("so_y_11            %.17g\n", s.y.h11);
    std::printf("so_x_01            %.17g\n", s.x.h01);
    std::printf("so_z_01            %.17g\n", s.z.h01);
    std::printf("so_composed_01     %.17g\n", en.h01);
    std::printf("so_composed_11     %.17g\n", en.h11);
    std::printf("so_e_mean          %.17g\n", s.e.v + 0.5 * (s.e.h00 * ua * ua + s.e.h11 * ub * ub));
    std::printf("so_direct_e_01     %.17g\n", s.e.h01);

    // Part B (channel 0 = eps, 1 = r)
    std::printf("lj_energy_var      %.17g\n", var2(E, ue, ur));
    std::printf("lj_force_var       %.17g\n", var2(Fo, ue, ur));
    std::printf("so_lj_energy_01    %.17g\n", E.h01);
    std::printf("so_lj_energy_11    %.17g\n", E.h11);
    std::printf("so_lj_force_01     %.17g\n", Fo.h01);
    std::printf("so_lj_force_11     %.17g\n", Fo.h11);

    // Part C
    const double m = 0.5, um = 0.05;
    std::printf("shared_u_var       %.17g\n", m * m * ue * ue + eps * eps * um * um);
    std::printf("shared_w_var       %.17g\n", um * um);
    // u = eps m, w = m share m: Var(u - w) = m^2 ue^2 + (eps - 1)^2 um^2
    std::printf("shared_u_minus_w_var %.17g\n", m * m * ue * ue + (eps - 1.0) * (eps - 1.0) * um * um);
    std::printf("shared_u_plus_w_var %.17g\n", m * m * ue * ue + (eps + 1.0) * (eps + 1.0) * um * um);
    return 0;
}
