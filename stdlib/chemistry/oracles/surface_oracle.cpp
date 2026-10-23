// Surface-microkinetics oracle — elementary adsorption/reaction/desorption
// steps on a site-conserving surface, solved to steady state.
//
// An INDEPENDENT check on stdlib/chemistry/surface.sio. This program is not a
// translation of that Sounio code. It follows the convention of its sibling
// stdlib/chemistry/oracles/catalysis_oracle.cpp and of the C++ oracles in
// sounio-uhs-coupled: reimplement the same physics from the formulas, in a
// different language and codebase, and diff the printed output.
//
// THE INDEPENDENCE HERE IS METHODOLOGICAL, NOT MERELY LEXICAL, AND THAT IS
// THE POINT. surface.sio finds the steady state by MARCHING RK4 FORWARD IN
// TIME until the residual stops changing. This program never integrates
// anything. It solves the algebraic steady-state system
//
//     F(theta) = nu^T r(theta) = 0,   subject to   sum(theta) = 1
//
// directly, by damped Newton iteration with a numerically differenced
// Jacobian and Gaussian elimination with partial pivoting. Two methods with
// different error structure -- one has a time-discretisation error and no
// linear algebra, the other has linear-algebra conditioning and no time step
// -- reaching the same coverages is evidence about the PHYSICS. Two RK4
// implementations agreeing would mostly be evidence about RK4.
//
// The site-conservation row is what makes this well posed. nu^T r = 0 is
// rank-deficient by exactly one, because every step conserves sites, so one
// species equation is a linear combination of the others. Replacing the
// vacant-site row with sum(theta) - 1 = 0 restores rank. That is the same
// structural fact surface.sio's sites_conserved() tests, arriving here as a
// requirement of the solver rather than as an assertion -- which is a second,
// independent reason to believe it.
//
// STEP KINDS, and the exact rate expression each implements. `v` is the free
// site fraction, read from the state vector like any other coverage.
//
//   0  non-dissociative adsorption   A(g) + *  <-> A*
//        r = kf * p_A * v  -  kr * th_A
//   1  dissociative adsorption       A2(g) + 2* <-> 2A*
//        r = kf * p_A2 * v^2  -  kr * th_A^2
//   2  Langmuir-Hinshelwood          A* + B* <-> C* + *
//        r = kf * th_A * th_B  -  kr * th_C * v
//   3  desorption                    C* <-> C(g) + *
//        r = kf * th_C  -  kr * p_C * v
//   4  Eley-Rideal                   A(g) + B* <-> C*
//        r = kf * p_A * th_B  -  kr * th_C
//
// Build and run:
//     g++ -std=c++23 -O2 -o surface_oracle surface_oracle.cpp && ./surface_oracle
//
// Exit status is 0 if every self-check passes, 1 otherwise.

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string_view>
#include <vector>

namespace {

constexpr int MAXS = 16;   // species, including the free site
constexpr int MAXR = 16;   // elementary steps

struct Step {
    int kind{-1};
    double kf{0.0};
    double kr{0.0};
    int gas{-1};
    int a{-1};
    int b{-1};
    int c{-1};
};

// nu[r][s]; the free site occupies its own column, so every row sums to zero.
using Nu = std::vector<std::array<double, MAXS>>;

double step_rate(const Step& st, std::span<const double> th,
                 std::span<const double> p, int vac) {
    const double v = th[static_cast<std::size_t>(vac)];
    switch (st.kind) {
        case 0:
            return st.kf * p[static_cast<std::size_t>(st.gas)] * v
                 - st.kr * th[static_cast<std::size_t>(st.a)];
        case 1: {
            const double ta = th[static_cast<std::size_t>(st.a)];
            return st.kf * p[static_cast<std::size_t>(st.gas)] * v * v - st.kr * ta * ta;
        }
        case 2:
            return st.kf * th[static_cast<std::size_t>(st.a)] * th[static_cast<std::size_t>(st.b)]
                 - st.kr * th[static_cast<std::size_t>(st.c)] * v;
        case 3:
            return st.kf * th[static_cast<std::size_t>(st.c)]
                 - st.kr * p[static_cast<std::size_t>(st.gas)] * v;
        case 4:
            return st.kf * p[static_cast<std::size_t>(st.gas)] * th[static_cast<std::size_t>(st.b)]
                 - st.kr * th[static_cast<std::size_t>(st.c)];
        default:
            return 0.0;
    }
}

void rates_of(const std::vector<Step>& steps, std::span<const double> th,
              std::span<const double> p, int vac, std::span<double> out) {
    for (std::size_t r = 0; r < steps.size(); ++r) {
        out[r] = step_rate(steps[r], th, p, vac);
    }
}

// The residual whose root is the steady state. The vacant-site row is
// replaced by the site-conservation constraint, for the rank reason argued
// in the header.
void residual(const std::vector<Step>& steps, const Nu& nu, int nspecies, int vac,
              std::span<const double> th, std::span<const double> p,
              std::span<double> f) {
    std::array<double, MAXR> r{};
    rates_of(steps, th, p, vac, r);
    for (int s = 0; s < nspecies; ++s) {
        double acc = 0.0;
        for (std::size_t k = 0; k < steps.size(); ++k) {
            acc += nu[k][static_cast<std::size_t>(s)] * r[k];
        }
        f[static_cast<std::size_t>(s)] = acc;
    }
    double total = 0.0;
    for (int s = 0; s < nspecies; ++s) total += th[static_cast<std::size_t>(s)];
    f[static_cast<std::size_t>(vac)] = total - 1.0;
}

// Gaussian elimination with partial pivoting. Returns false on a singular
// pivot rather than producing a number.
bool solve_linear(std::vector<double>& A, std::vector<double>& b, int n) {
    for (int col = 0; col < n; ++col) {
        int piv = col;
        double best = std::fabs(A[static_cast<std::size_t>(col * n + col)]);
        for (int row = col + 1; row < n; ++row) {
            const double v = std::fabs(A[static_cast<std::size_t>(row * n + col)]);
            if (v > best) { best = v; piv = row; }
        }
        if (best < 1e-300) return false;
        if (piv != col) {
            for (int k = 0; k < n; ++k) {
                std::swap(A[static_cast<std::size_t>(col * n + k)],
                          A[static_cast<std::size_t>(piv * n + k)]);
            }
            std::swap(b[static_cast<std::size_t>(col)], b[static_cast<std::size_t>(piv)]);
        }
        const double d = A[static_cast<std::size_t>(col * n + col)];
        for (int row = col + 1; row < n; ++row) {
            const double factor = A[static_cast<std::size_t>(row * n + col)] / d;
            if (factor == 0.0) continue;
            for (int k = col; k < n; ++k) {
                A[static_cast<std::size_t>(row * n + k)] -=
                    factor * A[static_cast<std::size_t>(col * n + k)];
            }
            b[static_cast<std::size_t>(row)] -= factor * b[static_cast<std::size_t>(col)];
        }
    }
    for (int row = n - 1; row >= 0; --row) {
        double acc = b[static_cast<std::size_t>(row)];
        for (int k = row + 1; k < n; ++k) {
            acc -= A[static_cast<std::size_t>(row * n + k)] * b[static_cast<std::size_t>(k)];
        }
        b[static_cast<std::size_t>(row)] = acc / A[static_cast<std::size_t>(row * n + row)];
    }
    return true;
}

// Damped Newton with a numerically differenced Jacobian and a coverage
// clamp -- a Newton step that leaves the physical simplex is halved rather
// than accepted, which is what keeps this converging from a bare surface.
bool newton_steady_state(const std::vector<Step>& steps, const Nu& nu, int nspecies,
                         int vac, std::span<const double> p, std::span<double> th,
                         double& residual_out) {
    std::vector<double> f(static_cast<std::size_t>(nspecies));
    std::vector<double> f2(static_cast<std::size_t>(nspecies));
    std::vector<double> J(static_cast<std::size_t>(nspecies * nspecies));
    std::vector<double> dx(static_cast<std::size_t>(nspecies));
    std::vector<double> trial(static_cast<std::size_t>(nspecies));

    for (int iter = 0; iter < 500; ++iter) {
        residual(steps, nu, nspecies, vac, th, p, f);
        double worst = 0.0;
        for (int s = 0; s < nspecies; ++s) worst = std::fmax(worst, std::fabs(f[static_cast<std::size_t>(s)]));
        residual_out = worst;
        if (worst < 1e-14) return true;

        for (int col = 0; col < nspecies; ++col) {
            const double x0 = th[static_cast<std::size_t>(col)];
            const double h = 1e-7 * std::fmax(1.0, std::fabs(x0));
            th[static_cast<std::size_t>(col)] = x0 + h;
            residual(steps, nu, nspecies, vac, th, p, f2);
            th[static_cast<std::size_t>(col)] = x0;
            for (int row = 0; row < nspecies; ++row) {
                J[static_cast<std::size_t>(row * nspecies + col)] =
                    (f2[static_cast<std::size_t>(row)] - f[static_cast<std::size_t>(row)]) / h;
            }
        }
        for (int s = 0; s < nspecies; ++s) dx[static_cast<std::size_t>(s)] = -f[static_cast<std::size_t>(s)];
        if (!solve_linear(J, dx, nspecies)) return false;

        double lambda = 1.0;
        bool stepped = false;
        for (int back = 0; back < 40; ++back) {
            bool physical = true;
            for (int s = 0; s < nspecies; ++s) {
                trial[static_cast<std::size_t>(s)] =
                    th[static_cast<std::size_t>(s)] + lambda * dx[static_cast<std::size_t>(s)];
                if (trial[static_cast<std::size_t>(s)] < -1e-12 ||
                    trial[static_cast<std::size_t>(s)] > 1.0 + 1e-12) physical = false;
            }
            if (physical) {
                residual(steps, nu, nspecies, vac, trial, p, f2);
                double w2 = 0.0;
                for (int s = 0; s < nspecies; ++s) w2 = std::fmax(w2, std::fabs(f2[static_cast<std::size_t>(s)]));
                if (w2 < worst || back == 39) {
                    for (int s = 0; s < nspecies; ++s) th[static_cast<std::size_t>(s)] = trial[static_cast<std::size_t>(s)];
                    stepped = true;
                    break;
                }
            }
            lambda *= 0.5;
        }
        if (!stepped) return false;
    }
    return false;
}

bool near(double a, double b, double tol) { return std::fabs(a - b) < tol; }

int failures = 0;

void check(std::string_view name, double got, double want, double tol) {
    const bool ok = near(got, want, tol);
    if (!ok) ++failures;
    std::printf("%-46s %18.12f  expected %18.12f  %s\n",
                name.data(), got, want, ok ? "OK" : "FAIL");
}

// ---------------------------------------------------------------------------
// Mechanism builders, matching the ones in the Sounio side by construction
// only -- the numbers are the physics, not a copy of a data structure.
// ---------------------------------------------------------------------------

struct Mechanism {
    std::vector<Step> steps;
    Nu nu;
    int nspecies{0};
    int vac{0};
};

Mechanism single_adsorption(bool dissociative, double kf, double kr) {
    Mechanism m;
    m.nspecies = 2;
    m.vac = 1;
    Step s;
    s.kind = dissociative ? 1 : 0;
    s.kf = kf; s.kr = kr; s.gas = 0; s.a = 0;
    m.steps.push_back(s);
    std::array<double, MAXS> row{};
    row[0] = dissociative ? 2.0 : 1.0;
    row[1] = dissociative ? -2.0 : -1.0;
    m.nu.push_back(row);
    return m;
}

// A(g)+* <-> A*, B(g)+* <-> B*, A*+B* -> C*+*, C* -> C(g)+*
Mechanism lh_four_step() {
    Mechanism m;
    m.nspecies = 4;
    m.vac = 3;
    Step a; a.kind = 0; a.kf = 100.0; a.kr = 50.0; a.gas = 0; a.a = 0;
    Step b; b.kind = 0; b.kf = 100.0; b.kr = 25.0; b.gas = 1; b.a = 1;
    Step c; c.kind = 2; c.kf = 0.001; c.kr = 0.0; c.a = 0; c.b = 1; c.c = 2;
    Step d; d.kind = 3; d.kf = 1000.0; d.kr = 0.0; d.gas = 2; d.c = 2;
    m.steps = {a, b, c, d};
    std::array<double, MAXS> r0{}; r0[0] = 1.0;  r0[3] = -1.0;
    std::array<double, MAXS> r1{}; r1[1] = 1.0;  r1[3] = -1.0;
    std::array<double, MAXS> r2{}; r2[0] = -1.0; r2[1] = -1.0; r2[2] = 1.0; r2[3] = 1.0;
    std::array<double, MAXS> r3{}; r3[2] = -1.0; r3[3] = 1.0;
    m.nu = {r0, r1, r2, r3};
    return m;
}

// H2 (or H2 molecular) + CO2 on a mineral surface, lumped irreversible
// surface step returning both sites.
Mechanism h2_co2_surface(bool dissociative) {
    Mechanism m;
    m.nspecies = 3;
    m.vac = 2;
    Step h; h.kind = dissociative ? 1 : 0; h.kf = 1.0; h.kr = 1.0; h.gas = 0; h.a = 0;
    Step c; c.kind = 0; c.kf = 1.0; c.kr = 10.0; c.gas = 1; c.a = 1;
    Step s; s.kind = 2; s.kf = 1e-6; s.kr = 0.0; s.a = 0; s.b = 1; s.c = 2;
    m.steps = {h, c, s};
    std::array<double, MAXS> r0{};
    r0[0] = dissociative ? 2.0 : 1.0;
    r0[2] = dissociative ? -2.0 : -1.0;
    std::array<double, MAXS> r1{}; r1[1] = 1.0;  r1[2] = -1.0;
    std::array<double, MAXS> r2{}; r2[0] = -1.0; r2[1] = -1.0; r2[2] = 2.0;
    m.nu = {r0, r1, r2};
    return m;
}

bool rows_sum_to_zero(const Mechanism& m) {
    for (const auto& row : m.nu) {
        double acc = 0.0;
        for (int s = 0; s < m.nspecies; ++s) acc += row[static_cast<std::size_t>(s)];
        if (std::fabs(acc) > 1e-15) return false;
    }
    return true;
}

double surface_rate_at(const Mechanism& m, double p_h2, double p_co2, int step_index) {
    std::array<double, MAXS> th{};
    th[static_cast<std::size_t>(m.vac)] = 1.0;
    std::array<double, MAXS> p{};
    p[0] = p_h2;
    p[1] = p_co2;
    double res = 0.0;
    if (!newton_steady_state(m.steps, m.nu, m.nspecies, m.vac, p, th, res)) return -1.0;
    std::array<double, MAXR> r{};
    rates_of(m.steps, th, p, m.vac, r);
    return r[static_cast<std::size_t>(step_index)];
}

double apparent_order(bool dissociative, double p_h2, double p_co2) {
    const Mechanism m = h2_co2_surface(dissociative);
    const double lo = p_h2 / 1.02;
    const double hi = p_h2 * 1.02;
    const double r_lo = surface_rate_at(m, lo, p_co2, 2);
    const double r_hi = surface_rate_at(m, hi, p_co2, 2);
    if (r_lo <= 0.0 || r_hi <= 0.0) return std::nan("");
    return (std::log(r_hi) - std::log(r_lo)) / (std::log(hi) - std::log(lo));
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    std::printf("SURFACE MICROKINETICS ORACLE (C++23, Newton on the algebraic\n");
    std::printf("steady state -- no time integration anywhere in this program)\n\n");

    std::printf("-- Langmuir isotherms, solved not integrated --\n");
    {
        const Mechanism m = single_adsorption(false, 2.0, 0.5);
        std::array<double, MAXS> th{}; th[1] = 1.0;
        std::array<double, MAXS> p{};  p[0] = 0.25;
        double res = 0.0;
        const bool ok = newton_steady_state(m.steps, m.nu, m.nspecies, m.vac, p, th, res);
        if (!ok) { ++failures; std::printf("newton failed (non-dissociative)\n"); }
        // K = kf/kr = 4, Kp = 1, theta = 1/2
        check("langmuir non-dissociative theta_A", th[0], 0.5, 1e-12);
    }
    {
        const Mechanism m = single_adsorption(true, 2.0, 0.5);
        std::array<double, MAXS> th{}; th[1] = 1.0;
        std::array<double, MAXS> p{};  p[0] = 1.0;
        double res = 0.0;
        const bool ok = newton_steady_state(m.steps, m.nu, m.nspecies, m.vac, p, th, res);
        if (!ok) { ++failures; std::printf("newton failed (dissociative)\n"); }
        // K = 4, sqrt(Kp) = 2, theta = 2/3
        check("langmuir dissociative theta_H (H2+2* <-> 2H*)", th[0], 2.0 / 3.0, 1e-12);
    }

    std::printf("\n-- site conservation is what makes the Newton system full rank --\n");
    {
        const Mechanism m = lh_four_step();
        std::printf("%-46s %s\n", "every nu row sums to zero",
                    rows_sum_to_zero(m) ? "OK" : "FAIL");
        if (!rows_sum_to_zero(m)) ++failures;
    }

    std::printf("\n-- dual-site Langmuir-Hinshelwood, mechanism vs closed form --\n");
    {
        const Mechanism m = lh_four_step();
        std::array<double, MAXS> th{}; th[3] = 1.0;
        std::array<double, MAXS> p{};  p[0] = 0.5; p[1] = 0.25; p[2] = 0.0;
        double res = 0.0;
        const bool ok = newton_steady_state(m.steps, m.nu, m.nspecies, m.vac, p, th, res);
        if (!ok) { ++failures; std::printf("newton failed (LH four step)\n"); }
        std::array<double, MAXR> r{};
        rates_of(m.steps, th, p, m.vac, r);
        // KA pA = (100/50)*0.5 = 1, KB pB = (100/25)*0.25 = 1, denom = 3
        const double closed = 0.001 * 1.0 * 1.0 / (3.0 * 3.0);
        check("theta_A", th[0], 1.0 / 3.0, 1e-4);
        check("theta_B", th[1], 1.0 / 3.0, 1e-4);
        check("surface rate vs closed form k/9", r[2], closed, 1e-8);
    }

    std::printf("\n-- apparent reaction order in H2, the UHS channel (c) result --\n");
    std::printf("%-14s %20s %20s\n", "p_H2", "dissociative", "molecular");
    {
        const double p_co2 = 0.1;
        const double pressures[] = {1e-4, 1e-2, 1.0, 1e2, 1e4};
        const double want_d[] = {0.490196078431, 0.409909855, 0.002487562, -0.408264, -0.490001};
        const double want_m[] = {0.999802, 0.980392, 0.004975, -0.980002, -0.999798};
        for (int i = 0; i < 5; ++i) {
            const double od = apparent_order(true, pressures[i], p_co2);
            const double om = apparent_order(false, pressures[i], p_co2);
            std::printf("%-14.6g %20.9f %20.9f\n", pressures[i], od, om);
            if (!near(od, want_d[i], 2e-5)) { ++failures; std::printf("   FAIL dissociative order\n"); }
            if (!near(om, want_m[i], 2e-5)) { ++failures; std::printf("   FAIL molecular order\n"); }
        }
    }

    std::printf("\n");
    if (failures == 0) {
        std::printf("SURFACE_ORACLE ALL PASS\n");
        return 0;
    }
    std::printf("SURFACE_ORACLE: %d failures\n", failures);
    return 1;
}
