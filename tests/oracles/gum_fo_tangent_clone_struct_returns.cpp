// Oracle for tests/run-pass/gum_fo_tangent_clone_struct_returns.sio.
// First-order GUM (JCGM 100:2008 eq. 13, with covariances from shared inputs)
// with analytic derivatives. Independent inputs.
//
// Part A, a Lennard-Jones shell sum (sigma = 1, shells i = 1..3 at distance i r):
//   q_i = 1 / (i r)
//   E(eps, r) = sum_i 4 eps (q_i^12 - q_i^6)
//   F(eps, r) = -dE/dr = sum_i 4 eps (12 q_i^12 - 6 q_i^6) / r
// eps = 0.997 +- 0.02, r = 1.12 +- 0.01.
//
// Part B, a screened site potential on an atom { pos: {x, y, z}, q }:
//   s2 = x^2 + y^2 + z^2
//   phi = q sum_{i=1..3} 1 / (i + s2)
//   fx  = -dphi/dx = 2 x q sum_i 1 / (i + s2)^2
// x = 0.8 +- 0.01, y = 0.5 +- 0.02, z = 0.6 +- 0.015, q = 0.9 +- 0.03.
//
// Build: g++ -std=c++23 -O2 gum_fo_tangent_clone_struct_returns.cpp && ./a.out
#include <array>
#include <cmath>
#include <cstdio>

struct Row {
    double v;                 // value
    std::array<double, 4> d;  // partials w.r.t. the inputs of the part
};

static double var(const Row& a, const std::array<double, 4>& u) {
    double s = 0.0;
    for (int k = 0; k < 4; ++k) s += a.d[k] * a.d[k] * u[k] * u[k];
    return s;
}
static double cov(const Row& a, const Row& b, const std::array<double, 4>& u) {
    double s = 0.0;
    for (int k = 0; k < 4; ++k) s += a.d[k] * b.d[k] * u[k] * u[k];
    return s;
}
static Row lin(const Row& a, double ca, const Row& b, double cb) {
    Row r{};
    r.v = ca * a.v + cb * b.v;
    for (int k = 0; k < 4; ++k) r.d[k] = ca * a.d[k] + cb * b.d[k];
    return r;
}

// d[0] = d/d eps, d[1] = d/d r
static void lj(double eps, double r, int nshell, Row& E, Row& F) {
    E = Row{};
    F = Row{};
    for (int i = 1; i <= nshell; ++i) {
        const double q = 1.0 / (i * r);
        const double q6 = std::pow(q, 6), q12 = q6 * q6;
        E.v += 4.0 * eps * (q12 - q6);
        E.d[0] += 4.0 * (q12 - q6);
        E.d[1] += 4.0 * eps * (-12.0 * q12 + 6.0 * q6) / r;
        F.v += 4.0 * eps * (12.0 * q12 - 6.0 * q6) / r;
        F.d[0] += 4.0 * (12.0 * q12 - 6.0 * q6) / r;
        F.d[1] += 4.0 * eps * (-156.0 * q12 + 42.0 * q6) / (r * r);
    }
}

// d[0] = d/dx, d[1] = d/dy, d[2] = d/dz, d[3] = d/dq
static void site(double x, double y, double z, double q, Row& phi, Row& fx) {
    phi = Row{};
    fx = Row{};
    const double s2 = x * x + y * y + z * z;
    double a1 = 0.0, a2 = 0.0, a3 = 0.0;  // sum 1/(i+s2)^n
    for (int i = 1; i <= 3; ++i) {
        const double t = 1.0 / (i + s2);
        a1 += t;
        a2 += t * t;
        a3 += t * t * t;
    }
    phi.v = q * a1;
    const double dphi_ds2 = -q * a2;
    phi.d = {dphi_ds2 * 2.0 * x, dphi_ds2 * 2.0 * y, dphi_ds2 * 2.0 * z, a1};
    fx.v = 2.0 * x * q * a2;
    const double dfx_ds2 = -2.0 * x * q * 2.0 * a3;
    fx.d = {2.0 * q * a2 + dfx_ds2 * 2.0 * x, dfx_ds2 * 2.0 * y, dfx_ds2 * 2.0 * z, 2.0 * x * a2};
}

int main() {
    const std::array<double, 4> ulj = {0.02, 0.01, 0.0, 0.0};
    const std::array<double, 4> ust = {0.01, 0.02, 0.015, 0.03};
    Row E, F;
    lj(0.997, 1.12, 3, E, F);
    std::printf("lj_energy_var      %.17g\n", var(E, ulj));
    std::printf("lj_force_var       %.17g\n", var(F, ulj));
    const Row ef = lin(E, 1.0, F, -1.0);
    std::printf("lj_energy_minus_force_var %.17g\n", var(ef, ulj));
    std::printf("lj_energy_minus_same_call_var %.17g\n", 0.0);
    const Row eh = lin(E, 1.0, E, -0.5);
    std::printf("lj_energy_minus_half_var %.17g\n", var(eh, ulj));
    // early return after 2 shells
    Row E2, F2;
    lj(0.997, 1.12, 2, E2, F2);
    std::printf("lj_cut2_energy_var %.17g\n", var(E2, ulj));
    std::printf("lj_cut2_force_var  %.17g\n", var(F2, ulj));
    // a measurement made inside the call: m = 0.5 +- 0.05, E = eps m, F = m
    {
        const double eps = 0.997, ue = 0.02, m = 0.5, um = 0.05;
        std::printf("noisy_energy_var   %.17g\n", m * m * ue * ue + eps * eps * um * um);
        std::printf("noisy_force_var    %.17g\n", um * um);
    }
    // cross-term sanity: Cov(E, F)
    std::printf("lj_cov_ef          %.17g\n", cov(E, F, ulj));

    Row phi, fx;
    site(0.8, 0.5, 0.6, 0.9, phi, fx);
    std::printf("site_phi_var       %.17g\n", var(phi, ust));
    std::printf("site_fx_var        %.17g\n", var(fx, ust));
    const Row pf = lin(phi, 1.0, fx, 1.0);
    std::printf("site_phi_plus_fx_var %.17g\n", var(pf, ust));
    std::printf("site_phi_minus_same_var %.17g\n", 0.0);
}
