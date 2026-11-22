// C++23 oracle for tests/stdlib/epistemic/test_stiff_ode_knightian_mc.sio.
//
// Recomputes every number that test compares, with the same algorithms the
// stdlib uses, and an independent quadrature for the Monte Carlo target:
//
//   - vdp_final_y: forward Euler van der Pol, mu = 5, x0 = 0.5, y0 = 0,
//     dt = 0.02, 80 steps.
//   - epistemic::propagate::monte_carlo: the LCG (6364136223846793005,
//     1442695040888963407, two's-complement wrap), 12-uniform normal, seed
//     12345, n = 5000, population variance.
//   - E[f(mu)] for mu ~ N(5, 0.25) on the sampler's |z| <= 6 support, by
//     Simpson quadrature: the value the Monte Carlo mean estimates,
//     independent of the generator.
//   - The p-box propagation of the test: pb_from_knowledge gives lo == hi, so
//     every pb_add / pb_sub / pb_mul and the 7-point grid on x*x keep a point
//     band, and [k_lo, k_hi] is the deterministic trajectory f(mu0).
//   - Delta-method width 2 * |df/dmu| * sigma with the test's forward
//     difference h = 0.001.
//   - kl_smooth_ce (alpha = 2, lambda = 1) and kl_precise_ce (lambda = 1) on
//     cs_neighborhood(0.04 m, 0.0004, 0.25, 0.01 m, 0.08 m, 800), m = MC mean.
//
// Build: g++ -std=c++23 -O2 stiff_ode_knightian_mc.cpp && ./a.out
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

static double vdp_final_y(double mu, double x0, double y0, double dt, int n) {
    double x = x0, y = y0;
    for (int i = 0; i < n; ++i) {
        const double dx = y;
        const double dy = mu * (1.0 - x * x) * y - x;
        x = x + dt * dx;
        y = y + dt * dy;
    }
    return y;
}

// i64 arithmetic with wrap-around, as the Sounio LCG computes it.
static int64_t lcg_next(int64_t s) {
    return static_cast<int64_t>(static_cast<uint64_t>(s) * 6364136223846793005ULL +
                                1442695040888963407ULL);
}
static int64_t wrap_neg(int64_t s) { return static_cast<int64_t>(0ULL - static_cast<uint64_t>(s)); }
static double lcg_uniform(int64_t seed) {
    int64_t s = seed;
    if (s < 0) s = wrap_neg(s);
    const int64_t m = 1073741824;
    int64_t r = s;
    if (r < 0) r = wrap_neg(r);
    const int64_t bucket = r - (r / m) * m;
    return static_cast<double>(bucket) / static_cast<double>(m);
}

struct Credal { double mu0, eps, s_lo, s_hi; };
static Credal cs_neighborhood(double mean, double eps, double lo, double hi) {
    eps = std::clamp(eps, 0.0, 1.0);
    const double s_lo = std::min(lo, hi), s_hi = std::max(lo, hi);
    return {std::clamp(mean, s_lo, s_hi), eps, s_lo, s_hi};
}
static double kl_precise_ce(const Credal& c, double lam) {
    return (1.0 - c.eps) * c.mu0 + c.eps * (lam * c.s_lo + (1.0 - lam) * c.s_hi);
}
static double kl_smooth_ce(const Credal& c, double alpha, double lam) {
    if (alpha <= 0.0) return kl_precise_ce(c, lam);
    const double w0 = 1.0 - c.eps, wl = c.eps * lam, wh = c.eps * (1.0 - lam);
    const double m = std::min({c.mu0, c.s_lo, c.s_hi});
    const double sum = w0 * std::exp(-alpha * (c.mu0 - m)) + wl * std::exp(-alpha * (c.s_lo - m)) +
                       wh * std::exp(-alpha * (c.s_hi - m));
    return m - std::log(sum) / alpha;
}

int main() {
    const double mu0 = 5.0, mu_var = 0.25, x0 = 0.5, y0 = 0.0, dt = 0.02;
    const int n_steps = 80;
    const double sigma = std::sqrt(mu_var);

    // Monte Carlo exactly as epistemic::propagate::monte_carlo.
    int64_t seed = 12345;
    double sum = 0.0, sum_sq = 0.0;
    const int n = 5000;
    for (int i = 0; i < n; ++i) {
        double s12 = 0.0;
        int64_t s = seed;
        for (int j = 0; j < 12; ++j) {
            s = lcg_next(s);
            s12 += lcg_uniform(s);
        }
        seed = s;
        const double y = vdp_final_y(mu0 + (s12 - 6.0) * sigma, x0, y0, dt, n_steps);
        sum += y;
        sum_sq += y * y;
    }
    const double mc_mean = sum / n;
    const double mc_var = std::max(sum_sq / n - mc_mean * mc_mean, 0.0);

    // E[f(mu)] for mu = mu0 + z sigma, z ~ N(0, 1) restricted to |z| <= 6 (the
    // support of the 12-uniform sampler; the excluded mass is 2e-9), composite
    // Simpson with 24000 panels. Beyond |z| ~ 6 the Euler map grows so fast
    // that the untruncated Gaussian expectation is dominated by its far tail,
    // which no sampler with bounded draws visits.
    double e1 = 0.0, e2 = 0.0, mass = 0.0;
    const int panels = 24000;
    const double za = -6.0, zb = 6.0, hz = (zb - za) / panels;
    for (int i = 0; i <= panels; ++i) {
        const double z = za + i * hz;
        const double wt = (i == 0 || i == panels) ? 1.0 : (i % 2 ? 4.0 : 2.0);
        const double phi = std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI);
        const double y = vdp_final_y(mu0 + z * sigma, x0, y0, dt, n_steps);
        e1 += wt * phi * y;
        e2 += wt * phi * y * y;
        mass += wt * phi;
    }
    e1 /= mass;
    e2 /= mass;
    const double true_std = std::sqrt(e2 - e1 * e1);

    // Point p-box band and delta method.
    const double f0 = vdp_final_y(mu0, x0, y0, dt, n_steps);
    const double fp = vdp_final_y(mu0 + 0.001, x0, y0, dt, n_steps);
    const double df = (fp - f0) / 0.001;
    const double delta_w = 2.0 * std::sqrt(df * df * mu_var);
    const double k_lo = f0, k_hi = f0, k_width = 0.0;

    const Credal c = cs_neighborhood(mc_mean * 0.04, 0.25, mc_mean * 0.01, mc_mean * 0.08);
    const double ce_k = kl_smooth_ce(c, 2.0, 1.0);
    const double ce_p = kl_precise_ce(c, 1.0);

    std::printf("MC_MEAN          %.17g\n", mc_mean);
    std::printf("MC_STD           %.17g\n", std::sqrt(mc_var));
    std::printf("E_F_QUADRATURE   %.17g\n", e1);
    std::printf("STD_F_QUADRATURE %.17g\n", true_std);
    std::printf("MC_SE            %.17g\n", true_std / std::sqrt(static_cast<double>(n)));
    std::printf("F0 = K_LO = K_HI %.17g\n", f0);
    std::printf("K_WIDTH          %.17g\n", k_width);
    std::printf("DF_DMU           %.17g\n", df);
    std::printf("DELTA_W          %.17g\n", delta_w);
    std::printf("CE_ALPHA2        %.17g\n", ce_k);
    std::printf("CE_PRECISE       %.17g\n", ce_p);
    std::printf("CE_DIFF          %.17g\n", ce_k - ce_p);
    std::printf("assert1 (k_lo <= mc_mean <= k_hi): %s\n", (k_lo <= mc_mean && mc_mean <= k_hi) ? "holds" : "FALSE");
    std::printf("assert2 (k_width > 1.01 * delta_w): %s\n", (k_width > delta_w * 1.01) ? "holds" : "FALSE");
    std::printf("assert3 (ce_k < ce_p - 0.001): %s\n", (ce_k < ce_p - 0.001) ? "holds" : "FALSE");
    std::printf("strict ce_k < ce_p: %s\n", (ce_k < ce_p) ? "holds" : "FALSE");
    return 0;
}
