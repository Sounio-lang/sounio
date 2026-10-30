// C++23 oracle for tests/stdlib/epistemic/test_active_belief_update.sio.
//
// stdlib/epistemic/active.sio updates a belief by precision weighting and its
// integer confidence (0..1000) by a Beta posterior: the prior Beta has mean
// c / 1000 and concentration n0 (2 = Beta(1, 1) by default), successes =
// trunc(10 * obs_precision / prior_precision), failures = 1, and the result
// is round-half-up(1000 * alpha / (alpha + beta)). This file recomputes every
// value the test compares, plus the expected free energy terms and the
// Klibanoff bridge's pragmatic values.
//
// Build: g++ -std=c++23 -O2 active_belief_update.cpp && ./a.out
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

static const double kFloor = 1e-7;

static int64_t clamp_conf(int64_t c) { return std::clamp<int64_t>(c, 0, 1000); }

static int64_t beta_posterior_confidence(int64_t c, double n0, double s, double f) {
    if (n0 <= 0.0 || s < 0.0 || f < 0.0) return -1;
    const double p = static_cast<double>(clamp_conf(c)) / 1000.0;
    const double a = n0 * p + s, b = n0 * (1.0 - p) + f;
    if (a + b <= 0.0) return -1;
    return clamp_conf(static_cast<int64_t>(1000.0 * a / (a + b) + 0.5));
}

struct Ep { double val, var; int64_t conf; };

static Ep update(const Ep& prior, double obs, double obs_var, double n0) {
    const double pp = 1.0 / std::max(prior.var, kFloor);
    const double op = 1.0 / std::max(obs_var, kFloor);
    const double tp = pp + op;
    const double s = std::trunc(op / pp * 10.0);
    int64_t c = beta_posterior_confidence(prior.conf, n0, s, 1.0);
    if (c < 0) c = 0;
    return {(pp * prior.val + op * obs) / tp, 1.0 / tp, c};
}

static double efe_total(const Ep& cur, double post_var, double reward, double w) {
    const double ig = 0.5 * std::log(std::max(cur.var, kFloor) / std::max(post_var, kFloor));
    return -std::max(ig, 0.0) - reward * w;
}

static double kl_smooth_ce(double mu0, double eps, double lo, double hi, double alpha, double lam) {
    if (alpha <= 0.0) return (1.0 - eps) * mu0 + eps * (lam * lo + (1.0 - lam) * hi);
    const double m = std::min({mu0, lo, hi});
    const double sum = (1.0 - eps) * std::exp(-alpha * (mu0 - m)) + eps * lam * std::exp(-alpha * (lo - m)) +
                       eps * (1.0 - lam) * std::exp(-alpha * (hi - m));
    return m - std::log(sum) / alpha;
}

int main() {
    const Ep a = update({10.0, 4.0, 500}, 12.0, 1.0, 2.0);
    std::printf("A  val %.17g var %.17g conf %lld\n", a.val, a.var, static_cast<long long>(a.conf));
    const Ep b = update({1.0, 1.0, 900}, 1.0, 4.0, 2.0);
    std::printf("B  val %.17g var %.17g conf %lld\n", b.val, b.var, static_cast<long long>(b.conf));
    const Ep c = update({1.0, 1.0, 900}, 1.0, 4.0, 20.0);
    std::printf("C  conf (n0 = 20) %lld\n", static_cast<long long>(c.conf));
    std::printf("D  bpc(1000, 2, 0, 1) %lld  bpc(0, 2, 0, 1) %lld  bpc(500, 0, 1, 1) %lld\n",
                static_cast<long long>(beta_posterior_confidence(1000, 2.0, 0.0, 1.0)),
                static_cast<long long>(beta_posterior_confidence(0, 2.0, 0.0, 1.0)),
                static_cast<long long>(beta_posterior_confidence(500, 0.0, 1.0, 1.0)));
    const Ep s1 = update({0.0, 1.0, 500}, 1.0, 1.0, 2.0);
    const Ep s2 = update(s1, 2.0, 0.5, 2.0);
    std::printf("SEQ val %.17g var %.17g conf %lld (after 1: conf %lld)\n", s2.val, s2.var,
                static_cast<long long>(s2.conf), static_cast<long long>(s1.conf));

    const Ep cur{0.0, 1.0, 500};
    const double vr[3] = {0.5, 0.1, 0.9}, rw[3] = {0.0, 0.0, 1.0};
    for (double w : {1.0, 2.0}) {
        int best = 0;
        double best_efe = 0.0;
        for (int i = 0; i < 3; ++i) {
            const double e = efe_total(cur, cur.var * vr[i], rw[i], w);
            std::printf("EFE w=%g action %d total %.17g\n", w, i, e);
            if (i == 0 || e < best_efe) { best_efe = e; best = i; }
        }
        std::printf("SELECT w=%g -> %d\n", w, best);
    }

    std::printf("ENTROPY(var=1) %.17g\n", 0.5 * std::log(2.0 * M_PI * M_E));
    std::printf("SCORE(0.65,0.01,500) %.17g SCORE(0.60,0.0004,500) %.17g\n",
                std::log(1.01) + 0.5 + 1.0, std::log(1.0004) + 0.5 + 1.0);

    // Bridge: cs_neighborhood(10, 1, 0.2, 5, 15, 800), lambda = 1, weight 1,
    // current variance 1, expected posterior variance 0.25.
    const double walley = 5.0;  // min{x_i : w_i > 0} at lambda = 1, eps > 0
    const double smooth = kl_smooth_ce(10.0, 0.2, 5.0, 15.0, 1.0, 1.0);
    const double precise = kl_smooth_ce(10.0, 0.2, 5.0, 15.0, 0.0, 1.0);
    std::printf("BRIDGE epistemic %.17g\n", 0.5 * std::log(4.0));
    std::printf("BRIDGE pragmatic walley %.17g smooth(alpha=1) %.17g precise %.17g\n", walley, smooth, precise);
    std::printf("BRIDGE total smooth %.17g precise %.17g\n", -0.5 * std::log(4.0) - smooth,
                -0.5 * std::log(4.0) - precise);
    return 0;
}
