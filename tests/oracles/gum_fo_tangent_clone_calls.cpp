// Oracle for tests/run-pass/gum_fo_tangent_clone_calls.sio.
// First-order GUM (JCGM 100:2008 eq. 10/13) with analytic derivatives.
// Build: g++ -std=c++23 -O2 gum_fo_tangent_clone_calls.cpp && ./a.out
#include <cmath>
#include <cstdio>

int main() {
    const double e = 10.0, ue = 1.0;
    // kn_noloop / kn_loop: s = e + 2e, ds/de = 3
    std::printf("reassign %.17g\n", 9.0 * ue * ue);
    std::printf("loop %.17g\n", 9.0 * ue * ue);
    // kn_loop(k) - 3e: the call stays correlated with its own input
    std::printf("corr %.17g\n", 0.0);
    // pw(e, 4) = e^4 by recursion, d/de = 4 e^3
    const double d2 = 4.0 * e * e * e;
    std::printf("recur %.17g\n", d2 * d2 * ue * ue);
    // lj_shells: E = sum_{i=1..3} 4 eps ((s/(i r))^12 - (s/(i r))^6), s = 1
    const double r = 1.12, ur = 0.01, ep = 0.997, uep = 0.02;
    double dE_dr = 0.0, dE_de = 0.0;
    for (int i = 1; i <= 3; ++i) {
        const double q = 1.0 / (i * r);
        const double q6 = std::pow(q, 6), q12 = q6 * q6;
        dE_de += 4.0 * (q12 - q6);
        dE_dr += 4.0 * ep * (-12.0 * q12 + 6.0 * q6) / r;
    }
    std::printf("lj %.17g\n", dE_dr * dE_dr * ur * ur + dE_de * dE_de * uep * uep);
    // noisy(e) = e * m with m = 2 +- 0.5 measured inside the call
    std::printf("inner %.17g\n", 2.0 * 2.0 * ue * ue + e * e * 0.25);
    // sq(e) - sq(e)
    std::printf("twice %.17g\n", 0.0);
    // partition(beta, dE) = sum_{n=0}^{5} exp(-beta n dE), with an early return
    const double beta = 0.8, ub = 0.05, dd = 1.3, udd = 0.1;
    double dZ_db = 0.0, dZ_dd = 0.0;
    for (int n = 0; n <= 5; ++n) {
        const double t = std::exp(-beta * n * dd);
        dZ_db += -n * dd * t;
        dZ_dd += -n * beta * t;
        if (t < 1e-3) break;
    }
    std::printf("partition %.17g\n", dZ_db * dZ_db * ub * ub + dZ_dd * dZ_dd * udd * udd);
    // boltz_sum(eps) = sum_{i=1..4} exp(-eps i / 100), eps = 34.2 +- 1
    double db = 0.0;
    for (int i = 1; i <= 4; ++i) db -= (i / 100.0) * std::exp(-34.2 * i / 100.0);
    std::printf("boltz %.17g\n", db * db);

    // tests/run-pass/gum_fo_tangent_clone_second_order_on.sio
    // nsqrt(x): six Newton steps from y = x, differentiated step by step
    // (forward-mode duals), x = 20 +- 1.
    {
        const double x = 20.0;
        double y = x, dy = 1.0;
        for (int i = 0; i < 6; ++i) {
            const double ny = 0.5 * (y + x / y);
            const double dny = 0.5 * (dy + (y - x * dy) / (y * y));
            y = ny;
            dy = dny;
        }
        std::printf("newton_sqrt %.17g\n", dy * dy);
    }
    // mymax(x, 3) with x = 20: the first argument is returned, d/dx = 1.
    std::printf("early_return_max %.17g\n", 1.0);
    // hessian_of(sq(x), 0, 0) = d2(x^2)/dx2.
    std::printf("hessian_sq %.17g\n", 2.0);
    return 0;
}
