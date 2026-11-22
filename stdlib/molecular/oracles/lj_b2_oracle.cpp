// stdlib/molecular/oracles/lj_b2_oracle.cpp
//
// Second virial coefficient of the Lennard-Jones potential truncated at rc
// with the mean-field tail energy c N^2 / V used by molecular::lj, in long
// double by composite Simpson quadrature:
//
//     B = -2 pi * integral_0^rc (exp(-u(r)/T) - 1) r^2 dr  +  c / T
//     c = (8/3) pi (rc^-9 / 3 - rc^-3)
//
// At this level the Widom mean of a dilute gas is m = exp(-2 B rho), which is
// the prediction examples/molecular/lj_widom_finite_size.sio tests (W1).
//
// Build and run:  g++ -std=c++23 -O2 lj_b2_oracle.cpp -o o && ./o
#include <cmath>
#include <cstdio>

int main() {
    const long double pi = std::acos(-1.0L);
    const long double temp = 2.0L, rc = 3.0L, rho = 0.001L;
    const auto integrand = [&](long double r) -> long double {
        if (r < 1e-12L) return 0.0L;
        const long double i6 = std::pow(1.0L / r, 6);
        return (std::exp(-4.0L * (i6 * i6 - i6) / temp) - 1.0L) * r * r;
    };
    const int n = 2000000;  // even
    const long double h = rc / n;
    long double s = 0.0L;
    for (int i = 0; i <= n; ++i)
        s += ((i == 0 || i == n) ? 1.0L : (i % 2 ? 4.0L : 2.0L)) * integrand(i * h);
    s *= h / 3.0L;
    const long double b_trunc = -2.0L * pi * s;
    const long double i3 = 1.0L / (rc * rc * rc), i9 = i3 * i3 * i3;
    const long double c = (8.0L / 3.0L) * pi * (i9 / 3.0L - i3);
    const long double b = b_trunc + c / temp;
    std::printf("B2 truncated   %.10Lf\n", b_trunc);
    std::printf("tail c / T     %.10Lf\n", c / temp);
    std::printf("B total        %.10Lf\n", b);
    std::printf("-ln m = 2 B rho at rho = %.4Lf :  %.8Lf\n", rho, 2.0L * b * rho);
    return 0;
}
