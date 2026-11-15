// stdlib/molecular/oracles/lj_config_oracle.cpp
//
// Independent reference for molecular::lj::mol_total_energy_virial: pair
// energy and pair virial of one fixed configuration, in long double, written
// without looking at the Sounio loop structure (minimum image by rounding
// rather than by branches, energy from r rather than from r^2 powers).
//
// Configuration: N = 400 points in a cubic box of side 10. Point i sits on a
// simple cubic grid of spacing 1.25 (site ix = i % 8, iy = (i / 8) % 8,
// iz = i / 64, centre (ix + 0.5) * 1.25) displaced along each axis by
// 0.3 (v - 0.5), v = s / 2^31 from the 31-bit LCG
// s <- (1103515245 s + 12345) mod 2^31 started at s = 20261113, one draw per
// coordinate in the order x0 y0 z0 x1 ... The jitter keeps every pair apart
// by at least 0.73, so the sum is not dominated by a few overlapping pairs
// and an error in the attractive region would show. Cutoff 3, no tail. tests/stdlib/molecular/test_lj_oracle.sio
// builds the same configuration and compares.
//
// Build and run:  g++ -std=c++23 -O2 lj_config_oracle.cpp -o o && ./o
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    const int n = 400;
    const long double box = 10.0L, rc = 3.0L;
    long long s = 20261113;
    std::vector<long double> p(3 * n);
    for (int i = 0; i < n; ++i) {
        const int site[3] = {i % 8, (i / 8) % 8, i / 64};
        for (int k = 0; k < 3; ++k) {
            s = (1103515245LL * s + 12345LL) % 2147483648LL;
            const long double v = static_cast<long double>(s) / 2147483648.0L;
            p[3 * i + k] = (site[k] + 0.5L) * 1.25L + 0.3L * (v - 0.5L);
        }
    }
    long double u = 0.0L, w = 0.0L;
    long long pairs = 0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            long double r2 = 0.0L;
            for (int k = 0; k < 3; ++k) {
                long double d = p[3 * i + k] - p[3 * j + k];
                d -= box * std::round(d / box);
                r2 += d * d;
            }
            const long double r = std::sqrt(r2);
            if (r < rc) {
                const long double sr6 = std::pow(1.0L / r, 6);
                u += 4.0L * (sr6 * sr6 - sr6);
                w += 24.0L * (2.0L * sr6 * sr6 - sr6);
                ++pairs;
            }
        }
    std::printf("pairs  %lld\n", pairs);
    std::printf("U      %.17Le\n", u);
    std::printf("W      %.17Le\n", w);
    return 0;
}
