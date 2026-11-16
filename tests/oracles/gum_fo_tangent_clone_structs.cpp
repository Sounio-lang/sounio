// Oracle for tests/run-pass/gum_fo_tangent_clone_structs.sio.
// First-order GUM (JCGM 100:2008 eq. 10) with analytic derivatives.
// E(r) = sum_{i=1..3} 4 eps ((s/(i r))^12 - (s/(i r))^6), s = 1,
// eps = 0.997 +- 0.02 (a struct field), r = 1.12 +- 0.01.
// Build: g++ -std=c++23 -O2 gum_fo_tangent_clone_structs.cpp && ./a.out
#include <cmath>
#include <cstdio>

static void dE(double r, double ep, double& de, double& dr) {
    de = 0.0;
    dr = 0.0;
    for (int i = 1; i <= 3; ++i) {
        const double q = 1.0 / (i * r);
        const double q6 = std::pow(q, 6), q12 = q6 * q6;
        de += 4.0 * (q12 - q6);
        dr += 4.0 * ep * (-12.0 * q12 + 6.0 * q6) / r;
    }
}

int main() {
    const double r = 1.12, ur = 0.01, ep = 0.997, uep = 0.02;
    double de1, dr1, de2, dr2;
    dE(r, ep, de1, dr1);
    dE(2.0 * r, ep, de2, dr2);
    const double single = de1 * de1 * uep * uep + dr1 * dr1 * ur * ur;
    // struct by value, struct by reference, self by value, self by reference
    std::printf("single %.17g\n", single);
    // shells(p, r) - p.energy(r): the same function of the same inputs
    std::printf("corr %.17g\n", 0.0);
    // total(r) = energy(r) + energy(2 r): d/dr picks up the factor 2
    const double de = de1 + de2, dr = dr1 + 2.0 * dr2;
    std::printf("nested %.17g\n", de * de * uep * uep + dr * dr * ur * ur);
    return 0;
}
