// Independent NB2 moment accumulator. Not the Sounio implementation.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

struct Result {
    int status = 0;
    long double alpha = 0;
    long double pearson = 0;
    long double weight = 0;
    bool truncated = false;
};

Result estimate(const std::int64_t* counts, const long double* means, int n, int parameters) {
    Result out;
    if (n < 2 || n > 256 || parameters < 0 || parameters >= n) { out.status = 1; return out; }
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(static_cast<double>(means[i]))) { out.status = 2; return out; }
    }
    for (int i = 0; i < n; ++i) if (!(means[i] > 0)) { out.status = 3; return out; }
    for (int i = 0; i < n; ++i) if (counts[i] < 0) { out.status = 4; return out; }
    long double moment = 0, weight = 0, pearson = 0;
    for (int i = 0; i < n; ++i) {
        long double y = static_cast<long double>(counts[i]);
        long double mu = means[i];
        long double square = (y - mu) * (y - mu);
        moment += square - mu;
        weight += mu * mu;
        pearson += square / mu;
    }
    if (!(weight > 0) || !std::isfinite(static_cast<double>(moment))) { out.status = 5; return out; }
    long double alpha = (moment / weight) * (static_cast<long double>(n) / static_cast<long double>(n - parameters));
    out.alpha = alpha;
    out.pearson = pearson;
    out.weight = weight;
    out.truncated = alpha < 0;
    return out;
}

int main() {
    const std::int64_t over_y[] = {0, 10};
    const long double over_mu[] = {5.0L, 5.0L};
    Result over = estimate(over_y, over_mu, 2, 0);
    if (over.status != 0 || fabsl(over.alpha - 0.8L) > 1e-18L || over.truncated) return 1;

    const std::int64_t scaled_y[] = {0, 0, 10, 10};
    const long double scaled_mu[] = {5.0L, 5.0L, 5.0L, 5.0L};
    Result scaled = estimate(scaled_y, scaled_mu, 4, 1);
    long double expect = 0.8L * 4.0L / 3.0L;
    if (scaled.status != 0 || fabsl(scaled.alpha - expect) > 1e-18L) return 2;

    const std::int64_t under_y[] = {2, 2, 2, 2};
    const long double under_mu[] = {2.0L, 2.0L, 2.0L, 2.0L};
    Result under = estimate(under_y, under_mu, 4, 0);
    if (under.status != 0 || !(under.alpha < 0) || !under.truncated) return 3;

    const std::int64_t bad[] = {1, -1};
    const long double means[] = {1.0L, 1.0L};
    if (estimate(bad, means, 2, 0).status != 4) return 4;
    if (estimate(over_y, over_mu, 2, 2).status != 1) return 5;
    std::cout << "NB_DISPERSION_REFERENCE_PASS\n";
    std::cout.precision(17);
    std::cout << static_cast<double>(over.alpha) << "\n";
    std::cout << static_cast<double>(scaled.alpha) << "\n";
    std::cout << static_cast<double>(under.alpha) << "\n";
    return 0;
}
