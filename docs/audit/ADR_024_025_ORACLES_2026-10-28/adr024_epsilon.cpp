#include <algorithm>
#include <cstdio>
int main() {
    const double mu = 3.2, u = 0.1, a = 0.0, b = 16 * 0.5;
    const double da = mu - a, db = b - mu, u2 = u * u;
    const double cantelli = u2 / (u2 + da * da) + u2 / (u2 + db * db);
    const double cheb = u2 / (std::min(da, db) * std::min(da, db));
    std::printf("cantelli=%.6e chebyshev=%.6e eps=%.6e\n", cantelli, cheb, std::min(cantelli, cheb));
}
