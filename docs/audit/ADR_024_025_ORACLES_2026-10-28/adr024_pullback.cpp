// Checks each rounding operator's pulled-back interval against the obligation 0 <= i < n
// (x0 = 0, dx = 1, n = 4) on a dense grid: an interval may never admit an out-of-range index.
#include <cmath>
#include <cstdio>
int main() {
    const int n = 4; int bad_floor = 0, bad_ceil = 0, bad_round = 0, miss_floor = 0, miss_ceil = 0;
    for (int k = -4000; k <= 8000; ++k) {
        const double q = k / 1000.0;
        auto in = [&](double i) { return i >= 0 && i < n; };
        const bool pf = q >= 0 && q < n, pc = q > -1 && q <= n - 1, pr = q > -0.5 && q < n - 0.5;
        if (pf && !in(std::floor(q))) ++bad_floor;
        if (pc && !in(std::ceil(q))) ++bad_ceil;
        if (pr && !in(std::round(q))) ++bad_round;
        if (pr && !in(std::nearbyint(q))) ++bad_round;   // half-to-even under the default mode
        if (!pf && in(std::floor(q))) ++miss_floor;      // equivalence for floor and ceil
        if (!pc && in(std::ceil(q))) ++miss_ceil;
    }
    std::printf("unsound: floor=%d ceil=%d round=%d   non-equivalent: floor=%d ceil=%d\n", bad_floor, bad_ceil, bad_round, miss_floor, miss_ceil);
    std::printf("x=-0.25: %g %g %g   x=3.75: %g %g %g   x=-0.5: %g %g round=%g nearbyint=%g\n",
                std::floor(-0.25), std::ceil(-0.25), std::round(-0.25), std::floor(3.75), std::ceil(3.75), std::round(3.75),
                std::floor(-0.5), std::ceil(-0.5), std::round(-0.5), std::nearbyint(-0.5));
}
