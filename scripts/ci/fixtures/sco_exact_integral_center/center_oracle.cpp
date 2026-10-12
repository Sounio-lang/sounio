#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <utility>
#pragma STDC FENV_ACCESS ON
using I = std::int64_t;
constexpr I bound = I{1} << 54;
static_assert(std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53);
static_assert(static_cast<__int128>(3) * bound < std::numeric_limits<I>::max());
static_assert(static_cast<__int128>(2) * 2000000000LL * 2000000000LL <= std::numeric_limits<I>::max());
std::pair<bool,I> guarded_add(I a, I b) {
    if (b > 0 && a > std::numeric_limits<I>::max() - b) return {false,0};
    if (b < 0 && a < std::numeric_limits<I>::min() - b) return {false,0};
    return {true,a+b};
}
bool integral_literal(double x) {
    if (!std::isfinite(x) || x < -static_cast<double>(bound) || x > static_cast<double>(bound)) return false;
    const I i = static_cast<I>(x);
    return static_cast<double>(i) == x;
}
bool exact_float(I x) {
    // Only the source's bounded pair/three-term sums are admissible here.
    if (x < -3*bound || x > 3*bound) return false;
    return static_cast<I>(static_cast<double>(x)) == x;
}
int main() {
    if (std::fesetround(FE_TONEAREST) != 0) return 2;
    const std::array<std::pair<I,I>,7> additions{{
        {std::numeric_limits<I>::max(),1}, {std::numeric_limits<I>::min(),-1},
        {std::numeric_limits<I>::max(),0}, {std::numeric_limits<I>::min(),0},
        {1,-1}, {-10000000000000000LL,1}, {10000000000000000LL,-1}}};
    for (auto [a,b] : additions) {
        const __int128 reference = static_cast<__int128>(a) + b;
        const bool allowed = reference >= std::numeric_limits<I>::min() && reference <= std::numeric_limits<I>::max();
        const auto observed = guarded_add(a,b);
        if (observed.first != allowed || (allowed && observed.second != static_cast<I>(reference))) return 3;
    }
    if (!integral_literal(1.0) || !integral_literal(-10000000000000000.0)
        || !integral_literal(10000000000000000.0) || !integral_literal(static_cast<double>(bound))
        || integral_literal(1.5) || integral_literal(std::numeric_limits<double>::infinity())
        || integral_literal(std::numeric_limits<double>::quiet_NaN())
        || integral_literal(std::nextafter(static_cast<double>(bound), std::numeric_limits<double>::infinity()))) return 4;
    constexpr I ai=1, bi=-10000000000000000LL, ci=10000000000000000LL;
    const __int128 mathematical = static_cast<__int128>(ai) + bi + ci;
    const auto ab=guarded_add(ai,bi), bc=guarded_add(bi,ci);
    if (!ab.first || !bc.first) return 5;
    const auto result=guarded_add(ab.second,ci);
    if (!result.first || mathematical != result.second || !exact_float(bc.second) || !exact_float(result.second)) return 6;
    volatile double a=static_cast<double>(ai), b=static_cast<double>(bi), c=static_cast<double>(ci);
    volatile double old_inner=a+b;
    volatile double old_center=old_inner+c;
    volatile double new_inner=b+c;
    volatile double new_center=a+new_inner;
    if (new_center != static_cast<double>(result.second)) return 7;
    // Fixed one-nonzero uncertainty avoids covariance and quantized RSS ties
    // being misrepresented as a center-accuracy certificate.
    constexpr I ua=1000000000, ub=0, uc=0;
    const __int128 original_variance=static_cast<__int128>(ua)*ua+static_cast<__int128>(ub)*ub+static_cast<__int128>(uc)*uc;
    const __int128 alternative_variance=static_cast<__int128>(ub)*ub+static_cast<__int128>(uc)*uc+static_cast<__int128>(ua)*ua;
    if (original_variance != alternative_variance) return 8;
    std::puts("OVERFLOW_GUARDS_INT128_REFERENCE_PASS");
    std::puts("FINITE_INTEGRAL_DOMAIN_GUARDS_PASS");
    std::puts("ALTERNATIVE_PAIR_AND_FINAL_EXACT_REPRESENTABILITY_PASS");
    std::puts("FIXED_ONE_NONZERO_VARIANCE_UNCHANGED_PASS");
    std::printf("fixed_exact_center=%lld\nfixed_source_left=%.17g\nfixed_alternative_right=%.17g\n",
        static_cast<long long>(result.second),static_cast<double>(old_center),static_cast<double>(new_center));
    return 0;
}
