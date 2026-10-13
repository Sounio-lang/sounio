// Independent constant oracle for tan_reduction_compensated_2608.sio.
// Build: c++ -std=c++23 -O2 -ffp-contract=off tests/oracles/tan_reduction_2608.cpp -o /tmp/tan_oracle
// Reference evaluates the exact binary64 input using long-double libm.
// This checks written reference values; Sounio execution is a separate test.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>

struct Case { double input; double expected; };
static constexpr Case cases[] = {
    {-479424.3176972952, -9806663.258355444},
    {-479424.31769729516, -9801068.58367296},
    {-479424.3176972953, -9812264.32382312},
    {479424.3176972952, 9806663.258355444},
    {1.5707963267948966, 1.633123935319537e+16},
    {1.5707963267948963, 3530114321217157.5},
    {1.5707963267948968, -6218431163823738.0},
    {-1.5707963267948966, -1.633123935319537e+16},
    {3.141592653589793, -1.2246467991473532e-16},
    {3.1415926535897936, 3.216245299353273e-16},
    {1647097.7583689587, 291056669292.4059},
    {1647097.758368959, -4359295077.378354},
    {1647097.7583689585, 4232510408.045062},
    {1647099.3291652855, -6.420676210313675e-11},
    {-1647099.3291652855, 6.420676210313675e-11},
    {1000000.0, -0.373624453987599},
    {-1000000.0, 0.373624453987599},
    {0.5, 0.5463024898437905},
    {-0.5, -0.5463024898437905},
};

int main() {
    static_assert(std::numeric_limits<double>::is_iec559);
    static_assert(std::numeric_limits<double>::digits == 53);
    std::size_t failed = 0;
    for (const auto& c : cases) {
        const long double reference = std::tan(static_cast<long double>(c.input));
        const long double error = std::fabs(static_cast<long double>(c.expected) - reference);
        const long double scale = std::max(1.0L, std::fabs(reference));
        // Reference transcription is checked more tightly than the Sounio kernel.
        if (!std::isfinite(reference) || error > 2.0e-16L * scale) {
            ++failed;
            std::cerr << std::setprecision(21) << "FAIL input=" << c.input
                      << " reference=" << reference << " written=" << c.expected << '\n';
        }
    }
    std::cout << "TAN_REDUCTION_ORACLE cases=" << std::size(cases)
              << " failed=" << failed << '\n';
    return failed ? 1 : 0;
}
