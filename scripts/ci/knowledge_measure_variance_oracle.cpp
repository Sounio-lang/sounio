#include <cassert>
#include <iomanip>
#include <iostream>

int main() {
    // Independent C++23 binary64 reference for the Sounio regression margins.
    const double a = 0.1 * 0.1;
    const double b = 0.2 * 0.2;
    const double independent = a + b;
    const double same_slot = 4.0 * a;
    const double large = 0.8 * 0.8 + 0.8 * 0.8;
    assert(independent > 0.045 && independent < 0.06);
    assert(same_slot > 0.03 && same_slot < 0.05);
    assert(large > 1.0);
    // Exact rational cross-checks for those same acceptance/refusal margins.
    static_assert(5 * 1000 > 45 * 100 && 5 * 100 < 6 * 100);
    static_assert(4 * 100 > 3 * 100 && 4 * 100 < 5 * 100);
    static_assert(128 > 100);
    std::cout << std::setprecision(17)
              << "C++23 variance oracle: independent=" << independent
              << " same_slot=" << same_slot << " large=" << large << '\n';
}
