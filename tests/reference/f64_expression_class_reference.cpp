#include <cmath>
#include <iostream>

int main() {
    for (const bool enabled : {false, true}) {
        const double actual = enabled ? std::sqrt(4.0) : 0.0;
        if (actual / 16.0 != (enabled ? 0.125 : 0.0)) return 1;
        const double literal = enabled ? 2.0 : 4.0;
        if (literal / 16.0 != (enabled ? 0.125 : 0.25)) return 2;
        if (enabled && 16.0 / actual != 8.0) return 3;
        const auto integer = enabled ? 7 : 9;
        if (integer / 2 != (enabled ? 3 : 4)) return 4;
        if (static_cast<long long>(2.0) / 2 != 1) return 5;
        if (static_cast<double>(7) / 2.0 != 3.5) return 6;
    }
    std::cout << "CXX23_F64_REFERENCE_OK\n";
}
