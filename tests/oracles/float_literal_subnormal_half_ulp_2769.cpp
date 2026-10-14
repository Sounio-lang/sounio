#include <boost/multiprecision/cpp_int.hpp>

#include <cstdint>
#include <iostream>
#include <string>

using boost::multiprecision::cpp_int;

struct Decimal {
    cpp_int significand;
    int exponent10;
};

Decimal parse_decimal(const std::string& text) {
    std::string digits = text;
    bool negative = false;
    if (!digits.empty() && digits.front() == '-') {
        negative = true;
        digits.erase(digits.begin());
    }

    const auto exponent_marker = digits.find_first_of("eE");
    int exponent10 = 0;
    if (exponent_marker != std::string::npos) {
        exponent10 = std::stoi(digits.substr(exponent_marker + 1));
        digits.resize(exponent_marker);
    }

    const auto decimal_point = digits.find('.');
    if (decimal_point != std::string::npos) {
        exponent10 -= static_cast<int>(digits.size() - decimal_point - 1);
        digits.erase(decimal_point, 1);
    }

    cpp_int significand = 0;
    for (const char digit : digits) {
        if (digit >= '0' && digit <= '9') {
            significand = significand * 10 + (digit - '0');
        }
    }
    if (negative) {
        significand = -significand;
    }

    return {significand, exponent10};
}

cpp_int power_of_ten(unsigned exponent) {
    cpp_int result = 1;
    for (unsigned i = 0; i < exponent; ++i) {
        result *= 10;
    }
    return result;
}

// Return the binary64 subnormal payload obtained by exact rational
// round-to-nearest, ties-to-even. The fixture's values are all nonnegative
// and small enough that the rounded payload fits in uint64_t.
std::uint64_t rounded_subnormal_payload(const Decimal& decimal) {
    cpp_int numerator = decimal.significand;
    cpp_int denominator = 1;
    if (decimal.exponent10 >= 0) {
        numerator *= power_of_ten(
            static_cast<unsigned>(decimal.exponent10));
    } else {
        denominator = power_of_ten(
            static_cast<unsigned>(-decimal.exponent10));
    }

    numerator <<= 1074;
    cpp_int quotient = numerator / denominator;
    const cpp_int remainder = numerator % denominator;
    const cpp_int twice_remainder = remainder * 2;

    if (twice_remainder > denominator ||
        (twice_remainder == denominator && (quotient & 1) != 0)) {
        ++quotient;
    }

    return quotient.convert_to<std::uint64_t>();
}

int main() {
    const std::string literals[] = {
        "4.9406564584124654e-324",
        "5e-324",
        "3e-324",
        "2.4703282292062328e-324",
        "2.470328229206232720882538e-324",
        "9.8813129168249309e-324",
        "7.4109846876186982e-324",
        "2.2250738585072014e-308",
        "1e-400",
    };

    for (const auto& literal : literals) {
        std::cout << literal << " bits="
                  << rounded_subnormal_payload(parse_decimal(literal)) << '\n';
    }
}
