// ADR009: independent C++23 numerical reference for Sounio Type-A witnesses.
// Build: c++ -std=c++23 -O2 -I/opt/homebrew/include calibration_cpp23_oracle.cpp -o oracle
// Uses decimal floating point with 100 decimal digits; it is a numerical
// reference, not a proof of arbitrary-input floating-point accuracy.
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <array>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>

using Decimal = boost::multiprecision::cpp_dec_float_100;

struct TypeA {
    Decimal mean;
    Decimal stddev;
    Decimal uncertainty;
};

TypeA evaluate(std::span<const Decimal> readings) {
    if (readings.size() < 2 || readings.size() > 16) {
        throw std::invalid_argument("Type-A reference requires 2..16 readings");
    }
    const Decimal count(readings.size());
    Decimal sum = 0;
    for (const auto& x : readings) sum += x;
    const Decimal mean = sum / count;
    Decimal squared = 0;
    for (const auto& x : readings) {
        const Decimal delta = x - mean;
        squared += delta * delta;
    }
    const Decimal variance = squared / Decimal(readings.size() - 1);
    const Decimal stddev = sqrt(variance);
    return {mean, stddev, stddev / sqrt(count)};
}

int main() {
    const std::array<Decimal, 5> symmetric = {
        Decimal(-2), Decimal(-1), Decimal(0), Decimal(1), Decimal(2)
    };
    const std::array<Decimal, 2> small = {Decimal("1e-12"), Decimal("2e-12")};
    const std::array<Decimal, 2> large = {Decimal("-1e150"), Decimal("1e150")};
    std::array<Decimal, 16> constant;
    constant.fill(Decimal(4));
    const auto a = evaluate(symmetric);
    const auto b = evaluate(small);
    const auto c = evaluate(large);
    const auto d = evaluate(constant);
    // Exact expected means and squares are an independent check on the
    // reference implementation before comparing its decimal square roots.
    const auto close = [](const Decimal& actual, const Decimal& expected) {
        return abs(actual - expected) <= abs(expected) * Decimal("1e-90");
    };
    if (a.mean != 0 || !close(a.stddev * a.stddev, Decimal("2.5")) ||
        b.mean != Decimal("1.5e-12") || !close(b.uncertainty, Decimal("5e-13")) ||
        c.mean != 0 || !close(c.uncertainty, Decimal("1e150")) ||
        d.mean != 4 || d.stddev != 0 || d.uncertainty != 0) return 2;
    for (const auto invalid_count : {0, 1, 17}) {
        std::array<Decimal, 17> readings;
        try {
            (void)evaluate(std::span<const Decimal>(readings.data(), invalid_count));
            return 3;
        } catch (const std::invalid_argument&) {
            // This is the expected domain rejection, also tested in Sounio.
        }
    }

    std::cout << std::setprecision(60)
              << "{\n"
              << "  \"language\": \"C++23\",\n"
              << "  \"precision\": \"boost::multiprecision::cpp_dec_float_100\",\n"
              << "  \"symmetric_stddev\": \"" << a.stddev << "\",\n"
              << "  \"symmetric_uncertainty\": \"" << a.uncertainty << "\",\n"
              << "  \"small_mean\": \"" << b.mean << "\",\n"
              << "  \"small_uncertainty\": \"" << b.uncertainty << "\",\n"
              << "  \"large_uncertainty\": \"" << c.uncertainty << "\",\n"
              << "  \"constant_uncertainty\": \"" << d.uncertainty << "\"\n"
              << "}\n";
}
