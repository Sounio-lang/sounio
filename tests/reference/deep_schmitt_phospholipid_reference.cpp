#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace {

// Schmitt (2008), doi:10.1016/j.tiv.2007.09.010.  The published
// corrigendum corrects Table 2, not Eqs. 17/18.  Wright (2015), OhioLINK
// accession wright1432890247, independently identifies the reversed 0.05 and
// 20 coefficients.  The arrays below are httk 2.7.4 tissue.data values.

constexpr double corrected(double kn_pl, double neutral, double zwitter,
                           double positive, double negative) {
  return kn_pl * (neutral + zwitter + 20.0 * positive + 0.05 * negative);
}

constexpr double reversed(double kn_pl, double neutral, double zwitter,
                          double positive, double negative) {
  return kn_pl * (neutral + zwitter + 0.05 * positive + 20.0 * negative);
}

constexpr std::array<double, 14> fa_plc{
    -1.0,     0.050510, 0.068530, 0.167800, 0.030900, 0.140000, 0.078840,
    0.007859, 0.050000, 0.138200, 0.040000, 0.072870, -1.0,     -1.0};

constexpr std::array<double, 14> flc{
    -1.0,     0.092310, 0.064070, 0.112300, 0.137500, 0.013160, 0.023870,
    0.934000, 0.070000, 0.283100, 0.020000, 0.039460, -1.0,     -1.0};

constexpr bool supported(std::size_t organ) { return organ > 0 && organ < 12; }

constexpr double httk_ratio(std::size_t organ) {
  return supported(organ) ? fa_plc[organ] * flc[organ] / 0.000501 : -1.0;
}

constexpr bool close(double a, double b, double tolerance = 1e-12) {
  const double difference = a - b;
  return (difference < 0.0 ? -difference : difference) <= tolerance;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2 || std::string_view{argv[1]} != "--self-test") {
    std::cerr << "usage: deep_schmitt_phospholipid_reference --self-test\n";
    return 64;
  }

  static_assert(close(corrected(2.0, 0.15, 0.10, 0.50, 0.25), 20.525));
  static_assert(close(reversed(2.0, 0.15, 0.10, 0.50, 0.25), 10.55));
  static_assert(corrected(2.0, 0.15, 0.10, 0.50, 0.25) -
                    reversed(2.0, 0.15, 0.10, 0.50, 0.25) >
                9.97);

  std::size_t coverage = 0;
  for (std::size_t organ = 0; organ < fa_plc.size(); ++organ) {
    if (supported(organ)) {
      ++coverage;
    } else if (httk_ratio(organ) >= 0.0) {
      std::cerr << "unsupported organ returned a nonnegative ratio\n";
      return 1;
    }
  }

  if (coverage != 11 || !close(httk_ratio(6), 3.7563089820359274) ||
      !close(httk_ratio(3), 37.61265469061876)) {
    std::cerr << std::setprecision(17)
              << "httk tissue:RBC ratio control failed: coverage=" << coverage
              << " muscle=" << httk_ratio(6) << " brain=" << httk_ratio(3)
              << '\n';
    return 1;
  }

  constexpr double rr_muscle_ratio = 2.42 / 0.57;
  if (!close(rr_muscle_ratio, 4.24561403508772) ||
      !(httk_ratio(6) < rr_muscle_ratio)) {
    std::cerr << "source-specific muscle ratio control failed\n";
    return 1;
  }

  std::cout << "PASS\n";
  return 0;
}
