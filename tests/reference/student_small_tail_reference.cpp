#include <cmath>
#include <iomanip>
#include <iostream>
int main() {
  std::cout << std::setprecision(17);
  for (double t : {1e-10, 1e-8, 1e-7}) {
    double r = t / std::sqrt(2.0);
    double x = 1.0 / (1.0 + r*r);
    double via_x = 1.0 - std::sqrt(1.0-x);
    long double exact = 1.0L - static_cast<long double>(t) /
      std::hypotl(static_cast<long double>(t), std::sqrt(2.0L));
    double complement_argument = (r*r) / (1.0+r*r);
    double complementary_closed_form = 1.0-std::sqrt(complement_argument);
    std::cout << "df=2 t=" << t << " original_x=" << x
      << " old_closed_form=" << via_x << " expected=" << static_cast<double>(exact)
      << " old_abs_error=" << static_cast<double>(std::abs(static_cast<long double>(via_x)-exact))
      << " complementary_closed_form=" << complementary_closed_form
      << " complementary_abs_error=" << static_cast<double>(std::abs(static_cast<long double>(complementary_closed_form)-exact))
      << "\n";
  }
  std::cout << "native_fixture_absolute_tolerance=2e-14\n";
}
