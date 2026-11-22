// Coefficients of the native sin / cos builtins (codegen_x86_linux.sio,
// emit_builtin_trig): the Taylor coefficients (-1)^j / k! rounded to double from
// long double, and the Cody-Waite split of 2 pi (hi has 24 significant bits so
// n * hi is exact for |n| < 2^29). Also prints, for the record, the truncation
// bound at |r| = pi of the series as emitted: sin through r^27, cos through r^28.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
static unsigned long long bits(double d) { unsigned long long u; std::memcpy(&u, &d, 8); return u; }
int main() {
  const long double PI = 3.141592653589793238462643383279502884L;
  long double f = 1;
  for (int k = 1; k <= 29; ++k) {
    f *= k;
    long double c = 1 / f;
    int j = (k % 2) ? (k - 1) / 2 : k / 2;
    if (j % 2) c = -c;
    std::printf("k=%2d 0x%016llX  // %s1/%d!\n", k, bits((double)c), c < 0 ? "-" : "", k);
  }
  double hi = (double)(float)(2 * PI);
  double lo = (double)(2 * PI - (long double)hi);
  std::printf("2pi hi 0x%016llX (%.17g)\n2pi lo 0x%016llX (%.17g)\n1/2pi 0x%016llX\n", bits(hi), hi, bits(lo), lo, bits((double)(1 / (2 * PI))));
  long double fs = 1, fc = 1;
  for (int k = 1; k <= 29; ++k) fs *= k;
  for (int k = 1; k <= 30; ++k) fc *= k;
  std::printf("truncation at pi: sin %.3Le  cos %.3Le\n", std::pow(PI, 29) / fs, std::pow(PI, 30) / fc);
}
