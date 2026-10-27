// Peelle's Pertinent Puzzle: additive-GLS vs log-domain combination.
// x1 = 1.5, x2 = 1.0; independent 10% each; shared normalisation 20%.
#include <cmath>
#include <cstdio>

struct Comb { double est, u, w1, w2; };

// GLS for two measurements with covariance [[a, c], [c, b]].
static Comb gls(double x1, double x2, double a, double b, double c) {
  const double det = a * b - c * c;
  const double s1 = (b - c) / det, s2 = (a - c) / det;  // V^-1 * 1
  const double sum = s1 + s2;
  return {(s1 * x1 + s2 * x2) / sum, std::sqrt(1.0 / sum), s1 / sum, s2 / sum};
}

int main() {
  const double x1 = 1.5, x2 = 1.0, ri = 0.10, rn = 0.20;

  // Additive model: relative components evaluated at the measured values.
  const double a = std::pow(ri * x1, 2) + std::pow(rn * x1, 2);
  const double b = std::pow(ri * x2, 2) + std::pow(rn * x2, 2);
  const double c = (rn * x1) * (rn * x2);
  const Comb add = gls(x1, x2, a, b, c);
  std::printf("additive GLS : %.4f +- %.4f  (w1 = %+.4f, w2 = %+.4f)\n",
              add.est, add.u, add.w1, add.w2);

  // Multiplicative model: combine in log space, where the shared factor is additive.
  const double la = ri * ri + rn * rn, lb = la, lc = rn * rn;
  const Comb lg = gls(std::log(x1), std::log(x2), la, lb, lc);
  const double est = std::exp(lg.est);
  std::printf("log-domain   : %.4f +- %.4f  (rel %.2f%%, w1 = %+.4f, w2 = %+.4f)\n",
              est, est * lg.u, 100.0 * lg.u, lg.w1, lg.w2);
  return 0;
}
