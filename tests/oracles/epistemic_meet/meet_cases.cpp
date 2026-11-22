// Oracle for tests/run-pass/epistemic_meet_*.sio (ADR-016 meet over GUM channels).
// Each source is a linear combination of independent channels, so V = S diag(s2) S^T
// exactly, and the GLS fusion, its variance, E_n (k = 2), the Birge ratio and
// n_eff = 1^T R^-1 1 follow in closed form. long double; printed to 17 digits.
//   A  two independent sources: a = 10.0 + c0 (0.3), b = 10.4 + c1 (0.4).
//   B  a shared channel: a = 10.0 + c0 (0.2) + s, b = 10.3 + c1 (0.3) + 0.5 s,
//      s with u = 0.25.  cov(fused, a) must equal var(fused) (GLS identity).
//   C  three independent, mutually inconsistent sources for meet_birge:
//      1.0 (0.1), 1.3 (0.1), 0.8 (0.1).
//   D  the same three as a contest: retract 1.3, then 0.8.
// Build: g++ -std=c++23 -O2 meet_cases.cpp && ./a.out
#include <cstdio>
#include <cmath>
#include <vector>
using R = long double;
using Mat = std::vector<std::vector<R>>;
static std::vector<R> solve(Mat A, std::vector<R> b) {
  int n = (int)b.size();
  for (int k = 0; k < n; ++k) for (int i = k + 1; i < n; ++i) {
    R f = A[i][k] / A[k][k];
    for (int j = k + 1; j < n; ++j) A[i][j] -= f * A[k][j];
    b[i] -= f * b[k];
  }
  for (int r = n - 1; r >= 0; --r) { R acc = b[r]; for (int j = r + 1; j < n; ++j) acc -= A[r][j] * b[j]; b[r] = acc / A[r][r]; }
  return b;
}
struct Res { std::vector<R> w; R x, var, chi2, rb, neff; };
static Res gls(const Mat& V, const std::vector<R>& x) {
  int n = (int)x.size();
  auto y = solve(V, std::vector<R>(n, 1));
  R s = 0; for (R v : y) s += v;
  Res r; r.w.resize(n); r.x = 0;
  for (int i = 0; i < n; ++i) { r.w[i] = y[i] / s; r.x += r.w[i] * x[i]; }
  r.var = 1 / s;
  std::vector<R> res(n); for (int i = 0; i < n; ++i) res[i] = x[i] - r.x;
  auto t = solve(V, res); r.chi2 = 0; for (int i = 0; i < n; ++i) r.chi2 += res[i] * t[i];
  r.rb = sqrtl(r.chi2 / (n - 1));
  std::vector<R> sg(n); for (int i = 0; i < n; ++i) sg[i] = sqrtl(V[i][i]);
  auto q = solve(V, sg); r.neff = 0; for (int i = 0; i < n; ++i) r.neff += sg[i] * q[i];
  return r;
}
static R en(const Mat& V, const std::vector<R>& x, int i, int j) {
  return fabsl(x[i] - x[j]) / (2 * sqrtl(V[i][i] + V[j][j] - 2 * V[i][j]));
}
int main() {
  {
    Mat V = {{0.09L, 0}, {0, 0.16L}}; std::vector<R> x = {10.0L, 10.4L};
    auto r = gls(V, x);
    std::printf("A en %.17Lg x %.17Lg var %.17Lg w0 %.17Lg neff %.17Lg\n", en(V, x, 0, 1), r.x, r.var, r.w[0], r.neff);
  }
  {
    R s2 = 0.0625L;
    Mat V = {{0.04L + s2, 0.5L * s2}, {0.5L * s2, 0.09L + 0.25L * s2}}; std::vector<R> x = {10.0L, 10.3L};
    auto r = gls(V, x);
    R cov_fa = r.w[0] * V[0][0] + r.w[1] * V[1][0];
    std::printf("B en %.17Lg x %.17Lg var %.17Lg w0 %.17Lg cov_fa %.17Lg neff %.17Lg\n", en(V, x, 0, 1), r.x, r.var, r.w[0], cov_fa, r.neff);
  }
  {
    Mat V = {{0.01L, 0, 0}, {0, 0.01L, 0}, {0, 0, 0.01L}}; std::vector<R> x = {1.0L, 1.3L, 0.8L};
    auto r = gls(V, x);
    R f = r.rb > 1 ? r.rb : 1;
    std::printf("C x %.17Lg var %.17Lg birge %.17Lg var_birge %.17Lg neff %.17Lg en01 %.17Lg\n", r.x, r.var, r.rb, r.var * f * f, r.neff, en(V, x, 0, 1));
  }
  {
    // D: the sources of C as a contest; retract 1.3, then 0.8.
    Mat V = {{0.01L, 0}, {0, 0.01L}}; std::vector<R> x = {1.0L, 0.8L};
    auto r = gls(V, x);
    std::printf("D after one retract: x %.17Lg var %.17Lg birge %.17Lg en %.17Lg; after two: x 1 var 0.01\n", r.x, r.var, r.rb, en(V, x, 0, 1));
  }
  return 0;
}
