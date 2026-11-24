// ADR-009 verified_foreign_reference (C++23 conformal held-out report formatter)
// Keeps the existing step4_gate rule (cp95_upper_1sided <= alpha) unchanged as RATE_DEMONSTRATED,
// reports GUARANTEE_REFUTED (cp95_lower_1sided > alpha) separately, and solves for the minimum
// sample size n needed to demonstrate alpha at the observed rate (cp95_upper_1sided <= alpha).

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

static double round_n(double x, int digits) {
  double p = std::pow(10.0, static_cast<double>(digits));
  return std::round(x * p) / p;
}

// Regularized incomplete beta I_x(a, b) via Lentz continued fraction.
static long double betacf(long double a, long double b, long double x) {
  const int maxit = 400;
  const long double eps = 3e-16L;
  const long double fpmin = 1e-300L;
  long double qab = a + b;
  long double qap = a + 1.0L;
  long double qam = a - 1.0L;
  long double c = 1.0L;
  long double d = 1.0L - qab * x / qap;
  if (std::fabsl(d) < fpmin) d = fpmin;
  d = 1.0L / d;
  long double h = d;
  for (int m = 1; m <= maxit; ++m) {
    int m2 = 2 * m;
    long double aa = static_cast<long double>(m) * (b - static_cast<long double>(m)) * x /
                     ((qam + static_cast<long double>(m2)) * (a + static_cast<long double>(m2)));
    d = 1.0L + aa * d;
    if (std::fabsl(d) < fpmin) d = fpmin;
    c = 1.0L + aa / c;
    if (std::fabsl(c) < fpmin) c = fpmin;
    d = 1.0L / d;
    h *= d * c;
    aa = -(a + static_cast<long double>(m)) * (qab + static_cast<long double>(m)) * x /
         ((a + static_cast<long double>(m2)) * (qap + static_cast<long double>(m2)));
    d = 1.0L + aa * d;
    if (std::fabsl(d) < fpmin) d = fpmin;
    c = 1.0L + aa / c;
    if (std::fabsl(c) < fpmin) c = fpmin;
    d = 1.0L / d;
    long double del = d * c;
    h *= del;
    if (std::fabsl(del - 1.0L) < eps) break;
  }
  return h;
}

static long double betai(long double a, long double b, long double x) {
  if (x <= 0.0L) return 0.0L;
  if (x >= 1.0L) return 1.0L;
  long double bt = std::expl(std::lgammal(a + b) - std::lgammal(a) - std::lgammal(b) +
                             a * std::logl(x) + b * std::logl(1.0L - x));
  if (x < (a + 1.0L) / (a + b + 2.0L)) {
    return bt * betacf(a, b, x) / a;
  }
  return 1.0L - bt * betacf(b, a, 1.0L - x) / b;
}

// Beta(a, b) quantile by bisection.
static double beta_quantile(double q, long double a, long double b) {
  long double lo = 0.0L;
  long double hi = 1.0L;
  for (int i = 0; i < 100; ++i) {
    long double mid = 0.5L * (lo + hi);
    if (betai(a, b, mid) < static_cast<long double>(q)) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return static_cast<double>(0.5L * (lo + hi));
}

// One-sided 95% Clopper-Pearson upper bound (matches step4_gate / BetaInv(0.95, k+1, n-k)).
static double clopper_pearson_upper_1sided(int64_t k, int64_t n) {
  if (k >= n) return 1.0;
  if (k <= 0) return 1.0 - std::pow(0.05, 1.0 / static_cast<double>(n));
  return beta_quantile(0.95, static_cast<long double>(k + 1), static_cast<long double>(n - k));
}

// One-sided 95% Clopper-Pearson lower bound (BetaInv(0.05, k, n-k+1)).
static double clopper_pearson_lower_1sided(int64_t k, int64_t n) {
  if (k <= 0) return 0.0;
  if (k >= n) return std::pow(0.05, 1.0 / static_cast<double>(n));
  return beta_quantile(0.05, static_cast<long double>(k), static_cast<long double>(n - k + 1));
}

static std::pair<double, double> clopper_pearson_2sided(int64_t k, int64_t n) {
  double lo = (k <= 0) ? 0.0
                       : beta_quantile(0.025, static_cast<long double>(k),
                                       static_cast<long double>(n - k + 1));
  double hi = (k >= n) ? 1.0
                       : beta_quantile(0.975, static_cast<long double>(k + 1),
                                       static_cast<long double>(n - k));
  return {lo, hi};
}

// Minimum n >= 1 such that one-sided 95% CP upper bound <= alpha at observed rate p_hat.
// For p_hat == 0: exact analytical ceil(ln(0.05) / ln(1 - alpha)).
// For 0 < p_hat < alpha: uses conservative k(n) = ceil(p_hat * n) and binary search with
// exact local scan around the boundary so non-monotonicity from ceil(p_hat * n) cannot understate n.
// Returns std::nullopt if p_hat >= alpha (impossible at any finite n).
static std::optional<int64_t> solve_n_needed(double p_hat, double alpha) {
  if (p_hat >= alpha) return std::nullopt;
  if (p_hat <= 0.0) {
    int64_t n = static_cast<int64_t>(std::ceil(std::log(0.05) / std::log(1.0 - alpha)));
    while (n > 1 && clopper_pearson_upper_1sided(0, n - 1) <= alpha) --n;
    while (clopper_pearson_upper_1sided(0, n) > alpha) ++n;
    return n;
  }

  auto ok_at = [&](int64_t n) -> bool {
    int64_t k = static_cast<int64_t>(std::ceil(p_hat * static_cast<double>(n) - 1e-15));
    return clopper_pearson_upper_1sided(k, n) <= alpha;
  };

  // Normal-approx starting scale to bracket quickly.
  double diff = alpha - p_hat;
  double z = 1.6448536269514722;
  int64_t est = std::max<int64_t>(
      1, static_cast<int64_t>(std::ceil(z * z * p_hat * (1.0 - p_hat) / (diff * diff))));
  int64_t hi = est;
  while (!ok_at(hi)) {
    if (hi > (1LL << 40)) return std::nullopt;
    hi *= 2;
  }
  int64_t lo = 1;
  while (lo < hi) {
    int64_t mid = lo + (hi - lo) / 2;
    if (ok_at(mid)) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  // Because ceil(p_hat * n) jumps by 1 every ~1/p_hat steps, ensure all n >= ans hold
  // (smallest n from which every larger sample size also passes at ceil(p_hat*n)).
  int64_t period = static_cast<int64_t>(std::ceil(1.0 / p_hat)) + 2;
  int64_t ans = lo;
  for (int64_t probe = lo; probe <= lo + 4 * period; ++probe) {
    if (!ok_at(probe)) ans = probe + 1;
  }
  return ans;
}

static void print_verdict(const std::string& label, int64_t k, int64_t n, double alpha) {
  double p_hat = static_cast<double>(k) / static_cast<double>(n);
  double cp_lo_1 = clopper_pearson_lower_1sided(k, n);
  double cp_hi_1 = clopper_pearson_upper_1sided(k, n);
  auto [cp_lo_2, cp_hi_2] = clopper_pearson_2sided(k, n);
  bool guarantee_refuted = (cp_lo_1 > alpha);
  bool rate_demonstrated = (cp_hi_1 <= alpha);
  auto n_need = solve_n_needed(p_hat, alpha);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "[" << label << "] k=" << k << "/" << n << " observed_rate=" << p_hat
            << " alpha=" << alpha << " cp95_1sided=[" << cp_lo_1 << ", " << cp_hi_1 << "]"
            << " cp95_2sided=[" << cp_lo_2 << ", " << cp_hi_2 << "]\n";
  std::cout << "GUARANTEE_REFUTED " << (guarantee_refuted ? "yes" : "no")
            << " (CP95 lower=" << cp_lo_1 << " > alpha=" << alpha << ")\n";
  std::cout << "RATE_DEMONSTRATED " << (rate_demonstrated ? "yes" : "no")
            << " (CP95 upper=" << cp_hi_1 << " <= alpha=" << alpha << ")\n";
  if (n_need.has_value()) {
    int64_t k_at_n = static_cast<int64_t>(std::ceil(p_hat * static_cast<double>(*n_need) - 1e-15));
    double ub_at_n = clopper_pearson_upper_1sided(k_at_n, *n_need);
    std::cout << "N_NEEDED_TO_DEMONSTRATE alpha=" << alpha << " at observed_rate=" << p_hat
              << ": n=" << *n_need << " (k=" << k_at_n << "/" << *n_need
              << ", CP95 upper=" << ub_at_n << " <= " << alpha << ")\n";
  } else {
    std::cout << "N_NEEDED_TO_DEMONSTRATE alpha=" << alpha << " at observed_rate=" << p_hat
              << ": impossible (observed_rate >= alpha)\n";
  }
}

static bool run_selftest() {
  if (std::abs(round_n(clopper_pearson_upper_1sided(6, 240), 6) - 0.048744) > 1e-6 ||
      std::abs(round_n(clopper_pearson_upper_1sided(5, 240), 6) - 0.043303) > 1e-6 ||
      std::abs(round_n(clopper_pearson_upper_1sided(0, 240), 6) - 0.012405) > 1e-6) {
    std::cerr << "SELFTEST_FAIL: CP95 upper reference mismatch\n";
    return false;
  }
  // 0/240 at alpha=0.01 -> n_needed == 299
  auto n01 = solve_n_needed(0.0, 0.01);
  if (!n01.has_value() || *n01 != 299) {
    std::cerr << "SELFTEST_FAIL: expected n=299 for p=0, alpha=0.01\n";
    return false;
  }
  // 0/240 at alpha=0.05 -> n_needed == 59
  auto n05 = solve_n_needed(0.0, 0.05);
  if (!n05.has_value() || *n05 != 59) {
    std::cerr << "SELFTEST_FAIL: expected n=59 for p=0, alpha=0.05\n";
    return false;
  }
  // 12/240 = 0.05 >= 0.05 -> impossible
  if (solve_n_needed(12.0 / 240.0, 0.05).has_value()) {
    std::cerr << "SELFTEST_FAIL: expected impossible for p=0.05, alpha=0.05\n";
    return false;
  }
  // Predicted L=256 case: k=11/240, alpha=0.05 -> GUARANTEE_REFUTED no, RATE_DEMONSTRATED no
  if (clopper_pearson_lower_1sided(11, 240) > 0.05) {
    std::cerr << "SELFTEST_FAIL: k=11/240 should not refute alpha=0.05\n";
    return false;
  }
  if (clopper_pearson_upper_1sided(11, 240) <= 0.05) {
    std::cerr << "SELFTEST_FAIL: k=11/240 should not demonstrate alpha=0.05\n";
    return false;
  }
  // Refutation case: k=20/240 -> CP95 lower > 0.05
  if (!(clopper_pearson_lower_1sided(20, 240) > 0.05)) {
    std::cerr << "SELFTEST_FAIL: k=20/240 should refute alpha=0.05\n";
    return false;
  }
  print_verdict("selftest_pred_11_of_240", 11, 240, 0.05);
  print_verdict("selftest_pass_6_of_240", 6, 240, 0.05);
  print_verdict("selftest_refute_20_of_240", 20, 240, 0.05);
  std::cout << "CONFORMAL_REPORT_SELFTEST_PASS\n";
  return true;
}

static bool report_from_json(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) {
    std::cerr << "ERROR: cannot open " << path << "\n";
    return false;
  }
  std::ostringstream ss;
  ss << ifs.rdbuf();
  std::string body = ss.str();

  auto pos = body.find("\"heldout_audit\"");
  if (pos == std::string::npos) {
    std::cerr << "ERROR: missing heldout_audit in " << path << "\n";
    return false;
  }
  auto end_pos = body.find("\"heldout40_timing\"", pos);
  std::string sub = (end_pos == std::string::npos) ? body.substr(pos) : body.substr(pos, end_pos - pos);

  int found = 0;
  for (const auto& [key, alpha] :
       std::initializer_list<std::pair<std::string, double>>{{"alpha_0.05", 0.05}, {"alpha_0.01", 0.01}}) {
    auto kpos = sub.find("\"" + key + "\"");
    if (kpos == std::string::npos) continue;
    auto block = sub.substr(kpos, 1200);
    std::smatch m_n, m_k;
    if (std::regex_search(block, m_n, std::regex("\"n_prompts\"\\s*:\\s*([0-9]+)")) &&
        std::regex_search(block, m_k, std::regex("\"mismatches\"\\s*:\\s*([0-9]+)"))) {
      int64_t n = std::stoll(m_n[1].str());
      int64_t k = std::stoll(m_k[1].str());
      print_verdict(key, k, n, alpha);
      ++found;
    }
  }
  if (found == 0) {
    std::cout << "[pending] heldout_audit is empty in " << path << " (HELD-OUT not yet complete)\n";
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--selftest") {
    return run_selftest() ? 0 : 1;
  }
  if (argc == 5 && std::string(argv[1]) == "--eval") {
    int64_t k = std::stoll(argv[2]);
    int64_t n = std::stoll(argv[3]);
    double alpha = std::stod(argv[4]);
    print_verdict("eval", k, n, alpha);
    return 0;
  }
  if (argc == 3 && std::string(argv[1]) == "--json") {
    return report_from_json(argv[2]) ? 0 : 1;
  }
  std::cerr << "usage:\n"
            << "  conformal_report --selftest\n"
            << "  conformal_report --eval <k> <n> <alpha>\n"
            << "  conformal_report --json <spark_qwen3_8b_cuda_q4_length_L*.json>\n";
  return 2;
}
