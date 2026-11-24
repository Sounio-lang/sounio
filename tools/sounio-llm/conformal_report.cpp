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

// Exact integer ceil((k_obs * n) / n_obs) for fixed observed rate p = k_obs / n_obs.
static int64_t ceil_rate_k(int64_t k_obs, int64_t n_obs, int64_t n) {
  __int128 num = static_cast<__int128>(k_obs) * static_cast<__int128>(n) +
                 static_cast<__int128>(n_obs) - 1;
  return static_cast<int64_t>(num / static_cast<__int128>(n_obs));
}

// One definition of N_NEEDED: hold observed rate p = k_obs / n_obs fixed, and find the
// smallest n >= 1 such that k = ceil(p * n) satisfies CP95_upper_1sided(k, n) <= alpha.
// Returns std::nullopt if p >= alpha (impossible at any finite n).
static std::optional<int64_t> solve_n_needed(int64_t k_obs, int64_t n_obs, double alpha) {
  double p_hat = static_cast<double>(k_obs) / static_cast<double>(n_obs);
  if (p_hat >= alpha) return std::nullopt;
  if (k_obs <= 0) {
    int64_t n = static_cast<int64_t>(std::ceil(std::log(0.05) / std::log(1.0 - alpha)));
    while (n > 1 && clopper_pearson_upper_1sided(0, n - 1) <= alpha) --n;
    while (clopper_pearson_upper_1sided(0, n) > alpha) ++n;
    return n;
  }

  // Within each constant-k plateau (k - 1) * n_obs < k_obs * n <= k * n_obs,
  // CP95_upper(k, n) is strictly decreasing in n and minimized at n_max(k) = floor(k * n_obs / k_obs).
  auto n_max_for_k = [&](int64_t k) -> int64_t {
    return static_cast<int64_t>((static_cast<__int128>(k) * static_cast<__int128>(n_obs)) /
                                static_cast<__int128>(k_obs));
  };
  auto n_min_for_k = [&](int64_t k) -> int64_t {
    return static_cast<int64_t>((static_cast<__int128>(k - 1) * static_cast<__int128>(n_obs)) /
                                static_cast<__int128>(k_obs)) +
           1;
  };
  auto plateau_passes = [&](int64_t k) -> bool {
    int64_t n_hi = n_max_for_k(k);
    return n_hi >= 1 && clopper_pearson_upper_1sided(k, n_hi) <= alpha;
  };

  int64_t k_hi = 1;
  while (!plateau_passes(k_hi)) {
    if (k_hi > (1LL << 38)) return std::nullopt;
    k_hi *= 2;
  }
  // Because n_max_for_k(k) = floor(k * n_obs / k_obs) can have tiny rounding jitter of < 1 in n
  // across consecutive k, first binary search for a passing k_hi, then scan down to verify the
  // exact smallest passing k plateau.
  int64_t k_lo = 1;
  while (k_lo < k_hi) {
    int64_t mid = k_lo + (k_hi - k_lo) / 2;
    if (plateau_passes(mid)) {
      k_hi = mid;
    } else {
      k_lo = mid + 1;
    }
  }
  int64_t best_k = k_lo;
  int64_t scan_lo = std::max<int64_t>(1, best_k - std::max<int64_t>(k_obs, 64));
  for (int64_t k = best_k - 1; k >= scan_lo; --k) {
    if (plateau_passes(k)) best_k = k;
  }

  // Binary search within [n_min(best_k), n_max(best_k)] for the exact smallest n.
  int64_t lo = n_min_for_k(best_k);
  int64_t hi = n_max_for_k(best_k);
  while (lo < hi) {
    int64_t mid = lo + (hi - lo) / 2;
    if (clopper_pearson_upper_1sided(best_k, mid) <= alpha) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

static std::string format_verdict(const std::string& label, int64_t k, int64_t n, double alpha) {
  double p_hat = static_cast<double>(k) / static_cast<double>(n);
  double cp_lo_1 = clopper_pearson_lower_1sided(k, n);
  double cp_hi_1 = clopper_pearson_upper_1sided(k, n);
  auto [cp_lo_2, cp_hi_2] = clopper_pearson_2sided(k, n);
  bool guarantee_refuted = (cp_lo_1 > alpha);
  bool rate_demonstrated = (cp_hi_1 <= alpha);

  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "[" << label << "] k=" << k << "/" << n << " observed_rate=" << p_hat
      << " alpha=" << alpha << " cp95_1sided=[" << cp_lo_1 << ", " << cp_hi_1 << "]"
      << " cp95_2sided=[" << cp_lo_2 << ", " << cp_hi_2 << "]\n";
  out << "GUARANTEE_REFUTED " << (guarantee_refuted ? "yes" : "no ")
      << " (CP95 lower=" << cp_lo_1 << (guarantee_refuted ? " >  " : " <= ")
      << "alpha=" << alpha << "; refuted iff lower > alpha)\n";
  out << "RATE_DEMONSTRATED " << (rate_demonstrated ? "yes" : "no ")
      << " (CP95 upper=" << cp_hi_1 << (rate_demonstrated ? " <= " : " >  ")
      << "alpha=" << alpha << "; demonstrated iff upper <= alpha)\n";

  if (rate_demonstrated) {
    out << "N_NEEDED_TO_DEMONSTRATE alpha=" << alpha << " at observed_rate=" << p_hat
        << ": already demonstrated at n=" << n << "\n";
  } else {
    auto n_need = solve_n_needed(k, n, alpha);
    if (n_need.has_value()) {
      int64_t k_at_n = ceil_rate_k(k, n, *n_need);
      double ub_at_n = clopper_pearson_upper_1sided(k_at_n, *n_need);
      out << "N_NEEDED_TO_DEMONSTRATE alpha=" << alpha << " at observed_rate=" << p_hat
          << ": n=" << *n_need << " (k=ceil(p*n)=" << k_at_n << "/" << *n_need
          << ", CP95 upper=" << ub_at_n << " <= " << alpha << ")\n";
    } else {
      out << "N_NEEDED_TO_DEMONSTRATE alpha=" << alpha << " at observed_rate=" << p_hat
          << ": impossible (observed_rate >= alpha)\n";
    }
  }
  return out.str();
}

static void print_verdict(const std::string& label, int64_t k, int64_t n, double alpha) {
  std::cout << format_verdict(label, k, n, alpha);
}

// Parse every printed verdict line and fail if the printed inequality is false for the printed
// numbers or inconsistent with the yes/no verdict.
static bool verify_printed_inequalities(const std::string& text) {
  std::istringstream iss(text);
  std::string line;
  std::regex re_verdict(
      R"(^(GUARANTEE_REFUTED|RATE_DEMONSTRATED)\s+(yes|no)\s+\(CP95\s+(lower|upper)=([0-9.]+)\s*(<=|>)\s*alpha=([0-9.]+);)");
  int checked = 0;
  while (std::getline(iss, line)) {
    if (line.rfind("GUARANTEE_REFUTED", 0) != 0 && line.rfind("RATE_DEMONSTRATED", 0) != 0) {
      continue;
    }
    std::smatch m;
    if (!std::regex_search(line, m, re_verdict)) {
      std::cerr << "SELFTEST_FAIL: unparseable verdict line: " << line << "\n";
      return false;
    }
    const std::string kind = m[1].str();
    const std::string yn = m[2].str();
    const double lhs = std::stod(m[4].str());
    const std::string op = m[5].str();
    const double rhs = std::stod(m[6].str());
    const bool rel_holds = (op == "<=") ? (lhs <= rhs) : (lhs > rhs);
    if (!rel_holds) {
      std::cerr << "SELFTEST_FAIL: false printed inequality in line: " << line << "\n";
      return false;
    }
    if (kind == "GUARANTEE_REFUTED") {
      if ((yn == "yes" && op != ">") || (yn == "no" && op != "<=")) {
        std::cerr << "SELFTEST_FAIL: GUARANTEE_REFUTED yes/no disagrees with op in: " << line << "\n";
        return false;
      }
    } else {
      if ((yn == "yes" && op != "<=") || (yn == "no" && op != ">")) {
        std::cerr << "SELFTEST_FAIL: RATE_DEMONSTRATED yes/no disagrees with op in: " << line << "\n";
        return false;
      }
    }
    ++checked;
  }
  return checked >= 2;
}

static bool run_selftest() {
  if (std::abs(round_n(clopper_pearson_upper_1sided(6, 240), 6) - 0.048744) > 1e-6 ||
      std::abs(round_n(clopper_pearson_upper_1sided(5, 240), 6) - 0.043303) > 1e-6 ||
      std::abs(round_n(clopper_pearson_upper_1sided(0, 240), 6) - 0.012405) > 1e-6) {
    std::cerr << "SELFTEST_FAIL: CP95 upper reference mismatch\n";
    return false;
  }
  // Negative check on verify_printed_inequalities: must reject false printed inequalities.
  if (verify_printed_inequalities(
          "GUARANTEE_REFUTED no  (CP95 lower=0.025913 >  alpha=0.050000; refuted iff lower > alpha)\n"
          "RATE_DEMONSTRATED no  (CP95 upper=0.074725 <= alpha=0.050000; demonstrated iff upper <= alpha)\n")) {
    std::cerr << "SELFTEST_FAIL: verifier failed to reject false printed inequalities\n";
    return false;
  }

  // 0/240 at alpha=0.01 -> n_needed == 299
  auto n01 = solve_n_needed(0, 240, 0.01);
  if (!n01.has_value() || *n01 != 299) {
    std::cerr << "SELFTEST_FAIL: expected n=299 for p=0, alpha=0.01\n";
    return false;
  }
  // 0/240 at alpha=0.05 -> n_needed == 59
  auto n05 = solve_n_needed(0, 240, 0.05);
  if (!n05.has_value() || *n05 != 59) {
    std::cerr << "SELFTEST_FAIL: expected n=59 for p=0, alpha=0.05\n";
    return false;
  }
  // 12/240 = 0.05 >= 0.05 -> impossible
  if (solve_n_needed(12, 240, 0.05).has_value()) {
    std::cerr << "SELFTEST_FAIL: expected impossible for p=0.05, alpha=0.05\n";
    return false;
  }
  // Minimality check for k=11/240, alpha=0.05: verify n_needed passes and every smaller n fails.
  auto n11 = solve_n_needed(11, 240, 0.05);
  if (!n11.has_value()) {
    std::cerr << "SELFTEST_FAIL: expected finite n_needed for k=11/240, alpha=0.05\n";
    return false;
  }
  if (clopper_pearson_upper_1sided(ceil_rate_k(11, 240, *n11), *n11) > 0.05) {
    std::cerr << "SELFTEST_FAIL: n11=" << *n11 << " does not satisfy CP95 upper <= 0.05\n";
    return false;
  }
  for (int64_t prev = std::max<int64_t>(1, *n11 - 500); prev < *n11; ++prev) {
    if (clopper_pearson_upper_1sided(ceil_rate_k(11, 240, prev), prev) <= 0.05) {
      std::cerr << "SELFTEST_FAIL: n11=" << *n11 << " is not smallest; prev=" << prev << " passes\n";
      return false;
    }
  }

  const std::string s11 = format_verdict("selftest_pred_11_of_240", 11, 240, 0.05);
  const std::string s6 = format_verdict("selftest_pass_6_of_240", 6, 240, 0.05);
  const std::string s20 = format_verdict("selftest_refute_20_of_240", 20, 240, 0.05);
  if (!verify_printed_inequalities(s11) || !verify_printed_inequalities(s6) ||
      !verify_printed_inequalities(s20)) {
    return false;
  }
  if (s6.find("already demonstrated at n=240") == std::string::npos) {
    std::cerr << "SELFTEST_FAIL: passing case must print 'already demonstrated at n=240'\n";
    return false;
  }
  std::cout << s11 << s6 << s20;
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
