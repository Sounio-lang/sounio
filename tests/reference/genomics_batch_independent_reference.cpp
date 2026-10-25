// C++23 binary128 independent oracle: frozen SIO171 G8 N12 P2 ML case0.
// Compile on Slurm: g++ -std=c++23 -O2 this.cpp -lquadmath -o reference.
// Usage: reference [native.stdout ...]. Solves heterogeneous-offset group
// score equations and outer profile stationary points independently of IRLS/QR.
// The existing qualified C++ direct-Gram working-prior model is reused below.
// Scope: case0 zero prior and --prior G12/N16 positive-prior ML fixture.
// CR and additional failure/outlier routes require further frozen fixtures.
#define main sio171_existing_prior_main
#include "deep_stats_dispersion_reference.cpp"
#undef main
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
namespace batch {
int G = 8, N = 12;
bool prior_case = false, original_prior_marker = false;
Q upper = 2;
using Row = std::array<Q, 16>;
Q q(const char *s) { return strtoflt128(s, nullptr); }
struct Profile {
  Q t = 0, alpha = 0, ll = 0, criterion = 0, score = 0;
  Row mu{};
};
struct Gene {
  std::array<Q, 24> v{};
  std::array<int, 12> s{};
  Q bh = 1, by = 1;
};
struct Model {
  std::array<Row, 12> y{};
  Row factor{}, offset{};
  std::array<Gene, 12> gene{};
  Fit prior{};
  int eligible = 0;
};
Q pmf(int y, Q mu, Q a) {
  if (a == 0)
    return y * logq(mu) - mu - lgammaq(Q(y + 1));
  Q sum = 0;
  for (int k = 0; k < y; ++k)
    sum += log1pq(a * k);
  return sum - lgammaq(Q(y + 1)) + y * logq(mu) - (y + 1 / a) * log1pq(a * mu);
}
Q scale(const Row &y, const Row &s, int group, Q a) {
  Q lo = q("1e100"), hi = 0, sy = 0, ss = 0;
  for (int j = group * (N / 2); j < (group + 1) * (N / 2); ++j) {
    lo = fminq(lo, y[j] / s[j]);
    hi = fmaxq(hi, y[j] / s[j]);
    sy += y[j];
    ss += s[j];
  }
  if (a == 0)
    return sy / ss;
  // Strictly decreasing score; normalized count extrema enclose unique root.
  for (int k = 0; k < 120; ++k) {
    Q mid = (lo + hi) / 2;
    if (mid == lo || mid == hi)
      break;
    Q z = 0;
    for (int j = group * (N / 2); j < (group + 1) * (N / 2); ++j) {
      Q mu = s[j] * mid;
      z += (y[j] - mu) / (1 + a * mu);
    }
    if (z > 0)
      lo = mid;
    else
      hi = mid;
  }
  return (lo + hi) / 2;
}
Profile eval(const Row &y, const Row &s, Q t, bool poisson = false,
             Q center = 0, Q variance = 0) {
  Profile p;
  p.t = poisson ? Q(0) : t;
  p.alpha = poisson ? Q(0) : expq(t);
  for (int g = 0; g < 2; ++g) {
    Q m = scale(y, s, g, p.alpha);
    require(m > 0, "finite group MLE");
    for (int j = g * (N / 2); j < (g + 1) * (N / 2); ++j) {
      Q mu = s[j] * m;
      p.mu[j] = mu;
      int count = int(y[j]);
      p.ll += pmf(count, mu, p.alpha);
      if (!poisson) {
        Q r = 1 / p.alpha, h = 0;
        for (int k = 0; k < count; ++k)
          h += 1 / (r + k);
        p.score += -r * (h - log1pq(mu / r) + (mu - y[j]) / (r + mu));
      }
    }
  }
  p.criterion = p.ll;
  if (variance > 0) {
    Q d = t - center;
    p.criterion -= d * d / (2 * variance);
    p.score -= d / variance;
  }
  return p;
}
Profile maximize(const Row &y, const Row &s, Q center = 0, Q variance = 0) {
  auto evaluate = [&](Q t) { return eval(y, s, t, false, center, variance); };
  Profile left = evaluate(-10), best = left;
  if (variance == 0) {
    auto poisson = eval(y, s, 0, true);
    if (poisson.criterion > best.criterion)
      best = poisson;
  }
  // Locate all sampled positive-to-negative derivative brackets.
  for (int k = 1; k <= 112; ++k) {
    auto right = evaluate(Q(-10) + (upper + 10) * k / 112);
    if (right.criterion > best.criterion)
      best = right;
    if (left.score > 0 && right.score < 0) {
      Q lo = left.t, hi = right.t;
      for (int j = 0; j < 112; ++j) {
        Q mid = (lo + hi) / 2;
        if (mid == lo || mid == hi)
          break;
        if (evaluate(mid).score > 0)
          lo = mid;
        else
          hi = mid;
      }
      auto p = evaluate((lo + hi) / 2);
      if (p.criterion > best.criterion)
        best = p;
    }
    left = right;
  }
  return best;
}
Q curvature(const Row &y, const Row &s, const Profile &p, Q h) {
  return (2 * p.ll - eval(y, s, p.t + h).ll - eval(y, s, p.t - h).ll) /
         (N * h * h);
}
Model expected() {
  Model m;
  std::vector<Row> logs;
  for (int i = 0; i < G; ++i) {
    Row v{};
    Q center = 0;
    int positives = 0;
    for (int j = 0; j < N; ++j) {
      int count = 0;
      if (prior_case) {
        int base = (i + 1) * 20, noise = ((j * 7 + i * 3) % 17) - 8;
        // Fixture uses f64 sqrt and truncates the positive product to i64.
        // Some products land on integer boundaries (base100, multiplier.7).
        // Audit the actual C++ f64 truncation against binary128 for every
        // frozen base/multiplier; never silently change the fixture counts.
        Q multiplier = i % 4 == 0   ? q(".7")
                       : i % 4 == 1 ? q("1.5")
                       : i % 4 == 2 ? Q(1)
                                    : Q(2);
        int amplitude = base / 30 + int(sqrtq(Q(base)) * multiplier);
        volatile double base64 = base;
        volatile double multiplier64 = i % 4 == 0   ? .7
                                       : i % 4 == 1 ? 1.5
                                       : i % 4 == 2 ? 1.0
                                                    : 2.0;
        double product64 = std::sqrt(base64) * multiplier64;
        int amplitude64 = base / 30 + int(product64);
        require(amplitude64 == amplitude,
                "frozen f64/binary128 amplitude agreement");
        if (j == 0)
          std::printf("AMPLITUDE_AUDIT %d %d %d\n", base, amplitude64,
                      amplitude);
        count = std::max(0, base + noise * amplitude);
        if (j >= N / 2)
          count *= 2;
      } else {
        int noise = i == 0 ? 0 : ((j * 7 + i * 3) % 11) - 5;
        count =
            (i + 1) * 20 + noise * (i + 1) + (j >= N / 2 ? (i + 1) * 10 : 0);
      }
      if (i == G - 1)
        count = 0;
      m.y[i][j] = count;
      if (count > 0) {
        v[j] = logq(Q(count));
        center += v[j];
        ++positives;
      }
    }
    if (positives) {
      center /= positives;
      for (int j = 0; j < N; ++j)
        if (m.y[i][j] > 0)
          v[j] -= center;
      logs.push_back(v);
    }
  }
  Q center = 0;
  for (int j = 0; j < N; ++j) {
    std::vector<Q> col;
    for (size_t i = 0; i < logs.size(); ++i)
      if (m.y[i][j] > 0)
        col.push_back(logs[i][j]);
    std::sort(col.begin(), col.end());
    require(!col.empty(), "normalization sample coverage");
    m.offset[j] = col[col.size() / 2];
    if (col.size() % 2 == 0)
      m.offset[j] = (m.offset[j] + col[col.size() / 2 - 1]) / 2;
    center += m.offset[j] / N;
  }
  for (int j = 0; j < N; ++j) {
    m.offset[j] -= center;
    m.factor[j] = expq(m.offset[j]);
  }
  std::vector<Q> means, alpha, variance;
  std::vector<int> ids;
  for (int i = 0; i < G; ++i) {
    auto &g = m.gene[i];
    g.s.fill(-1);
    g.s[5] = 300;
    for (int j = 0; j < N; ++j)
      g.v[0] += m.y[i][j] / m.factor[j] / N;
    if (i == G - 1) {
      g.s[0] = 3;
      g.s[1] = -1;
      g.s[3] = 1;
      continue;
    }
    auto p = maximize(m.y[i], m.factor);
    g.s[0] = 0;
    g.v[1] = p.alpha;
    g.v[2] = p.t;
    g.v[3] = g.v[4] = p.ll / N;
    g.s[1] = p.alpha == 0 ? 1 : 0;
    g.s[3] = p.alpha == 0 ? 2 : 0;
    g.s[4] = p.alpha == 0 ? 0 : 1;
    if (p.alpha == 0) {
      g.v[20] = 1;
      continue;
    }
    require(p.t > -10 && p.t < upper, "fixture interior");
    require(fabsq(p.score) < q("1e-24"),
            "independent profile score stationary");
    Q h = Q(1) / 256;
    g.v[22] = curvature(m.y[i], m.factor, p, h);
    g.v[23] = curvature(m.y[i], m.factor, p, h / 2);
    require(g.v[22] > 0 && g.v[23] > 0, "positive curvature");
    g.v[5] = 1 / (N * g.v[23]);
    means.push_back(g.v[0]);
    alpha.push_back(p.alpha);
    variance.push_back(g.v[5]);
    ids.push_back(i);
  }
  m.eligible = int(ids.size());
  require(m.eligible == (prior_case ? 11 : 4), "frozen eligible set");
  m.prior = fit(means, alpha, variance, prior_case ? q(".01") : q(".05"),
                prior_case ? Q(10) : Q(1), 2);
  require(prior_case ? m.prior.prior > 0 : m.prior.prior == 0,
          "frozen prior variance class");
  for (size_t k = 0; k < ids.size(); ++k) {
    int i = ids[k];
    auto &g = m.gene[i];
    Q trend = logq(m.prior.a0 + m.prior.a1 / g.v[0]);
    bool high = m.prior.flags[k] != 0;
    g.s[5] = g.s[7] = g.s[8] = g.s[9] = 0;
    g.s[10] = high ? 1 : 0;
    g.s[11] = high ? 3 : m.prior.prior > 0 ? 1 : 2;
    g.v[6] = m.prior.mapped[k];
    g.v[7] = m.prior.variance[k];
    g.v[8] = trend;
    Q final_t = high ? g.v[2] : trend;
    if (!high && m.prior.prior > 0) {
      auto refit = maximize(m.y[i], m.factor, trend, m.prior.prior);
      require(fabsq(refit.score) < q("1e-24"),
              "penalized profile score stationary");
      final_t = refit.t;
      g.v[11] = refit.criterion / N;
      g.v[12] = refit.ll / N;
    }
    g.v[9] = expq(final_t);
    g.v[10] = final_t;
    auto p = eval(m.y[i], m.factor, final_t);
    Q info[2]{};
    for (int j = 0; j < N; ++j)
      info[j / (N / 2)] += p.mu[j] / (1 + g.v[9] * p.mu[j]);
    g.v[13] = logq(scale(m.y[i], m.factor, 1, g.v[9])) -
              logq(scale(m.y[i], m.factor, 0, g.v[9]));
    g.v[14] = sqrtq(1 / info[0] + 1 / info[1]);
    g.v[15] = g.v[13] / g.v[14];
    // erfc is independent of production GammaQ(1/2,z^2/2).
    g.v[16] = erfcq(fabsq(g.v[15]) / sqrtq(Q(2)));
    g.v[17] = logq(g.v[16]);
    g.v[18] = g.v[19] = 1;
  }
  std::array<int, 12> order{};
  std::iota(order.begin(), order.begin() + G, 0);
  auto pv = [&](int i) {
    return m.gene[i].v[18] == 1 ? m.gene[i].v[16] : Q(1);
  };
  std::stable_sort(order.begin(), order.begin() + G,
                   [&](int a, int b) { return pv(a) < pv(b); });
  Q harmonic = 0;
  for (int i = 1; i <= G; ++i)
    harmonic += Q(1) / i;
  Q bh = 1, by = 1;
  for (int rank = G; rank > 0; --rank) {
    int i = order[rank - 1];
    bh = fminq(bh, pv(i) * G / rank);
    by = fminq(by, pv(i) * G * harmonic / rank);
    m.gene[i].bh = bh;
    m.gene[i].by = by;
  }
  return m;
}
struct Reader {
  std::ifstream in;
  std::string path;
  size_t line = 0;
  Reader(std::string p) : in(p), path(p) { require(bool(in), "native open"); }
  std::string get() {
    std::string s;
    require(bool(std::getline(in, s)), "native EOF");
    ++line;
    if (!s.empty() && s.back() == '\r')
      s.pop_back();
    return s;
  }
  void literal(std::string e) {
    auto a = get();
    if (a != e)
      throw std::runtime_error(path + ":" + std::to_string(line) +
                               ": expected " + e + " got " + a);
  }
  Q number(std::string prefix) {
    auto s = get();
    require(s.starts_with(prefix), "native prefix");
    auto n = s.substr(prefix.size());
    char *end = nullptr;
    Q x = strtoflt128(n.c_str(), &end);
    require(end != n.c_str() && *end == '\0' && !isnanq(x) && !isinfq(x),
            "finite native number");
    return x;
  }
};
void close(Q a, Q e, std::string field, Q rel, Q abs) {
  if (fabsq(a - e) > rel * fabsq(e) + abs) {
    char av[100], ev[100];
    quadmath_snprintf(av, sizeof(av), "%.24Qg", a);
    quadmath_snprintf(ev, sizeof(ev), "%.24Qg", e);
    throw std::runtime_error(field + ": actual=" + av + " independent=" + ev);
  }
}
void probability_close(Q actual, Q expected, const std::string &field) {
  require(actual >= 0 && actual <= 1 && expected >= 0 && expected <= 1,
          "probability bounds");
  if (expected > 0)
    require(actual > 0, "positive reference probability must stay positive");
  // Zero absolute floor prevents a small positive tail/BH/BY being accepted
  // as zero or with an arbitrarily large relative error. Frozen tails are
  // within the finite binary64 range; underflow is not claimed here.
  close(actual, expected, field, q(".0002"), 0);
}
std::vector<std::string> split(std::string s) {
  std::vector<std::string> v;
  std::istringstream in(s);
  std::string x;
  while (std::getline(in, x, '\t'))
    v.push_back(x);
  return v;
}
int integer(Q x) {
  require(floorq(x) == x && x >= -1 && x <= 10000000, "integer status");
  return int(x);
}
void validate(const Model &m, std::string path) {
  Reader r(path);
  r.literal("PREP 0");
  r.literal("ELIGIBLE " + std::to_string(m.eligible));
  r.literal("SHRINK 0");
  std::istringstream work(r.get());
  std::string key, extra;
  int wp = -1, wr = -1;
  require(bool(work >> key >> wp >> wr) && key == "WORK" && !(work >> extra),
          "WORK schema");
  close(r.number("PRIOR_A0 "), m.prior.a0, "prior a0", q(".0002"), q("1e-10"));
  close(r.number("PRIOR_A1 "), m.prior.a1, "prior a1", q(".0002"), q("1e-10"));
  close(r.number("PRIOR_VAR "), m.prior.prior, "prior variance", q(".0002"),
        q("2e-6"));
  for (int j = 0; j < N; ++j) {
    r.literal("SAMPLE " + std::to_string(j));
    close(r.number("FACTOR "), m.factor[j], "factor", q("2e-12"), q("2e-14"));
    close(r.number("OFFSET "), m.offset[j], "offset", q("2e-12"), q("2e-14"));
  }
  int evals = 0, refits = 0;
  std::array<Gene, 12> actual{};
  for (int i = 0; i < G; ++i) {
    r.literal("GENE " + std::to_string(i));
    auto &a = actual[i];
    auto &e = m.gene[i];
    for (int j = 0; j < 12; ++j) {
      a.s[j] = integer(r.number("STATUS " + std::to_string(j) + " "));
      if (j == 2) {
        require(a.s[j] >= 1 && a.s[j] <= 128, "point budget");
        evals += a.s[j];
      } else if (j == 6) {
        if (e.s[11] == 1) {
          require(a.s[j] >= 1 && a.s[j] <= 128, "refit budget");
          refits += a.s[j];
        } else
          require(a.s[j] == -1, "no profile refit evaluations");
      } else
        require(a.s[j] == e.s[j], "gene status/availability");
    }
    for (int j = 0; j < 24; ++j) {
      r.literal("VALUE " + std::to_string(j));
      a.v[j] = r.number("V ");
      Q rel = q(".0002"), abs = q("2e-6");
      if (j == 0) {
        rel = q("2e-12");
        abs = q("2e-12");
      }
      if (e.v[j] == 0 || j == 18 || j == 19 || j == 20 || j == 21)
        rel = abs = 0;
      if (j == 3 || j == 4) {
        rel = q("2e-10");
        abs = q("2e-10");
      }
      if (j == 16) {
        probability_close(a.v[j], e.v[j], "raw probability");
        continue;
      }
      close(a.v[j], e.v[j],
            "gene " + std::to_string(i) + " value " + std::to_string(j), rel,
            abs);
    }
    a.bh = r.number("BH ");
    a.by = r.number("BY ");
    probability_close(a.bh, e.bh, "full family BH");
    probability_close(a.by, e.by, "full family BY");
  }
  require(evals == wp && refits == wr, "actual work sums");
  r.literal("#model=NB2_log_link;dispersion_prior=Gaussian_working;alpha=plug_"
            "in;assumptions_assessed=false;finite_sample_calibrated=false;"
            "global_optimality_assessed=false");
  r.literal("#BH_requirement=valid_p_values_and_independence_or_PRDS;BY_"
            "requirement=valid_p_values;family=all_original_gene_rows;"
            "unavailable_effective_p=1;display_significant_digits=15");
  r.literal(
      "gene_row\tgene_id_hex\tpoint_status\tcurvature_status\tprior_"
      "eligible\trefit_status\trefit_route\tcovariance_status\tcontrast_"
      "status\ttail_status\toutlier\tpoint_available\trefit_available\tp_"
      "available\tlog_p_available\tbase_mean\tpoint_alpha\tgaussian_mapped_log_"
      "alpha\tfinal_alpha\tcontrast\tstandard_error\tz\tp\tlog_p\tBH\tBY");
  for (int i = 0; i < G; ++i) {
    auto f = split(r.get());
    require(f.size() == 26, "TSV width");
    auto &a = actual[i];
    auto &e = m.gene[i];
    require(f[0] == std::to_string(i), "TSV row order");
    require(f[1] == ([](int row) {
              char b[20];
              std::snprintf(b, sizeof(b), "47414141%02x", 65 + row);
              return std::string(b);
            })(i),
            "gene ID");
    const int sc[8]{0, 3, 4, 5, 11, 7, 8, 9};
    for (int k = 0; k < 8; ++k)
      require(f[k + 2] ==
                  std::to_string(k == 2 ? (a.s[4] == 1 ? 1 : 0) : a.s[sc[k]]),
              "TSV statuses");
    bool po = e.s[0] == 0, fi = e.s[5] == 0;
    require(f[10] == std::to_string(e.s[10] == 1 ? 1 : 0), "TSV outlier");
    require(f[11] == (po ? "1" : "0"), "TSV point availability");
    for (int k = 12; k <= 14; ++k)
      require(f[k] == (fi ? "1" : "0"), "TSV final availability");
    const int vc[9]{0, 1, 6, 9, 13, 14, 15, 16, 17};
    for (int k = 0; k < 9; ++k) {
      bool available = k == 0 || (k == 1 ? po : fi);
      if (!available) {
        require(f[15 + k] == "NA", "TSV unavailable must be NA");
        continue;
      }
      char *end = nullptr;
      Q x = strtoflt128(f[15 + k].c_str(), &end);
      require(end != f[15 + k].c_str() && *end == '\0' && !isnanq(x) &&
                  !isinfq(x),
              "TSV finite");
      close(x, a.v[vc[k]], "TSV scientific consistency", 0, 0);
    }
    for (int k = 0; k < 2; ++k) {
      char *end = nullptr;
      Q x = strtoflt128(f[24 + k].c_str(), &end);
      require(end != f[24 + k].c_str() && *end == '\0' && !isnanq(x) &&
                  !isinfq(x),
              "TSV adjustment");
      close(x, k == 0 ? a.bh : a.by, "TSV adjustment consistency", 0, 0);
    }
  }
  r.literal(prior_case && !original_prior_marker ? "GENOMICS_BATCH_PRIOR_PASS"
                                                 : "GENOMICS_BATCH_PASS");
  std::string tail;
  require(!std::getline(r.in, tail), "trailing native rows");
  std::printf("GENOMICS_BATCH_INDEPENDENT_CONCORDANCE_PASS %s\n", path.c_str());
}
} // namespace batch
int main(int argc, char **argv) {
  try {
    int first = 1;
    if (argc > 1 && (std::string(argv[1]) == "--prior" ||
                     std::string(argv[1]) == "--prior-original-marker")) {
      batch::original_prior_marker =
          std::string(argv[1]) == "--prior-original-marker";
      batch::prior_case = true;
      batch::G = 12;
      batch::N = 16;
      batch::upper = 4;
      first = 2;
    }
    auto m = batch::expected();
    std::printf("CASE%d G%d N%d P2 ML eligible=%d\n", batch::prior_case ? 1 : 0,
                batch::G, batch::N, m.eligible);
    row("independent_prior_a0", m.prior.a0);
    row("independent_prior_a1", m.prior.a1);
    row("independent_prior_variance", m.prior.prior);
    for (int i = 0; i < batch::G; ++i) {
      std::printf("INDEPENDENT_GENE %d\n", i);
      for (int j : {0, 1, 2, 3, 5, 9, 13, 14, 15, 16, 17, 22, 23}) {
        auto label = "v" + std::to_string(j);
        row(label.c_str(), m.gene[i].v[j]);
      }
      row("BH", m.gene[i].bh);
      row("BY", m.gene[i].by);
    }
    if (batch::prior_case) {
      int changed = 0;
      for (int i = 0; i < batch::G; ++i)
        if (m.gene[i].s[11] == 1 &&
            fabsq(m.gene[i].v[10] - m.gene[i].v[6]) > batch::q(".001"))
          ++changed;
      require(changed > 0, "genuine profile differs from Gaussian map");
      std::printf("GENUINE_REFIT_DIFFERS_FROM_GAUSSIAN %d\n", changed);
    }
    for (int i = first; i < argc; ++i) {
      batch::validate(m, argv[i]);
    }
    std::puts("GENOMICS_BATCH_INDEPENDENT_CPP_PASS");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "GENOMICS_BATCH_ORACLE_REJECT %s\n", e.what());
    return 2;
  }
}
