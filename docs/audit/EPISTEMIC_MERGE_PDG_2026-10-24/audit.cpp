// Retrospective audit of the ADR-016 default `meet` on PDG 2025 computed averages.
// Implements PROTOCOL.md exactly; post-hoc analyses are labelled as such.
// Build: c++ -std=c++2b -O2 -Wall -Wextra audit.cpp -o audit
// Run:   ./audit inputs.csv results.csv
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---- CSV (quoted fields allowed) ------------------------------------------
std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  bool q = false;
  for (char c : line) {
    if (c == '"') q = !q;
    else if (c == ',' && !q) { out.push_back(cur); cur.clear(); }
    else cur.push_back(c);
  }
  out.push_back(cur);
  return out;
}

// ---- Regularised upper incomplete gamma Q(a, x) ----------------------------
double gamma_p_series(double a, double x) {
  double sum = 1.0 / a, term = sum;
  for (int n = 1; n < 10000; ++n) {
    term *= x / (a + n);
    sum += term;
    if (std::fabs(term) < std::fabs(sum) * 1e-16) break;
  }
  return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}
double gamma_q_cf(double a, double x) {  // modified Lentz
  const double tiny = 1e-300;
  double b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
  for (int i = 1; i < 10000; ++i) {
    const double an = -i * (i - a);
    b += 2.0;
    d = an * d + b; if (std::fabs(d) < tiny) d = tiny;
    c = b + an / c; if (std::fabs(c) < tiny) c = tiny;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1.0) < 1e-16) break;
  }
  return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}
double gamma_q(double a, double x) {
  if (x <= 0) return 1.0;
  return x < a + 1.0 ? 1.0 - gamma_p_series(a, x) : gamma_q_cf(a, x);
}
// P(chi2_k >= x)
double chi2_sf(double x, int k) { return gamma_q(0.5 * k, 0.5 * x); }

struct Wilson { double lo, hi; };
Wilson wilson(long k, long n) {
  const double z = 1.959963984540054, p = double(k) / n, z2 = z * z;
  const double c = (p + z2 / (2 * n)) / (1 + z2 / n);
  const double h = z * std::sqrt(p * (1 - p) / n + z2 / (4.0 * n * n)) / (1 + z2 / n);
  return {c - h, c + h};
}

struct In { double x, u, asym; bool pdg_ref; };
struct Avg {
  std::string pdgid;
  double pdg_s = 1.0;
  std::vector<In> in;
  // results
  double chi2 = 0, rb = 0, p = 1;
  bool refuse = false, pair_en = false, asym = false, has_pdg_ref = false;
};

double quantile(std::vector<double> v, double q) {
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  const double pos = q * (v.size() - 1);
  const size_t i = size_t(pos);
  const double f = pos - i;
  return i + 1 < v.size() ? v[i] * (1 - f) + v[i + 1] * f : v[i];
}

struct Tally { long n = 0, k = 0; };
void report(const char* label, const Tally& t) {
  const Wilson w = wilson(t.k, t.n);
  std::printf("%-58s %5ld / %5ld = %6.2f %%  [95%% CI %5.2f, %5.2f]\n", label, t.k, t.n,
              100.0 * t.k / t.n, 100.0 * w.lo, 100.0 * w.hi);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: audit inputs.csv results.csv\n"); return 2; }
  std::ifstream f(argv[1]);
  std::string line;
  std::getline(f, line);  // header
  std::map<long, Avg> avgs;
  std::vector<long> order;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const auto c = split_csv(line);
    const long id = std::stol(c[0]);
    auto [it, fresh] = avgs.try_emplace(id);
    if (fresh) { order.push_back(id); it->second.pdgid = c[1]; it->second.pdg_s = c[5].empty() ? 1.0 : std::stod(c[5]); }
    const double ep = std::stod(c[7]), en = std::stod(c[8]);
    In in{std::stod(c[6]), 0.5 * (ep + en), std::fabs(ep - en) / (ep + en), c[9].rfind("PDG ", 0) == 0};
    it->second.in.push_back(in);
  }

  const double p2 = chi2_sf(4.0, 1);  // = P(chi2_1 >= 4)
  std::printf("check: p2 = P(chi2_1 >= 4) = %.6f  (erfc(sqrt 2) = %.6f)\n", p2, std::erfc(std::sqrt(2.0)));
  std::printf("check: P(chi2_10 >= 18.307) = %.5f (expected 0.05000)\n\n", chi2_sf(18.307, 10));

  std::ofstream out(argv[2]);
  out << "pdgid,n,chi2,birge,p,refuse,pair_en_gt1,pdg_s,asym_gt_0.2,has_pdg_ref\n";

  Tally prim, pair, sensA, sensB, posthoc;
  long disc_total = 0, disc_explained = 0;   // post hoc (d)
  std::vector<double> rel_diff_rule;          // post hoc (e)
  long tab[2][2] = {{0, 0}, {0, 0}};  // [refuse][pdg_inflated]
  std::vector<double> s_refused, rel_diff;
  long total_avgs = long(avgs.size()), excluded_n1 = 0;

  for (long id : order) {
    Avg& a = avgs[id];
    const int n = int(a.in.size());
    if (n < 2) { ++excluded_n1; continue; }
    double sw = 0, swx = 0;
    for (const In& i : a.in) { const double w = 1.0 / (i.u * i.u); sw += w; swx += w * i.x; }
    const double mean = swx / sw;
    for (const In& i : a.in) {
      a.chi2 += std::pow((i.x - mean) / i.u, 2);
      a.asym = a.asym || i.asym > 0.2;
      a.has_pdg_ref = a.has_pdg_ref || i.pdg_ref;
    }
    a.rb = std::sqrt(a.chi2 / (n - 1));
    a.p = chi2_sf(a.chi2, n - 1);
    a.refuse = a.p < p2;
    for (int i = 0; i < n && !a.pair_en; ++i)
      for (int j = i + 1; j < n; ++j) {
        const double en = std::fabs(a.in[i].x - a.in[j].x) /
                          (2.0 * std::sqrt(a.in[i].u * a.in[i].u + a.in[j].u * a.in[j].u));
        if (en > 1.0) { a.pair_en = true; break; }
      }
    const bool inflated = a.pdg_s > 1.0 + 1e-9;

    // Post hoc: the PDG rule for S excludes inputs with u_i > 3 sqrt(N) u_mean.
    const double u_mean = 1.0 / std::sqrt(sw), cut = 3.0 * std::sqrt(double(n)) * u_mean;
    double sw2 = 0, swx2 = 0; int n2 = 0;
    for (const In& i : a.in) if (i.u <= cut) { const double w = 1.0 / (i.u * i.u); sw2 += w; swx2 += w * i.x; ++n2; }
    double s_rule = 1.0;
    if (n2 >= 2) {
      const double m2 = swx2 / sw2; double c2 = 0;
      for (const In& i : a.in) if (i.u <= cut) c2 += std::pow((i.x - m2) / i.u, 2);
      s_rule = std::max(1.0, std::sqrt(c2 / (n2 - 1)));
    }
    if (a.refuse && !inflated) { ++disc_total; disc_explained += (s_rule <= 1.0 + 1e-9); }
    if (inflated) rel_diff_rule.push_back(std::fabs(s_rule - a.pdg_s) / a.pdg_s);

    ++prim.n; prim.k += a.refuse;
    ++pair.n; pair.k += a.pair_en;
    tab[a.refuse][inflated]++;
    if (a.refuse) s_refused.push_back(a.pdg_s);
    if (inflated) rel_diff.push_back(std::fabs(a.rb - a.pdg_s) / a.pdg_s);
    if (!a.asym) { ++sensA.n; sensA.k += a.refuse; }
    if (n >= 3) { ++sensB.n; sensB.k += a.refuse; }
    if (!a.has_pdg_ref) { ++posthoc.n; posthoc.k += a.refuse; }

    char buf[512];
    std::snprintf(buf, sizeof buf, "%s,%d,%.6g,%.6g,%.6g,%d,%d,%.6g,%d,%d\n", a.pdgid.c_str(), n,
                  a.chi2, a.rb, a.p, a.refuse, a.pair_en, a.pdg_s, a.asym, a.has_pdg_ref);
    out << buf;
  }

  std::printf("averages with >= 1 eligible input: %ld; excluded (N = 1): %ld; included (N >= 2): %ld\n\n",
              total_avgs, excluded_n1, prim.n);
  std::printf("PRIMARY\n");
  report("refused by default meet (chi2 tail < p2)", prim);
  std::printf("\nSECONDARY\n");
  report("(1) at least one input pair with E_n > 1", pair);
  std::printf("(2) rule x PDG inflation (S > 1):\n");
  std::printf("      %-22s %10s %10s\n", "", "PDG S>1", "PDG S=1");
  std::printf("      %-22s %10ld %10ld\n", "refused", tab[1][1], tab[1][0]);
  std::printf("      %-22s %10ld %10ld\n", "accepted", tab[0][1], tab[0][0]);
  std::printf("(3) PDG S among refused: median %.3f, IQR [%.3f, %.3f], max %.3f (n = %zu)\n",
              quantile(s_refused, 0.5), quantile(s_refused, 0.25), quantile(s_refused, 0.75),
              s_refused.empty() ? NAN : *std::max_element(s_refused.begin(), s_refused.end()),
              s_refused.size());
  std::printf("(4) validation, PDG S > 1: median |R_B - S| / S = %.4f, IQR [%.4f, %.4f] (n = %zu)\n",
              quantile(rel_diff, 0.5), quantile(rel_diff, 0.25), quantile(rel_diff, 0.75), rel_diff.size());
  std::printf("\nSENSITIVITY\n");
  report("(a) excluding input asymmetry > 0.2", sensA);
  report("(b) N >= 3 only", sensB);
  std::printf("\nPOST HOC (not in protocol)\n");
  report("(c) excluding averages with a PDG-derived input", posthoc);
  std::printf("(d) refused but PDG S = 1, explained by the PDG exclusion rule (S_rule = 1): %ld / %ld\n",
              disc_explained, disc_total);
  std::printf("(e) validation with the PDG exclusion rule, PDG S > 1: median |S_rule - S| / S = %.4f, IQR [%.4f, %.4f]\n",
              quantile(rel_diff_rule, 0.5), quantile(rel_diff_rule, 0.25), quantile(rel_diff_rule, 0.75));
  return 0;
}
