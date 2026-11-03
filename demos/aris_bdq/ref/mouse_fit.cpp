// mouse_fit.cpp — C++23. Refits the mouse cellular-lesion (Table S12) and caseum
// (Table S14) data of Karakitsios, Della Pasqua & Dokoumetzidis (2025, BJCP,
// doi:10.1002/bcp.70163) and reports the parameter covariance the paper does not.
//
// Plasma driver: the paper fitted these data with literature plasma models for
// BALB/c and C3HeB/FeJ (Irwin et al. 2016) that are not reproduced in the SI.
// This program uses the mouse population model of Table S5 (Janssen data, typical
// 0.025 kg mouse) as a SURROGATE. Every estimate is therefore conditional on that
// substitution; `pub` mode shows how well the published parameters fit under it.
//
// Fit: least squares on log concentrations (the paper's proportional error model),
// Levenberg–Marquardt in log-parameters, central-difference Jacobian.
// Covariance: s^2 (J'J)^-1 in log-parameters, s^2 = RSS/(n-p).
//
// Modes:  mouse_fit pub      — predictions with the published estimates vs data
//         mouse_fit lesion   — refit rate_mphi, Kp_mphi/(fu fni)     (S12, BALB/c)
//         mouse_fit caseum   — refit rate_fmphi, Kp_fmphi/(fu fni), Kp_caseum, k_cas, h (S14)
//         mouse_fit pde      — caseum as radial diffusion–binding (task 4)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

namespace {

double W(double pH) {  // 1/fni, diprotic base pKa 8.5, 1.1 (see bdq.cpp)
  return 1.0 + std::pow(10.0, 8.5 - pH) + std::pow(10.0, 9.6 - 2.0 * pH);
}

struct Mouse {
  // Table S5 (typical 0.025 kg mouse)
  double F = 0.2, ka = 0.13, CL = 0.0089, V1 = 0.00033, Q2 = 0.0024, V2 = 0.14, Q3 = 0.006, V3 = 0.0094;
  // Table S3 (mice) and S4
  double Q = 840.6, SA = 500.0, Vpb = 0.0314, Vew = 0.0192, Vlys = 0.0015, Viw = 0.098;
  double pHpl = 7.39, pHew = 7.4, pHiw = 7.36;
  double fu_pls = 7.2e-4, fu_ew = 1.1e-3, BP = 1.0;
  double perm = 79.1;           // strain: BALB/c 79.1, Kramnik 29.6 cm/h (main text)
  double kplys_fu = 3.48e6;     // S8
  // lesion / caseum (S11, S13)
  double rate_m = 1.43e5, R_m = 5.68e6;
  double rate_f = 5.78e3, R_f = 1.44e6, Kp_cas = 0.71, kcas = 0.25, h = 0.75, tau = 1e-3;
  double ratio1 = 1.67, flys = 0.078;
  double weight = 0.025;        // kg
};

// Irwin et al. 2016 (ACS Infect Dis 2:251), Supplemental Table 2: population GM of the
// three-compartment plasma model, per kg (apparent, F = 1). The paper's own plasma
// driver for these two experiments. Volumes/clearances in L/kg, L/h/kg; dose per kg.
Mouse balbc() {
  Mouse m;
  m.F = 1.0; m.weight = 1.0;
  m.ka = 5.39; m.CL = 0.71; m.V1 = 6.27; m.Q2 = 3.63; m.V2 = 6.46; m.Q3 = 0.41; m.V3 = 41.86;
  m.perm = 79.1;  // BALB/c cellular permeability (main text)
  return m;
}
Mouse c3heb() {
  Mouse m;
  m.F = 1.0; m.weight = 1.0;
  m.ka = 4.32; m.CL = 0.71; m.V1 = 7.90; m.Q2 = 5.37; m.V2 = 7.51; m.Q3 = 0.59; m.V3 = 61.71;
  m.perm = 29.6;  // Kramnik (C3HeB/FeJ) cellular permeability (main text)
  return m;
}

constexpr int N = 14;
enum { DEP = 0, CEN, P1, P2, CUEW, CIW, CM, C1, C2, C3, C4, C5, C6, ELIM };

using Mat = std::array<std::array<double, N>, N>;
using Vec = std::array<double, N>;

void buildA(const Mouse& m, double t, Mat& A) {
  for (auto& r : A) r.fill(0.0);
  double k10 = m.CL / m.V1, k12 = m.Q2 / m.V1, k21 = m.Q2 / m.V2, k13 = m.Q3 / m.V1, k31 = m.Q3 / m.V3;
  A[DEP][DEP] = -m.ka;
  A[CEN][DEP] = m.ka;
  A[CEN][CEN] = -(k10 + k12 + k13); A[CEN][P1] = k21; A[CEN][P2] = k31;
  A[P1][CEN] = k12; A[P1][P1] = -k21; A[P2][CEN] = k13; A[P2][P2] = -k31;
  A[ELIM][CEN] = k10;
  double Kewb = m.fu_pls * (1.0 / W(m.pHpl)) / (m.fu_ew * (1.0 / W(m.pHew)) * m.BP);
  double capEW = (m.Vew + m.Vpb / Kewb) / m.fu_ew;
  double PS = m.perm * m.SA, Wew = W(m.pHew), aiw = W(m.pHiw);
  // plasma conc in mg/L = amount(mg)/V1(L); volumes below in mL, consistent ratios
  A[CUEW][CEN] = m.Q * m.BP / m.V1 / capEW;
  A[CUEW][CUEW] = -(m.Q / (m.fu_ew * Kewb) + PS / Wew) / capEW;
  A[CUEW][CIW] = PS / aiw / capEW;
  double capIW = m.Viw + m.kplys_fu * m.Vlys;
  A[CIW][CUEW] = PS / (Wew * capIW);
  A[CIW][CIW] = -PS / (aiw * capIW);
  double fIW = m.Viw / (m.Vpb + m.Vew + m.Viw + m.Vlys), r = m.flys / fIW;
  double denm = (1.0 + m.R_m * r) / (1.0 + r);
  A[CM][CUEW] = m.rate_m / Wew;
  A[CM][CM] = -m.rate_m / denm;
  double denf = (1.0 + m.R_f * r) / (1.0 + r);
  double g = 1.0 / (1.0 + m.ratio1 / m.Kp_cas);
  double kc = m.kcas / std::pow(t + m.tau, m.h);
  A[C1][CUEW] = m.rate_f * m.ratio1 * g / Wew;
  A[C1][C1] = -m.rate_f * m.ratio1 * g / (denf * m.Kp_cas) - kc * g;
  A[C1][C2] = kc * g;
  for (int n = C2; n <= C5; ++n) { A[n][n - 1] = kc; A[n][n] = -2 * kc; A[n][n + 1] = kc; }
  A[C6][C5] = kc; A[C6][C6] = -kc;
}

void solve(const Mat& A, double c, Vec& b) {
  Mat M;
  for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) M[i][j] = (i == j) - c * A[i][j];
  for (int k = 0; k < N; ++k) {
    int p = k;
    for (int i = k + 1; i < N; ++i) if (std::abs(M[i][k]) > std::abs(M[p][k])) p = i;
    std::swap(M[k], M[p]); std::swap(b[k], b[p]);
    for (int i = k + 1; i < N; ++i) {
      double f = M[i][k] / M[k][k];
      if (f == 0) continue;
      for (int j = k; j < N; ++j) M[i][j] -= f * M[k][j];
      b[i] -= f * b[k];
    }
  }
  for (int i = N - 1; i >= 0; --i) {
    double s = b[i];
    for (int j = i + 1; j < N; ++j) s -= M[i][j] * b[j];
    b[i] = s / M[i][i];
  }
}

struct Obs { double t; int what; double y; };  // what: 0 lesion, 1 outer caseum, 2..7 cas1..cas6

// simulate and return predictions at observation times (all in ug/g == mg/L)
std::vector<double> predict(const Mouse& m, int ndoses, const std::vector<Obs>& obs, double dt = 0.05) {
  std::vector<double> times;
  for (auto& o : obs) times.push_back(o.t);
  double tend = *std::max_element(times.begin(), times.end());
  Vec x{};
  const double gam = 2.0 - std::sqrt(2.0);
  std::vector<double> out(obs.size(), 0.0);
  long nsteps = std::lround(tend / dt);
  Mat A0, Ag, A1;
  double dose = 25.0 * m.weight;  // mg
  for (long n = 0; n <= nsteps; ++n) {
    double t = n * dt;
    for (int d = 0; d < ndoses; ++d)
      if (std::abs(t - 24.0 * d) < 1e-9) x[DEP] += m.F * dose;
    for (size_t i = 0; i < obs.size(); ++i) {
      if (std::abs(t - obs[i].t) < 0.5 * dt) {
        if (obs[i].what == 0) out[i] = x[CM];
        else if (obs[i].what == 1) out[i] = x[C1] / m.Kp_cas;
        else out[i] = x[C1 + obs[i].what - 2];
      }
    }
    if (n == nsteps) break;
    buildA(m, t, A0); buildA(m, t + gam * dt, Ag); buildA(m, t + dt, A1);
    Vec f0{};
    for (int i = 0; i < N; ++i) { double s = 0; for (int j = 0; j < N; ++j) s += A0[i][j] * x[j]; f0[i] = s; }
    Vec y{};
    for (int i = 0; i < N; ++i) y[i] = x[i] + 0.5 * gam * dt * f0[i];
    solve(Ag, 0.5 * gam * dt, y);
    Vec z{};
    double cy = 1.0 / (gam * (2.0 - gam)), cx = (1.0 - gam) * (1.0 - gam) / (gam * (2.0 - gam));
    for (int i = 0; i < N; ++i) z[i] = cy * y[i] - cx * x[i];
    solve(A1, (1.0 - gam) / (2.0 - gam) * dt, z);
    x = z;
  }
  return out;
}

// ---- data
std::vector<Obs> lesion_data() {  // Table S12, BALB/c, 25 mg/kg single oral; DVID 2 = cellular lesion
  return {{0.5, 0, 8.888889}, {3, 0, 26}, {8, 0, 31.77778}, {24, 0, 11.54782},
          {48, 0, 3.805022}, {72, 0, 2.323119}, {96, 0, 1.604564}, {168, 0, 0.979651}};
}
std::vector<Obs> caseum_data() {  // Table S14, Kramnik; Regr 2 = outer, 3..8 = cas1..cas6
  return {{5, 1, 1.30}, {24, 1, 2.62},
          {5, 2, 1.19}, {5, 3, 0.374}, {5, 4, 0.0196}, {5, 6, 0.00158},
          {24, 2, 2.19}, {24, 3, 0.179}, {24, 4, 0.0664}, {24, 5, 0.0187}, {24, 6, 0.00282},
          {389, 1, 6.78}, {408, 1, 15.4},
          {389, 2, 5.65}, {389, 3, 2.14}, {389, 4, 1.20}, {389, 5, 0.699}, {389, 6, 0.433},
          {408, 2, 11.9}, {408, 3, 3.49}, {408, 4, 2.28}, {408, 5, 1.27}, {408, 6, 0.577}, {408, 7, 0.241}};
}
// Caveat: in S14 the 5 h and 24 h single-dose and the 389/408 h multiple-dose rows
// belong to different animals (IDs 1-4); all share the typical-animal prediction here.
// The single-dose rows come from a 17-dose design file (ADDL=16) — the 5/24 h points
// precede the second dose, so they are single-dose observations either way.

struct Fit {
  std::vector<double> est, se;
  std::vector<std::vector<double>> corr;
  double rss, s;
  int n, p;
};

using Setter = void (*)(Mouse&, const std::vector<double>&);

Fit lm(Mouse base, int ndoses, const std::vector<Obs>& obs, std::vector<double> theta0, Setter set) {
  int p = theta0.size(), n = obs.size();
  std::vector<double> th(p);
  for (int i = 0; i < p; ++i) th[i] = std::log(theta0[i]);
  auto resid = [&](const std::vector<double>& lt) {
    Mouse m = base;
    std::vector<double> v(p);
    for (int i = 0; i < p; ++i) v[i] = std::exp(lt[i]);
    set(m, v);
    auto pr = predict(m, ndoses, obs);
    std::vector<double> r(n);
    for (int i = 0; i < n; ++i) r[i] = std::log(std::max(pr[i], 1e-30)) - std::log(obs[i].y);
    return r;
  };
  auto rss = [](const std::vector<double>& r) { double s = 0; for (double a : r) s += a * a; return s; };
  double lam = 1e-2;
  std::vector<double> r = resid(th);
  double cur = rss(r);
  std::vector<std::vector<double>> J(n, std::vector<double>(p));
  for (int it = 0; it < 60; ++it) {
    for (int j = 0; j < p; ++j) {
      auto tp = th, tm = th;
      tp[j] += 1e-4; tm[j] -= 1e-4;
      auto rp = resid(tp), rm = resid(tm);
      for (int i = 0; i < n; ++i) J[i][j] = (rp[i] - rm[i]) / 2e-4;
    }
    std::vector<std::vector<double>> H(p, std::vector<double>(p, 0));
    std::vector<double> g(p, 0);
    for (int a = 0; a < p; ++a) {
      for (int b = 0; b < p; ++b) for (int i = 0; i < n; ++i) H[a][b] += J[i][a] * J[i][b];
      for (int i = 0; i < n; ++i) g[a] += J[i][a] * r[i];
    }
    bool improved = false;
    for (int tries = 0; tries < 12 && !improved; ++tries) {
      // solve (H + lam diag H) d = -g  (small p: Gaussian elimination)
      std::vector<std::vector<double>> M = H;
      std::vector<double> d(p);
      for (int a = 0; a < p; ++a) { M[a][a] *= (1 + lam); d[a] = -g[a]; }
      for (int k = 0; k < p; ++k) {
        int piv = k;
        for (int i = k + 1; i < p; ++i) if (std::abs(M[i][k]) > std::abs(M[piv][k])) piv = i;
        std::swap(M[k], M[piv]); std::swap(d[k], d[piv]);
        for (int i = k + 1; i < p; ++i) { double f = M[i][k] / M[k][k]; for (int j = k; j < p; ++j) M[i][j] -= f * M[k][j]; d[i] -= f * d[k]; }
      }
      for (int i = p - 1; i >= 0; --i) { double s = d[i]; for (int j = i + 1; j < p; ++j) s -= M[i][j] * d[j]; d[i] = s / M[i][i]; }
      auto tn = th;
      for (int a = 0; a < p; ++a) tn[a] += std::clamp(d[a], -2.0, 2.0);
      auto rn = resid(tn);
      double nv = rss(rn);
      if (nv < cur) { th = tn; r = rn; double rel = (cur - nv) / cur; cur = nv; lam = std::max(lam / 5, 1e-9); improved = true;
        if (rel < 1e-10) it = 1000; }
      else lam *= 8;
    }
    if (!improved) break;
  }
  // covariance in log-parameters
  for (int j = 0; j < p; ++j) {
    auto tp = th, tm = th;
    tp[j] += 1e-4; tm[j] -= 1e-4;
    auto rp = resid(tp), rm = resid(tm);
    for (int i = 0; i < n; ++i) J[i][j] = (rp[i] - rm[i]) / 2e-4;
  }
  std::vector<std::vector<double>> H(p, std::vector<double>(p, 0)), C(p, std::vector<double>(p, 0));
  for (int a = 0; a < p; ++a) for (int b = 0; b < p; ++b) for (int i = 0; i < n; ++i) H[a][b] += J[i][a] * J[i][b];
  // invert H (Gauss-Jordan)
  std::vector<std::vector<double>> I(p, std::vector<double>(p, 0));
  for (int a = 0; a < p; ++a) I[a][a] = 1;
  auto Hc = H;
  for (int k = 0; k < p; ++k) {
    int piv = k;
    for (int i = k + 1; i < p; ++i) if (std::abs(Hc[i][k]) > std::abs(Hc[piv][k])) piv = i;
    std::swap(Hc[k], Hc[piv]); std::swap(I[k], I[piv]);
    double d = Hc[k][k];
    for (int j = 0; j < p; ++j) { Hc[k][j] /= d; I[k][j] /= d; }
    for (int i = 0; i < p; ++i) if (i != k) { double f = Hc[i][k]; for (int j = 0; j < p; ++j) { Hc[i][j] -= f * Hc[k][j]; I[i][j] -= f * I[k][j]; } }
  }
  Fit F;
  F.n = n; F.p = p; F.rss = cur; F.s = std::sqrt(cur / std::max(1, n - p));
  for (int a = 0; a < p; ++a) for (int b = 0; b < p; ++b) C[a][b] = F.s * F.s * I[a][b];
  F.corr.assign(p, std::vector<double>(p));
  for (int a = 0; a < p; ++a) {
    F.est.push_back(std::exp(th[a]));
    F.se.push_back(std::exp(th[a]) * std::sqrt(C[a][a]));  // delta method
    for (int b = 0; b < p; ++b) F.corr[a][b] = C[a][b] / std::sqrt(C[a][a] * C[b][b]);
  }
  // condition number of H (power iteration bounds would do; report ratio of diag of inverse)
  return F;
}

void report(const char* title, const Fit& F, const std::vector<std::string>& names, const std::vector<double>& pub,
            const std::vector<double>& pub_se) {
  std::printf("%s\nn=%d p=%d RSS(log)=%.4f s=%.4f (paper's proportional b for comparison)\n", title, F.n, F.p, F.rss, F.s);
  std::printf("%-22s %14s %12s %8s | %14s %12s %8s\n", "parameter", "refit", "SE", "RSE%", "published", "SE", "RSE%");
  for (int a = 0; a < F.p; ++a)
    std::printf("%-22s %14.5g %12.4g %8.1f | %14.5g %12.4g %8.1f\n", names[a].c_str(), F.est[a], F.se[a],
                100 * F.se[a] / F.est[a], pub[a], pub_se[a], 100 * pub_se[a] / pub[a]);
  std::printf("correlation matrix (log-parameters):\n");
  for (int a = 0; a < F.p; ++a) {
    std::printf("  %-20s", names[a].c_str());
    for (int b = 0; b < F.p; ++b) std::printf(" %7.3f", F.corr[a][b]);
    std::printf("\n");
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "pub";
  if (mode == "pub") {
    Mouse b = balbc();
    auto ld = lesion_data();
    auto pl = predict(b, 1, ld);
    double rss = 0;
    std::printf("LESION (BALB/c) published params, Irwin BALB/c plasma\n  t      obs       pred\n");
    for (size_t i = 0; i < ld.size(); ++i) { std::printf("  %-6g %-9.4g %-9.4g\n", ld[i].t, ld[i].y, pl[i]); rss += std::pow(std::log(pl[i] / ld[i].y), 2); }
    std::printf("  RSS(log) = %.4f\n", rss);
    Mouse k = c3heb();
    auto cd = caseum_data();
    auto pc = predict(k, 17, cd);
    rss = 0;
    std::printf("CASEUM (Kramnik) published params, Irwin C3HeB plasma\n  t      comp  obs       pred\n");
    for (size_t i = 0; i < cd.size(); ++i) { std::printf("  %-6g %-5d %-9.4g %-9.4g\n", cd[i].t, cd[i].what, cd[i].y, pc[i]); rss += std::pow(std::log(pc[i] / cd[i].y), 2); }
    std::printf("  RSS(log) = %.4f\n", rss);
    return 0;
  }
  if (mode == "nca") {  // plasma drivers vs Irwin 2016 Table 3 NCA (BALB/c: Cmax 2.93, AUC168 33.7, t1/2 89.3)
   for (int which = 0; which < 3; ++which) {
    Mouse b = which == 0 ? Mouse{} : which == 1 ? balbc() : c3heb();
    const char* nm = which == 0 ? "Table S5 (Janssen CD1)" : which == 1 ? "Irwin BALB/c" : "Irwin C3HeB/FeJ";
    double dt = 0.05, auc = 0, cmax = 0, tmax = 0, prev = 0;
    Vec x{}; x[DEP] = b.F * 25.0 * b.weight;
    Mat A0, Ag, A1; const double gam = 2.0 - std::sqrt(2.0);
    double c72 = 0, c168 = 0;
    for (int n = 0; n < 3360; ++n) {
      double t = n * dt;
      buildA(b, t, A0); buildA(b, t + gam * dt, Ag); buildA(b, t + dt, A1);
      Vec f0{}; for (int i = 0; i < N; ++i) { double s = 0; for (int j = 0; j < N; ++j) s += A0[i][j] * x[j]; f0[i] = s; }
      Vec y{}; for (int i = 0; i < N; ++i) y[i] = x[i] + 0.5 * gam * dt * f0[i];
      solve(Ag, 0.5 * gam * dt, y);
      Vec z{}; double cy = 1.0 / (gam * (2.0 - gam)), cx = (1.0 - gam) * (1.0 - gam) / (gam * (2.0 - gam));
      for (int i = 0; i < N; ++i) z[i] = cy * y[i] - cx * x[i];
      solve(A1, (1.0 - gam) / (2.0 - gam) * dt, z); x = z;
      double c = x[CEN] / b.V1;
      auc += 0.5 * (c + prev) * dt; prev = c;
      if (c > cmax) { cmax = c; tmax = t + dt; }
      if (std::abs(t + dt - 72) < 1e-6) c72 = c;
      if (std::abs(t + dt - 168) < 1e-6) c168 = c;
    }
    std::printf("%-24s 25 mg/kg: Cmax %.3f ug/mL at %.2f h, AUC0-168 %.2f ug.h/mL, t1/2(72-168) %.1f h\n",
                nm, cmax, tmax, auc, std::log(2.0) * 96.0 / std::log(c72 / c168));
   }
    std::printf("Irwin 2016 Table 3 NCA:  BALB/c Cmax 2.93 (0.5 h), AUC 33.7, t1/2 89.3 | C3HeB Cmax 2.72 (4 h), AUC 35.9, t1/2 116.4\n");
    return 0;
  }
  if (mode == "lesion") {
    Mouse b = balbc();
    auto F = lm(b, 1, lesion_data(), {1.43e5, 5.68e6},
                [](Mouse& m, const std::vector<double>& v) { m.rate_m = v[0]; m.R_m = v[1]; });
    report("LESION refit (S12, BALB/c, Irwin 2016 plasma)", F, {"rate_mphi (1/h)", "Kp_mphi/(fu fni)"},
           {1.43e5, 5.68e6}, {1.12e4, 2.03e5});
    return 0;
  }
  if (mode == "caseum") {
    Mouse k = c3heb();
    auto F = lm(k, 17, caseum_data(), {5.78e3, 1.44e6, 0.71, 0.25, 0.75},
                [](Mouse& m, const std::vector<double>& v) {
                  m.rate_f = v[0]; m.R_f = v[1]; m.Kp_cas = v[2]; m.kcas = v[3]; m.h = std::min(v[4], 1.0);
                });
    report("CASEUM refit (S14, Kramnik, Irwin 2016 plasma)", F,
           {"rate_fmphi (1/h)", "Kp_fmphi/(fu fni)", "Kp_caseum", "k_cas", "h"},
           {5.78e3, 1.44e6, 0.71, 0.25, 0.75}, {1.48e3, 3.95e5, 0.21, 0.048, 0.05});
    return 0;
  }
  return 1;
}
