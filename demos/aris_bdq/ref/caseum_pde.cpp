// caseum_pde.cpp — C++23. Is the fractal rate needed?
//
// Karakitsios et al. (2025) model drug penetration into the caseum of C3HeB/FeJ mice
// as a chain of six 300-µm compartments with a time-dependent rate
// k(t) = k_cas / (t + tau)^h (fractal kinetics, h = 0.75). This program fits, to the
// same LCM data (Table S14), two models that share the plasma (Irwin 2016, C3HeB/FeJ),
// lung and foamy-macrophage parts of the published model and differ only in the caseum:
//
//   A  fractal: 6 compartments, k(t) = k_cas/(t+tau)^h            (5 parameters)
//   B  Fickian: slab diffusion, constant D, 36 cells of 50 µm,
//      observations = mean of the 6 cells of each 300-µm bin       (4 parameters)
//
// Both: outer caseum in instant equilibrium with the first caseum element
// (C_first = Kp_caseum · C_outer), outer-caseum volume fixed as in the paper
// (1.67 × the volume of the first 300 µm). Slab geometry, as the paper's
// equal-volume ratios imply. Fit: least squares on log concentrations, LM.
// Comparison: RSS, AIC = n ln(RSS/n) + 2p, and the residual pattern by distance.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

double W(double pH) { return 1.0 + std::pow(10.0, 8.5 - pH) + std::pow(10.0, 9.6 - 2.0 * pH); }

struct P {
  // Irwin 2016 Supplemental Table 2, C3HeB/FeJ, per kg
  double ka = 4.32, CL = 0.71, V1 = 7.90, Q2 = 5.37, V2 = 7.51, Q3 = 0.59, V3 = 61.71;
  // mouse lung (Tables S3, S4, S8; Kramnik permeability)
  double Q = 840.6, SA = 500.0, Vpb = 0.0314, Vew = 0.0192, Vlys = 0.0015, Viw = 0.098;
  double fu_pls = 7.2e-4, fu_ew = 1.1e-3, perm = 29.6, kplys_fu = 3.48e6;
  // caseum
  double rate_f = 5.78e3, R_f = 1.44e6, Kp_cas = 0.71, kcas = 0.25, h = 0.75, tau = 1e-3, D = 1e-4;
  double ratio1_300 = 1.67, flys = 0.078;
  bool fractal = true;
  int cells = 6;           // 6 (A) or 36 (B)
  double dx_um = 300.0;    // 300 (A) or 50 (B)
};

struct Obs { double t; int what; double y; };  // what: 1 outer, 2..7 bins cas1..cas6

std::vector<Obs> data() {  // Table S14 (Regr 2 = outer, 3..8 = bins 1..6)
  return {{5, 1, 1.30}, {24, 1, 2.62}, {5, 2, 1.19}, {5, 3, 0.374}, {5, 4, 0.0196}, {5, 6, 0.00158},
          {24, 2, 2.19}, {24, 3, 0.179}, {24, 4, 0.0664}, {24, 5, 0.0187}, {24, 6, 0.00282},
          {389, 1, 6.78}, {408, 1, 15.4}, {389, 2, 5.65}, {389, 3, 2.14}, {389, 4, 1.20}, {389, 5, 0.699},
          {389, 6, 0.433}, {408, 2, 11.9}, {408, 3, 3.49}, {408, 4, 2.28}, {408, 5, 1.27}, {408, 6, 0.577},
          {408, 7, 0.241}};
}

using Mat = std::vector<std::vector<double>>;

// state: 0 DEP, 1 CEN, 2 P1, 3 P2, 4 CuEW, 5 CIW, 6.. caseum cells (cell 0 lumped with outer)
void buildA(const P& p, double t, Mat& A) {
  int n = 6 + p.cells;
  for (auto& r : A) std::fill(r.begin(), r.end(), 0.0);
  double k10 = p.CL / p.V1, k12 = p.Q2 / p.V1, k21 = p.Q2 / p.V2, k13 = p.Q3 / p.V1, k31 = p.Q3 / p.V3;
  A[0][0] = -p.ka; A[1][0] = p.ka;
  A[1][1] = -(k10 + k12 + k13); A[1][2] = k21; A[1][3] = k31;
  A[2][1] = k12; A[2][2] = -k21; A[3][1] = k13; A[3][3] = -k31;
  double Kewb = p.fu_pls * (1 / W(7.39)) / (p.fu_ew * (1 / W(7.4)));
  double capEW = (p.Vew + p.Vpb / Kewb) / p.fu_ew, PS = p.perm * p.SA, Wew = W(7.4), aiw = W(7.36);
  A[4][1] = p.Q / p.V1 / capEW;
  A[4][4] = -(p.Q / (p.fu_ew * Kewb) + PS / Wew) / capEW;
  A[4][5] = PS / aiw / capEW;
  double capIW = p.Viw + p.kplys_fu * p.Vlys;
  A[5][4] = PS / (Wew * capIW); A[5][5] = -PS / (aiw * capIW);
  double fIW = p.Viw / (p.Vpb + p.Vew + p.Viw + p.Vlys), r = p.flys / fIW;
  double denf = (1 + p.R_f * r) / (1 + r);
  double ratio1 = p.ratio1_300 * (300.0 / p.dx_um);   // outer volume / first-cell volume
  double g = 1.0 / (1.0 + ratio1 / p.Kp_cas);
  double k = p.fractal ? p.kcas / std::pow(t + p.tau, p.h) : p.D / std::pow(p.dx_um * 1e-4, 2);  // D in cm^2/h
  int c0 = 6, cl = 6 + p.cells - 1;
  A[c0][4] = p.rate_f * ratio1 * g / Wew;
  A[c0][c0] = -p.rate_f * ratio1 * g / (denf * p.Kp_cas) - k * g;
  A[c0][c0 + 1] = k * g;
  for (int i = c0 + 1; i < cl; ++i) { A[i][i - 1] = k; A[i][i] = -2 * k; A[i][i + 1] = k; }
  A[cl][cl - 1] = k; A[cl][cl] = -k;
  (void)n;
}

void solve(const Mat& A, double c, std::vector<double>& b) {
  int n = b.size();
  Mat M(n, std::vector<double>(n));
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) M[i][j] = (i == j) - c * A[i][j];
  for (int k = 0; k < n; ++k) {
    int p = k;
    for (int i = k + 1; i < n; ++i) if (std::abs(M[i][k]) > std::abs(M[p][k])) p = i;
    std::swap(M[k], M[p]); std::swap(b[k], b[p]);
    for (int i = k + 1; i < n; ++i) {
      double f = M[i][k] / M[k][k];
      if (f == 0) continue;
      for (int j = k; j < n; ++j) M[i][j] -= f * M[k][j];
      b[i] -= f * b[k];
    }
  }
  for (int i = n - 1; i >= 0; --i) { double s = b[i]; for (int j = i + 1; j < n; ++j) s -= M[i][j] * b[j]; b[i] = s / M[i][i]; }
}

std::vector<double> predict(const P& p, const std::vector<Obs>& obs, double dt = 0.1) {
  int n = 6 + p.cells;
  std::vector<double> x(n, 0.0), out(obs.size(), 0.0);
  const double gam = 2.0 - std::sqrt(2.0);
  long ns = std::lround(408.0 / dt);
  Mat A0(n, std::vector<double>(n)), Ag = A0, A1 = A0;
  int per_bin = p.cells / 6;
  for (long s = 0; s <= ns; ++s) {
    double t = s * dt;
    for (int d = 0; d < 17; ++d) if (std::abs(t - 24.0 * d) < 1e-9) x[0] += 25.0;  // mg/kg, F = 1
    for (size_t i = 0; i < obs.size(); ++i) {
      if (std::abs(t - obs[i].t) >= 0.5 * dt) continue;
      if (obs[i].what == 1) out[i] = x[6] / p.Kp_cas;
      else {
        int b = obs[i].what - 2;
        double m = 0;
        for (int c = 0; c < per_bin; ++c) m += x[6 + b * per_bin + c];
        out[i] = m / per_bin;
      }
    }
    if (s == ns) break;
    buildA(p, t, A0); buildA(p, t + gam * dt, Ag); buildA(p, t + dt, A1);
    std::vector<double> y(n), f0(n, 0.0);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) f0[i] += A0[i][j] * x[j];
    for (int i = 0; i < n; ++i) y[i] = x[i] + 0.5 * gam * dt * f0[i];
    solve(Ag, 0.5 * gam * dt, y);
    double cy = 1 / (gam * (2 - gam)), cx = (1 - gam) * (1 - gam) / (gam * (2 - gam));
    for (int i = 0; i < n; ++i) y[i] = cy * y[i] - cx * x[i];
    solve(A1, (1 - gam) / (2 - gam) * dt, y);
    x = y;
  }
  return out;
}

using Setter = void (*)(P&, const std::vector<double>&);

struct Fit { std::vector<double> th; double rss; std::vector<double> res; };

Fit lm(P base, std::vector<double> init, Setter set) {
  auto obs = data();
  int p = init.size(), n = obs.size();
  std::vector<double> th(p);
  for (int i = 0; i < p; ++i) th[i] = std::log(init[i]);
  auto resid = [&](const std::vector<double>& lt) {
    P q = base; std::vector<double> v(p);
    for (int i = 0; i < p; ++i) v[i] = std::exp(lt[i]);
    set(q, v);
    auto pr = predict(q, obs);
    std::vector<double> r(n);
    for (int i = 0; i < n; ++i) r[i] = std::log(std::max(pr[i], 1e-30)) - std::log(obs[i].y);
    return r;
  };
  auto ss = [](const std::vector<double>& r) { double s = 0; for (double a : r) s += a * a; return s; };
  auto r = resid(th);
  double cur = ss(r), lam = 1e-2;
  for (int it = 0; it < 50; ++it) {
    std::vector<std::vector<double>> J(n, std::vector<double>(p));
    for (int j = 0; j < p; ++j) {
      auto a = th, b = th; a[j] += 1e-4; b[j] -= 1e-4;
      auto ra = resid(a), rb = resid(b);
      for (int i = 0; i < n; ++i) J[i][j] = (ra[i] - rb[i]) / 2e-4;
    }
    std::vector<std::vector<double>> H(p, std::vector<double>(p, 0)); std::vector<double> g(p, 0);
    for (int a = 0; a < p; ++a) { for (int b = 0; b < p; ++b) for (int i = 0; i < n; ++i) H[a][b] += J[i][a] * J[i][b];
      for (int i = 0; i < n; ++i) g[a] += J[i][a] * r[i]; }
    bool ok = false;
    for (int tr = 0; tr < 12 && !ok; ++tr) {
      auto M = H; std::vector<double> d(p);
      for (int a = 0; a < p; ++a) { M[a][a] *= 1 + lam; d[a] = -g[a]; }
      for (int k = 0; k < p; ++k) { int pv = k; for (int i = k + 1; i < p; ++i) if (std::abs(M[i][k]) > std::abs(M[pv][k])) pv = i;
        std::swap(M[k], M[pv]); std::swap(d[k], d[pv]);
        for (int i = k + 1; i < p; ++i) { double f = M[i][k] / M[k][k]; for (int j = k; j < p; ++j) M[i][j] -= f * M[k][j]; d[i] -= f * d[k]; } }
      for (int i = p - 1; i >= 0; --i) { double s = d[i]; for (int j = i + 1; j < p; ++j) s -= M[i][j] * d[j]; d[i] = s / M[i][i]; }
      auto tn = th; for (int a = 0; a < p; ++a) tn[a] += std::clamp(d[a], -2.0, 2.0);
      auto rn = resid(tn); double nv = ss(rn);
      if (nv < cur) { double rel = (cur - nv) / cur; th = tn; r = rn; cur = nv; lam = std::max(lam / 5, 1e-9); ok = true; if (rel < 1e-9) it = 1000; }
      else lam *= 8;
    }
    if (!ok) break;
  }
  Fit F; F.rss = cur; F.res = r;
  for (double a : th) F.th.push_back(std::exp(a));
  return F;
}

}  // namespace

int main() {
  auto obs = data();
  int n = obs.size();
  P a; a.fractal = true; a.cells = 6; a.dx_um = 300;
  auto FA = lm(a, {5.78e3, 1.44e6, 0.71, 0.25, 0.75}, [](P& q, const std::vector<double>& v) {
    q.rate_f = v[0]; q.R_f = v[1]; q.Kp_cas = v[2]; q.kcas = v[3]; q.h = std::min(v[4], 1.0); });
  P b; b.fractal = false; b.cells = 36; b.dx_um = 50;
  auto FB = lm(b, {5.78e3, 1.44e6, 0.71, 2e-5}, [](P& q, const std::vector<double>& v) {
    q.rate_f = v[0]; q.R_f = v[1]; q.Kp_cas = v[2]; q.D = v[3]; });
  auto aic = [&](double rss, int p) { return n * std::log(rss / n) + 2 * p; };
  std::printf("model,params,RSS_log,AIC\n");
  std::printf("A fractal k(t), 6 compartments,5,%.4f,%.2f\n", FA.rss, aic(FA.rss, 5));
  std::printf("B Fickian slab, constant D, 36 cells,4,%.4f,%.2f\n", FB.rss, aic(FB.rss, 4));
  std::printf("A: rate_f=%.4g R_f=%.4g Kp_cas=%.4g k_cas=%.4g h=%.4g\n", FA.th[0], FA.th[1], FA.th[2], FA.th[3], FA.th[4]);
  std::printf("B: rate_f=%.4g R_f=%.4g Kp_cas=%.4g D=%.4g cm^2/h (%.3g um^2/s)\n", FB.th[0], FB.th[1], FB.th[2], FB.th[3],
              FB.th[3] * 1e8 / 3600.0);
  std::printf("\nresiduals log(pred/obs) by observation:\nt,what,obs,resA,resB\n");
  for (int i = 0; i < n; ++i) std::printf("%g,%d,%g,%+.3f,%+.3f\n", obs[i].t, obs[i].what, obs[i].y, FA.res[i], FB.res[i]);
  return 0;
}
