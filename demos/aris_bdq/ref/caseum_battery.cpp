// caseum_battery.cpp — C++23. Is the fractal caseum rate needed? The strong version.
//
// caseum_pde.cpp compared the paper's fractal rate with ONE Fickian rival. A reviewer can
// answer that the rival lost for reasons that have nothing to do with fractal kinetics:
// grid resolution, slab geometry, the coarseness of six compartments, or the absence of
// binding. This program fits, to the same LCM data (Karakitsios 2025, Table S14), the
// whole family below. All models share the plasma (Irwin 2016, C3HeB/FeJ), lung and
// foamy-macrophage rim of the published model and differ only in the caseum.
//
//   A6    fractal k(t) = k_cas/(t+tau)^h, 6 compartments of 300 µm (the paper)   5 p
//   A36   the same law as a continuum: D(t) = k_cas·(300 µm)^2/(t+tau)^h,
//         36 cells of 50 µm (anomalous, time-dependent diffusion)               5 p
//   B36   Fick, constant D, slab, 36 cells                                      4 p
//   B72   Fick, constant D, slab, 72 cells (grid convergence of B36)            4 p
//   BS36  Fick, constant D, SPHERE of radius 1800 µm, 36 shells                 4 p
//   C36   Fick + reversible binding to immobile sites (D, k_on, k_off),
//         slab, 36 cells: the classical mechanism for non-Fickian transients     6 p
//
// The six fit parameters common to all: rate_f, R_f, Kp_caseum (rim and outer caseum).
// Fit: least squares on log concentrations, Levenberg–Marquardt, multi-start.
// Evidence: RSS, AIC, AICc, BIC, and leave-one-time-out cross-validation (four sampling
// times; each fold refits the model without that time and predicts it).
//
// Structure: the plasma–lung subsystem drives the caseum and receives nothing back, so
// it is integrated once (TR-BDF2) and its extracellular-water concentration is stored at
// every step and stage; the caseum is then integrated with the same scheme as a forced
// linear system with a banded solve. Regression lock: A6 and B36 must reproduce the
// RSS of caseum_pde.cpp (8.1943 and 40.6039).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr double kDt = 0.1, kTend = 408.0, kGam = 2.0 - 1.4142135623730951;

double W(double pH) { return 1.0 + std::pow(10.0, 8.5 - pH) + std::pow(10.0, 9.6 - 2.0 * pH); }

struct Obs { double t; int what; double y; };  // what: 1 outer, 2..7 bins cas1..cas6
std::vector<Obs> data() {  // Table S14
  return {{5, 1, 1.30}, {24, 1, 2.62}, {5, 2, 1.19}, {5, 3, 0.374}, {5, 4, 0.0196}, {5, 6, 0.00158},
          {24, 2, 2.19}, {24, 3, 0.179}, {24, 4, 0.0664}, {24, 5, 0.0187}, {24, 6, 0.00282},
          {389, 1, 6.78}, {408, 1, 15.4}, {389, 2, 5.65}, {389, 3, 2.14}, {389, 4, 1.20}, {389, 5, 0.699},
          {389, 6, 0.433}, {408, 2, 11.9}, {408, 3, 3.49}, {408, 4, 2.28}, {408, 5, 1.27}, {408, 6, 0.577},
          {408, 7, 0.241}};
}

// ---------------------------------------------------------------- plasma + lung, once
struct Lung {  // CuEW at step n (u0[n]) and at the TR stage t_n + gam·dt (ug[n])
  std::vector<double> u0, ug;
  double Wew;
};

Lung lung_trajectory() {
  // Irwin 2016 Supplemental Table 2 (C3HeB/FeJ, per kg); mouse lung, Tables S3, S4, S8
  const double ka = 4.32, CL = 0.71, V1 = 7.90, Q2 = 5.37, V2 = 7.51, Q3 = 0.59, V3 = 61.71;
  const double Q = 840.6, SA = 500.0, Vpb = 0.0314, Vew = 0.0192, Vlys = 0.0015, Viw = 0.098;
  const double fu_pls = 7.2e-4, fu_ew = 1.1e-3, perm = 29.6, kplys_fu = 3.48e6;
  double A[6][6] = {};
  double k10 = CL / V1, k12 = Q2 / V1, k21 = Q2 / V2, k13 = Q3 / V1, k31 = Q3 / V3;
  A[0][0] = -ka; A[1][0] = ka;
  A[1][1] = -(k10 + k12 + k13); A[1][2] = k21; A[1][3] = k31;
  A[2][1] = k12; A[2][2] = -k21; A[3][1] = k13; A[3][3] = -k31;
  double Kewb = fu_pls * (1 / W(7.39)) / (fu_ew * (1 / W(7.4)));
  double capEW = (Vew + Vpb / Kewb) / fu_ew, PS = perm * SA, Wew = W(7.4), aiw = W(7.36);
  A[4][1] = Q / V1 / capEW;
  A[4][4] = -(Q / (fu_ew * Kewb) + PS / Wew) / capEW;
  A[4][5] = PS / aiw / capEW;
  double capIW = Viw + kplys_fu * Vlys;
  A[5][4] = PS / (Wew * capIW); A[5][5] = -PS / (aiw * capIW);
  auto solve6 = [&](double c, std::array<double, 6>& b) {
    double M[6][6];
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) M[i][j] = (i == j) - c * A[i][j];
    for (int k = 0; k < 6; ++k) {
      int p = k;
      for (int i = k + 1; i < 6; ++i) if (std::abs(M[i][k]) > std::abs(M[p][k])) p = i;
      for (int j = 0; j < 6; ++j) std::swap(M[k][j], M[p][j]);
      std::swap(b[k], b[p]);
      for (int i = k + 1; i < 6; ++i) { double f = M[i][k] / M[k][k]; for (int j = k; j < 6; ++j) M[i][j] -= f * M[k][j]; b[i] -= f * b[k]; }
    }
    for (int i = 5; i >= 0; --i) { double s = b[i]; for (int j = i + 1; j < 6; ++j) s -= M[i][j] * b[j]; b[i] = s / M[i][i]; }
  };
  long ns = std::lround(kTend / kDt);
  Lung L; L.Wew = Wew; L.u0.resize(ns + 1); L.ug.resize(ns + 1);
  std::array<double, 6> x{};
  for (long s = 0; s <= ns; ++s) {
    double t = s * kDt;
    for (int d = 0; d < 17; ++d) if (std::abs(t - 24.0 * d) < 1e-9) x[0] += 25.0;  // mg/kg qd, F = 1
    L.u0[s] = x[4];
    if (s == ns) { L.ug[s] = x[4]; break; }
    std::array<double, 6> y{}, f0{};
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) f0[i] += A[i][j] * x[j];
    for (int i = 0; i < 6; ++i) y[i] = x[i] + 0.5 * kGam * kDt * f0[i];
    solve6(0.5 * kGam * kDt, y);
    L.ug[s] = y[4];
    double cy = 1 / (kGam * (2 - kGam)), cx = (1 - kGam) * (1 - kGam) / (kGam * (2 - kGam));
    for (int i = 0; i < 6; ++i) y[i] = cy * y[i] - cx * x[i];
    solve6((1 - kGam) / (2 - kGam) * kDt, y);
    x = y;
  }
  return L;
}

// ---------------------------------------------------------------- caseum models
enum class Law { Fractal, Fick };
struct Model {
  const char* name;
  Law law;
  int cells;        // caseum cells over 1800 µm
  bool sphere;      // shells of a sphere of radius 1800 µm (else slab)
  bool binding;     // reversible binding to immobile sites
  int np;           // number of fitted parameters
  std::vector<double> init;  // natural-scale starting values
  std::vector<double> lo, hi;  // natural-scale box bounds (empty = unbounded)
  std::vector<int> fit_idx;    // indices of init that are fitted (empty = all)
  double K_fixed = 0;          // >0: pin k_on/k_off to this (measured caseum binding)
};

std::vector<int> fitted(const Model& m) {
  if (!m.fit_idx.empty()) return m.fit_idx;
  std::vector<int> v; for (int i = 0; i < m.np; ++i) v.push_back(i); return v;
}
int n_fitted(const Model& m) { return (int)fitted(m).size(); }

// fitted vector: [rate_f, R_f, Kp_cas, a, b, c]
//   Fractal: a = k_cas (1/h^(1-h)), b = h        Fick: a = D (cm^2/h); binding: b = k_on, c = k_off (1/h)
struct Geo { std::vector<double> V, G; int per_bin; };  // cell volumes, interface conductance factors

Geo geometry(const Model& m) {
  Geo g; int N = m.cells; double dx = 1800.0 / N;  // µm
  g.V.resize(N); g.G.resize(N > 0 ? N - 1 : 0); g.per_bin = N / 6;
  if (!m.sphere) {
    for (int i = 0; i < N; ++i) g.V[i] = dx;                 // per unit area
    for (int i = 0; i + 1 < N; ++i) g.G[i] = 1.0 / dx;       // area / distance
  } else {
    const double R = 1800.0;  // cell 0 at the surface, cell N-1 at the centre
    for (int i = 0; i < N; ++i) { double ro = R - i * dx, ri = R - (i + 1) * dx; g.V[i] = (ro * ro * ro - ri * ri * ri) / 3.0; }
    for (int i = 0; i + 1 < N; ++i) { double r = R - (i + 1) * dx; g.G[i] = r * r / dx; }
  }
  return g;
}

// Banded solve of (I - c·B) y = rhs for a matrix of half-bandwidth w (no pivoting:
// I - cB is a nonsingular M-matrix for every model here).
void band_solve(std::vector<double>& M, int n, int w, std::vector<double>& b) {
  int ld = 2 * w + 1;
  auto at = [&](int i, int j) -> double& { return M[i * ld + (j - i + w)]; };
  for (int k = 0; k < n; ++k) {
    double piv = at(k, k);
    for (int i = k + 1; i <= std::min(n - 1, k + w); ++i) {
      double f = at(i, k) / piv;
      if (f == 0) continue;
      for (int j = k; j <= std::min(n - 1, k + w); ++j) at(i, j) -= f * at(k, j);
      b[i] -= f * b[k];
    }
  }
  for (int i = n - 1; i >= 0; --i) {
    double s = b[i];
    for (int j = i + 1; j <= std::min(n - 1, i + w); ++j) s -= at(i, j) * b[j];
    b[i] = s / at(i, i);
  }
}

struct System {  // dy/dt = B(t) y + f(t), banded
  int n, w;
  std::vector<double> B;  // band storage, ld = 2w+1
  std::vector<double> f;
};

// Build the caseum system at time t with lung forcing u (CuEW).
void build(const Model& m, const Geo& g, const std::vector<double>& th, double t, double u, double Wew, System& S) {
  const double rate_f = th[0], R_f = th[1], Kp = th[2];
  const double tau = 1e-3, flys = 0.078;
  const double fIW = 0.098 / (0.0314 + 0.0192 + 0.098 + 0.0015), r = flys / fIW;
  const double denf = (1 + R_f * r) / (1 + r);
  int N = m.cells, sb = m.binding ? 2 : 1;
  S.n = N * sb; S.w = m.binding ? 2 : 1;
  int ld = 2 * S.w + 1;
  S.B.assign(S.n * ld, 0.0); S.f.assign(S.n, 0.0);
  auto at = [&](int i, int j) -> double& { return S.B[i * ld + (j - i + S.w)]; };
  // diffusivity in µm^2/h
  double D;
  if (m.law == Law::Fractal) D = th[3] * 300.0 * 300.0 / std::pow(t + tau, th[4]);
  else D = th[3] * 1e8;  // cm^2/h -> µm^2/h
  // outer caseum lumped with cell 0: its volume is 1.67 × the first 300 µm (paper)
  double V300 = 0; for (int i = 0; i < g.per_bin; ++i) V300 += g.V[i];
  double Vout = 1.67 * V300;
  std::vector<double> cap(N);
  for (int i = 0; i < N; ++i) cap[i] = g.V[i];
  cap[0] += Vout / Kp;
  auto F = [&](int i) { return i * sb; };  // index of free conc. of cell i
  // rim exchange into the outer caseum (eq. S12), scaled to outer-caseum volume
  double kin = rate_f * Vout;
  S.f[F(0)] = kin * u / Wew / cap[0];
  at(F(0), F(0)) -= kin / (denf * Kp) / cap[0];
  for (int i = 0; i + 1 < N; ++i) {
    double Gc = D * g.G[i];
    at(F(i), F(i)) -= Gc / cap[i];       at(F(i), F(i + 1)) += Gc / cap[i];
    at(F(i + 1), F(i + 1)) -= Gc / cap[i + 1]; at(F(i + 1), F(i)) += Gc / cap[i + 1];
  }
  if (m.binding) {
    double kon = th[4], koff = th[5];
    for (int i = 0; i < N; ++i) {
      double vf = g.V[i] / cap[i];  // binding acts on the caseum part of the cell only
      at(F(i), F(i)) -= kon * vf;     at(F(i), F(i) + 1) += koff * vf;
      at(F(i) + 1, F(i)) += kon;      at(F(i) + 1, F(i) + 1) -= koff;
    }
  }
}

std::vector<double> predict(const Model& m, const Lung& L, const std::vector<double>& th, const std::vector<Obs>& obs) {
  Geo g = geometry(m);
  int N = m.cells, sb = m.binding ? 2 : 1, n = N * sb;
  std::vector<double> x(n, 0.0), out(obs.size(), 0.0);
  long ns = std::lround(kTend / kDt);
  System S0, Sg, S1;
  double cy = 1 / (kGam * (2 - kGam)), cx = (1 - kGam) * (1 - kGam) / (kGam * (2 - kGam));
  for (long s = 0; s <= ns; ++s) {
    double t = s * kDt;
    for (size_t i = 0; i < obs.size(); ++i) {
      if (std::abs(t - obs[i].t) >= 0.5 * kDt) continue;
      if (obs[i].what == 1) out[i] = x[0] / th[2];  // outer caseum = C_first / Kp
      else {
        int b = obs[i].what - 2; double num = 0, den = 0;
        for (int c = 0; c < g.per_bin; ++c) {
          int k = b * g.per_bin + c;
          double tot = x[k * sb] + (m.binding ? x[k * sb + 1] : 0.0);  // LCM measures total
          num += tot * g.V[k]; den += g.V[k];
        }
        out[i] = num / den;
      }
    }
    if (s == ns) break;
    build(m, g, th, t, L.u0[s], L.Wew, S0);
    build(m, g, th, t + kGam * kDt, L.ug[s], L.Wew, Sg);
    build(m, g, th, t + kDt, L.u0[s + 1], L.Wew, S1);
    int ld = 2 * S0.w + 1;
    auto mul = [&](const System& S, const std::vector<double>& v) {
      std::vector<double> r(n, 0.0);
      for (int i = 0; i < n; ++i) for (int j = std::max(0, i - S.w); j <= std::min(n - 1, i + S.w); ++j) r[i] += S.B[i * ld + (j - i + S.w)] * v[j];
      return r;
    };
    auto lhs = [&](const System& S, double c) {
      std::vector<double> M(n * ld, 0.0);
      for (int i = 0; i < n; ++i) for (int j = std::max(0, i - S.w); j <= std::min(n - 1, i + S.w); ++j)
        M[i * ld + (j - i + S.w)] = (i == j) - c * S.B[i * ld + (j - i + S.w)];
      return M;
    };
    // TR stage
    double c1 = 0.5 * kGam * kDt;
    auto Bx = mul(S0, x);
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) y[i] = x[i] + c1 * (Bx[i] + S0.f[i]) + c1 * Sg.f[i];
    auto Mg = lhs(Sg, c1); band_solve(Mg, n, S0.w, y);
    // BDF2 stage
    double c2 = (1 - kGam) / (2 - kGam) * kDt;
    for (int i = 0; i < n; ++i) y[i] = cy * y[i] - cx * x[i] + c2 * S1.f[i];
    auto M1 = lhs(S1, c2); band_solve(M1, n, S0.w, y);
    x = y;
  }
  return out;
}

// ---------------------------------------------------------------- fitting
struct Fit { std::vector<double> th; double rss = 1e300; };

double sumsq(const std::vector<double>& r) { double s = 0; for (double a : r) s += a * a; return s; }

std::vector<double> resid(const Model& m, const Lung& L, const std::vector<double>& th, const std::vector<Obs>& obs) {
  auto pr = predict(m, L, th, obs);
  std::vector<double> r(obs.size());
  for (size_t i = 0; i < obs.size(); ++i) r[i] = std::log(std::max(pr[i], 1e-30)) - std::log(obs[i].y);
  return r;
}

std::vector<double> clampTh(const Model& m, std::vector<double> th) {
  if (m.law == Law::Fractal) th[4] = std::min(th[4], 1.0);
  if (m.K_fixed > 0) th[4] = m.K_fixed * th[5];   // K = k_on/k_off pinned
  for (size_t i = 0; i < th.size() && i < m.lo.size(); ++i) th[i] = std::clamp(th[i], m.lo[i], m.hi[i]);
  return th;
}

Fit lm(const Model& m, const Lung& L, std::vector<double> init, const std::vector<Obs>& obs) {
  auto idx = fitted(m);
  int p = (int)idx.size(), n = obs.size();
  std::vector<double> base = init;              // non-fitted entries stay here
  std::vector<double> lt(p);
  for (int i = 0; i < p; ++i) lt[i] = std::log(init[idx[i]]);
  auto expand = [&](const std::vector<double>& l) {
    std::vector<double> th = base;
    for (int i = 0; i < p; ++i) th[idx[i]] = std::exp(l[i]);
    return clampTh(m, th);
  };
  auto R = [&](const std::vector<double>& l) { return resid(m, L, expand(l), obs); };
  auto r = R(lt); double cur = sumsq(r), lam = 1e-2;
  for (int it = 0; it < 80; ++it) {
    std::vector<std::vector<double>> J(n, std::vector<double>(p));
    for (int j = 0; j < p; ++j) {
      auto a = lt, b = lt; a[j] += 1e-4; b[j] -= 1e-4;
      auto ra = R(a), rb = R(b);
      for (int i = 0; i < n; ++i) J[i][j] = (ra[i] - rb[i]) / 2e-4;
    }
    std::vector<std::vector<double>> H(p, std::vector<double>(p, 0)); std::vector<double> gr(p, 0);
    for (int a = 0; a < p; ++a) { for (int b = 0; b < p; ++b) for (int i = 0; i < n; ++i) H[a][b] += J[i][a] * J[i][b];
      for (int i = 0; i < n; ++i) gr[a] += J[i][a] * r[i]; }
    bool ok = false;
    for (int tr = 0; tr < 14 && !ok; ++tr) {
      auto M = H; std::vector<double> d(p);
      for (int a = 0; a < p; ++a) { M[a][a] = M[a][a] * (1 + lam) + 1e-12; d[a] = -gr[a]; }
      for (int k = 0; k < p; ++k) { int pv = k; for (int i = k + 1; i < p; ++i) if (std::abs(M[i][k]) > std::abs(M[pv][k])) pv = i;
        std::swap(M[k], M[pv]); std::swap(d[k], d[pv]);
        for (int i = k + 1; i < p; ++i) { double f = M[i][k] / M[k][k]; for (int j = k; j < p; ++j) M[i][j] -= f * M[k][j]; d[i] -= f * d[k]; } }
      for (int i = p - 1; i >= 0; --i) { double s = d[i]; for (int j = i + 1; j < p; ++j) s -= M[i][j] * d[j]; d[i] = s / M[i][i]; }
      auto tn = lt; for (int a = 0; a < p; ++a) tn[a] += std::clamp(d[a], -2.0, 2.0);
      auto rn = R(tn); double nv = sumsq(rn);
      if (std::isfinite(nv) && nv < cur) { double rel = (cur - nv) / cur; lt = tn; r = rn; cur = nv; lam = std::max(lam / 5, 1e-9); ok = true; if (rel < 1e-10) it = 1000; }
      else lam *= 8;
    }
    if (!ok) break;
  }
  Fit F; F.rss = cur; F.th = expand(lt);
  return F;
}

// multi-start: the given start plus log-spaced perturbations of the mechanism parameters
Fit fit_best(const Model& m, const Lung& L, const std::vector<Obs>& obs, const std::vector<double>* warm = nullptr) {
  std::vector<std::vector<double>> starts;
  if (warm) starts.push_back(*warm);
  starts.push_back(m.init);
  const double mult[] = {0.1, 10.0, 0.03, 30.0};
  auto idx = fitted(m);
  for (double f : mult) {
    auto s = m.init;
    for (int k : idx) {
      if (k < 3) continue;
      if (m.law == Law::Fractal && k == 4) continue;  // h stays in (0, 1]
      s[k] *= (k % 2 ? f : 1.0 / f);
    }
    starts.push_back(s);
  }
  auto is_fitted = [&](int k) { return std::find(idx.begin(), idx.end(), k) != idx.end(); };
  if (m.law == Law::Fractal && is_fitted(4))
    for (double h : {0.4, 0.95}) { auto s = m.init; s[4] = h; starts.push_back(s); }
  if (m.binding && is_fitted(4) && is_fitted(5))
    for (double kk : {1e-3, 1.0, 1e3}) { auto s = m.init; s[4] = kk * 10; s[5] = kk; starts.push_back(s); }
  Fit best;
  for (auto& s : starts) { auto F = lm(m, L, s, obs); if (F.rss < best.rss) best = F; }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  bool quick = argc > 1 && std::strcmp(argv[1], "quick") == 0;
  bool resid_mode = argc > 1 && std::strcmp(argv[1], "resid") == 0;
  if (resid_mode) quick = true;
  auto obs = data();
  int n = obs.size();
  Lung L = lung_trajectory();
  // Physical ceiling on a constant diffusivity: free aqueous diffusion of a molecule of
  // bedaquiline's size (MW 555) is about 400 um^2/s by Stokes-Einstein. Caseum is a dense,
  // lipid- and protein-rich matrix, so its diffusivity must be BELOW the free-solution value;
  // 400 um^2/s = 0.0144 cm^2/h is therefore a generous ceiling, not a tight prior.
  const double Dmax = 0.0144;                       // cm^2/h  (= 400 um^2/s, free water)
  const std::vector<double> loB = {1.0, 1e3, 1e-2, 1e-12};
  const std::vector<double> hiB = {1e7, 1e9, 1e2, Dmax};
  const std::vector<double> loC = {1.0, 1e3, 1e-2, 1e-12, 1e-6, 1e-6};
  const std::vector<double> hiC = {1e7, 1e9, 1e2, Dmax, 1e9, 1e9};
  std::vector<Model> models = {
    {"A6   fractal k(t), 6 compartments (paper)", Law::Fractal, 6, false, false, 5, {5.78e3, 1.44e6, 0.71, 0.25, 0.75}, {}, {}},
    {"A36  fractal law as continuum D(t), 36 cells", Law::Fractal, 36, false, false, 5, {5.78e3, 1.44e6, 0.71, 0.25, 0.75}, {}, {}},
    {"B36  Fick, constant D, slab, 36 cells", Law::Fick, 36, false, false, 4, {5.78e3, 1.44e6, 0.71, 2e-4}, {}, {}},
    {"B72  Fick, constant D, slab, 72 cells", Law::Fick, 72, false, false, 4, {5.78e3, 1.44e6, 0.71, 2e-4}, {}, {}},
    {"BS36 Fick, constant D, sphere, 36 shells", Law::Fick, 36, true, false, 4, {5.78e3, 1.44e6, 0.71, 2e-4}, {}, {}},
    {"C36  Fick + binding, free parameters", Law::Fick, 36, false, true, 6, {5.78e3, 1.44e6, 0.71, 2e-4, 0.1, 0.01}, {}, {}},
    {"C36p Fick + binding, D <= free-water 400 um2/s", Law::Fick, 36, false, true, 6, {5.78e3, 1.44e6, 0.71, 2e-4, 0.1, 0.01}, loC, hiC},
    {"B36p Fick, constant D, D <= free water", Law::Fick, 36, false, false, 4, {5.78e3, 1.44e6, 0.71, 2e-4}, loB, hiB},
    // Rim held at the paper's published Table S13 estimates (rate_fmphi, Kp_fmphi/(fu*fni),
    // Kp_caseum). Only the caseum transport law is fitted, so the comparison cannot be won
    // or lost on the rim: it is a comparison of transport laws alone.
    {"R-A  rim fixed | fractal k(t), 6 compartments", Law::Fractal, 6, false, false, 5, {5.78e3, 1.44e6, 0.71, 0.25, 0.75}, {}, {}, {3, 4}},
    {"R-B  rim fixed | Fick, constant D, 36 cells", Law::Fick, 36, false, false, 4, {5.78e3, 1.44e6, 0.71, 2e-4}, loB, hiB, {3}},
    {"R-C  rim fixed | Fick + binding, 36 cells", Law::Fick, 36, false, true, 6, {5.78e3, 1.44e6, 0.71, 2e-4, 0.1, 0.01}, loC, hiC, {3, 4, 5}},
    // Independent constraint. Sarathy et al., ACS Infect Dis 2016 (doi 10.1021/acsinfecdis.6b00051),
    // measured caseum binding by rapid equilibrium dialysis: bedaquiline is the most bound drug of
    // the panel, fu(caseum) < 0.01 %. A retardation factor R = 1/fu therefore EXCEEDS 1e4.
    // M-C pins k_on/k_off to that measured value and fits only D (<= free water) and k_off.
    // M-B is its rapid-equilibrium limit: Fick with D_eff <= D_free/R = 400/1e4 um2/s.
    {"M-C  rim fixed | binding pinned at measured fu<1e-4", Law::Fick, 36, false, true, 6, {5.78e3, 1.44e6, 0.71, 2e-4, 100.0, 0.01}, loC, hiC, {3, 5}, 1e4},
    // Nested test: the fractal law contains the constant-rate chain at h = 0, so h is one
    // extra parameter over the paper's own 6-compartment structure. R-A0 is that null.
    {"R-A0 rim fixed | 6 compartments, constant rate h=0", Law::Fractal, 6, false, false, 5, {5.78e3, 1.44e6, 0.71, 0.25, 1e-9}, {}, {}, {3}},
    {"M-B  rim fixed | Fick, D_eff <= free water / 1e4", Law::Fick, 36, false, false, 4, {5.78e3, 1.44e6, 0.71, 1e-6},
       {1.0, 1e3, 1e-2, 1e-14}, {1e7, 1e9, 1e2, 1.44e-6}, {3}},
  };

  // regression lock against caseum_pde.cpp at its published optima
  {
    double rA = sumsq(resid(models[0], L, {4967, 1.668e6, 0.5791, 0.2163, 0.6995}, obs));
    double rB = sumsq(resid(models[2], L, {5968, 1.175e6, 0.1746, 2.255e-4}, obs));
    std::printf("LOCK A6 RSS at caseum_pde optimum = %.4f (expect 8.194)  B36 = %.4f (expect 40.60)\n", rA, rB);
  }
  std::printf("\nmodel,p,RSS_log,AIC,AICc,BIC,LOTO_CV_SSE,params\n");
  double rssA = -1, rssA0 = -1;
  const double times[4] = {5, 24, 389, 408};
  for (auto& m : models) {
    Fit F = fit_best(m, L, obs);
    int p = n_fitted(m);
    double aic = n * std::log(F.rss / n) + 2 * p;
    double aicc = aic + 2.0 * p * (p + 1) / (n - p - 1);
    double bic = n * std::log(F.rss / n) + p * std::log((double)n);
    double cv = 0;
    if (!quick) {
      for (double tt : times) {
        std::vector<Obs> tr, te;
        for (auto& o : obs) (o.t == tt ? te : tr).push_back(o);
        Fit Fk = fit_best(m, L, tr, &F.th);
        cv += sumsq(resid(m, L, Fk.th, te));
      }
    }
    if (resid_mode) {
      if (m.name[0] != 'R' && m.name[0] != 'M') continue;
      auto r = resid(m, L, F.th, obs);
      for (int i = 0; i < n; ++i) std::printf("RESID,%.4s,%g,%d,%.5f\n", m.name, obs[i].t, obs[i].what, r[i]);
      continue;
    }
    if (std::strncmp(m.name, "R-A ", 4) == 0) rssA = F.rss;
    if (std::strncmp(m.name, "R-A0", 4) == 0) rssA0 = F.rss;
    std::printf("%s,%d,%.4f,%.2f,%.2f,%.2f,%.3f,", m.name, p, F.rss, aic, aicc, bic, quick ? NAN : cv);
    for (double v : F.th) std::printf("%.4g ", v);
    std::printf("\n");
    if (m.law == Law::Fick) {
      double Dus = F.th[3] * 1e8 / 3600.0;  // cm^2/h -> um^2/s
      std::printf("    D = %.4g um2/s (free water for MW 555 ~ 400)", Dus);
      if (m.binding) {
        double K = F.th[4] / F.th[5], R = 1 + K;
        std::printf("; K = k_on/k_off = %.3g, retardation R = %.3g, D_eff = D/R = %.4g um2/s; k_off t_1/2 = %.3g h",
                    K, R, Dus / R, std::log(2.0) / F.th[5]);
      }
      std::printf("\n");
    }
    std::fflush(stdout);
    if (m.name[0] == 'A' || m.name[0] == 'C' || m.name[0] == 'R' || std::strncmp(m.name, "B36", 3) == 0) {
      auto r = resid(m, L, F.th, obs);
      std::printf("  residuals %s t=408 bins 1..6:", m.name);
      for (int i = 0; i < n; ++i) if (obs[i].t == 408 && obs[i].what >= 2) std::printf(" %+.2f", r[i]);
      std::printf("\n");
    }
  }
  if (rssA > 0 && rssA0 > 0) {
    // The fractal law contains the constant-rate chain at h = 0: one nested parameter.
    double chi2 = n * std::log(rssA0 / rssA);
    double z = std::sqrt(chi2);
    double pval = std::erfc(z / std::sqrt(2.0));   // two-sided normal tail = chi2_1 upper tail
    std::printf("\nNested test on the fractal exponent h (same 6-compartment structure, same rim):\n");
    std::printf("  RSS(h = 0) = %.4f   RSS(h fitted = free) = %.4f\n", rssA0, rssA);
    std::printf("  LRT chi2(1) = n ln(RSS0/RSS1) = %.2f   p = %.3g\n", chi2, pval);
  }
  return 0;
}
