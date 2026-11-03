// bdq.cpp — C++23 reference reimplementation of the human lung PBPK prediction for
// bedaquiline in Karakitsios, Della Pasqua & Dokoumetzidis (2025), Br J Clin Pharmacol
// 91:3167-3178, doi:10.1002/bcp.70163 (equations S1-S19, Tables S2-S4, S8, S11, S13),
// driven by the human plasma model of Svensson, Dosne & Karlsson (2016), CPT:PSP 5:682,
// doi:10.1002/psp4.12147 (Table 3, typical patient at reference covariates).
//
// Every state equation is linear in the states (k_caseum(t) is time-varying but does not
// depend on the state), so the system is integrated with TR-BDF2 (L-stable, 2nd order)
// using a dense LU solve per stage. Dose events are aligned to step boundaries.
//
// Modes:
//   bdq nominal <dt> [ion]      -> nominal trajectories (CSV on stdout)
//   bdq conv                    -> step-refinement table
//   bdq gum   <budget>          -> first-order GUM budget of endpoints
//   bdq mc    <budget> <N> <seed> -> Monte Carlo of endpoints
//   bdq ion                     -> endpoints under the three ionisation readings
// ion: 0 = diprotic base (pKa 8.5, 1.1)   [default; consistent with the published figures]
//      1 = formula as printed in S1 with Table S2 labels (pKa1 = 8.5, pKa2 = 1.1)
//      2 = formula as printed in S1 with text labels (pKa1 = 1.1, pKa2 = 8.5)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int NS = 17;  // number of states
// state indices
enum : int {
  T1 = 0, T2, DEP, CEN, P1, P2, ELIM,  // plasma (amounts, mg); ELIM = cumulative eliminated
  CUEW, CIW,                           // healthy lung: unbound EW conc, total IW conc (mg/L)
  CMPH,                                // cellular lesion macrophage total conc (mg/L ~ ug/g)
  CAS1, CAS2, CAS3, CAS4, CAS5, CAS6,  // caseum total concs (mg/L ~ ug/g)
  CASIN                                // cumulative amount-equivalent entering cas chain
};

struct Params {
  // --- Svensson 2016, Table 3 (typical: 70 kg, non-Black, 32 y, albumin at steady state)
  double CL = 2.62, V = 198.0, Q1 = 3.66, VP1 = 8550.0, Q2 = 7.34, VP2 = 2690.0;  // L/h, L
  double MATfrac = 0.66, FR = 0.47;  // MAT as fraction of 6 h; FR = transit fraction of MAT
  double F = 1.0;
  // --- Karakitsios 2025, human physiology (Table S3) — mL, mL/h, cm^2
  double Qlung = 356000.0, SA = 960000.0;
  double V_PB = 101.2, V_EW = 89.2, V_LYS = 5.47, V_IW = 351.0;
  double pH_pl = 7.4, pH_EW = 7.4, pH_IW = 6.69, pH_LYS = 5.0;
  // Table S4 (human)
  double BP = 1.0, fu_pls = 5.0e-4, fu_EW = 8.2e-4;
  // drug (Table S2)
  double pKa_base_hi = 8.5, pKa_base_lo = 1.1;
  int ion = 0;
  // --- fitted in mice (S8, S11, S13), carried to humans
  double Perm = 55.7;            // cm/h
  double KpLYS_fu_mouse = 3.48e6;
  double pH_IW_mouse = 7.36;
  double rate_m = 1.43e5;        // 1/h
  double R_m = 5.68e6;           // Kp_mphi / (fu*fni)
  double rate_f = 5.78e3;        // 1/h
  double R_f = 1.44e6;           // Kp_fmphi / (fu*fni)
  double Kp_cas = 0.71;
  double kcas = 0.25;            // h^(h-1)
  double h = 0.75;
  double tau = 1.0e-3;           // h (not reported: "a very small positive constant")
  // structural constants
  double flys = 0.078;           // lysosomal fraction of cell volume (Ufuk 2017)
  double ratio1 = 1.67;          // V_outer / V_cas1 ; ratio2..6 = 1
  double pH_foamy = 5.84;
  bool cas_sign_conservative = true;  // eq S12: +k*C2/ratio2 (conserving) vs printed sign
};

// 1/fraction non-ionised.  ion==0: diprotic base.  ion 1/2: formula as printed in S1.
double Wfac(const Params& p, double pH) {
  if (p.ion == 0) {
    return 1.0 + std::pow(10.0, p.pKa_base_hi - pH) +
           std::pow(10.0, p.pKa_base_hi + p.pKa_base_lo - 2.0 * pH);
  }
  double pKa1 = (p.ion == 1) ? 8.5 : 1.1;
  double pKa2 = (p.ion == 1) ? 1.1 : 8.5;
  return 1.0 + std::pow(10.0, pKa1 - pH) + std::pow(10.0, pH - pKa2) +
         std::pow(10.0, pKa1 + pKa2 - 2.0 * pH);
}
double fni(const Params& p, double pH) { return 1.0 / Wfac(p, pH); }

struct Derived {
  double ktr, ka;
  double K_EWB;          // total EW / total blood at equilibrium of unbound-unionised
  double PS;             // mL/h
  double W_EW, a_IW;
  double KpLYS_fu_h;     // human Kp_LYS:IW / fu_IW (eq S9)
  double fIW, r;         // r = 0.078/fIW
  double cap_EW;         // (V_EW + V_PB/K_EWB)/fu_EW
};

Derived derive(const Params& p) {
  Derived d{};
  double MAT = p.MATfrac * 6.0;
  double MTT = p.FR * MAT;
  d.ktr = 2.0 / MTT;                               // two transit compartments
  d.ka = std::log(10.0) / ((1.0 - p.FR) * MAT);    // time to 90 % of first-order part
  // equal unbound-unionised conc in plasma and EW:
  //   fu_EW*fni_EW*C_EW = fu_pls*fni_pl*C_pl ;  C_blood = BP*C_pl
  d.K_EWB = (p.fu_pls * fni(p, p.pH_pl)) / (p.fu_EW * fni(p, p.pH_EW) * p.BP);
  d.PS = p.Perm * p.SA;
  d.W_EW = Wfac(p, p.pH_EW);
  d.a_IW = Wfac(p, p.pH_IW);
  // eq S8/S9: (fuLYS*fniLYS) fitted in mice and held constant
  double fulys_fnilys = fni(p, p.pH_IW_mouse) / p.KpLYS_fu_mouse;
  d.KpLYS_fu_h = fni(p, p.pH_IW) / fulys_fnilys;
  d.fIW = p.V_IW / (p.V_PB + p.V_EW + p.V_IW + p.V_LYS);
  d.r = p.flys / d.fIW;
  d.cap_EW = (p.V_EW + p.V_PB / d.K_EWB) / p.fu_EW;
  return d;
}

using Mat = std::array<std::array<double, NS>, NS>;
using Vec = std::array<double, NS>;

double kcase(const Params& p, double t) { return p.kcas / std::pow(t + p.tau, p.h); }

// Build A(t) such that dx/dt = A(t) x  (all linear, no forcing besides doses)
void buildA(const Params& p, const Derived& d, double t, Mat& A) {
  for (auto& row : A) row.fill(0.0);
  // ---- plasma (amounts, mg; volumes in L)
  A[T1][T1] = -d.ktr;
  A[T2][T1] = d.ktr;   A[T2][T2] = -d.ktr;
  A[DEP][T2] = d.ktr;  A[DEP][DEP] = -d.ka;
  A[CEN][DEP] = d.ka;
  double k10 = p.CL / p.V, k12 = p.Q1 / p.V, k21 = p.Q1 / p.VP1, k13 = p.Q2 / p.V, k31 = p.Q2 / p.VP2;
  A[CEN][CEN] = -(k10 + k12 + k13);
  A[CEN][P1] = k21; A[CEN][P2] = k31;
  A[P1][CEN] = k12; A[P1][P1] = -k21;
  A[P2][CEN] = k13; A[P2][P2] = -k31;
  A[ELIM][CEN] = k10;
  // C_pl (mg/L) = CEN / V ; C_blood = BP*C_pl
  // ---- EW (eq S1), state Cu_EW (mg/L), volumes mL, flows mL/h
  // cap_EW * dCu/dt = Q*Cb - Q*Cu/(fu_EW*K) - PS*(Cu/W - C_IW*fu_IW/a), fu_IW = 1
  double inv = 1.0 / d.cap_EW;
  A[CUEW][CEN] = inv * p.Qlung * p.BP / p.V;
  A[CUEW][CUEW] = -inv * (p.Qlung / (p.fu_EW * d.K_EWB) + d.PS / d.W_EW);
  A[CUEW][CIW] = inv * d.PS / d.a_IW;
  // ---- IW (eq S4): (V_IW + Kp*V_LYS) dC_IW/dt = PS*(Cu_EW/W - Cu_IW/a), fu_IW=1 => Kp = KpLYS_fu_h
  double capIW = p.V_IW + d.KpLYS_fu_h * p.V_LYS;
  A[CIW][CUEW] = d.PS / (d.W_EW * capIW);
  A[CIW][CIW] = -d.PS / (d.a_IW * capIW);
  // ---- cellular lesion (eq S11), fu*fni fixed to 1 => Kp_mphi = R_m
  double denm = 1.0 / (d.r + 1.0) + p.R_m * d.r / (d.r + 1.0);
  A[CMPH][CUEW] = p.rate_m / d.W_EW;
  A[CMPH][CMPH] = -p.rate_m / denm;
  // ---- outer caseum + caseum chain (eq S12-S14)
  double denf = (1.0 + p.R_f * d.r) / (d.r + 1.0);
  double g = 1.0 / (1.0 + p.ratio1 / p.Kp_cas);
  double kc = kcase(p, t);
  A[CAS1][CUEW] = p.rate_f * p.ratio1 * g / d.W_EW;
  A[CAS1][CAS1] = -p.rate_f * p.ratio1 * g / (denf * p.Kp_cas) - kc * g;
  A[CAS1][CAS2] = (p.cas_sign_conservative ? +1.0 : -1.0) * kc * g;  // ratio2 = 1
  for (int n = CAS2; n <= CAS5; ++n) {
    A[n][n - 1] = kc;  // ratio_n = 1
    A[n][n] = -2.0 * kc;
    A[n][n + 1] = kc;
  }
  A[CAS6][CAS5] = kc;
  A[CAS6][CAS6] = -kc;
  // influx bookkeeping into cas chain (as dC_cas1 units; outer caseum excluded)
  // conserved quantity (equal volumes V_cas): C1/g + C2 + ... + C6, since the outer caseum
  // (volume ratio1*V_cas1, conc C1/Kp_cas) is lumped with cas1.  Its source term is:
  A[CASIN][CUEW] = A[CAS1][CUEW] / g;
  A[CASIN][CAS1] = -p.rate_f * p.ratio1 / (denf * p.Kp_cas);
}

// solve (I - c*A) x = b in place, dense LU with partial pivoting
void solve(const Mat& A, double c, Vec& b) {
  Mat M;
  for (int i = 0; i < NS; ++i)
    for (int j = 0; j < NS; ++j) M[i][j] = (i == j ? 1.0 : 0.0) - c * A[i][j];
  std::array<int, NS> piv;
  std::iota(piv.begin(), piv.end(), 0);
  for (int k = 0; k < NS; ++k) {
    int pmax = k;
    for (int i = k + 1; i < NS; ++i)
      if (std::abs(M[i][k]) > std::abs(M[pmax][k])) pmax = i;
    std::swap(M[k], M[pmax]);
    std::swap(b[k], b[pmax]);
    for (int i = k + 1; i < NS; ++i) {
      double f = M[i][k] / M[k][k];
      if (f == 0.0) continue;
      for (int j = k; j < NS; ++j) M[i][j] -= f * M[k][j];
      b[i] -= f * b[k];
    }
  }
  for (int i = NS - 1; i >= 0; --i) {
    double s = b[i];
    for (int j = i + 1; j < NS; ++j) s -= M[i][j] * b[j];
    b[i] = s / M[i][i];
  }
}

Vec matvec(const Mat& A, const Vec& x) {
  Vec y{};
  for (int i = 0; i < NS; ++i) {
    double s = 0;
    for (int j = 0; j < NS; ++j) s += A[i][j] * x[j];
    y[i] = s;
  }
  return y;
}

// dose schedule: 400 mg qd days 0..13, then 200 mg Mon/Wed/Fri until end of week 24
std::vector<std::pair<double, double>> schedule() {
  std::vector<std::pair<double, double>> s;
  for (int dday = 0; dday < 14; ++dday) s.push_back({24.0 * dday, 400.0});
  // day 14 is a Monday-equivalent; pattern +2,+2,+3
  int day = 14;
  int pat[3] = {2, 2, 3};
  int k = 0;
  while (day < 24 * 7) {
    s.push_back({24.0 * day, 200.0});
    day += pat[k % 3];
    ++k;
  }
  return s;
}

struct Out {
  // outputs recorded on hourly grid
  std::vector<double> t, cp_u, lys_m, cyt_m, lys_f, cyt_f, outer, cas[6];
  double mass_resid_rel = 0;  // plasma ledger
  double cas_resid_rel = 0;   // caseum chain ledger
};

struct Obs {
  double cp_u, cyt_m, lys_m, cyt_f, lys_f, outer;
  std::array<double, 6> cas;
};

Obs observe(const Params& p, const Derived& d, const Vec& x) {
  Obs o{};
  double Cp = x[CEN] / p.V;            // mg/L
  o.cp_u = p.fu_pls * Cp * 1e3;        // ng/mL
  // macrophage: total C_mphi -> C_IW (total, fu=1 convention) -> C_LYS
  double denm = (1.0 + p.R_m * d.r) / (1.0 + d.r);
  double CIWm = x[CMPH] / denm;        // fu*fni = 1 convention: this is the unionised-unbound eq. conc
  double CLYSm = p.R_m * CIWm;
  // eq S18/S19: unbound cytosol = C_LYS/(R*fni_IW(foamy pH)); unbound lys = * fni_IW/fni_LYS
  double fniIW = fni(p, p.pH_foamy), fniLYS = fni(p, p.pH_LYS);
  o.cyt_m = CLYSm / (p.R_m * fniIW) * 1e3;
  o.lys_m = o.cyt_m * fniIW / fniLYS;
  // foamy: outer caseum total = C_cas1/Kp_cas
  double Cout = x[CAS1] / p.Kp_cas;
  double denf = (1.0 + p.R_f * d.r) / (1.0 + d.r);
  double CLYSf = p.R_f * Cout / denf;  // eq S17 with fu*fni=1 convention
  o.cyt_f = CLYSf / (p.R_f * fniIW) * 1e3;
  o.lys_f = o.cyt_f * fniIW / fniLYS;
  o.outer = Cout * 1e3;
  for (int i = 0; i < 6; ++i) o.cas[i] = x[CAS1 + i] * 1e3;  // ng/g
  return o;
}

Out simulate(const Params& p, double dt, double tend = 4032.0, bool record = true) {
  Derived d = derive(p);
  auto sched = schedule();
  Vec x{};
  Out out;
  double dosed = 0;
  const double gam = 2.0 - std::sqrt(2.0);
  size_t di = 0;
  int nsteps = static_cast<int>(std::llround(tend / dt));
  Mat A0, Ag, A1;
  double nextRec = 0.0;
  for (int n = 0; n <= nsteps; ++n) {
    double t = n * dt;
    while (di < sched.size() && sched[di].first <= t + 1e-9) {
      x[T1] += p.F * sched[di].second;
      dosed += p.F * sched[di].second;
      ++di;
    }
    if (record && t + 1e-9 >= nextRec) {
      Obs o = observe(p, d, x);
      out.t.push_back(t);
      out.cp_u.push_back(o.cp_u); out.lys_m.push_back(o.lys_m); out.cyt_m.push_back(o.cyt_m);
      out.lys_f.push_back(o.lys_f); out.cyt_f.push_back(o.cyt_f); out.outer.push_back(o.outer);
      for (int i = 0; i < 6; ++i) out.cas[i].push_back(o.cas[i]);
      nextRec += 1.0;
    }
    if (n == nsteps) break;
    // TR-BDF2 step
    buildA(p, d, t, A0);
    buildA(p, d, t + gam * dt, Ag);
    buildA(p, d, t + dt, A1);
    Vec f0 = matvec(A0, x);
    Vec y{};
    for (int i = 0; i < NS; ++i) y[i] = x[i] + 0.5 * gam * dt * f0[i];
    solve(Ag, 0.5 * gam * dt, y);  // trapezoid stage
    Vec z{};
    for (int i = 0; i < NS; ++i)
      z[i] = (1.0 / (gam * (2.0 - gam))) * y[i] - ((1.0 - gam) * (1.0 - gam) / (gam * (2.0 - gam))) * x[i];
    solve(A1, (1.0 - gam) / (2.0 - gam) * dt, z);  // BDF2 stage
    x = z;
  }
  double inbody = x[T1] + x[T2] + x[DEP] + x[CEN] + x[P1] + x[P2];
  out.mass_resid_rel = std::abs(dosed - inbody - x[ELIM]) / dosed;
  // caseum ledger in volume-weighted cas1-equivalent units (ratio1 handled inside g); with
  // equal ratios the chain sum is conserved except influx term
  double gg = 1.0 / (1.0 + p.ratio1 / p.Kp_cas);
  double chain = x[CAS1] / gg;
  for (int i = 1; i < 6; ++i) chain += x[CAS1 + i];
  out.cas_resid_rel = std::abs(chain - x[CASIN]) / std::max(1e-300, std::abs(chain));
  return out;
}

// ---------- endpoints
struct End {
  double lysm_wk2_max, lysm_maint_min, lysf_maint_min, cpu_wk24_trough, cas1_w24, cas6_w24,
      cas_above_wcc;  // number of cas compartments above WCC90 at week 24
};
constexpr int NEND = 7;
const char* endName[NEND] = {"lysM_max_w0-2 (ng/g)", "lysM_min_w3-24 (ng/g)", "lysF_min_w3-24 (ng/g)",
                             "Cp_u_w24 (ng/mL)", "cas1_w24 (ng/g)", "cas6_w24 (ng/g)", "n_cas>WCC90_w24"};

std::array<double, NEND> endpoints(const Out& o) {
  std::array<double, NEND> e{};
  double mx = 0, mnm = 1e300, mnf = 1e300;
  for (size_t i = 0; i < o.t.size(); ++i) {
    if (o.t[i] <= 336) mx = std::max(mx, o.lys_m[i]);
    if (o.t[i] >= 504) { mnm = std::min(mnm, o.lys_m[i]); mnf = std::min(mnf, o.lys_f[i]); }
  }
  size_t last = o.t.size() - 1;
  e[0] = mx; e[1] = mnm; e[2] = mnf; e[3] = o.cp_u[last];
  e[4] = o.cas[0][last]; e[5] = o.cas[5][last];
  int nab = 0;
  for (int k = 0; k < 6; ++k) nab += (o.cas[k][last] > 2780.0);
  e[6] = nab;
  return e;
}

// ---------- uncertainty: parameters with reported SEs (+ optional assumption ranges)
struct UP { const char* name; double Params::*m; double sd_rel; };

std::vector<UP> budget(int which) {
  std::vector<UP> u = {
      // Karakitsios S8/S11/S13 (SE/estimate)
      {"Perm", &Params::Perm, 10.5 / 55.7},
      {"KpLYS/fu (mouse)", &Params::KpLYS_fu_mouse, 1.26e5 / 3.48e6},
      {"rate_mphi", &Params::rate_m, 1.12e4 / 1.43e5},
      {"Kp_mphi/(fu fni)", &Params::R_m, 2.03e5 / 5.68e6},
      {"rate_fmphi", &Params::rate_f, 1.48e3 / 5.78e3},
      {"Kp_fmphi/(fu fni)", &Params::R_f, 3.95e5 / 1.44e6},
      {"Kp_caseum", &Params::Kp_cas, 0.21 / 0.71},
      {"k_cas", &Params::kcas, 0.048 / 0.25},
      {"h", &Params::h, 0.05 / 0.75},
      // Svensson Table 3: SE ~ (CI width / 3.92)
      {"CL/F", &Params::CL, (2.75 - 2.49) / 3.92 / 2.62},
      {"V/F", &Params::V, (215.0 - 184.0) / 3.92 / 198.0},
      {"Q1/F", &Params::Q1, (3.97 - 3.35) / 3.92 / 3.66},
      {"VP1/F", &Params::VP1, (9230.0 - 7940.0) / 3.92 / 8550.0},
      {"Q2/F", &Params::Q2, (7.84 - 6.80) / 3.92 / 7.34},
      {"VP2/F", &Params::VP2, (2980.0 - 2390.0) / 3.92 / 2690.0},
  };
  if (which == 1) {
    // assumptions the paper fixed without uncertainty; spreads stated in the paper or
    // its sources (S10 tested fu_pls 0.00018-0.00144; Ufuk lysosomal 3.7-13.9 %).
    // Treated as lognormal with these relative SDs — an explicit, contestable choice.
    u.push_back({"fu_pls", &Params::fu_pls, 0.5});
    u.push_back({"f_lys (0.078)", &Params::flys, 0.4});
  }
  return u;
}
// pH assumptions handled as absolute SDs (in pH units) in budget 1
struct UPabs { const char* name; double Params::*m; double sd; };
std::vector<UPabs> budget_abs(int which) {
  if (which != 1) return {};
  return {{"pH_foamy (5.84)", &Params::pH_foamy, 0.2},
          {"pH_lys (5.0)", &Params::pH_LYS, 0.2},
          {"pH_IW human (6.69)", &Params::pH_IW, 0.2}};
}

int run_gum(int which, double dt) {
  Params p0;
  auto e0 = endpoints(simulate(p0, dt));
  auto ur = budget(which);
  auto ua = budget_abs(which);
  std::vector<std::array<double, NEND>> contrib;
  std::vector<std::string> names;
  for (auto& u : ur) {
    Params pp = p0, pm = p0;
    double v = p0.*(u.m), hstep = 0.01 * v;
    pp.*(u.m) = v + hstep; pm.*(u.m) = v - hstep;
    auto ep = endpoints(simulate(pp, dt)), em = endpoints(simulate(pm, dt));
    std::array<double, NEND> c{};
    for (int k = 0; k < NEND; ++k) { double s = (ep[k] - em[k]) / (2 * hstep); c[k] = std::pow(s * u.sd_rel * v, 2); }
    contrib.push_back(c); names.push_back(u.name);
  }
  for (auto& u : ua) {
    Params pp = p0, pm = p0;
    double v = p0.*(u.m), hstep = 0.01;
    pp.*(u.m) = v + hstep; pm.*(u.m) = v - hstep;
    auto ep = endpoints(simulate(pp, dt)), em = endpoints(simulate(pm, dt));
    std::array<double, NEND> c{};
    for (int k = 0; k < NEND; ++k) { double s = (ep[k] - em[k]) / (2 * hstep); c[k] = std::pow(s * u.sd, 2); }
    contrib.push_back(c); names.push_back(u.name);
  }
  std::printf("budget,%d\nendpoint,nominal,u_GUM1,rel", which);
  for (auto& n : names) std::printf(",%s", n.c_str());
  std::printf("\n");
  for (int k = 0; k < NEND - 1; ++k) {
    double var = 0;
    for (auto& c : contrib) var += c[k];
    std::printf("%s,%.6g,%.6g,%.4f", endName[k], e0[k], std::sqrt(var), std::sqrt(var) / e0[k]);
    for (auto& c : contrib) std::printf(",%.4f", var > 0 ? c[k] / var : 0.0);
    std::printf("\n");
  }
  return 0;
}

int run_mc(int which, int N, unsigned seed, double dt) {
  Params p0;
  auto ur = budget(which);
  auto ua = budget_abs(which);
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> z(0.0, 1.0);
  std::vector<std::array<double, NEND>> E;
  E.reserve(N);
  for (int i = 0; i < N; ++i) {
    Params p = p0;
    for (auto& u : ur) {
      double s2 = std::log(1.0 + u.sd_rel * u.sd_rel);  // lognormal with mean = nominal
      p.*(u.m) = (p0.*(u.m)) * std::exp(z(rng) * std::sqrt(s2) - 0.5 * s2);
    }
    for (auto& u : ua) p.*(u.m) = (p0.*(u.m)) + u.sd * z(rng);
    if (p.h > 1.0) p.h = 1.0;
    E.push_back(endpoints(simulate(p, dt)));
  }
  std::printf("budget,%d,N,%d,seed,%u\nendpoint,mean,sd,p2.5,p50,p97.5,P(>MIC50=60),P(>MIC90=120)\n", which, N, seed);
  for (int k = 0; k < NEND; ++k) {
    std::vector<double> v(N);
    for (int i = 0; i < N; ++i) v[i] = E[i][k];
    double m = std::accumulate(v.begin(), v.end(), 0.0) / N, s = 0;
    for (double a : v) s += (a - m) * (a - m);
    s = std::sqrt(s / (N - 1));
    std::sort(v.begin(), v.end());
    auto q = [&](double f) { return v[std::min<size_t>(N - 1, static_cast<size_t>(f * (N - 1) + 0.5))]; };
    double p50 = 0, p120 = 0;
    for (double a : v) { p50 += a > 60.0; p120 += a > 120.0; }
    std::printf("%s,%.6g,%.6g,%.6g,%.6g,%.6g,%.4f,%.4f\n", endName[k], m, s, q(0.025), q(0.5), q(0.975), p50 / N, p120 / N);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string_view mode = argc > 1 ? argv[1] : "nominal";
  if (mode == "nominal") {
    Params p;
    double dt = argc > 2 ? std::atof(argv[2]) : 0.25;
    if (argc > 3) p.ion = std::atoi(argv[3]);
    if (argc > 4) p.cas_sign_conservative = std::atoi(argv[4]) != 0;
    Out o = simulate(p, dt);
    Derived d = derive(p);
    std::fprintf(stderr, "ktr=%.4f ka=%.4f K_EWB=%.4f PS=%.4g W_EW=%.4g a_IW=%.4g KpLYS_fu_h=%.4g fIW=%.4f r=%.4f\n",
                 d.ktr, d.ka, d.K_EWB, d.PS, d.W_EW, d.a_IW, d.KpLYS_fu_h, d.fIW, d.r);
    std::fprintf(stderr, "plasma mass ledger rel residual = %.3e ; caseum chain ledger rel residual = %.3e\n",
                 o.mass_resid_rel, o.cas_resid_rel);
    std::printf("t,cp_u,cyt_m,lys_m,cyt_f,lys_f,outer,cas1,cas2,cas3,cas4,cas5,cas6\n");
    for (size_t i = 0; i < o.t.size(); ++i) {
      std::printf("%.2f,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g", o.t[i], o.cp_u[i], o.cyt_m[i], o.lys_m[i], o.cyt_f[i], o.lys_f[i], o.outer[i]);
      for (int k = 0; k < 6; ++k) std::printf(",%.6g", o.cas[k][i]);
      std::printf("\n");
    }
    return 0;
  }
  if (mode == "band") {  // per-hour 2.5/50/97.5 percentiles of lys_m under a budget
    int which = argc > 2 ? std::atoi(argv[2]) : 0;
    int N = argc > 3 ? std::atoi(argv[3]) : 300;
    Params p0;
    auto ur = budget(which);
    auto ua = budget_abs(which);
    std::mt19937_64 rng(1729);
    std::normal_distribution<double> z(0.0, 1.0);
    std::vector<std::vector<double>> L;
    std::vector<double> tt;
    for (int i = 0; i < N; ++i) {
      Params p = p0;
      for (auto& u : ur) {
        double s2 = std::log(1.0 + u.sd_rel * u.sd_rel);
        p.*(u.m) = (p0.*(u.m)) * std::exp(z(rng) * std::sqrt(s2) - 0.5 * s2);
      }
      for (auto& u : ua) p.*(u.m) = (p0.*(u.m)) + u.sd * z(rng);
      if (p.h > 1.0) p.h = 1.0;
      Out o = simulate(p, 0.25);
      L.push_back(o.lys_m);
      tt = o.t;
    }
    std::printf("t,p025,p50,p975\n");
    std::vector<double> col(N);
    for (size_t k = 0; k < tt.size(); ++k) {
      for (int i = 0; i < N; ++i) col[i] = L[i][k];
      std::sort(col.begin(), col.end());
      auto q = [&](double f) { return col[static_cast<size_t>(f * (N - 1) + 0.5)]; };
      std::printf("%.1f,%.6g,%.6g,%.6g\n", tt[k], q(0.025), q(0.5), q(0.975));
    }
    return 0;
  }
  if (mode == "conv") {
    Params p;
    double dts[] = {1.0, 0.5, 0.25, 0.125, 0.0625};
    std::array<double, NEND> prev{};
    std::printf("dt");
    for (auto n : endName) std::printf(",%s", n);
    std::printf(",plasma_ledger\n");
    for (double dt : dts) {
      Out o = simulate(p, dt);
      auto e = endpoints(o);
      std::printf("%.4f", dt);
      for (double v : e) std::printf(",%.10g", v);
      std::printf(",%.2e\n", o.mass_resid_rel);
      prev = e;
    }
    // tau sensitivity
    for (double tau : {1e-6, 1e-4, 1e-3, 1e-2, 1e-1}) {
      Params q; q.tau = tau;
      auto e = endpoints(simulate(q, 0.125));
      std::printf("tau=%.0e,cas1=%.6g,cas6=%.6g,lysF=%.6g\n", tau, e[4], e[5], e[2]);
    }
    return 0;
  }
  if (mode == "ion") {
    for (int ion = 0; ion < 3; ++ion) {
      Params p; p.ion = ion;
      Derived d = derive(p);
      auto e = endpoints(simulate(p, 0.25));
      std::printf("ion=%d W_EW=%.4g a_IW=%.4g fniIW(5.84)/fniLYS(5.0)=%.4g", ion, d.W_EW, d.a_IW,
                  fni(p, 5.84) / fni(p, 5.0));
      for (int k = 0; k < NEND; ++k) std::printf(" | %s=%.4g", endName[k], e[k]);
      std::printf("\n");
    }
    for (int s = 0; s < 2; ++s) {
      Params p; p.cas_sign_conservative = (s == 1);
      auto e = endpoints(simulate(p, 0.25));
      std::printf("cas_sign_conservative=%d cas1_w24=%.5g cas6_w24=%.5g\n", s, e[4], e[5]);
    }
    return 0;
  }
  if (mode == "parity") {  // same lines and order as demos/aris_bdq/bdq_lung.sio
    Params p;
    Out o = simulate(p, 0.25);
    auto e = endpoints(o);
    double cpumax = 0;
    for (size_t i = 0; i < o.t.size(); ++i)
      if (o.t[i] <= 336) cpumax = std::max(cpumax, o.cp_u[i]);
    size_t last = o.t.size() - 1;
    std::printf("BDQ|cpu_max_w0-2_ng_per_mL=%.6f\n", cpumax);
    std::printf("BDQ|lysM_max_w0-2_ng_per_g=%.6f\n", e[0]);
    std::printf("BDQ|lysM_min_w3-24_ng_per_g=%.6f\n", e[1]);
    std::printf("BDQ|lysF_min_w3-24_ng_per_g=%.6f\n", e[2]);
    std::printf("BDQ|cpu_w24_ng_per_mL=%.6f\n", e[3]);
    std::printf("BDQ|cas1_w24_ng_per_g=%.6f\n", e[4]);
    std::printf("BDQ|cas2_w24_ng_per_g=%.6f\n", o.cas[1][last]);
    std::printf("BDQ|cas6_w24_ng_per_g=%.6f\n", e[5]);
    // first-order GUM, reported-SE budget (budget 0), same as the Sounio GUM| lines
    double vl = 0, vc = 0;
    for (auto& u : budget(0)) {
      Params pp = p, pm = p;
      pp.*(u.m) = p.*(u.m) * 1.01; pm.*(u.m) = p.*(u.m) * 0.99;
      auto ep = endpoints(simulate(pp, 0.25)), em = endpoints(simulate(pm, 0.25));
      double dl = (ep[1] - em[1]) / 0.02 * u.sd_rel, dc = (ep[5] - em[5]) / 0.02 * u.sd_rel;
      vl += dl * dl; vc += dc * dc;
    }
    std::printf("GUM|lysM_min_w3-24|u=%.6f\n", std::sqrt(vl));
    std::printf("GUM|cas6_w24|u=%.6f\n", std::sqrt(vc));
    return 0;
  }
  if (mode == "gumcov") {
    // Caseum parameters with the covariance recovered by mouse_fit.cpp (refit of Table S14
    // with the Irwin 2016 C3HeB/FeJ plasma model): log-scale SDs and correlations.
    // Human caseum endpoints at week 24, first-order GUM: full covariance vs diagonal.
    Params p0;
    double Params::*m[5] = {&Params::rate_f, &Params::R_f, &Params::Kp_cas, &Params::kcas, &Params::h};
    const char* nm[5] = {"rate_fmphi", "Kp_fmphi/(fu fni)", "Kp_caseum", "k_cas", "h"};
    double sd[5] = {0.348, 0.381, 0.403, 0.248, 0.093};
    double R[5][5] = {{1.000, 0.150, -0.416, -0.464, -0.270},
                      {0.150, 1.000, -0.679, 0.316, 0.251},
                      {-0.416, -0.679, 1.000, -0.222, 0.102},
                      {-0.464, 0.316, -0.222, 1.000, 0.723},
                      {-0.270, 0.251, 0.102, 0.723, 1.000}};
    auto e0 = endpoints(simulate(p0, 0.25));
    double g[5][NEND];
    for (int i = 0; i < 5; ++i) {
      Params pp = p0, pm = p0;
      pp.*(m[i]) *= std::exp(0.01); pm.*(m[i]) *= std::exp(-0.01);
      auto ep = endpoints(simulate(pp, 0.25)), em = endpoints(simulate(pm, 0.25));
      for (int k = 0; k < NEND; ++k) g[i][k] = (ep[k] - em[k]) / 0.02;  // dE/dlog(theta)
    }
    std::printf("endpoint,nominal,u_diagonal,u_full_covariance,ratio\n");
    for (int k : {2, 4, 5}) {
      double vd = 0, vf = 0;
      for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j) {
          double c = R[i][j] * sd[i] * sd[j] * g[i][k] * g[j][k];
          vf += c;
          if (i == j) vd += c;
        }
      std::printf("%s,%.6g,%.6g,%.6g,%.3f\n", endName[k], e0[k], std::sqrt(vd), std::sqrt(vf), std::sqrt(vf / vd));
    }
    (void)nm;
    return 0;
  }
  if (mode == "gum") return run_gum(argc > 2 ? std::atoi(argv[2]) : 0, 0.25);
  if (mode == "mc")
    return run_mc(argc > 2 ? std::atoi(argv[2]) : 0, argc > 3 ? std::atoi(argv[3]) : 1000,
                  argc > 4 ? static_cast<unsigned>(std::atoi(argv[4])) : 1729u, 0.25);
  std::fprintf(stderr, "unknown mode\n");
  return 1;
}
