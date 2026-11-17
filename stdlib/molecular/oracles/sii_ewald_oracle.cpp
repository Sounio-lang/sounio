// sII proton configuration (ice rules, zero bond winding), SPC/E host charges,
// Alavi three-site H2, tinfoil Ewald for guest-host and guest-guest electrostatics.
// Oracle for molecular::host_guest charged guests. long double throughout.
// Predictions were written before this file ran: predictions.txt (sha256 in predictions.sha256).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using R = long double;
using V = std::array<R, 3>;
static const R PI = 3.141592653589793238462643383279502884L;
static const R KE_KJ = 1389.35457644382L;            // e^2/(4 pi eps0) in kJ/mol A
static const R KB_KJ = 0.00831446261815324L;         // kJ/mol/K
static const R A = 17.31L;

static V add(V a, V b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
static V sub(V a, V b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
static V mul(V a, R s) { return {a[0] * s, a[1] * s, a[2] * s}; }
static R dot(V a, V b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
static R nrm(V a) { return std::sqrt(dot(a, a)); }
static V unit(V a) { return mul(a, 1 / nrm(a)); }

// ---------------- Fd-3m, origin choice 2 (same generators as host_guest) ----------------
struct Op { int r[3][3]; R t[3]; };
static R fr(R v) { v -= std::floor(v); if (v > 1 - 1e-12L) v = 0; return v; }
static V act(const Op& o, const V& p) { V q; for (int i = 0; i < 3; ++i) { q[i] = o.t[i]; for (int j = 0; j < 3; ++j) q[i] += o.r[i][j] * p[j]; q[i] = fr(q[i]); } return q; }
static Op comp(const Op& a, const Op& b) { Op c{}; for (int i = 0; i < 3; ++i) { c.t[i] = a.t[i]; for (int j = 0; j < 3; ++j) { c.t[i] += a.r[i][j] * b.t[j]; c.r[i][j] = 0; for (int k = 0; k < 3; ++k) c.r[i][j] += a.r[i][k] * b.r[k][j]; } c.t[i] = fr(c.t[i]); } return c; }
static bool same(const Op& a, const Op& b) { for (int i = 0; i < 3; ++i) { for (int j = 0; j < 3; ++j) if (a.r[i][j] != b.r[i][j]) return false; R d = std::fabs(a.t[i] - b.t[i]); if (d > 1e-9L && d < 1 - 1e-9L) return false; } return true; }
static std::vector<Op> group() {
  std::vector<Op> g{
      {{{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}}, {0.75L, 0.25L, 0.5L}},
      {{{-1, 0, 0}, {0, 1, 0}, {0, 0, -1}}, {0.25L, 0.5L, 0.75L}},
      {{{0, 0, 1}, {1, 0, 0}, {0, 1, 0}}, {0, 0, 0}},
      {{{0, 1, 0}, {1, 0, 0}, {0, 0, -1}}, {0.75L, 0.25L, 0.5L}},
      {{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}}, {0, 0, 0}},
      {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {0, 0.5L, 0.5L}},
      {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {0.5L, 0, 0.5L}},
      {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {0.5L, 0.5L, 0}}};
  std::vector<Op> G{{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {0, 0, 0}}};
  for (bool grew = true; grew;) {
    grew = false; size_t n = G.size();
    for (size_t i = 0; i < n; ++i) for (auto& h : g) { Op c = comp(h, G[i]); bool f = false; for (auto& e : G) if (same(e, c)) { f = true; break; } if (!f) { G.push_back(c); grew = true; } }
  }
  return G;
}
static std::vector<V> orbit(const std::vector<Op>& G, V p) {
  std::vector<V> o;
  for (auto& op : G) { V q = act(op, p); bool f = false; for (auto& e : o) { R d = 0; for (int i = 0; i < 3; ++i) { R x = std::fabs(e[i] - q[i]); x = std::min(x, 1 - x); d += x * x; } if (d < 1e-14L) { f = true; break; } } if (!f) o.push_back(q); }
  return o;
}

// ---------------- proton graph ----------------
struct Edge { int u, v; int s[3]; };   // bond vector u->v = r_v + s*A - r_u (cartesian)
struct Host { std::vector<V> O; std::vector<Edge> E; std::vector<int> dir; };  // dir[e]=+1: u donates to v

static V bondvec(const Host& h, int e, int sign) {
  const Edge& ed = h.E[e];
  V b; for (int i = 0; i < 3; ++i) b[i] = h.O[ed.v][i] + ed.s[i] * A - h.O[ed.u][i];
  return sign > 0 ? b : mul(b, -1);
}
static std::array<long, 3> winding(const Host& h) {
  std::array<long, 3> w{0, 0, 0};
  for (size_t e = 0; e < h.E.size(); ++e) for (int i = 0; i < 3; ++i) w[i] += (long)h.dir[e] * h.E[e].s[i];
  return w;
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  const unsigned long long seed = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 20261116ULL;
  auto G = group();
  auto o8 = orbit(G, {0.125L, 0.125L, 0.125L}), o32 = orbit(G, {0.21796L, 0.21796L, 0.21796L}), o96 = orbit(G, {0.18198L, 0.18198L, 0.37081L});
  Host h;
  for (auto* o : {&o8, &o32, &o96}) for (auto& f : *o) h.O.push_back(mul(f, A));
  const int NO = (int)h.O.size();
  std::printf("group %zu  O %d (8+32+96 = %zu+%zu+%zu)\n", G.size(), NO, o8.size(), o32.size(), o96.size());
  // bonds: u<v pairs (or image pairs) within 3.2 A
  for (int u = 0; u < NO; ++u) for (int v = u; v < NO; ++v) for (int a = -1; a <= 1; ++a) for (int b = -1; b <= 1; ++b) for (int c = -1; c <= 1; ++c) {
    if (u == v) continue;
    V d{h.O[v][0] + a * A - h.O[u][0], h.O[v][1] + b * A - h.O[u][1], h.O[v][2] + c * A - h.O[u][2]};
    R r = nrm(d); if (r < 3.2L) h.E.push_back({u, v, {a, b, c}});
  }
  const int NE = (int)h.E.size();
  std::vector<std::vector<int>> inc(NO);
  for (int e = 0; e < NE; ++e) { inc[h.E[e].u].push_back(e); inc[h.E[e].v].push_back(e); }
  int deg_bad = 0; R bmin = 9, bmax = 0;
  for (int v = 0; v < NO; ++v) if (inc[v].size() != 4) ++deg_bad;
  for (int e = 0; e < NE; ++e) { R r = nrm(bondvec(h, e, 1)); bmin = std::min(bmin, r); bmax = std::max(bmax, r); }
  std::printf("bonds %d  degree!=4: %d  O-O %.5Lf..%.5Lf\n", NE, deg_bad, bmin, bmax);
  // parity invariant of the winding: sum |s| mod 2 over undirected edges
  long par[3] = {0, 0, 0};
  for (auto& e : h.E) for (int i = 0; i < 3; ++i) par[i] += std::labs(e.s[i]);
  std::printf("winding parity %ld %ld %ld\n", par[0] % 2, par[1] % 2, par[2] % 2);

  // Euler circuit (Hierholzer) -> in = out = 2 at every vertex
  h.dir.assign(NE, 0);
  {
    std::vector<int> used(NE, 0), ptr(NO, 0), stack{0};
    std::vector<int> estack;  // edge taken to reach stack top, with orientation
    std::vector<std::pair<int, int>> path;  // (edge, sign) in circuit order
    std::vector<std::pair<int, int>> es;
    std::vector<int> vs{0};
    while (!vs.empty()) {
      int v = vs.back();
      while (ptr[v] < (int)inc[v].size() && used[inc[v][ptr[v]]]) ++ptr[v];
      if (ptr[v] == (int)inc[v].size()) { vs.pop_back(); if (!es.empty()) { path.push_back(es.back()); es.pop_back(); } continue; }
      int e = inc[v][ptr[v]]; used[e] = 1;
      int w = h.E[e].u == v ? h.E[e].v : h.E[e].u; int sg = h.E[e].u == v ? 1 : -1;
      es.push_back({e, sg}); vs.push_back(w);
    }
    for (auto [e, sg] : path) h.dir[e] = sg;
    int z = 0; for (int e = 0; e < NE; ++e) if (h.dir[e] == 0) ++z;
    std::printf("euler circuit length %zu  unoriented %d\n", path.size(), z);
  }
  auto out_in_ok = [&]() { int bad = 0; for (int v = 0; v < NO; ++v) { int o = 0; for (int e : inc[v]) { bool tail = (h.E[e].u == v) == (h.dir[e] > 0); o += tail; } if (o != 2) ++bad; } return bad; };
  std::printf("ice-rule violations after euler: %d\n", out_in_ok());
  { long fp = 0; for (int e = 0; e < NE; ++e) fp += (long)h.dir[e] * (e + 1); long fo = 0; for (int v = 0; v < NO; ++v) fo += (long)std::llround(h.O[v][0] * 1000) * (v + 1); long fe = 0; for (int e = 0; e < NE; ++e) fe += ((long)h.E[e].u * 7 + (long)h.E[e].v * 3 + (long)h.E[e].s[0] * 5) * (e + 1); std::printf("fingerprint euler %ld oxy %ld edges %ld\n", fp, fo, fe); }

  // loop moves: random directed walk until a vertex repeats, reverse the cycle.
  long long rs = (long long)(seed % 2147483646ULL) + 1;
  auto rng = [&]() { rs = (rs * 48271LL) % 2147483647LL; return (unsigned long long)rs; };
  auto w2 = [](std::array<long, 3> w) { return w[0] * w[0] + w[1] * w[1] + w[2] * w[2]; };
  auto W = winding(h);
  std::printf("initial winding %ld %ld %ld\n", W[0], W[1], W[2]);
  long acc_local = 0, acc_wind = 0, steps = 0;
  const long NSTEP = 200000;
  for (steps = 0; steps < NSTEP || w2(W) != 0; ++steps) {
    if (steps > 50 * NSTEP) break;
    int v0 = (int)(rng() % NO);
    std::vector<int> seen(NO, -1), ev, vv;
    int v = v0;
    while (seen[v] < 0) {
      seen[v] = (int)vv.size(); vv.push_back(v);
      int outs[2], k = 0;
      for (int e : inc[v]) { bool tail = (h.E[e].u == v) == (h.dir[e] > 0); if (tail) outs[k++] = e; }
      int e = outs[rng() % 2]; ev.push_back(e);
      v = h.E[e].u == v ? h.E[e].v : h.E[e].u;
    }
    std::array<long, 3> dw{0, 0, 0};
    for (size_t i = seen[v]; i < ev.size(); ++i) for (int c = 0; c < 3; ++c) dw[c] += (long)h.dir[ev[i]] * h.E[ev[i]].s[c];
    std::array<long, 3> Wn{W[0] - 2 * dw[0], W[1] - 2 * dw[1], W[2] - 2 * dw[2]};
    bool wind = w2(dw) != 0;
    bool accept = !wind ? true : (w2(Wn) < w2(W) || (steps < NSTEP && w2(Wn) == w2(W)));
    if (accept) { for (size_t i = seen[v]; i < ev.size(); ++i) h.dir[ev[i]] = -h.dir[ev[i]]; W = Wn; if (wind) ++acc_wind; else ++acc_local; }
  }
  std::printf("loop moves %ld  local %ld  winding %ld  final W %ld %ld %ld  ice-rule violations %d\n", steps, acc_local, acc_wind, W[0], W[1], W[2], out_in_ok());

  // SPC/E sites
  const R qO = -0.8476L, qHw = 0.4238L, th = 109.47L / 2 * PI / 180;
  std::vector<V> site; std::vector<R> q;
  V M{0, 0, 0}; R hh_min = 9, hh_max = 0;
  for (int v = 0; v < NO; ++v) {
    V u[2]; int k = 0;
    for (int e : inc[v]) { bool tail = (h.E[e].u == v) == (h.dir[e] > 0); if (tail) { V b = bondvec(h, e, h.E[e].u == v ? 1 : -1); u[k++] = unit(b); } }
    V b = unit(add(u[0], u[1])), p = unit(sub(u[0], u[1]));
    V h1 = add(h.O[v], add(mul(b, std::cos(th)), mul(p, std::sin(th))));
    V h2 = add(h.O[v], sub(mul(b, std::cos(th)), mul(p, std::sin(th))));
    site.push_back(h.O[v]); q.push_back(qO);
    site.push_back(h1); q.push_back(qHw);
    site.push_back(h2); q.push_back(qHw);
    M = add(M, add(mul(h1, qHw), add(mul(h2, qHw), mul(h.O[v], qO))));
    // angle of the H to the O-O bond it donates along
    R a1 = std::acos(std::min((R)1, dot(unit(sub(h1, h.O[v])), u[0]))) * 180 / PI;
    hh_min = std::min(hh_min, a1); hh_max = std::max(hh_max, a1);
  }
  // M above uses absolute positions of a neutral molecule set: independent of origin.
  std::printf("SPC/E host: %zu sites  dipole M = (%.6Lf %.6Lf %.6Lf) |M| = %.6Lf e A (aligned bound %.2Lf)  H off-bond angle %.3Lf..%.3Lf deg\n",
              site.size(), M[0], M[1], M[2], nrm(M), NO * 2 * qHw * std::cos(th), hh_min, hh_max);
  // dump host for the Sounio side: oxygen index, donor neighbour list
  if (FILE* f = std::fopen("sii_protons.txt", "w")) {
    std::fprintf(f, "# seed %llu  winding %ld %ld %ld\n", seed, W[0], W[1], W[2]);
    for (size_t i = 0; i < site.size(); ++i) std::fprintf(f, "%.12Lf %.12Lf %.12Lf %.4Lf\n", site[i][0], site[i][1], site[i][2], q[i]);
    std::fclose(f);
  }

  // ---------------- Ewald (tinfoil) ----------------
  const R Vol = A * A * A;
  auto erfc_l = [](R x) { return std::erfc(x); };
  struct Ew { R alpha, rc; int kmax; std::vector<std::array<int, 3>> ks; std::vector<R> kf; std::vector<R> Sre, Sim; };
  auto make_ew = [&](R alpha) {
    Ew w; w.alpha = alpha; w.rc = 5.6L / alpha; if (w.rc > A - 0.01L) w.rc = 5.6L / 0.35L; R kc = 2 * alpha * 5.6L; w.kmax = (int)std::ceil(kc * A / (2 * PI));
    for (int a = -w.kmax; a <= w.kmax; ++a) for (int b = -w.kmax; b <= w.kmax; ++b) for (int c = -w.kmax; c <= w.kmax; ++c) {
      if (!a && !b && !c) continue;
      V k{2 * PI * a / A, 2 * PI * b / A, 2 * PI * c / A}; R k2 = dot(k, k); if (k2 > kc * kc) continue;
      w.ks.push_back({a, b, c}); w.kf.push_back(4 * PI / Vol * std::exp(-k2 / (4 * alpha * alpha)) / k2);
      R sr = 0, si = 0; for (size_t j = 0; j < site.size(); ++j) { R ph = dot(k, site[j]); sr += q[j] * std::cos(ph); si += q[j] * std::sin(ph); }
      w.Sre.push_back(sr); w.Sim.push_back(si);
    }
    return w;
  };
  auto phi_ew = [&](const Ew& w, V r) {  // host potential at r, e/A
    R s = 0; int n = (int)std::ceil(w.rc / A) + 1;
    for (size_t j = 0; j < site.size(); ++j) for (int a = -n; a <= n; ++a) for (int b = -n; b <= n; ++b) for (int c = -n; c <= n; ++c) {
      V d{r[0] - site[j][0] - a * A, r[1] - site[j][1] - b * A, r[2] - site[j][2] - c * A}; R x = nrm(d);
      if (x < w.rc) s += q[j] * erfc_l(w.alpha * x) / x;
    }
    for (size_t i = 0; i < w.ks.size(); ++i) { V k{2 * PI * w.ks[i][0] / A, 2 * PI * w.ks[i][1] / A, 2 * PI * w.ks[i][2] / A}; R ph = dot(k, r); s += w.kf[i] * (w.Sre[i] * std::cos(ph) + w.Sim[i] * std::sin(ph)); }
    return s;
  };
  auto phi_direct = [&](V r, int N) {  // cubic image shells |n|_inf <= N
    R s = 0;
    for (int a = -N; a <= N; ++a) for (int b = -N; b <= N; ++b) for (int c = -N; c <= N; ++c)
      for (size_t j = 0; j < site.size(); ++j) { V d{r[0] - site[j][0] - a * A, r[1] - site[j][1] - b * A, r[2] - site[j][2] - c * A}; s += q[j] / nrm(d); }
    return s;
  };
  // H2 Alavi
  const R dH2 = 0.7414L, qH2 = 0.4932L;
  auto h2_sites = [&](V c, V n, V out[3]) { out[0] = add(c, mul(n, dH2 / 2)); out[1] = sub(c, mul(n, dH2 / 2)); out[2] = c; };
  auto e_gh = [&](auto&& phi, V c, V n) { V s[3]; h2_sites(c, n, s); return KE_KJ * (qH2 * phi(s[0]) + qH2 * phi(s[1]) - 2 * qH2 * phi(s[2])); };

  // cage centres
  auto c16 = orbit(G, {0, 0, 0}), c8 = orbit(G, {0.375L, 0.375L, 0.375L});
  std::vector<V> cen; for (auto& f : c16) cen.push_back(mul(f, A)); for (auto& f : c8) cen.push_back(mul(f, A));
  std::printf("cages %zu small + %zu large\n", c16.size(), c8.size());

  Ew e25 = make_ew(0.35L), e30 = make_ew(0.40L), e35 = make_ew(0.45L);
  std::printf("ewald k-vectors: %zu %zu %zu  rc %.2Lf %.2Lf %.2Lf\n", e25.ks.size(), e30.ks.size(), e35.ks.size(), e25.rc, e30.rc, e35.rc);
  std::printf("EXPECT M %.15Le %.15Le %.15Le\n", M[0], M[1], M[2]);
  std::printf("EXPECT phi123 %.15Le\n", phi_ew(e25, V{1, 2, 3}));
  // P3/P4 at an off-centre point in a small and a large cage, tilted axis
  V nt = unit(V{0.3L, -0.5L, 0.81L});
  for (int ci : {0, 16}) {
    V c = add(cen[ci], V{0.31L, -0.17L, 0.22L});
    R a = e_gh([&](V r) { return phi_ew(e25, r); }, c, nt), b = e_gh([&](V r) { return phi_ew(e30, r); }, c, nt), d = e_gh([&](V r) { return phi_ew(e35, r); }, c, nt);
    std::printf("EXPECT p3 %d %.17Le %.17Le %.17Le %.15Le %.15Le\n", ci, c[0], c[1], c[2], a, b);
    std::printf("P3 cage %d: E(a=0.35)=%.12Lf E(0.40)=%.12Lf E(0.45)=%.12Lf kJ/mol  rel spread %.2Le\n", ci, a, b, d, (std::max({a, b, d}) - std::min({a, b, d})) / std::fabs(b));
    for (int N : {4, 8, 16, 24}) { R e = e_gh([&](V r) { return phi_direct(r, N); }, c, nt); std::printf("   P4 direct N=%2d: %.12Lf  rel %.2Le\n", N, e, (e - b) / std::fabs(b)); }
  }

  // P5/P6/P7: orientation scan at every cage centre (Fibonacci sphere, 2000 directions)
  const int ND = 240;
  std::vector<V> dirs; for (int i = 0; i < ND; ++i) { R z = 1 - (2 * i + 1) / (R)ND, rr = std::sqrt(1 - z * z), ph = i * PI * (3 - std::sqrt((R)5)); dirs.push_back({rr * std::cos(ph), rr * std::sin(ph), z}); }
  R hgap = 1e-3L;
  std::printf("cage  type   <E>_Omega    min        max       spread     <E>/spread  quad-limit rel.err   C_ratio(250K)\n");
  for (size_t ci = 0; ci < cen.size(); ++ci) {
    auto phi = [&](V r) { return phi_ew(e25, r); };
    V c = cen[ci];
    // Hessian of phi by central differences (for P5)
    R H[3][3]; R p0 = phi(c);
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
      V ei{0, 0, 0}, ej{0, 0, 0}; ei[i] = hgap; ej[j] = hgap;
      H[i][j] = (phi(add(c, add(ei, ej))) - phi(add(c, sub(ei, ej))) - phi(sub(c, sub(ei, ej))) + phi(sub(c, add(ei, ej)))) / (4 * hgap * hgap);
    }
    (void)p0;
    R tr = H[0][0] + H[1][1] + H[2][2];
    R sum = 0, mn = 1e9, mx = -1e9, boltz = 0, maxrel = 0, qmn = 1e9, qmx = -1e9;
    const R kT = KB_KJ * 250;
    for (int i = 0; i < ND; ++i) {
      V n = dirs[i];
      R quad = 0; for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) quad += n[a] * n[b] * H[a][b];
      R eq = KE_KJ * 0.5L * 2 * qH2 * (dH2 / 2) * (dH2 / 2) * quad;
      R ex = e_gh(phi, c, n);
      maxrel = std::max(maxrel, std::fabs(ex - eq));
      sum += ex; mn = std::min(mn, ex); mx = std::max(mx, ex); boltz += std::exp(-ex / kT);
      qmn = std::min(qmn, eq); qmx = std::max(qmx, eq);
    }
    R spread = mx - mn;
    std::printf("EXPECT cage %zu %.15Le %.15Le %.15Le %.15Le\n", ci, sum / ND, mn, mx, boltz / ND);
    std::printf("%3zu  %s  %+.6Lf  %+.5Lf  %+.5Lf  %.5Lf  %+.2Le   %.2Le (tr H %.1Le)   %.6Lf\n", ci, ci < 16 ? "small" : "large", sum / ND, mn, mx, spread, (sum / ND) / spread, maxrel / spread, tr, boltz / ND);
  }

  // P8: guest-guest Ewald vs direct image sum for 5 H2 in the cell
  std::vector<V> gc, gn; std::mt19937_64 g2(7);
  std::uniform_real_distribution<double> U(0, 1);
  for (int m = 0; m < 5; ++m) { gc.push_back(add(cen[m * 4], V{0.2L * (R)U(g2), -0.2L * (R)U(g2), 0.1L})); R z = 2 * (R)U(g2) - 1, ph = 2 * PI * (R)U(g2), rr = std::sqrt(1 - z * z); gn.push_back({rr * std::cos(ph), rr * std::sin(ph), z}); }
  std::vector<V> gs; std::vector<R> gq; std::vector<int> gm;
  for (int m = 0; m < 5; ++m) { V s[3]; h2_sites(gc[m], gn[m], s); for (int k = 0; k < 3; ++k) { gs.push_back(s[k]); gq.push_back(k < 2 ? qH2 : -2 * qH2); gm.push_back(m); } }
  auto gg_ewald = [&](R alpha) {
    R rc = 5.6L / alpha, kc = 2 * alpha * 5.6L; int kmax = (int)std::ceil(kc * A / (2 * PI)), n = (int)std::ceil(rc / A) + 1;
    R er = 0;
    for (size_t i = 0; i < gs.size(); ++i) for (size_t j = 0; j < gs.size(); ++j) for (int a = -n; a <= n; ++a) for (int b = -n; b <= n; ++b) for (int c = -n; c <= n; ++c) {
      if (i == j && !a && !b && !c) continue;
      bool same_mol_cell = gm[i] == gm[j] && !a && !b && !c;
      V d{gs[i][0] - gs[j][0] - a * A, gs[i][1] - gs[j][1] - b * A, gs[i][2] - gs[j][2] - c * A}; R x = nrm(d);
      if (same_mol_cell) { er -= 0.5L * gq[i] * gq[j] * std::erf(alpha * x) / x; continue; }
      if (x < rc) er += 0.5L * gq[i] * gq[j] * std::erfc(alpha * x) / x;
    }
    R ek = 0;
    for (int a = -kmax; a <= kmax; ++a) for (int b = -kmax; b <= kmax; ++b) for (int c = -kmax; c <= kmax; ++c) {
      if (!a && !b && !c) continue; V k{2 * PI * a / A, 2 * PI * b / A, 2 * PI * c / A}; R k2 = dot(k, k); if (k2 > kc * kc) continue;
      R sr = 0, si = 0; for (size_t j = 0; j < gs.size(); ++j) { R ph = dot(k, gs[j]); sr += gq[j] * std::cos(ph); si += gq[j] * std::sin(ph); }
      ek += 2 * PI / Vol * std::exp(-k2 / (4 * alpha * alpha)) / k2 * (sr * sr + si * si);
    }
    R es = 0; for (R qq : gq) es -= alpha / std::sqrt(PI) * qq * qq;
    return KE_KJ * (er + ek + es);
  };
  auto gg_direct = [&](int N) {
    R s = 0;
    for (size_t i = 0; i < gs.size(); ++i) for (size_t j = 0; j < gs.size(); ++j) for (int a = -N; a <= N; ++a) for (int b = -N; b <= N; ++b) for (int c = -N; c <= N; ++c) {
      if (gm[i] == gm[j] && !a && !b && !c) continue;
      V d{gs[i][0] - gs[j][0] - a * A, gs[i][1] - gs[j][1] - b * A, gs[i][2] - gs[j][2] - c * A}; s += 0.5L * gq[i] * gq[j] / nrm(d);
    }
    return KE_KJ * s;
  };
  R g25 = gg_ewald(0.35L), g30 = gg_ewald(0.40L), g35 = gg_ewald(0.45L);
  std::printf("EXPECT gg %.15Le\n", g25);
  for (int m = 0; m < 5; ++m) std::printf("EXPECT gmol %d %.17Le %.17Le %.17Le %.17Le %.17Le %.17Le\n", m, gc[m][0], gc[m][1], gc[m][2], gn[m][0], gn[m][1], gn[m][2]);
  std::printf("P8 guest-guest Ewald: %.12Lf %.12Lf %.12Lf kJ/mol  rel spread %.2Le\n", g25, g30, g35, (std::max({g25, g30, g35}) - std::min({g25, g30, g35})) / std::fabs(g30));
  for (int N : {2, 4, 8, 16}) { R d = gg_direct(N); std::printf("   direct N=%2d: %.12Lf  rel %.2Le\n", N, d, (d - g30) / std::fabs(g30)); }
  return 0;
}
