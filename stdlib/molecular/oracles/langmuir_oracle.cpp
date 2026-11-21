// Oracle for molecular::langmuir (tests/stdlib/molecular/test_langmuir_oracle.sio).
// The lattice, the proton configuration and the SPC/E sites are those of sii_ewald_oracle.cpp
// (long double). Everything the library computes is then mirrored in double precision on the
// same random stream: molecular::lj's xorshift128+ (mol_seed / mol_uniform), Box-Muller normals,
// Levy bridges, rejection draws, the path-integral ratio, the pair ratios and the charged-guest
// quadrature. Predictions were written before this file: langmuir_predictions.txt
// (sha256 in langmuir_predictions.sha256).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstdint>
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

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  const unsigned long long seed = 20261116ULL;
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

  // ======================= double-precision mirror of molecular::langmuir =======================
  const double a = 17.31, T = 250.0, cut = 12.0, rho = 136.0 / (a * a * a);
  // host sites: 27 images of the 136 oxygens (hg_replicate with m = 1)
  std::vector<std::array<double, 3>> host;
  for (int ix = -1; ix <= 1; ++ix) for (int iy = -1; iy <= 1; ++iy) for (int iz = -1; iz <= 1; ++iz)
    for (int v = 0; v < NO; ++v) host.push_back({(double)h.O[v][0] + ix * a, (double)h.O[v][1] + iy * a, (double)h.O[v][2] + iz * a});
  std::vector<std::array<double, 3>> cenv;
  for (auto& f : orbit(G, {0, 0, 0})) cenv.push_back({(double)(f[0] * A), (double)(f[1] * A), (double)(f[2] * A)});
  for (auto& f : orbit(G, {0.375L, 0.375L, 0.375L})) cenv.push_back({(double)(f[0] * A), (double)(f[1] * A), (double)(f[2] * A)});
  const int NC = (int)cenv.size();
  auto mimg = [&](double d) { double hh = 0.5 * a; if (d > hh) return d - a; if (d < -hh) return d + a; return d; };
  auto cage_of = [&](double x, double y, double z) { double best = 1e300; int bi = 0;
    for (int c = 0; c < NC; ++c) { double dx = mimg(x - cenv[c][0]), dy = mimg(y - cenv[c][1]), dz = mimg(z - cenv[c][2]), r2 = dx * dx + dy * dy + dz * dz; if (r2 < best) { best = r2; bi = c; } }
    return bi; };
  auto wrapf = [&](double x) { return x - a * std::floor(x / a); };
  auto tail_of = [&](double eps, double sig) { double s3 = sig * sig * sig, s6 = s3 * s3, s12 = s6 * s6, c3 = cut * cut * cut, c9 = c3 * c3 * c3;
    return 16.0 * 3.14159265358979323846 * rho * eps * (s12 / (9.0 * c9) - s6 / (3.0 * c3)); };
  auto field = [&](double x, double y, double z, double eps, double sig, double lam) {
    x = wrapf(x); y = wrapf(y); z = wrapf(z);
    double s2 = sig * sig, s6 = s2 * s2 * s2, s12 = s6 * s6, c2 = cut * cut, u = 0, l = 0;
    for (auto& s : host) { double dx = x - s[0], dy = y - s[1], dz = z - s[2], r2 = dx * dx + dy * dy + dz * dz; if (r2 > c2) continue;
      double i2 = 1 / r2, i6 = i2 * i2 * i2, aa = s12 * i6 * i6, bb = s6 * i6; u += 4 * eps * (aa - bb); l += 4 * eps * (132 * aa - 30 * bb) * i2; }
    return u + tail_of(eps, sig) + lam * l; };
  // xorshift128+ of molecular::lj
  int64_t RA = 314159265, RB = 271828182;
  auto mol_seed = [&](int64_t s) { if (s <= 0) s = 987654321; int64_t aa = (1103515245LL * s + 12345) % 2147483648LL, bb = (1103515245LL * aa + 12345) % 2147483648LL;
    RA = aa * 65536 + 314159265; RB = bb * 65536 + 271828182;
    for (int i = 0; i < 16; ++i) { uint64_t x = (uint64_t)RA, y = (uint64_t)RB; RA = (int64_t)y; x ^= x << 23; x ^= (x >> 17) & 140737488355327ULL; x ^= y ^ ((y >> 26) & 274877906943ULL); RB = (int64_t)x; } };
  auto mol_uniform = [&]() { uint64_t x = (uint64_t)RA, y = (uint64_t)RB; RA = (int64_t)y; x ^= x << 23; x ^= (x >> 17) & 140737488355327ULL; x ^= y ^ ((y >> 26) & 274877906943ULL); RB = (int64_t)x;
    uint64_t r = ((x + y) >> 11) & 9007199254740991ULL; return (double)r / 9007199254740992.0; };
  auto normal = [&]() { double u1 = mol_uniform(); if (u1 < 1e-300) u1 = 1e-300; double u2 = mol_uniform(); return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2); };
  auto lambda2 = [](double t, double m) { double hb = 1.054571817e-34, kb = 1.380649e-23, amu = 1.66053906660e-27; return hb * hb / (m * amu * kb * t) * 1.0e20; };
  auto fib = [](int i, int n) { double z = 1.0 - (2.0 * i + 1.0) / n, r = std::sqrt(1.0 - z * z), ph = i * 3.14159265358979323846 * (3.0 - std::sqrt(5.0)); return std::array<double, 3>{r * std::cos(ph), r * std::sin(ph), z}; };
  auto envelope = [&](int cage, double rmax, int ng, double eps, double sig, double lam) { double wmin = 1e300;
    for (int i = 0; i <= ng; ++i) for (int j = 0; j <= ng; ++j) for (int k = 0; k <= ng; ++k) {
      double dx = rmax * (2.0 * i / ng - 1.0), dy = rmax * (2.0 * j / ng - 1.0), dz = rmax * (2.0 * k / ng - 1.0);
      if (dx * dx + dy * dy + dz * dz <= rmax * rmax && cage_of(cenv[cage][0] + dx, cenv[cage][1] + dy, cenv[cage][2] + dz) == cage) wmin = std::min(wmin, field(cenv[cage][0] + dx, cenv[cage][1] + dy, cenv[cage][2] + dz, eps, sig, lam)); }
    return wmin - 50.0; };
  auto draw = [&](int cage, double rmax, double env, double eps, double sig, double lam) {
    for (long tr = 0; tr < 100000000; ++tr) { double dx = rmax * (2.0 * mol_uniform() - 1.0), dy = rmax * (2.0 * mol_uniform() - 1.0), dz = rmax * (2.0 * mol_uniform() - 1.0);
      if (dx * dx + dy * dy + dz * dz <= rmax * rmax) { double x = cenv[cage][0] + dx, y = cenv[cage][1] + dy, z = cenv[cage][2] + dz;
        if (cage_of(x, y, z) == cage) { double w = field(x, y, z, eps, sig, lam); if (mol_uniform() < std::exp(-(w - env) / T)) return std::array<double, 4>{x, y, z, w}; } } }
    return std::array<double, 4>{0, 0, 0, 1e300}; };
  auto bridge = [&](std::vector<double>& path, int p, double x0, double y0, double z0, double lam2) {
    path[0] = x0; path[1] = y0; path[2] = z0; double x = x0, y = y0, z = z0;
    for (int k = 1; k < p; ++k) { double left = p - k + 1, sd = std::sqrt(lam2 / p * (left - 1.0) / left);
      x = x + (x0 - x) / left + sd * normal(); y = y + (y0 - y) / left + sd * normal(); z = z + (z0 - z) / left + sd * normal();
      path[3 * k] = x; path[3 * k + 1] = y; path[3 * k + 2] = z; } };
  auto path_w = [&](const std::vector<double>& path, int p, double eps, double sig) { double s = 0; for (int k = 0; k < p; ++k) s += field(path[3 * k], path[3 * k + 1], path[3 * k + 2], eps, sig, 0.0); return s / p; };
  auto stats = [](const std::vector<double>& v) { double m = 0; for (double x : v) m += x; m /= v.size(); double s = 0; for (double x : v) s += (x - m) * (x - m); return std::make_pair(m, std::sqrt(s / (v.size() - 1) / v.size())); };
  auto qratio = [&](int cage, double rmax, double env, double eps, double sig, double lam2, int p, int n0, int m, int nb) {
    std::vector<double> bat(nb), path(768); int per = n0 / nb;
    for (int b = 0; b < nb; ++b) { double acc = 0; for (int i = 0; i < per; ++i) { auto d = draw(cage, rmax, env, eps, sig, 0.0); double r = 0;
        for (int j = 0; j < m; ++j) { bridge(path, p, d[0], d[1], d[2], lam2); r += std::exp(-(path_w(path, p, eps, sig) - d[3]) / T); } acc += r / m; }
      bat[b] = acc / per; }
    return stats(bat); };
  auto pair_r2 = [](double r2, double eh, double sh, double rc2) { if (r2 > rc2) return 0.0; double s2 = sh * sh, s6 = s2 * s2 * s2, s12 = s6 * s6, i2 = 1 / r2, i6 = i2 * i2 * i2; return 4 * eh * (s12 * i6 * i6 - s6 * i6); };
  auto pair_u = [&](const std::vector<double>& pa, const std::vector<double>& pb, int p, double eh, double sh, double rc2) { double s = 0;
    for (int k = 0; k < p; ++k) { double dx = mimg(pa[3 * k] - pb[3 * k]), dy = mimg(pa[3 * k + 1] - pb[3 * k + 1]), dz = mimg(pa[3 * k + 2] - pb[3 * k + 2]); s += pair_r2(dx * dx + dy * dy + dz * dz, eh, sh, rc2); }
    return s / p; };
  auto pratios = [&](int cage, double rmax, double env, double eps, double sig, double lam2, int p, double eh, double sh, double rc, int ns, int nb) {
    std::vector<double> b2(nb), b3(nb), bc(nb), pa(768), pb(768), pc(768); double rc2 = rc * rc; int per = ns / nb;
    for (int b = 0; b < nb; ++b) { double n2 = 0, n3 = 0, nw = 0, cl = 0;
      for (int i = 0; i < per; ++i) {
        auto da = draw(cage, rmax, env, eps, sig, 0.0); bridge(pa, p, da[0], da[1], da[2], lam2); double wa = std::exp(-(path_w(pa, p, eps, sig) - da[3]) / T);
        auto db = draw(cage, rmax, env, eps, sig, 0.0); bridge(pb, p, db[0], db[1], db[2], lam2); double wb = std::exp(-(path_w(pb, p, eps, sig) - db[3]) / T);
        auto dc = draw(cage, rmax, env, eps, sig, 0.0); bridge(pc, p, dc[0], dc[1], dc[2], lam2); double wc = std::exp(-(path_w(pc, p, eps, sig) - dc[3]) / T);
        double uab = pair_u(pa, pb, p, eh, sh, rc2), uac = pair_u(pa, pc, p, eh, sh, rc2), ubc = pair_u(pb, pc, p, eh, sh, rc2);
        n2 += wa * wb * std::exp(-uab / T); n3 += wa * wb * wc * std::exp(-(uab + uac + ubc) / T); nw += wa + wb + wc;
        double ex = mimg(da[0] - db[0]), ey = mimg(da[1] - db[1]), ez = mimg(da[2] - db[2]); cl += std::exp(-pair_r2(ex * ex + ey * ey + ez * ez, eh, sh, rc2) / T); }
      double fp = per, mw = nw / (3.0 * fp); b2[b] = n2 / fp / (mw * mw); b3[b] = n3 / fp / (mw * mw * mw); bc[b] = cl / fp; }
    auto r2 = stats(b2), r3 = stats(b3), r0 = stats(bc); return std::array<double, 6>{r2.first, r2.second, r3.first, r3.second, r0.first, r0.second}; };

  // ---- Ewald of molecular::ewald (double) ----
  std::vector<double> sx, sy, sz, sq;
  for (size_t j = 0; j < site.size(); ++j) { sx.push_back((double)site[j][0]); sy.push_back((double)site[j][1]); sz.push_back((double)site[j][2]); sq.push_back((double)q[j]); }
  const double ealpha = 0.35, erc = 16.0, ekc = 3.92, tp = 2.0 * 3.14159265358979323846 / a;
  std::vector<double> KX, KY, KZ, KF, SC, SS; int mm = 0; while (mm * tp <= ekc) ++mm;
  double pref = 8.0 * 3.14159265358979323846 / (a * a * a);
  for (int ia = 0; ia <= mm; ++ia) for (int ib = -mm; ib <= mm; ++ib) for (int ic = -mm; ic <= mm; ++ic) {
    bool half = ia > 0 || (ia == 0 && ib > 0) || (ia == 0 && ib == 0 && ic > 0); if (!half) continue;
    double kx = tp * ia, ky = tp * ib, kz = tp * ic, k2 = kx * kx + ky * ky + kz * kz; if (k2 > ekc * ekc) continue;
    KX.push_back(kx); KY.push_back(ky); KZ.push_back(kz); KF.push_back(pref * std::exp(-k2 / (4.0 * ealpha * ealpha)) / k2);
    double scs = 0, sss = 0; for (size_t j = 0; j < sx.size(); ++j) { double ph = kx * sx[j] + ky * sy[j] + kz * sz[j]; scs += sq[j] * std::cos(ph); sss += sq[j] * std::sin(ph); }
    SC.push_back(scs); SS.push_back(sss); }
  auto ewrap = [&](double x) { double y = x; while (y < 0.0) y += a; while (y >= a) y -= a; return y; };
  auto phi = [&](double x, double y, double z) { double px = ewrap(x), py = ewrap(y), pz = ewrap(z), rc2 = erc * erc, re = 0;
    for (size_t j = 0; j < sx.size(); ++j) { double dx0 = px - ewrap(sx[j]), dy0 = py - ewrap(sy[j]), dz0 = pz - ewrap(sz[j]);
      for (int ia = -1; ia <= 1; ++ia) for (int ib = -1; ib <= 1; ++ib) for (int ic = -1; ic <= 1; ++ic) { double dx = dx0 - ia * a, dy = dy0 - ib * a, dz = dz0 - ic * a, r2 = dx * dx + dy * dy + dz * dz;
          if (r2 < rc2) { double r = std::sqrt(r2); re += sq[j] * std::erfc(ealpha * r) / r; } } }
    double rp = 0; for (size_t k = 0; k < KX.size(); ++k) { double ph = KX[k] * px + KY[k] * py + KZ[k] * pz; rp += KF[k] * (SC[k] * std::cos(ph) + SS[k] * std::sin(ph)); }
    return re + rp; };
  auto glx = [](int n, std::vector<double>& x, std::vector<double>& w) { x.assign(64, 0); w.assign(64, 0);
    for (int i = 0; i < n; ++i) { double z = std::cos(3.14159265358979323846 * (i + 0.75) / (n + 0.5)), pp = 1; int it = 0, done = 0;
      while (it < 100 && !done) { double p1 = 1, p2 = 0; for (int j = 0; j < n; ++j) { double p3 = p2; p2 = p1; p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0); }
        pp = n * (z * p1 - p2) / (z * z - 1.0); double z1 = z; z = z1 - p1 / pp; if (std::fabs(z - z1) < 1e-15) done = 1; ++it; }
      x[i] = z; w[i] = 2.0 / ((1.0 - z * z) * pp * pp); } };
  auto charged = [&](int cage, double rmax, int nr, int ndir, int nori, double eps, double sig, double lam, double dh, double qh) {
    std::vector<double> gx, gw; glx(nr, gx, gw); double kt = 0.00831446261815324 * T, ke = 1389.35457644382, hh = 0.5 * dh, c0 = 0, cq = 0;
    for (int i = 0; i < nr; ++i) { double r = 0.5 * rmax * (gx[i] + 1.0), wr = 0.5 * rmax * gw[i] * r * r * 4.0 * 3.14159265358979323846 / ndir;
      for (int d = 0; d < ndir; ++d) { auto u = fib(d, ndir); double x = cenv[cage][0] + r * u[0], y = cenv[cage][1] + r * u[1], z = cenv[cage][2] + r * u[2];
        if (cage_of(x, y, z) != cage) continue;
        double wl = std::exp(-field(x, y, z, eps, sig, lam) / T); c0 += wr * wl; double p0 = phi(x, y, z), s = 0;
        for (int o = 0; o < nori; ++o) { auto n = fib(o, nori); double e = ke * qh * (phi(x + hh * n[0], y + hh * n[1], z + hh * n[2]) + phi(x - hh * n[0], y - hh * n[1], z - hh * n[2]) - 2.0 * p0); s += std::exp(-e / kt); }
        cq += wr * wl * s / nori; } }
    return std::log(cq / c0); };

  // ---- the test's sequence ----
  int small = cage_of(0, 0, 0), large = cage_of(0.375 * a, 0.375 * a, 0.375 * a);
  std::printf("EXPECT cages %d %d %d\n", NC, small, large);
  const double eB = std::sqrt(78.20 * 34.2), sB = 0.5 * (3.166 + 2.96), l2 = lambda2(T, 2.01588);
  std::printf("EXPECT lambda2 %.17e\n", l2);
  mol_seed(20261117);
  double n1 = normal(), n2 = normal(), n3 = normal();
  std::printf("EXPECT normals %.17e %.17e %.17e\n", n1, n2, n3);
  { std::vector<double> path(768); bridge(path, 8, 0.1, 0.2, 0.3, l2); double s = 0; for (int k = 0; k < 24; ++k) s += path[k] * (k + 1);
    std::printf("EXPECT bridge %.17e %.17e %.17e %.17e\n", path[21], path[22], path[23], s); }
  double envS = envelope(small, 3.3, 20, eB, sB, 0.0), envL = envelope(large, 4.6, 20, eB, sB, 0.0);
  std::printf("EXPECT env %.17e %.17e\n", envS, envL);
  auto qs = qratio(small, 3.3, envS, eB, sB, l2, 16, 400, 2, 10);
  auto ql = qratio(large, 4.6, envL, eB, sB, l2, 16, 400, 2, 10);
  std::printf("EXPECT qratio %.17e %.17e %.17e %.17e\n", qs.first, qs.second, ql.first, ql.second);
  std::printf("  ln(C_cl/C_q) small %.4f +- %.4f  large %.4f +- %.4f\n", -std::log(qs.first), qs.second / qs.first, -std::log(ql.first), ql.second / ql.first);
  auto pr = pratios(large, 4.6, envL, eB, sB, l2, 16, 34.2, 2.96, 0.5 * a, 300, 10);
  std::printf("EXPECT pairs %.17e %.17e %.17e %.17e %.17e %.17e\n", pr[0], pr[1], pr[2], pr[3], pr[4], pr[5]);
  std::printf("  ln(R2cl/R2q) %.4f  (R2q %.4f +- %.4f, R2cl %.4f +- %.4f)\n", std::log(pr[4] / pr[0]), pr[0], pr[1], pr[4], pr[5]);
  const double eA = std::sqrt(78.20 * (0.2852 / 0.00831446261815324)), sA = 0.5 * (3.166 + 3.038), lamA = l2 / 24.0;
  double rcs = charged(small, 3.3, 6, 26, 14, eA, sA, lamA, 0.7414, 0.4932), rl = charged(large, 4.1, 6, 26, 14, eA, sA, lamA, 0.7414, 0.4932);
  std::printf("EXPECT charged %.17e %.17e\n", rcs, rl);
  return 0;
}
