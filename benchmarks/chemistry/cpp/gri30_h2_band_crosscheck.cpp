// Independent C++20 cross-check of the Sounio GRI-Mech 3.0 H/O kinetics and of
// the first-order diagonal GUM uncertainty band, written from the published
// protocol rather than translated from either the Sounio module or the Python
// replica.  Third implementation: Sounio (native), Python (replica), C++ (this).
//
// Purpose: settle the band scaling law empirically.  The per-step parameter
// term (nu_ir * net_r * dt * u_r)^2 accumulated in quadrature over N = T/dt
// steps gives a variance ~ N * dt^2 = T*dt, hence a standard uncertainty
// ~ sqrt(T*dt).  A factor f in dt must therefore move the band by sqrt(f),
// and a factor f in T must move it by sqrt(f) as well.
//
// Build:  g++ -std=c++20 -O2 -o band_crosscheck gri30_h2_band_crosscheck.cpp
// Run:    ./band_crosscheck ../gri30_h2_mechanism.json
//
// No external dependencies: the JSON reader below handles exactly the shape of
// benchmarks/chemistry/gri30_h2_mechanism.json and nothing more.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------- tiny JSON
struct JV {
    enum K { NUL, NUM, STR, ARR, OBJ } k = NUL;
    double num = 0.0;
    std::string str;
    std::vector<JV> arr;
    std::vector<std::pair<std::string, JV>> obj;
    const JV* find(const std::string& key) const {
        for (auto& kv : obj) if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

struct JParser {
    const std::string& s; size_t i = 0;
    explicit JParser(const std::string& src) : s(src) {}
    void ws() { while (i < s.size() && (s[i]==' '||s[i]=='\n'||s[i]=='\t'||s[i]=='\r')) ++i; }
    JV parse() {
        ws(); JV v;
        if (s[i] == '{') {
            v.k = JV::OBJ; ++i; ws();
            if (s[i] == '}') { ++i; return v; }
            for (;;) { ws(); std::string key = pstr(); ws(); ++i; // ':'
                v.obj.emplace_back(key, parse()); ws();
                if (s[i] == ',') { ++i; continue; } ++i; break; }
        } else if (s[i] == '[') {
            v.k = JV::ARR; ++i; ws();
            if (s[i] == ']') { ++i; return v; }
            for (;;) { v.arr.push_back(parse()); ws();
                if (s[i] == ',') { ++i; continue; } ++i; break; }
        } else if (s[i] == '"') { v.k = JV::STR; v.str = pstr(); }
        else if (s.compare(i, 4, "null") == 0) { v.k = JV::NUL; i += 4; }
        else if (s.compare(i, 4, "true") == 0) { v.k = JV::NUM; v.num = 1; i += 4; }
        else if (s.compare(i, 5, "false") == 0) { v.k = JV::NUM; v.num = 0; i += 5; }
        else { v.k = JV::NUM; size_t e; v.num = std::stod(s.substr(i), &e); i += e; }
        return v;
    }
    std::string pstr() {
        ++i; std::string out;
        while (s[i] != '"') { if (s[i] == '\\') ++i; out += s[i++]; }
        ++i; return out;
    }
};

// ------------------------------------------------------------- mechanism
static const std::array<const char*, 10> SPN =
    {"H2","H","O","O2","OH","H2O","HO2","H2O2","N2","AR"};
constexpr int NSP = 10;
constexpr double R_SI = 8.314462618;
constexpr double R_CAL = 1.9872041;      // cal/mol/K, CHEMKIN activation-energy R
constexpr double P0 = 101325.0;          // GRI-Mech NASA-7 reference pressure, 1 atm

struct Rxn {
    std::string eq; int type = 0;        // 0 arrhenius, 1 three-body, 2 falloff
    double A = 0, b = 0, Ea = 0;
    bool has_low = false; double lA = 0, lb = 0, lEa = 0;
    bool has_troe = false; double t_a = 0, t3 = 0, t1 = 0, t2 = 0;
    std::array<double, NSP> eff{}, reac{}, prod{}, nu{};
    double dn = 0; double u = 0.30;
};

static int sp_index(const std::string& n) {
    for (int i = 0; i < NSP; ++i) if (n == SPN[i]) return i;
    return -1;
}

struct Mech {
    std::vector<Rxn> rx;
    std::array<std::array<std::array<double,7>,2>, NSP> nasa{};
};

static Mech load_mech(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    std::stringstream ss; ss << f.rdbuf(); std::string src = ss.str();
    JParser p(src); JV root = p.parse();
    Mech m;
    const JV* spec = root.find("species");
    for (int i = 0; i < NSP; ++i) {
        const JV* s = spec->find(SPN[i]);
        const JV* co = s->find("coeffs");
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 7; ++c) m.nasa[i][r][c] = co->arr[r].arr[c].num;
    }
    // representative 1-sigma relative uncertainties on k (same set as the replica:
    // Baulch 2005 / Konnov 2008 / Hong 2011 order-of-magnitude fidelity, not a refit)
    const std::map<std::string, double> named = {
        {"H + O2 <=> O + OH", 0.10}, {"OH + H2 <=> H + H2O", 0.10},
        {"O + H2 <=> H + OH", 0.15}, {"2 OH <=> O + H2O", 0.20},
        {"H + O2 + M <=> HO2 + M", 0.25}, {"H + OH + M <=> H2O + M", 0.30},
        {"2 O + M <=> O2 + M", 0.25}, {"O + H + M <=> OH + M", 0.25}};
    for (auto& rv : root.find("reactions")->arr) {
        Rxn r;
        r.eq = rv.find("eq")->str;
        const std::string ty = rv.find("type")->str;
        r.type = (ty == "three-body") ? 1 : (ty == "falloff") ? 2 : 0;
        const JV* fw = rv.find("fwd");
        r.A = fw->arr[0].num; r.b = fw->arr[1].num; r.Ea = fw->arr[2].num;
        const JV* lo = rv.find("low");
        if (lo && lo->k == JV::ARR) { r.has_low = true;
            r.lA = lo->arr[0].num; r.lb = lo->arr[1].num; r.lEa = lo->arr[2].num; }
        const JV* tr = rv.find("troe");
        if (tr && tr->k == JV::ARR) { r.has_troe = true;
            r.t_a = tr->arr[0].num; r.t3 = tr->arr[1].num;
            r.t1 = tr->arr[2].num; r.t2 = tr->arr[3].num; }
        r.eff.fill(1.0);
        for (auto& kv : rv.find("eff")->obj) { int j = sp_index(kv.first);
            if (j >= 0) r.eff[j] = kv.second.num; }
        for (auto& kv : rv.find("react")->obj) { int j = sp_index(kv.first);
            if (j >= 0) { r.reac[j] = kv.second.num; r.nu[j] -= kv.second.num; } }
        for (auto& kv : rv.find("prod")->obj) { int j = sp_index(kv.first);
            if (j >= 0) { r.prod[j] = kv.second.num; r.nu[j] += kv.second.num; } }
        for (int j = 0; j < NSP; ++j) r.dn += r.nu[j];
        auto it = named.find(r.eq); if (it != named.end()) r.u = it->second;
        m.rx.push_back(r);
    }
    return m;
}

// ------------------------------------------------------------ thermochemistry
static double g_rt(const Mech& m, int s, double T) {
    const auto& a = (T <= 1000.0) ? m.nasa[s][0] : m.nasa[s][1];
    const double h = a[0] + a[1]*T/2 + a[2]*T*T/3 + a[3]*T*T*T/4 + a[4]*T*T*T*T/5 + a[5]/T;
    const double sr = a[0]*std::log(T) + a[1]*T + a[2]*T*T/2 + a[3]*T*T*T/3
                    + a[4]*T*T*T*T/4 + a[6];
    return h - sr;
}

static std::vector<double> kc_all(const Mech& m, double T) {
    std::vector<double> kc(m.rx.size());
    const double c0 = P0 / (R_SI * T) * 1e-6;   // mol/cm^3
    for (size_t r = 0; r < m.rx.size(); ++r) {
        double e = 0; for (int s = 0; s < NSP; ++s) e -= m.rx[r].nu[s] * g_rt(m, s, T);
        kc[r] = std::exp(e) * std::pow(c0, m.rx[r].dn);
    }
    return kc;
}

static double kfwd(const Mech& m, size_t r, double T, double meff) {
    const Rxn& x = m.rx[r];
    const double kf = x.A * std::pow(T, x.b) * std::exp(-x.Ea / (R_CAL * T));
    if (x.type == 1) return kf * meff;
    if (x.type == 2) {
        const double k0 = x.lA * std::pow(T, x.lb) * std::exp(-x.lEa / (R_CAL * T));
        const double pr = k0 * meff / kf;
        const double fc = (1 - x.t_a) * std::exp(-T / x.t3) + x.t_a * std::exp(-T / x.t1)
                        + std::exp(-x.t2 / T);
        const double lfc = std::log10(fc);
        const double c = -0.4 - 0.67 * lfc;
        const double lpr = std::log10(pr);
        const double xx = (lpr + c) / (0.75 - 1.27 * lfc - 0.14 * (lpr + c));
        return kf * (pr / (1 + pr)) * std::pow(10.0, lfc / (1 + xx * xx));
    }
    return kf;
}

using Vec = std::array<double, NSP>;

static std::vector<double> rates_net(const Mech& m, double T, const Vec& c,
                                     const std::vector<double>& kc) {
    std::vector<double> out(m.rx.size());
    for (size_t r = 0; r < m.rx.size(); ++r) {
        const Rxn& x = m.rx[r];
        double meff = 0; for (int s = 0; s < NSP; ++s) meff += x.eff[s] * c[s];
        const double kf = kfwd(m, r, T, meff);
        double f = kf, b = kf / kc[r];
        for (int s = 0; s < NSP; ++s) {
            if (x.reac[s] > 0) f *= std::pow(c[s], x.reac[s]);
            if (x.prod[s] > 0) b *= std::pow(c[s], x.prod[s]);
        }
        out[r] = f - b;
    }
    return out;
}

static Vec dcdt(const Mech& m, double T, const Vec& c, const std::vector<double>& kc) {
    const auto rn = rates_net(m, T, c, kc);
    Vec d{};
    for (int s = 0; s < NSP; ++s) { double a = 0;
        for (size_t r = 0; r < m.rx.size(); ++r) a += m.rx[r].nu[s] * rn[r];
        d[s] = a; }
    return d;
}

static Vec rk4(const Mech& m, double T, const Vec& c, double dt,
               const std::vector<double>& kc) {
    const Vec k1 = dcdt(m, T, c, kc);
    Vec t2c, t3c, t4c;
    for (int i = 0; i < NSP; ++i) t2c[i] = c[i] + 0.5 * dt * k1[i];
    const Vec k2 = dcdt(m, T, t2c, kc);
    for (int i = 0; i < NSP; ++i) t3c[i] = c[i] + 0.5 * dt * k2[i];
    const Vec k3 = dcdt(m, T, t3c, kc);
    for (int i = 0; i < NSP; ++i) t4c[i] = c[i] + dt * k3[i];
    const Vec k4 = dcdt(m, T, t4c, kc);
    Vec o{};
    for (int i = 0; i < NSP; ++i)
        o[i] = c[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    return o;
}

// First-order diagonal GUM delta method, per step, dt^2 parameter scaling.
static Vec prop_unc(const Mech& m, double T, const Vec& c,
                    const std::vector<double>& kc, const Vec& v, double dt) {
    const double eps = 1e-12;
    const auto rn = rates_net(m, T, c, kc);
    std::array<Vec, NSP> J{};
    for (int k = 0; k < NSP; ++k) {
        Vec cp = c, cm = c; cp[k] += eps; cm[k] -= eps;
        const Vec dp = dcdt(m, T, cp, kc), dm = dcdt(m, T, cm, kc);
        for (int i = 0; i < NSP; ++i) J[k][i] = (dp[i] - dm[i]) / (2 * eps);
    }
    Vec out{};
    for (int i = 0; i < NSP; ++i) {
        double acc = v[i] + 2.0 * J[i][i] * v[i] * dt;
        for (int k = 0; k < NSP; ++k) { if (k == i) continue;
            const double t = J[k][i] * dt; acc += t * t * v[k]; }
        for (size_t r = 0; r < m.rx.size(); ++r) {
            const double t = m.rx[r].nu[i] * rn[r] * dt * m.rx[r].u; acc += t * t; }
        out[i] = std::max(acc, 0.0);
    }
    return out;
}

struct Run { Vec c, u; long n; };

static Run band(const Mech& m, double T, double t_end, double dt, bool with_unc) {
    const auto kc = kc_all(m, T);
    const double mtot = 1.0 / (82.057 * T);
    Vec c{}; c[0] = mtot * 0.02; c[3] = mtot * 0.01; c[8] = mtot * 0.97; c[1] = 1e-11;
    Vec v{}; v[0] = std::pow(0.01 * c[0], 2); v[3] = std::pow(0.01 * c[3], 2);
    const long n = std::lround(t_end / dt);
    for (long i = 0; i < n; ++i) {
        c = rk4(m, T, c, dt, kc);
        if (with_unc) v = prop_unc(m, T, c, kc, v, dt);
    }
    Vec u{}; for (int i = 0; i < NSP; ++i) u[i] = std::sqrt(v[i]);
    return {c, u, n};
}

int main(int argc, char** argv) {
    const std::string path = (argc > 1) ? argv[1] : "../gri30_h2_mechanism.json";
    const Mech m = load_mech(path);
    std::printf("mechanism: %s  (%zu reactions, %d species)\n",
                path.c_str(), m.rx.size(), NSP);
    const double T = 1500.0;

    std::printf("\n=== deterministic checkpoint, T=%.0f K, t=1e-4 s, dt=1e-8 ===\n", T);
    const Run d = band(m, T, 1e-4, 1e-8, false);
    for (int i = 0; i < 8; ++i) std::printf("  %-5s %.17e\n", SPN[i], d.c[i]);

    std::printf("\n=== STEP 5a: band vs dt at fixed T = 1e-6 s ===\n");
    const double TE = 1e-6;
    const double dts[3] = {4e-9, 2e-9, 1e-9};
    Run rr[3];
    for (int j = 0; j < 3; ++j) {
        rr[j] = band(m, T, TE, dts[j], true);
        std::printf("dt=%.0e n=%6ld : ", dts[j], rr[j].n);
        for (int i = 0; i < 8; ++i) std::printf("%s=%.6e ", SPN[i], rr[j].u[i]);
        std::printf("\n");
    }
    const int pairs[3][2] = {{0,1},{1,2},{0,2}};
    for (auto& p : pairs) {
        const double f = dts[p[0]] / dts[p[1]];
        std::printf("\n  dt factor f=%.0f  -> predicted band ratio sqrt(f)=%.6f\n",
                    f, std::sqrt(f));
        for (int i = 0; i < 8; ++i)
            if (rr[p[1]].u[i] > 0)
                std::printf("    %-5s measured ratio %.6f\n",
                            SPN[i], rr[p[0]].u[i] / rr[p[1]].u[i]);
    }

    std::printf("\n=== STEP 5b: band vs T at fixed dt = 1e-8 s ===\n");
    const double tes[3] = {1e-6, 1e-5, 1e-4};
    Run tr_[3];
    for (int j = 0; j < 3; ++j) {
        tr_[j] = band(m, T, tes[j], 1e-8, true);
        std::printf("T=%.0e n=%6ld sqrt(T/dt)=%9.4f : ", tes[j], tr_[j].n,
                    std::sqrt(tes[j] / 1e-8));
        for (int i : {0, 1, 5}) std::printf("%s=%.6e ", SPN[i], tr_[j].u[i]);
        std::printf("\n");
    }
    for (int j = 1; j < 3; ++j) {
        const double f = tes[j] / tes[j-1];
        std::printf("\n  T factor f=%.0f -> predicted band ratio sqrt(f)=%.6f\n",
                    f, std::sqrt(f));
        for (int i = 0; i < 8; ++i)
            if (tr_[j-1].u[i] > 0)
                std::printf("    %-5s measured ratio %.6f\n",
                            SPN[i], tr_[j].u[i] / tr_[j-1].u[i]);
    }
    return 0;
}
