// C++23 oracle for tests/stdlib/ode/test_pbpk14_rk4_oracle.sio.
// Same 14-compartment model as stdlib/ode/pbpk14_rk4.sio (default_params,
// 100 mg oral dose). Shows that plain RK4 at the caller's step diverges and
// that the stability-bounded substepping the solver now uses agrees with a
// converged fine-step reference.
// Build: g++ -std=c++23 -O2 pbpk14_rk4_stability.cpp && ./a.out
#include <array>
#include <cmath>
#include <cstdio>
// 0 gut 1 art 2 ven 3 lung 4 heart 5 brain 6 muscle 7 adipose 8 skin 9 bone
// 10 spleen 11 gut_tis 12 liver 13 kidney
using State = std::array<double, 14>;
static State ode(const State& s) {
    const double ka = 1, rb = 1, fu = 0.3, cl = 50, gfr = 7.5;
    const double v_art = 1, v_ven = 3.9, v_lung = .5, v_heart = .33, v_brain = 1.4, v_muscle = 28,
                 v_adipose = 14, v_skin = 2.6, v_bone = 4, v_spleen = .15, v_gut = 1, v_liver = 1.5,
                 v_kidney = .28;
    const double q_lung = 175, q_heart = 7, q_brain = 21, q_muscle = 31.5, q_adipose = 8.75,
                 q_skin = 8.75, q_bone = 8.75, q_spleen = 5.25, q_gut = 31.5, q_la = 10.5,
                 q_kidney = 38.5;
    const double kp_lung = .8, kp_heart = 1, kp_brain = .5, kp_muscle = .8, kp_adipose = 2,
                 kp_skin = 1, kp_bone = .5, kp_spleen = 1, kp_gut = 1, kp_liver = 1.2, kp_kidney = 1.5;
    const double ca = s[1] / v_art / rb, cv = s[2] / v_ven / rb, clu = s[3] / v_lung,
                 ch = s[4] / v_heart, cb = s[5] / v_brain, cm = s[6] / v_muscle,
                 cad = s[7] / v_adipose, cs = s[8] / v_skin, cbo = s[9] / v_bone,
                 csp = s[10] / v_spleen, cg = s[11] / v_gut, cli = s[12] / v_liver,
                 ck = s[13] / v_kidney;
    const double qh = q_gut + q_spleen + q_la;
    State d;
    d[0] = -ka * s[0];
    d[1] = q_lung * clu / kp_lung - (q_heart + q_brain + q_muscle + q_adipose + q_skin + q_bone + q_gut + q_spleen + q_la + q_kidney) * ca;
    d[2] = q_heart * ch / kp_heart + q_brain * cb / kp_brain + q_muscle * cm / kp_muscle + q_adipose * cad / kp_adipose + q_skin * cs / kp_skin + q_bone * cbo / kp_bone + qh * cli / kp_liver + q_kidney * ck / kp_kidney - q_lung * cv;
    d[3] = q_lung * (cv - clu / kp_lung);
    d[4] = q_heart * (ca - ch / kp_heart);
    d[5] = q_brain * (ca - cb / kp_brain);
    d[6] = q_muscle * (ca - cm / kp_muscle);
    d[7] = q_adipose * (ca - cad / kp_adipose);
    d[8] = q_skin * (ca - cs / kp_skin);
    d[9] = q_bone * (ca - cbo / kp_bone);
    d[10] = q_spleen * (ca - csp / kp_spleen);
    d[11] = q_gut * (ca - cg / kp_gut) + ka * s[0];
    d[12] = q_la * ca + q_gut * cg / kp_gut + q_spleen * csp / kp_spleen - qh * cli / kp_liver - cl * fu * cli;
    d[13] = q_kidney * (ca - ck / kp_kidney) - gfr * fu * ca * rb;
    return d;
}
static State axpy(const State& a, double h, const State& b) {
    State r;
    for (int i = 0; i < 14; ++i) r[i] = a[i] + h * b[i];
    return r;
}
static void rk4(State& s, double h) {
    const State k1 = ode(s), k2 = ode(axpy(s, .5 * h, k1)), k3 = ode(axpy(s, .5 * h, k2)), k4 = ode(axpy(s, h, k3));
    for (int j = 0; j < 14; ++j) s[j] += h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
}
static double norm1() {  // ||A||_1: A is linear, so ode(e_j) is column j
    double mx = 0;
    for (int j = 0; j < 14; ++j) {
        State e{};
        e[j] = 1;
        double s = 0;
        for (double x : ode(e)) s += std::fabs(x);
        mx = std::max(mx, s);
    }
    return mx;
}
static State solve(double t, long n, bool substep, long* taken) {
    const double dt = t / n;
    long m = 1;
    if (substep) {
        const double x = dt * norm1() / 2.5;
        if (x > 1) m = static_cast<long>(std::ceil(x));
    }
    State s{};
    s[0] = 100;
    for (long i = 0; i < n; ++i)
        for (long k = 0; k < m; ++k) rk4(s, dt / m);
    if (taken) *taken = n * m;
    return s;
}
static double total6(const State& s) { return s[0] + s[1] + s[2] + s[3] + s[12] + s[13]; }
int main() {
    std::printf("norm1 = %.17g\n", norm1());
    long taken = 0;
    const State plain = solve(24, 100, false, nullptr);
    std::printf("plain RK4, 24 h / 100 steps: total6 = %.17g (diverged)\n", total6(plain));
    const State sub = solve(24, 100, true, &taken);
    const State ref = solve(24, 1000000, false, nullptr);
    std::printf("substepped 24 h / 100: total6 = %.17g liver = %.17g substeps = %ld\n", total6(sub), sub[12], taken);
    std::printf("reference  24 h / 1e6: total6 = %.17g liver = %.17g\n", total6(ref), ref[12]);
    const State sub2 = solve(2, 50, true, &taken);
    const State ref2 = solve(2, 1000000, false, nullptr);
    std::printf("substepped 2 h / 50: liver = %.17g substeps = %ld\n", sub2[12], taken);
    std::printf("reference  2 h / 1e6: liver = %.17g\n", ref2[12]);
}
