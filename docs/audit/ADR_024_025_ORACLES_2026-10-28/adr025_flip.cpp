// Cockcroft-Gault decision "CrCl < 30 mL/min": first-order u, one-sided Cantelli flip bound,
// Euler allocation of the variance across inputs, and the bound after remeasuring SCr.
#include <cstdio>
struct In { const char* name; double mu, u; };
static void eval(double age, double wt, double scr, double u_wt, double u_scr, double t) {
    const double s = (140.0 - age) * wt / (72.0 * scr);
    const double c_wt = (140.0 - age) / (72.0 * scr);
    const double c_scr = -(140.0 - age) * wt / (72.0 * scr * scr);
    const double v_wt = c_wt * c_wt * u_wt * u_wt, v_scr = c_scr * c_scr * u_scr * u_scr;  // independent inputs
    const double v = v_wt + v_scr, d = s - t;
    const double eps = v / (v + d * d);
    std::printf("CrCl=%.3f u=%.3f d=%.3f eps=%.4f w_weight=%.3f w_scr=%.3f\n", s, __builtin_sqrt(v), d, eps, v_wt / v, v_scr / v);
}
int main() {
    eval(70, 70, 2.2, 1.0, 0.2, 30.0);   // as measured
    eval(70, 70, 1.8, 1.0, 0.2, 30.0);
    eval(70, 70, 1.8, 1.0, 0.05, 30.0);
    eval(70, 70, 2.2, 1.0, 0.05, 30.0);  // SCr remeasured with smaller uncertainty
}
