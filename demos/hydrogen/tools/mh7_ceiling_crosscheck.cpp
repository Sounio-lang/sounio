// demos/hydrogen/tools/mh7_ceiling_crosscheck.cpp
// Independent C++23 cross-check of mh7_coupled_ceiling.sio, written from the
// protocol (Gkanas et al. 2020 Tables 3-4, flat-plateau van 't Hoff, P0 = 1 bar,
// R = 8.314462618), not translated from the .sio.
// Build: g++ -std=c++23 -O2 -o mh7x tools/mh7_ceiling_crosscheck.cpp && ./mh7x
#include <array>
#include <cmath>
#include <cstdio>

namespace {
constexpr double R = 8.314462618;
struct Stage { double h_abs, s_abs, h_des, s_des; };
constexpr std::array<Stage, 7> table3{{
    {25242, 104.6, 28195, 106.8}, {21466, 94.7, 26133, 107.1}, {20354, 101.1, 24823, 108.7},
    {19991, 100.2, 20252, 100.8}, {18198, 98.12, 19856, 101.4}, {16232, 98.05, 19125, 101.5},
    {14702, 98.1, 18916, 106.2}}};
struct Case { double t_hot, p_final, t_cycle; };
constexpr std::array<Case, 6> table4{{
    {353.15, 374.0, 6625}, {363.15, 483.2, 5760}, {373.15, 606.4, 5090},
    {378.15, 682.0, 4720}, {383.15, 742.0, 4530}, {393.15, 830.0, 3760}}};
constexpr double t_cold = 283.15;

double plateau(double h, double s, double t) { return std::exp(s / R - h / (R * t)); }
double drive(std::size_t i, double th) {
  return std::log(plateau(table3[i].h_des, table3[i].s_des, th) /
                  plateau(table3[i + 1].h_abs, table3[i + 1].s_abs, t_cold));
}
}  // namespace

int main() {
  int pairs = 0, conc_p = 0, conc_t = 0, infeasible = 0, bottleneck_s6s7 = 0;
  for (std::size_t c = 0; c < table4.size(); ++c) {
    const auto th = table4[c].t_hot;
    const double ceil = plateau(table3[6].h_des, table3[6].s_des, th);
    std::size_t kmin = 0;
    for (std::size_t k = 0; k < 6; ++k) {
      if (drive(k, th) <= 0) ++infeasible;
      if (drive(k, th) < drive(kmin, th)) kmin = k;
    }
    bottleneck_s6s7 += (kmin == 5);
    std::printf("case %zu ceiling %.4f bar  paper/ceiling %.4f  min drive %.4f (S%zu>S%zu)\n", c + 1,
                 ceil, table4[c].p_final / ceil, drive(kmin, th), kmin + 1, kmin + 2);
    for (std::size_t d = c + 1; d < table4.size(); ++d) {
      ++pairs;
      const double dc = plateau(table3[6].h_des, table3[6].s_des, table4[d].t_hot) - ceil;
      conc_p += (dc * (table4[d].p_final - table4[c].p_final) > 0);
      conc_t += ((drive(5, table4[d].t_hot) - drive(5, th)) * (table4[d].t_cycle - table4[c].t_cycle) < 0);
    }
  }
  std::printf("infeasible couplings %d  bottleneck S6>S7 in %d/6  pressure concordant %d/%d  cycle concordant %d/%d\n",
               infeasible, bottleneck_s6s7, conc_p, pairs, conc_t, pairs);
}
