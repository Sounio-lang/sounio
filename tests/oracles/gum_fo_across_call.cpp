// C++23 oracle for tests/run-pass/gum_fo_across_call.sio (#1706, KL-11).
// Build: g++ -std=c++23 -O2 gum_fo_across_call.cpp && ./a.out
#include <cstdio>
#include <cmath>
// c_{n+1} = c_n + 0.1 * (-(cl*fu) * c_n / 5), 20 steps, c_0 = 1.2.
// First order in (cl, fu): forward-mode duals on each input.
int main() {
    long double cl = 12.4L, fu = 0.08L, ucl = 1.24L, ufu = 0.008L;
    long double c = 1.2L, dc_dcl = 0.0L, dc_dfu = 0.0L;
    for (int s = 0; s < 20; ++s) {
        long double d = -(cl * fu) * c / 5.0L;
        long double dd_dcl = -(fu * c + cl * fu * dc_dcl) / 5.0L;
        long double dd_dfu = -(cl * c + cl * fu * dc_dfu) / 5.0L;
        c += 0.1L * d; dc_dcl += 0.1L * dd_dcl; dc_dfu += 0.1L * dd_dfu;
    }
    std::printf("var=%.15Lg\n", dc_dcl * dc_dcl * ucl * ucl + dc_dfu * dc_dfu * ufu * ufu);
}
