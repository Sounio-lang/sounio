// Oracle for tests/run-pass/epistemic_meet_second_order.sio.
// a = c0^2 with c0 = 2 +- 0.1 (channel 0), b = 4.1 + c1 with u(c1) = 0.3.
// First-order V: var(a) = (2 c0 u)^2, var(b) = 0.09, independent.
// meet weights w_a = var(b) / (var(a) + var(b)); the fusion is w_a a + w_b b,
// so its Hessian in c0 is 2 w_a and its second-order mean adds 0.5 H u^2.
// Build: g++ -std=c++23 -O2 meet_second_order.cpp && ./a.out
#include <cstdio>
int main() {
  long double c0 = 2, u0 = 0.1L, b = 4.1L, vb = 0.09L;
  long double a = c0 * c0, va = (2 * c0 * u0) * (2 * c0 * u0);
  long double wa = vb / (va + vb), m = wa * a + (1 - wa) * b, h = 2 * wa;
  std::printf("meet %.17Lg var %.17Lg hessian00 %.17Lg mean2 %.17Lg\n", m, va * vb / (va + vb), h, m + 0.5L * h * u0 * u0);
}
