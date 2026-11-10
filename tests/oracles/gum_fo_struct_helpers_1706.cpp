// C++23 oracle for tests/run-pass/gum_fo_struct_helpers_1706.sio (#1706).
// Build: g++ -std=c++23 -O2 gum_fo_struct_helpers_1706.cpp && ./a.out
// First-order GUM, var(y) = sum_k (dy/dx_k)^2 sigma_k^2 over the independent
// inputs x, v, s, with the partial derivatives written out for each row.
#include <cstdio>

int main() {
    const long double x = 5.0L, v = 50.0L, s = 2.0L;
    const long double vx = 0.3L * 0.3L, vv = 2.0L * 2.0L, vs = 0.1L * 0.1L;
    std::printf("param_value=%.12Lf\n", 4.0L * vx);                 // 2 * p.cl0 = 2x
    std::printf("param_ref=%.12Lf\n", 4.0L * vx);                   // same through &p
    std::printf("param_two=%.12Lf\n", vx + vv);                     // p.cl0 + q.cl0 = x + v
    std::printf("method_first=%.12Lf\n", vx);                       // p.cl0 = x
    std::printf("method_prod=%.12Lf\n", v * v * vx + x * x * vv);   // x * v
    std::printf("method_cancel=%.12Lf\n", 0.0L);                    // p.first() - x
    std::printf("method_arg=%.12Lf\n", s * s * vx + x * x * vs);    // x * s
    std::printf("ctor_local_method=%.12Lf\n", v * v * vx + x * x * vv); // mk(x, v).prod()
}
