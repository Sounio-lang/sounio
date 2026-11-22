// C++23 oracle for tests/run-pass/gum_fo_struct_fields_channels_2776.sio (#2776).
// Build: g++ -std=c++23 -O2 gum_fo_struct_fields_channels_2776.cpp && ./a.out
// First-order GUM, var(y) = sum_k (dy/dx_k)^2 sigma_k^2 over independent inputs,
// with the partial derivatives written out by hand for each line of the test.
#include <cstdio>

int main() {
    // x = 5 +- 0.3, v = 50 +- 2, s = 2 +- 0.1
    const long double x = 5.0L, sx = 0.3L, v = 50.0L, sv = 2.0L, s = 2.0L, ss = 0.1L;
    const long double vx = sx * sx, vv = sv * sv, vs = ss * ss;
    std::printf("field=%.12Lf\n", vx);                           // p.cl0 = x
    std::printf("nested=%.12Lf\n", vx);                          // o.inner.cl0 = x
    std::printf("sum_fields=%.12Lf\n", vx + vv);                 // p.cl0 + p.v0
    std::printf("call_field=%.12Lf\n", s * s * vx + x * x * vs); // mk(x, s).cl0 = x*s
    std::printf("param_field=%.12Lf\n", 4.0L * vx);              // twice(p) = 2*p.cl0
    std::printf("param_ref=%.12Lf\n", 4.0L * vx);                // twice_ref(&p) = 2*p.cl0
    std::printf("method=%.12Lf\n", v * v * vx + x * x * vv);     // p.prod() = cl0*v0
    std::printf("nested_arg=%.12Lf\n", vv + 4.0L * vx);          // add2(v, dbl(x)) = v + 2x
    std::printf("fn_value=%.12Lf\n", 4.0L * vx);                 // fdbl(x) = 2x
    std::printf("fn_value_hof=%.12Lf\n", 4.0L * vx);             // apply1(dbl, x) = 2x
    std::printf("closure_after_def=%.12Lf\n", 9.0L * vx);        // 3x, declared before a closure
    std::printf("closure_params=%.12Lf\n", s * s * vx + x * x * vs); // cmul(x, s) = x s
    std::printf("closure_cancel=%.12Lf\n", 0.0L);                // cmul(x, s) - x s
    std::printf("closure_capture=%.12Lf\n", 4.0L * vv);          // 2 v, v captured
    std::printf("closure_capture_cancel=%.12Lf\n", 0.0L);        // cap(2) - 2 v
    // 10 inputs: a is the 10th (sigma 0.05, a = 2), b the 11th (sigma 0.2, b = 3)
    const long double a = 2.0L, sa = 0.05L, b = 3.0L, sb = 0.2L;
    std::printf("cube10=%.12Lf\n", 9.0L * a * a * a * a * sa * sa);   // a^3
    // (a+b)(a-b) = a^2 - b^2: d/da = 2a, d/db = -2b
    std::printf("diffsq10=%.12Lf\n", 4.0L * a * a * sa * sa + 4.0L * b * b * sb * sb);
    std::printf("cancel10=%.12Lf\n", 0.0L);                      // (a + b) - a - b
    std::printf("call10_cancel=%.12Lf\n", 0.0L);                 // sq(a) - a*a
    std::printf("field10_cancel=%.12Lf\n", 0.0L);                // pr.a - a
    std::printf("method10=%.12Lf\n", sa * sa + sb * sb);         // pr.d() = a - b
    std::printf("method10_minus_a=%.12Lf\n", sb * sb);           // pr.d() - a = -b
    std::printf("param10_cancel=%.12Lf\n", 0.0L);                // twice_a(pr) - 2a
    std::printf("nested_arg10=%.12Lf\n", sb * sb + 4.0L * sa * sa); // add2(b, dbl(a))
    // loop: acc = sum_i w_i y_i, y_i measured in the loop with u_i, w_i = 1/u_i^2
    const long double u[3] = {0.1L, 0.2L, 0.15L};
    long double W = 0.0L;
    for (long double ui : u) W += 1.0L / (ui * ui);
    std::printf("loop_acc=%.12Lf\n", W);
    std::printf("pooled=%.12Lf\n", 1.0L / W);
    // variance_of(k) of a Knowledge returned by a function: 0.05^2
    std::printf("knowledge_ret=%.12Lf\n", 0.05L * 0.05L);
}
