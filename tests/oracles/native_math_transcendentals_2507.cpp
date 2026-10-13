// C++23 oracle for the transcendental fixtures of #2507:
//   tests/run-pass/native_math_transcendentals_values.sio  (the function values)
//   tests/run-pass/epistemic_hessian_transcendentals.sio   (f'' at each point)
// Build and run:
//   g++ -std=c++23 -O2 tests/oracles/native_math_transcendentals_2507.cpp -o /tmp/o2507 && /tmp/o2507
// Each constant written in the .sio files is compared with C libm (values) or
// with the closed-form second derivative evaluated in long double (Hessians).
// Exit status 0 means every constant agrees; any disagreement is printed.
#include <cmath>
#include <cstdio>

enum Fn { atan_k, tan_k, tanh_k, asin_k, acos_k };

struct Row { const char* label; Fn f; double x; double want; };

static double eval(Fn f, double x) {
    switch (f) {
        case atan_k: return std::atan(x);
        case tan_k:  return std::tan(x);
        case tanh_k: return std::tanh(x);
        case asin_k: return std::asin(x);
        case acos_k: return std::acos(x);
    }
    return 0.0;
}

static const Row rows[] = {
    {"atan(0.1)", atan_k, 0.1, 0.09966865249116204},
    {"atan(0.25)", atan_k, 0.25, 0.24497866312686414},
    {"atan(0.4142135623730951)", atan_k, 0.4142135623730951, 0.3926990816987242},
    {"atan(0.5)", atan_k, 0.5, 0.4636476090008061},
    {"atan(0.9)", atan_k, 0.9, 0.7328151017865066},
    {"atan(0.99999)", atan_k, 0.99999, 0.7853931633724482},
    {"atan(1.0)", atan_k, 1.0, 0.7853981633974483},
    {"atan(1.5)", atan_k, 1.5, 0.982793723247329},
    {"atan(2.0)", atan_k, 2.0, 1.1071487177940904},
    {"atan(10.0)", atan_k, 10.0, 1.4711276743037347},
    {"atan(1e10)", atan_k, 10000000000.0, 1.5707963266948965},
    {"atan(1e300)", atan_k, 1.0e300, 1.5707963267948966},
    {"atan(-0.5)", atan_k, -(0.5), -(0.4636476090008061)},
    {"atan(-2.0)", atan_k, -(2.0), -(1.1071487177940904)},
    {"tan(0.1)", tan_k, 0.1, 0.10033467208545055},
    {"tan(0.5)", tan_k, 0.5, 0.5463024898437905},
    {"tan(1.0)", tan_k, 1.0, 1.5574077246549023},
    {"tan(1.5)", tan_k, 1.5, 14.101419947171719},
    {"tan(-1.5)", tan_k, -(1.5), -(14.101419947171719)},
    {"tan(2.0)", tan_k, 2.0, -(2.185039863261519)},
    {"tan(3.0)", tan_k, 3.0, -(0.1425465430742778)},
    {"tan(4.0)", tan_k, 4.0, 1.1578212823495775},
    {"tan(10.0)", tan_k, 10.0, 0.6483608274590866},
    {"tan(100.0)", tan_k, 100.0, -(0.5872139151569291)},
    {"tan(1e6)", tan_k, 1000000.0, -(0.373624453987599)},
    {"tan(1647099.0)", tan_k, 1647099.0, -(0.34159248796294767)},
    {"tan(-0.5)", tan_k, -(0.5), -(0.5463024898437905)},
    {"tanh(1e-10)", tanh_k, 0.0000000001, 0.0000000001},
    {"tanh(0.1)", tanh_k, 0.1, 0.09966799462495582},
    {"tanh(0.5)", tanh_k, 0.5, 0.46211715726000974},
    {"tanh(0.9)", tanh_k, 0.9, 0.7162978701990245},
    {"tanh(0.99999)", tanh_k, 0.99999, 0.7615899561803637},
    {"tanh(1.0)", tanh_k, 1.0, 0.7615941559557649},
    {"tanh(1.5)", tanh_k, 1.5, 0.9051482536448664},
    {"tanh(2.0)", tanh_k, 2.0, 0.9640275800758169},
    {"tanh(10.0)", tanh_k, 10.0, 0.9999999958776927},
    {"tanh(-2.0)", tanh_k, -(2.0), -(0.9640275800758169)},
    {"asin(1e-10)", asin_k, 0.0000000001, 0.0000000001},
    {"asin(0.1)", asin_k, 0.1, 0.1001674211615598},
    {"asin(0.5)", asin_k, 0.5, 0.5235987755982989},
    {"asin(0.9)", asin_k, 0.9, 1.1197695149986342},
    {"asin(0.99999)", asin_k, 0.99999, 1.5663241871131188},
    {"asin(1.0)", asin_k, 1.0, 1.5707963267948966},
    {"asin(-0.5)", asin_k, -(0.5), -(0.5235987755982989)},
    {"acos(0.0)", acos_k, 0.0, 1.5707963267948966},
    {"acos(1e-10)", acos_k, 0.0000000001, 1.5707963266948965},
    {"acos(0.1)", acos_k, 0.1, 1.4706289056333368},
    {"acos(0.5)", acos_k, 0.5, 1.0471975511965979},
    {"acos(0.9)", acos_k, 0.9, 0.45102681179626236},
    {"acos(0.99999)", acos_k, 0.99999, 0.0044721396817777506},
    {"acos(-0.5)", acos_k, -(0.5), 2.0943951023931957},
    {"acos(-1.0)", acos_k, -(1.0), 3.141592653589793},
};

// Second derivatives, closed form, in long double.
static long double d2_atan(long double x) { return -2.0L * x / ((1.0L + x * x) * (1.0L + x * x)); }
static long double d2_tan(long double x)  { long double t = std::tan(x); return 2.0L * t * (1.0L + t * t); }
static long double d2_tanh(long double x) { long double t = std::tanh(x); return -2.0L * t * (1.0L - t * t); }
static long double d2_asin(long double x) { return x / std::pow(1.0L - x * x, 1.5L); }
static long double d2_acos(long double x) { return -x / std::pow(1.0L - x * x, 1.5L); }
static long double d1_atan(long double x) { return 1.0L / (1.0L + x * x); }
static long double d1_asin(long double x) { return 1.0L / std::sqrt(1.0L - x * x); }

struct Hrow { const char* label; long double analytic; double written; };

int main() {
    int bad = 0;
    for (const Row& r : rows) {
        const double got = eval(r.f, r.x);
        double m = std::fabs(r.want);
        if (m < 1.0) m = 1.0;
        if (!(std::fabs(got - r.want) <= 4.0e-15 * m)) {
            std::printf("VALUE MISMATCH %s libm=%.17g written=%.17g\n", r.label, got, r.want);
            ++bad;
        }
    }
    // The constants asserted by epistemic_hessian_transcendentals.sio.
    const Hrow h[] = {
        {"atan''(1)",          d2_atan(1.0L),  -0.5},
        {"tan''(0)",           d2_tan(0.0L),    0.0},
        {"d2(x*atan(y))/dxdy at y=1 = atan'(1)", d1_atan(1.0L), 0.5},
        {"tanh''(0)",          d2_tanh(0.0L),   0.0},
        {"d2(asin(x)*y)/dxdy at x=0 = asin'(0)", d1_asin(0.0L), 1.0},
        {"acos''(0)",          d2_acos(0.0L),   0.0},
        {"atan''(0.5)",        d2_atan(0.5L),  -0.64},
        {"tan''(0.5)",         d2_tan(0.5L),    1.4186890138709112},
        {"tanh''(0.5)",        d2_tanh(0.5L),  -0.7268619813835873},
        {"asin''(0.5)",        d2_asin(0.5L),   0.769800358919501},
        {"acos''(0.5)",        d2_acos(0.5L),  -0.769800358919501},
    };
    for (const Hrow& r : h) {
        std::printf("%-40s analytic=%.17Lg written=%.17g\n", r.label, r.analytic, r.written);
        if (!(std::fabs(static_cast<double>(r.analytic) - r.written) <= 1.0e-15)) {
            std::printf("HESSIAN MISMATCH %s\n", r.label);
            ++bad;
        }
    }
    std::printf("%zu values, %zu hessians, %d mismatches\n", sizeof rows / sizeof rows[0], sizeof h / sizeof h[0], bad);
    return bad == 0 ? 0 : 1;
}
