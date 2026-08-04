#include <ap_int.h>

using q_t = ap_int<128>;
using wide_t = ap_int<256>;

struct interval_t {
    q_t lo;
    q_t hi;
};

static const int FRAC_BITS = 96;
static const int ORDER = 16;
static const q_t ONE = q_t(1) << FRAC_BITS;
static const q_t DOMAIN_LIMIT = q_t(1) << 111;
static const q_t STEP_RAW = ONE >> 8;
static const wide_t ROUND_MASK = (wide_t(1) << FRAC_BITS) - 1;

static q_t qmin(q_t a, q_t b) {
#pragma HLS INLINE
    return a < b ? a : b;
}

static q_t qmax(q_t a, q_t b) {
#pragma HLS INLINE
    return a > b ? a : b;
}

static q_t qabs(q_t value) {
#pragma HLS INLINE
    return value < 0 ? q_t(-value) : value;
}

static bool in_domain(q_t value) {
#pragma HLS INLINE
    return value > -DOMAIN_LIMIT && value < DOMAIN_LIMIT;
}

static q_t mul_floor(q_t left, q_t right) {
#pragma HLS INLINE
    wide_t product = wide_t(left) * wide_t(right);
    return q_t(product >> FRAC_BITS);
}

static q_t mul_ceil(q_t left, q_t right) {
#pragma HLS INLINE
    wide_t product = wide_t(left) * wide_t(right);
    return q_t((product + ROUND_MASK) >> FRAC_BITS);
}

static q_t div_floor(q_t value, q_t divisor) {
#pragma HLS INLINE
    q_t quotient = value / divisor;
    q_t remainder = value % divisor;
    return quotient - (remainder < 0 ? 1 : 0);
}

static q_t div_ceil(q_t value, q_t divisor) {
#pragma HLS INLINE
    q_t quotient = value / divisor;
    q_t remainder = value % divisor;
    return quotient + (remainder > 0 ? 1 : 0);
}

static interval_t add(interval_t a, interval_t b) {
#pragma HLS INLINE
    return {a.lo + b.lo, a.hi + b.hi};
}

static interval_t sub(interval_t a, interval_t b) {
#pragma HLS INLINE
    return {a.lo - b.hi, a.hi - b.lo};
}

static interval_t div2(interval_t value) {
#pragma HLS INLINE
    return {div_floor(value.lo, 2), div_ceil(value.hi, 2)};
}

static interval_t mul(interval_t a, interval_t b) {
#pragma HLS INLINE
    q_t p00_lo = mul_floor(a.lo, b.lo);
    q_t p01_lo = mul_floor(a.lo, b.hi);
    q_t p10_lo = mul_floor(a.hi, b.lo);
    q_t p11_lo = mul_floor(a.hi, b.hi);
    q_t p00_hi = mul_ceil(a.lo, b.lo);
    q_t p01_hi = mul_ceil(a.lo, b.hi);
    q_t p10_hi = mul_ceil(a.hi, b.lo);
    q_t p11_hi = mul_ceil(a.hi, b.hi);
    return {
        qmin(qmin(p00_lo, p01_lo), qmin(p10_lo, p11_lo)),
        qmax(qmax(p00_hi, p01_hi), qmax(p10_hi, p11_hi)),
    };
}

static interval_t step_div(interval_t value, int divisor) {
#pragma HLS INLINE
    interval_t step = {STEP_RAW, STEP_RAW};
    interval_t scaled = mul(value, step);
    return {div_floor(scaled.lo, divisor), div_ceil(scaled.hi, divisor)};
}

static q_t absolute_upper(interval_t value) {
#pragma HLS INLINE
    return qmax(qabs(value.lo), qabs(value.hi));
}

static void evaluate_field(const interval_t state[4], interval_t zs, interval_t result[4]) {
#pragma HLS INLINE
    interval_t yy = mul(state[1], state[1]);
    interval_t xy = mul(state[0], state[1]);
    interval_t wzs = add(state[2], zs);
    interval_t two_yy = {yy.lo << 1, yy.hi << 1};
    interval_t one = {ONE, ONE};
    result[0] = sub(two_yy, xy);
    result[1] = sub(xy, div2(mul(state[1], wzs)));
    result[2] = sub(sub(xy, state[2]), zs);
    result[3] = sub(sub(sub(state[0], state[1]), div2(wzs)), one);
}

static void lipschitz_rows(const interval_t box[4], interval_t zs, q_t rows[4]) {
#pragma HLS INLINE
    interval_t four_y = {box[1].lo << 2, box[1].hi << 2};
    interval_t diagonal_y = sub(box[0], div2(add(box[2], zs)));
    rows[0] = absolute_upper(box[1]) + absolute_upper(sub(four_y, box[0]));
    rows[1] = absolute_upper(box[1]) + absolute_upper(diagonal_y)
        + div_ceil(absolute_upper(box[1]), 2);
    rows[2] = absolute_upper(box[1]) + absolute_upper(box[0]) + ONE;
    rows[3] = (ONE << 1) + (ONE >> 1);
}

static void coefficients(const interval_t state[4], interval_t zs,
                         interval_t coeff[4][ORDER + 1], int maximum) {
#pragma HLS INLINE off
    interval_t zero = {0, 0};
    for (int axis = 0; axis < 4; ++axis) {
        for (int degree = 0; degree <= ORDER; ++degree) {
            coeff[axis][degree] = zero;
        }
        coeff[axis][0] = state[axis];
    }

degree_loop:
    for (int degree = 0; degree < maximum; ++degree) {
#pragma HLS LOOP_TRIPCOUNT min=15 max=16
        interval_t xy = zero;
        interval_t yy = zero;
        interval_t yw = zero;
convolution_loop:
        for (int j = 0; j <= degree; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
            xy = add(xy, mul(coeff[0][j], coeff[1][degree - j]));
            yy = add(yy, mul(coeff[1][j], coeff[1][degree - j]));
            yw = add(yw, mul(coeff[1][j], coeff[2][degree - j]));
        }
        interval_t two_yy = {yy.lo << 1, yy.hi << 1};
        interval_t y_rhs = sub(xy, div2(add(yw, mul(zs, coeff[1][degree]))));
        interval_t w_constant = degree == 0 ? zs : zero;
        interval_t ell_constant = degree == 0
            ? add(div2(zs), interval_t{ONE, ONE}) : zero;
        coeff[0][degree + 1] = step_div(sub(two_yy, xy), degree + 1);
        coeff[1][degree + 1] = step_div(y_rhs, degree + 1);
        coeff[2][degree + 1] = step_div(
            sub(xy, add(coeff[2][degree], w_constant)), degree + 1);
        coeff[3][degree + 1] = step_div(
            sub(sub(sub(coeff[0][degree], coeff[1][degree]),
                    div2(coeff[2][degree])), ell_constant), degree + 1);
    }
}

extern "C" void target23_scaled_taylor16(const q_t *input, q_t *output, int n_cases) {
#pragma HLS INTERFACE m_axi port=input bundle=gmem0 depth=54 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=output bundle=gmem1 depth=459 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=n_cases
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS ALLOCATION function instances=coefficients limit=1

case_loop:
    for (int index = 0; index < n_cases; ++index) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
        const int in_base = 18 * index;
        const int out_base = 153 * index;
        interval_t initial[4];
        interval_t box[4];
#pragma HLS ARRAY_PARTITION variable=initial complete
#pragma HLS ARRAY_PARTITION variable=box complete
        for (int component = 0; component < 4; ++component) {
#pragma HLS UNROLL
            initial[component] = {input[in_base + 2 * component], input[in_base + 2 * component + 1]};
            box[component] = {input[in_base + 8 + 2 * component], input[in_base + 9 + 2 * component]};
        }
        interval_t zs = {input[in_base + 16], input[in_base + 17]};
        q_t status = 1;
        for (int component = 0; component < 4; ++component) {
#pragma HLS UNROLL
            if (initial[component].lo > initial[component].hi || box[component].lo > box[component].hi) {
                status = -1;
            }
        }
        if (zs.lo > zs.hi) {
            status = -1;
        }
        if (status == 1) {
            for (int component = 0; component < 4; ++component) {
#pragma HLS UNROLL
                if (!in_domain(initial[component].lo) || !in_domain(initial[component].hi)
                    || !in_domain(box[component].lo) || !in_domain(box[component].hi)) {
                    status = -2;
                }
            }
            if (!in_domain(zs.lo) || !in_domain(zs.hi)) {
                status = -2;
            }
        }

        interval_t derivative[4];
        q_t rows[4];
#pragma HLS ARRAY_PARTITION variable=derivative complete
#pragma HLS ARRAY_PARTITION variable=rows complete
        if (status == 1) {
            evaluate_field(box, zs, derivative);
            interval_t time = {0, STEP_RAW};
            for (int component = 0; component < 4; ++component) {
#pragma HLS UNROLL
                interval_t image = add(initial[component], mul(time, derivative[component]));
                if (!(box[component].lo < image.lo && image.hi < box[component].hi)) {
                    status = -4;
                }
            }
        }
        if (status == 1) {
            lipschitz_rows(box, zs, rows);
            q_t maximum = qmax(qmax(rows[0], rows[1]), qmax(rows[2], rows[3]));
            if (div_ceil(maximum, 256) >= ONE) {
                status = -5;
            }
        }
        if (status != 1) {
            for (int word = 0; word < 152; ++word) {
                output[out_base + word] = 0;
            }
            output[out_base + 152] = status;
            continue;
        }

        interval_t center[4][ORDER + 1];
        interval_t wide[4][ORDER + 1];
        coefficients(initial, zs, center, ORDER - 1);
        coefficients(box, zs, wide, ORDER);
        int out_word = 0;
        for (int degree = 0; degree < ORDER; ++degree) {
            for (int axis = 0; axis < 4; ++axis) {
                output[out_base + out_word++] = center[axis][degree].lo;
                output[out_base + out_word++] = center[axis][degree].hi;
            }
        }
        interval_t polynomial[4];
        interval_t remainder[4];
        interval_t next_state[4];
#pragma HLS ARRAY_PARTITION variable=polynomial complete
#pragma HLS ARRAY_PARTITION variable=remainder complete
#pragma HLS ARRAY_PARTITION variable=next_state complete
        for (int axis = 0; axis < 4; ++axis) {
#pragma HLS UNROLL
            polynomial[axis] = {0, 0};
            for (int degree = 0; degree < ORDER; ++degree) {
                polynomial[axis] = add(polynomial[axis], center[axis][degree]);
            }
            remainder[axis] = wide[axis][ORDER];
            next_state[axis] = add(polynomial[axis], remainder[axis]);
        }
        for (int axis = 0; axis < 4; ++axis) {
            output[out_base + out_word++] = remainder[axis].lo;
            output[out_base + out_word++] = remainder[axis].hi;
        }
        for (int axis = 0; axis < 4; ++axis) {
            output[out_base + out_word++] = polynomial[axis].lo;
            output[out_base + out_word++] = polynomial[axis].hi;
        }
        for (int axis = 0; axis < 4; ++axis) {
            output[out_base + out_word++] = next_state[axis].lo;
            output[out_base + out_word++] = next_state[axis].hi;
        }
        output[out_base + 152] = status;
    }
}
