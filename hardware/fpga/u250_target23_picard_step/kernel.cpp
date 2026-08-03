#include <ap_int.h>

using q_t = ap_int<128>;
using wide_t = ap_int<256>;

struct interval_t {
    q_t lo;
    q_t hi;
};

static const int FRAC_BITS = 96;
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
    q_t negative = q_t(-value);
    return value < 0 ? negative : value;
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
    q_t lower = qmin(qmin(p00_lo, p01_lo), qmin(p10_lo, p11_lo));
    q_t upper = qmax(qmax(p00_hi, p01_hi), qmax(p10_hi, p11_hi));
    return {lower, upper};
}

static q_t absolute_upper(interval_t value) {
#pragma HLS INLINE
    return qmax(qabs(value.lo), qabs(value.hi));
}

static void evaluate_field(const interval_t state[4], interval_t zs, interval_t result[4]) {
#pragma HLS INLINE
    interval_t x = state[0];
    interval_t y = state[1];
    interval_t w = state[2];
    interval_t yy = mul(y, y);
    interval_t xy = mul(x, y);
    interval_t wzs = add(w, zs);
    interval_t two_yy = {yy.lo << 1, yy.hi << 1};
    interval_t one = {ONE, ONE};
    result[0] = sub(two_yy, xy);
    result[1] = sub(xy, div2(mul(y, wzs)));
    result[2] = sub(sub(xy, w), zs);
    result[3] = sub(sub(sub(x, y), div2(wzs)), one);
}

static void lipschitz_rows(const interval_t box[4], interval_t zs, q_t rows[4]) {
#pragma HLS INLINE
    interval_t x = box[0];
    interval_t y = box[1];
    interval_t w = box[2];
    interval_t four_y = {y.lo << 2, y.hi << 2};
    interval_t four_y_minus_x = sub(four_y, x);
    interval_t diagonal_y = sub(x, div2(add(w, zs)));
    rows[0] = absolute_upper(y) + absolute_upper(four_y_minus_x);
    rows[1] = absolute_upper(y) + absolute_upper(diagonal_y)
        + div_ceil(absolute_upper(y), 2);
    rows[2] = absolute_upper(y) + absolute_upper(x) + ONE;
    rows[3] = (ONE << 1) + (ONE >> 1);
}

extern "C" void target23_picard_step(const q_t *input, q_t *output, int n_cases) {
#pragma HLS INTERFACE m_axi port=input bundle=gmem0 depth=72 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=output bundle=gmem1 depth=88 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=n_cases
#pragma HLS INTERFACE s_axilite port=return

case_loop:
    for (int index = 0; index < n_cases; ++index) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
        const int in_base = 18 * index;
        const int out_base = 22 * index;
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
        interval_t image[4];
        q_t rows[4];
#pragma HLS ARRAY_PARTITION variable=derivative complete
#pragma HLS ARRAY_PARTITION variable=image complete
#pragma HLS ARRAY_PARTITION variable=rows complete
        if (status == 1) {
            evaluate_field(box, zs, derivative);
            interval_t time = {0, STEP_RAW};
            for (int component = 0; component < 4; ++component) {
#pragma HLS UNROLL
                image[component] = add(initial[component], mul(time, derivative[component]));
                if (!(box[component].lo < image[component].lo
                    && image[component].hi < box[component].hi)) {
                    status = -4;
                }
            }
        }
        q_t contraction = 0;
        if (status == 1) {
            lipschitz_rows(box, zs, rows);
            q_t maximum = qmax(qmax(rows[0], rows[1]), qmax(rows[2], rows[3]));
            contraction = div_ceil(maximum, 256);
            if (contraction >= ONE) {
                status = -5;
            }
        }
        if (status != 1) {
            for (int word = 0; word < 21; ++word) {
#pragma HLS UNROLL
                output[out_base + word] = 0;
            }
            output[out_base + 21] = status;
            continue;
        }
        for (int component = 0; component < 4; ++component) {
#pragma HLS UNROLL
            output[out_base + 2 * component] = derivative[component].lo;
            output[out_base + 2 * component + 1] = derivative[component].hi;
            output[out_base + 8 + 2 * component] = image[component].lo;
            output[out_base + 9 + 2 * component] = image[component].hi;
            output[out_base + 16 + component] = rows[component];
        }
        output[out_base + 20] = contraction;
        output[out_base + 21] = status;
    }
}
