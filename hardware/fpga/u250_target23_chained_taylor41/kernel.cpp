#include <ap_int.h>

using q_t = ap_int<224>;
using uq_t = ap_uint<224>;
using wide_t = ap_int<448>;
using limb_t = ap_uint<32>;

struct interval_t { q_t lo; q_t hi; };

static const int F = 192;
static const int ORDER = 41;
static const int STEPS_PER_PARTITION = 843;
static const q_t ONE = q_t(1) << F;
static const q_t STEP = ONE >> 8;
static const q_t PAD = q_t(1) << 96;
static const q_t DOMAIN_LIMIT = ONE << 15;
static const q_t RADIUS_LIMIT = ONE >> 16;
static const wide_t ROUND_MASK = (wide_t(1) << F) - 1;

static q_t qmin(q_t a, q_t b) { return a < b ? a : b; }
static q_t qmax(q_t a, q_t b) { return a > b ? a : b; }
static q_t qabs(q_t a) { return a < 0 ? q_t(-a) : a; }

static q_t floor_div(q_t value, int divisor) {
    q_t quotient = value / divisor;
    q_t remainder = value % divisor;
    return quotient - (remainder < 0 ? 1 : 0);
}

static q_t ceil_div(q_t value, int divisor) {
    q_t quotient = value / divisor;
    q_t remainder = value % divisor;
    return quotient + (remainder > 0 ? 1 : 0);
}

static q_t ceil_wide(wide_t numerator, wide_t denominator) {
    wide_t quotient = numerator / denominator;
    wide_t remainder = numerator % denominator;
    return q_t(quotient + (remainder > 0 ? 1 : 0));
}

static q_t scaled_floor(wide_t product) { return q_t(product >> F); }
static q_t scaled_ceil(wide_t product) { return q_t((product + ROUND_MASK) >> F); }

static interval_t plus_i(interval_t a, interval_t b) { return {a.lo + b.lo, a.hi + b.hi}; }
static interval_t minus_i(interval_t a, interval_t b) { return {a.lo - b.hi, a.hi - b.lo}; }
static interval_t half_i(interval_t a) { return {floor_div(a.lo, 2), ceil_div(a.hi, 2)}; }
static q_t magnitude(interval_t a) { return qmax(qabs(a.lo), qabs(a.hi)); }

static interval_t times_i(interval_t a, interval_t b) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION operation instances=mul limit=1
    wide_t p00 = wide_t(a.lo) * wide_t(b.lo);
    wide_t p01 = wide_t(a.lo) * wide_t(b.hi);
    wide_t p10 = wide_t(a.hi) * wide_t(b.lo);
    wide_t p11 = wide_t(a.hi) * wide_t(b.hi);
    wide_t lower = p00;
    wide_t upper = p00;
    if (p01 < lower) lower = p01; if (p01 > upper) upper = p01;
    if (p10 < lower) lower = p10; if (p10 > upper) upper = p10;
    if (p11 < lower) lower = p11; if (p11 > upper) upper = p11;
    return {scaled_floor(lower), scaled_ceil(upper)};
}

static interval_t scaled_divide(interval_t value, q_t step, int divisor) {
    interval_t scaled = times_i(value, {step, step});
    return {floor_div(scaled.lo, divisor), ceil_div(scaled.hi, divisor)};
}

static void field(const interval_t state[4], interval_t zs, interval_t out[4]) {
#pragma HLS ALLOCATION function instances=times_i limit=1
    interval_t xy = times_i(state[0], state[1]);
    interval_t yy = times_i(state[1], state[1]);
    interval_t wz = plus_i(state[2], zs);
    out[0] = minus_i({yy.lo << 1, yy.hi << 1}, xy);
    out[1] = minus_i(xy, half_i(times_i(state[1], wz)));
    out[2] = minus_i(minus_i(xy, state[2]), zs);
    out[3] = minus_i(minus_i(minus_i(state[0], state[1]), half_i(wz)), {ONE, ONE});
}

static void picard_image(const interval_t initial[4], const interval_t box[4], interval_t zs, q_t step, interval_t out[4]) {
#pragma HLS ALLOCATION function instances=times_i limit=1
    interval_t derivative[4];
    field(box, zs, derivative);
    for (int axis = 0; axis < 4; ++axis) out[axis] = plus_i(initial[axis], times_i({0, step}, derivative[axis]));
}

static q_t ordinary_lipschitz(const interval_t box[4], interval_t zs) {
    interval_t four_y = {box[1].lo << 2, box[1].hi << 2};
    q_t row0 = magnitude(box[1]) + magnitude(minus_i(four_y, box[0]));
    q_t row1 = magnitude(box[1]) + magnitude(minus_i(box[0], half_i(plus_i(box[2], zs)))) + ceil_div(magnitude(box[1]), 2);
    q_t row2 = magnitude(box[1]) + magnitude(box[0]) + ONE;
    return qmax(qmax(row0, row1), qmax(row2, q_t(5) * ONE / 2));
}

static q_t logarithmic_norm(const interval_t box[4], interval_t zs) {
    interval_t four_y = {box[1].lo << 2, box[1].hi << 2};
    q_t row0 = -box[1].lo + magnitude(minus_i(four_y, box[0]));
    q_t row1 = minus_i(box[0], half_i(plus_i(box[2], zs))).hi + magnitude(box[1]) + ceil_div(magnitude(box[1]), 2);
    q_t row2 = -ONE + magnitude(box[1]) + magnitude(box[0]);
    return qmax(qmax(row0, row1), qmax(row2, q_t(5) * ONE / 2));
}

static bool close_box(const interval_t initial[4], interval_t zs, q_t step, interval_t result[4], int &iterations, q_t &contraction) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION function instances=picard_image limit=1
    for (int axis = 0; axis < 4; ++axis) {
        if (initial[axis].lo > initial[axis].hi || initial[axis].lo <= -DOMAIN_LIMIT || initial[axis].hi >= DOMAIN_LIMIT) return false;
        result[axis] = initial[axis];
    }
    for (int iteration = 1; iteration <= 512; ++iteration) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=100 avg=24
        interval_t mapped[4];
        picard_image(initial, result, zs, step, mapped);
        bool stable = true;
        interval_t widened[4];
        for (int axis = 0; axis < 4; ++axis) {
            widened[axis] = {qmin(result[axis].lo, mapped[axis].lo), qmax(result[axis].hi, mapped[axis].hi)};
            if (widened[axis].lo != result[axis].lo || widened[axis].hi != result[axis].hi) stable = false;
        }
        if (stable) {
            interval_t padded[4];
            interval_t checked[4];
            for (int axis = 0; axis < 4; ++axis) padded[axis] = {result[axis].lo - PAD, result[axis].hi + PAD};
            picard_image(initial, padded, zs, step, checked);
            for (int axis = 0; axis < 4; ++axis) {
                if (!(padded[axis].lo < checked[axis].lo && checked[axis].hi < padded[axis].hi)) return false;
                result[axis] = padded[axis];
            }
            contraction = scaled_ceil(wide_t(ordinary_lipschitz(result, zs)) * wide_t(step));
            iterations = iteration;
            return contraction < ONE;
        }
        for (int axis = 0; axis < 4; ++axis) result[axis] = widened[axis];
    }
    return false;
}

static void coefficients(const interval_t state[4], interval_t zs, q_t step, interval_t coeff[4][ORDER + 1], int maximum) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION function instances=times_i limit=1
    for (int axis = 0; axis < 4; ++axis) {
        for (int degree = 0; degree <= ORDER; ++degree) coeff[axis][degree] = {0, 0};
        coeff[axis][0] = state[axis];
    }
    for (int degree = 0; degree < maximum; ++degree) {
#pragma HLS LOOP_TRIPCOUNT min=40 max=41
        interval_t xy = {0, 0};
        interval_t yy = {0, 0};
        interval_t yw = {0, 0};
        for (int j = 0; j <= degree; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=41 avg=21
            xy = plus_i(xy, times_i(coeff[0][j], coeff[1][degree - j]));
            yy = plus_i(yy, times_i(coeff[1][j], coeff[1][degree - j]));
            yw = plus_i(yw, times_i(coeff[1][j], coeff[2][degree - j]));
        }
        interval_t forcing_w = degree == 0 ? zs : interval_t{0, 0};
        interval_t forcing_ell = degree == 0 ? plus_i(half_i(zs), {ONE, ONE}) : interval_t{0, 0};
        coeff[0][degree + 1] = scaled_divide(minus_i({yy.lo << 1, yy.hi << 1}, xy), step, degree + 1);
        coeff[1][degree + 1] = scaled_divide(minus_i(xy, half_i(plus_i(yw, times_i(zs, coeff[1][degree])))), step, degree + 1);
        coeff[2][degree + 1] = scaled_divide(minus_i(xy, plus_i(coeff[2][degree], forcing_w)), step, degree + 1);
        coeff[3][degree + 1] = scaled_divide(minus_i(minus_i(minus_i(coeff[0][degree], coeff[1][degree]), half_i(coeff[2][degree])), forcing_ell), step, degree + 1);
    }
}

static q_t exp_upper(q_t argument) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION operation instances=mul limit=1
    if (argument <= 0) return ONE;
    q_t term = ONE;
    q_t result = ONE;
    for (int degree = 1; degree <= 32; ++degree) {
        term = ceil_wide(wide_t(term) * wide_t(argument), wide_t(ONE) * degree);
        result += term;
    }
    q_t following = ceil_wide(wide_t(term) * wide_t(argument), wide_t(ONE) * 33);
    q_t ratio = ceil_div(argument, 34);
    q_t tail = ceil_wide(wide_t(following) * wide_t(ONE), wide_t(ONE - ratio));
    return result + tail;
}

static int section_sign(q_t center, q_t radius) {
    if (center + radius < 0) return -1;
    if (center - radius > 0) return 1;
    return 0;
}

static bool advance(const q_t center[4], q_t radius, interval_t zs, q_t step,
                    q_t next_center[4], q_t &next_radius) {
#pragma HLS INLINE off
#pragma HLS ALLOCATION function instances=close_box limit=1
#pragma HLS ALLOCATION function instances=coefficients limit=1
    interval_t initial[4];
    interval_t point[4];
    for (int axis = 0; axis < 4; ++axis) {
        initial[axis] = {center[axis] - radius, center[axis] + radius};
        point[axis] = {center[axis], center[axis]};
    }
    interval_t box[4];
    interval_t point_box[4];
    int iterations = 0;
    int point_iterations = 0;
    q_t contraction = 0;
    q_t point_contraction = 0;
    if (!close_box(initial, zs, step, box, iterations, contraction)) return false;
    if (!close_box(point, zs, step, point_box, point_iterations, point_contraction)) return false;
    interval_t center_coeff[4][ORDER + 1];
    interval_t wide_coeff[4][ORDER + 1];
    coefficients(point, zs, step, center_coeff, ORDER - 1);
    coefficients(point_box, zs, step, wide_coeff, ORDER);
    q_t local = 0;
    for (int axis = 0; axis < 4; ++axis) {
        interval_t polynomial = {0, 0};
        for (int degree = 0; degree < ORDER; ++degree) polynomial = plus_i(polynomial, center_coeff[axis][degree]);
        interval_t enclosure = plus_i(polynomial, wide_coeff[axis][ORDER]);
        q_t midpoint = floor_div(enclosure.lo + enclosure.hi, 2);
        next_center[axis] = midpoint;
        local = qmax(local, qmax(midpoint - enclosure.lo, enclosure.hi - midpoint));
    }
    q_t mu = logarithmic_norm(box, zs);
    q_t mu_h = mu > 0 ? scaled_ceil(wide_t(mu) * wide_t(step)) : q_t(0);
    q_t amplification = exp_upper(mu_h);
    q_t propagated = scaled_ceil(wide_t(radius) * wide_t(amplification));
    next_radius = propagated + local;
    return next_radius < RADIUS_LIMIT;
}

static bool localize_event(const q_t center[4], q_t radius, interval_t zs, q_t step) {
#pragma HLS INLINE
#pragma HLS ALLOCATION function instances=advance limit=1
    q_t low = 0;
    q_t high = step;
    q_t low_center[4];
    q_t high_center[4];
    for (int axis = 0; axis < 4; ++axis) low_center[axis] = center[axis];
    q_t low_radius = radius;
    q_t high_radius = 0;
    if (!advance(center, radius, zs, high, high_center, high_radius) || section_sign(high_center[2], high_radius) != 1) return false;
    for (int iteration = 0; iteration < 42; ++iteration) {
#pragma HLS LOOP_TRIPCOUNT min=42 max=42
        q_t middle = floor_div(low + high, 2);
        q_t middle_center[4];
        q_t middle_radius = 0;
        if (!advance(center, radius, zs, middle, middle_center, middle_radius)) return false;
        int sign = section_sign(middle_center[2], middle_radius);
        if (sign < 0) {
            low = middle;
            low_radius = middle_radius;
            for (int axis = 0; axis < 4; ++axis) low_center[axis] = middle_center[axis];
        } else if (sign > 0) {
            high = middle;
            high_radius = middle_radius;
            for (int axis = 0; axis < 4; ++axis) high_center[axis] = middle_center[axis];
        } else {
            break;
        }
    }
    if (section_sign(low_center[2], low_radius) != -1 || section_sign(high_center[2], high_radius) != 1 || high - low > (ONE >> 50)) return false;
    interval_t event_initial[4];
    for (int axis = 0; axis < 4; ++axis) event_initial[axis] = {low_center[axis] - low_radius, low_center[axis] + low_radius};
    interval_t event_box[4];
    int iterations = 0;
    q_t contraction = 0;
    if (!close_box(event_initial, zs, high - low, event_box, iterations, contraction)) return false;
    interval_t normal = minus_i(times_i(event_box[0], event_box[1]), zs);
    return normal.lo > 0;
}

static q_t load_q(const limb_t *memory, int word) {
    uq_t bits = 0;
    for (int limb = 0; limb < 7; ++limb) bits.range(32 * limb + 31, 32 * limb) = memory[7 * word + limb];
    return q_t(bits);
}

static void store_q(limb_t *memory, int word, q_t value) {
    uq_t bits = uq_t(value);
    for (int limb = 0; limb < 7; ++limb) memory[7 * word + limb] = bits.range(32 * limb + 31, 32 * limb);
}

extern "C" void target23_chained_taylor41(const limb_t *input, limb_t *output, int partition) {
#pragma HLS INTERFACE m_axi port=input bundle=gmem0 depth=182 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=output bundle=gmem1 depth=59010 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=partition
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS ALLOCATION function instances=times_i limit=1
#pragma HLS ALLOCATION function instances=coefficients limit=1
#pragma HLS ALLOCATION function instances=advance limit=1
    const int base = 13 * partition;
    int partition_id = load_q(input, base + 0).to_int();
    int start_step = load_q(input, base + 1).to_int();
    int count = load_q(input, base + 2).to_int();
    q_t time = load_q(input, base + 3);
    q_t center[4];
    for (int axis = 0; axis < 4; ++axis) center[axis] = load_q(input, base + 4 + axis);
    q_t radius = load_q(input, base + 8);
    interval_t zs = {load_q(input, base + 9), load_q(input, base + 10)};
    bool armed = load_q(input, base + 11) != 0;
    int events = load_q(input, base + 12).to_int();
    if (partition_id != partition || count != STEPS_PER_PARTITION || start_step != partition * STEPS_PER_PARTITION) return;
    for (int local_step = 0; local_step < STEPS_PER_PARTITION; ++local_step) {
#pragma HLS LOOP_TRIPCOUNT min=843 max=843
        int before = section_sign(center[2], radius);
        q_t following[4];
        q_t following_radius = 0;
        bool valid = advance(center, radius, zs, STEP, following, following_radius);
        int after = valid ? section_sign(following[2], following_radius) : 0;
        int event_index = 0;
        if (after < 0) armed = true;
        if (valid && armed && before < 0 && after > 0) {
            if (localize_event(center, radius, zs, STEP)) {
                ++events;
                event_index = events;
                armed = false;
            } else {
                valid = false;
            }
        }
        const int out = 10 * local_step;
        store_q(output, out + 0, time);
        store_q(output, out + 1, time + STEP);
        store_q(output, out + 2, STEP);
        store_q(output, out + 3, 8);
        for (int axis = 0; axis < 4; ++axis) store_q(output, out + 4 + axis, valid ? following[axis] : q_t(0));
        store_q(output, out + 8, valid ? following_radius : q_t(-1));
        store_q(output, out + 9, valid ? q_t(event_index) : q_t(-9));
        if (!valid) return;
        for (int axis = 0; axis < 4; ++axis) center[axis] = following[axis];
        radius = following_radius;
        time += STEP;
    }
}
