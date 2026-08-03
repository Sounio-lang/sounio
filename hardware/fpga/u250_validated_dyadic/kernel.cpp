#include <ap_int.h>

using q_t = ap_int<128>;
using wide_t = ap_int<256>;

static const int FRAC_BITS = 96;
static const q_t DOMAIN_LIMIT = q_t(1) << 111;
static const wide_t ROUND_MASK = (wide_t(1) << FRAC_BITS) - 1;

static bool in_domain(q_t value) {
#pragma HLS INLINE
    return value > -DOMAIN_LIMIT && value < DOMAIN_LIMIT;
}

static bool allowed_divisor(q_t divisor) {
#pragma HLS INLINE
    return divisor == 2 || divisor == 3 || divisor == 6 || divisor == 41;
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

static q_t min4(q_t a, q_t b, q_t c, q_t d) {
#pragma HLS INLINE
    q_t ab = a < b ? a : b;
    q_t cd = c < d ? c : d;
    return ab < cd ? ab : cd;
}

static q_t max4(q_t a, q_t b, q_t c, q_t d) {
#pragma HLS INLINE
    q_t ab = a > b ? a : b;
    q_t cd = c > d ? c : d;
    return ab > cd ? ab : cd;
}

extern "C" void validated_dyadic_kat(const q_t *input, q_t *output, int n_cases) {
#pragma HLS INTERFACE m_axi port=input bundle=gmem0 depth=480 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=output bundle=gmem1 depth=864 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=n_cases
#pragma HLS INTERFACE s_axilite port=return

case_loop:
    for (int index = 0; index < n_cases; ++index) {
#pragma HLS LOOP_TRIPCOUNT min=96 max=96
        const int in_base = 5 * index;
        const int out_base = 9 * index;
        q_t a_lo = input[in_base + 0];
        q_t a_hi = input[in_base + 1];
        q_t b_lo = input[in_base + 2];
        q_t b_hi = input[in_base + 3];
        q_t divisor = input[in_base + 4];
        q_t status = 1;
        if (a_lo > a_hi || b_lo > b_hi) {
            status = -1;
        } else if (!in_domain(a_lo) || !in_domain(a_hi)
                || !in_domain(b_lo) || !in_domain(b_hi)) {
            status = -2;
        } else if (!allowed_divisor(divisor)) {
            status = -3;
        }
        if (status != 1) {
            for (int word = 0; word < 8; ++word) {
#pragma HLS UNROLL
                output[out_base + word] = 0;
            }
            output[out_base + 8] = status;
            continue;
        }

        q_t p00_lo = mul_floor(a_lo, b_lo);
        q_t p01_lo = mul_floor(a_lo, b_hi);
        q_t p10_lo = mul_floor(a_hi, b_lo);
        q_t p11_lo = mul_floor(a_hi, b_hi);
        q_t p00_hi = mul_ceil(a_lo, b_lo);
        q_t p01_hi = mul_ceil(a_lo, b_hi);
        q_t p10_hi = mul_ceil(a_hi, b_lo);
        q_t p11_hi = mul_ceil(a_hi, b_hi);

        output[out_base + 0] = a_lo + b_lo;
        output[out_base + 1] = a_hi + b_hi;
        output[out_base + 2] = a_lo - b_hi;
        output[out_base + 3] = a_hi - b_lo;
        output[out_base + 4] = min4(p00_lo, p01_lo, p10_lo, p11_lo);
        output[out_base + 5] = max4(p00_hi, p01_hi, p10_hi, p11_hi);
        output[out_base + 6] = div_floor(a_lo, divisor);
        output[out_base + 7] = div_ceil(a_hi, divisor);
        output[out_base + 8] = status;
    }
}
