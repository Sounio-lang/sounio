// paged_attn_f64.cu -- C reference for paged_attn_f64.ptx.
//
// Same op order as ll_runs_layer's row path in stdlib/llm/llama.sio,
// f64, separate mul/add, same exp emitter (Cody-Waite constants,
// Horner mul-then-add from 1/13!, exponent-field reconstruction,
// round-to-nearest-even for n). Compile with a C++ compiler for a
// host-side oracle; the PTX is the device artifact under test.

static double pow2_ref(long long m) {
    // exact 2^m like exp_pow2_pos (products of exact powers of two),
    // single rounding only for negative m (1.0 / exact).
    unsigned long long am = (unsigned long long)(m < 0 ? -m : m);
    unsigned long long bits = (1023ULL + am) << 52;
    double e;
    __builtin_memcpy(&e, &bits, 8);
    if (m < 0) e = 1.0 / e;
    return e;
}

// math::pure::exp replica (stdlib/math/pure.sio): the host forward
// calls this Sounio exp, so the device must match it op-for-op,
// including truncation, nesting order, and the split powers.
static double pure_exp_ref(double x) {
    if (x != x) return x;
    if (x > 709.782712893384) return 1.0e308 * 10.0;
    if (x < -745.1332191019412) return 0.0;
    double q = x * 1.4426950408889634;
    long long k = (long long)(q >= 0.0 ? q + 0.5 : q - 0.5);
    double kf = (double)k;
    double r = (x - kf * 6.93147180369123816490e-01)
                   - kf * 1.90821492927058770002e-10;
    double acc = r * 1.6059043836821613e-10;
    acc = 2.0876756987868100e-09 + r * acc;
    acc = 2.5052108385441720e-08 + r * acc;
    acc = 2.7557319223985888e-07 + r * acc;
    acc = 2.7557319223985893e-06 + r * acc;
    acc = 2.4801587301587302e-05 + r * acc;
    acc = 1.9841269841269841e-04 + r * acc;
    acc = 1.3888888888888889e-03 + r * acc;
    acc = 8.3333333333333332e-03 + r * acc;
    acc = 4.1666666666666664e-02 + r * acc;
    acc = 1.6666666666666666e-01 + r * acc;
    acc = 0.5 + r * acc;
    acc = 1.0 + r * acc;
    double p = 1.0 + r * acc;
    long long k1 = k / 2;               // truncating, like idiv/div.s64
    double e1 = pow2_ref(k1), e2 = pow2_ref(k - k1);
    return (p * e1) * e2;
}

// q      [nrows, heads, dh]      post-rope queries, f64
// pool_k [npool, B, kv_heads, dh] f64 ; pool_v same layout
// table  [nrows, maxblocks] u64 ; lens [nrows] u64 (host stages i64)
// scale  passed by value here; the PTX takes a device pointer to
//        one f64 so the Sounio host never needs f64 bit conversion.
// sco    [nrows, heads, maxblocks*B] f64 scratch, one lane
//        per (row, head); shared layout was a same-row race
// out    [nrows, heads, dh]     f64 ; proof[0] += finished rows*heads
void paged_attn_f64_ref(const double *q, const double *pool_k,
                        const double *pool_v,
                        const unsigned long long *table,
                        const unsigned long long *lens,
                        double *sco, double *out,
                        unsigned long long *proof,
                        int nrows, int heads, int kv_heads, int dh,
                        int B, int maxblocks, double scale) {
    int kvdim = kv_heads * dh;
    int rep = heads / kv_heads;
    for (int row = 0; row < nrows; row++) {
        int len = (int)lens[row];
        for (int h = 0; h < heads; h++) {
            int kvh = h / rep;
            double mx = -1e9;
            for (int t = 0; t < len; t++) {
                int bi = t / B, pos = t % B;
                unsigned long long bo =
                    table[row * maxblocks + bi] * (unsigned)B + (unsigned)pos;
                double dot = 0.0;
                for (int d = 0; d < dh; d++)
                    dot += q[(row * heads + h) * dh + d]
                         * pool_k[(bo * kv_heads + kvh) * dh + d];
                double sc = dot * scale;
                sco[(row * heads + h) * maxblocks * B + t] = sc;
                if (sc > mx) mx = sc;
            }
            double sum = 0.0;
            for (int t = 0; t < len; t++) {
                double e = pure_exp_ref(sco[(row * heads + h) * maxblocks * B + t] - mx);
                sco[(row * heads + h) * maxblocks * B + t] = e;
                sum += e;
            }
            for (int d = 0; d < dh; d++) {
                double acc = 0.0;
                for (int t = 0; t < len; t++) {
                    int bi = t / B, pos = t % B;
                    unsigned long long bo =
                        table[row * maxblocks + bi] * (unsigned)B + (unsigned)pos;
                    acc += sco[(row * heads + h) * maxblocks * B + t] / sum
                         * pool_v[(bo * kv_heads + kvh) * dh + d];
                }
                out[(row * heads + h) * dh + d] = acc;
            }
            proof[0] += 1;
        }
    }
    (void)kvdim;
}
