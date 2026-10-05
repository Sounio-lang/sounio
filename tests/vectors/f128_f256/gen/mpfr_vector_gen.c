/* mpfr_vector_gen.c
 *
 * Deterministic generator for the Sounio f128/f256 external-oracle test
 * vector corpus. Wave 1, lane ws-g-mpfr-vectors.
 *
 * Inputs and outputs are IEEE-754 binary128 / binary256 wire bit
 * patterns (sign + biased-exponent + trailing-significand). Operations
 * are computed with MPFR at extended precision, then rounded back to
 * the target format using IEEE-754 round-to-nearest-even with gradual
 * underflow and all the IEEE-754-2008 special-case rules.
 *
 * Output is JSONL (one vector per line) on stdout. The companion
 * shell wrapper records tool versions and seeds in
 * tests/vectors/f128_f256/GENERATION_RECEIPT.md.
 *
 * Determinism: PCG-XSH-RR with a fixed seed. No clock reads, no
 * /dev/urandom.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <mpfr.h>
#include <gmp.h>

/* -------------------------------------------------------------------------
 * IEEE-754 binaryN parameters
 * ----------------------------------------------------------------------- */
#define F128_STORAGE_BITS 128
#define F128_K             15      /* exponent bits */
#define F128_P             113     /* precision bits (incl. leading) */
#define F128_BIAS          16383
#define F128_EMAX          16383
#define F128_TRAIL_BITS    112

#define F256_STORAGE_BITS 256
#define F256_K             19
#define F256_P             237
#define F256_BIAS          262143
#define F256_EMAX          262143
#define F256_TRAIL_BITS    236

#define EXT_PREC_MULT 8   /* extended precision = MULT * target_p */

/* -------------------------------------------------------------------------
 * PCG-XSH-RR (fixed seed). Public-domain reference implementation.
 * ----------------------------------------------------------------------- */
static uint64_t pcg_state = 0x853c49e6748fea9bULL;
static uint64_t pcg_inc   = 0xda3e39cb94b95bdbULL;

static uint32_t pcg32(void) {
    uint64_t oldstate = pcg_state;
    pcg_state = oldstate * 6364136223846793005ULL + (pcg_inc | 1ULL);
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

static uint64_t pcg64(void) {
    uint64_t hi = (uint64_t)pcg32();
    uint64_t lo = (uint64_t)pcg32();
    return (hi << 32) | lo;
}

/* -------------------------------------------------------------------------
 * binary128 wire type (trailing is 112 bits → __uint128_t)
 * ----------------------------------------------------------------------- */
typedef struct {
    bool        is_nan;
    bool        is_inf;
    bool        is_zero;
    int         sign;
    uint64_t    exponent;       /* biased 0..(2*F128_EMAX+1) = 32767 */
    __uint128_t trailing;       /* F128_TRAIL_BITS = 112 */
} f128_bits;

static void f128_decode(f128_bits *out, uint64_t hi, uint64_t lo) {
    uint64_t sign   = (hi >> 63) & 1ULL;
    uint64_t biased = (hi >> 48) & 0x7FFFULL;
    __uint128_t tra_hi = (__uint128_t)(hi & 0xFFFFFFFFFFFFULL) << 64;
    __uint128_t tra_lo = (__uint128_t)lo;
    out->sign     = (int)sign;
    out->exponent = biased;
    out->trailing = tra_hi | tra_lo;
    out->trailing &= (((__uint128_t)1 << F128_TRAIL_BITS) - 1);
    out->is_nan  = (biased == 0x7FFF) && (out->trailing != 0);
    out->is_inf  = (biased == 0x7FFF) && (out->trailing == 0);
    out->is_zero = (biased == 0) && (out->trailing == 0);
}

static void f128_encode(uint64_t *hi, uint64_t *lo, const f128_bits *v) {
    __uint128_t t = v->trailing & (((__uint128_t)1 << F128_TRAIL_BITS) - 1);
    uint64_t hi_word = ((uint64_t)v->sign & 1ULL) << 63;
    hi_word |= (v->exponent & 0x7FFFULL) << 48;
    hi_word |= (uint64_t)(t >> 64);
    *hi = hi_word;
    *lo = (uint64_t)t;
}

/* -------------------------------------------------------------------------
 * binary256 wire type (trailing is 236 bits across 4 limbs)
 * ----------------------------------------------------------------------- */
typedef struct {
    bool     is_nan;
    bool     is_inf;
    bool     is_zero;
    int      sign;
    uint64_t exponent;                /* biased 0..(2*F256_EMAX+1) = 524287 */
    /* trailing field is F256_TRAIL_BITS = 236 bits; stored across 4 limbs,
     * little-endian; bits 192..235 (44 bits) go in limb 3 low. */
    uint64_t tra_limbs[4];
} f256_bits;

static void f256_decode(f256_bits *out,
                        uint64_t l0, uint64_t l1, uint64_t l2, uint64_t l3) {
    uint64_t sign   = (l3 >> 63) & 1ULL;
    uint64_t biased = (l3 >> 44) & 0x7FFFFULL;             /* 19 bits */
    uint64_t t_hi   = l3 & 0x000FFFFFFFFFFFFFULL;           /* 44 bits */
    out->sign       = (int)sign;
    out->exponent   = biased;
    out->tra_limbs[0] = l0;
    out->tra_limbs[1] = l1;
    out->tra_limbs[2] = l2;
    out->tra_limbs[3] = t_hi;
    bool nz = (l0 | l1 | l2 | t_hi) != 0;
    out->is_nan = (biased == 0x7FFFF) && nz;
    out->is_inf = (biased == 0x7FFFF) && !nz;
    out->is_zero= (biased == 0) && !nz;
}

static void f256_encode(uint64_t *l0, uint64_t *l1, uint64_t *l2, uint64_t *l3,
                        const f256_bits *v) {
    *l0 = v->tra_limbs[0];
    *l1 = v->tra_limbs[1];
    *l2 = v->tra_limbs[2];
    uint64_t hi = ((uint64_t)v->sign & 1ULL) << 63;
    hi |= (v->exponent & 0x7FFFFULL) << 44;
    hi |= v->tra_limbs[3] & 0x000FFFFFFFFFFFFFULL;
    *l3 = hi;
}

/* -------------------------------------------------------------------------
 * Generic MPFR → IEEE binaryN round-to-nearest-even with gradual underflow.
 *
 * Operates on an mpfr_t with EXT_PREC_MULT*target_p bits of precision,
 * i.e. an "exact" representation of the operation result.
 *
 * On exit:
 *   - NaN/Inf/Zero handled (sign preserved, qNaN with deterministic
 *     payload generated for "compute produced NaN" cases)
 *   - Normal range: trailing rounded to (P-1) bits with RNE; if
 *     rounding up overflows the trailing, exponent bumps by 1 (may
 *     overflow to Inf)
 *   - Subnormal range: gradual underflow, RNE with sticky bit, with
 *     the documented IEEE behavior that rounding up out of the
 *     subnormal range lands on the smallest normal
 * ----------------------------------------------------------------------- */

/* Pack a trailing-field mpz_t into 4 little-endian uint64 limbs.
 * Bits beyond the (p-1)-bit trailing field are ignored. */
static inline void pack_trailing(int p, mpz_srcptr z, uint64_t out_limbs[4]) {
    size_t nbits = mpz_sizeinbase(z, 2);
    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    for (size_t b = 0; b < nbits && b < (size_t)(p - 1); b++) {
        if (mpz_tstbit(z, (mp_bitcnt_t)b)) {
            uint64_t bit = (uint64_t)1 << (b & 63);
            switch (b >> 6) {
                case 0: t0 |= bit; break;
                case 1: t1 |= bit; break;
                case 2: t2 |= bit; break;
                case 3: t3 |= bit; break;
            }
        }
    }
    out_limbs[0] = t0;
    out_limbs[1] = t1;
    out_limbs[2] = t2;
    out_limbs[3] = t3;
}

static void round_to_binaryN(int p, int bias, int emax,
                             int sign,
                             bool *out_nan, bool *out_inf, bool *out_zero,
                             int *out_sign,
                             uint64_t *out_exp,
                             /* trailing P-1 bits; we accept up to 4 limbs */
                             uint64_t out_tra_limbs[4],
                             const mpfr_t exact)
{
    *out_nan = *out_inf = *out_zero = false;
    *out_sign = sign;
    out_tra_limbs[0] = out_tra_limbs[1] = out_tra_limbs[2] = out_tra_limbs[3] = 0;

    if (mpfr_nan_p(exact)) {
        *out_nan = true;
        *out_exp = (uint64_t)(2 * emax + 1);
        /* qNaN payload: set the top bit of the trailing field. The trailing
         * field occupies bits 0..(p-2) of the wire, so the top bit is at
         * absolute position (p-2). Place it in the right 64-bit limb. */
        size_t top = (size_t)(p - 2);
        out_tra_limbs[top >> 6] = (uint64_t)1 << (top & 63);
        return;
    }
    if (mpfr_inf_p(exact)) {
        *out_inf = true;
        *out_exp = (uint64_t)(2 * emax + 1);
        return;
    }
    if (mpfr_zero_p(exact)) {
        *out_zero = true;
        return;
    }

    /* Get exact integer significand and binary exponent. */
    mpz_t sig;
    mpz_init(sig);
    mpfr_exp_t e = mpfr_get_z_2exp(sig, exact);
    size_t sig_bits = mpz_sizeinbase(sig, 2);
    int64_t E_unbiased = (int64_t)sig_bits - 1 + (int64_t)e;

    /* Helper: round sig to `target_bits` bits with RNE, write to `rounded`.
     * If after rounding `rounded` has `target_bits + 1` bits (i.e., = 2^target_bits),
     * returns true; otherwise false. */
    #define ROUND_SIG_TO(target_bits, rounded) do {                              \
        if (sig_bits <= (size_t)(target_bits)) {                                \
            mpz_set(rounded, sig);                                              \
        } else {                                                                \
            size_t rp = sig_bits - (target_bits);                               \
            uint64_t rb = mpz_tstbit(sig, (mp_bitcnt_t)rp) ? 1 : 0;             \
            mpz_t bl; mpz_init(bl);                                             \
            mpz_fdiv_r_2exp(bl, sig, (mp_bitcnt_t)rp);                          \
            uint64_t st = (mpz_sgn(bl) != 0) ? 1 : 0;                           \
            mpz_fdiv_q_2exp(rounded, sig, (mp_bitcnt_t)rp);                     \
            uint64_t tb = mpz_tstbit(rounded, 0);                               \
            if (rb && (st || tb)) mpz_add_ui(rounded, rounded, 1);              \
            mpz_clear(bl);                                                      \
        }                                                                       \
    } while (0)

    if (E_unbiased >= 1 - bias) {
        /* === NORMAL range (or overflow) === */
        mpz_t rounded; mpz_init(rounded);
        ROUND_SIG_TO(p, rounded);
        /* If rounding up overflowed to 2^p, bump exponent and divide. */
        int64_t new_E = E_unbiased;
        if (mpz_sizeinbase(rounded, 2) > (size_t)p) {
            mpz_fdiv_q_2exp(rounded, rounded, 1);
            new_E += 1;
        }
        if (new_E - bias >= (int64_t)(2 * emax + 1)) {
            *out_inf = true;
            *out_exp = (uint64_t)(2 * emax + 1);
            mpz_clear(rounded); mpz_clear(sig);
            return;
        }
        /* Extract trailing = rounded - 2^(p-1). Compute 2^(p-1) in an mpz
         * to avoid undefined shift behavior on systems where mp_bitcnt_t is
         * narrower than (p-1). */
        mpz_t two_pow; mpz_init(two_pow);
        mpz_setbit(two_pow, (mp_bitcnt_t)(p - 1));
        mpz_t trailing; mpz_init(trailing);
        mpz_sub(trailing, rounded, two_pow);
        mpz_clear(two_pow);
        *out_exp = (uint64_t)(new_E + bias);
        pack_trailing(p, trailing, out_tra_limbs);
        mpz_clear(trailing); mpz_clear(rounded); mpz_clear(sig);
        return;
    }

    /* === SUBNORMAL range === */
    mpz_t rounded; mpz_init(rounded);
    ROUND_SIG_TO(p - 1, rounded);
    /* If rounding up overflowed to 2^(p-1), we landed on the smallest
     * normal (trailing = 0, E_biased = 1). Otherwise subnormal. */
    if (mpz_sizeinbase(rounded, 2) == (size_t)p) {
        *out_exp = 1;
        /* trailing stays all-zero */
    } else {
        *out_exp = 0;
        pack_trailing(p, rounded, out_tra_limbs);
    }
    mpz_clear(rounded); mpz_clear(sig);
    #undef ROUND_SIG_TO
}

/* -------------------------------------------------------------------------
 * Convert a binary128 wire value → mpfr_t at extended precision.
 * ----------------------------------------------------------------------- */
static void f128_to_mpfr(mpfr_t x, const f128_bits *v) {
    if (v->is_nan) { mpfr_set_nan(x); return; }
    if (v->is_inf) { mpfr_set_inf(x, v->sign ? -1 : 1); return; }
    if (v->is_zero){ mpfr_set_zero(x, v->sign ? -1 : 1); return; }
    /* Build mpz = trailing + 2^(P-1). */
    mpz_t sig;
    mpz_init(sig);
    /* Copy the 16 bytes of __uint128_t into a local buffer so mpz_import
     * reads native bytes. On little-endian hosts the bytes are stored
     * least-significant first; "order = -1, endian = -1" tells mpz_import
     * to use native byte order, so it reconstructs the integer correctly. */
    unsigned char buf[16];
    memcpy(buf, &v->trailing, 16);
    mpz_import(sig, 2, /*order=*/-1, /*size=*/8, /*endian=*/-1, /*nails=*/0, buf);
    /* Mask to the 112-bit trailing field. */
    mpz_fdiv_r_2exp(sig, sig, F128_TRAIL_BITS);
    /* Insert the implicit leading 1 at bit P-1 = 112. */
    mpz_setbit(sig, F128_P - 1);
    /* Numerical value = sig * 2^E_unbiased = (1 + trailing/2^112) * 2^E_unbiased.
     * So mpfr_set_z_2exp(x, sig, E_unbiased - (P-1)). */
    mpfr_exp_t E_unbiased = (mpfr_exp_t)v->exponent - (mpfr_exp_t)F128_BIAS;
    mpfr_set_z_2exp(x, sig, E_unbiased - (F128_P - 1), MPFR_RNDN);
    if (v->sign) mpfr_neg(x, x, MPFR_RNDN);
    mpz_clear(sig);
}

/* -------------------------------------------------------------------------
 * Round an extended-precision mpfr_t into IEEE binary128.
 * ----------------------------------------------------------------------- */
static void f128_round_from_mpfr(f128_bits *out, const mpfr_t exact) {
    uint64_t limbs[4] = {0};
    round_to_binaryN(F128_P, F128_BIAS, F128_EMAX,
                     mpfr_signbit(exact) ? 1 : 0,
                     &out->is_nan, &out->is_inf, &out->is_zero,
                     &out->sign, &out->exponent, limbs, exact);
    /* The 4 limbs returned are the trailing field packed across 4 little-
     * endian uint64; for binary128 only the bottom 2 limbs are meaningful,
     * and only 112 bits within them. The top 2 limbs are zero since the
     * trailing field of f128 is only 112 bits. */
    __uint128_t t = ((__uint128_t)limbs[1] << 64) | limbs[0];
    t &= (((__uint128_t)1 << F128_TRAIL_BITS) - 1);
    out->trailing = t;
}

/* -------------------------------------------------------------------------
 * Convert a binary256 wire value → mpfr_t at extended precision.
 * ----------------------------------------------------------------------- */
static void f256_to_mpfr(mpfr_t x, const f256_bits *v) {
    if (v->is_nan)  { mpfr_set_nan(x); return; }
    if (v->is_inf)  { mpfr_set_inf(x, v->sign ? -1 : 1); return; }
    if (v->is_zero) { mpfr_set_zero(x, v->sign ? -1 : 1); return; }
    mpz_t sig;
    mpz_init(sig);
    /* tra_limbs[0..3] are stored little-endian: limb 0 holds the lowest
     * 64 bits of the 236-bit trailing field. Use native byte order so
     * mpz_import reconstructs the integer correctly. */
    unsigned char buf[32];
    memcpy(buf, v->tra_limbs, 32);
    mpz_import(sig, 4, /*order=*/-1, /*size=*/8, /*endian=*/-1, /*nails=*/0, buf);
    /* Mask to 236 bits. */
    mpz_fdiv_r_2exp(sig, sig, F256_TRAIL_BITS);
    /* Insert the implicit leading 1 at bit P-1 = 236. */
    mpz_setbit(sig, F256_P - 1);
    /* mpfr_set_z_2exp(x, sig, E_unbiased - (P-1)). */
    mpfr_exp_t E_unbiased = (mpfr_exp_t)v->exponent - (mpfr_exp_t)F256_BIAS;
    mpfr_set_z_2exp(x, sig, E_unbiased - (F256_P - 1), MPFR_RNDN);
    if (v->sign) mpfr_neg(x, x, MPFR_RNDN);
    mpz_clear(sig);
}

/* -------------------------------------------------------------------------
 * Round an extended-precision mpfr_t into IEEE binary256.
 * ----------------------------------------------------------------------- */
static void f256_round_from_mpfr(f256_bits *out, const mpfr_t exact) {
    uint64_t limbs[4] = {0};
    round_to_binaryN(F256_P, F256_BIAS, F256_EMAX,
                     mpfr_signbit(exact) ? 1 : 0,
                     &out->is_nan, &out->is_inf, &out->is_zero,
                     &out->sign, &out->exponent, limbs, exact);
    out->tra_limbs[0] = limbs[0];
    out->tra_limbs[1] = limbs[1];
    out->tra_limbs[2] = limbs[2];
    out->tra_limbs[3] = limbs[3] & 0x000FFFFFFFFFFFFFULL;
}

/* -------------------------------------------------------------------------
 * JSONL output helpers
 * ----------------------------------------------------------------------- */
static const char *class_str(bool is_nan, bool is_inf, bool is_zero,
                             uint64_t exponent) {
    if (is_nan)  return "nan";
    if (is_inf)  return "inf";
    if (is_zero) return "zero";
    if (exponent == 0) return "subnormal";
    return "normal";
}

static void write_f128(FILE *f, const char *field, const f128_bits *v) {
    uint64_t hi, lo;
    f128_encode(&hi, &lo, v);
    fprintf(f, "\"%s\":{", field);
    fprintf(f, "\"class\":\"%s\",", class_str(v->is_nan, v->is_inf, v->is_zero, v->exponent));
    fprintf(f, "\"sign\":%d,", v->sign);
    fprintf(f, "\"exponent\":%" PRIu64 ",", v->exponent);
    fprintf(f, "\"trailing_hex\":\"%016" PRIx64 "%016" PRIx64 "\",", (uint64_t)(v->trailing >> 64), (uint64_t)v->trailing);
    fprintf(f, "\"limbs\":[%" PRIu64 ",%" PRIu64 "]}", lo, hi);
}

static void write_f256(FILE *f, const char *field, const f256_bits *v) {
    uint64_t l0, l1, l2, l3;
    f256_encode(&l0, &l1, &l2, &l3, v);
    fprintf(f, "\"%s\":{", field);
    fprintf(f, "\"class\":\"%s\",", class_str(v->is_nan, v->is_inf, v->is_zero, v->exponent));
    fprintf(f, "\"sign\":%d,", v->sign);
    fprintf(f, "\"exponent\":%" PRIu64 ",", v->exponent);
    fprintf(f, "\"trailing_hex\":\"%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "\",",
            v->tra_limbs[3], v->tra_limbs[2], v->tra_limbs[1], v->tra_limbs[0]);
    fprintf(f, "\"limbs\":[%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "]}", l0, l1, l2, l3);
}

/* -------------------------------------------------------------------------
 * Special operand builders
 * ----------------------------------------------------------------------- */
static void f128_zero(f128_bits *v, int sign) {
    memset(v, 0, sizeof(*v)); v->sign = sign; v->is_zero = true;
}
static void f128_inf(f128_bits *v, int sign) {
    memset(v, 0, sizeof(*v)); v->sign = sign;
    v->exponent = (uint64_t)(2*F128_EMAX + 1);
    v->is_inf = true;
}
static void f128_qnan(f128_bits *v, int sign) {
    memset(v, 0, sizeof(*v)); v->sign = sign;
    v->exponent = (uint64_t)(2*F128_EMAX + 1);
    v->trailing = (__uint128_t)1 << (F128_TRAIL_BITS - 1); /* top bit set */
    v->is_nan = true;
}
static void f128_snan(f128_bits *v, int sign) {
    memset(v, 0, sizeof(*v)); v->sign = sign;
    v->exponent = (uint64_t)(2*F128_EMAX + 1);
    v->trailing = (__uint128_t)1; /* top bit clear → signaling NaN */
    v->is_nan = true;
}
static void f128_max_normal(f128_bits *v, int sign) {
    memset(v, 0, sizeof(*v)); v->sign = sign;
    v->exponent = (uint64_t)(2*F128_EMAX);
    v->trailing = (((__uint128_t)1) << F128_TRAIL_BITS) - 1;
}
static void f128_min_normal(f128_bits *v, int sign) {
    memset(v, 0, sizeof(*v)); v->sign = sign;
    v->exponent = 1;
    v->trailing = 0;
}
static void f128_max_subnormal(f128_bits *v, int sign) {
    memset(v, 0, sizeof(*v)); v->sign = sign;
    v->trailing = (((__uint128_t)1) << F128_TRAIL_BITS) - 1;
}
static void f128_min_subnormal(f128_bits *v, int sign) {
    memset(v, 0, sizeof(*v)); v->sign = sign;
    v->trailing = 1;
}

static void f256_zero(f256_bits *v, int sign) {
    memset(v, 0, sizeof(*v)); v->sign = sign; v->is_zero = true;
}
static void f256_inf(f256_bits *v, int sign) {
    memset(v, 0, sizeof(*v)); v->sign = sign;
    v->exponent = (uint64_t)(2*F256_EMAX + 1);
    v->is_inf = true;
}
static void f256_qnan(f256_bits *v, int sign) {
    memset(v, 0, sizeof(*v)); v->sign = sign;
    v->exponent = (uint64_t)(2*F256_EMAX + 1);
    v->tra_limbs[0] = (uint64_t)1 << 63; /* top bit of low 64 bits of trailing */
    v->is_nan = true;
}
static void f256_snan(f256_bits *v, int sign) {
    memset(v, 0, sizeof(*v)); v->sign = sign;
    v->exponent = (uint64_t)(2*F256_EMAX + 1);
    v->tra_limbs[0] = 1;
    v->is_nan = true;
}
static void f256_max_normal(f256_bits *v, int sign) {
    memset(v, 0, sizeof(*v)); v->sign = sign;
    v->exponent = (uint64_t)(2*F256_EMAX);
    v->tra_limbs[0] = 0xFFFFFFFFFFFFFFFFULL;
    v->tra_limbs[1] = 0xFFFFFFFFFFFFFFFFULL;
    v->tra_limbs[2] = 0xFFFFFFFFFFFFFFFFULL;
    v->tra_limbs[3] = 0x000FFFFFFFFFFFFFULL;
}
static void f256_min_normal(f256_bits *v, int sign) {
    memset(v, 0, sizeof(*v)); v->sign = sign;
    v->exponent = 1;
}
static void f256_max_subnormal(f256_bits *v, int sign) {
    memset(v, 0, sizeof(*v)); v->sign = sign;
    v->tra_limbs[0] = 0xFFFFFFFFFFFFFFFFULL;
    v->tra_limbs[1] = 0xFFFFFFFFFFFFFFFFULL;
    v->tra_limbs[2] = 0xFFFFFFFFFFFFFFFFULL;
    v->tra_limbs[3] = 0x000FFFFFFFFFFFFFULL;
}
static void f256_min_subnormal(f256_bits *v, int sign) {
    memset(v, 0, sizeof(*v)); v->sign = sign;
    v->tra_limbs[0] = 1;
}

/* -------------------------------------------------------------------------
 * Random operand generators (deterministic via PCG)
 * ----------------------------------------------------------------------- */
static void random_f128_normal(f128_bits *v) {
    v->sign = (int)(pcg32() & 1);
    v->exponent = 1 + (pcg64() % (uint64_t)(2 * F128_EMAX));
    v->trailing = ((__uint128_t)pcg64() << 64) | pcg64();
    v->trailing &= (((__uint128_t)1) << F128_TRAIL_BITS) - 1;
    v->is_nan = v->is_inf = v->is_zero = false;
}
static void random_f128_subnormal(f128_bits *v) {
    v->sign = (int)(pcg32() & 1);
    v->exponent = 0;
    v->trailing = ((__uint128_t)pcg64() << 64) | pcg64();
    v->trailing &= (((__uint128_t)1) << F128_TRAIL_BITS) - 1;
    v->is_nan = v->is_inf = v->is_zero = false;
}
static void random_f256_normal(f256_bits *v) {
    v->sign = (int)(pcg32() & 1);
    v->exponent = 1 + (pcg64() % (uint64_t)(2 * F256_EMAX));
    v->tra_limbs[0] = pcg64();
    v->tra_limbs[1] = pcg64();
    v->tra_limbs[2] = pcg64();
    v->tra_limbs[3] = pcg64() & 0x000FFFFFFFFFFFFFULL;
    v->is_nan = v->is_inf = v->is_zero = false;
}
static void random_f256_subnormal(f256_bits *v) {
    v->sign = (int)(pcg32() & 1);
    v->exponent = 0;
    v->tra_limbs[0] = pcg64();
    v->tra_limbs[1] = pcg64();
    v->tra_limbs[2] = pcg64();
    v->tra_limbs[3] = pcg64() & 0x000FFFFFFFFFFFFFULL;
    v->is_nan = v->is_inf = v->is_zero = false;
}

/* -------------------------------------------------------------------------
 * Cmp predicates
 * ----------------------------------------------------------------------- */
typedef enum { CMP_EQ, CMP_NE, CMP_LT, CMP_LE, CMP_GT, CMP_GE, CMP_UNORD } cmp_kind;
static const char *cmp_name(cmp_kind k) {
    switch (k) {
        case CMP_EQ: return "eq";
        case CMP_NE: return "ne";
        case CMP_LT: return "lt";
        case CMP_LE: return "le";
        case CMP_GT: return "gt";
        case CMP_GE: return "ge";
        case CMP_UNORD: return "unord";
    }
    return "?";
}

/* ordered comparison: NaN input → false for EQ/NE/LT/LE/GT/GE; UNORD is the inverse. */
static bool cmp_ordered(mpfr_t a, mpfr_t b, cmp_kind k) {
    bool an = mpfr_nan_p(a), bn = mpfr_nan_p(b);
    if (k == CMP_UNORD) return an || bn;
    if (an || bn) return false;
    int sgn = mpfr_cmp(a, b);
    switch (k) {
        case CMP_EQ: return sgn == 0;
        case CMP_NE: return sgn != 0;
        case CMP_LT: return sgn < 0;
        case CMP_LE: return sgn <= 0;
        case CMP_GT: return sgn > 0;
        case CMP_GE: return sgn >= 0;
        default: return false;
    }
}

/* -------------------------------------------------------------------------
 * Operation dispatchers (binary128)
 * ----------------------------------------------------------------------- */
static void do_f128_binary(FILE *f, const char *op, const f128_bits *a, const f128_bits *b) {
    mpfr_t x, y, r;
    mpfr_inits2(EXT_PREC_MULT * F128_P, x, y, r, (mpfr_ptr)0);
    f128_to_mpfr(x, a);
    f128_to_mpfr(y, b);

    if (!strcmp(op, "add"))      mpfr_add(r, x, y, MPFR_RNDN);
    else if (!strcmp(op, "sub")) mpfr_sub(r, x, y, MPFR_RNDN);
    else if (!strcmp(op, "mul")) {
        /* 0*Inf = NaN per IEEE */
        if ((mpfr_zero_p(x) && mpfr_inf_p(y)) ||
            (mpfr_inf_p(x) && mpfr_zero_p(y)))
            mpfr_set_nan(r);
        else mpfr_mul(r, x, y, MPFR_RNDN);
    }
    else if (!strcmp(op, "div")) {
        if ((mpfr_zero_p(x) && mpfr_zero_p(y)) ||
            (mpfr_inf_p(x) && mpfr_inf_p(y)))
            mpfr_set_nan(r);
        else mpfr_div(r, x, y, MPFR_RNDN);
    }

    f128_bits rv;
    f128_round_from_mpfr(&rv, r);

    fprintf(f, "{\"op\":\"f128_%s\",", op);
    write_f128(f, "a", a); fputc(',', f);
    write_f128(f, "b", b); fputc(',', f);
    write_f128(f, "result", &rv);
    fprintf(f, ",\"rounding\":\"rne\"}\n");

    mpfr_clears(x, y, r, (mpfr_ptr)0);
}

static void do_f128_cmp(FILE *f, cmp_kind k, const f128_bits *a, const f128_bits *b) {
    mpfr_t x, y;
    mpfr_inits2(EXT_PREC_MULT * F128_P, x, y, (mpfr_ptr)0);
    f128_to_mpfr(x, a);
    f128_to_mpfr(y, b);
    bool result = cmp_ordered(x, y, k);
    fprintf(f, "{\"op\":\"f128_cmp_%s\",", cmp_name(k));
    write_f128(f, "a", a); fputc(',', f);
    write_f128(f, "b", b); fputc(',', f);
    fprintf(f, "\"result\":%s}\n", result ? "true" : "false");
    mpfr_clears(x, y, (mpfr_ptr)0);
}

/* -------------------------------------------------------------------------
 * Operation dispatchers (binary256)
 * ----------------------------------------------------------------------- */
static void do_f256_binary(FILE *f, const char *op, const f256_bits *a, const f256_bits *b) {
    mpfr_t x, y, r;
    mpfr_inits2(EXT_PREC_MULT * F256_P, x, y, r, (mpfr_ptr)0);
    f256_to_mpfr(x, a);
    f256_to_mpfr(y, b);

    if (mpfr_nan_p(x) || mpfr_nan_p(y)) {
        mpfr_set_nan(r);
    } else if (!strcmp(op, "add"))      mpfr_add(r, x, y, MPFR_RNDN);
    else if (!strcmp(op, "sub"))        mpfr_sub(r, x, y, MPFR_RNDN);
    else if (!strcmp(op, "mul")) {
        if ((mpfr_zero_p(x) && mpfr_inf_p(y)) ||
            (mpfr_inf_p(x) && mpfr_zero_p(y)))
            mpfr_set_nan(r);
        else mpfr_mul(r, x, y, MPFR_RNDN);
    }
    else if (!strcmp(op, "div")) {
        if ((mpfr_zero_p(x) && mpfr_zero_p(y)) ||
            (mpfr_inf_p(x) && mpfr_inf_p(y)))
            mpfr_set_nan(r);
        else mpfr_div(r, x, y, MPFR_RNDN);
    }

    f256_bits rv;
    f256_round_from_mpfr(&rv, r);

    fprintf(f, "{\"op\":\"f256_%s\",", op);
    write_f256(f, "a", a); fputc(',', f);
    write_f256(f, "b", b); fputc(',', f);
    write_f256(f, "result", &rv);
    fprintf(f, ",\"rounding\":\"rne\"}\n");

    mpfr_clears(x, y, r, (mpfr_ptr)0);
}

static void do_f256_cmp(FILE *f, cmp_kind k, const f256_bits *a, const f256_bits *b) {
    mpfr_t x, y;
    mpfr_inits2(EXT_PREC_MULT * F256_P, x, y, (mpfr_ptr)0);
    f256_to_mpfr(x, a);
    f256_to_mpfr(y, b);
    bool result = cmp_ordered(x, y, k);
    fprintf(f, "{\"op\":\"f256_cmp_%s\",", cmp_name(k));
    write_f256(f, "a", a); fputc(',', f);
    write_f256(f, "b", b); fputc(',', f);
    fprintf(f, "\"result\":%s}\n", result ? "true" : "false");
    mpfr_clears(x, y, (mpfr_ptr)0);
}

/* -------------------------------------------------------------------------
 * Test drivers
 * ----------------------------------------------------------------------- */
static void gen_f128_corpus(FILE *f) {
    f128_bits a, b;

    const char *binops[] = {"add", "sub", "mul", "div"};
    for (size_t i = 0; i < sizeof(binops)/sizeof(binops[0]); i++) {
        f128_zero(&a, 0); f128_zero(&b, 1);  do_f128_binary(f, binops[i], &a, &b);
        f128_zero(&a, 1); f128_zero(&b, 0);  do_f128_binary(f, binops[i], &a, &b);
        f128_inf(&a, 0);  f128_inf(&b, 1);   do_f128_binary(f, binops[i], &a, &b);
        f128_inf(&a, 0);  f128_inf(&b, 0);   do_f128_binary(f, binops[i], &a, &b);
        f128_zero(&a, 0); f128_inf(&b, 0);   do_f128_binary(f, binops[i], &a, &b);
        f128_zero(&a, 0); f128_zero(&b, 0);  do_f128_binary(f, binops[i], &a, &b);
        f128_inf(&a, 0);  f128_inf(&b, 0);   do_f128_binary(f, binops[i], &a, &b);
        f128_min_normal(&a, 0); f128_zero(&b, 0); do_f128_binary(f, binops[i], &a, &b);
        f128_min_normal(&a, 0); f128_zero(&b, 1); do_f128_binary(f, binops[i], &a, &b);
    }

    f128_max_normal(&a, 0); f128_max_normal(&b, 0);
    do_f128_binary(f, "add", &a, &b); do_f128_binary(f, "mul", &a, &b);
    f128_max_normal(&a, 0); f128_min_normal(&b, 0);
    do_f128_binary(f, "add", &a, &b); do_f128_binary(f, "sub", &a, &b);
    do_f128_binary(f, "mul", &a, &b); do_f128_binary(f, "div", &a, &b);

    f128_max_subnormal(&a, 0); f128_max_subnormal(&b, 0);
    do_f128_binary(f, "add", &a, &b); do_f128_binary(f, "mul", &a, &b);
    f128_max_subnormal(&a, 0); f128_min_subnormal(&b, 0);
    do_f128_binary(f, "add", &a, &b); do_f128_binary(f, "sub", &a, &b);
    do_f128_binary(f, "mul", &a, &b);
    f128_min_subnormal(&a, 0); f128_min_subnormal(&b, 0);
    do_f128_binary(f, "add", &a, &b); do_f128_binary(f, "mul", &a, &b);

    /* Tie-to-even tests: 1.0 + ulp/2 = 1 (RNE picks even); 1+ulp + ulp/2 = 1+ulp. */
    {
        f128_bits one; memset(&one, 0, sizeof(one));
        one.exponent = F128_BIAS;
        f128_bits half_ulp; memset(&half_ulp, 0, sizeof(half_ulp));
        half_ulp.exponent = F128_BIAS - F128_P;        /* 2^-113 */
        half_ulp.trailing = 1;
        f128_bits ulp; memset(&ulp, 0, sizeof(ulp));
        ulp.exponent = F128_BIAS - (F128_P - 1);       /* 2^-112 */
        ulp.trailing = 0;
        f128_bits two_ulps; memset(&two_ulps, 0, sizeof(two_ulps));
        two_ulps.exponent = F128_BIAS - (F128_P - 2);  /* 2^-111 */
        two_ulps.trailing = 0;
        f128_bits three_ulps; memset(&three_ulps, 0, sizeof(three_ulps));
        three_ulps.exponent = F128_BIAS - (F128_P - 2);
        three_ulps.trailing = 1;

        do_f128_binary(f, "add", &one, &half_ulp);     /* 1 + 1ulp/2 → 1 (RNE even) */
        do_f128_binary(f, "add", &one, &ulp);          /* 1 + 1ulp → 1+ulp */
        do_f128_binary(f, "add", &ulp, &half_ulp);     /* 1ulp + ulp/2 → 1ulp (RNE even) */
        do_f128_binary(f, "add", &two_ulps, &half_ulp);/* 2ulp + ulp/2 → 2ulp (RNE even) */
        do_f128_binary(f, "add", &three_ulps, &half_ulp);/* 3ulp + ulp/2 → 3ulp (RNE even) */
        do_f128_binary(f, "add", &ulp, &ulp);          /* 1ulp + 1ulp → 2ulp */
    }

    for (int i = 0; i < 256; i++) {
        random_f128_normal(&a); random_f128_normal(&b);
        do_f128_binary(f, "add", &a, &b);
        do_f128_binary(f, "sub", &a, &b);
        do_f128_binary(f, "mul", &a, &b);
        do_f128_binary(f, "div", &a, &b);
    }
    for (int i = 0; i < 128; i++) {
        random_f128_subnormal(&a); random_f128_subnormal(&b);
        do_f128_binary(f, "add", &a, &b);
        do_f128_binary(f, "sub", &a, &b);
        do_f128_binary(f, "mul", &a, &b);
        do_f128_binary(f, "div", &a, &b);
    }
    for (int i = 0; i < 128; i++) {
        random_f128_normal(&a); random_f128_subnormal(&b);
        do_f128_binary(f, "add", &a, &b);
        do_f128_binary(f, "sub", &a, &b);
        do_f128_binary(f, "mul", &a, &b);
        do_f128_binary(f, "div", &a, &b);
    }

    cmp_kind kinds[] = {CMP_EQ, CMP_NE, CMP_LT, CMP_LE, CMP_GT, CMP_GE, CMP_UNORD};
    for (size_t k = 0; k < sizeof(kinds)/sizeof(kinds[0]); k++) {
        f128_zero(&a, 0); f128_zero(&b, 0);  do_f128_cmp(f, kinds[k], &a, &b);
        f128_zero(&a, 0); f128_zero(&b, 1);  do_f128_cmp(f, kinds[k], &a, &b);
        f128_inf(&a, 0);  f128_inf(&b, 0);   do_f128_cmp(f, kinds[k], &a, &b);
        f128_inf(&a, 0);  f128_inf(&b, 1);   do_f128_cmp(f, kinds[k], &a, &b);
        f128_inf(&a, 0);  f128_zero(&b, 0);  do_f128_cmp(f, kinds[k], &a, &b);
        f128_qnan(&a, 0); f128_zero(&b, 0);  do_f128_cmp(f, kinds[k], &a, &b);
        f128_qnan(&a, 0); f128_qnan(&b, 1);  do_f128_cmp(f, kinds[k], &a, &b);
        f128_max_normal(&a, 0); f128_max_normal(&b, 0); do_f128_cmp(f, kinds[k], &a, &b);
        f128_max_normal(&a, 0); f128_max_normal(&b, 1); do_f128_cmp(f, kinds[k], &a, &b);
        f128_min_subnormal(&a, 0); f128_max_subnormal(&b, 0); do_f128_cmp(f, kinds[k], &a, &b);
    }
    for (int i = 0; i < 256; i++) {
        random_f128_normal(&a); random_f128_normal(&b);
        for (size_t k = 0; k < sizeof(kinds)/sizeof(kinds[0]); k++)
            do_f128_cmp(f, kinds[k], &a, &b);
    }
    for (int i = 0; i < 64; i++) {
        f128_qnan(&a, 0); random_f128_normal(&b);
        for (size_t k = 0; k < sizeof(kinds)/sizeof(kinds[0]); k++)
            do_f128_cmp(f, kinds[k], &a, &b);
    }
    /* sNaN propagation: ensure sNaN inputs yield qNaN. */
    {
        f128_bits snan; f128_snan(&snan, 0);
        f128_bits one; memset(&one, 0, sizeof(one));
        one.exponent = F128_BIAS;
        do_f128_binary(f, "add", &snan, &one);
    }
}

static void gen_f256_corpus(FILE *f) {
    f256_bits a, b;

    const char *binops[] = {"add", "sub", "mul", "div"};
    for (size_t i = 0; i < sizeof(binops)/sizeof(binops[0]); i++) {
        f256_zero(&a, 0); f256_zero(&b, 1);  do_f256_binary(f, binops[i], &a, &b);
        f256_zero(&a, 1); f256_zero(&b, 0);  do_f256_binary(f, binops[i], &a, &b);
        f256_inf(&a, 0);  f256_inf(&b, 1);   do_f256_binary(f, binops[i], &a, &b);
        f256_inf(&a, 0);  f256_inf(&b, 0);   do_f256_binary(f, binops[i], &a, &b);
        f256_zero(&a, 0); f256_inf(&b, 0);   do_f256_binary(f, binops[i], &a, &b);
        f256_zero(&a, 0); f256_zero(&b, 0);  do_f256_binary(f, binops[i], &a, &b);
        f256_inf(&a, 0);  f256_inf(&b, 0);   do_f256_binary(f, binops[i], &a, &b);
        f256_min_normal(&a, 0); f256_zero(&b, 0); do_f256_binary(f, binops[i], &a, &b);
        f256_min_normal(&a, 0); f256_zero(&b, 1); do_f256_binary(f, binops[i], &a, &b);
    }

    f256_max_normal(&a, 0); f256_max_normal(&b, 0);
    do_f256_binary(f, "add", &a, &b); do_f256_binary(f, "mul", &a, &b);
    f256_max_normal(&a, 0); f256_min_normal(&b, 0);
    do_f256_binary(f, "add", &a, &b); do_f256_binary(f, "sub", &a, &b);
    do_f256_binary(f, "mul", &a, &b); do_f256_binary(f, "div", &a, &b);

    f256_max_subnormal(&a, 0); f256_max_subnormal(&b, 0);
    do_f256_binary(f, "add", &a, &b); do_f256_binary(f, "mul", &a, &b);
    f256_max_subnormal(&a, 0); f256_min_subnormal(&b, 0);
    do_f256_binary(f, "add", &a, &b); do_f256_binary(f, "sub", &a, &b);
    do_f256_binary(f, "mul", &a, &b);
    f256_min_subnormal(&a, 0); f256_min_subnormal(&b, 0);
    do_f256_binary(f, "add", &a, &b); do_f256_binary(f, "mul", &a, &b);

    /* Tie-to-even test at f256 (ulp at 1.0 = 2^-236). */
    {
        f256_bits one; memset(&one, 0, sizeof(one));
        one.exponent = F256_BIAS;
        f256_bits half_ulp; memset(&half_ulp, 0, sizeof(half_ulp));
        half_ulp.exponent = F256_BIAS - F256_P;
        half_ulp.tra_limbs[0] = 1;
        f256_bits ulp; memset(&ulp, 0, sizeof(ulp));
        ulp.exponent = F256_BIAS - (F256_P - 1);
        do_f256_binary(f, "add", &one, &half_ulp);
        do_f256_binary(f, "add", &one, &ulp);
        do_f256_binary(f, "add", &ulp, &half_ulp);
    }

    for (int i = 0; i < 256; i++) {
        random_f256_normal(&a); random_f256_normal(&b);
        do_f256_binary(f, "add", &a, &b);
        do_f256_binary(f, "sub", &a, &b);
        do_f256_binary(f, "mul", &a, &b);
        do_f256_binary(f, "div", &a, &b);
    }
    for (int i = 0; i < 128; i++) {
        random_f256_subnormal(&a); random_f256_subnormal(&b);
        do_f256_binary(f, "add", &a, &b);
        do_f256_binary(f, "sub", &a, &b);
        do_f256_binary(f, "mul", &a, &b);
        do_f256_binary(f, "div", &a, &b);
    }
    for (int i = 0; i < 128; i++) {
        random_f256_normal(&a); random_f256_subnormal(&b);
        do_f256_binary(f, "add", &a, &b);
        do_f256_binary(f, "sub", &a, &b);
        do_f256_binary(f, "mul", &a, &b);
        do_f256_binary(f, "div", &a, &b);
    }

    cmp_kind kinds[] = {CMP_EQ, CMP_NE, CMP_LT, CMP_LE, CMP_GT, CMP_GE, CMP_UNORD};
    for (size_t k = 0; k < sizeof(kinds)/sizeof(kinds[0]); k++) {
        f256_zero(&a, 0); f256_zero(&b, 0);  do_f256_cmp(f, kinds[k], &a, &b);
        f256_zero(&a, 0); f256_zero(&b, 1);  do_f256_cmp(f, kinds[k], &a, &b);
        f256_inf(&a, 0);  f256_inf(&b, 0);   do_f256_cmp(f, kinds[k], &a, &b);
        f256_inf(&a, 0);  f256_inf(&b, 1);   do_f256_cmp(f, kinds[k], &a, &b);
        f256_inf(&a, 0);  f256_zero(&b, 0);  do_f256_cmp(f, kinds[k], &a, &b);
        f256_qnan(&a, 0); f256_zero(&b, 0);  do_f256_cmp(f, kinds[k], &a, &b);
        f256_qnan(&a, 0); f256_qnan(&b, 1);  do_f256_cmp(f, kinds[k], &a, &b);
        f256_max_normal(&a, 0); f256_max_normal(&b, 0); do_f256_cmp(f, kinds[k], &a, &b);
        f256_max_normal(&a, 0); f256_max_normal(&b, 1); do_f256_cmp(f, kinds[k], &a, &b);
        f256_min_subnormal(&a, 0); f256_max_subnormal(&b, 0); do_f256_cmp(f, kinds[k], &a, &b);
    }
    for (int i = 0; i < 256; i++) {
        random_f256_normal(&a); random_f256_normal(&b);
        for (size_t k = 0; k < sizeof(kinds)/sizeof(kinds[0]); k++)
            do_f256_cmp(f, kinds[k], &a, &b);
    }
    for (int i = 0; i < 64; i++) {
        f256_qnan(&a, 0); random_f256_normal(&b);
        for (size_t k = 0; k < sizeof(kinds)/sizeof(kinds[0]); k++)
            do_f256_cmp(f, kinds[k], &a, &b);
    }
    {
        f256_bits snan; f256_snan(&snan, 0);
        f256_bits one; memset(&one, 0, sizeof(one));
        one.exponent = F256_BIAS;
        do_f256_binary(f, "add", &snan, &one);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <f128|f256|both> <output.jsonl>\n", argv[0]);
        return 2;
    }
    /* Set MPFR exponent range to the widest available so that operations
     * on tiny / huge values don't trap. */
    mpfr_set_emin(MPFR_EMIN_MIN);
    mpfr_set_emax(MPFR_EMAX_MAX);

    FILE *f = strcmp(argv[2], "-") ? fopen(argv[2], "w") : stdout;
    if (!f) { perror(argv[2]); return 1; }

    if (!strcmp(argv[1], "f128") || !strcmp(argv[1], "both"))
        gen_f128_corpus(f);
    if (!strcmp(argv[1], "f256") || !strcmp(argv[1], "both"))
        gen_f256_corpus(f);

    if (f != stdout) fclose(f);
    mpfr_free_cache();
    return 0;
}
