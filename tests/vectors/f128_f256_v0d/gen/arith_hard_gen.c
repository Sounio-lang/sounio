/* arith_hard_gen.c
 *
 * V0-D softfloat hard-case corpus — external oracle only (MPFR).
 * No Sounio arithmetic. Produce the answers before any implementation
 * has an interest in them.
 *
 * Families (structural, not random):
 *   halfway_tie_even     — exact result midway; RNE chooses even LSB
 *   sticky_bit           — bits below half-ulp set; must round away from tie
 *   catastrophic_cancel  — near-equal operands destroy leading digits
 *   rump                 — Rump 1988 poly (a=77617,b=33096); f64 sign flips
 *   sqrt_hard            — sqrt halfway / subnormal / non-square
 *   overflow_underflow   — results that overflow/underflow under RNE
 *
 * Ops: add, sub, mul, div, sqrt (unary). Formats: binary128 and binary256.
 *
 * Build:
 *   gcc -O2 -Wall -Wextra -o arith_hard_gen arith_hard_gen.c -lmpfr -lgmp
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <mpfr.h>
#include <gmp.h>

#define F128_P 113
#define F128_BIAS 16383
#define F128_EMAX 16383
#define F128_TRAIL 112
#define F128_EMIN (1 - F128_EMAX)

#define F256_P 237
#define F256_BIAS 262143
#define F256_EMAX 262143
#define F256_TRAIL 236
#define F256_EMIN (1 - F256_EMAX)

#define EXT 4096 /* >> 237 */

typedef struct {
    int sign;
    uint64_t exp_biased;
    uint64_t limbs[4];
    const char *klass;
} wire_t;

static const char *g_mpfr_ver;
static int g_id;

/* ---------- wire encode (same contract as literal_boundary_gen / minimax) ---------- */

static void mpfr_to_f128(wire_t *out, mpfr_srcptr x) {
    memset(out, 0, sizeof(*out));
    if (mpfr_nan_p(x)) {
        out->klass = "nan";
        out->exp_biased = 0x7FFF;
        out->limbs[1] = ((uint64_t)0x7FFF << 48) | (1ULL << 47);
        return;
    }
    if (mpfr_inf_p(x)) {
        out->klass = "inf";
        out->sign = mpfr_signbit(x) ? 1 : 0;
        out->exp_biased = 0x7FFF;
        out->limbs[1] = ((uint64_t)out->sign << 63) | ((uint64_t)0x7FFF << 48);
        return;
    }
    if (mpfr_zero_p(x)) {
        out->klass = "zero";
        out->sign = mpfr_signbit(x) ? 1 : 0;
        out->limbs[1] = (uint64_t)out->sign << 63;
        return;
    }

    mpfr_t y;
    mpfr_init2(y, F128_P);
    mpfr_set(y, x, MPFR_RNDN);
    out->sign = mpfr_signbit(y) ? 1 : 0;
    if (out->sign)
        mpfr_neg(y, y, MPFR_RNDN);

    mpfr_exp_t e = mpfr_get_exp(y);
    int64_t E = (int64_t)e - 1;

    if (E > F128_EMAX) {
        out->klass = "inf";
        out->exp_biased = 0x7FFF;
        out->limbs[1] = ((uint64_t)out->sign << 63) | ((uint64_t)0x7FFF << 48);
        mpfr_clear(y);
        return;
    }

    if (E < F128_EMIN) {
        mpfr_t z;
        mpfr_init2(z, F128_P + 32);
        mpfr_set(z, y, MPFR_RNDN);
        mpfr_mul_2si(z, z, -F128_EMIN, MPFR_RNDN);
        if (mpfr_cmp_ui(z, 1) < 0) {
            mpfr_mul_2ui(z, z, F128_TRAIL, MPFR_RNDN);
            mpz_t zi;
            mpz_init(zi);
            mpfr_get_z(zi, z, MPFR_RNDN);
            if (mpz_sgn(zi) == 0) {
                out->klass = "zero";
                out->limbs[1] = (uint64_t)out->sign << 63;
            } else {
                out->klass = "subnormal";
                out->exp_biased = 0;
                out->limbs[0] = (uint64_t)mpz_get_ui(zi);
                mpz_tdiv_q_2exp(zi, zi, 64);
                uint64_t hi_t = (uint64_t)mpz_get_ui(zi);
                out->limbs[1] = ((uint64_t)out->sign << 63) | (hi_t & 0xFFFFFFFFFFFFULL);
            }
            mpz_clear(zi);
            mpfr_clear(z);
            mpfr_clear(y);
            return;
        }
        mpfr_clear(z);
        E = F128_EMIN;
        mpfr_set_ui_2exp(y, 1, E, MPFR_RNDN);
    }

    out->klass = "normal";
    out->exp_biased = (uint64_t)(E + F128_BIAS);
    mpfr_t frac;
    mpfr_init2(frac, F128_P);
    mpfr_set(frac, y, MPFR_RNDN);
    mpfr_div_2si(frac, frac, E, MPFR_RNDN);
    mpfr_sub_ui(frac, frac, 1, MPFR_RNDN);
    mpfr_mul_2ui(frac, frac, F128_TRAIL, MPFR_RNDN);
    mpz_t zi;
    mpz_init(zi);
    mpfr_get_z(zi, frac, MPFR_RNDN);
    out->limbs[0] = (uint64_t)mpz_get_ui(zi);
    mpz_tdiv_q_2exp(zi, zi, 64);
    uint64_t hi_t = (uint64_t)mpz_get_ui(zi);
    out->limbs[1] = ((uint64_t)out->sign << 63)
                  | ((out->exp_biased & 0x7FFFULL) << 48)
                  | (hi_t & 0xFFFFFFFFFFFFULL);
    mpz_clear(zi);
    mpfr_clear(frac);
    mpfr_clear(y);
}

static void mpfr_to_f256(wire_t *out, mpfr_srcptr x) {
    memset(out, 0, sizeof(*out));
    if (mpfr_nan_p(x)) {
        out->klass = "nan";
        out->exp_biased = 0x7FFFF;
        out->limbs[3] = ((uint64_t)0x7FFFF << 44) | (1ULL << 43);
        return;
    }
    if (mpfr_inf_p(x)) {
        out->klass = "inf";
        out->sign = mpfr_signbit(x) ? 1 : 0;
        out->exp_biased = 0x7FFFF;
        out->limbs[3] = ((uint64_t)out->sign << 63) | ((uint64_t)0x7FFFF << 44);
        return;
    }
    if (mpfr_zero_p(x)) {
        out->klass = "zero";
        out->sign = mpfr_signbit(x) ? 1 : 0;
        out->limbs[3] = (uint64_t)out->sign << 63;
        return;
    }

    mpfr_t y;
    mpfr_init2(y, F256_P);
    mpfr_set(y, x, MPFR_RNDN);
    out->sign = mpfr_signbit(y) ? 1 : 0;
    if (out->sign)
        mpfr_neg(y, y, MPFR_RNDN);

    mpfr_exp_t e = mpfr_get_exp(y);
    int64_t E = (int64_t)e - 1;

    if (E > F256_EMAX) {
        out->klass = "inf";
        out->exp_biased = 0x7FFFF;
        out->limbs[3] = ((uint64_t)out->sign << 63) | ((uint64_t)0x7FFFF << 44);
        mpfr_clear(y);
        return;
    }

    if (E < F256_EMIN) {
        mpfr_t z;
        mpfr_init2(z, F256_P + 32);
        mpfr_set(z, y, MPFR_RNDN);
        mpfr_mul_2si(z, z, -F256_EMIN, MPFR_RNDN);
        if (mpfr_cmp_ui(z, 1) < 0) {
            mpfr_mul_2ui(z, z, F256_TRAIL, MPFR_RNDN);
            mpz_t zi;
            mpz_init(zi);
            mpfr_get_z(zi, z, MPFR_RNDN);
            if (mpz_sgn(zi) == 0) {
                out->klass = "zero";
                out->limbs[3] = (uint64_t)out->sign << 63;
            } else {
                out->klass = "subnormal";
                for (int i = 0; i < 4; i++) {
                    out->limbs[i] = (uint64_t)mpz_get_ui(zi);
                    mpz_tdiv_q_2exp(zi, zi, 64);
                }
                out->limbs[3] &= 0x000FFFFFFFFFFFFFULL;
                out->limbs[3] |= (uint64_t)out->sign << 63;
            }
            mpz_clear(zi);
            mpfr_clear(z);
            mpfr_clear(y);
            return;
        }
        mpfr_clear(z);
        E = F256_EMIN;
        mpfr_set_ui_2exp(y, 1, E, MPFR_RNDN);
    }

    out->klass = "normal";
    out->exp_biased = (uint64_t)(E + F256_BIAS);
    mpfr_t frac;
    mpfr_init2(frac, F256_P);
    mpfr_set(frac, y, MPFR_RNDN);
    mpfr_div_2si(frac, frac, E, MPFR_RNDN);
    mpfr_sub_ui(frac, frac, 1, MPFR_RNDN);
    mpfr_mul_2ui(frac, frac, F256_TRAIL, MPFR_RNDN);
    mpz_t zi;
    mpz_init(zi);
    mpfr_get_z(zi, frac, MPFR_RNDN);
    for (int i = 0; i < 4; i++) {
        out->limbs[i] = (uint64_t)mpz_get_ui(zi);
        mpz_tdiv_q_2exp(zi, zi, 64);
    }
    uint64_t t_hi = out->limbs[3] & 0x000FFFFFFFFFFFFFULL;
    out->limbs[3] = ((uint64_t)out->sign << 63)
                  | ((out->exp_biased & 0x7FFFFULL) << 44)
                  | t_hi;
    mpz_clear(zi);
    mpfr_clear(frac);
    mpfr_clear(y);
}

static void print_trailing_f128(const wire_t *w) {
    uint64_t hi_t = w->limbs[1] & 0xFFFFFFFFFFFFULL;
    printf("%012" PRIx64 "%016" PRIx64, hi_t, w->limbs[0]);
}

static void print_trailing_f256(const wire_t *w) {
    uint64_t t3 = w->limbs[3] & 0x000FFFFFFFFFFFFFULL;
    printf("%011" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64,
           t3, w->limbs[2], w->limbs[1], w->limbs[0]);
}

static void emit_wire(const wire_t *w, int is_f256) {
    printf("{\"class\":\"%s\",\"sign\":%d,\"exponent\":%" PRIu64
           ",\"trailing_hex\":\"",
           w->klass, w->sign, w->exp_biased);
    if (is_f256)
        print_trailing_f256(w);
    else
        print_trailing_f128(w);
    printf("\",\"limbs\":[");
    int n = is_f256 ? 4 : 2;
    for (int i = 0; i < n; i++) {
        if (i)
            printf(",");
        printf("%" PRId64, (int64_t)w->limbs[i]);
    }
    printf("]}");
}

static void from_str(mpfr_t out, const char *s) {
    if (mpfr_set_str(out, s, 0, MPFR_RNDN) != 0) {
        fprintf(stderr, "parse fail: %s\n", s);
        abort();
    }
}

static void set_pow2(mpfr_t out, int64_t e) {
    mpfr_set_ui_2exp(out, 1, e, MPFR_RNDN);
}

/* ---------- vector emission ---------- */

static void emit_bin(const char *fmt_name, int is_f256, const char *op,
                     const char *family, mpfr_srcptr a, mpfr_srcptr b,
                     const char *notes) {
    mpfr_t r;
    mpfr_init2(r, EXT);
    if (!strcmp(op, "add"))
        mpfr_add(r, a, b, MPFR_RNDN);
    else if (!strcmp(op, "sub"))
        mpfr_sub(r, a, b, MPFR_RNDN);
    else if (!strcmp(op, "mul"))
        mpfr_mul(r, a, b, MPFR_RNDN);
    else if (!strcmp(op, "div"))
        mpfr_div(r, a, b, MPFR_RNDN);
    else {
        fprintf(stderr, "bad op %s\n", op);
        abort();
    }

    wire_t wa, wb, wr;
    if (is_f256) {
        mpfr_to_f256(&wa, a);
        mpfr_to_f256(&wb, b);
        mpfr_to_f256(&wr, r);
    } else {
        mpfr_to_f128(&wa, a);
        mpfr_to_f128(&wb, b);
        mpfr_to_f128(&wr, r);
    }

    /* Also record f64 evaluation of same op for rump/diagnostics */
    mpfr_t da, db, dr;
    mpfr_init2(da, 53);
    mpfr_init2(db, 53);
    mpfr_init2(dr, 53);
    mpfr_set(da, a, MPFR_RNDN);
    mpfr_set(db, b, MPFR_RNDN);
    if (!strcmp(op, "add"))
        mpfr_add(dr, da, db, MPFR_RNDN);
    else if (!strcmp(op, "sub"))
        mpfr_sub(dr, da, db, MPFR_RNDN);
    else if (!strcmp(op, "mul"))
        mpfr_mul(dr, da, db, MPFR_RNDN);
    else
        mpfr_div(dr, da, db, MPFR_RNDN);
    int f64_sign = mpfr_signbit(dr) ? 1 : 0;
    int exact_sign = mpfr_signbit(r) ? 1 : 0;
    bool f64_sign_flip =
        !mpfr_nan_p(dr) && !mpfr_zero_p(dr) && !mpfr_nan_p(r) && !mpfr_zero_p(r) &&
        (f64_sign != exact_sign);

    g_id++;
    printf("{\"id\":\"%s_arith_%04d\",\"format\":\"%s\",\"op\":\"%s_%s\","
           "\"family\":\"%s\",\"arity\":2,\"a\":",
           is_f256 ? "f256" : "f128", g_id, fmt_name, is_f256 ? "f256" : "f128",
           op, family);
    emit_wire(&wa, is_f256);
    printf(",\"b\":");
    emit_wire(&wb, is_f256);
    printf(",\"result\":");
    emit_wire(&wr, is_f256);
    printf(",\"rounding\":\"rne\","
           "\"f64_sign_differs\":%s,"
           "\"provenance\":{"
           "\"tool\":\"MPFR\","
           "\"version\":\"%s\","
           "\"rounding_mode\":\"MPFR_RNDN\","
           "\"extended_precision_bits\":%d,"
           "\"invocation\":\"mpfr_%s(r,a,b,MPFR_RNDN) at p=%d then RNE-encode to %s\","
           "\"generator\":\"tests/vectors/f128_f256_v0d/gen/arith_hard_gen.c\","
           "\"notes\":\"%s\""
           "}}\n",
           f64_sign_flip ? "true" : "false", g_mpfr_ver, EXT, op, EXT, fmt_name,
           notes);

    mpfr_clear(r);
    mpfr_clears(da, db, dr, (mpfr_ptr)0);
}

static void emit_sqrt(const char *fmt_name, int is_f256, const char *family,
                      mpfr_srcptr a, const char *notes) {
    mpfr_t r;
    mpfr_init2(r, EXT);
    if (mpfr_sgn(a) < 0)
        mpfr_set_nan(r);
    else
        mpfr_sqrt(r, a, MPFR_RNDN);

    wire_t wa, wr;
    if (is_f256) {
        mpfr_to_f256(&wa, a);
        mpfr_to_f256(&wr, r);
    } else {
        mpfr_to_f128(&wa, a);
        mpfr_to_f128(&wr, r);
    }

    g_id++;
    printf("{\"id\":\"%s_arith_%04d\",\"format\":\"%s\",\"op\":\"%s_sqrt\","
           "\"family\":\"%s\",\"arity\":1,\"a\":",
           is_f256 ? "f256" : "f128", g_id, fmt_name, is_f256 ? "f256" : "f128",
           family);
    emit_wire(&wa, is_f256);
    printf(",\"b\":null,\"result\":");
    emit_wire(&wr, is_f256);
    printf(",\"rounding\":\"rne\","
           "\"f64_sign_differs\":false,"
           "\"provenance\":{"
           "\"tool\":\"MPFR\","
           "\"version\":\"%s\","
           "\"rounding_mode\":\"MPFR_RNDN\","
           "\"extended_precision_bits\":%d,"
           "\"invocation\":\"mpfr_sqrt(r,a,MPFR_RNDN) at p=%d then RNE-encode to %s\","
           "\"generator\":\"tests/vectors/f128_f256_v0d/gen/arith_hard_gen.c\","
           "\"notes\":\"%s\""
           "}}\n",
           g_mpfr_ver, EXT, EXT, fmt_name, notes);
    mpfr_clear(r);
}

/* Rump closed form as one result vector with a,b inputs */
static void emit_rump(const char *fmt_name, int is_f256) {
    /* f = 333.75 b^6 + a^2 (11 a^2 b^2 - b^6 - 121 b^4 - 2) + 5.5 b^8 + a/(2b)
     * a=77617, b=33096  (Rump 1988) */
    mpfr_t a, b, t, t2, t3, t4, t6, t8, a2, a4, term1, term2, term3, term4, f;
    mpfr_inits2(EXT, a, b, t, t2, t3, t4, t6, t8, a2, a4, term1, term2, term3,
                term4, f, (mpfr_ptr)0);
    mpfr_set_ui(a, 77617, MPFR_RNDN);
    mpfr_set_ui(b, 33096, MPFR_RNDN);

    mpfr_pow_ui(t2, b, 2, MPFR_RNDN);
    mpfr_pow_ui(t4, b, 4, MPFR_RNDN);
    mpfr_pow_ui(t6, b, 6, MPFR_RNDN);
    mpfr_pow_ui(t8, b, 8, MPFR_RNDN);
    mpfr_mul(a2, a, a, MPFR_RNDN);
    mpfr_mul(a4, a2, a2, MPFR_RNDN);

    /* term1 = 333.75 * b^6 */
    mpfr_set_d(t, 333.75, MPFR_RNDN);
    mpfr_mul(term1, t, t6, MPFR_RNDN);

    /* inner = 11 a^2 b^2 - b^6 - 121 b^4 - 2 */
    mpfr_mul(t, a2, t2, MPFR_RNDN);
    mpfr_mul_ui(t, t, 11, MPFR_RNDN);
    mpfr_sub(t, t, t6, MPFR_RNDN);
    mpfr_mul_ui(t3, t4, 121, MPFR_RNDN);
    mpfr_sub(t, t, t3, MPFR_RNDN);
    mpfr_sub_ui(t, t, 2, MPFR_RNDN);
    mpfr_mul(term2, a2, t, MPFR_RNDN);

    /* term3 = 5.5 * b^8 */
    mpfr_set_d(t, 5.5, MPFR_RNDN);
    mpfr_mul(term3, t, t8, MPFR_RNDN);

    /* term4 = a/(2b) */
    mpfr_mul_ui(t, b, 2, MPFR_RNDN);
    mpfr_div(term4, a, t, MPFR_RNDN);

    mpfr_add(f, term1, term2, MPFR_RNDN);
    mpfr_add(f, f, term3, MPFR_RNDN);
    mpfr_add(f, f, term4, MPFR_RNDN);

    /* Hardware double evaluation of the same AST (pow via libm).
     * On modern IEEE-754 binary64 this typically yields ~ -1.18e21
     * (catastrophic magnitude error), not the true ~ -0.827…;
     * older literature also reports a positive wrong value depending
     * on intermediate association. We record the host double result
     * re-encoded into the target format for comparison. */
    {
        double ad = 77617.0, bd = 33096.0;
        double b2 = bd * bd;
        double b4 = b2 * b2;
        double b6 = b4 * b2;
        double b8 = b4 * b4;
        double a2 = ad * ad;
        double fd = 333.75 * b6 + a2 * (11.0 * a2 * b2 - b6 - 121.0 * b4 - 2.0) +
                    5.5 * b8 + ad / (2.0 * bd);
        mpfr_t f64;
        mpfr_init2(f64, 53);
        mpfr_set_d(f64, fd, MPFR_RNDN);

        wire_t wa, wb, wr, w64;
        if (is_f256) {
            mpfr_to_f256(&wa, a);
            mpfr_to_f256(&wb, b);
            mpfr_to_f256(&wr, f);
            mpfr_to_f256(&w64, f64);
        } else {
            mpfr_to_f128(&wa, a);
            mpfr_to_f128(&wb, b);
            mpfr_to_f128(&wr, f);
            mpfr_to_f128(&w64, f64);
        }

        bool sign_flip =
            (mpfr_signbit(f) != 0) != (signbit(fd) != 0);
        bool bits_differ =
            (wr.sign != w64.sign) || (wr.exp_biased != w64.exp_biased) ||
            (wr.limbs[0] != w64.limbs[0]) || (wr.limbs[1] != w64.limbs[1]) ||
            (is_f256 && (wr.limbs[2] != w64.limbs[2] || wr.limbs[3] != w64.limbs[3]));

        g_id++;
        printf("{\"id\":\"%s_arith_%04d\",\"format\":\"%s\",\"op\":\"%s_rump1988\","
               "\"family\":\"rump\",\"arity\":2,\"a\":",
               is_f256 ? "f256" : "f128", g_id, fmt_name,
               is_f256 ? "f256" : "f128");
        emit_wire(&wa, is_f256);
        printf(",\"b\":");
        emit_wire(&wb, is_f256);
        printf(",\"result\":");
        emit_wire(&wr, is_f256);
        printf(",\"f64_result\":");
        emit_wire(&w64, is_f256);
        printf(",\"rounding\":\"rne\","
               "\"f64_sign_differs\":%s,"
               "\"f64_bits_differ\":%s,"
               "\"f64_host_double\":%.17g,"
               "\"expression\":\"333.75*b^6 + a^2*(11*a^2*b^2 - b^6 - 121*b^4 - 2) + "
               "5.5*b^8 + a/(2*b)\","
               "\"provenance\":{"
               "\"tool\":\"MPFR\","
               "\"version\":\"%s\","
               "\"rounding_mode\":\"MPFR_RNDN\","
               "\"extended_precision_bits\":%d,"
               "\"invocation\":\"Rump 1988 closed form at a=77617 b=33096; "
               "exact: all ops MPFR p=%d RNDN then RNE-encode to %s; "
               "f64_result: same AST in IEEE-754 binary64 (C double/libm) "
               "then mpfr_set_d + RNE-encode to %s\","
               "\"generator\":\"tests/vectors/f128_f256_v0d/gen/arith_hard_gen.c\","
               "\"notes\":\"Ill-conditioned poly; insufficient precision destroys "
               "correct magnitude (and historically sign). Repo EISA fixtures use "
               "same (a,b). Exact ~ -0.827396059946821…\","
               "\"citation\":\"Rump 1988; tools/eisa rump_build (77617,33096)\""
               "}}\n",
               sign_flip ? "true" : "false", bits_differ ? "true" : "false", fd,
               g_mpfr_ver, EXT, EXT, fmt_name, fmt_name);

        mpfr_clear(f64);
    }

    mpfr_clears(a, b, t, t2, t3, t4, t6, t8, a2, a4, term1, term2, term3, term4, f,
                (mpfr_ptr)0);
}

static void emit_format(int is_f256) {
    const char *fmt = is_f256 ? "binary256" : "binary128";
    int p = is_f256 ? F256_P : F128_P;
    /* ulp(1) = 2^(1-p) */
    int64_t e_ulp = 1 - p;       /* exponent of ulp at 1.0 */
    int64_t e_half = -p;         /* half ulp = 2^-p */

    mpfr_t one, half_ulp, ulp, two_ulps, three_ulps, a, b, eps, big, small;
    mpfr_inits2(EXT, one, half_ulp, ulp, two_ulps, three_ulps, a, b, eps, big,
                small, (mpfr_ptr)0);

    mpfr_set_ui(one, 1, MPFR_RNDN);
    set_pow2(half_ulp, e_half);
    set_pow2(ulp, e_ulp);
    set_pow2(two_ulps, e_ulp + 1);
    /* three_ulps = 3 * ulp */
    mpfr_mul_ui(three_ulps, ulp, 3, MPFR_RNDN);

    /* ---- halfway / tie-to-even ---- */
    /* 1 + half_ulp → exact mid between 1 and 1+ulp; 1 is even → stays 1 */
    emit_bin(fmt, is_f256, "add", "halfway_tie_even", one, half_ulp,
             "1 + ulp/2: RNE ties to even (1.0 has LSB 0)");
    /* (1+ulp) + half_ulp: mid between 1+ulp and 1+2ulp; 1+ulp has odd trailing → up */
    mpfr_add(a, one, ulp, MPFR_RNDN);
    emit_bin(fmt, is_f256, "add", "halfway_tie_even", a, half_ulp,
             "(1+ulp)+ulp/2: RNE ties away from odd LSB");
    emit_bin(fmt, is_f256, "add", "halfway_tie_even", ulp, half_ulp,
             "ulp + ulp/2 at subnormal/min scale");
    emit_bin(fmt, is_f256, "add", "halfway_tie_even", two_ulps, half_ulp,
             "2ulp + ulp/2: even → stay");
    emit_bin(fmt, is_f256, "add", "halfway_tie_even", three_ulps, half_ulp,
             "3ulp + ulp/2: odd → up");
    /* mul halfway: sqrt-style product midpoints */
    /* (1+2^-k) for k that creates mid product */
    set_pow2(eps, -(p / 2));
    mpfr_add(a, one, eps, MPFR_RNDN);
    emit_bin(fmt, is_f256, "mul", "halfway_tie_even", a, a,
             "(1+2^-(p/2))^2: product may land near mid between representables");

    /* ---- sticky bit: force bits below half-ulp ---- */
    /* 1 + half_ulp + tinier  → must round UP (sticky set), not tie-to-even stay */
    set_pow2(small, e_half - 8);
    mpfr_add(a, half_ulp, small, MPFR_RNDN);
    emit_bin(fmt, is_f256, "add", "sticky_bit", one, a,
             "1 + (ulp/2 + 2^(e_half-8)): sticky prevents pure-tie; round up");
    /* mul: (1+2^-(p-2)) * (1+2^-(p-2)) creates low product bits */
    set_pow2(eps, -(p - 2));
    mpfr_add(a, one, eps, MPFR_RNDN);
    mpfr_add(b, one, eps, MPFR_RNDN);
    emit_bin(fmt, is_f256, "mul", "sticky_bit", a, b,
             "(1+2^-(p-2))^2: low bits set sticky on RNE to p bits");
    /* div: 1 / (1 - 2^-k) expansion sticky */
    set_pow2(eps, -(p - 3));
    mpfr_sub(b, one, eps, MPFR_RNDN);
    emit_bin(fmt, is_f256, "div", "sticky_bit", one, b,
             "1/(1-2^-(p-3)): infinite series truncated → sticky");
    /* sub that leaves sticky residue */
    mpfr_set_ui(big, 1, MPFR_RNDN);
    mpfr_add(a, big, half_ulp, MPFR_RNDN);
    mpfr_add(a, a, small, MPFR_RNDN); /* 1 + half + tiny, then will be encoded */
    /* use exact mpfr operands not pre-rounded */
    mpfr_set(a, one, MPFR_RNDN);
    mpfr_add(a, a, half_ulp, MPFR_RNDN);
    mpfr_add(a, a, small, MPFR_RNDN);
    emit_bin(fmt, is_f256, "sub", "sticky_bit", a, one,
             "(1+ulp/2+tiny) - 1: residual near half-ulp with sticky");

    /* ---- catastrophic cancellation (parity target ~20+ per format) ----
     * Two nearly-equal operands destroy leading digits; a correct softfloat
     * must retain enough guard bits for the residual. Ladder of eps scales,
     * large-magnitude Sterbenz pairs, opposite-sign adds, and product forms.
     */
    {
        /* (1 + 2^-k) - 1 for k in a ladder spanning coarse → near-p */
        static const int k_ladder[] = {1, 2, 4, 8, 12, 16, 20, 24, 30, 40,
                                       48, 52, 60, 80};
        for (size_t i = 0; i < sizeof(k_ladder) / sizeof(k_ladder[0]); i++) {
            int k = k_ladder[i];
            if (k >= p - 2)
                continue; /* residual must remain normal or at least nonzero */
            set_pow2(eps, -k);
            mpfr_add(a, one, eps, MPFR_RNDN);
            char note[96];
            snprintf(note, sizeof(note),
                     "(1+2^-%d)-1: cancel leaves 2^-%d", k, k);
            emit_bin(fmt, is_f256, "sub", "catastrophic_cancel", a, one, note);
        }
        /* Negative nearly-equal: (-1 - 2^-k) - (-1) */
        set_pow2(eps, -16);
        mpfr_set_si(a, -1, MPFR_RNDN);
        mpfr_sub(a, a, eps, MPFR_RNDN); /* -1 - eps */
        mpfr_set_si(b, -1, MPFR_RNDN);
        emit_bin(fmt, is_f256, "sub", "catastrophic_cancel", a, b,
                 "(-1-2^-16)-(-1): signed cancel");
        /* Opposite-sign add that cancels: (1+e) + (-1) */
        set_pow2(eps, -25);
        mpfr_add(a, one, eps, MPFR_RNDN);
        mpfr_set_si(b, -1, MPFR_RNDN);
        emit_bin(fmt, is_f256, "add", "catastrophic_cancel", a, b,
                 "(1+2^-25)+(-1): cancel via add of opposites");
        /* Sterbenz large magnitude, several decades */
        static const int e_big[] = {10, 40, 80, 100, 200};
        for (size_t i = 0; i < sizeof(e_big) / sizeof(e_big[0]); i++) {
            int eb = e_big[i];
            if (is_f256 == 0 && eb > F128_EMAX - 2)
                continue;
            if (is_f256 && eb > 1000)
                continue;
            mpfr_set_ui_2exp(big, 1, eb, MPFR_RNDN);
            set_pow2(eps, eb - 30);
            mpfr_add(a, big, eps, MPFR_RNDN);
            char note[96];
            snprintf(note, sizeof(note), "2^%d+2^%d - 2^%d", eb, eb - 30, eb);
            emit_bin(fmt, is_f256, "sub", "catastrophic_cancel", a, big, note);
        }
        /* Adjacent integers around 1e6 */
        mpfr_set_ui(a, 1000003, MPFR_RNDN);
        mpfr_set_ui(b, 1000001, MPFR_RNDN);
        emit_bin(fmt, is_f256, "sub", "catastrophic_cancel", a, b,
                 "1000003-1000001: small difference of large ints");
        mpfr_set_ui(a, 16777217, MPFR_RNDN); /* past f32 exact int range */
        mpfr_set_ui(b, 16777215, MPFR_RNDN);
        emit_bin(fmt, is_f256, "sub", "catastrophic_cancel", a, b,
                 "2^24+1 - (2^24-1): cancel past binary32 integer span");
        /* (1+e)-(1+e/2) residual e/2 */
        set_pow2(eps, -18);
        mpfr_add(a, one, eps, MPFR_RNDN);
        mpfr_div_ui(b, eps, 2, MPFR_RNDN);
        mpfr_add(b, one, b, MPFR_RNDN);
        emit_bin(fmt, is_f256, "sub", "catastrophic_cancel", a, b,
                 "(1+2^-18)-(1+2^-19)");
        /* Product cancellation: (1+e)(1-e)=1-e^2 for several e */
        static const int e_prod[] = {8, 16, 24, 30, 40};
        for (size_t i = 0; i < sizeof(e_prod) / sizeof(e_prod[0]); i++) {
            set_pow2(eps, -e_prod[i]);
            mpfr_add(a, one, eps, MPFR_RNDN);
            mpfr_sub(b, one, eps, MPFR_RNDN);
            char note[96];
            snprintf(note, sizeof(note),
                     "(1+2^-%d)(1-2^-%d)=1-2^-%d", e_prod[i], e_prod[i],
                     2 * e_prod[i]);
            emit_bin(fmt, is_f256, "mul", "catastrophic_cancel", a, b, note);
        }
        /* div cancel: (1+e)-1 then implicit; or (x-y)/x ≈ 0 */
        set_pow2(eps, -22);
        mpfr_add(a, one, eps, MPFR_RNDN);
        mpfr_sub(b, a, one, MPFR_RNDN); /* b should be eps if a exact */
        /* a/a - 1 style: compute (a - a) is trivial; use (a+eps)/a */
        mpfr_add(a, one, eps, MPFR_RNDN);
        emit_bin(fmt, is_f256, "div", "catastrophic_cancel", a, one,
                 "(1+2^-22)/1: near-1 quotient; residual after -1 is cancel-class");
        /* Nearly equal subnormals (if representable) */
        if (!is_f256) {
            set_pow2(a, -16400);
            set_pow2(b, -16401);
            emit_bin(fmt, is_f256, "sub", "catastrophic_cancel", a, b,
                     "near-equal deep subnormals: 2^-16400 - 2^-16401");
        }
    }

    /* ---- overflow / underflow structural ---- */
    if (!is_f256) {
        mpfr_set_ui_2exp(a, 1, F128_EMAX, MPFR_RNDN);
        mpfr_set_ui(b, 2, MPFR_RNDN);
        emit_bin(fmt, is_f256, "mul", "overflow_underflow", a, b,
                 "2^emax * 2 → overflow to inf");
        set_pow2(a, F128_EMIN);
        set_pow2(b, -10);
        emit_bin(fmt, is_f256, "mul", "overflow_underflow", a, b,
                 "min_normal * 2^-10 → subnormal/underflow domain");
        set_pow2(a, -16494); /* min subnormal */
        set_pow2(b, -10);
        emit_bin(fmt, is_f256, "mul", "overflow_underflow", a, b,
                 "min_subnormal * 2^-10 → deeper underflow / zero");
    } else {
        mpfr_set_ui_2exp(a, 1, F256_EMAX, MPFR_RNDN);
        mpfr_set_ui(b, 2, MPFR_RNDN);
        emit_bin(fmt, is_f256, "mul", "overflow_underflow", a, b,
                 "2^emax * 2 → overflow");
        set_pow2(a, F256_EMIN);
        set_pow2(b, -20);
        emit_bin(fmt, is_f256, "mul", "overflow_underflow", a, b,
                 "min_normal * 2^-20 → subnormal domain");
    }

    /* ---- sqrt hard ---- */
    mpfr_set_ui(a, 4, MPFR_RNDN);
    emit_sqrt(fmt, is_f256, "sqrt_hard", a, "sqrt(4)=2 exact");
    mpfr_set_ui(a, 2, MPFR_RNDN);
    emit_sqrt(fmt, is_f256, "sqrt_hard", a, "sqrt(2) irrational RNE");
    /* perfect square just below a midpoint */
    mpfr_set_ui(a, 1, MPFR_RNDN);
    emit_sqrt(fmt, is_f256, "sqrt_hard", a, "sqrt(1)=1");
    /* subnormal sqrt */
    if (!is_f256) {
        set_pow2(a, -16494);
        emit_sqrt(fmt, is_f256, "sqrt_hard", a, "sqrt(min_subnormal)");
        set_pow2(a, F128_EMIN);
        emit_sqrt(fmt, is_f256, "sqrt_hard", a, "sqrt(min_normal)");
    } else {
        set_pow2(a, F256_EMIN - (F256_P - 1));
        emit_sqrt(fmt, is_f256, "sqrt_hard", a, "sqrt(min_subnormal)");
    }
    /* (1+ulp) not a perfect square */
    mpfr_add(a, one, ulp, MPFR_RNDN);
    emit_sqrt(fmt, is_f256, "sqrt_hard", a, "sqrt(1+ulp)");

    /* div hard: /3 recurring */
    mpfr_set_ui(a, 1, MPFR_RNDN);
    mpfr_set_ui(b, 3, MPFR_RNDN);
    emit_bin(fmt, is_f256, "div", "sticky_bit", a, b,
             "1/3: recurring binary; sticky-critical RNE");
    mpfr_set_ui(a, 1, MPFR_RNDN);
    mpfr_set_ui(b, 10, MPFR_RNDN);
    emit_bin(fmt, is_f256, "div", "sticky_bit", a, b, "1/10 sticky/recurring");

    /* Rump */
    emit_rump(fmt, is_f256);

    mpfr_clears(one, half_ulp, ulp, two_ulps, three_ulps, a, b, eps, big, small,
                (mpfr_ptr)0);
}

int main(void) {
    g_mpfr_ver = mpfr_get_version();
    fprintf(stderr,
            "arith_hard_gen: MPFR %s RNDN ext=%d generator=f128_f256_v0d\n",
            g_mpfr_ver, EXT);
    g_id = 0;
    emit_format(0);
    int n128 = g_id;
    emit_format(1);
    fprintf(stderr, "emitted f128=%d f256=%d total=%d\n", n128, g_id - n128,
            g_id);
    return 0;
}
