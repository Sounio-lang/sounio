/* f256_only_traps_gen.c
 *
 * Structural traps that appear for binary256 softfloat but have no
 * binary128 analogue of the same shape. Oracle: MPFR. No Sounio.
 *
 * Claims under test (measured, not assumed):
 *
 * 1) f128-cascade mul:
 *      a,b free in R;  a256 = RNE_f256(a); b256 = RNE_f256(b);
 *      direct  = RNE_f256(a256 * b256)     [exact product of decoded values]
 *      cascade = widen_f256( RNE_f128(a256) * RNE_f128(b256) )
 *    If direct != cascade, a correct f256 mul must not factor through f128.
 *    There is no f128 analogue "factor through f64" that is *the same*
 *    structure at a wider free precision — f128-through-f64 is double-round;
 *    f256-through-f128 is a *new* intermediate width.
 *
 * 2) triple-round path on a *product*:
 *      direct vs RNE_f256(RNE_f128(RNE_f64(a)*RNE_f64(b))) style chains
 *    Differs from literal double-round (single value) by being op-shaped.
 *
 * 3) schoolbook limb truncate (4-limb product kept, low 4 discarded w/o sticky):
 *      Simulate mul of 4×64 significand limbs keeping only top 237 bits of the
 *      474-bit product with NO sticky — wrong whenever low 237 bits are nonzero
 *      and affect RNE. At f128, schoolbook is 2×2→4 limbs; the *4-limb
 *      significand* schoolbook error class only exists at f256.
 *
 * Build: gcc -O2 -Wall -o f256_only_traps_gen f256_only_traps_gen.c -lmpfr -lgmp -lm
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

#define EXT 8192

typedef struct {
    int sign;
    uint64_t exp_biased;
    uint64_t limbs[4];
    const char *klass;
} wire_t;

static const char *g_ver;
static int g_id;

/* ---- encode (same contract as arith_hard / literal_boundary) ---- */

static void mpfr_to_f128(wire_t *out, mpfr_srcptr x);
static void mpfr_to_f256(wire_t *out, mpfr_srcptr x);

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
                out->limbs[0] = (uint64_t)mpz_get_ui(zi);
                mpz_tdiv_q_2exp(zi, zi, 64);
                out->limbs[1] = ((uint64_t)out->sign << 63) |
                                ((uint64_t)mpz_get_ui(zi) & 0xFFFFFFFFFFFFULL);
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
    out->limbs[1] = ((uint64_t)out->sign << 63) |
                    ((out->exp_biased & 0x7FFFULL) << 48) |
                    ((uint64_t)mpz_get_ui(zi) & 0xFFFFFFFFFFFFULL);
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
    out->limbs[3] = ((uint64_t)out->sign << 63) |
                    ((out->exp_biased & 0x7FFFFULL) << 44) | t_hi;
    mpz_clear(zi);
    mpfr_clear(frac);
    mpfr_clear(y);
}

static bool wire_eq(const wire_t *a, const wire_t *b) {
    if (a->sign != b->sign || a->exp_biased != b->exp_biased)
        return false;
    for (int i = 0; i < 4; i++)
        if (a->limbs[i] != b->limbs[i])
            return false;
    return true;
}

static void print_trail256(const wire_t *w) {
    uint64_t t3 = w->limbs[3] & 0x000FFFFFFFFFFFFFULL;
    printf("%011" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64, t3,
           w->limbs[2], w->limbs[1], w->limbs[0]);
}

static void emit_wire(const wire_t *w) {
    printf("{\"class\":\"%s\",\"sign\":%d,\"exponent\":%" PRIu64
           ",\"trailing_hex\":\"",
           w->klass, w->sign, w->exp_biased);
    print_trail256(w);
    printf("\",\"limbs\":[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]}",
           (int64_t)w->limbs[0], (int64_t)w->limbs[1], (int64_t)w->limbs[2],
           (int64_t)w->limbs[3]);
}

/* Decode wire back to mpfr (for schoolbook on significands) */
static void f256_wire_to_mpfr(mpfr_t out, const wire_t *w) {
    if (!strcmp(w->klass, "nan")) {
        mpfr_set_nan(out);
        return;
    }
    if (!strcmp(w->klass, "inf")) {
        mpfr_set_inf(out, w->sign ? -1 : 1);
        return;
    }
    if (!strcmp(w->klass, "zero")) {
        mpfr_set_zero(out, w->sign ? -1 : 1);
        return;
    }
    /* reconstruct significand integer */
    mpz_t sig;
    mpz_init(sig);
    uint64_t trail[4];
    trail[0] = w->limbs[0];
    trail[1] = w->limbs[1];
    trail[2] = w->limbs[2];
    trail[3] = w->limbs[3] & 0x000FFFFFFFFFFFFFULL;
    mpz_import(sig, 4, -1, 8, -1, 0, trail);
    mpz_fdiv_r_2exp(sig, sig, F256_TRAIL);
    if (!strcmp(w->klass, "subnormal")) {
        /* value = trail * 2^(emin - (p-1)) = trail * 2^(F256_EMIN - 236) */
        mpfr_set_z_2exp(out, sig, F256_EMIN - (F256_P - 1), MPFR_RNDN);
    } else {
        mpz_setbit(sig, F256_P - 1); /* hidden 1 */
        int64_t E = (int64_t)w->exp_biased - F256_BIAS;
        mpfr_set_z_2exp(out, sig, E - (F256_P - 1), MPFR_RNDN);
    }
    if (w->sign)
        mpfr_neg(out, out, MPFR_RNDN);
    mpz_clear(sig);
}

/* Schoolbook mul of two f256 values keeping only top F256_P bits of product
 * significand with NO sticky (truncate toward zero in the extended sig, then
 * force RNE as if discarded bits were 0). This is the naive 4-limb error. */
static void schoolbook_no_sticky_mul(wire_t *out, const wire_t *a,
                                     const wire_t *b) {
    mpfr_t xa, xb, prod, trunc;
    mpfr_inits2(EXT, xa, xb, prod, trunc, (mpfr_ptr)0);
    f256_wire_to_mpfr(xa, a);
    f256_wire_to_mpfr(xb, b);
    mpfr_mul(prod, xa, xb, MPFR_RNDN); /* exact at EXT */

    /* Truncate significand to p bits toward zero (chop sticky region),
     * then the true RNE from full prod is computed separately as oracle.
     * Here we encode truncate-then-as-if-exact: use MPFR_RNDZ to p bits. */
    mpfr_init2(trunc, F256_P);
    mpfr_set(trunc, prod, MPFR_RNDZ); /* toward zero = drop sticky without round-up */
    mpfr_to_f256(out, trunc);
    /* Mark class from encode */
    mpfr_clears(xa, xb, prod, trunc, (mpfr_ptr)0);
}

static void emit_case(const char *trap_id, const char *family, const char *op,
                      const wire_t *a, const wire_t *b, const wire_t *direct,
                      const wire_t *wrong_path, const char *wrong_name,
                      const char *derivation, const char *why_f256_only) {
    g_id++;
    printf("{\"id\":\"f256_only_%04d\",\"format\":\"binary256\","
           "\"trap_id\":\"%s\",\"family\":\"%s\",\"op\":\"%s\","
           "\"a\":",
           g_id, trap_id, family, op);
    emit_wire(a);
    printf(",\"b\":");
    emit_wire(b);
    printf(",\"result_direct\":");
    emit_wire(direct);
    printf(",\"result_wrong_path\":");
    emit_wire(wrong_path);
    printf(",\"wrong_path_name\":\"%s\","
           "\"paths_differ\":true,"
           "\"rounding\":\"rne\","
           "\"provenance\":{"
           "\"tool\":\"MPFR\","
           "\"version\":\"%s\","
           "\"rounding_mode\":\"MPFR_RNDN\","
           "\"extended_precision_bits\":%d,"
           "\"generator\":\"tests/vectors/f128_f256_v0d/gen/f256_only_traps_gen.c\","
           "\"derivation\":\"%s\","
           "\"why_f256_only\":\"%s\","
           "\"invocation\":\"see derivation\""
           "}}\n",
           wrong_name, g_ver, EXT, derivation, why_f256_only);
}

/* Build f256 from string via high prec then RNE */
static void from_str_f256(wire_t *w, const char *s) {
    mpfr_t x;
    mpfr_init2(x, EXT);
    if (mpfr_set_str(x, s, 0, MPFR_RNDN) != 0) {
        fprintf(stderr, "parse fail %s\n", s);
        abort();
    }
    mpfr_to_f256(w, x);
    mpfr_clear(x);
}

static void mul_direct_f256(wire_t *out, const wire_t *a, const wire_t *b) {
    mpfr_t xa, xb, r;
    mpfr_inits2(EXT, xa, xb, r, (mpfr_ptr)0);
    f256_wire_to_mpfr(xa, a);
    f256_wire_to_mpfr(xb, b);
    mpfr_mul(r, xa, xb, MPFR_RNDN);
    mpfr_to_f256(out, r);
    mpfr_clears(xa, xb, r, (mpfr_ptr)0);
}

static void mul_via_f128_cascade(wire_t *out, const wire_t *a, const wire_t *b) {
    /* RNE_f128(a) * RNE_f128(b) in high prec, then RNE to f256 */
    mpfr_t xa, xb, a128, b128, r;
    mpfr_inits2(EXT, xa, xb, a128, b128, r, (mpfr_ptr)0);
    f256_wire_to_mpfr(xa, a);
    f256_wire_to_mpfr(xb, b);
    mpfr_set(a128, xa, MPFR_RNDN);
    mpfr_prec_round(a128, F128_P, MPFR_RNDN);
    mpfr_set(b128, xb, MPFR_RNDN);
    mpfr_prec_round(b128, F128_P, MPFR_RNDN);
    mpfr_mul(r, a128, b128, MPFR_RNDN);
    /* product of f128 values: round to f128 first (simulate f128 mul), then widen */
    mpfr_prec_round(r, F128_P, MPFR_RNDN);
    mpfr_to_f256(out, r);
    mpfr_clears(xa, xb, a128, b128, r, (mpfr_ptr)0);
}

static void mul_via_triple(wire_t *out, const wire_t *a, const wire_t *b) {
    /* a→f64, b→f64, mul f64, →f128, →f256 */
    mpfr_t xa, xb, a64, b64, r;
    mpfr_inits2(EXT, xa, xb, a64, b64, r, (mpfr_ptr)0);
    f256_wire_to_mpfr(xa, a);
    f256_wire_to_mpfr(xb, b);
    mpfr_set(a64, xa, MPFR_RNDN);
    mpfr_prec_round(a64, 53, MPFR_RNDN);
    mpfr_set(b64, xb, MPFR_RNDN);
    mpfr_prec_round(b64, 53, MPFR_RNDN);
    mpfr_mul(r, a64, b64, MPFR_RNDN);
    mpfr_prec_round(r, 53, MPFR_RNDN);
    mpfr_prec_round(r, F128_P, MPFR_RNDN);
    mpfr_to_f256(out, r);
    mpfr_clears(xa, xb, a64, b64, r, (mpfr_ptr)0);
}

/* f128-only analogue of cascade: via f64 — used to show same values may
 * differ at f128 via f64 AND at f256 via f128 (document both). */
static void mul_f128_via_f64(wire_t *out128, const wire_t *a256,
                             const wire_t *b256) {
    mpfr_t xa, xb, a64, b64, r;
    mpfr_inits2(EXT, xa, xb, a64, b64, r, (mpfr_ptr)0);
    f256_wire_to_mpfr(xa, a256);
    f256_wire_to_mpfr(xb, b256);
    mpfr_set(a64, xa, MPFR_RNDN);
    mpfr_prec_round(a64, 53, MPFR_RNDN);
    mpfr_set(b64, xb, MPFR_RNDN);
    mpfr_prec_round(b64, 53, MPFR_RNDN);
    mpfr_mul(r, a64, b64, MPFR_RNDN);
    mpfr_prec_round(r, 53, MPFR_RNDN);
    mpfr_to_f128(out128, r);
    mpfr_clears(xa, xb, a64, b64, r, (mpfr_ptr)0);
}

static void mul_f128_direct(wire_t *out128, const wire_t *a256,
                            const wire_t *b256) {
    mpfr_t xa, xb, a128, b128, r;
    mpfr_inits2(EXT, xa, xb, a128, b128, r, (mpfr_ptr)0);
    f256_wire_to_mpfr(xa, a256);
    f256_wire_to_mpfr(xb, b256);
    mpfr_set(a128, xa, MPFR_RNDN);
    mpfr_prec_round(a128, F128_P, MPFR_RNDN);
    mpfr_set(b128, xb, MPFR_RNDN);
    mpfr_prec_round(b128, F128_P, MPFR_RNDN);
    mpfr_mul(r, a128, b128, MPFR_RNDN);
    mpfr_to_f128(out128, r);
    mpfr_clears(xa, xb, a128, b128, r, (mpfr_ptr)0);
}

static void try_pair(const char *tag, const char *sa, const char *sb) {
    wire_t a, b, direct, via128, via_tri, school;
    from_str_f256(&a, sa);
    from_str_f256(&b, sb);
    mul_direct_f256(&direct, &a, &b);
    mul_via_f128_cascade(&via128, &a, &b);
    mul_via_triple(&via_tri, &a, &b);
    schoolbook_no_sticky_mul(&school, &a, &b);

    if (!wire_eq(&direct, &via128)) {
        char der[512];
        snprintf(der, sizeof(der),
                 "a=RNE_f256(%s); b=RNE_f256(%s); "
                 "direct=RNE_f256(a*b) at p=%d; "
                 "cascade=RNE_f256(RNE_f128(RNE_f128(a)*RNE_f128(b))) "
                 "i.e. operands and product rounded through binary128",
                 sa, sb, EXT);
        emit_case(tag, "f128_cascade_mul", "f256_mul", &a, &b, &direct, &via128,
                  "via_f128_mul_widen", der,
                  "Intermediate width is binary128; f128 softfloat has no "
                  "wider-than-self IEEE binary intermediate of this form "
                  "(its cascade analogue is f64, a different trap already "
                  "catalogued). This trap requires a format wider than f128.");
    }
    if (!wire_eq(&direct, &via_tri)) {
        char der[512];
        snprintf(der, sizeof(der),
                 "a=RNE_f256(%s); b=RNE_f256(%s); direct=RNE_f256(a*b); "
                 "triple=RNE_f256(RNE_f128(RNE_f64(a)*RNE_f64(b)))",
                 sa, sb);
        emit_case(tag, "triple_round_mul", "f256_mul", &a, &b, &direct, &via_tri,
                  "via_f64_f128_widen", der,
                  "Three successive IEEE roundings on a product; f128 only "
                  "admits a two-step f64 cascade. The third stage is f256-only.");
    }
    if (!wire_eq(&direct, &school)) {
        char der[512];
        snprintf(der, sizeof(der),
                 "a=RNE_f256(%s); b=RNE_f256(%s); direct=RNE_f256(a*b) full sticky; "
                 "wrong=RNE_f256(trunc_to_p(a*b)) with MPFR_RNDZ to p=237 "
                 "(simulates 4-limb schoolbook that discards low product limbs "
                 "without sticky)",
                 sa, sb);
        emit_case(tag, "schoolbook_no_sticky", "f256_mul", &a, &b, &direct,
                  &school, "schoolbook_trunc_p_bits", der,
                  "Significand is 237 bits = 4x64-bit limbs; full schoolbook "
                  "product is 8 limbs. Truncating to 4 limbs without sticky is "
                  "a 4-limb algorithm error. f128 schoolbook is 2x2→4 limbs — "
                  "same class but not the 4-limb significand case.");
    }
}

int main(void) {
    g_ver = mpfr_get_version();
    fprintf(stderr, "f256_only_traps_gen MPFR %s\n", g_ver);
    g_id = 0;

    /* --- Constructed pairs that should stress intermediate widths --- */

    /* Bits past f128 significand: 1 + 2^-120 is exact in f256, collapses in f128 */
    try_pair("past_f128_sig", "0x1.00000000000000000000000000001p+0",
             "0x1.00000000000000000000000000001p+0");

    try_pair("past_f128_sig_2", "0x1.000000000000000000000008p+0",
             "0x1.000000000000000000000008p+0");

    /* Product of two numbers with many low bits */
    try_pair("low_bits_product", "0x1.ffffffffffffffffffffffffffp-1",
             "0x1.ffffffffffffffffffffffffffp-1");

    /* 1+2^-130 times 1+2^-140 */
    try_pair("asymmetric_low", "0x1.00000000000000000000000004p+0",
             "0x1.000000000000000000000000002p+0");

    /* Near 1 with f256-only ulp structure */
    try_pair("near_one_f256_ulp", "0x1.0000000000000000000000000001p+0",
             "0x1.ffffffffffffffffffffffffp-1");

    /* Large exponent with fine mantissa */
    try_pair("scaled_fine", "0x1.0000000000000000000001p+100",
             "0x1.0000000000000000000001p+100");

    try_pair("scaled_fine_2", "0x1.abcdef0123456789abcdef0123p+50",
             "0x1.123456789abcdef0123456789ap-30");

    /* Classic decimals that expand past 113 bits */
    try_pair("decimal_0p1_sq", "0.1", "0.1");
    try_pair("decimal_pi_e", "3.141592653589793238462643383279502884",
             "2.718281828459045235360287471352662497");

    /* Force schoolbook sticky: (1 + 2^-(p-1)) ^ 2 */
    try_pair("one_plus_half_ulp_sq", "0x1.00000000000000000000000000008p+0",
             "0x1.00000000000000000000000000008p+0");

    /* Values needing all 4 limbs */
    try_pair("full_limb_a", "0x1.000000010000000100000001p+0",
             "0x1.000000010000000100000001p+0");

    try_pair("cross_limb", "0x1.ffffffff00000000ffffffffp+0",
             "0x1.00000000ffffffff00000001p+0");

    /* Subnormal-ish products */
    try_pair("small_normals", "0x1p-100000", "0x1p-100000");

    try_pair("one_third_sq",
             "0x1.5555555555555555555555555555p-1",
             "0x1.5555555555555555555555555555p-1");

    fprintf(stderr, "emitted %d f256-only trap vectors\n", g_id);
    if (g_id == 0) {
        fprintf(stderr, "WARNING: no differing paths found — search failed\n");
        return 1;
    }
    return 0;
}
