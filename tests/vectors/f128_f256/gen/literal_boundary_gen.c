/* literal_boundary_gen.c
 *
 * Wave 3 / WS-G — external-oracle reference vectors for V0-B *literal*
 * and value-boundary probes. Complements mpfr_vector_gen.c (arithmetic
 * ops) with cases that answer: "what bit pattern must a correct f128/f256
 * literal parse produce?"
 *
 * Oracle is MPFR, not Sounio. Double-rounding traps compare:
 *   direct:   string → MPFR(high) → RNE → binaryN
 *   via f64:  string → MPFR → RNE → binary64 → (exact) widen → binaryN
 * A widen-f64 parser shortcut that only checks self-consistency will miss
 * exactly these traps.
 *
 * Build:
 *   gcc -O2 -Wall -Wextra -o literal_boundary_gen literal_boundary_gen.c \
 *       $(pkg-config --cflags --libs mpfr) -lgmp
 *
 * Output: JSONL on stdout (one vector per line).
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <mpfr.h>
#include <gmp.h>

#define F128_K 15
#define F128_P 113
#define F128_BIAS 16383
#define F128_EMAX 16383
#define F128_TRAIL 112
#define F128_EMIN (1 - F128_EMAX) /* -16382 */

#define F256_K 19
#define F256_P 237
#define F256_BIAS 262143
#define F256_EMAX 262143
#define F256_TRAIL 236

#define EXT_PREC 2048 /* >> 237; "exact" for our decimal/hex inputs */

typedef struct {
    int sign;
    uint64_t exp_biased;
    uint64_t limbs[4]; /* LE; f128 uses limbs[0..1], f256 uses [0..3] */
    const char *klass; /* zero|subnormal|normal|inf|nan */
} wire_bits;

static const char *MPFR_VERSION_STR = NULL;
static char PROVENANCE_BUF[512];

static void init_provenance(void) {
    MPFR_VERSION_STR = mpfr_get_version();
    snprintf(PROVENANCE_BUF, sizeof(PROVENANCE_BUF),
             "tool=MPFR version=%s rounding=MPFR_RNDN "
             "ext_prec=%d build=gcc+libmpfr+libgmp "
             "generator=tests/vectors/f128_f256/gen/literal_boundary_gen.c",
             MPFR_VERSION_STR, EXT_PREC);
}

/* ---- binary128 encode from MPFR (already at target or higher) ---- */

static void mpfr_to_f128(wire_bits *out, mpfr_t x) {
    memset(out, 0, sizeof(*out));
    if (mpfr_nan_p(x)) {
        out->klass = "nan";
        out->sign = 0;
        out->exp_biased = 0x7FFF;
        /* quiet NaN: top trailing bit set */
        out->limbs[1] = (1ULL << 63) | (1ULL << 47); /* sign=0, exp=all1s high, qNaN */
        /* rebuild properly below */
        out->limbs[0] = 0;
        out->limbs[1] = ((uint64_t)0x7FFF << 48) | (1ULL << 47);
        return;
    }
    if (mpfr_inf_p(x)) {
        out->klass = "inf";
        out->sign = mpfr_signbit(x) ? 1 : 0;
        out->exp_biased = 0x7FFF;
        out->limbs[0] = 0;
        out->limbs[1] = ((uint64_t)out->sign << 63) | ((uint64_t)0x7FFF << 48);
        return;
    }
    if (mpfr_zero_p(x)) {
        out->klass = "zero";
        out->sign = mpfr_signbit(x) ? 1 : 0;
        out->exp_biased = 0;
        out->limbs[0] = 0;
        out->limbs[1] = (uint64_t)out->sign << 63;
        return;
    }

    /* Round to binary128 via MPFR's binary128 format if available.
     * MPFR 4.x: mpfr_set_float128 not always present; use manual RNE via
     * mpfr_get_z_2exp after setting prec=113. */
    mpfr_t y;
    mpfr_init2(y, F128_P);
    mpfr_set(y, x, MPFR_RNDN);

    out->sign = mpfr_signbit(y) ? 1 : 0;
    if (out->sign)
        mpfr_neg(y, y, MPFR_RNDN);

    mpfr_exp_t e = mpfr_get_exp(y); /* exponent of 2^e * 0.1mmm… with m in [0.5,1) in MPFR terms: exp of significant */
    /* MPFR: significand in [0.5, 1) so value = m * 2^e with 0.5 <= m < 1.
     * IEEE normal: 1.f * 2^E with unbiased E = e-1. */
    int64_t E = (int64_t)e - 1;

    if (E > F128_EMAX) {
        /* overflow to inf */
        out->klass = "inf";
        out->exp_biased = 0x7FFF;
        out->limbs[0] = 0;
        out->limbs[1] = ((uint64_t)out->sign << 63) | ((uint64_t)0x7FFF << 48);
        mpfr_clear(y);
        return;
    }

    /* Subnormal / underflow domain: E < emin = -16382 */
    if (E < F128_EMIN) {
        /* gradual underflow: round to 2^(emin-1) * (0.ffff…) scale
         * value = m * 2^e = m * 2^(E+1). For subnormals, encode as
         * 0.frac * 2^emin with frac having 112 bits (no hidden 1). */
        /* Use MPFR subnormalize if available; else scale. */
        mpfr_t z;
        mpfr_init2(z, F128_P + 16);
        /* z = y / 2^emin  → integer part is subnormal trailing+hidden layout */
        mpfr_set(z, y, MPFR_RNDN);
        /* y = significand * 2^E with significand in [1,2). Rebuild: */
        mpfr_mul_2si(z, z, -F128_EMIN, MPFR_RNDN); /* z in units of 2^emin */
        /* Now z should be in (0, 1) for deep underflow or [1, 2) near min normal.
         * Subnormal max is just below 1.0 * 2^emin. */
        if (mpfr_cmp_ui(z, 1) < 0) {
            /* pure subnormal or zero */
            mpfr_mul_2ui(z, z, F128_TRAIL, MPFR_RNDN); /* scale to integer 112-bit */
            mpz_t zi;
            mpz_init(zi);
            mpfr_get_z(zi, z, MPFR_RNDN);
            if (mpz_sgn(zi) == 0) {
                out->klass = "zero";
                out->exp_biased = 0;
                out->limbs[0] = 0;
                out->limbs[1] = (uint64_t)out->sign << 63;
            } else {
                out->klass = "subnormal";
                out->exp_biased = 0;
                /* zi is up to 112 bits */
                uint64_t lo = (uint64_t)mpz_get_ui(zi);
                mpz_tdiv_q_2exp(zi, zi, 64);
                uint64_t hi_t = (uint64_t)mpz_get_ui(zi);
                out->limbs[0] = lo;
                out->limbs[1] = ((uint64_t)out->sign << 63) | (hi_t & 0xFFFFFFFFFFFFULL);
            }
            mpz_clear(zi);
            mpfr_clear(z);
            mpfr_clear(y);
            return;
        }
        mpfr_clear(z);
        /* fell through: min normal */
        E = F128_EMIN;
        mpfr_set_ui_2exp(y, 1, E, MPFR_RNDN); /* 1.0 * 2^emin */
    }

    /* Normal: 1.trailing * 2^E */
    out->klass = "normal";
    out->exp_biased = (uint64_t)(E + F128_BIAS);
    /* Extract 112 trailing bits: y = 1.f * 2^E → frac = y/2^E - 1 */
    mpfr_t frac;
    mpfr_init2(frac, F128_P);
    mpfr_set(frac, y, MPFR_RNDN);
    mpfr_div_2si(frac, frac, E, MPFR_RNDN); /* in [1,2) */
    mpfr_sub_ui(frac, frac, 1, MPFR_RNDN);  /* in [0,1) */
    mpfr_mul_2ui(frac, frac, F128_TRAIL, MPFR_RNDN);
    mpz_t zi;
    mpz_init(zi);
    mpfr_get_z(zi, frac, MPFR_RNDN);
    uint64_t lo = (uint64_t)mpz_get_ui(zi);
    mpz_tdiv_q_2exp(zi, zi, 64);
    uint64_t hi_t = (uint64_t)mpz_get_ui(zi);
    out->limbs[0] = lo;
    out->limbs[1] = ((uint64_t)out->sign << 63)
                  | ((out->exp_biased & 0x7FFFULL) << 48)
                  | (hi_t & 0xFFFFFFFFFFFFULL);
    mpz_clear(zi);
    mpfr_clear(frac);
    mpfr_clear(y);
}

static void mpfr_to_f256(wire_bits *out, mpfr_t x) {
    memset(out, 0, sizeof(*out));
    if (mpfr_nan_p(x)) {
        out->klass = "nan";
        out->sign = 0;
        out->exp_biased = 0x7FFFF;
        out->limbs[3] = ((uint64_t)0x7FFFF << 44) | (1ULL << 43); /* qNaN */
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
        out->exp_biased = 0;
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
    int64_t emin = 1 - F256_EMAX; /* -262142 */

    if (E > F256_EMAX) {
        out->klass = "inf";
        out->exp_biased = 0x7FFFF;
        out->limbs[3] = ((uint64_t)out->sign << 63) | ((uint64_t)0x7FFFF << 44);
        mpfr_clear(y);
        return;
    }

    if (E < emin) {
        mpfr_t z;
        mpfr_init2(z, F256_P + 16);
        mpfr_set(z, y, MPFR_RNDN);
        mpfr_mul_2si(z, z, -emin, MPFR_RNDN);
        if (mpfr_cmp_ui(z, 1) < 0) {
            mpfr_mul_2ui(z, z, F256_TRAIL, MPFR_RNDN);
            mpz_t zi;
            mpz_init(zi);
            mpfr_get_z(zi, z, MPFR_RNDN);
            if (mpz_sgn(zi) == 0) {
                out->klass = "zero";
                out->exp_biased = 0;
                out->limbs[3] = (uint64_t)out->sign << 63;
            } else {
                out->klass = "subnormal";
                out->exp_biased = 0;
                /* 236 bits across limbs 0..3 low 44 */
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
        E = emin;
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
    /* limb3 low 44 bits are trailing; pack exp+sign */
    uint64_t t_hi = out->limbs[3] & 0x000FFFFFFFFFFFFFULL;
    out->limbs[3] = ((uint64_t)out->sign << 63)
                  | ((out->exp_biased & 0x7FFFFULL) << 44)
                  | t_hi;
    mpz_clear(zi);
    mpfr_clear(frac);
    mpfr_clear(y);
}


/* Correct: 48-bit hi_t + 64-bit lo = 112 bits → 12+16 = 28 hex digits */
static void print_trailing_hex_f128_fix(const wire_bits *w) {
    uint64_t hi_t = w->limbs[1] & 0xFFFFFFFFFFFFULL;
    uint64_t lo = w->limbs[0];
    printf("%012" PRIx64 "%016" PRIx64, hi_t, lo);
}

static void print_trailing_hex_f256(const wire_bits *w) {
    /* 236 bits: limb3 low 44 + limb2 + limb1 + limb0 */
    uint64_t t3 = w->limbs[3] & 0x000FFFFFFFFFFFFFULL;
    printf("%011" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64,
           t3, w->limbs[2], w->limbs[1], w->limbs[0]);
}

static void emit_wire_json(const wire_bits *w, int is_f256) {
    printf("{\"class\":\"%s\",\"sign\":%d,\"exponent\":%" PRIu64
           ",\"trailing_hex\":\"",
           w->klass, w->sign, w->exp_biased);
    if (is_f256)
        print_trailing_hex_f256(w);
    else
        print_trailing_hex_f128_fix(w);
    printf("\",\"limbs\":[");
    int n = is_f256 ? 4 : 2;
    for (int i = 0; i < n; i++) {
        if (i)
            printf(",");
        printf("%" PRId64, (int64_t)w->limbs[i]);
    }
    printf("]}");
}

static bool wire_eq(const wire_bits *a, const wire_bits *b, int is_f256) {
    if (a->sign != b->sign || a->exp_biased != b->exp_biased)
        return false;
    int n = is_f256 ? 4 : 2;
    for (int i = 0; i < n; i++)
        if (a->limbs[i] != b->limbs[i])
            return false;
    return true;
}

static void parse_string_to_mpfr(mpfr_t out, const char *s) {
    /* MPFR accepts decimal and hexfloat 0x1.fp+3 */
    int r = mpfr_set_str(out, s, 0, MPFR_RNDN);
    if (r != 0) {
        fprintf(stderr, "mpfr_set_str failed for '%s'\n", s);
        abort();
    }
}

/* Direct RNE to format */
static void direct_f128(wire_bits *out, const char *s) {
    mpfr_t x;
    mpfr_init2(x, EXT_PREC);
    parse_string_to_mpfr(x, s);
    mpfr_to_f128(out, x);
    mpfr_clear(x);
}

static void direct_f256(wire_bits *out, const char *s) {
    mpfr_t x;
    mpfr_init2(x, EXT_PREC);
    parse_string_to_mpfr(x, s);
    mpfr_to_f256(out, x);
    mpfr_clear(x);
}

/* via f64: string → RNE binary64 → exact reinterpret as real → RNE binaryN */
static void via_f64_f128(wire_bits *out, const char *s) {
    mpfr_t x, d;
    mpfr_init2(x, EXT_PREC);
    mpfr_init2(d, 53); /* binary64 precision */
    parse_string_to_mpfr(x, s);
    mpfr_set(d, x, MPFR_RNDN); /* round to f64 */
    /* d is exact f64 value; widen by re-encoding at f128 */
    mpfr_to_f128(out, d);
    mpfr_clear(d);
    mpfr_clear(x);
}

static void via_f64_f256(wire_bits *out, const char *s) {
    mpfr_t x, d;
    mpfr_init2(x, EXT_PREC);
    mpfr_init2(d, 53);
    parse_string_to_mpfr(x, s);
    mpfr_set(d, x, MPFR_RNDN);
    mpfr_to_f256(out, d);
    mpfr_clear(d);
    mpfr_clear(x);
}

static int id_counter = 0;

static void emit_vector(const char *format, const char *family,
                        const char *source_literal, const char *source_kind,
                        const char *notes, int is_f256) {
    wire_bits direct, via;
    if (is_f256) {
        direct_f256(&direct, source_literal);
        via_f64_f256(&via, source_literal);
    } else {
        direct_f128(&direct, source_literal);
        via_f64_f128(&via, source_literal);
    }
    bool differs = !wire_eq(&direct, &via, is_f256);
    id_counter++;
    printf("{\"id\":\"%s_lit_%04d\",\"format\":\"%s\",\"family\":\"%s\","
           "\"source_literal\":\"%s\",\"source_kind\":\"%s\","
           "\"expected\":",
           is_f256 ? "f256" : "f128", id_counter, format, family,
           source_literal, source_kind);
    emit_wire_json(&direct, is_f256);
    printf(",\"via_f64\":");
    emit_wire_json(&via, is_f256);
    printf(",\"double_rounds_differs\":%s,"
           "\"rounding\":\"rne\","
           "\"provenance\":{"
           "\"tool\":\"MPFR\","
           "\"version\":\"%s\","
           "\"rounding_mode\":\"MPFR_RNDN\","
           "\"extended_precision_bits\":%d,"
           "\"invocation\":\"mpfr_set_str(s,0,MPFR_RNDN); "
           "direct: mpfr_set(y,x,MPFR_RNDN) at p=%d; "
           "via_f64: mpfr_set(d,x,MPFR_RNDN) at p=53 then re-encode\","
           "\"generator\":\"tests/vectors/f128_f256/gen/literal_boundary_gen.c\","
           "\"notes\":\"%s\""
           "}}\n",
           differs ? "true" : "false", MPFR_VERSION_STR, EXT_PREC,
           is_f256 ? F256_P : F128_P, notes);
}

/* ---- case tables ---- */

typedef struct {
    const char *lit;
    const char *kind;
    const char *family;
    const char *notes;
} case_t;

static void emit_all_for_format(int is_f256) {
    const char *fmt = is_f256 ? "binary256" : "binary128";

    /* Exactly representable (finite, no rounding ambiguity for these strings) */
    static const case_t exact[] = {
        {"0", "decimal", "exactly_representable", "integer zero"},
        {"-0", "decimal", "exactly_representable", "signed zero"},
        {"1", "decimal", "exactly_representable", "integer one"},
        {"-1", "decimal", "exactly_representable", "integer minus one"},
        {"2", "decimal", "exactly_representable", "integer two"},
        {"0.5", "decimal", "exactly_representable", "dyadic 2^-1"},
        {"0.25", "decimal", "exactly_representable", "dyadic 2^-2"},
        {"0.125", "decimal", "exactly_representable", "dyadic 2^-3"},
        {"1.0", "decimal", "exactly_representable", "decimal one with point"},
        {"2.0", "decimal", "exactly_representable", "decimal two"},
        {"0x1p+0", "hexfloat", "exactly_representable", "hexfloat 1.0"},
        {"0x1.0p+0", "hexfloat", "exactly_representable", "hexfloat 1.0 trailing"},
        {"0x1p+1", "hexfloat", "exactly_representable", "hexfloat 2.0"},
        {"0x1p-1", "hexfloat", "exactly_representable", "hexfloat 0.5"},
        {"0x1.8p+0", "hexfloat", "exactly_representable", "1.5 exactly"},
        {"0x1p-10", "hexfloat", "exactly_representable", "2^-10"},
        {"0x1p+10", "hexfloat", "exactly_representable", "2^10=1024"},
    };
    for (size_t i = 0; i < sizeof(exact) / sizeof(exact[0]); i++)
        emit_vector(fmt, exact[i].family, exact[i].lit, exact[i].kind,
                    exact[i].notes, is_f256);

    /* Provably not exactly representable (need RNE) — decimal non-dyadic */
    static const case_t not_exact[] = {
        {"0.1", "decimal", "provably_not_representable",
         "1/10 not dyadic; RNE to format"},
        {"0.2", "decimal", "provably_not_representable", "1/5"},
        {"0.3", "decimal", "provably_not_representable", "3/10"},
        {"0.1", "decimal", "provably_not_representable", "repeat for id stability"},
        {"1.1", "decimal", "provably_not_representable", "11/10"},
        {"3.14159265358979323846264338327950288", "decimal",
         "provably_not_representable", "pi digits beyond any binary format"},
        {"2.71828182845904523536028747135266249", "decimal",
         "provably_not_representable", "e digits"},
    };
    for (size_t i = 0; i < sizeof(not_exact) / sizeof(not_exact[0]); i++) {
        if (i == 3)
            continue; /* skip accidental duplicate entry above */
        emit_vector(fmt, not_exact[i].family, not_exact[i].lit, not_exact[i].kind,
                    not_exact[i].notes, is_f256);
    }

    /* Subnormals: 2^(emin-1) and min subnormal 2^(emin - (p-1)) */
    if (!is_f256) {
        /* emin = -16382; min subnormal = 2^(-16494); max subnormal just below 2^-16382 */
        emit_vector(fmt, "subnormal", "0x1p-16494", "hexfloat",
                    "binary128 minimum positive subnormal", 0);
        emit_vector(fmt, "subnormal", "0x1p-16493", "hexfloat",
                    "2 * min subnormal", 0);
        emit_vector(fmt, "subnormal", "0x1p-16383", "hexfloat",
                    "near subnormal range high", 0);
        emit_vector(fmt, "subnormal", "-0x1p-16494", "hexfloat",
                    "negative min subnormal", 0);
    } else {
        /* emin = -262142; min subnormal 2^(emin-(p-1)) = 2^(-262142-236) = 2^-262378 */
        emit_vector(fmt, "subnormal", "0x1p-262378", "hexfloat",
                    "binary256 minimum positive subnormal", 1);
        emit_vector(fmt, "subnormal", "0x1p-262377", "hexfloat",
                    "2 * min subnormal", 1);
        emit_vector(fmt, "subnormal", "-0x1p-262378", "hexfloat",
                    "negative min subnormal", 1);
    }

    /* Smallest normal */
    if (!is_f256) {
        emit_vector(fmt, "min_normal", "0x1p-16382", "hexfloat",
                    "binary128 smallest positive normal", 0);
        emit_vector(fmt, "min_normal", "-0x1p-16382", "hexfloat",
                    "binary128 smallest magnitude negative normal", 0);
    } else {
        emit_vector(fmt, "min_normal", "0x1p-262142", "hexfloat",
                    "binary256 smallest positive normal", 1);
        emit_vector(fmt, "min_normal", "-0x1p-262142", "hexfloat",
                    "binary256 neg min normal", 1);
    }

    /* Largest finite: (2 - 2^(1-p)) * 2^emax */
    if (!is_f256) {
        emit_vector(fmt, "max_finite", "0x1.ffffffffffffffffffffffffffffp+16383",
                    "hexfloat",
                    "binary128 max finite (all 112 trailing bits 1)", 0);
        emit_vector(fmt, "max_finite", "-0x1.ffffffffffffffffffffffffffffp+16383",
                    "hexfloat", "binary128 most negative finite", 0);
    } else {
        /* 236 trailing ones — hex length: 236/4 = 59 hex digits */
        emit_vector(fmt, "max_finite",
                    "0x1.ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffp+262143",
                    "hexfloat",
                    "binary256 max finite (236 trailing ones)", 1);
    }

    /* ULP either side of 1.0 */
    if (!is_f256) {
        /* ulp(1.0) in f128 = 2^-112 */
        emit_vector(fmt, "ulp_neighbors", "0x1p+0", "hexfloat",
                    "1.0 anchor", 0);
        emit_vector(fmt, "ulp_neighbors", "0x1.0000000000000000000000000001p+0",
                    "hexfloat", "1.0 + 1 ulp (trailing bit 0 set)", 0);
        emit_vector(fmt, "ulp_neighbors", "0x1.fffffffffffffp-1", "hexfloat",
                    "next below 1.0 toward 0.5 (max below 1 as 0.111... * 2^0)", 0);
        emit_vector(fmt, "ulp_neighbors", "0x1p-112", "hexfloat",
                    "ulp(1.0) as a standalone value", 0);
    } else {
        emit_vector(fmt, "ulp_neighbors", "0x1p+0", "hexfloat", "1.0 anchor", 1);
        emit_vector(fmt, "ulp_neighbors", "0x1p-236", "hexfloat",
                    "ulp(1.0) in binary256", 1);
    }

    /* Double-rounding traps: more than 53 bits of significance so f64
     * intermediate loses bits that f128/f256 keep. */
    static const case_t traps[] = {
        {"0.1", "decimal", "double_rounding_trap",
         "classic 0.1; f64 and f128 expansions differ"},
        {"0.3", "decimal", "double_rounding_trap",
         "0.3 double-round trap"},
        {"1e-20", "decimal", "double_rounding_trap",
         "scientific decimal outside exact f64 dyadic set"},
        {"1.0000000000000002", "decimal", "double_rounding_trap",
         "near 1+eps; many digits force f64 RNE before widen"},
        {"3.14159265358979323846", "decimal", "double_rounding_trap",
         "pi truncated; f64 vs wide RNE differ"},
        {"0x1.00000000000008p+0", "hexfloat", "double_rounding_trap",
         "hex with bits past f64 mantissa (53 bits)"},
        {"0x1.00000000000001p+0", "hexfloat", "double_rounding_trap",
         "1 + 2^-52; boundary of f64 ulp at 1.0"},
        {"0x1.000000000000001p+0", "hexfloat", "double_rounding_trap",
         "1 + 2^-56; invisible to f64, visible to f128"},
        {"0x1.fffffffffffffp+1023", "hexfloat", "double_rounding_trap",
         "near f64 max; wide format still finite normal"},
        {"9.999999999999999e-1", "decimal", "double_rounding_trap",
         "just below 1.0 with long decimal"},
    };
    for (size_t i = 0; i < sizeof(traps) / sizeof(traps[0]); i++)
        emit_vector(fmt, traps[i].family, traps[i].lit, traps[i].kind,
                    traps[i].notes, is_f256);

    /* Literal boundary spelling variants */
    static const case_t boundaries[] = {
        {"1e0", "decimal", "literal_boundary", "scientific integer exponent"},
        {"1E0", "decimal", "literal_boundary", "capital E"},
        {"1.0e+0", "decimal", "literal_boundary", "explicit + exponent"},
        {"1.0e-0", "decimal", "literal_boundary", "exponent -0"},
        {"0x1.0p+0", "hexfloat", "literal_boundary", "hex with trailing .0"},
        {"0X1.0P+0", "hexfloat", "literal_boundary", "capital hex markers"},
        {"0x0.1p+4", "hexfloat", "literal_boundary", "equiv 1.0 via exponent"},
        {"0x2p-1", "hexfloat", "literal_boundary", "equiv 1.0"},
    };
    for (size_t i = 0; i < sizeof(boundaries) / sizeof(boundaries[0]); i++)
        emit_vector(fmt, boundaries[i].family, boundaries[i].lit,
                    boundaries[i].kind, boundaries[i].notes, is_f256);
}

int main(void) {
    init_provenance();
    fprintf(stderr, "literal_boundary_gen: %s\n", PROVENANCE_BUF);
    id_counter = 0;
    emit_all_for_format(0); /* f128 */
    int f128_n = id_counter;
    emit_all_for_format(1); /* f256 */
    fprintf(stderr, "emitted f128=%d f256=%d total=%d vectors\n", f128_n,
            id_counter - f128_n, id_counter);
    return 0;
}
