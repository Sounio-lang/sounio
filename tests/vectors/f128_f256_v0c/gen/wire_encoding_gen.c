/* wire_encoding_gen.c
 *
 * V0-C wire-format / limb-pool corpus — encodings, not arithmetic values.
 * Authored before any limb-pool implementation has an interest in the answer.
 *
 * Families:
 *   valid_edge       — IEEE class edges (0, subnormal, normal, inf, nan)
 *   limb_boundary    — single bits at LSW/MSW and cross-limb edges
 *   class_consistent — class field matches (exp,trailing) derivation rules
 *   malformed_reject — truncated, wrong limb arity, illegal sign/exp, garbage
 *
 * Valid vectors carry a full wire object. Reject vectors carry a partial
 * encoding plus reject_reason; a correct decoder MUST fail closed.
 *
 * Provenance: IEEE 754-2008 §3.6 layout constants (same as
 * f128_f256_format_descriptor_probe) + generator path. No Sounio.
 *
 * Build: gcc -O2 -Wall -Wextra -o wire_encoding_gen wire_encoding_gen.c
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

#define F128_EMAX 16383
#define F128_TRAIL 112
#define F128_ALL1_EXP ((uint64_t)(2 * F128_EMAX + 1)) /* 32767 */

#define F256_EMAX 262143
#define F256_TRAIL 236
#define F256_ALL1_EXP ((uint64_t)(2 * F256_EMAX + 1)) /* 524287 */

static int g_id;
static const char *GEN =
    "tests/vectors/f128_f256_v0c/gen/wire_encoding_gen.c";
static const char *CITATION =
    "IEEE 754-2008 §3.6 binaryN; matches self-hosted/compiler "
    "f128_f256_format_descriptor_probe.sio (K/P/bias/emax)";

static void emit_prov(const char *invocation, const char *notes) {
    printf("\"provenance\":{"
           "\"tool\":\"structural-ieee754\","
           "\"version\":\"IEEE-754-2008\","
           "\"rounding_mode\":\"n/a-encoding\","
           "\"invocation\":\"%s\","
           "\"generator\":\"%s\","
           "\"citation\":\"%s\","
           "\"notes\":\"%s\""
           "}",
           invocation, GEN, CITATION, notes);
}

static void emit_accept(const char *fmt, int is_f256, const char *family,
                        const char *klass, int sign, uint64_t exp,
                        const char *trail_hex, const int64_t *limbs, int nlimbs,
                        const char *notes) {
    g_id++;
    printf("{\"id\":\"%s_wire_%04d\",\"format\":\"%s\",\"family\":\"%s\","
           "\"verdict\":\"accept\",\"reject_reason\":null,"
           "\"encoding\":{\"class\":\"%s\",\"sign\":%d,\"exponent\":%" PRIu64
           ",\"trailing_hex\":\"%s\",\"limbs\":[",
           is_f256 ? "f256" : "f128", g_id, fmt, family, klass, sign, exp,
           trail_hex);
    for (int i = 0; i < nlimbs; i++) {
        if (i)
            printf(",");
        printf("%" PRId64, limbs[i]);
    }
    printf("]},");
    emit_prov("hand-constructed IEEE wire from format parameters", notes);
    printf("}\n");
}

static void emit_reject(const char *fmt, int is_f256, const char *family,
                        const char *reason, const char *encoding_json,
                        const char *notes) {
    g_id++;
    printf("{\"id\":\"%s_wire_%04d\",\"format\":\"%s\",\"family\":\"%s\","
           "\"verdict\":\"reject\",\"reject_reason\":\"%s\","
           "\"encoding\":%s,",
           is_f256 ? "f256" : "f128", g_id, fmt, family, reason, encoding_json);
    emit_prov("negative structural case for fail-closed limb decode", notes);
    printf("}\n");
}

/* Pack f128: limbs LE [lo, hi] with hi = sign|exp|top48 trailing */
static void f128_pack(int sign, uint64_t exp, uint64_t trail_hi48, uint64_t trail_lo,
                      int64_t limbs[2], char *hex28) {
    limbs[0] = (int64_t)trail_lo;
    limbs[1] = (int64_t)(((uint64_t)sign << 63) | ((exp & 0x7FFFULL) << 48) |
                         (trail_hi48 & 0xFFFFFFFFFFFFULL));
    snprintf(hex28, 29, "%012" PRIx64 "%016" PRIx64, trail_hi48 & 0xFFFFFFFFFFFFULL,
             trail_lo);
}

static void f256_pack(int sign, uint64_t exp, uint64_t l0, uint64_t l1, uint64_t l2,
                      uint64_t trail_hi44, int64_t limbs[4], char *hex) {
    limbs[0] = (int64_t)l0;
    limbs[1] = (int64_t)l1;
    limbs[2] = (int64_t)l2;
    limbs[3] = (int64_t)(((uint64_t)sign << 63) | ((exp & 0x7FFFFULL) << 44) |
                         (trail_hi44 & 0x000FFFFFFFFFFFFFULL));
    snprintf(hex, 64, "%011" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64,
             trail_hi44 & 0x000FFFFFFFFFFFFFULL, l2, l1, l0);
}

static void emit_f128_valid(void) {
    const char *fmt = "binary128";
    int64_t L[2];
    char hx[40];

    /* +0 */
    f128_pack(0, 0, 0, 0, L, hx);
    emit_accept(fmt, 0, "valid_edge", "zero", 0, 0, hx, L, 2, "+0");
    /* -0 */
    f128_pack(1, 0, 0, 0, L, hx);
    emit_accept(fmt, 0, "valid_edge", "zero", 1, 0, hx, L, 2, "-0");
    /* min subnormal: trailing=1 */
    f128_pack(0, 0, 0, 1, L, hx);
    emit_accept(fmt, 0, "valid_edge", "subnormal", 0, 0, hx, L, 2,
                "min positive subnormal");
    /* max subnormal: all 112 trailing ones */
    f128_pack(0, 0, 0xFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, L, hx);
    emit_accept(fmt, 0, "valid_edge", "subnormal", 0, 0, hx, L, 2,
                "max subnormal");
    /* min normal: exp=1, trail=0 */
    f128_pack(0, 1, 0, 0, L, hx);
    emit_accept(fmt, 0, "valid_edge", "normal", 0, 1, hx, L, 2, "min normal");
    /* 1.0 */
    f128_pack(0, 16383, 0, 0, L, hx);
    emit_accept(fmt, 0, "valid_edge", "normal", 0, 16383, hx, L, 2, "1.0");
    /* -1.0 */
    f128_pack(1, 16383, 0, 0, L, hx);
    emit_accept(fmt, 0, "valid_edge", "normal", 1, 16383, hx, L, 2, "-1.0");
    /* max finite: exp=32766, all trailing 1 */
    f128_pack(0, 32766, 0xFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, L, hx);
    emit_accept(fmt, 0, "valid_edge", "normal", 0, 32766, hx, L, 2, "max finite");
    /* +inf */
    f128_pack(0, F128_ALL1_EXP, 0, 0, L, hx);
    emit_accept(fmt, 0, "valid_edge", "inf", 0, F128_ALL1_EXP, hx, L, 2, "+inf");
    /* -inf */
    f128_pack(1, F128_ALL1_EXP, 0, 0, L, hx);
    emit_accept(fmt, 0, "valid_edge", "inf", 1, F128_ALL1_EXP, hx, L, 2, "-inf");
    /* qNaN: top trailing bit set */
    f128_pack(0, F128_ALL1_EXP, 1ULL << 47, 0, L, hx);
    emit_accept(fmt, 0, "valid_edge", "nan", 0, F128_ALL1_EXP, hx, L, 2,
                "canonical qNaN (quiet bit)");
    /* sNaN: exp all1, trailing nonzero, quiet bit clear */
    f128_pack(0, F128_ALL1_EXP, 0, 1, L, hx);
    emit_accept(fmt, 0, "valid_edge", "snan", 0, F128_ALL1_EXP, hx, L, 2,
                "sNaN payload=1");

    /* limb boundaries */
    f128_pack(0, 16383, 0, 1, L, hx); /* bit 0 of LSW */
    emit_accept(fmt, 0, "limb_boundary", "normal", 0, 16383, hx, L, 2,
                "LSB of limb0 set (1+ulp-scale trail bit0 at exp 1.0 field)");
    f128_pack(0, 16383, 0, 1ULL << 63, L, hx);
    emit_accept(fmt, 0, "limb_boundary", "normal", 0, 16383, hx, L, 2,
                "MSB of limb0 set");
    f128_pack(0, 16383, 1, 0, L, hx); /* bit 64 of trailing */
    emit_accept(fmt, 0, "limb_boundary", "normal", 0, 16383, hx, L, 2,
                "LSB of trail_hi (cross-limb bit 64)");
    f128_pack(0, 16383, 1ULL << 47, 0, L, hx); /* top trailing bit */
    emit_accept(fmt, 0, "limb_boundary", "normal", 0, 16383, hx, L, 2,
                "MSB of 112-bit trailing field");

    /* class consistency accept: class matches derivation */
    f128_pack(0, 0, 0, 2, L, hx);
    emit_accept(fmt, 0, "class_consistent", "subnormal", 0, 0, hx, L, 2,
                "class=subnormal with exp=0 trail!=0");
}

static void emit_f256_valid(void) {
    const char *fmt = "binary256";
    int64_t L[4];
    char hx[80];

    f256_pack(0, 0, 0, 0, 0, 0, L, hx);
    emit_accept(fmt, 1, "valid_edge", "zero", 0, 0, hx, L, 4, "+0");
    f256_pack(1, 0, 0, 0, 0, 0, L, hx);
    emit_accept(fmt, 1, "valid_edge", "zero", 1, 0, hx, L, 4, "-0");
    f256_pack(0, 0, 1, 0, 0, 0, L, hx);
    emit_accept(fmt, 1, "valid_edge", "subnormal", 0, 0, hx, L, 4,
                "min positive subnormal");
    f256_pack(0, 0, ~0ULL, ~0ULL, ~0ULL, 0x000FFFFFFFFFFFFFULL, L, hx);
    emit_accept(fmt, 1, "valid_edge", "subnormal", 0, 0, hx, L, 4, "max subnormal");
    f256_pack(0, 1, 0, 0, 0, 0, L, hx);
    emit_accept(fmt, 1, "valid_edge", "normal", 0, 1, hx, L, 4, "min normal");
    f256_pack(0, 262143, 0, 0, 0, 0, L, hx);
    emit_accept(fmt, 1, "valid_edge", "normal", 0, 262143, hx, L, 4, "1.0");
    f256_pack(0, 524286, ~0ULL, ~0ULL, ~0ULL, 0x000FFFFFFFFFFFFFULL, L, hx);
    emit_accept(fmt, 1, "valid_edge", "normal", 0, 524286, hx, L, 4, "max finite");
    f256_pack(0, F256_ALL1_EXP, 0, 0, 0, 0, L, hx);
    emit_accept(fmt, 1, "valid_edge", "inf", 0, F256_ALL1_EXP, hx, L, 4, "+inf");
    f256_pack(1, F256_ALL1_EXP, 0, 0, 0, 0, L, hx);
    emit_accept(fmt, 1, "valid_edge", "inf", 1, F256_ALL1_EXP, hx, L, 4, "-inf");
    f256_pack(0, F256_ALL1_EXP, 0, 0, 0, 1ULL << 43, L, hx);
    emit_accept(fmt, 1, "valid_edge", "nan", 0, F256_ALL1_EXP, hx, L, 4, "qNaN");
    f256_pack(0, F256_ALL1_EXP, 1, 0, 0, 0, L, hx);
    emit_accept(fmt, 1, "valid_edge", "snan", 0, F256_ALL1_EXP, hx, L, 4, "sNaN");

    f256_pack(0, 262143, 1, 0, 0, 0, L, hx);
    emit_accept(fmt, 1, "limb_boundary", "normal", 0, 262143, hx, L, 4,
                "LSB limb0");
    f256_pack(0, 262143, 0, 1, 0, 0, L, hx);
    emit_accept(fmt, 1, "limb_boundary", "normal", 0, 262143, hx, L, 4,
                "LSB limb1");
    f256_pack(0, 262143, 0, 0, 1, 0, L, hx);
    emit_accept(fmt, 1, "limb_boundary", "normal", 0, 262143, hx, L, 4,
                "LSB limb2");
    f256_pack(0, 262143, 0, 0, 0, 1, L, hx);
    emit_accept(fmt, 1, "limb_boundary", "normal", 0, 262143, hx, L, 4,
                "LSB of trail_hi in limb3");
    f256_pack(0, 262143, 0, 0, 0, 1ULL << 43, L, hx);
    emit_accept(fmt, 1, "limb_boundary", "normal", 0, 262143, hx, L, 4,
                "MSB of 236-bit trailing (bit 235)");
}

static void emit_f128_reject(void) {
    const char *fmt = "binary128";
    /* wrong limb count */
    emit_reject(fmt, 0, "malformed_reject", "limb_count",
                "{\"class\":\"normal\",\"sign\":0,\"exponent\":16383,"
                "\"trailing_hex\":\"0000000000000000000000000000\","
                "\"limbs\":[0]}",
                "f128 requires exactly 2 limbs");
    emit_reject(fmt, 0, "malformed_reject", "limb_count",
                "{\"class\":\"normal\",\"sign\":0,\"exponent\":16383,"
                "\"trailing_hex\":\"0000000000000000000000000000\","
                "\"limbs\":[0,0,0]}",
                "f128 requires exactly 2 limbs (got 3)");
    /* truncated trailing hex */
    emit_reject(fmt, 0, "malformed_reject", "truncated_trailing_hex",
                "{\"class\":\"normal\",\"sign\":0,\"exponent\":16383,"
                "\"trailing_hex\":\"abc\",\"limbs\":[0,4611404543450677248]}",
                "trailing_hex shorter than 28 hex digits");
    /* missing fields */
    emit_reject(fmt, 0, "malformed_reject", "missing_field",
                "{\"class\":\"normal\",\"sign\":0,\"limbs\":[0,0]}",
                "exponent and trailing_hex absent");
    /* illegal sign */
    emit_reject(fmt, 0, "malformed_reject", "illegal_sign",
                "{\"class\":\"zero\",\"sign\":2,\"exponent\":0,"
                "\"trailing_hex\":\"0000000000000000000000000000\","
                "\"limbs\":[0,0]}",
                "sign must be 0 or 1");
    /* exponent out of range */
    emit_reject(fmt, 0, "malformed_reject", "exponent_range",
                "{\"class\":\"normal\",\"sign\":0,\"exponent\":40000,"
                "\"trailing_hex\":\"0000000000000000000000000000\","
                "\"limbs\":[0,0]}",
                "biased exponent > 32767");
    /* class claims normal but exp=0 trail=0 → zero */
    emit_reject(fmt, 0, "malformed_reject", "class_mismatch",
                "{\"class\":\"normal\",\"sign\":0,\"exponent\":0,"
                "\"trailing_hex\":\"0000000000000000000000000000\","
                "\"limbs\":[0,0]}",
                "class=normal but exp=0 trail=0 is zero");
    /* class claims zero but trail nonzero */
    emit_reject(fmt, 0, "malformed_reject", "class_mismatch",
                "{\"class\":\"zero\",\"sign\":0,\"exponent\":0,"
                "\"trailing_hex\":\"0000000000000000000000000001\","
                "\"limbs\":[1,0]}",
                "class=zero but trailing nonzero");
    /* class claims inf but trail nonzero */
    emit_reject(fmt, 0, "malformed_reject", "class_mismatch",
                "{\"class\":\"inf\",\"sign\":0,\"exponent\":32767,"
                "\"trailing_hex\":\"0000000000000000000000000001\","
                "\"limbs\":[1,-4611686018427387904]}",
                "class=inf requires trail=0");
    /* class claims nan but trail=0 */
    emit_reject(fmt, 0, "malformed_reject", "class_mismatch",
                "{\"class\":\"nan\",\"sign\":0,\"exponent\":32767,"
                "\"trailing_hex\":\"0000000000000000000000000000\","
                "\"limbs\":[0,-4611686018427387904]}",
                "class=nan requires trail!=0");
    /* non-hex trailing */
    emit_reject(fmt, 0, "malformed_reject", "non_hex_trailing",
                "{\"class\":\"normal\",\"sign\":0,\"exponent\":16383,"
                "\"trailing_hex\":\"xxxxxxxxxxxxxxxxxxxxxxxxxxxx\","
                "\"limbs\":[0,4611404543450677248]}",
                "trailing_hex must be hex");
    /* empty limbs */
    emit_reject(fmt, 0, "malformed_reject", "empty_limbs",
                "{\"class\":\"normal\",\"sign\":0,\"exponent\":16383,"
                "\"trailing_hex\":\"0000000000000000000000000000\","
                "\"limbs\":[]}",
                "empty limbs array");
    /* limbs/trailing disagree (decoder must detect inconsistency if both given) */
    emit_reject(fmt, 0, "malformed_reject", "limbs_trailing_disagree",
                "{\"class\":\"normal\",\"sign\":0,\"exponent\":16383,"
                "\"trailing_hex\":\"0000000000000000000000000001\","
                "\"limbs\":[0,4611404543450677248]}",
                "trailing claims bit0 set but limbs are exact 1.0");
    /* high garbage: if a third semantic field claimed bits above 128 */
    emit_reject(fmt, 0, "malformed_reject", "overwidth",
                "{\"class\":\"normal\",\"sign\":0,\"exponent\":16383,"
                "\"trailing_hex\":\"0000000000000000000000000000\","
                "\"limbs\":[0,4611404543450677248],"
                "\"extra_limb\":1}",
                "unknown extra_limb field / overwidth payload");
}

static void emit_f256_reject(void) {
    const char *fmt = "binary256";
    emit_reject(fmt, 1, "malformed_reject", "limb_count",
                "{\"class\":\"normal\",\"sign\":0,\"exponent\":262143,"
                "\"trailing_hex\":\"00000000000000000000000000000000000000000000000000000000000\","
                "\"limbs\":[0,0]}",
                "f256 requires exactly 4 limbs");
    emit_reject(fmt, 1, "malformed_reject", "limb_count",
                "{\"class\":\"normal\",\"sign\":0,\"exponent\":262143,"
                "\"trailing_hex\":\"00000000000000000000000000000000000000000000000000000000000\","
                "\"limbs\":[0,0,0,0,0]}",
                "f256 requires exactly 4 limbs (got 5)");
    emit_reject(fmt, 1, "malformed_reject", "truncated_trailing_hex",
                "{\"class\":\"normal\",\"sign\":0,\"exponent\":262143,"
                "\"trailing_hex\":\"00\",\"limbs\":[0,0,0,4611686018427387904]}",
                "trailing_hex truncated for 236-bit field");
    emit_reject(fmt, 1, "malformed_reject", "exponent_range",
                "{\"class\":\"normal\",\"sign\":0,\"exponent\":999999,"
                "\"trailing_hex\":\"00000000000000000000000000000000000000000000000000000000000\","
                "\"limbs\":[0,0,0,0]}",
                "biased exponent > 524287");
    emit_reject(fmt, 1, "malformed_reject", "class_mismatch",
                "{\"class\":\"subnormal\",\"sign\":0,\"exponent\":1,"
                "\"trailing_hex\":\"00000000000000000000000000000000000000000000000000000000000\","
                "\"limbs\":[0,0,0,17592186044416]}",
                "class=subnormal requires exp=0");
    emit_reject(fmt, 1, "malformed_reject", "illegal_sign",
                "{\"class\":\"zero\",\"sign\":-1,\"exponent\":0,"
                "\"trailing_hex\":\"00000000000000000000000000000000000000000000000000000000000\","
                "\"limbs\":[0,0,0,0]}",
                "sign must be 0 or 1");
    emit_reject(fmt, 1, "malformed_reject", "missing_field",
                "{\"sign\":0,\"exponent\":0}",
                "class/limbs/trailing_hex missing");
    emit_reject(fmt, 1, "malformed_reject", "empty_limbs",
                "{\"class\":\"zero\",\"sign\":0,\"exponent\":0,"
                "\"trailing_hex\":\"00000000000000000000000000000000000000000000000000000000000\","
                "\"limbs\":[]}",
                "empty limbs");
}

int main(void) {
    fprintf(stderr, "wire_encoding_gen: IEEE-754-2008 structural encodings\n");
    g_id = 0;
    emit_f128_valid();
    int a = g_id;
    emit_f128_reject();
    int b = g_id;
    emit_f256_valid();
    int c = g_id;
    emit_f256_reject();
    fprintf(stderr, "emitted f128_valid+reject through %d, f256 through %d, total %d\n",
            b, g_id, g_id);
    (void)a;
    (void)c;
    return 0;
}
