/*
 * l9_zd_census_fast.c — fast exact level-9 (512-dim) canonical zero-divisor
 * census with exact independent verification.
 *
 * Companion to:
 *   docs/research/l9_zd_census_spec_2026-07-26.md
 *   docs/research/l9_zd_census_falsifiers_2026-07-26.md
 *   scripts/research/l8_zd_census_fast.c           (L8 parent harness)
 *   scripts/ci/l9_zd_census_gate.sh                (CI gate)
 *
 * Two exact methods, sharing only the Cayley-Dickson sign table S:
 *
 *   Method 1 (census, the audited 2-cycle criterion of
 *   scripts/research/routon_zd_contract.py): for 1 <= i < j < 512,
 *   l = i^j, p(k) = S[i,k]*S[j,k]*S[i,k^l]*S[j,k^l] in {+1,-1};
 *   (i,j) is a canonical ZD pair iff some p(k) = +1, and
 *   nullity(L_a) = #{k : p(k)=+1}/2 exactly (both signs sgn simultaneously).
 *
 *   Method 2 (verifier, generic exact linear algebra): build the 512x512
 *   matrix M(sgn) = I + sgn*Q over GF(65521), Q[k][k^l] = S[i,k]*S[j,k^l],
 *   and compute its rank by Gaussian elimination with partial pivoting.
 *   rank_GF(p)(M) = rank_Q(M) for every odd prime p, because M decomposes
 *   into 2x2 blocks [[1, sgn*q'], [sgn*q, 1]] whose rank (1 iff q*q' = +1,
 *   else 2) is the same over any field of characteristic != 2.  So Method 2
 *   is an exact Q-rank computation, not a modular heuristic.  It uses no
 *   closed-form nullity formula — only generic GE — and therefore audits
 *   Method 1 independently, for BOTH signs and for every candidate pair
 *   (nullity 0 expected for non-ZD pairs).
 *
 * New at this level: the census is additionally checked against the solved
 * nullity-histogram counting law of
 *   docs/research/nullity_histogram_law_spec_2026-07-26.md
 * whose level-9 prediction (its live falsification target) is the
 * multiplicity multiset
 *   {1344x32, 2016x16, 2736x8, 3480x4, 4236x2, 4998x1},
 * i.e. mu_s(9) = 2^(9-s+1)*c0(s-1) attained by exactly 2^(9-s) distinct
 * nullity values for s = 4..9, with c0(b) = 3*(2b-3)*2^(b-2) + 3.
 *
 * Build: cc -O2 -o l9_zd_census_fast l9_zd_census_fast.c
 * Fast census-only build (skips Method 2):
 *        cc -O2 -DL9_SKIP_VERIFY -o l9_zd_census_fast l9_zd_census_fast.c
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BITS 9
#define N (1 << BITS)
#define PRIME 65521u /* largest 16-bit prime; odd, so char != 2 */

static int8_t S[N][N];

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Cayley-Dickson sign of e_a * e_b at level `bits`; mirrors cds() in
 * scripts/research/routon_zd_contract.py exactly. */
static int cds(int a, int b, int bits) {
    int s = 1;
    while (bits > 0) {
        if (a == 0 || b == 0) return s;
        if (bits == 1) return -s;
        int h = 1 << (bits - 1);
        int ah = a >= h, bh = b >= h;
        int al = a & (h - 1), bl = b & (h - 1);
        if (!ah && !bh)      { a = al; b = bl; }
        else if (!ah && bh)  { a = bl; b = al; }
        else if (ah && !bh)  { if (bl == 0) { a = al; b = 0; }
                               else { a = al; b = bl; s = -s; } }
        else                 { if (bl == 0) { a = 0; b = al; s = -s; }
                               else { a = bl; b = al; } }
        bits--;
    }
    return s;
}

static void build_sign_table(void) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            S[i][j] = (int8_t)cds(i, j, BITS);
}

/* FNV-1a 64-bit over the raw int8 table, for cross-implementation audit
 * against the NumPy sign table of the Python baseline. */
static uint64_t sign_table_hash(void) {
    uint64_t h = 1469598103934665603ULL;
    const uint8_t *p = (const uint8_t *)S;
    for (size_t k = 0; k < sizeof(S); k++) {
        h ^= p[k];
        h *= 1099511628211ULL;
    }
    return h;
}

#ifndef L9_SKIP_VERIFY
/* Exact rank over GF(PRIME) by Gaussian elimination with partial pivoting.
 * M is N x N row-major, destroyed in place. */
static uint16_t Mbuf[N * N];

static uint32_t modinv(uint32_t a) {
    /* PRIME is prime: a^(PRIME-2) mod PRIME */
    uint64_t base = a, acc = 1;
    uint32_t e = PRIME - 2;
    while (e) {
        if (e & 1) acc = acc * base % PRIME;
        base = base * base % PRIME;
        e >>= 1;
    }
    return (uint32_t)acc;
}

static int rank_gfp(void) {
    int rank = 0;
    for (int col = 0; col < N && rank < N; col++) {
        int piv = -1;
        for (int r = rank; r < N; r++)
            if (Mbuf[r * N + col]) { piv = r; break; }
        if (piv < 0) continue;
        if (piv != rank) {
            for (int c = col; c < N; c++) {
                uint16_t t = Mbuf[rank * N + c];
                Mbuf[rank * N + c] = Mbuf[piv * N + c];
                Mbuf[piv * N + c] = t;
            }
        }
        uint32_t inv = modinv(Mbuf[rank * N + col]);
        for (int r = rank + 1; r < N; r++) {
            uint32_t v = Mbuf[r * N + col];
            if (!v) continue;
            uint32_t f = (uint32_t)((uint64_t)v * inv % PRIME);
            for (int c = col; c < N; c++) {
                uint32_t sub = (uint32_t)((uint64_t)f * Mbuf[rank * N + c] % PRIME);
                uint32_t val = Mbuf[r * N + c] + PRIME - sub;
                Mbuf[r * N + c] = (uint16_t)(val >= PRIME ? val - PRIME : val);
            }
        }
        rank++;
    }
    return rank;
}

/* Load M(sgn) = I + sgn*Q, Q[k][k^l] = S[i][k]*S[j][k^l], into Mbuf. */
static void load_matrix(int i, int j, int sgn) {
    int l = i ^ j;
    memset(Mbuf, 0, sizeof(Mbuf));
    for (int k = 0; k < N; k++) {
        Mbuf[k * N + k] = 1;
        int q = S[i][k] * S[j][k ^ l]; /* in {+1,-1} */
        Mbuf[k * N + (k ^ l)] = (uint16_t)(sgn * q > 0 ? 1 : PRIME - 1);
    }
}
#endif /* L9_SKIP_VERIFY */

/* c0(b) = number of invertible canonical candidate pairs at level b. */
static uint64_t c0(int b) {
    return 3ULL * (uint64_t)(2 * b - 3) * (1ULL << (b - 2)) + 3ULL;
}

int main(void) {
    double t0 = now_sec();

    build_sign_table();
    double t1 = now_sec();
    printf("L9_FAST_SIGN_TABLE seconds=%.6f fnv1a=%016llx\n",
           t1 - t0, (unsigned long long)sign_table_hash());

    /* Method 1: census via the exact 2-cycle criterion. */
    uint64_t hist[N / 2 + 1];      /* hist[nullity] over index pairs */
    uint64_t fiber_size[N];        /* fiber_size[label] over index pairs */
    memset(hist, 0, sizeof(hist));
    memset(fiber_size, 0, sizeof(fiber_size));
    uint64_t zd_pairs = 0;
    /* per-birth-class odd-part tracking for the nullity law: class m allows
     * t odd, 1 <= t <= 2^(m-3)-1 (64 t-values max at m = 9; t/2 index fits
     * in a 64-bit mask). */
    uint64_t seen_t[BITS + 1];     /* bitmask of odd parts seen, per class */
    memset(seen_t, 0, sizeof(seen_t));
    int nullity_law_ok = 1;
#ifndef L9_SKIP_VERIFY
    /* nullity of every candidate pair (0 = not a ZD), needed to audit
     * Method 2 against Method 1 on the full candidate set. */
    static uint8_t nullity_tab[N][N]; /* [i][j], i < j */
#endif
    for (int i = 1; i < N; i++) {
        const int8_t *Si = S[i];
        for (int j = i + 1; j < N; j++) {
            int l = i ^ j;
            const int8_t *Sj = S[j];
            int cnt = 0;
            for (int k = 0; k < N; k++) {
                int p = Si[k] * Sj[k] * Si[k ^ l] * Sj[k ^ l];
                cnt += (p == 1);
            }
            int nullity = cnt / 2;
#ifndef L9_SKIP_VERIFY
            nullity_tab[i][j] = (uint8_t)nullity;
#endif
            if (nullity > 0) {
                zd_pairs++;
                hist[nullity]++;
                fiber_size[l]++;
                /* nullity law: m-born pair at level b has nullity
                 * 2^(b-m+2)*t, t odd, 1 <= t <= 2^(m-3)-1. */
                int m = 32 - __builtin_clz((unsigned)l); /* bit_length(l) */
                int base = 1 << (BITS - m + 2);
                if (nullity % base != 0) {
                    nullity_law_ok = 0;
                } else {
                    int t = nullity / base;
                    if (t < 1 || t % 2 == 0 || t > (1 << (m - 3)) - 1) {
                        nullity_law_ok = 0;
                    } else {
                        seen_t[m] |= 1ULL << (t / 2);
                    }
                }
            }
        }
    }
    double t2 = now_sec();
    /* census law prediction Z(9) = 4^9 - 26*2^9 + 2^8 - 4 = 249084 triples */
    uint64_t law = 249084;
    uint64_t triples = 2 * zd_pairs;
    int census_ok = (triples == law);
    printf("L9_FAST_CENSUS seconds=%.6f index_pairs=%llu triples=%llu "
           "law_Z9=%llu census_ok=%d\n",
           t2 - t1, (unsigned long long)zd_pairs, (unsigned long long)triples,
           (unsigned long long)law, census_ok);
    printf("L9_FAST_HISTOGRAM");
    for (int v = 0; v <= N / 2; v++)
        if (hist[v]) printf(" %d:%llu", v, (unsigned long long)hist[v]);
    printf("\n");

    /* Fiber laws: labels are exactly {l in [8, 512) : l not a power of 2}
     * (F(9) = 2^9 - 9 - 5 = 498); an m-born fiber has size
     * 2^9 - 2^(9-m+2) triples... note: fiber_size here counts index
     * pairs, i.e. half the triple count used by the L4-L7 contracts. */
    int fibers_ok = 1;
    uint64_t n_fibers = 0;
    int max_nullity = 0;
    for (int v = 0; v <= N / 2; v++)
        if (hist[v]) max_nullity = v;
    for (int l = 1; l < N; l++) {
        int is_power_of_2 = (l & (l - 1)) == 0;
        int expect_fiber = (l >= 8) && !is_power_of_2;
        if (expect_fiber) {
            n_fibers++;
            int m = 32 - __builtin_clz((unsigned)l);
            uint64_t want = ((uint64_t)N - (1u << (BITS - m + 2))) / 2;
            if (fiber_size[l] != want) {
                fibers_ok = 0;
                printf("L9_FAST_FIBER_MISMATCH label=%d size=%llu want=%llu\n",
                       l, (unsigned long long)fiber_size[l],
                       (unsigned long long)want);
            }
        } else if (fiber_size[l] != 0) {
            fibers_ok = 0;
            printf("L9_FAST_FIBER_UNEXPECTED label=%d size=%llu\n",
                   l, (unsigned long long)fiber_size[l]);
        }
    }
    /* completeness: every allowed odd part occurs in every birth class */
    int completeness_ok = 1;
    for (int m = 4; m <= BITS; m++) {
        /* number of allowed odd t values is 2^(m-4); mask over t/2 index */
        uint64_t want_mask = (1ULL << (1 << (m - 4))) - 1;
        if (seen_t[m] != want_mask) completeness_ok = 0;
    }
    int max_ok = (max_nullity == (N / 2) - 4); /* 2^(b-1) - 4 = 252 */
    printf("L9_FAST_FIBERS count=%llu law_F9=498 size_law_ok=%d\n",
           (unsigned long long)n_fibers, fibers_ok);
    printf("L9_FAST_NULLITY_LAW odd_part_law=%d completeness=%d "
           "max_nullity=%d want=252 max_ok=%d\n",
           nullity_law_ok, completeness_ok, max_nullity, max_ok);

    /* Nullity-histogram counting law (the solved multiplicity law): the
     * multiset of multiplicities must be exactly
     *   { mu_s(9) attained 2^(9-s) times : s = 4..9 },
     * mu_s(9) = 2^(9-s+1)*c0(s-1), i.e.
     *   1344x32, 2016x16, 2736x8, 3480x4, 4236x2, 4998x1. */
    static uint64_t obs_mult[1 << 17]; /* obs_mult[mu] = #nullity values
                                        * with multiplicity mu; mu <=
                                        * total pairs < 2^17 */
    memset(obs_mult, 0, sizeof(obs_mult));
    uint64_t distinct_nullities = 0;
    for (int v = 0; v <= N / 2; v++)
        if (hist[v]) {
            obs_mult[hist[v]]++;
            distinct_nullities++;
        }
    int hist_law_ok = 1;
    uint64_t law_values = 0, law_mass = 0;
    printf("L9_FAST_HISTOGRAM_LAW");
    for (int s = 4; s <= BITS; s++) {
        uint64_t mu = (1ULL << (BITS - s + 1)) * c0(s - 1);
        uint64_t want_count = 1ULL << (BITS - s);
        law_values += want_count;
        law_mass += mu * want_count;
        int ok = (obs_mult[mu] == want_count);
        if (!ok) hist_law_ok = 0;
        printf(" s=%d:mu=%llu,want=%llu,got=%llu%s", s,
               (unsigned long long)mu, (unsigned long long)want_count,
               (unsigned long long)obs_mult[mu], ok ? "" : " MISMATCH");
    }
    /* no other multiplicity value may occur */
    for (int v = 0; v < (1 << 17); v++) {
        if (!obs_mult[v]) continue;
        int expected = 0;
        for (int s = 4; s <= BITS; s++)
            if ((1ULL << (BITS - s + 1)) * c0(s - 1) == (uint64_t)v)
                expected = 1;
        if (!expected) {
            hist_law_ok = 0;
            printf(" unexpected_mult=%d(count=%llu)", v,
                   (unsigned long long)obs_mult[v]);
        }
    }
    if (distinct_nullities != law_values || law_mass != zd_pairs)
        hist_law_ok = 0;
    printf(" distinct=%llu want=%llu mass=%llu ok=%d\n",
           (unsigned long long)distinct_nullities,
           (unsigned long long)law_values, (unsigned long long)law_mass,
           hist_law_ok);

#ifndef L9_SKIP_VERIFY
    /* Method 2: exact GF(65521)-rank audit of every candidate pair, both
     * signs.  nullity must match Method 1 exactly (0 for non-ZD pairs). */
    uint64_t checked = 0, mismatches = 0;
    for (int i = 1; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                load_matrix(i, j, sgn);
                int rk = rank_gfp();
                int nullity_ge = N - rk;
                checked++;
                if (nullity_ge != (int)nullity_tab[i][j]) {
                    if (mismatches < 10)
                        printf("L9_FAST_VERIFY_MISMATCH i=%d j=%d sgn=%d "
                               "census=%d ge=%d\n",
                               i, j, sgn, (int)nullity_tab[i][j], nullity_ge);
                    mismatches++;
                }
            }
        }
        if ((i & 63) == 0)
            fprintf(stderr, "L9_FAST_VERIFY_PROGRESS i=%d/%d checked=%llu\n",
                    i, N, (unsigned long long)checked);
    }
    double t3 = now_sec();
    printf("L9_FAST_VERIFY seconds=%.6f pair_signs_checked=%llu mismatches=%llu\n",
           t3 - t2, (unsigned long long)checked, (unsigned long long)mismatches);
#else
    double t3 = now_sec();
    printf("L9_FAST_VERIFY skipped (L9_SKIP_VERIFY build)\n");
#endif

    printf("L9_FAST_TOTAL seconds=%.6f\n", t3 - t0);
    int pass = census_ok && fibers_ok && n_fibers == 498 &&
               nullity_law_ok && completeness_ok && max_ok && hist_law_ok
#ifdef L9_SKIP_VERIFY
               ;
#else
               && mismatches == 0;
#endif
    printf("L9_ZD_FAST_VERDICT %s%s\n", pass ? "PASS" : "FAIL",
#ifdef L9_SKIP_VERIFY
           " (census-only; GF(65521) verification skipped)"
#else
           ""
#endif
           );
    return pass ? 0 : 1;
}
