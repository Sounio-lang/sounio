/*
 * l9_nullity_histogram_law_contract.c — level-9 (512-dim) out-of-sample
 * verification of the nullity-histogram counting law.
 *
 * Companion to:
 *   docs/research/l9_nullity_histogram_law_spec_2026-07-26.md
 *   docs/research/nullity_histogram_law_spec_2026-07-26.md  (the law)
 *   scripts/research/nullity_histogram_law_contract.py      (levels 4..7 + L8 tabulation)
 *   scripts/research/l8_zd_census_fast.c                    (implementation template)
 *   scripts/ci/l9_nullity_histogram_law_gate.sh             (CI gate)
 *
 * The law (parent spec, section 1): at every level b >= 4 the canonical ZD
 * nullity histogram has exactly b-3 distinct multiplicities; for each
 * terminal level s in {4..b} exactly 2^(b-s) distinct nullity values occur
 * with multiplicity
 *
 *     mu_s(b) = 2^(b-s+1) * c0(s-1),   c0(b) = 3*(2b-3)*2^(b-2) + 3.
 *
 * Per-pair form: N(m, b, t) = 2^(b-m+V+1) * c0(m_s - 1), where (V, m_s) is
 * the 2-adic descent of the odd part t at birth level m (see descent()).
 *
 * Level-9 prediction (falsification target stated in the parent spec):
 * multiplicities 1344, 2016, 2736, 3480, 4236, 4998 attained by
 * 32, 16, 8, 4, 2, 1 distinct nullity values; total 124542 index pairs
 * = Z(9)/2 with Z(9) = 249084.
 *
 * What this program does (all exact integer arithmetic):
 *
 *   Census:    exact 2-cycle criterion (audited method of
 *              scripts/research/routon_zd_contract.py) at level 9:
 *              l = i^j, p(k) = S[i,k]*S[j,k]*S[i,k^l]*S[j,k^l],
 *              nullity = #{k : p(k)=+1}/2.  Also an independent exact
 *              level-8 census (k in [0,256)) for the lemma checks.
 *   Lemmas:    L1 eps-identity, L2 left=right nullity, L3 native recursion
 *              (nullity = 256 - 2*nu - 4), L4 doubling (embedded/high =
 *              2*nu) — verified at level 9 (lemmas were previously verified
 *              exhaustively only at b <= 7; levels 8 and 9 are new).
 *   Law:       per-class N(m,9,t), full histogram, and terminal
 *              multiplicity structure compared against the exact scan.
 *   Audit:     independent GF(65521)-rank verification (dense Gaussian
 *              elimination, exact Q-rank by the 2x2 block argument of the
 *              L8 contract) on a deterministic 1/32 subsample of all
 *              candidate pair-signs.  Set L9_FULL_VERIFY=1 for the complete
 *              audit of all 260610 pair-signs (~2.5 min).
 *
 * Build: cc -O2 -o l9_nullity_histogram_law_contract l9_nullity_histogram_law_contract.c
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BITS 9
#define N (1 << BITS)        /* 512 */
#define H (1 << (BITS - 1))  /* 256 */
#define PRIME 65521u /* largest 16-bit prime; odd, so char != 2 */

static int8_t S[N][N];
static int8_t ST[N][N]; /* transpose, for the right p-function */

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
        for (int j = 0; j < N; j++) {
            S[i][j] = (int8_t)cds(i, j, BITS);
            ST[j][i] = S[i][j];
        }
}

/* FNV-1a 64-bit over the raw int8 table, for cross-implementation audit
 * against the NumPy sign table of scripts/research/routon_zd_contract.py. */
static uint64_t sign_table_hash(void) {
    uint64_t h = 1469598103934665603ULL;
    const uint8_t *p = (const uint8_t *)S;
    for (size_t k = 0; k < sizeof(S); k++) {
        h ^= p[k];
        h *= 1099511628211ULL;
    }
    return h;
}

/* ------------------------------------------------------------------ */
/* The counting law                                                    */
/* ------------------------------------------------------------------ */

static uint64_t c0_law(int b) {
    /* invertible canonical candidate pairs at level b */
    return 3ULL * (uint64_t)(2 * b - 3) * (1ULL << (b - 2)) + 3;
}

/* 2-adic descent of the odd part t at birth level m: iterate
 * (m, t) -> (m - v, (max-t)/2^v) with max = 2^(m-3)-1, v = v2(max-t),
 * until t = max.  Returns accumulated valuation V, sets *m_s = terminal
 * level.  Terminates because m strictly decreases and the only odd t at
 * m = 4 is t = 1 = max. */
static int descent(int m, int t, int *m_s) {
    int V = 0;
    for (;;) {
        int mx = (1 << (m - 3)) - 1;
        if (t == mx) { *m_s = m; return V; }
        int u = mx - t;
        int v = __builtin_ctz((unsigned)u);
        V += v;
        m -= v;
        t = u >> v;
    }
}

/* N(m, b, t): # index pairs at level b, born at m, nullity 2^(b-m+2)*t. */
static uint64_t law_N(int m, int b, int t) {
    int m_s;
    int V = descent(m, t, &m_s);
    return (1ULL << (b - m + V + 1)) * c0_law(m_s - 1);
}

/* ------------------------------------------------------------------ */
/* Exact GF(PRIME) rank (independent audit)                            */
/* ------------------------------------------------------------------ */

static uint16_t Mbuf[N * N];

static uint32_t modinv(uint32_t a) {
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

/* ------------------------------------------------------------------ */

static uint8_t nullity_tab[N][N];  /* level-9 nullity of {i,j} (0 = invertible) */
static uint8_t nullity8[H][H];     /* exact level-8 nullity of {i,j} */

int main(void) {
    double t0 = now_sec();

    build_sign_table();
    double t1 = now_sec();
    printf("L9_SIGN_TABLE seconds=%.6f fnv1a=%016llx\n",
           t1 - t0, (unsigned long long)sign_table_hash());

    /* Exact level-8 census (k in [0,256), sign table restricted to the
     * low block — the level-8 algebra), needed for the L3/L4 checks. */
    for (int i = 1; i < H; i++) {
        const int8_t *Si = S[i];
        for (int j = i + 1; j < H; j++) {
            int l = i ^ j;
            const int8_t *Sj = S[j];
            int cnt = 0;
            for (int k = 0; k < H; k++)
                cnt += (Si[k] * Sj[k] * Si[k ^ l] * Sj[k ^ l] == 1);
            nullity8[i][j] = (uint8_t)(cnt / 2);
        }
    }
    double t8 = now_sec();
    printf("L9_L8_CENSUS seconds=%.6f\n", t8 - t1);

    /* Level-9 census via the exact 2-cycle criterion. */
    static uint64_t hist[N / 2 + 1];            /* hist[nullity] */
    static uint64_t chist[BITS + 1][N / 2 + 1]; /* per birth class */
    static uint64_t fiber_size[N];
    uint64_t zd_pairs = 0;
    uint64_t seen_t[BITS + 1];
    memset(seen_t, 0, sizeof(seen_t));
    int odd_part_law_ok = 1;
    uint64_t l1_violations = 0, l2_mismatches = 0;
    uint64_t pair_index = 0;
    for (int i = 1; i < N; i++) {
        const int8_t *Si = S[i];
        const int8_t *Ti = ST[i];
        for (int j = i + 1; j < N; j++, pair_index++) {
            int l = i ^ j;
            const int8_t *Sj = S[j];
            const int8_t *Tj = ST[j];
            /* L1: eps-identity S(i,l)*S(j,l) = -1 */
            if (Si[l] * Sj[l] != -1) l1_violations++;
            int cnt = 0, cntR = 0;
            for (int k = 0; k < N; k++) {
                cnt += (Si[k] * Sj[k] * Si[k ^ l] * Sj[k ^ l] == 1);
                cntR += (Ti[k] * Tj[k] * Ti[k ^ l] * Tj[k ^ l] == 1);
            }
            /* L2: nullity(L_a) = nullity(R_a) */
            if (cnt != cntR) l2_mismatches++;
            int nullity = cnt / 2;
            nullity_tab[i][j] = (uint8_t)nullity;
            if (nullity > 0) {
                zd_pairs++;
                hist[nullity]++;
                fiber_size[l]++;
                int m = 32 - __builtin_clz((unsigned)l); /* bit_length(l) */
                chist[m][nullity]++;
                /* odd-part law: nullity = 2^(b-m+2)*t, t odd,
                 * 1 <= t <= 2^(m-3)-1 */
                int base = 1 << (BITS - m + 2);
                if (nullity % base != 0) {
                    odd_part_law_ok = 0;
                } else {
                    int t = nullity / base;
                    if (t < 1 || t % 2 == 0 || t > (1 << (m - 3)) - 1)
                        odd_part_law_ok = 0;
                    else
                        seen_t[m] |= 1ULL << (t / 2);
                }
            }
        }
    }
    double t2 = now_sec();

    /* census law Z(9) = 4^9 - (3*9-1)*2^9 + 2^8 - 4 = 249084 triples */
    uint64_t law_z9 = (1ULL << (2 * BITS)) - (uint64_t)(3 * BITS - 1) * N
                      + H - 4;
    uint64_t triples = 2 * zd_pairs;
    int census_ok = (triples == law_z9);
    printf("L9_CENSUS seconds=%.6f index_pairs=%llu triples=%llu "
           "law_Z9=%llu census_ok=%d\n",
           t2 - t8, (unsigned long long)zd_pairs, (unsigned long long)triples,
           (unsigned long long)law_z9, census_ok);
    printf("L9_HISTOGRAM");
    int distinct_values = 0, max_nullity = 0;
    for (int v = 0; v <= N / 2; v++)
        if (hist[v]) {
            printf(" %d:%llu", v, (unsigned long long)hist[v]);
            distinct_values++;
            max_nullity = v;
        }
    printf("\n");

    printf("L9_LEMMAS l1_violations=%llu l2_mismatches=%llu "
           "pairs_checked=%llu\n",
           (unsigned long long)l1_violations,
           (unsigned long long)l2_mismatches,
           (unsigned long long)pair_index);

    /* L3 (native recursion) and L4 (doubling) at level 9, against the
     * independent exact level-8 census. */
    uint64_t l3_mismatches = 0, l3_checked = 0;
    uint64_t l4e_mismatches = 0, l4h_mismatches = 0, l4_checked = 0;
    for (int i = 1; i < H; i++) {
        for (int j = i + 1; j < H; j++) {
            int nu8 = nullity8[i][j];
            /* L4: embedded {i,j} and high {H+i,H+j} at level 9 have
             * nullity 2*nu8 (including nu8 = 0). */
            l4_checked++;
            if ((int)nullity_tab[i][j] != 2 * nu8) l4e_mismatches++;
            if ((int)nullity_tab[H + i][H + j] != 2 * nu8) l4h_mismatches++;
        }
    }
    /* L3: native pair {i0, H+j0}, label H+r (1 <= r <= H-1), j0 = i0^r,
     * i0 != r: nullity = H - 2*nu8 - 4. */
    for (int r = 1; r < H; r++) {
        for (int i0 = 1; i0 < H; i0++) {
            if (i0 == r) continue;
            int j0 = i0 ^ r;
            int nu8 = nullity8[i0 < j0 ? i0 : j0][i0 < j0 ? j0 : i0];
            int pred = H - 2 * nu8 - 4;
            l3_checked++;
            if ((int)nullity_tab[i0][H + j0] != pred) l3_mismatches++;
        }
    }
    printf("L9_L3_L4 l3_mismatches=%llu l3_checked=%llu "
           "l4_embedded_mismatches=%llu l4_high_mismatches=%llu "
           "l4_checked=%llu\n",
           (unsigned long long)l3_mismatches, (unsigned long long)l3_checked,
           (unsigned long long)l4e_mismatches,
           (unsigned long long)l4h_mismatches,
           (unsigned long long)l4_checked);

    /* Fiber laws: labels exactly {l in [512, ...) : ...} i.e.
     * {l in [8,512) : l not a power of 2}, F(9) = 2^9 - 9 - 5 = 498;
     * an m-born fiber has (N - 2^(BITS-m+2))/2 index pairs. */
    int fibers_ok = 1;
    uint64_t n_fibers = 0;
    for (int l = 1; l < N; l++) {
        int is_power_of_2 = (l & (l - 1)) == 0;
        int expect_fiber = (l >= 8) && !is_power_of_2;
        if (expect_fiber) {
            n_fibers++;
            int m = 32 - __builtin_clz((unsigned)l);
            uint64_t want = ((uint64_t)N - (1u << (BITS - m + 2))) / 2;
            if (fiber_size[l] != want) {
                fibers_ok = 0;
                printf("L9_FIBER_MISMATCH label=%d size=%llu want=%llu\n",
                       l, (unsigned long long)fiber_size[l],
                       (unsigned long long)want);
            }
        } else if (fiber_size[l] != 0) {
            fibers_ok = 0;
            printf("L9_FIBER_UNEXPECTED label=%d size=%llu\n",
                   l, (unsigned long long)fiber_size[l]);
        }
    }
    int completeness_ok = 1;
    for (int m = 4; m <= BITS; m++) {
        uint64_t want_mask = (1ULL << (1 << (m - 4))) - 1;
        if (seen_t[m] != want_mask) completeness_ok = 0;
    }
    int max_ok = (max_nullity == (N / 2) - 4); /* 2^(b-1) - 4 = 252 */
    printf("L9_FIBERS count=%llu law_F9=%d size_law_ok=%d "
           "odd_part_law_ok=%d completeness_ok=%d max_nullity=%d "
           "want=252 max_ok=%d\n",
           (unsigned long long)n_fibers, (1 << BITS) - BITS - 5,
           fibers_ok, odd_part_law_ok, completeness_ok, max_nullity, max_ok);

    /* The law at level 9: full histogram and per-class histograms. */
    static uint64_t law_hist[N / 2 + 1];
    static uint64_t law_chist[BITS + 1][N / 2 + 1];
    for (int m = 4; m <= BITS; m++)
        for (int t = 1; t < (1 << (m - 3)); t += 2) {
            int nullity = (1 << (BITS - m + 2)) * t;
            uint64_t cnt = law_N(m, BITS, t);
            law_hist[nullity] += cnt;
            law_chist[m][nullity] += cnt;
        }
    int law_hist_ok = 1, law_class_ok = 1;
    for (int v = 0; v <= N / 2; v++)
        if (law_hist[v] != hist[v]) law_hist_ok = 0;
    for (int m = 4; m <= BITS; m++)
        for (int v = 0; v <= N / 2; v++)
            if (law_chist[m][v] != chist[m][v]) {
                law_class_ok = 0;
                printf("L9_LAW_CLASS_MISMATCH m=%d nullity=%d law=%llu "
                       "scan=%llu\n", m, v,
                       (unsigned long long)law_chist[m][v],
                       (unsigned long long)chist[m][v]);
            }
    uint64_t law_total = 0;
    for (int v = 0; v <= N / 2; v++) law_total += law_hist[v];
    printf("L9_LAW_HISTOGRAM match=%d per_class_match=%d "
           "distinct_values=%d law_total=%llu\n",
           law_hist_ok, law_class_ok, distinct_values,
           (unsigned long long)law_total);

    /* Terminal-level structure: exactly b-3 = 6 distinct multiplicities;
     * mu_s = 3*2^(b-s+1)*((2s-5)*2^(s-3)+1) attained by exactly 2^(b-s)
     * distinct nullity values, s = 4..9. */
    /* multiplicity histogram: mh[mult] = #nullity values with that mult */
    static uint64_t mh[5000 + 1];
    int distinct_mults = 0;
    for (int v = 0; v <= N / 2; v++)
        if (hist[v]) {
            if (hist[v] > 5000) { printf("L9_TERMINAL_OVERFLOW\n"); return 1; }
            if (mh[hist[v]] == 0) distinct_mults++;
            mh[hist[v]]++;
        }
    int terminal_ok = (distinct_mults == BITS - 3);
    for (int s = 4; s <= BITS; s++) {
        uint64_t mu = 3ULL * (1ULL << (BITS - s + 1))
                      * ((uint64_t)(2 * s - 5) * (1ULL << (s - 3)) + 1);
        uint64_t want = 1ULL << (BITS - s);
        if (mh[mu] != want) {
            terminal_ok = 0;
            printf("L9_TERMINAL_MISMATCH s=%d mu=%llu values=%llu want=%llu\n",
                   s, (unsigned long long)mu, (unsigned long long)mh[mu],
                   (unsigned long long)want);
        }
    }
    printf("L9_TERMINAL match=%d distinct_multiplicities=%d "
           "structure={1344:32,2016:16,2736:8,3480:4,4236:2,4998:1}\n",
           terminal_ok, distinct_mults);

    /* Independent GF(65521)-rank audit.  Default: deterministic 1/32
     * subsample of all candidate pair-signs; L9_FULL_VERIFY=1 audits all
     * 261120 pair-signs. */
    int full = getenv("L9_FULL_VERIFY") && getenv("L9_FULL_VERIFY")[0] == '1';
    uint64_t checked = 0, mismatches = 0;
    pair_index = 0;
    for (int i = 1; i < N; i++) {
        for (int j = i + 1; j < N; j++, pair_index++) {
            if (!full && (pair_index % 32) != 0) continue;
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                load_matrix(i, j, sgn);
                int nullity_ge = N - rank_gfp();
                checked++;
                if (nullity_ge != (int)nullity_tab[i][j]) {
                    if (mismatches < 10)
                        printf("L9_VERIFY_MISMATCH i=%d j=%d sgn=%d "
                               "census=%d ge=%d\n",
                               i, j, sgn, (int)nullity_tab[i][j], nullity_ge);
                    mismatches++;
                }
            }
        }
    }
    double t3 = now_sec();
    printf("L9_VERIFY mode=%s seconds=%.6f pair_signs_checked=%llu "
           "mismatches=%llu\n",
           full ? "full" : "subsample_1of32", t3 - t2,
           (unsigned long long)checked, (unsigned long long)mismatches);

    printf("L9_TOTAL seconds=%.6f\n", t3 - t0);
    int pass = census_ok && law_hist_ok && law_class_ok && terminal_ok &&
               distinct_values == 63 && law_total == 124542 &&
               l1_violations == 0 && l2_mismatches == 0 &&
               l3_mismatches == 0 && l4e_mismatches == 0 &&
               l4h_mismatches == 0 &&
               fibers_ok && n_fibers == 498 && odd_part_law_ok &&
               completeness_ok && max_ok && mismatches == 0;
    printf("L9_NULLITY_LAW_VERDICT %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
