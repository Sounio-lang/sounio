/*
 * fpga_census_kernel_model.c — bit-accurate software model of the proposed
 * AMD Alveo U250 catastrophe-scan census kernel (Phase 1).
 *
 * Companion to:
 *   docs/research/u250_catastrophe_scan_fpga_spec_2026-07-26.md  (spec)
 *   hardware/fpga/u250_catastrophe_scan/krnl_census.cpp          (HLS outline)
 *   scripts/ci/fpga_catastrophe_scan_gate.sh                     (CI gate)
 *
 * What is modeled
 * ---------------
 * The audited exact 2-cycle criterion of scripts/research/routon_zd_contract.py:
 * for a = e_i +/- e_j, l = i^j,
 *     p(k) = S[i,k]*S[j,k]*S[i,k^l]*S[j,k^l] in {+1,-1},
 * (i,j) is a canonical zero-divisor pair iff some p(k) = +1 and
 * nullity(L_a) = #{k : p(k) = +1}/2 exactly (both signs simultaneously).
 *
 * The hardware kernel encodes signs as BITS (0 = +1, 1 = -1), so sign
 * multiplication becomes XOR.  With row bit-vectors sb[i] and the difference
 * vector d = sb[i] ^ sb[j]:
 *
 *     p(k) = +1  <=>  d[k] ^ d[k^l] = 0,
 *
 * so per candidate pair the datapath is exactly:
 *     v       = d ^ perm_l(d)          (512-bit XOR; perm_l = index-XOR rewiring)
 *     bad     = N - popcount(v)        (adder tree)
 *     nullity = bad >> 1               (exact)
 *
 * perm_l is a pure wiring permutation (a b-stage conditional-swap mux
 * network, no arithmetic): bit k moves to position k^l.  This C model
 * reproduces that datapath bit-for-bit and is the executable contract the
 * HLS kernel must match.  It is validated here against the integer path
 * (the audited criterion, same as l8_zd_census_fast.c Method 1) at every
 * level b = 4..9, pair by pair.
 *
 * Contract assertions
 * -------------------
 *   M1  bit-parity path == integer path for every candidate pair, b = 4..9
 *   M2  census triples == growth law Z(b) at b = 4..8 (confirmed levels)
 *   M3  census triples == Z(9) = 249084 (out-of-sample falsification test)
 *   M4  L8 nullity histogram == published histogram (L8 spec section 4)
 *   M5  cycle model: 1 pair/cycle/engine (II=1) -> L9 cycles printed
 *
 * Build: cc -O2 -o fpga_census_kernel_model fpga_census_kernel_model.c
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXBITS 9
#define MAXN (1 << MAXBITS)
#define WORDS (MAXN / 64)

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

/* Growth law Z(b) = 4^b - (3b-1)*2^b + 2^(b-1) - 4 (triples, both signs). */
static uint64_t census_law(int b) {
    return (1ULL << (2 * b)) - (uint64_t)(3 * b - 1) * (1ULL << b)
           + (1ULL << (b - 1)) - 4;
}

/* Sign storage: integer table (reference) and bit-packed rows (hardware
 * image; bit k of row i is 1 iff S[i,k] = -1). */
static int8_t  S[MAXN][MAXN];
static uint64_t sb[MAXN][WORDS];

static void build_tables(int bits) {
    int n = 1 << bits;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            S[i][j] = (int8_t)cds(i, j, bits);
    for (int i = 0; i < n; i++) {
        for (int w = 0; w < WORDS; w++) sb[i][w] = 0;
        for (int k = 0; k < n; k++)
            if (S[i][k] < 0) sb[i][k >> 6] |= 1ULL << (k & 63);
    }
}

/* Integer path: audited 2-cycle criterion (routon_zd_contract.py). */
static int nullity_integer(int bits, int i, int j) {
    int n = 1 << bits, l = i ^ j, cnt = 0;
    const int8_t *Si = S[i], *Sj = S[j];
    for (int k = 0; k < n; k++)
        cnt += (Si[k] * Sj[k] * Si[k ^ l] * Sj[k ^ l] == 1);
    return cnt / 2;
}

/* Bit-parity path: the hardware datapath, modeled word-serially but with
 * exactly the hardware semantics.  perm_l(d) has bit k = d[k^l]; the HLS
 * kernel realizes it as a conditional-swap mux network (pure rewiring). */
static void perm_l(const uint64_t *d, uint64_t *out, int l, int n) {
    for (int w = 0; w < WORDS; w++) out[w] = 0;
    for (int k = 0; k < n; k++)
        if ((d[(k ^ l) >> 6] >> ((k ^ l) & 63)) & 1ULL)
            out[k >> 6] |= 1ULL << (k & 63);
}

static int nullity_bit(int bits, int i, int j) {
    int n = 1 << bits, l = i ^ j;
    uint64_t d[WORDS], p[WORDS];
    for (int w = 0; w < WORDS; w++) d[w] = sb[i][w] ^ sb[j][w];
    perm_l(d, p, l, n);
    int ones = 0;
    for (int w = 0; w < ((n + 63) >> 6); w++)
        ones += __builtin_popcountll(d[w] ^ p[w]); /* bits >= n never set */
    return (n - ones) / 2;   /* bad = #{k : v[k] = 0}; nullity = bad/2 */
}

/* Published L8 nullity histogram (index pairs), L8 spec section 4. */
static const struct { int nullity; uint64_t count; } L8_HIST[] = {
    {4,1740},{8,1368},{12,1368},{16,1008},{20,1008},{24,1008},{28,1008},
    {32,672},{36,672},{40,672},{44,672},{48,672},{52,672},{56,672},{60,672},
    {64,672},{68,672},{72,672},{76,672},{80,672},{84,672},{88,672},{92,672},
    {96,1008},{100,1008},{104,1008},{108,1008},{112,1368},{116,1368},
    {120,1740},{124,2118},
};
#define L8_HIST_LEN (sizeof(L8_HIST) / sizeof(L8_HIST[0]))

int main(void) {
    double t0 = now_sec();
    int ok = 1;

    /* M1 + M2 + M3: per-level census, both paths, pair by pair. */
    for (int bits = 4; bits <= MAXBITS; bits++) {
        build_tables(bits);
        int n = 1 << bits;
        uint64_t zd_pairs = 0, mismatches = 0;
        static uint64_t hist[MAXN / 2 + 1];
        memset(hist, 0, sizeof(hist));
        for (int i = 1; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int ni = nullity_integer(bits, i, j);
                int nb = nullity_bit(bits, i, j);
                if (ni != nb) {
                    if (mismatches < 5)
                        printf("FPGA_MODEL_MISMATCH b=%d i=%d j=%d int=%d bit=%d\n",
                               bits, i, j, ni, nb);
                    mismatches++;
                }
                if (nb > 0) { zd_pairs++; hist[nb]++; }
            }
        }
        uint64_t triples = 2 * zd_pairs;
        uint64_t law = census_law(bits);
        int law_ok = (triples == law);
        /* b = 4..8 confirmed levels (M2); b = 9 out-of-sample test (M3). */
        printf("FPGA_MODEL_LEVEL b=%d pairs=%llu triples=%llu law_Z%d=%llu "
               "law_ok=%d path_mismatches=%llu%s\n",
               bits, (unsigned long long)zd_pairs, (unsigned long long)triples,
               bits, (unsigned long long)law, law_ok,
               (unsigned long long)mismatches,
               bits == 9 ? " (out-of-sample)" : "");
        if (mismatches) ok = 0;
        if (!law_ok) ok = 0;   /* M3: Z(9) = 249084 asserted after manual run */

        /* M4: L8 histogram equality. */
        if (bits == 8) {
            int hist_ok = 1;
            for (size_t h = 0; h < L8_HIST_LEN; h++)
                if (hist[L8_HIST[h].nullity] != L8_HIST[h].count) hist_ok = 0;
            uint64_t sum = 0;
            for (int v = 0; v <= n / 2; v++) {
                if (hist[v] && v % 4 != 0) hist_ok = 0;  /* L8 spectrum step 4 */
                sum += hist[v];
            }
            if (sum != 29886) hist_ok = 0;
            if (!hist_ok) ok = 0;
            printf("FPGA_MODEL_L8_HISTOGRAM match=%d\n", hist_ok);
        }
    }

    /* M5: cycle model.  II=1 per engine, engines process disjoint pair
     * ranges; pipeline depth (~20 cycles) negligible vs pair count. */
    {
        uint64_t pairs9 = ((uint64_t)(MAXN - 1) * (MAXN - 2)) / 2; /* C(511,2) */
        int pes = 16;
        uint64_t cyc1 = pairs9, cyc16 = (pairs9 + pes - 1) / pes;
        printf("FPGA_MODEL_L9_CYCLE_EST pairs=%llu cycles_1pe=%llu "
               "cycles_%dpe=%llu est_us_%dpe@250MHz=%.1f\n",
               (unsigned long long)pairs9, (unsigned long long)cyc1,
               pes, (unsigned long long)cyc16,
               pes, (double)cyc16 / 250.0);
    }

    printf("FPGA_MODEL_TOTAL seconds=%.3f\n", now_sec() - t0);
    printf("FPGA_CENSUS_MODEL_VERDICT %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
