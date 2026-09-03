// tb_san_scan.cpp — C testbench for krnl_san_scan (csim + cosim).
//
// Independent golden model (different algorithm than the kernel's
// per-lane trees: sequential cumulative scan over plain arrays) checks
// exit histogram, catastrophe count, and MAC total EXACTLY. Covers the
// boundary cases the spec calls out:
//   * conf == q_delta must SETTLE (>= semantics, spec T2)
//   * all-below-threshold sample -> final head (catastrophe)
//   * first-point hit
//   * n_samples not a multiple of LANES (tail beat)
//   * n_points = 5 (ResNet-50 geometry) and 7 (ViT-L geometry)
// plus a deterministic LCG random cohort.
#include <ap_int.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_POINTS 8
#define LANES 4
#define HIST_BINS MAX_POINTS

typedef ap_uint<15> conf_t;
typedef ap_uint<15> thresh_t;
typedef ap_uint<64> flop_t;
typedef ap_uint<32> count_t;
typedef ap_uint<128> sample_w_t;
typedef ap_uint<512> bus_t;

extern "C" void krnl_san_scan(const bus_t *samples, const flop_t *lut,
                              thresh_t q_delta, int n_points, int n_samples,
                              count_t *hist_out, count_t *catastrophe_out,
                              flop_t *flops_out);

// independent golden: plain sequential scan, cumulative-any semantics
static void golden(const unsigned short *q, int n_samples, int n_points,
                   unsigned short q_delta, const unsigned long long *lut,
                   unsigned long long hist[HIST_BINS],
                   unsigned long long *cat, unsigned long long *flops) {
    for (int b = 0; b < HIST_BINS; b++) hist[b] = 0;
    *cat = 0;
    *flops = 0;
    for (int i = 0; i < n_samples; i++) {
        int idx = n_points - 1;
        for (int k = 0; k < n_points - 1; k++) {
            if (q[i * (n_points - 1) + k] >= q_delta) { idx = k; break; }
        }
        hist[idx]++;
        if (idx == n_points - 1) (*cat)++;
        *flops += lut[idx];
    }
}

static unsigned long long lcg_state = 0x9E3779B97F4A7C15ULL;
static unsigned lcg() {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (unsigned)(lcg_state >> 33);
}

static int run_case(const char *name, int n_points, unsigned short q_delta,
                    const unsigned short *q, int n_samples) {
    const int n_conf = n_points - 1;
    unsigned long long lut[MAX_POINTS];
    for (int k = 0; k < n_points; k++)
        lut[k] = 1000000ULL * (k + 1) * (k + 1);  // arbitrary exact prefixes

    // pack samples into 512-bit beats (128 bits per sample, 15-bit fields)
    int n_words = (n_samples + LANES - 1) / LANES;
    bus_t *beats = (bus_t *)malloc(sizeof(bus_t) * n_words);
    for (int w = 0; w < n_words; w++) {
        bus_t beat = 0;
        for (int p = 0; p < LANES; p++) {
            int s = w * LANES + p;
            sample_w_t rec = 0;
            if (s < n_samples)
                for (int k = 0; k < n_conf; k++)
                    rec(15 * k + 14, 15 * k) = q[s * n_conf + k];
            beat(128 * p + 127, 128 * p) = rec;
        }
        beats[w] = beat;
    }

    flop_t lut_k[MAX_POINTS];
    for (int k = 0; k < MAX_POINTS; k++) lut_k[k] = lut[k < n_points ? k : n_points - 1];

    count_t hist_out[HIST_BINS] = {0};
    count_t cat_out = 0;
    flop_t flops_out = 0;
    krnl_san_scan(beats, lut_k, q_delta, n_points, n_samples,
                  hist_out, &cat_out, &flops_out);

    unsigned long long g_hist[HIST_BINS], g_cat, g_flops;
    golden(q, n_samples, n_points, q_delta, lut, g_hist, &g_cat, &g_flops);

    int fail = 0;
    for (int b = 0; b < n_points; b++)
        if (hist_out[b] != (count_t)g_hist[b]) {
            printf("  FAIL %s hist[%d]: kernel=%u golden=%llu\n", name, b,
                   (unsigned)hist_out[b], g_hist[b]);
            fail = 1;
        }
    if (cat_out != (count_t)g_cat) {
        printf("  FAIL %s cat: kernel=%u golden=%llu\n", name,
               (unsigned)cat_out, g_cat);
        fail = 1;
    }
    if (flops_out != (flop_t)g_flops) {
        printf("  FAIL %s flops: kernel=%llu golden=%llu\n", name,
               (unsigned long long)flops_out, g_flops);
        fail = 1;
    }
    printf("  %s %s (n=%d points=%d cat=%llu flops=%llu)\n",
           fail ? "FAIL" : "PASS", name, n_samples, n_points, g_cat, g_flops);
    free(beats);
    return fail;
}

int main() {
    int fail = 0;
    const unsigned short QD = 24576;  // round(0.75 * 2^15)

    // edge: boundary value must settle; zeros must be catastrophes;
    // first-point hit; tail beat (n not multiple of LANES)
    {
        enum { NP = 5, NC = NP - 1, N = 7 };
        unsigned short q[N * NC] = {
            QD, 0, 0, 0,          // boundary: settles at point 0
            0, 0, 0, 0,           // catastrophe
            100, QD - 1, 0, 0,    // below threshold everywhere -> catastrophe
            0, 32767, 0, 0,       // settles at point 1
            1, 2, 3, QD,          // settles at point 3 (last gate)
            0, 0, 0, 0,           // catastrophe
            QD + 1, 9, 9, 9,      // settles at point 0
        };
        fail |= run_case("edge-resnet-geom", NP, QD, q, N);
    }
    // random cohort, resnet geometry (5 points), tail beat
    {
        enum { NP = 5, NC = NP - 1, N = 100003 };
        unsigned short *q = (unsigned short *)malloc(sizeof(unsigned short) * N * NC);
        for (int i = 0; i < N * NC; i++) q[i] = lcg() & 0x7FFF;
        fail |= run_case("random-resnet-geom", NP, QD, q, N);
        free(q);
    }
    // random cohort, ViT geometry (7 points), aligned beats
    {
        enum { NP = 7, NC = NP - 1, N = 65536 };
        unsigned short *q = (unsigned short *)malloc(sizeof(unsigned short) * N * NC);
        for (int i = 0; i < N * NC; i++) q[i] = lcg() & 0x7FFF;
        fail |= run_case("random-vit-geom", NP, 18022, q, N);  // round(0.55*2^15)
        free(q);
    }
    printf("TB_SAN_SCAN_%s\n", fail ? "FAIL" : "PASS");
    return fail;
}
