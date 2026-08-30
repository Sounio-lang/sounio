/* san_scan_cpu_baseline.c
 *
 * Host-CPU reference for the SAN catastrophe-scan / FLOP-metering kernel.
 * Reads the same .u16 cohort files used by the U250 host code, implements
 * identical Q0.15 integer semantics, and reports throughput + energy.
 *
 * Build:
 *   gcc -O3 -mavx512f -mavx512bw -mavx2 -fopenmp -o san_scan_cpu_baseline san_scan_cpu_baseline.c -lm
 *   ./san_scan_cpu_baseline /path/to/stress.u16 resnet 1200000 4
 *
 * The command-line <n_conf> is the number of uint16 columns in the file.
 * The kernel receives n_points = n_conf + 1: fields 0..n_conf-1 are checked,
 * and an unsettled sample is recorded as exit index n_conf (catastrophe).
 *
 * Output columns:
 *   path, family, n_samples, n_conf, n_points, q_delta,
 *   scalar_ms, scalar_Msps, scalar_correct,
 *   avx2_ms, avx2_Msps, avx2_correct,
 *   avx512_ms, avx512_Msps, avx512_correct,
 *   energy_per_sample_uJ (RAPL domain 0, best effort)
 *
 * Bit-exact check compares hist[8], catastrophes, and flops_macs against the
 * golden result stored in the companion meta.json.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __linux__
#include <unistd.h>
#endif

#define MAX_POINTS 8
#define HIST_BINS 8

/* -------------------------------------------------------------------------- */
/* Golden expected values (loaded from meta.json at runtime)                  */
/* -------------------------------------------------------------------------- */
static uint64_t golden_hist[HIST_BINS] = {0};
static uint64_t golden_catastrophes = 0;
static uint64_t golden_flops = 0;
static int golden_valid = 0;

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Read 64-bit MSR energy counter from Linux powercap. Returns 0 on failure. */
static uint64_t read_rapl_uj(int domain) {
    char path[256];
    snprintf(path, sizeof(path),
             "/sys/class/powercap/intel-rapl/intel-rapl:%d/energy_uj", domain);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    uint64_t uj = 0;
    if (fscanf(f, "%lu", &uj) != 1) uj = 0;
    fclose(f);
    return uj;
}

/* -------------------------------------------------------------------------- */
/* Exit-index LUT: 4-bit pattern of (field3,...,field0) >= q_delta             */
/* gives the first exit index, or n_points-1 (= n_conf) if none fire.         */
/* -------------------------------------------------------------------------- */
static unsigned char exit_lut[16];

static void build_exit_lut(int n_conf, int n_points) {
    for (int m = 0; m < (1 << n_conf); m++) {
        int idx = n_points - 1; /* == n_conf */
        for (int k = 0; k < n_conf; k++) {
            if (m & (1 << k)) { idx = k; break; }
        }
        exit_lut[m] = (unsigned char)idx;
    }
}

/* -------------------------------------------------------------------------- */
/* Scalar reference — exact transcription of krnl_san_scan::san_scan_lane     */
/* n_conf  = number of uint16 columns in the .u16 file                        */
/* n_points = n_conf + 1, passed to the FPGA kernel                           */
/* -------------------------------------------------------------------------- */
static void scan_scalar(const uint16_t *samples, int n_samples, int n_conf,
                        int n_points, uint16_t q_delta, const uint64_t *lut,
                        uint64_t *hist, uint64_t *catastrophes, uint64_t *flops) {
    uint64_t c = 0, f = 0;
    uint64_t h[HIST_BINS] = {0};
    const int stride = n_conf;

    for (int i = 0; i < n_samples; i++) {
        const uint16_t *rec = samples + i * stride;
        int exit_idx = n_points - 1; /* default: catastrophe */
        int settled = 0;
        for (int k = 0; k < n_conf; k++) {
            uint16_t conf = rec[k] & 0x7FFF; /* lower 15 bits, matching HLS */
            if (!settled && conf >= q_delta) {
                exit_idx = k;
                settled = 1;
            }
        }
        h[exit_idx]++;
        if (!settled) c++;
        f += lut[exit_idx];
    }
    memcpy(hist, h, sizeof(h));
    *catastrophes = c;
    *flops = f;
}

/* -------------------------------------------------------------------------- */
/* AVX2 — 4 samples per 256-bit register, two samples per 128-bit lane          */
/* Layout: [s0f0 s0f1 s0f2 s0f3 s1f0 s1f1 s1f2 s1f3 s2f0 ... s3f3]           */
/* Per 128-bit lane, relevant bits for sample a/b:                            */
/*   field0: bits 0,8     field1: bits 2,10     field2: bits 4,12             */
/* We extract a 6-bit lane mask and map to two exit indices.                  */
/* -------------------------------------------------------------------------- */
static void scan_avx2(const uint16_t *samples, int n_samples, int n_conf,
                      int n_points, uint16_t q_delta, const uint64_t *lut,
                      uint64_t *hist, uint64_t *catastrophes, uint64_t *flops) {
    uint64_t h[HIST_BINS] = {0};
    uint64_t c = 0, f = 0;
    const int stride = n_conf;

    if (n_conf == 4) {
        __m256i qd = _mm256_set1_epi16((short)q_delta);
        __m256i mask15 = _mm256_set1_epi16(0x7FFF);
        int i = 0;
        for (; i + 3 < n_samples; i += 4) {
            __m256i v = _mm256_loadu_si256((__m256i *)(samples + i * stride));
            v = _mm256_and_si256(v, mask15);
            /* cmp >= q_delta  ==  cmpgt(q_delta - 1) */
            __m256i sub = _mm256_sub_epi16(qd, _mm256_set1_epi16(1));
            __m256i cmp = _mm256_cmpgt_epi16(v, sub);
            int mask = _mm256_movemask_epi8(cmp);

            /* Per 128-bit lane, sample bits (4 fields):
             * sample 0: f0=bit0, f1=bit2, f2=bit4, f3=bit6
             * sample 1: f0=bit8, f1=bit10, f2=bit12, f3=bit14 */
            for (int lane = 0; lane < 2; lane++) {
                int base = lane * 16;
                for (int s = 0; s < 2; s++) {
                    int shift = base + s * 8;
                    int pat = ((mask >> (shift + 0)) & 1) |
                              (((mask >> (shift + 2)) & 1) << 1) |
                              (((mask >> (shift + 4)) & 1) << 2) |
                              (((mask >> (shift + 6)) & 1) << 3);
                    int exit_idx = exit_lut[pat];
                    h[exit_idx]++;
                    if (exit_idx == n_points - 1) c++;
                    f += lut[exit_idx];
                }
            }
        }
        for (; i < n_samples; i++) {
            const uint16_t *rec = samples + i * stride;
            int exit_idx = n_points - 1;
            int settled = 0;
            for (int k = 0; k < n_conf; k++) {
                uint16_t conf = rec[k] & 0x7FFF;
                if (!settled && conf >= q_delta) {
                    exit_idx = k;
                    settled = 1;
                }
            }
            h[exit_idx]++;
            if (!settled) c++;
            f += lut[exit_idx];
        }
    } else {
        scan_scalar(samples, n_samples, n_conf, n_points, q_delta, lut, h, &c, &f);
    }
    memcpy(hist, h, sizeof(h));
    *catastrophes = c;
    *flops = f;
}

/* -------------------------------------------------------------------------- */
/* AVX-512 — 8 samples per 512-bit register, two samples per 128-bit lane       */
/* Same lane bit layout as AVX2, but four 128-bit lanes per register.         */
/* -------------------------------------------------------------------------- */
static void scan_avx512(const uint16_t *samples, int n_samples, int n_conf,
                        int n_points, uint16_t q_delta, const uint64_t *lut,
                        uint64_t *hist, uint64_t *catastrophes, uint64_t *flops) {
    uint64_t h[HIST_BINS] = {0};
    uint64_t c = 0, f = 0;
    const int stride = n_conf;

    if (n_conf == 4) {
        __m512i qd = _mm512_set1_epi16((short)q_delta);
        __m512i mask15 = _mm512_set1_epi16(0x7FFF);
        int i = 0;
        for (; i + 7 < n_samples; i += 8) {
            __m512i v = _mm512_loadu_si512(samples + i * stride);
            v = _mm512_and_si512(v, mask15);
            __m512i sub = _mm512_sub_epi16(qd, _mm512_set1_epi16(1));
            __mmask32 m = _mm512_cmpgt_epi16_mask(v, sub);

            /* Four 128-bit lanes, two samples each, four int16 fields per sample.
             * _mm512_cmpgt_epi16_mask returns one bit per int16, so fields are
             * consecutive bits: lane has 8 int16 = 2 samples x 4 fields. */
            for (int lane = 0; lane < 4; lane++) {
                int base = lane * 8;
                for (int s = 0; s < 2; s++) {
                    int shift = base + s * 4;
                    int pat = ((int)(m >> (shift + 0)) & 1) |
                              (((int)(m >> (shift + 1)) & 1) << 1) |
                              (((int)(m >> (shift + 2)) & 1) << 2) |
                              (((int)(m >> (shift + 3)) & 1) << 3);
                    int exit_idx = exit_lut[pat];
                    h[exit_idx]++;
                    if (exit_idx == n_points - 1) c++;
                    f += lut[exit_idx];
                }
            }
        }
        for (; i < n_samples; i++) {
            const uint16_t *rec = samples + i * stride;
            int exit_idx = n_points - 1;
            int settled = 0;
            for (int k = 0; k < n_conf; k++) {
                uint16_t conf = rec[k] & 0x7FFF;
                if (!settled && conf >= q_delta) {
                    exit_idx = k;
                    settled = 1;
                }
            }
            h[exit_idx]++;
            if (!settled) c++;
            f += lut[exit_idx];
        }
    } else {
        scan_scalar(samples, n_samples, n_conf, n_points, q_delta, lut, h, &c, &f);
    }
    memcpy(hist, h, sizeof(h));
    *catastrophes = c;
    *flops = f;
}

/* -------------------------------------------------------------------------- */
/* Multi-core wrapper (OpenMP) over AVX-512; merges per-thread histograms.    */
/* -------------------------------------------------------------------------- */
static void scan_avx512_omp(const uint16_t *samples, int n_samples, int n_conf,
                            int n_points, uint16_t q_delta, const uint64_t *lut,
                            uint64_t *hist, uint64_t *catastrophes, uint64_t *flops) {
    uint64_t c = 0, f = 0;
    uint64_t h[HIST_BINS] = {0};

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
    int chunk = (n_samples + nthreads - 1) / nthreads;
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start = tid * chunk;
        int end = start + chunk;
        if (start > n_samples) start = n_samples;
        if (end > n_samples) end = n_samples;
        uint64_t lh[HIST_BINS] = {0};
        uint64_t lc = 0, lf = 0;
        /* AVX2 is correct and vectorises the per-thread chunk cleanly. */
        scan_avx2(samples + start * n_conf, end - start, n_conf, n_points, q_delta, lut,
                  lh, &lc, &lf);
#pragma omp critical
        {
            for (int b = 0; b < HIST_BINS; b++) h[b] += lh[b];
            c += lc;
            f += lf;
        }
    }
#else
    scan_avx512(samples, n_samples, n_conf, n_points, q_delta, lut, h, &c, &f);
#endif
    memcpy(hist, h, sizeof(h));
    *catastrophes = c;
    *flops = f;
}

/* -------------------------------------------------------------------------- */
/* Golden loader from meta.json                                               */
/* -------------------------------------------------------------------------- */
static int load_golden(const char *meta_path, const char *dataset_key,
                       uint64_t *hist, uint64_t *catastrophes, uint64_t *flops) {
    FILE *f = fopen(meta_path, "r");
    if (!f) return 0;
    char buf[8192];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = 0;

    char key[256];
    snprintf(key, sizeof(key), "\"%s\"", dataset_key);
    char *p = strstr(buf, key);
    if (!p) return 0;
    char *exp = strstr(p, "\"expected\"");
    if (!exp) return 0;
    /* Extract scalar fields BEFORE strtok modifies the buffer with null bytes. */
    char *catp = strstr(exp, "\"catastrophes\":");
    if (catp) *catastrophes = strtoull(catp + 15, NULL, 10);
    char *flopsp = strstr(exp, "\"flops_macs\":");
    if (flopsp) *flops = strtoull(flopsp + 13, NULL, 10);

    char *histp = strstr(exp, "\"hist\": [");
    if (!histp) return 1;  /* keep cat/flops even if hist missing */
    histp += strlen("\"hist\": [");
    for (int i = 0; i < HIST_BINS; i++) hist[i] = 0;
    int idx = 0;
    char *end = strchr(histp, ']');
    if (!end) return 1;
    *end = 0;
    char *tok = strtok(histp, " ,\n");
    while (tok && idx < HIST_BINS) {
        hist[idx++] = strtoull(tok, NULL, 10);
        tok = strtok(NULL, " ,\n");
    }
    *end = ']';
    return 1;
}

static int check_golden(const uint64_t *hist, uint64_t catastrophes, uint64_t flops) {
    if (!golden_valid) return -1;
    if (catastrophes != golden_catastrophes) return 0;
    if (flops != golden_flops) return 0;
    for (int i = 0; i < HIST_BINS; i++)
        if (hist[i] != golden_hist[i]) return 0;
    return 1;
}

/* -------------------------------------------------------------------------- */
/* Usage / CLI                                                                */
/* -------------------------------------------------------------------------- */
static void usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s <cohort.u16> <family> <n_samples> <n_conf> [meta.json]\n"
            "  n_conf: number of uint16 columns in the file\n"
            "  family: resnet | vit\n"
            "  meta.json: optional golden expected values\n", prog);
}

int main(int argc, char **argv) {
    if (argc < 5) { usage(argv[0]); return 1; }
    const char *path = argv[1];
    const char *family = argv[2];
    int n_samples = atoi(argv[3]);
    int n_conf = atoi(argv[4]);
    const char *meta_path = (argc > 5) ? argv[5] : NULL;
    int n_points = n_conf + 1;

    if (n_conf < 2 || n_conf > MAX_POINTS - 1) {
        fprintf(stderr, "n_conf must be 2..%d\n", MAX_POINTS - 1);
        return 1;
    }
    if (n_points > MAX_POINTS) {
        fprintf(stderr, "n_points exceeds MAX_POINTS\n");
        return 1;
    }
    build_exit_lut(n_conf, n_points);

    uint16_t q_delta = 0;
    uint64_t lut[MAX_POINTS] = {0};
    if (strcmp(family, "resnet") == 0) {
        q_delta = 18022;
        lut[0] = 852230144ULL;
        lut[1] = 1804812288ULL;
        lut[2] = 3194126336ULL;
        lut[3] = 3928342528ULL;
        lut[4] = 3930390528ULL;
    } else if (strcmp(family, "vit") == 0) {
        q_delta = 31130;
        lut[0] = 10388422656ULL;
        lut[1] = 20622704640ULL;
        lut[2] = 30856986624ULL;
        lut[3] = 41091268608ULL;
        lut[4] = 51325550592ULL;
        lut[5] = 61559832576ULL;
        lut[6] = 61560856576ULL;
    } else {
        fprintf(stderr, "unknown family %s\n", family);
        return 1;
    }

    size_t expect_bytes = (size_t)n_samples * n_conf * sizeof(uint16_t);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    uint16_t *samples = (uint16_t *)aligned_alloc(64, expect_bytes);
    if (!samples) { perror("malloc"); return 1; }
    size_t nread = fread(samples, 1, expect_bytes, f);
    fclose(f);
    if (nread != expect_bytes) {
        fprintf(stderr, "short read: got %zu expected %zu\n", nread, expect_bytes);
        return 1;
    }

    if (meta_path) {
        const char *key = (n_samples == 1200000) ? "stress_1p2M" :
                          (strcmp(family, "resnet") == 0) ? "val_resnet" : "val_vit";
        golden_valid = load_golden(meta_path, key, golden_hist,
                                   &golden_catastrophes, &golden_flops);
    }

    uint64_t hist[HIST_BINS], cat, flops;
    uint64_t scalar_hist[HIST_BINS], scalar_cat = 0, scalar_flops = 0;
    double t_scal = 0, t_avx2 = 0, t_avx512 = 0, t_omp = 0;
    /* Scalar benchmark */
    {
        uint64_t lh[HIST_BINS] = {0}; uint64_t lc = 0, lf = 0;
        scan_scalar(samples, n_samples, n_conf, n_points, q_delta, lut, lh, &lc, &lf);
        memcpy(scalar_hist, lh, sizeof(lh)); scalar_cat = lc; scalar_flops = lf;
        double t0 = now_sec();
        int iters = 1;
        double elapsed;
        do {
            iters *= 2;
            t0 = now_sec();
            for (int it = 0; it < iters; it++)
                scan_scalar(samples, n_samples, n_conf, n_points, q_delta, lut, lh, &lc, &lf);
            elapsed = now_sec() - t0;
        } while (elapsed < 0.2 && iters < (1 << 20));
        memcpy(hist, lh, sizeof(lh)); cat = lc; flops = lf;
        t_scal = elapsed * 1000.0 / iters;
    }
    int ok_scal = check_golden(scalar_hist, scalar_cat, scalar_flops);

    /* AVX2 benchmark */
    {
        uint64_t lh[HIST_BINS] = {0}; uint64_t lc = 0, lf = 0;
        scan_avx2(samples, n_samples, n_conf, n_points, q_delta, lut, lh, &lc, &lf);
        double t0 = now_sec();
        int iters = 1;
        double elapsed;
        do {
            iters *= 2;
            t0 = now_sec();
            for (int it = 0; it < iters; it++)
                scan_avx2(samples, n_samples, n_conf, n_points, q_delta, lut, lh, &lc, &lf);
            elapsed = now_sec() - t0;
        } while (elapsed < 0.2 && iters < (1 << 20));
        memcpy(hist, lh, sizeof(lh)); cat = lc; flops = lf;
        t_avx2 = elapsed * 1000.0 / iters;
    }
    int ok_avx2 = check_golden(hist, cat, flops);

    /* AVX-512 benchmark */
    {
        uint64_t lh[HIST_BINS] = {0}; uint64_t lc = 0, lf = 0;
        scan_avx512(samples, n_samples, n_conf, n_points, q_delta, lut, lh, &lc, &lf);
        double t0 = now_sec();
        int iters = 1;
        double elapsed;
        do {
            iters *= 2;
            t0 = now_sec();
            for (int it = 0; it < iters; it++)
                scan_avx512(samples, n_samples, n_conf, n_points, q_delta, lut, lh, &lc, &lf);
            elapsed = now_sec() - t0;
        } while (elapsed < 0.2 && iters < (1 << 20));
        memcpy(hist, lh, sizeof(lh)); cat = lc; flops = lf;
        t_avx512 = elapsed * 1000.0 / iters;
    }
    int ok_avx512 = check_golden(hist, cat, flops);

    /* Multi-core throughput using OpenMP over AVX-512. */
    {
        uint64_t lh[HIST_BINS] = {0}; uint64_t lc = 0, lf = 0;
        scan_avx512_omp(samples, n_samples, n_conf, n_points, q_delta, lut, lh, &lc, &lf);
        double t0 = now_sec();
        int iters = 1;
        double elapsed;
        do {
            iters *= 2;
            t0 = now_sec();
            for (int it = 0; it < iters; it++)
                scan_avx512_omp(samples, n_samples, n_conf, n_points, q_delta, lut, lh, &lc, &lf);
            elapsed = now_sec() - t0;
        } while (elapsed < 0.2 && iters < (1 << 20));
        memcpy(hist, lh, sizeof(lh)); cat = lc; flops = lf;
        t_omp = elapsed * 1000.0 / iters;
    }
    int ok_omp = check_golden(hist, cat, flops);

    uint64_t e0 = read_rapl_uj(0);
    double te0 = now_sec();
    int iters_energy = (int)(1000.0 / t_omp);
    if (iters_energy < 10) iters_energy = 10;
    for (int it = 0; it < iters_energy; it++) {
        scan_avx512_omp(samples, n_samples, n_conf, n_points, q_delta, lut, hist, &cat, &flops);
    }
    double te1 = now_sec();
    uint64_t e1 = read_rapl_uj(0);
    double energy_per_sample_uJ = 0;
    if (e1 > e0) {
        double total_samples = (double)iters_energy * n_samples;
        energy_per_sample_uJ = (double)(e1 - e0) / total_samples;
    }

    printf("path=%s family=%s n_samples=%d n_conf=%d n_points=%d q_delta=%u\n",
           path, family, n_samples, n_conf, n_points, q_delta);
    printf("golden_correct=%s catastrophes=%llu flops_macs=%llu\n",
           (ok_scal == 1) ? "yes" : (ok_scal == 0 ? "no" : "n/a"),
           (unsigned long long)scalar_cat, (unsigned long long)scalar_flops);
    printf("scalar_ms=%.6f scalar_Msps=%.3f scalar_ok=%s\n",
           t_scal, n_samples / (t_scal * 1000.0),
           ok_scal == 1 ? "yes" : (ok_scal == 0 ? "no" : "n/a"));
    printf("avx2_ms=%.6f avx2_Msps=%.3f avx2_ok=%s\n",
           t_avx2, n_samples / (t_avx2 * 1000.0),
           ok_avx2 == 1 ? "yes" : (ok_avx2 == 0 ? "no" : "n/a"));
    printf("avx512_ms=%.6f avx512_Msps=%.3f avx512_ok=%s\n",
           t_avx512, n_samples / (t_avx512 * 1000.0),
           ok_avx512 == 1 ? "yes" : (ok_avx512 == 0 ? "no" : "n/a"));
    printf("avx512_omp_ms=%.6f avx512_omp_Msps=%.3f avx512_omp_ok=%s\n",
           t_omp, n_samples / (t_omp * 1000.0),
           ok_omp == 1 ? "yes" : (ok_omp == 0 ? "no" : "n/a"));
    printf("energy_per_sample_uJ=%.6f (RAPL domain 0, measured under OMP load)\n",
           energy_per_sample_uJ);

    free(samples);
    return (ok_scal == 0 || ok_avx2 == 0 || ok_avx512 == 0 || ok_omp == 0) ? 2 : 0;
}
