// kaxi_pbpk_2comp_gum_sampler — Phase Y input generator.
//
// Generates 4 binary files (f32, little-endian) for the GUM kernel:
//   init_c1.bin   C1 initial values  (blood compartment)
//   init_c2.bin   C2 initial values  (tissue compartment)
//   init_v11.bin  V11 initial values (GUM variance of C1)
//   init_v22.bin  V22 initial values (GUM variance of C2)
//
// and expected.summary with CPU-analytic digests for truth-claim verification.
//
// ODE constants (must match pbpk_2comp_gum_4step.ptx):
//   K_in=1.0, k12=0.5, k21=0.25, dt=0.5, n_steps=4
//
// Initial conditions (rapamycin 2-comp, Ferron 1997):
//   C1 ~ LogNormal(mean_dose, dose_sigma)
//   C2 = frac_c2 * C1
//
// GUM initial variances (JCGM 100:2008, exact log-normal formula):
//   V11_init = C1² × (exp(dose_sigma²) − 1)
//   V22_init = frac_c2² × V11_init
//   V12_init = 0  (independent initial compartments)
//
// CPU analytic propagation tracks full 2×2 covariance matrix via J·Σ·Jᵀ.

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>

// ODE constants — must match pbpk_2comp_gum_4step.ptx
#define K_IN  1.0f
#define K12   0.5f
#define K21   0.25f
#define DT    0.5f
#define N_STEPS 4

// Jacobian entries (constant — linear ODE)
#define J11  0.75f      // 1 - DT*K12
#define J12  0.125f     // DT*K21
#define J21  0.25f      // DT*K12
#define J22  0.875f     // 1 - DT*K21
#define K_IN_DT 0.5f    // K_IN * DT

static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double next_uniform(uint64_t *state) {
    return ((double)(splitmix64_next(state) >> 11) + 1.0) * (1.0 / 9007199254740993.0);
}
static double next_normal(uint64_t *state) {
    double u1 = next_uniform(state), u2 = next_uniform(state);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.141592653589793238 * u2);
}

// Precomputed GUM constants (matching pbpk_2comp_gum_4step.ptx mov.f32 lines).
// All exact in f32 since each is a finite sum/product of {0.75, 0.875, 0.5, 0.25, 0.125}.
#define J11SQ        0.5625f    // J11*J11
#define TWO_J11_J12  0.1875f    // 2*J11*J12 == J11*J21 (shared with V12 lane)
#define J12SQ        0.015625f  // J12*J12
#define J11_J22_PLUS_J12_J21 0.6875f  // J11*J22 + J12*J21
#define J12_J22      0.109375f  // J12*J22
#define J21SQ        0.0625f    // J21*J21
#define TWO_J21_J22  0.4375f    // 2*J21*J22
#define J22SQ        0.765625f  // J22*J22

// One Euler step in f32 — matches PTX mul.rn / fma.rn / add.rn chain bit-exactly.
//
// PTX:
//   %f23 = mul(J11, c1); %f23 = fma(J12, c2, %f23); %f23 = add(%f23, K_in_dt);
//   %f24 = mul(J21, c1); %f24 = fma(J22, c2, %f24);
static void euler_step_f32(float c1, float c2, float *c1_out, float *c2_out) {
    float t = J11 * c1;
    t = fmaf(J12, c2, t);
    *c1_out = t + K_IN_DT;
    t = J21 * c1;
    *c2_out = fmaf(J22, c2, t);
}

// Full GUM covariance propagation step via J·Σ·Jᵀ.
// Σ = [[v11, v12], [v12, v22]]
//
// PTX uses mul.rn + 2× fma.rn (single rounding for each fma's mul-add pair),
// so the CPU oracle must use fmaf() in the same order.
static void gum_step(float v11, float v12, float v22,
                     float *v11_out, float *v12_out, float *v22_out) {
    float t;
    t = J11SQ * v11;
    t = fmaf(TWO_J11_J12, v12, t);
    *v11_out = fmaf(J12SQ, v22, t);

    t = TWO_J11_J12 * v11;     // J11*J21 == 2*J11*J12 = 0.1875
    t = fmaf(J11_J22_PLUS_J12_J21, v12, t);
    *v12_out = fmaf(J12_J22, v22, t);

    t = J21SQ * v11;
    t = fmaf(TWO_J21_J22, v12, t);
    *v22_out = fmaf(J22SQ, v22, t);
}

static int write_all(const char *path, const void *buf, size_t bytes) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t w = fwrite(buf, 1, bytes, f);
    fclose(f);
    return (w == bytes) ? 0 : -1;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s --out-dir DIR --cohort N [options]\n"
        "  --out-dir DIR        output directory\n"
        "  --cohort N           total cohort size\n"
        "  --seed N             PRNG seed (default 42)\n"
        "  --shard-index K      shard index, 0-based (default 0)\n"
        "  --shard-count S      total shards (default 1)\n"
        "  --mean-dose F        log-normal location for C1 (default 1.0)\n"
        "  --dose-sigma F       log-normal scale sigma (default 0.50)\n"
        "  --frac-c2 F          C2 = frac_c2*C1 initially (default 0.1)\n"
        "  --counts-only        skip writing binary files\n"
        "\n"
        "Outputs:\n"
        "  init_c1.bin    shard_patients × 4 bytes (f32)\n"
        "  init_c2.bin    shard_patients × 4 bytes (f32)\n"
        "  init_v11.bin   shard_patients × 4 bytes (f32, GUM variance of C1)\n"
        "  init_v22.bin   shard_patients × 4 bytes (f32, GUM variance of C2)\n"
        "  expected.summary   digests for truth-claim verification\n",
        prog);
}

int main(int argc, char **argv) {
    const char *out_dir = NULL;
    long cohort = 0;
    uint64_t seed = 42;
    double mean_dose = 1.0;
    double dose_sigma = 0.50;
    double frac_c2 = 0.1;
    long shard_index = 0, shard_count = 1;
    int counts_only = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else if (!strcmp(argv[i], "--out-dir")     && i+1<argc) { out_dir     = argv[++i]; }
        else if (!strcmp(argv[i], "--cohort")      && i+1<argc) { cohort      = strtol(argv[++i],NULL,10); }
        else if (!strcmp(argv[i], "--seed")        && i+1<argc) { seed        = strtoull(argv[++i],NULL,10); }
        else if (!strcmp(argv[i], "--shard-index") && i+1<argc) { shard_index = strtol(argv[++i],NULL,10); }
        else if (!strcmp(argv[i], "--shard-count") && i+1<argc) { shard_count = strtol(argv[++i],NULL,10); }
        else if (!strcmp(argv[i], "--mean-dose")   && i+1<argc) { mean_dose   = strtod(argv[++i],NULL); }
        else if (!strcmp(argv[i], "--dose-sigma")  && i+1<argc) { dose_sigma  = strtod(argv[++i],NULL); }
        else if (!strcmp(argv[i], "--frac-c2")     && i+1<argc) { frac_c2     = strtod(argv[++i],NULL); }
        else if (!strcmp(argv[i], "--counts-only"))              { counts_only = 1; }
        else { fprintf(stderr, "error: unknown arg: %s\n", argv[i]); usage(argv[0]); return 2; }
    }
    if (!out_dir || cohort <= 0) { usage(argv[0]); return 2; }
    if (shard_count < 1 || shard_index < 0 || shard_index >= shard_count) {
        fprintf(stderr, "error: shard_index=%ld out of [0, %ld)\n", shard_index, shard_count);
        return 2;
    }

    long shard_size     = cohort / shard_count;
    long patient_start  = shard_index * shard_size;
    long patient_end    = (shard_index == shard_count - 1) ? cohort : patient_start + shard_size;
    long n              = patient_end - patient_start;
    size_t bytes        = (size_t)n * sizeof(float);

    float *c1_buf  = counts_only ? NULL : (float *)malloc(bytes);
    float *c2_buf  = counts_only ? NULL : (float *)malloc(bytes);
    float *v11_buf = counts_only ? NULL : (float *)malloc(bytes);
    float *v22_buf = counts_only ? NULL : (float *)malloc(bytes);
    if (!counts_only && (!c1_buf || !c2_buf || !v11_buf || !v22_buf)) {
        fprintf(stderr, "error: malloc(%zu)\n", bytes); return 1;
    }

    // GUM log-normal variance multiplier: Var[X] = E[X]² × (exp(σ²) − 1)
    // For a specific realisation x, we treat u(x) = x × sqrt(exp(σ²)−1).
    const float lnvar_mult = (float)(exp(dose_sigma * dose_sigma) - 1.0);

    // CPU-analytic digests of the FINAL state (after 4 Euler + GUM steps).
    uint64_t c1f_dig = 0xcbf29ce484222325ULL;
    uint64_t v11f_dig = 0xcbf29ce484222325ULL;
    // Initial-state digests for the runner to verify input loading.
    uint64_t c1i_dig = 0xcbf29ce484222325ULL;
    uint64_t v11i_dig = 0xcbf29ce484222325ULL;

    // GUM aggregate for budget CSV computation.
    double sum_v11_final = 0.0, sum_c1_final = 0.0, sum_c1sq_final = 0.0;

    static const uint64_t SPLITMIX_INC = 0x9E3779B97F4A7C15ULL;
    uint64_t state = seed + (uint64_t)patient_start * 4ULL * SPLITMIX_INC;
    const double ln_mean = log(mean_dose);

    for (long i = 0; i < n; i++) {
        double z1 = next_normal(&state);
        // consume second Box-Muller pair for skip-ahead alignment
        splitmix64_next(&state);
        splitmix64_next(&state);

        float c1_init = (float)exp(ln_mean + dose_sigma * z1);
        float c2_init = (float)(frac_c2 * (double)c1_init);

        // GUM initial variances (exact log-normal formula, float arithmetic)
        float v11_init = c1_init * c1_init * lnvar_mult;
        float v22_init = (float)(frac_c2 * frac_c2) * v11_init;
        float v12_init = 0.0f;  // compartments start uncorrelated

        // CPU analytic: propagate 4 Euler steps
        float c1 = c1_init, c2 = c2_init;
        float v11 = v11_init, v12 = v12_init, v22 = v22_init;
        for (int s = 0; s < N_STEPS; s++) {
            float c1n, c2n, v11n, v12n, v22n;
            euler_step_f32(c1, c2, &c1n, &c2n);
            gum_step(v11, v12, v22, &v11n, &v12n, &v22n);
            c1 = c1n; c2 = c2n;
            v11 = v11n; v12 = v12n; v22 = v22n;
        }

        // accumulate for budget
        sum_c1_final  += (double)c1;
        sum_c1sq_final += (double)c1 * (double)c1;
        sum_v11_final += (double)v11;

        // FNV-1a 64-bit digests (byte-wise, must match scripts/gpu/kaxi_ptx_runner.c).
        unsigned char c1i_bytes[4], v11i_bytes[4], c1f_bytes[4], v11f_bytes[4];
        memcpy(c1i_bytes,  &c1_init,  4);
        memcpy(v11i_bytes, &v11_init, 4);
        memcpy(c1f_bytes,  &c1,       4);
        memcpy(v11f_bytes, &v11,      4);
        for (int b = 0; b < 4; b++) {
            c1i_dig  = (c1i_dig  ^ c1i_bytes[b])  * 0x100000001b3ULL;
            v11i_dig = (v11i_dig ^ v11i_bytes[b]) * 0x100000001b3ULL;
            c1f_dig  = (c1f_dig  ^ c1f_bytes[b])  * 0x100000001b3ULL;
            v11f_dig = (v11f_dig ^ v11f_bytes[b]) * 0x100000001b3ULL;
        }

        if (!counts_only) {
            c1_buf[i]  = c1_init;
            c2_buf[i]  = c2_init;
            v11_buf[i] = v11_init;
            v22_buf[i] = v22_init;
        }
    }

    if (!counts_only) {
        char path[2048];
        snprintf(path, sizeof(path), "%s/init_c1.bin",  out_dir);
        if (write_all(path, c1_buf,  bytes) != 0) { fprintf(stderr, "error: write %s\n", path); return 1; }
        snprintf(path, sizeof(path), "%s/init_c2.bin",  out_dir);
        if (write_all(path, c2_buf,  bytes) != 0) { fprintf(stderr, "error: write %s\n", path); return 1; }
        snprintf(path, sizeof(path), "%s/init_v11.bin", out_dir);
        if (write_all(path, v11_buf, bytes) != 0) { fprintf(stderr, "error: write %s\n", path); return 1; }
        snprintf(path, sizeof(path), "%s/init_v22.bin", out_dir);
        if (write_all(path, v22_buf, bytes) != 0) { fprintf(stderr, "error: write %s\n", path); return 1; }
    }

    // population statistics for ISO budget
    double mean_c1_final   = sum_c1_final / (double)n;
    double var_c1_pop      = sum_c1sq_final / (double)n - mean_c1_final * mean_c1_final;
    double stdev_c1_pop    = sqrt(var_c1_pop > 0.0 ? var_c1_pop : 0.0);
    double mean_v11_final  = sum_v11_final / (double)n;
    double mean_u_b        = sqrt(mean_v11_final > 0.0 ? mean_v11_final : 0.0);
    // u_c = sqrt(u_B² + u_A²) combining dosing uncertainty and population variability
    double u_c             = sqrt(mean_v11_final + var_c1_pop);
    double u95             = 2.0 * u_c;
    double pct_type_b      = (u_c > 0.0) ? 100.0 * mean_v11_final / (mean_v11_final + var_c1_pop) : 0.0;
    double pct_type_a      = 100.0 - pct_type_b;

    char path[2048];
    snprintf(path, sizeof(path), "%s/expected.summary", out_dir);
    FILE *fs = fopen(path, "w");
    if (!fs) { fprintf(stderr, "error: write %s\n", path); return 1; }
    fprintf(fs,
        "cohort=%ld\n"
        "seed=%llu\n"
        "shard_index=%ld\n"
        "shard_count=%ld\n"
        "patient_start=%ld\n"
        "patient_end=%ld\n"
        "shard_patients=%ld\n"
        "c1_init_digest=%016llx\n"
        "v11_init_digest=%016llx\n"
        "c1_final_digest=%016llx\n"
        "v11_final_digest=%016llx\n"
        "mean_dose=%.17g\n"
        "dose_sigma=%.17g\n"
        "frac_c2=%.17g\n"
        "lnvar_mult=%.17g\n"
        "k_in=%.17g\n"
        "k12=%.17g\n"
        "k21=%.17g\n"
        "dt=%.17g\n"
        "n_steps=%d\n"
        "mean_c1_final=%.17g\n"
        "stdev_c1_pop=%.17g\n"
        "mean_u_b=%.17g\n"
        "u_c=%.17g\n"
        "u95_k2=%.17g\n"
        "pct_type_b=%.6g\n"
        "pct_type_a=%.6g\n",
        cohort, (unsigned long long)seed,
        shard_index, shard_count,
        patient_start, patient_end, n,
        (unsigned long long)c1i_dig, (unsigned long long)v11i_dig,
        (unsigned long long)c1f_dig, (unsigned long long)v11f_dig,
        mean_dose, dose_sigma, frac_c2, (double)lnvar_mult,
        (double)K_IN, (double)K12, (double)K21, (double)DT, N_STEPS,
        mean_c1_final, stdev_c1_pop, mean_u_b, u_c, u95,
        pct_type_b, pct_type_a);
    fclose(fs);

    fprintf(stdout,
        "kaxi_pbpk_2comp_gum_sampler: cohort=%ld seed=%llu "
        "shard=%ld/%ld patients=%ld..%ld "
        "c1i_dig=%016llx v11i_dig=%016llx "
        "c1f_dig=%016llx v11f_dig=%016llx "
        "mean_u_b=%.4f u95=%.4f pct_type_b=%.1f%%\n",
        cohort, (unsigned long long)seed,
        shard_index, shard_count, patient_start, patient_end,
        (unsigned long long)c1i_dig, (unsigned long long)v11i_dig,
        (unsigned long long)c1f_dig, (unsigned long long)v11f_dig,
        mean_u_b, u95, pct_type_b);

    free(c1_buf); free(c2_buf); free(v11_buf); free(v22_buf);
    return 0;
}
