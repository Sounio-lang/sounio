// kaxi_pbpk_2comp_sampler — Phase X input generator for the K-AXI 2-compartment
// PBPK pipeline. Produces init.mem.bin (C₁) + init.var.bin (C₂) as raw binary
// f32 streams matching kaxi_ptx_runner's --mode epistemic layout:
//   param_mem[gid] = C₁[gid]   (blood compartment concentration)
//   param_var[gid] = C₂[gid]   (tissue compartment concentration)
//
// CPU analytic reference runs 4 Euler steps of the same 2-comp ODE to predict
// the post-GPU classification. The gate decision is: C₁_out ∈ [window_low, window_high].
//
// ODE constants (must match kaxi_emit_pbpk_2comp_4step_mb_asm in kretikos_emit_kaxi.sio):
//   k_in = 1.0f, k12 = 0.5f, k21 = 0.25f, dt = 0.5f
//
// Initial condition distributions (rapamycin Cypher-stent PBPK, Ferron 1997):
//   C₁: log-normal(mean_dose, dose_sigma) — blood compartment
//   C₂: frac_c2 * C₁ — tissue compartment (default 0.1 = 10% initial tissue load)
//
// Default therapeutic window [0.4, 2.5] (rapamycin in PK units, ~5-15 ng/mL scaled).
//
// Shard skip-ahead: same splitmix64 O(1) mechanism as kaxi_pbpk_sampler.c.

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>

// ODE constants — must match kretikos_emit_kaxi.sio kaxi_emit_pbpk_2comp_4step_mb_asm
#define K_IN  1.0f
#define K12   0.5f
#define K21   0.25f
#define DT    0.5f
#define N_STEPS 4

static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double next_uniform(uint64_t *state) {
    uint64_t r = splitmix64_next(state) >> 11;
    return ((double)r + 1.0) * (1.0 / 9007199254740993.0);
}

static double next_normal(uint64_t *state) {
    double u1 = next_uniform(state);
    double u2 = next_uniform(state);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.141592653589793238 * u2);
}

// Run N_STEPS Euler steps in f32 — must match GPU PTX exactly.
static void euler_4step(float c1, float c2, float *c1_out, float *c2_out) {
    for (int s = 0; s < N_STEPS; s++) {
        float kc1 = K12 * c1;
        float kc2 = K21 * c2;
        float d1  = (K_IN - kc1) + kc2;
        float d2  = kc1 - kc2;
        c1 = c1 + DT * d1;
        c2 = c2 + DT * d2;
    }
    *c1_out = c1;
    *c2_out = c2;
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
        "  --out-dir DIR        directory for init.mem.bin, init.var.bin, expected.summary\n"
        "  --cohort N           TOTAL cohort size (across all shards)\n"
        "  --seed N             PRNG seed (default 42)\n"
        "  --shard-index K      shard to generate (0-based, default 0)\n"
        "  --shard-count S      total number of shards (default 1)\n"
        "  --mean-dose F        log-normal location for C₁ (default 1.0, PK units)\n"
        "  --dose-sigma F       log-normal scale for C₁ (default 0.50)\n"
        "  --frac-c2 F          C₂ = frac_c2 * C₁ initial (default 0.1)\n"
        "  --window-low F       therapeutic window lower bound (default 0.4)\n"
        "  --window-high F      therapeutic window upper bound (default 2.5)\n"
        "  --counts-only        skip writing init.mem.bin / init.var.bin\n"
        "\n"
        "Sharding: --shard-index K --shard-count S generates patients [K*floor(N/S),\n"
        "  (K+1)*floor(N/S)) with exact PRNG skip-ahead. Last shard absorbs remainder.\n"
        "\n"
        "Outputs (raw binary f32, little-endian):\n"
        "  init.mem.bin   shard_patients × 4 bytes (C₁ values)\n"
        "  init.var.bin   shard_patients × 4 bytes (C₂ values)\n"
        "  expected.summary  text: in_window, out_of_window, analytic digests\n",
        prog);
}

int main(int argc, char **argv) {
    const char *out_dir = NULL;
    long cohort = 0;
    uint64_t seed = 42;
    double mean_dose = 1.0;
    double dose_sigma = 0.50;
    double frac_c2 = 0.1;
    double window_low = 0.4;
    double window_high = 2.5;
    long shard_index = 0;
    long shard_count = 1;
    int counts_only = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else if (!strcmp(argv[i], "--out-dir")     && i+1 < argc) { out_dir     = argv[++i]; }
        else if (!strcmp(argv[i], "--cohort")      && i+1 < argc) { cohort      = strtol(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--seed")        && i+1 < argc) { seed        = strtoull(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--shard-index") && i+1 < argc) { shard_index = strtol(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--shard-count") && i+1 < argc) { shard_count = strtol(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--mean-dose")   && i+1 < argc) { mean_dose   = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--dose-sigma")  && i+1 < argc) { dose_sigma  = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--frac-c2")     && i+1 < argc) { frac_c2     = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--window-low")  && i+1 < argc) { window_low  = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--window-high") && i+1 < argc) { window_high = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--counts-only"))               { counts_only = 1; }
        else { fprintf(stderr, "error: unknown arg: %s\n", argv[i]); usage(argv[0]); return 2; }
    }
    if (!out_dir || cohort <= 0) { usage(argv[0]); return 2; }
    if (shard_count < 1 || shard_index < 0 || shard_index >= shard_count) {
        fprintf(stderr, "error: shard_index=%ld must be in [0, shard_count=%ld)\n",
                shard_index, shard_count);
        return 2;
    }

    long shard_size    = cohort / shard_count;
    long patient_start = shard_index * shard_size;
    long patient_end   = (shard_index == shard_count - 1) ? cohort : patient_start + shard_size;
    long shard_patients = patient_end - patient_start;
    size_t bytes = (size_t)shard_patients * sizeof(float);

    float *c1_buf = counts_only ? NULL : (float *)malloc(bytes);
    float *c2_buf = counts_only ? NULL : (float *)malloc(bytes);
    if (!counts_only && (!c1_buf || !c2_buf)) {
        fprintf(stderr, "error: malloc(%zu) failed\n", bytes); return 1;
    }

    uint64_t mem_digest = 0xcbf29ce484222325ULL;
    uint64_t var_digest = 0xcbf29ce484222325ULL;
    long in_window = 0, out_of_window = 0;

    // Skip-ahead: each patient consumes exactly 4 splitmix64 steps
    // (2 for Box-Muller z1, 2 for Box-Muller z2 — though we only use z1).
    // We draw 2 for Box-Muller but only use the first deviate per patient.
    // To keep parity with future multi-variate sampling, advance by 4 per patient.
    static const uint64_t SPLITMIX_INC = 0x9E3779B97F4A7C15ULL;
    uint64_t state = seed + (uint64_t)patient_start * 4ULL * SPLITMIX_INC;
    const double ln_mean = log(mean_dose);
    const float wl = (float)window_low, wh = (float)window_high;

    for (long i = 0; i < shard_patients; i++) {
        // Generate C₁ from log-normal; C₂ = frac_c2 * C₁.
        double z1 = next_normal(&state);
        // Consume second Box-Muller pair for skip-ahead alignment.
        splitmix64_next(&state);
        splitmix64_next(&state);
        float c1_init = (float)exp(ln_mean + dose_sigma * z1);
        float c2_init = (float)(frac_c2 * (double)c1_init);

        // CPU analytic: 4 Euler steps in f32 (must match GPU PTX).
        float c1_out, c2_out;
        euler_4step(c1_init, c2_init, &c1_out, &c2_out);

        // Gate: C₁_out in therapeutic window?
        if (c1_out >= wl && c1_out <= wh) in_window++; else out_of_window++;

        if (!counts_only) {
            c1_buf[i] = c1_init;
            c2_buf[i] = c2_init;
            uint32_t c1_bits, c2_bits;
            memcpy(&c1_bits, &c1_init, 4);
            memcpy(&c2_bits, &c2_init, 4);
            mem_digest ^= (uint64_t)c1_bits; mem_digest *= 0x100000001b3ULL;
            var_digest ^= (uint64_t)c2_bits; var_digest *= 0x100000001b3ULL;
        }
    }

    if (!counts_only) {
        char path[2048];
        snprintf(path, sizeof(path), "%s/init.mem.bin", out_dir);
        if (write_all(path, c1_buf, bytes) != 0) {
            fprintf(stderr, "error: write %s: %s\n", path, strerror(errno)); return 1;
        }
        snprintf(path, sizeof(path), "%s/init.var.bin", out_dir);
        if (write_all(path, c2_buf, bytes) != 0) {
            fprintf(stderr, "error: write %s: %s\n", path, strerror(errno)); return 1;
        }
    }

    char path[2048];
    snprintf(path, sizeof(path), "%s/expected.summary", out_dir);
    FILE *fs = fopen(path, "w");
    if (!fs) { fprintf(stderr, "error: write %s: %s\n", path, strerror(errno)); return 1; }
    fprintf(fs,
        "cohort=%ld\n"
        "seed=%llu\n"
        "shard_index=%ld\n"
        "shard_count=%ld\n"
        "patient_start=%ld\n"
        "patient_end=%ld\n"
        "shard_patients=%ld\n"
        "in_window=%ld\n"
        "out_of_window=%ld\n"
        "input_mem_digest=%016llx\n"
        "input_var_digest=%016llx\n"
        "window_low=%.17g\n"
        "window_high=%.17g\n"
        "mean_dose=%.17g\n"
        "dose_sigma=%.17g\n"
        "frac_c2=%.17g\n"
        "k_in=%.17g\n"
        "k12=%.17g\n"
        "k21=%.17g\n"
        "dt=%.17g\n"
        "n_steps=%d\n",
        cohort, (unsigned long long)seed,
        shard_index, shard_count,
        patient_start, patient_end, shard_patients,
        in_window, out_of_window,
        (unsigned long long)mem_digest,
        (unsigned long long)var_digest,
        window_low, window_high,
        mean_dose, dose_sigma, frac_c2,
        (double)K_IN, (double)K12, (double)K21, (double)DT, N_STEPS);
    fclose(fs);

    fprintf(stdout,
        "kaxi_pbpk_2comp_sampler: cohort=%ld seed=%llu "
        "shard=%ld/%ld patients=%ld..%ld "
        "in_window=%ld out_of_window=%ld "
        "mem_digest=%016llx var_digest=%016llx\n",
        cohort, (unsigned long long)seed,
        shard_index, shard_count, patient_start, patient_end,
        in_window, out_of_window,
        (unsigned long long)mem_digest, (unsigned long long)var_digest);

    free(c1_buf); free(c2_buf);
    return 0;
}
