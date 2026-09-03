// kaxi_pbpk_rapamycin_sampler — Phase X input generator for the K-AXI
// rapamycin 1-compartment epistemic PBPK pipeline.
//
// Kernel: kaxi_emit_pbpk_rapamycin_1comp_epistemic_asm() in kretikos_emit_kaxi.sio
//   - Loads C[gid] from param_mem[gid], σ²(C[gid]) from param_var[gid]
//   - Computes C_scaled = C * 3.125 (converts internal PK units → ng/mL)
//   - GUM propagation: σ²(C_scaled) = 3.125² × σ²(C)
//   - Gate: if 5.0 ≤ C_scaled ≤ 15.0 (rapamycin therapeutic window), write
//     C_scaled to mem[gid]; else write NaN sentinel (0x7FC00000)
//
// CPU analytic: same gate decision in f32 arithmetic. Outputs in_budget
// (= in-window) and nan_count (= out-of-window) for truth-claim comparison.
//
// Distributional binding (rapamycin/Cypher-stent PBPK, Ferron 1997):
//   C ~ log-normal(mean_dose, dose_sigma) in internal PK units
//   [C * 3.125 ng/mL] centred near 6.25 ng/mL (therapeutic midpoint)
//   σ²(C) ~ truncated-normal(sigma0_mean, sigma0_sigma) at zero
//
// Shard skip-ahead: splitmix64 O(1) skip-ahead, 4 draws per patient.
//
// Outputs (f32, raw binary):
//   init.mem.bin   shard_patients × 4 bytes  (C values in PK units)
//   init.var.bin   shard_patients × 4 bytes  (σ²(C) values)
//   expected.summary  text: in_budget, nan_count, shard metadata

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>

// Kernel constants — must match kretikos_emit_kaxi.sio
// pbpk_rapamycin_1comp_epistemic_asm()
#define RAPA_SCALE    3.125f   // 0f40480000 — PK units → ng/mL conversion
#define RAPA_WIN_LOW  5.0f     // 0f40A00000 — therapeutic window lower bound
#define RAPA_WIN_HIGH 15.0f    // 0f41700000 — therapeutic window upper bound
#define RAPA_NAN      0x7FC00000u  // f32 quiet-NaN sentinel for out-of-window

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
        "  --mean-dose F        log-normal mean for C in PK units (default 2.0)\n"
        "                       2.0 PK × 3.125 = 6.25 ng/mL (mid therapeutic window)\n"
        "  --dose-sigma F       log-normal sigma for C (default 0.50, CV~50%%)\n"
        "  --sigma0-mean F      mean σ²(C) across cohort (default 0.50)\n"
        "  --sigma0-sigma F     stddev of σ²(C) (default 0.15)\n"
        "  --counts-only        skip writing init.mem.bin / init.var.bin\n"
        "\n"
        "Sharding: last shard absorbs remainder; skip-ahead is O(1).\n"
        "\n"
        "Gate: kernel writes NaN if C × 3.125 ∉ [5.0, 15.0] ng/mL.\n"
        "CPU analytic replicates exact f32 arithmetic (no FMA).\n",
        prog);
}

int main(int argc, char **argv) {
    const char *out_dir = NULL;
    long cohort = 0;
    uint64_t seed = 42;
    double mean_dose = 2.0;
    double dose_sigma = 0.50;
    double sigma0_mean = 0.50;
    double sigma0_sigma = 0.15;
    long shard_index = 0;
    long shard_count = 1;
    int counts_only = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else if (!strcmp(argv[i], "--out-dir")     && i+1 < argc) { out_dir      = argv[++i]; }
        else if (!strcmp(argv[i], "--cohort")      && i+1 < argc) { cohort       = strtol(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--seed")        && i+1 < argc) { seed         = strtoull(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--shard-index") && i+1 < argc) { shard_index  = strtol(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--shard-count") && i+1 < argc) { shard_count  = strtol(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--mean-dose")   && i+1 < argc) { mean_dose    = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--dose-sigma")  && i+1 < argc) { dose_sigma   = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--sigma0-mean") && i+1 < argc) { sigma0_mean  = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--sigma0-sigma")&& i+1 < argc) { sigma0_sigma = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--counts-only"))               { counts_only  = 1; }
        else { fprintf(stderr, "error: unknown arg: %s\n", argv[i]); usage(argv[0]); return 2; }
    }
    if (!out_dir || cohort <= 0) { usage(argv[0]); return 2; }
    if (shard_count < 1 || shard_index < 0 || shard_index >= shard_count) {
        fprintf(stderr, "error: shard_index=%ld must be in [0, shard_count=%ld)\n",
                shard_index, shard_count);
        return 2;
    }

    long shard_size     = cohort / shard_count;
    long patient_start  = shard_index * shard_size;
    long patient_end    = (shard_index == shard_count - 1) ? cohort : patient_start + shard_size;
    long shard_patients = patient_end - patient_start;
    size_t bytes = (size_t)shard_patients * sizeof(float);

    float *c_buf  = counts_only ? NULL : (float *)malloc(bytes);
    float *s2_buf = counts_only ? NULL : (float *)malloc(bytes);
    if (!counts_only && (!c_buf || !s2_buf)) {
        fprintf(stderr, "error: malloc(%zu) failed\n", bytes); return 1;
    }

    uint64_t mem_digest = 0xcbf29ce484222325ULL;
    uint64_t var_digest = 0xcbf29ce484222325ULL;
    long in_budget = 0, nan_count = 0;

    // Skip-ahead: 4 splitmix64 steps per patient (2 Box-Muller for z1, 2 for z2).
    static const uint64_t SPLITMIX_INC = 0x9E3779B97F4A7C15ULL;
    uint64_t state = seed + (uint64_t)patient_start * 4ULL * SPLITMIX_INC;
    const double ln_mean = log(mean_dose);

    for (long i = 0; i < shard_patients; i++) {
        double z1 = next_normal(&state);
        double z2 = next_normal(&state);

        // C in internal PK units, log-normal
        float c = (float)exp(ln_mean + dose_sigma * z1);

        // σ²(C): truncated-normal at 0
        double s2d = sigma0_mean + sigma0_sigma * z2;
        if (s2d < 1e-9) s2d = 1e-9;
        float s2 = (float)s2d;

        // CPU analytic: replicate f32 kernel arithmetic (no FMA).
        // Kernel: f10 = c * RAPA_SCALE; gate: 5.0 ≤ f10 ≤ 15.0
        float c_scaled = c * RAPA_SCALE;
        int gate = (c_scaled >= RAPA_WIN_LOW && c_scaled <= RAPA_WIN_HIGH);
        if (gate) in_budget++; else nan_count++;

        if (!counts_only) {
            c_buf[i]  = c;
            s2_buf[i] = s2;
            uint32_t cb, sb;
            memcpy(&cb, &c,  4);
            memcpy(&sb, &s2, 4);
            mem_digest ^= (uint64_t)cb; mem_digest *= 0x100000001b3ULL;
            var_digest ^= (uint64_t)sb; var_digest *= 0x100000001b3ULL;
        }
    }

    if (!counts_only) {
        char path[2048];
        snprintf(path, sizeof(path), "%s/init.mem.bin", out_dir);
        if (write_all(path, c_buf, bytes) != 0) {
            fprintf(stderr, "error: write %s: %s\n", path, strerror(errno)); return 1;
        }
        snprintf(path, sizeof(path), "%s/init.var.bin", out_dir);
        if (write_all(path, s2_buf, bytes) != 0) {
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
        "in_budget=%ld\n"
        "nan_count=%ld\n"
        "input_mem_digest=%016llx\n"
        "input_var_digest=%016llx\n"
        "mean_dose=%.17g\n"
        "dose_sigma=%.17g\n"
        "sigma0_mean=%.17g\n"
        "sigma0_sigma=%.17g\n"
        "rapa_scale=%.17g\n"
        "window_low=%.17g\n"
        "window_high=%.17g\n",
        cohort, (unsigned long long)seed,
        shard_index, shard_count,
        patient_start, patient_end, shard_patients,
        in_budget, nan_count,
        (unsigned long long)mem_digest,
        (unsigned long long)var_digest,
        mean_dose, dose_sigma, sigma0_mean, sigma0_sigma,
        (double)RAPA_SCALE, (double)RAPA_WIN_LOW, (double)RAPA_WIN_HIGH);
    fclose(fs);

    fprintf(stdout,
        "kaxi_pbpk_rapamycin_sampler: cohort=%ld seed=%llu "
        "shard=%ld/%ld patients=%ld..%ld "
        "in_budget=%ld nan_count=%ld "
        "mem_digest=%016llx var_digest=%016llx\n",
        cohort, (unsigned long long)seed,
        shard_index, shard_count, patient_start, patient_end,
        in_budget, nan_count,
        (unsigned long long)mem_digest, (unsigned long long)var_digest);

    free(c_buf); free(s2_buf);
    return 0;
}
