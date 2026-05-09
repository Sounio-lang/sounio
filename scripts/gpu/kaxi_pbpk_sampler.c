// kaxi_pbpk_sampler — Phase W input generator for the K-AXI compute-then-gate
// pipeline. Produces init.mem.bin + init.var.bin as raw binary u64 streams
// matching the layout consumed by kaxi_ptx_runner via --init-file/--init-var-file.
//
// Each "patient" is one 8-byte slot in mem (drug concentration v as f64
// reinterpreted to u64) and one 8-byte slot in var (prior variance σ²₀ as
// f64 reinterpreted to u64). The K-AXI multi-block kernel emitted by
// kaxi_emit_vec_sqrt_gate_var_mb_asm() reads paired (v, σ²₀) per gid.
//
// Distributional binding (Phase W requirement: PBPK fixture corpus):
//
//   Source: stdlib/darwin_pbpk/drugs/rapamycin.sio:65-99
//   - rapamycin_mean_params() — Ferron 1997, Schreiber 1991, Lampen 1998
//   - CYP3A4-driven CV~60% on clearance → log-normal σ on dose
//   - Confidence floor 0.4-0.95 across 14 compartments → σ²₀ scaling
//
// Defaults reflect the rapamycin/Cypher-stent dissertation chapter, tuned
// so σ²(√v) = σ²₀/(4v) clusters near the gate threshold (1.0) and the
// cohort splits roughly 60/40 in-budget vs NaN — consistent with Phase V
// A5000 measurements (645/379 of 1024):
//   --mean-dose      1.0   (representative brain concentration, PK units)
//   --dose-sigma     0.50  (CV~50% on log-scale, CYP3A4 inter-individual)
//   --sigma0-mean    3.0   (prior variance baseline, confidence~0.65)
//   --sigma0-sigma   1.0   (variance heterogeneity)
//
// Determinism: PCG-style splitmix64 from --seed; identical (seed, cohort)
// reproduces byte-identical buffers across machines.
//
// IMPORTANT — what the in_budget/nan_count summary fields mean:
//   These are CPU-ANALYTIC reference counts: σ²(√v) = σ²₀/(4·v) < threshold.
//   The current Phase V/W PTX kernel does NOT actually compute sqrt at the
//   GPU level (`// unhandled` for the gvr opcode in the lowered PTX), so the
//   GPU's mem buffer comes back as deterministic-zero. The in_budget split
//   describes what the kernel WOULD gate if it lowered sqrt+gvr correctly.
//   When that landing happens (a future emitter phase), the GPU mem_digest
//   will reflect this split byte-for-byte. Until then, treat in_budget as
//   "the analytic gate decision the architecture is wired to compute,"
//   not "patients the GPU actually classified in this run."

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>

static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// uniform double in (0, 1) — open interval, never returns 0 or 1
static double next_uniform(uint64_t *state) {
    uint64_t r = splitmix64_next(state) >> 11;       // 53 random bits
    return ((double)r + 1.0) * (1.0 / 9007199254740993.0);
}

// Box-Muller standard normal — pulls two uniforms, returns one normal
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
        "  --cohort N           number of patients (must be > 0)\n"
        "  --seed N             PRNG seed (default 42)\n"
        "  --mean-dose F        log-normal location for v (default 2.0)\n"
        "  --dose-sigma F       log-normal scale for v (default 0.30)\n"
        "  --sigma0-mean F      mean of prior variance σ²₀ (default 0.50)\n"
        "  --sigma0-sigma F     stddev of σ²₀ across cohort (default 0.15)\n"
        "  --threshold F        gate threshold on σ²(sqrt) (default 1.0, matches Phase V)\n"
        "\n"
        "Outputs (raw binary, little-endian):\n"
        "  init.mem.bin   N × 8 bytes (f64 v reinterpret)\n"
        "  init.var.bin   N × 8 bytes (f64 σ²₀ reinterpret)\n"
        "  expected.summary    text: cohort, in_budget, nan_count, mem_digest, var_digest\n",
        prog);
}

int main(int argc, char **argv) {
    const char *out_dir = NULL;
    long cohort = 0;
    uint64_t seed = 42;
    double mean_dose = 1.0;
    double dose_sigma = 0.50;
    double sigma0_mean = 3.0;
    double sigma0_sigma = 1.0;
    double threshold = 1.0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) { out_dir = argv[++i]; }
        else if (!strcmp(argv[i], "--cohort") && i + 1 < argc) { cohort = strtol(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) { seed = strtoull(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--mean-dose") && i + 1 < argc) { mean_dose = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--dose-sigma") && i + 1 < argc) { dose_sigma = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--sigma0-mean") && i + 1 < argc) { sigma0_mean = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--sigma0-sigma") && i + 1 < argc) { sigma0_sigma = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--threshold") && i + 1 < argc) { threshold = strtod(argv[++i], NULL); }
        else { fprintf(stderr, "error: unknown arg: %s\n", argv[i]); usage(argv[0]); return 2; }
    }
    if (!out_dir || cohort <= 0) { usage(argv[0]); return 2; }

    size_t bytes = (size_t)cohort * sizeof(uint64_t);
    double *v_buf = (double *)malloc(bytes);
    double *s_buf = (double *)malloc(bytes);
    if (!v_buf || !s_buf) { fprintf(stderr, "error: malloc(%zu) failed\n", bytes); return 1; }

    // FNV-1a 64-bit accumulators for output digests
    uint64_t mem_digest = 0xcbf29ce484222325ULL;
    uint64_t var_digest = 0xcbf29ce484222325ULL;
    long in_budget = 0;
    long nan_count = 0;

    uint64_t state = seed;
    const double ln_mean = log(mean_dose);
    for (long i = 0; i < cohort; i++) {
        double z1 = next_normal(&state);
        double z2 = next_normal(&state);
        // dose: log-normal sample around mean_dose with log-sigma dose_sigma
        double v = exp(ln_mean + dose_sigma * z1);
        // σ²₀: truncated normal at 0 (positive variance) around sigma0_mean
        double s = sigma0_mean + sigma0_sigma * z2;
        if (s < 1e-9) s = 1e-9;
        v_buf[i] = v;
        s_buf[i] = s;

        // CPU reference: σ²(√v) = σ²₀ / (4·v); in-budget iff < threshold
        double sigma_sq_sqrt = s / (4.0 * v);
        int gate_pass = (sigma_sq_sqrt < threshold);
        if (gate_pass) in_budget++; else nan_count++;

        // Roll digests over the input buffers (verification compares input
        // determinism, not GPU output — actual GPU output is digested in the
        // runner and compared cross-stream / cross-launch).
        uint64_t v_bits, s_bits;
        memcpy(&v_bits, &v, 8); memcpy(&s_bits, &s, 8);
        mem_digest ^= v_bits; mem_digest *= 0x100000001b3ULL;
        var_digest ^= s_bits; var_digest *= 0x100000001b3ULL;
    }

    char path[2048];
    snprintf(path, sizeof(path), "%s/init.mem.bin", out_dir);
    if (write_all(path, v_buf, bytes) != 0) {
        fprintf(stderr, "error: write %s: %s\n", path, strerror(errno)); return 1;
    }
    snprintf(path, sizeof(path), "%s/init.var.bin", out_dir);
    if (write_all(path, s_buf, bytes) != 0) {
        fprintf(stderr, "error: write %s: %s\n", path, strerror(errno)); return 1;
    }
    snprintf(path, sizeof(path), "%s/expected.summary", out_dir);
    FILE *fs = fopen(path, "w");
    if (!fs) { fprintf(stderr, "error: write %s: %s\n", path, strerror(errno)); return 1; }
    fprintf(fs,
        "cohort=%ld\n"
        "seed=%llu\n"
        "in_budget=%ld\n"
        "nan_count=%ld\n"
        "input_mem_digest=%016llx\n"
        "input_var_digest=%016llx\n"
        "threshold=%.17g\n"
        "mean_dose=%.17g\n"
        "dose_sigma=%.17g\n"
        "sigma0_mean=%.17g\n"
        "sigma0_sigma=%.17g\n",
        cohort,
        (unsigned long long)seed,
        in_budget, nan_count,
        (unsigned long long)mem_digest,
        (unsigned long long)var_digest,
        threshold, mean_dose, dose_sigma, sigma0_mean, sigma0_sigma);
    fclose(fs);

    fprintf(stdout,
        "kaxi_pbpk_sampler: cohort=%ld seed=%llu in_budget=%ld nan_count=%ld "
        "mem_digest=%016llx var_digest=%016llx\n",
        cohort, (unsigned long long)seed, in_budget, nan_count,
        (unsigned long long)mem_digest, (unsigned long long)var_digest);

    free(v_buf); free(s_buf);
    return 0;
}
