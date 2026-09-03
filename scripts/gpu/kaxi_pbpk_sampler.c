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
//   These are CPU-ANALYTIC reference counts: σ²(√v) = σ²₀/(4·v) < threshold
//   (for the default sqrt-gate kernel) or Cp = D*scale ∈ [lo,hi] (for
//   pbpk_rapamycin_1comp_epistemic and future PBPK kernels).
//
//   Phase X (type=f32): the f32e vec_sqrt_gate_var_mb kernel fully computes
//   the epistemic gate on the GPU. The GPU outputs sqrt(v) for in-budget
//   patients and NaN sentinel (0x7FC00000) for those gated out. The runner
//   counts NaN sentinels to derive nan_count/in_budget. The truth-claim in
//   kretikos_kaxi_phase_w2_gate.sh compares these GPU counts against the
//   CPU-analytic reference, and they match exactly when both use --type f32.

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
        "  --cohort N           TOTAL cohort size (across all shards, must be > 0)\n"
        "  --seed N             PRNG seed (default 42)\n"
        "  --shard-index K      shard to generate (0-based, default 0)\n"
        "  --shard-count S      total number of shards (default 1 = no sharding)\n"
        "  --mean-dose F        log-normal location for v (default 2.0)\n"
        "  --dose-sigma F       log-normal scale for v (default 0.30)\n"
        "  --sigma0-mean F      mean of prior variance σ²₀ (default 0.50)\n"
        "  --sigma0-sigma F     stddev of σ²₀ across cohort (default 0.15)\n"
        "  --threshold F        gate threshold on σ²(sqrt) (default 1.0, matches Phase V)\n"
        "\n"
        "Sharding: --shard-index K --shard-count S generates patients [K*floor(N/S),\n"
        "  (K+1)*floor(N/S)) with exact PRNG skip-ahead — byte-identical to slicing\n"
        "  the full cohort. Last shard absorbs any remainder.\n"
        "\n"
        "Outputs (raw binary, little-endian):\n"
        "  init.mem.bin   shard_patients × elem bytes\n"
        "  init.var.bin   shard_patients × elem bytes\n"
        "  expected.summary    text: cohort, shard, in_budget, nan_count, digests\n",
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
    // Phase X.1: --type f32 emits 4-byte float per slot to match the f32e
    // PTX kernel's element stride. Default is f64 (8-byte) for backward
    // compatibility with Phase W gate's epistemic-u64 path.
    int type_f32 = 0;
    // Phase W.2: deterministic shard decomposition. shard_index K with
    // shard_count S generates patients [patient_start, patient_end) via
    // splitmix64 skip-ahead: state = seed + patient_start*4*INC (mod 2^64).
    // The factor 4 = 2 uniforms per Box-Muller call × 2 calls per patient.
    long shard_index = 0;
    long shard_count = 1;
    // Phase W.2: --counts-only skips writing init.mem.bin / init.var.bin.
    // Used for the full-cohort analytic reference where only in_budget /
    // nan_count are needed (avoids large file writes).
    int counts_only = 0;
    // Phase X: gate_type selects the CPU analytic used for the truth claim.
    //   0 = sqrt-gate (default): in_budget if σ²(√v) = σ²₀/(4v) < threshold
    //   1 = rapamycin_1comp: in_budget if v * pk_scale ∈ [pk_lo, pk_hi]
    // Corresponds to pbpk_rapamycin_1comp_epistemic kernel (scale=3.125, lo=5, hi=15).
    int gate_type = 0;
    double pk_scale = 3.125;
    double pk_lo = 5.0;
    double pk_hi = 15.0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) { out_dir = argv[++i]; }
        else if (!strcmp(argv[i], "--cohort") && i + 1 < argc) { cohort = strtol(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) { seed = strtoull(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--shard-index") && i + 1 < argc) { shard_index = strtol(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--shard-count") && i + 1 < argc) { shard_count = strtol(argv[++i], NULL, 10); }
        else if (!strcmp(argv[i], "--mean-dose") && i + 1 < argc) { mean_dose = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--dose-sigma") && i + 1 < argc) { dose_sigma = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--sigma0-mean") && i + 1 < argc) { sigma0_mean = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--sigma0-sigma") && i + 1 < argc) { sigma0_sigma = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--threshold") && i + 1 < argc) { threshold = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--gate") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "sqrt")) gate_type = 0;
            else if (!strcmp(argv[i], "rapamycin_1comp")) gate_type = 1;
            else { fprintf(stderr, "error: --gate must be sqrt or rapamycin_1comp\n"); return 2; }
        }
        else if (!strcmp(argv[i], "--pk-scale") && i + 1 < argc) { pk_scale = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--pk-lo") && i + 1 < argc) { pk_lo = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--pk-hi") && i + 1 < argc) { pk_hi = strtod(argv[++i], NULL); }
        else if (!strcmp(argv[i], "--type") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "f32")) type_f32 = 1;
            else if (!strcmp(argv[i], "f64")) type_f32 = 0;
            else { fprintf(stderr, "error: --type must be f32 or f64\n"); return 2; }
        }
        else if (!strcmp(argv[i], "--counts-only")) { counts_only = 1; }
        else { fprintf(stderr, "error: unknown arg: %s\n", argv[i]); usage(argv[0]); return 2; }
    }
    if (!out_dir || cohort <= 0) { usage(argv[0]); return 2; }
    if (shard_count < 1 || shard_index < 0 || shard_index >= shard_count) {
        fprintf(stderr, "error: shard_index=%ld must be in [0, shard_count=%ld)\n", shard_index, shard_count);
        return 2;
    }

    // Shard range: last shard absorbs remainder so all shards cover exactly cohort patients.
    long shard_size = cohort / shard_count;
    long patient_start = shard_index * shard_size;
    long patient_end   = (shard_index == shard_count - 1) ? cohort : patient_start + shard_size;
    long shard_patients = patient_end - patient_start;

    size_t elem = type_f32 ? sizeof(float) : sizeof(double);
    size_t bytes = (size_t)shard_patients * elem;
    // --counts-only: skip buffer allocation; loop streams without storage.
    double *v_buf = counts_only ? NULL : (double *)malloc((size_t)shard_patients * sizeof(double));
    double *s_buf = counts_only ? NULL : (double *)malloc((size_t)shard_patients * sizeof(double));
    if (!counts_only && (!v_buf || !s_buf)) {
        fprintf(stderr, "error: malloc(%zu) failed\n", bytes); return 1;
    }

    // FNV-1a 64-bit accumulators for output digests
    uint64_t mem_digest = 0xcbf29ce484222325ULL;
    uint64_t var_digest = 0xcbf29ce484222325ULL;
    long in_budget = 0;
    long nan_count = 0;

    // Advance PRNG to patient_start. Each patient consumes exactly 4 splitmix64
    // steps (2 for z1 via next_normal, 2 for z2). State after N steps from
    // seed S is S + N*INC (mod 2^64) — linear in N, so skip-ahead is O(1).
    static const uint64_t SPLITMIX_INC = 0x9E3779B97F4A7C15ULL;
    uint64_t state = seed + (uint64_t)patient_start * 4ULL * SPLITMIX_INC;
    const double ln_mean = log(mean_dose);
    for (long i = 0; i < shard_patients; i++) {
        double z1 = next_normal(&state);
        double z2 = next_normal(&state);
        // dose: log-normal sample around mean_dose with log-sigma dose_sigma
        double v = exp(ln_mean + dose_sigma * z1);
        // σ²₀: truncated normal at 0 (positive variance) around sigma0_mean
        double s = sigma0_mean + sigma0_sigma * z2;
        if (s < 1e-9) s = 1e-9;
        if (!counts_only) { v_buf[i] = v; s_buf[i] = s; }

        // CPU reference gate — two modes:
        //   sqrt (gate_type=0): σ²(√v) = σ²₀/(4v) < threshold
        //   rapamycin_1comp (gate_type=1): v * pk_scale ∈ [pk_lo, pk_hi]
        // For --type f32, run in float precision to match GPU rounding.
        int gate_pass;
        if (gate_type == 1) {
            if (type_f32) {
                float vf = (float)v, scf = (float)pk_scale;
                float cp = vf * scf;
                gate_pass = (cp >= (float)pk_lo && cp <= (float)pk_hi);
            } else {
                double cp = v * pk_scale;
                gate_pass = (cp >= pk_lo && cp <= pk_hi);
            }
        } else if (type_f32) {
            float vf = (float)v, sf = (float)s, thf = (float)threshold;
            float sigma_sq_sqrt = sf / (4.0f * vf);
            gate_pass = (sigma_sq_sqrt < thf);
        } else {
            double sigma_sq_sqrt = s / (4.0 * v);
            gate_pass = (sigma_sq_sqrt < threshold);
        }
        if (gate_pass) in_budget++; else nan_count++;

        // Roll digests (skipped in counts-only mode — no output buffers).
        if (!counts_only) {
            uint64_t v_bits, s_bits;
            memcpy(&v_bits, &v, 8); memcpy(&s_bits, &s, 8);
            mem_digest ^= v_bits; mem_digest *= 0x100000001b3ULL;
            var_digest ^= s_bits; var_digest *= 0x100000001b3ULL;
        }
    }

    // Convert to f32 and write init files — skipped in --counts-only mode.
    char path[2048];
    void *v_out = v_buf, *s_out = s_buf;
    float *vf_buf = NULL, *sf_buf = NULL;
    if (!counts_only) {
        if (type_f32) {
            vf_buf = (float *)malloc((size_t)shard_patients * sizeof(float));
            sf_buf = (float *)malloc((size_t)shard_patients * sizeof(float));
            if (!vf_buf || !sf_buf) { fprintf(stderr, "error: f32 buffer malloc\n"); return 1; }
            for (long i = 0; i < shard_patients; i++) { vf_buf[i] = (float)v_buf[i]; sf_buf[i] = (float)s_buf[i]; }
            v_out = vf_buf; s_out = sf_buf;
        }
        char path2[2048];
        snprintf(path2, sizeof(path2), "%s/init.mem.bin", out_dir);
        if (write_all(path2, v_out, bytes) != 0) {
            fprintf(stderr, "error: write %s: %s\n", path2, strerror(errno)); return 1;
        }
        snprintf(path2, sizeof(path2), "%s/init.var.bin", out_dir);
        if (write_all(path2, s_out, bytes) != 0) {
            fprintf(stderr, "error: write %s: %s\n", path2, strerror(errno)); return 1;
        }
    }
    snprintf(path, sizeof(path), "%s/expected.summary", out_dir);
    FILE *fs = fopen(path, "w");
    if (!fs) { fprintf(stderr, "error: write %s: %s\n", path, strerror(errno)); return 1; }
    fprintf(fs,
        "cohort=%ld\n"
        "seed=%llu\n"
        "type=%s\n"
        "shard_index=%ld\n"
        "shard_count=%ld\n"
        "patient_start=%ld\n"
        "patient_end=%ld\n"
        "shard_patients=%ld\n"
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
        type_f32 ? "f32" : "f64",
        shard_index, shard_count,
        patient_start, patient_end, shard_patients,
        in_budget, nan_count,
        (unsigned long long)mem_digest,
        (unsigned long long)var_digest,
        threshold, mean_dose, dose_sigma, sigma0_mean, sigma0_sigma);
    fclose(fs);

    fprintf(stdout,
        "kaxi_pbpk_sampler: cohort=%ld seed=%llu type=%s "
        "shard=%ld/%ld patients=%ld..%ld "
        "in_budget=%ld nan_count=%ld "
        "mem_digest=%016llx var_digest=%016llx\n",
        cohort, (unsigned long long)seed, type_f32 ? "f32" : "f64",
        shard_index, shard_count, patient_start, patient_end,
        in_budget, nan_count,
        (unsigned long long)mem_digest, (unsigned long long)var_digest);

    free(v_buf); free(s_buf);
    if (vf_buf) free(vf_buf);
    if (sf_buf) free(sf_buf);
    return 0;
}
