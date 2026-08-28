// kaxi_pbpk_8comp_assoc_sampler — Phase Z input generator + CPU oracle.
//
// L1 SCAFFOLD for the AssociatorField butterfly thread. This is the 8-compartment
// octonion-indexed extension of the Phase Y 2-comp GUM sampler
// (scripts/gpu/kaxi_pbpk_2comp_gum_sampler.c). It is fully runnable on CPU today;
// the matching GPU PTX kernel is the deferred multi-session piece documented by
// scripts/ci/kretikos_kaxi_phase_z_assoc_gate.sh.
//
// Per patient:
//   - 8 compartment concentrations C = (c0..c7) treated as one octonion.
//   - Diagonal GUM variance V = (v0..v7), one per compartment (log-normal init).
//   - ONE associator-augmentation scalar: the structural (order-indeterminacy)
//     uncertainty contribution from non-associative drug-interaction ordering.
//       A = C,  B = shift1(C),  Cc = shift2(C)   (cyclic component shifts)
//       aug = KAPPA * ||[A,B,Cc]||²            where [·,·,·] is the octonion associator
//     This mirrors stdlib/algebra/associator_field.sio::af_augmented_variance.
//     For associative inputs (e.g. all mass in the real slot c0) aug == 0 exactly,
//     which the gate checks as an invariant.
//
// Outputs (f32, little-endian), n = shard patient count:
//   init_c0.bin .. init_c7.bin     8 files, concentrations
//   init_v0.bin .. init_v7.bin     8 files, GUM diagonal variances
//   init_aug.bin                   associator-augmentation per patient
//   expected.summary               FNV digests + counts for truth-claim checks
//
// Build:
//   cc -O2 scripts/gpu/kaxi_pbpk_8comp_assoc_sampler.c -lm -o /tmp/pz_sampler

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define NCOMP 8
#define KAPPA 1.0f

// --- PRNG (matches the Phase Y sampler) ---
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

// --- Octonion multiplication (Cayley-Dickson sign table; matches
//     stdlib/algebra/octonion.sio::oct_mul exactly) ---
static void oct_mul(const float a[8], const float b[8], float out[8]) {
    out[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3] - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7];
    out[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2] + a[4]*b[5] - a[5]*b[4] - a[6]*b[7] + a[7]*b[6];
    out[2] = a[0]*b[2] + a[2]*b[0] - a[1]*b[3] + a[3]*b[1] + a[4]*b[6] - a[6]*b[4] + a[5]*b[7] - a[7]*b[5];
    out[3] = a[0]*b[3] + a[3]*b[0] + a[1]*b[2] - a[2]*b[1] + a[4]*b[7] - a[7]*b[4] - a[5]*b[6] + a[6]*b[5];
    out[4] = a[0]*b[4] + a[4]*b[0] - a[1]*b[5] + a[5]*b[1] - a[2]*b[6] + a[6]*b[2] - a[3]*b[7] + a[7]*b[3];
    out[5] = a[0]*b[5] + a[5]*b[0] + a[1]*b[4] - a[4]*b[1] + a[3]*b[6] - a[6]*b[3] - a[2]*b[7] + a[7]*b[2];
    out[6] = a[0]*b[6] + a[6]*b[0] + a[2]*b[4] - a[4]*b[2] - a[3]*b[5] + a[5]*b[3] + a[1]*b[7] - a[7]*b[1];
    out[7] = a[0]*b[7] + a[7]*b[0] - a[1]*b[6] + a[6]*b[1] + a[2]*b[5] - a[5]*b[2] + a[3]*b[4] - a[4]*b[3];
}

// associator-augmentation scalar: KAPPA * ||[A,B,C]||²
static float assoc_augmentation(const float c[8]) {
    float A[8], B[8], Cc[8];
    for (int i = 0; i < 8; i++) {
        A[i]  = c[i];
        B[i]  = c[(i + 1) % 8];   // shift1
        Cc[i] = c[(i + 2) % 8];   // shift2
    }
    float ab[8], ab_c[8], bc[8], a_bc[8];
    oct_mul(A, B, ab);   oct_mul(ab, Cc, ab_c);   // (A*B)*C
    oct_mul(B, Cc, bc);  oct_mul(A, bc, a_bc);    // A*(B*C)
    float n2 = 0.0f;
    for (int i = 0; i < 8; i++) { float d = ab_c[i] - a_bc[i]; n2 += d * d; }
    return KAPPA * n2;
}

// FNV-1a over raw f32 bytes — deterministic digest for truth claims.
static uint64_t fnv1a(const float *buf, long n) {
    uint64_t h = 1469598103934665603ULL;
    const uint8_t *p = (const uint8_t *)buf;
    for (long i = 0; i < n * (long)sizeof(float); i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
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
        "  --dose-sigma F       log-normal scale sigma (default 0.50)\n"
        "  --associative        force all mass into c0 (real slot) -> aug==0 invariant\n"
        "  --counts-only        skip writing binary files\n",
        prog);
}

int main(int argc, char **argv) {
    const char *out_dir = NULL;
    long cohort = 0;
    uint64_t seed = 42;
    double dose_sigma = 0.50;
    int associative = 0, counts_only = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else if (!strcmp(argv[i], "--out-dir")    && i+1<argc) { out_dir    = argv[++i]; }
        else if (!strcmp(argv[i], "--cohort")     && i+1<argc) { cohort     = strtol(argv[++i],NULL,10); }
        else if (!strcmp(argv[i], "--seed")       && i+1<argc) { seed       = strtoull(argv[++i],NULL,10); }
        else if (!strcmp(argv[i], "--dose-sigma") && i+1<argc) { dose_sigma = strtod(argv[++i],NULL); }
        else if (!strcmp(argv[i], "--associative"))             { associative = 1; }
        else if (!strcmp(argv[i], "--counts-only"))             { counts_only = 1; }
        else { fprintf(stderr, "error: unknown arg: %s\n", argv[i]); usage(argv[0]); return 2; }
    }
    if (!out_dir || cohort <= 0) { usage(argv[0]); return 2; }

    long n = cohort;
    size_t bytes = (size_t)n * sizeof(float);
    float *c[NCOMP], *v[NCOMP];
    float *aug = counts_only ? NULL : (float *)malloc(bytes);
    for (int k = 0; k < NCOMP; k++) {
        c[k] = counts_only ? NULL : (float *)malloc(bytes);
        v[k] = counts_only ? NULL : (float *)malloc(bytes);
    }

    double var_mult = exp(dose_sigma * dose_sigma) - 1.0;  // GUM log-normal multiplier
    uint64_t st = seed;
    double aug_sum = 0.0;
    long aug_zero = 0;

    for (long p = 0; p < n; p++) {
        float cc[NCOMP];
        for (int k = 0; k < NCOMP; k++) {
            double base = exp(next_normal(&st) * dose_sigma);   // log-normal dose
            if (associative && k > 0) base = 0.0;               // all mass in real slot
            cc[k] = (float)base;
            float var = (float)(base * base * var_mult);
            if (!counts_only) { c[k][p] = cc[k]; v[k][p] = var; }
        }
        float a = assoc_augmentation(cc);
        if (!counts_only) aug[p] = a;
        aug_sum += a;
        if (a == 0.0f) aug_zero++;
    }

    // Write buffers.
    if (!counts_only) {
        char path[512];
        for (int k = 0; k < NCOMP; k++) {
            snprintf(path, sizeof(path), "%s/init_c%d.bin", out_dir, k);
            if (write_all(path, c[k], bytes)) { fprintf(stderr, "write fail %s\n", path); return 1; }
            snprintf(path, sizeof(path), "%s/init_v%d.bin", out_dir, k);
            if (write_all(path, v[k], bytes)) { fprintf(stderr, "write fail %s\n", path); return 1; }
        }
        snprintf(path, sizeof(path), "%s/init_aug.bin", out_dir);
        if (write_all(path, aug, bytes)) { fprintf(stderr, "write fail %s\n", path); return 1; }

        snprintf(path, sizeof(path), "%s/expected.summary", out_dir);
        FILE *s = fopen(path, "w");
        if (!s) { fprintf(stderr, "write fail %s\n", path); return 1; }
        fprintf(s, "ncomp %d\n", NCOMP);
        fprintf(s, "patients %ld\n", n);
        fprintf(s, "kappa %.7g\n", (double)KAPPA);
        for (int k = 0; k < NCOMP; k++)
            fprintf(s, "fnv_c%d %016llx\n", k, (unsigned long long)fnv1a(c[k], n));
        fprintf(s, "fnv_aug %016llx\n", (unsigned long long)fnv1a(aug, n));
        fprintf(s, "aug_sum %.9g\n", aug_sum);
        fprintf(s, "aug_zero_count %ld\n", aug_zero);
        fclose(s);
    }

    // Machine-grep summary line for the gate.
    printf("phase_z_sampler patients=%ld ncomp=%d aug_sum=%.9g aug_zero=%ld associative=%d\n",
           n, NCOMP, aug_sum, aug_zero, associative);

    if (!counts_only) {
        for (int k = 0; k < NCOMP; k++) { free(c[k]); free(v[k]); }
        free(aug);
    }
    return 0;
}
