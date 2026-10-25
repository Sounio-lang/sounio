# Exact one-entry NB2 slot memoization experiment

The reusable scalar NB2 slot can retain one successful result keyed by the exact triple
(count, f64_to_bits(mean), f64_to_bits(alpha)). A hit returns the same private result.
A miss calls the original engine, whose body is byte-for-byte unchanged after its
private-function rename. Every explicit reset invalidates the cache. Failures are
never cached. Private scratch has no public mutator or getter that bypasses these
two entry points; snapshot/convenience APIs and difference-slot children preserve
their existing public behavior.

This branch is isolated from main 7ce5eddeef78c76c7e744150ed9c169c53a3d3b0.
It does not modify compiler code, fitting algorithms, compensated accumulation
order, model selection, numerical tolerances, statistical assumptions or quotas.
The cache is bounded by four private scalar fields and the already present result.
Signed zero and adjacent binary64 values have distinct keys. The original
notion of successful main status remains intact even when a separate derived
payload is unavailable.

## Native and independent qualification

| Slurm job | Frozen scope | Result |
|---|---|---|
| 17437 | before02: canonical main kernel, both compiler pins | PASS |
| 17439 | after01: same fixtures, cached kernel, both pins | PASS |
| 17442 | perf01: AB/BA/AB native pairs, both pins and both workloads | PASS |
| 17445 | reset01: strengthened same-key-after-reset fixture, both roots/pins | PASS |

Both compiler pins are immutable external dependencies, not rebuilt compilers:

- Modular c21 a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3.
- Committed c21 lean_single 8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc.

The native corpus includes all 64 scalar records with exact output bits and all
staged availability fields; changed count, adjacent mean/alpha and alpha signed
zero; invalid counts/means/alphas; actual NaN/Inf class checks; positive subnormal
alpha; unavailable derived deviance; repeated failures and explicit reset.
Each ordinary record is compared with a repeated call and a newly constructed
slot. The final reset probe deliberately calls the exact same successful key
after clearing its result.

Existing controls are retained unchanged: the 216-case scalar fixture and its
independent 400-decimal C++ reference, the 20,000-call scratch fixture, and the
13-case profile fixture with its independent C++ profile reference. All six
native programs per compiler have byte-identical stdout before and after,
including selected alpha, evaluation counts, availability and the emitted
binary64 bits of the 512-row profile means and linear predictors.

The strengthened reset case changed only the new scalar fixture; 17445 recompiled
it against the frozen old and new kernels. The kernel and measured profile ELFs
were unchanged. The profile/control fixture expectation annotations were then
corrected to their already emitted NB2_CACHE_PROFILE_PASS marker; this is an
annotation-only correction, not a changed numerical test or skipped witness.

Darwin sounio_verificado was attempted first with two candidates. Both used
nonexistent APIs and failed; the candidate was corrected using the actual public
slot functions. The first native preparation (17436/before01) then exposed an
existing lowering limitation for a direct intrinsic argument to println; typed
i64 bit locals in the new fixture resolved it. These preparation receipts remain
on the worker; the production baseline kernel was unchanged.

## Bounded performance result

The two real 512-observation ML profile fits use the same counts/histogram and
one-hot two-group design. The repeated workload groups 32 equal counts
consecutively. The control interleaves distinct adjacent counts. Both execute
the canonical dynamic_dispersion_fit, including its inner GLM and full scalar
likelihood, with the same initial values and budgets. This is not a scalar-only
loop or a cross-gene result cache.

There are three before/after pairs per compiler and workload, in actual order
AB, BA, AB. Every one of the 24 measured native stdout files is compared against
its frozen uncached output before reporting timings.

| Compiler / workload | Before CPU seconds, three runs | After CPU seconds, three runs | Median before/after CPU ratio | Peak RSS before / after, KiB |
|---|---|---|---|---|
| Modular / repeated | 2.40, 2.39, 2.40 | 1.33, 1.33, 1.33 | 1.804511 | 3608 / 3612 |
| lean_single / repeated | 4.87, 4.87, 4.87 | 2.26, 2.27, 2.26 | 2.154867 | 4108 / 4108 |
| Modular / interleaved control | 2.41, 2.42, 2.41 | 2.41, 2.42, 2.44 | 1.000000 | 3608 / 3612 |
| lean_single / interleaved control | 4.87, 4.86, 4.89 | 4.89, 4.88, 4.87 | 0.995910 | 4108 / 4108 |

C++23 parses the native resource receipts and computes these ratios. Native
timings use one Slurm CPU, 4096 MiB reservation, UID1000 and stack524288.
The sample size does not establish statistical confidence or universal speedup.
No 20,000-gene production speedup, whole-pipeline latency improvement, memory
reduction or extra inferential validity is claimed. Inputs without exact
consecutive repetitions retain the cache-key overhead; the interleaved control
records its observed cost.

Run the paired gate on an allocated Slurm CPU with explicit immutable roots:

    NB2_CACHE_BEFORE_ROOT=/scratch/.../before02 \
    NB2_CACHE_AFTER_ROOT=/scratch/.../after01 \
    NB2_CACHE_OUTPUT=/scratch/.../fresh-output \
    bash scripts/stdlib/nb2_slot_cache_gate.sh

The gate refuses existing output directories and missing successful qualification
receipts, preserves each native resource report, checks stdout identity and
requires all pairs before reporting performance.

## Custody

Raw immutable source/executables/output/job receipts remain under:

    /scratch/sounio-stdlib-inference-scale-20261024/SIO171-nb2-slot-cache/

The compact evidence01.tar.gz contains before/after source catalogs, exact kernel
copies, compiler/artifact hashes, native stdout, compile diagnostics, independent
reference results, reset receipts and all paired resource reports. It excludes
compiler and program executables, which remain on the worker. Its internal
MANIFEST.sha256 covers every included receipt. The visible external checksum
protects the archive; absolute paths inside historical artifact manifests refer
to original worker custody.

Archive bytes: 425803.
SHA256: 7188eec9dfa90117e9e2402e81f06a43d29615a30020c485e35dbbb9794289a7.

No public push, PR, main integration or merge is established by these receipts.

## Provenance-hardened replay

Slurm17448/perf02 reruns all 24 native measurements with the same numerical ELFs.
It finishes successfully only after these additional checks:

- The before kernel has the exact main7ce SHA256
  26b0d7332c7f9c25d37e4c7993ddf0e43f9491c72caadfa9ee896ef2656f7d56.
- The measured after kernel equals the canonical kernel beside the gate.
- Both frozen compiler manifests are identical, contain exactly the two
  expected a292/895 pins, and pass actual file checksum verification in each
  source root before and after measurement.
- Frozen before/after benchmark callers are identical. The canonical callers
  have identical bodies from line 4 onward; their corrected first 3 test
  annotation lines are explicitly outside this body identity check.
- SOURCE.sha256 and ELFS.sha256 both pass final checksum verification after
  every native measurement. The paired stdout identity checks still apply.

This replay reuses the already qualified numerical binaries. It does not claim
a fresh compiler or numeric executable build for annotation-only source changes.
Old packets and their receipts remain unchanged.

| Compiler / workload | Replay median CPU before/after ratio | Replay maximum RSS before / after, KiB |
|---|---|---|
| Modular / repeated | 1.796992 | 3608 / 3612 |
| lean_single / repeated | 2.154185 | 4108 / 4108 |
| Modular / interleaved control | 1.000000 | 3608 / 3612 |
| lean_single / interleaved control | 1.000000 | 4108 / 4108 |

All 48 measurements across the original and hardened paired runs remain
workload-specific observations, without a statistical-confidence claim.
The new evidence02.tar.gz contains the exact hardened gate/canonical source
inputs, native stdout/resources and initial/final checksum-verification records.
It is 113930 bytes; SHA256
3c028d459d53333d0f9b34b7f3098ff13a9255a6d9b6bb04f055d63de74758a1.
The internal manifest and the visible external checksum were verified.
