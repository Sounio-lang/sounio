# ML profile-refit count-model bootstrap

`dynamic_count_profile_lrt_bootstrap` is a separate method from the accepted
fixed-supplied-alpha bootstrap. It fits both beta and NB2 dispersion under the
null and alternative for the observed counts and for every generated replicate.
The generator uses the observed null's fitted means and dispersion. Exact
Poisson (alpha zero) is an explicit candidate, not a floor on positive alpha.

This is a model-conditional, plug-in parametric bootstrap, not an exact test or
an assessed model. Nesting, the NB2/Poisson model and observation independence
must be explicitly declared. `assumptions_assessed`, `global_optimality_assessed`
and `finite_sample_validity_assessed` remain false. Reestimating alpha does not
by itself establish marginal p-value validity or license FDR claims.

## Fit policy

Every fit calls the existing `dynamic_dispersion_fit` with method0 (ML), supplied
log-alpha bracket and controls, no prior and no curvature computation. Cox–Reid
and log-coordinate MAP criteria are deliberately absent from this LR API.
A resolved positive-alpha interior fit is accepted. A selected Poisson candidate
is also accepted despite its profile's left-censor diagnostic. Any selected
positive-alpha range edge stops the entire aggregate with status12. The search
is not a certificate of global optimality or a proof that an unexplored positive
alpha outside the bracket could not win.

Each model has its own initial coefficients, design, offsets and caller-owned
workspace. Initial coefficients are reused unchanged for every fit; there are
no data-dependent warm starts or hidden retries. Every failed or unresolved fit
stops the aggregate. No replicate is discarded. A fit's output owners and all
candidate/best/factor owners are scratch, not transactional public estimates.
Scientific aggregate fields may only be consumed when `available` is true.

The statistic is twice n times the difference in full mean log PMFs evaluated
at the two separately profiled fits, including alpha-dependent normalization.
Negative LR is retained diagnostically and makes the aggregate unavailable;
it is never clipped to zero. Overflow/unrepresentability is unavailable.
Subtracting nearly equal rounded mean log likelihoods is cancellation-sensitive;
zero is not an exact equality certificate. Identical-model fixtures deliberately
exercise identical computations.

## Interface and work

Use `dynamic_count_profile_bootstrap_workspace(n,p0,p1)`, with p0<=p1 and n>p1.
The returned sizes are low-level caller scratch requirements. Create distinct
owners for both models. Per model, provide its regular coefficient/means/eta
owners, count-GLM work, candidate coefficient/means/eta and best copies. ML
factor-R, factor-scale and permutation owners have length0. No numeric or
integer owner is allocated inside the driver. One reusable count cursor and
one draw slot serve all rows, and one likelihood slot serves full-PMF sums.
There is no B-by-n sample history; the last generated sample remains in scratch.
Payload byte counts exclude allocator/scalar runtime overhead and are not RSS.

`max_profile_attempts` budgets calls to the dispersion profiler (observed and
replicate null/alternative fits), not individual beta fits. The result's inherited
`fit_attempts` field counts these profile attempts. Each has its own
`max_outer_evaluations`; their cumulative reported count is `profile_evaluations`.
The product is overflow-checked. The driver does not count hidden work as free:
inner iteration bounds and outer-evaluation limits apply to every profile call.
The nested profiler's diagnostic count may be partial on a failed evaluation;
it is not a complete count of IRLS operations.

A pair is not begun unless two profile calls remain. Sampling uniforms/proposals
have per-draw and total budgets. The actual advanced RNG and counters are
returned on failure and success. Precondition failure consumes no RNG. A sampler
precision warning (`resolution_limited` or `rate_underflow`) makes calibration
unavailable even if the draw status itself is0.

## Status and aggregate

Status0 means all requested replicates resolved. Status1 is shape/owner failure;
2 invalid controls/RNG/counter capacity;3 nested profile failure;4 arithmetic;
5 profile-attempt/evaluation/sampler budget;6 negative LR;7 sampler failure;
8 sampler precision;9 full likelihood;11 missing declarations;12 positive-alpha
profile edge. Last nested fit/profile status and failure stage/replica/row locate
failure. Main status5 can wrap a profiler's status5. No p/MC output is available
on any failure, even when observed fit diagnostics or partial hits exist.

On success, hits counts replicate LR >= observed LR, including native ties,
and p=(hits+1)/(B+1). Raw hits/B and its Bernoulli standard-error estimate are
separate MC diagnostics; endpoint zero SE does not imply certainty. B<=2^53-1
keeps exact integer-to-binary64 rank representation. The selected B2 fixture is
for deterministic concordance, not adequate general Monte Carlo precision.

## Independent validation target

`deep_stats_count_profile_bootstrap_reference.cpp` uses C++23 binary128 analytic
group means, direct lgamma log PMFs and finite harmonic dispersion-score sums.
It locates positive-to-negative score crossings independently, then bisects;
it does not reuse Sounio IRLS, QR, likelihood, grid or golden search. The only
shared reference dependency is the independent C++ unsigned128 PCG/sampler.
Selected native observed alpha/LR, plus-one tail, hits, RNG and last sample must
match, including both last-refit dispersions, and a full fixture marker is mandatory.
`--mutant-fixed-alpha actual-native.stdout` evaluates the old fixed-alpha
replicate algorithm as an explicit negative control and must fail concordance.
This is an independent algorithm-control comparison, not a production mutation. This finite selected grid does
not certify the profile over all designs or parameter ranges.

Compile centrally under Slurm:

```
g++ -std=c++23 -O2 tests/reference/deep_stats_count_profile_bootstrap_reference.cpp -lquadmath -o profile_bootstrap_reference
profile_bootstrap_reference actual-native.stdout
```

Fixture `stdlib_deep_stats_count_profile_bootstrap.sio` covers grouped and
identical models, positive-edge rejection, declarations, attempt/sampler budgets,
invalid count conversion, and explicit observed Poisson selection. Source
preparation alone is not native acceptance, numeric concordance, calibrated
coverage, a full corpus pass or a performance result.

## Prepared bounded qualification protocol

`tests/reference/count_profile_calibration.cpp` emits independent C++-sampled
observations and a generated Sounio driver that invokes this real kernel; it
never replaces production fitting with C++ fits. Its driver owner/call shape is
extracted from the frozen native fixture, which is included in campaign hashes.
The observed null is a balanced two-group equal-mean model (mean8), for n16/64/256
and alpha0/.25/1. Generation and bootstrap seed schedules are explicit per dataset.

`scripts/stdlib/count_profile_calibration.sh` is a Slurm-only orchestrator,
requires an explicit passing functional-gate receipt and attested compiler,
and never submits itself. Qualification uses B199, batches16, at most128 datasets
per cell and at most24h cumulative wall budget. A separate pilot namespace uses
B31, n16/alpha.25 and16 datasets; it is not the B199 qualification result. Input
source/compiler/gate hashes must remain identical on resume. A hard kill reserves
the remaining wall budget conservatively, requiring review before further work.
Incomplete batches remain intact and block resume rather than disappearing.

Available p values, unavailable statuses and failing-replica metadata are kept.
Summaries report rejection among available datasets, availability, a conditional
Monte Carlo SE and approximate Wilson95 binomial interval; they also report the
extreme overall rejection range if all unavailable datasets were nonrejections
or rejections. Conditioning on solver/sampler availability can bias rejection
fractions. Pseudorandom draws and these finite simulated cells do not certify
universal validity, independent observations or scientific model adequacy.
Preparation of this protocol is not evidence it was executed.

The qualification gate receipt is exactly seven nonempty `key=value` lines,
with unique keys and no executable shell content:

```
format=stdlib-advanced-functional-v1
status=0
compiler_sha256=<SHA256 of the selected current-source ELF>
canonical_inputs_sha256=<SHA256 of the canonical input manifest>
canonical_inputs_path=<readable canonical manifest path>
results_sha256=<SHA256 of the functional RESULTS.tsv>
results_path=<readable functional RESULTS.tsv path>
```

The canonical manifest is produced from the frozen source root by
`find -L bin scripts self-hosted stdlib tests examples -type f -print0 | sort -z | xargs -0 sha256sum`.
Following bin symlinks binds their actual payloads. No `.git` directory is needed.
The orchestrator requires byte identity with its recomputed manifest and binds
both the selected compiler and RESULTS hashes. It checks the result schema,
all recorded exit statuses, required native/marker and independent-concordance
rows, and the fixed-alpha mutant rejection. A readable arbitrary file is not
sufficient. This is a consistency gate under a coordinator-controlled receipt,
not a cryptographic signature or independent attestation of execution.

Qualification visits one batch in each of the nine cells before starting the
next round, so an exhausted time budget does not intentionally fill only the
smallest n. `COUNT_PROFILE_WALL_BUDGET_SECONDS` accepts1..86400. The coordinating
Slurm allocation caps the combined pilot and qualification at24h and supplies
the remaining budget; the script alone is not a global scheduler certificate.

### Process isolation within qualification batches

The calibration generator still emits one ELF per batch with the same4K count
array and unchanged dataset/seed/B schedules. At execution it selects exactly
one local case through `COUNT_PROFILE_CALIBRATION_CASE=0..15`. A Linux-only
bounded `/proc/self/environ` scan accepts a unique canonical decimal value and
rejects missing, duplicate, malformed or out-of-range selections. The driver
never logs other environment values. It emits one `CASE COMPLETE` marker.

The orchestrator runs that ELF in sixteen separate processes and retains each
case's stdout, stderr, exit status, GNUtime resource receipt and shared-ELF hash
verification. It concatenates outputs in case order and emits exactly one final
`BATCH COMPLETE` only after every process and required marker succeeds. A
numerically unavailable bootstrap is retained as its status/p placeholder, not
discarded; a crashed process or incomplete record leaves the batch partial and
blocks aggregate publication. No partial batch is silently normalized into a
smaller sample of successful datasets.

The summary interface is now `--summarize B batch_size first_expected stdout...`.
It verifies exact expected indices for each input batch, bounded B/hits/status
and failure metadata, individual case completion markers and a single final
batch marker. Missing, reordered, duplicated or incomplete datasets are errors.
This process boundary resets process-lifetime managed temporaries between
observations. It does not establish bounded RSS within one B-replicate bootstrap
and does not replace the separate kernel hot-allocation fixes or measurements.

The bounded cell-to-base-seed map is evaluated by the C++ `--seeds cell_index`
mode; Bash retains only operational batch/index and wall-budget bookkeeping.
