<!-- docs:meta
topic_id: repo.docs.audit.r2-4-pcg64-algorithmic-bug.synthesis-c
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-4-pcg64-algorithmic-bug.synthesis-c
-->

# Phase C Synthesis — Statistical sanity (2026-05-17)

## Status

**PASS (6/6).** Mean and variance of the three core samplers
(`uniform_sample`, `normal_sample`, `exponential_sample`) backed by the
new canonical PCG64 fall inside CLT-derived tolerance bands at N=20000.

## Probe

`reference/phase_c_stat_sanity.sio` — seed=31415, N=20000 per distribution,
shared rng stream across the three blocks (sequential draws, not three
independent seeds). Theoretical (μ, σ²) compared against running
sum / sum-of-squares variance.

## Results

| dist        | N    | empirical mean | theoretical μ | empirical var | theoretical σ² | tol_mean | tol_var | mean | var |
|---          |---   |---             |---            |---            |---             |---       |---      |---   |---  |
| Uniform(0,1)| 20000| 0.499561       | 0.500000      | 0.084340      | 0.083333       | 0.01     | 0.01    | ✓    | ✓   |
| Normal(0,1) | 20000| 0.001510       | 0.000000      | 1.000464      | 1.000000       | 0.03     | 0.05    | ✓    | ✓   |
| Exp(rate=1) | 20000| 1.002394       | 1.000000      | 1.010144      | 1.000000       | 0.03     | 0.06    | ✓    | ✓   |

Total: **6/6 PASS** (3 means × 3 variances).

## Tolerance derivation

For each distribution, CLT gives `σ_mean = sqrt(σ²/N)`. Bands set at
≈4σ_mean for the mean (≈99.99% CLT band, conservative against rare seeds)
and at a hand-set loose σ² band big enough to absorb the χ² tail at
N=20000 without flake.

- Uniform: σ_M ≈ √(0.0833/20000) ≈ 0.00204 → 4σ_M ≈ 0.0082 → tol 0.01.
- Normal:  σ_M ≈ 1/√20000 ≈ 0.00707 → 4σ_M ≈ 0.028 → tol 0.03.
- Exp:     σ_M ≈ 1/√20000 ≈ 0.00707 → 4σ_M ≈ 0.028 → tol 0.03; variance band
  bumped to 0.06 since Exp σ² is noisier than its mean.

## Degenerate-stream guard

The pre-fix dst_pcg64 collapsed to a single stuck value after the first
step (Cause B SRET aliasing + Cause A dead-state). Either failure would
have driven empirical variance toward 0 — outside every band above by
orders of magnitude. The fact that 6/6 PASS confirms not only that the
mean is centred but that the *stream remains live* across 20000 steps.

## Out of scope

- Gamma/Beta/Poisson — same RNG backend, same per-call algorithmic
  structure as the three above; not gated here to keep the probe small.
- KS test / spectral test — not part of the dissertation's RNG-quality
  budget; pcg-cpp bit-exactness from Phase A is the canonical guarantee.

## Phase C complete — Phase D authorized

The PCG64 backend is now algorithmically correct (Phase A), wired into
stdlib (Phase B), and statistically sane (Phase C). Phase D removes the
"BROKEN PCG64" guidance from `stdlib/random/lib.sio` and updates any
README that still steers users toward Park-Miller as a workaround for
this specific bug. Park-Miller remains the recommended simple default
on quality-vs-complexity grounds, not as a defect workaround.
