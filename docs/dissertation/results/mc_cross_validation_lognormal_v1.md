<!-- docs:meta
topic_id: repo.docs.dissertation.results.mc-cross-validation-lognormal-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.mc-cross-validation-lognormal-v1
-->

# §4.12 — Monte Carlo Cross-Validation (Lognormal Prior)

**Source**: `stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio`

> **Engine dependency (verified 2026-08-17).** `pbpk28_mc_cross_validation.sio` runs to
> completion under `SOUNIO_SOUC_ENGINE=lean_single` (`rc=0`, `PASS`). Under default Madaros
> (`bin/souc`), the same file compiles clean but **crashes at runtime with `rc=182`**
> (`madaros: handles full`) partway through the N=2000 Monte Carlo loop — a resource-ceiling
> abort, not a numerical disagreement. Every number and gate marker on this page was produced
> under lean_single; it has not been reproduced under the project's default engine.

**Parameters**: N=2000, seed=1729, rapamycin, 5 mg dose, 168 h.
**Prior**: LogNormal for all 7 parameters (σ²_log = ln(1 + CV²), μ_log = ln(mean) − σ²_log/2).

---

## Results

| Method    | u (mg·h/L) | |u − u_MC| / u_MC | Criterion | Pass? |
|-----------|-----------|-------------------|-----------|-------|
| GUM first-order | 0.317 | 0.423 | ≤ 0.05 | NO |
| Hessian 2nd-order | 0.464 | 0.155 | ≤ 0.10 | NO |
| **MC lognormal** | **0.549** | — | — | — |

**MC mean AUC**: 1.204 mg·h/L (vs GUM reference 0.517 mg·h/L — Jensen bias from convex AUC(CL)).

Gate marker: **MC_CROSS_VALIDATION_PBPK28_LOGNORMAL_OUTPUT**

---

## Interpretation

Neither convergence criterion is met, even under lognormal prior. This is expected
for rapamycin because:

1. **CL_hep CV = 58%** — far above the quasi-linear threshold (~20%). Even the
   lognormal distribution has substantial right-skew in 1/CL space.

2. **Jensen bias**: AUC ∝ 1/(CL × fu/fu_ref). Under lognormal priors for both
   CL and fu, E[AUC_MC] > AUC(μ) by the convexity of 1/CL. The MC mean
   (1.204 mg·h/L) is ~2.3× the GUM reference (0.517 mg·h/L), confirming
   that the first-order GUM systematically underestimates mean AUC at high CV.

3. **Multiplicative CL × fu interaction**: Both CL and fu have CV=25–58%.
   Their product creates larger effective variance than each parameter alone.
   The GUM's additive variance formula misses this second-order cross-term.

### Why lognormal is better than unrestricted Gaussian

Under unrestricted Gaussian, CL ~ N(12.4, 51.85) places ~4.3% probability on
CL ≤ 0. Near-zero CL draws yield AUC ≫ 10 mg·h/L, inflating u_MC by hundreds
of percent (the "880% discord" motivating this analysis). Under lognormal,
all CL samples are positive, removing the pathological tail. The residual
42% GUM/MC disagreement reflects genuine pharmacokinetic nonlinearity, not
numerical artefact.

### Motivation for §4.13 (truncated normal)

The lognormal prior is physiologically coherent (CL > 0 always) but may over-
represent large CL values relative to clinical data. A truncated normal prior
with physiological bounds (CL_lo=0.5, CL_hi=50 L/h; fu_lo=0.001, fu_hi=0.5)
would reduce both Jensen bias and the multiplicative tail, potentially bringing
GUM/MC agreement within the ≤10% Hessian criterion. This is the §4.13 hypothesis.

---

## §4.12 Wording Guide

**Safe to write**: "Under a lognormal prior for all pharmacokinetic parameters,
the first-order GUM uncertainty estimate deviates by 42% from Monte Carlo
(N=2000, seed=1729), confirming that the high coefficient of variation of CL_hepatic
(CV=58%) places the model in the moderately nonlinear regime where linearized
propagation is insufficient."

**Safe to write**: "The second-order Hessian correction reduces the GUM/MC
discrepancy from 42% (first-order) to 16% (Hessian), demonstrating that
the curvature correction accounts for approximately 60% of the linearization error."

**Do NOT write**: "GUM fails for rapamycin" — the GUM is correct within its
domain of validity (CV < ~20%). The statement should be: "At CV=58%, the
first-order GUM requires Hessian correction to achieve metrological convergence."
