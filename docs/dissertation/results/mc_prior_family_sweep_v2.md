<!-- docs:meta
topic_id: repo.docs.dissertation.results.mc-prior-family-sweep-v2
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.mc-prior-family-sweep-v2
-->

# PBPK28 Prior-Family Sweep — v2

**Date:** 2026-05-13  
**Replaces:** `mc_prior_family_sweep_v1.md`  
**Determinism verified:** Yes — `determinism_audit_v1.md`; probe gate `MC_PBPK28_DETERMINISTIC_RESULTS_PASS`.  
**Harness:** `stdlib/darwin_pbpk/validation/pbpk28_mc_prior_family_sweep.sio`  

> **Engine dependency (verified 2026-08-17).** `pbpk28_mc_prior_family_sweep.sio` runs to
> completion under `SOUNIO_SOUC_ENGINE=lean_single` (`rc=0`, `PASS`). Under default Madaros
> (`bin/souc`), the same file compiles clean but **crashes at runtime with `rc=182`**
> (`madaros: handles full`) partway through the N=2000 Monte Carlo loop — a resource-ceiling
> abort, not a numerical disagreement. Every number and gate marker on this page was produced
> under lean_single; it has not been reproduced under the project's default engine.

**Configuration:** Drug = rapamycin, N = 2000, seed = 1729, dose = 5 mg, t = 168 h

**IMPORTANT CORRECTION:** The v1 sweep reported LogNormal as "winner"
(rel_Hess = 0.026). This was entirely an artifact of a Taylor-series defect in
`ms28_exp` (`var t = rx` instead of `var t: f64 = 1.0`) that compressed lognormal
samples by ~15%, artificially reducing u_MC to 0.477 and making it closer to
u_Hessian = 0.464. The correct value is u_MC(LogNormal) = 0.549, matching E1 exactly.

---

## Corrected results

| Family | u_MC (mg·h/L) | rel_GUM | rel_Hess | Hess criterion (≤10%)? |
|---|---|---|---|---|
| Gaussian (positive) | 1.511514 | 79.0% | 69.3% | NO |
| **LogNormal** | **0.549197** | **42.3%** | **15.5%** | **NO** |
| TruncNormal | 0.540790 | 41.4% | 14.2% | NO |

**GUM:** u_GUM = 0.317093 mg·h/L (same for all families — linearisation is family-independent)  
**Hessian:** u_Hessian = 0.464032 mg·h/L (same for all families)

**Gate:** `MC_PRIOR_FAMILY_SWEEP_OUTPUT`  
No prior family meets the Hessian criterion (rel_Hess ≤ 0.10).

---

## v1 vs v2 comparison

| Family | u_MC v1 | rel_Hess v1 | u_MC v2 | rel_Hess v2 | Root cause of difference |
|---|---|---|---|---|---|
| Gaussian | 1.511514 | 0.693 | 1.511514 | 0.693 | None — Gaussian does not use exp() |
| **LogNormal** | **0.476630** | **0.026** | **0.549197** | **0.155** | **ms28_exp Taylor bug fixed** |
| TruncNormal | 0.540790 | 0.142 | 0.540790 | 0.142 | None — TruncNormal does not use exp() |

The bug affected only the LogNormal family, since only lognormal sampling uses
`ms28_exp` to compute `exp(μ_log + σ_log · z)`. Gaussian sampling uses only
`sqrt(variance)`; TruncNormal uses only normal samples with rejection.

---

## Scientific conclusion (corrected)

**No prior family achieves rel_Hess ≤ 0.10 for rapamycin PBPK28 (CL_hep CV = 58%).**

The v1 conclusion "LogNormal is the best prior" was a consequence of
numerically suppressed u_MC. With correct arithmetic:

- TruncNormal performs marginally better than LogNormal (rel_Hess 14.2% vs 15.5%),
  consistent with tighter physiological bounds slightly moderating the CL_hep tail.
  The improvement is modest (~1.3 percentage points) — not the step-change
  originally attributed.
- The §4.13 hypothesis (TruncNormal ≤ 10%) remains **unconfirmed**.
- The dominant driver of GUM/MC discord is CL_hep CV = 58% (strongly nonlinear
  regime), not the distributional family. Reducing CV below ~20% via Bayesian
  posterior updating would be required to achieve convergence.

---

## Dissertation wording (§4.13)

**Safe to cite:**
- No prior family passes the Hessian criterion under the rapamycin uncertainty budget
- TruncNormal marginally reduces rel_Hess relative to LogNormal (14.2% vs 15.5%), but
  neither meets the ≤ 10% threshold
- The model is in a strongly nonlinear regime driven by CL_hep CV = 58%
- The v1 LogNormal "win" (rel_Hess = 0.026) was an artifact of a numerical defect,
  now corrected and documented in `determinism_audit_v1.md`

**Forbidden until further analysis:**
- Claiming LogNormal is the optimal prior on the basis of v1 numbers
- Citing rel_Hess(LogNormal) = 0.026
