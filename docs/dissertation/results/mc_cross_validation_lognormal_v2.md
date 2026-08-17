<!-- docs:meta
topic_id: repo.docs.dissertation.results.mc-cross-validation-lognormal-v2
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.mc-cross-validation-lognormal-v2
-->

# PBPK28 MC Cross-Validation — LogNormal Prior — v2

**Date:** 2026-05-13  
**Replaces:** `mc_cross_validation_lognormal_v1.md`  
**Determinism verified:** Yes — see `determinism_audit_v1.md` and
`docs/compiler/numerical_determinism.md`.  
**Harness:** `stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio`  

> **Engine dependency (verified 2026-08-17).** `pbpk28_mc_cross_validation.sio` runs to
> completion under `SOUNIO_SOUC_ENGINE=lean_single` (`rc=0`, `PASS`). Under default Madaros
> (`bin/souc`), the same file compiles clean but **crashes at runtime with `rc=182`**
> (`madaros: handles full`) partway through the N=2000 Monte Carlo loop — a resource-ceiling
> abort, not a numerical disagreement. Every number and gate marker on this page was produced
> under lean_single; it has not been reproduced under the project's default engine.

**Configuration:** Drug = rapamycin, N = 2000, seed = 1729, prior = LogNormal all 7 parameters  
**Computation:** Welford online accumulator (v2); exp correct (v2)

---

## Results

| Method | u (mg·h/L) | rel. to u_MC | Criterion | Verdict |
|---|---|---|---|---|
| GUM first-order | 0.317093 | 42.3% | ≤ 5% | NOT MET |
| Hessian second-order | 0.464032 | 15.5% | ≤ 10% | NOT MET |
| **MC (reference)** | **0.549197** | — | — | — |

**MC mean AUC:** 1.204004 mg·h/L (Jensen upward bias from convexity of 1/CL)  
**n_valid:** 2000/2000

**Gate:** `MC_CROSS_VALIDATION_PBPK28_LOGNORMAL_OUTPUT`  
(Neither GUM nor Hessian criterion met — motivates §4.13 truncated-prior analysis.)

---

## Determinism note

v1 produced identical numbers (u_MC = 0.549197, rel_Hess = 0.155073). E1 was
unaffected by the ms28_exp Taylor bug because it uses `mc28_exp` (correct
implementation). The v2 re-run with the Welford accumulator produces the same
result: the biased two-pass estimator and Welford agree to all printed digits for
N = 2000.

---

## Change from v1

| Quantity | v1 | v2 | Change |
|---|---|---|---|
| u_MC | 0.549197 | 0.549197 | None — E1 was already correct |
| rel_Hess | 0.155073 | 0.155073 | None |
| Variance estimator | two-pass (Σy²/n − ȳ²) | Welford online | Numerically equivalent at N=2000 |

E1 was always correct. The v1 interval [0.03, 0.16] for rel_Hess across the two
sprints arose entirely from the E4 ms28_exp bug, not from E1.

---

## Dissertation wording (§4.12)

**Safe to cite:**
- u_MC = 0.549 mg·h/L (±0.5% MC sampling error for N=2000)
- u_Hessian = 0.464 mg·h/L
- rel_Hess = 15.5% — exceeds the 10% Hessian criterion
- CL_hep CV = 58% places the model in the moderately-to-strongly nonlinear regime
- Second-order Hessian correction reduces the GUM/MC gap from 42% to 16% but does
  not fully resolve it

**Forbidden until further analysis:**
- Claiming GUM adequacy
- Citing rel_Hess = 0.026 (that was from the buggy E4, not E1)
