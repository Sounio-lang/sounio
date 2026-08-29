<!-- docs:meta
topic_id: repo.docs.dissertation.results.determinism-audit-summary-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.determinism-audit-summary-v1
-->

# PBPK28 MC Numerical-Determinism Audit — Summary

**Date:** 2026-05-13  
**Branch:** `codex/pbpk28-numerical-determinism-audit`

---

## 1. Root cause of the E1/E4 discrepancy

The v1 prior-evolution sprint reported:
- E1 rel_Hess(LogNormal) = 0.155
- E4 rel_Hess(LogNormal) = 0.026

The spread was attributed to "JIT compilation context effect." **That attribution
was wrong.** The actual root cause is a Taylor-series coding defect on line 81 of
`pbpk28_mc_prior_family_sweep.sio`:

```sounio
// BUGGY (before D3 fix):
var t = rx          // Taylor series starts at rx^1, missing constant term 1

// CORRECT (mc28_exp in E1, and ms28_exp after D3 fix):
var t: f64 = 1.0    // Taylor series starts at 1 (correct)
```

This causes `ms28_exp` to compute:

    1 + rx² + rx³/2! + rx⁴/3! + ...

instead of the correct:

    1 + rx + rx²/2! + rx³/3! + ...

For `x = 1.0` (representative of ln(mean_CL_hep)): the buggy implementation
returns 2.221 (−18.3% error). For typical lognormal arguments the error is 17–25%.

The compressed exp() suppresses lognormal parameter samples, reducing u_MC from
the correct 0.549 to 0.477 — making it spuriously close to u_Hessian = 0.464,
producing the artificially low rel_Hess = 0.026.

**The non-determinism was not non-determinism at all.** Both harnesses were fully
deterministic; they were just computing different (wrong vs correct) values.

---

## 2. Corrective actions

| Deliverable | Action | Outcome |
|---|---|---|
| D1 | Probe binary + shell script documenting the bug | Root cause confirmed quantitatively |
| D2 | RNG isolation audit | Pre-existing design already correct; no change needed |
| D3 | Fix `var t = rx` → `var t: f64 = 1.0` | u_MC(E4/LogNormal) = u_MC(E1) = 0.549197 |
| D3 | Add Welford accumulator to both harnesses | Numerical stability for lognormal tails |
| D4 | Compiler audit document | Confirmed: no compiler-level issue; IEEE 754 compliant |
| D5 | Re-execute both harnesses | Canonical numbers committed to dissertation results |

The fix was a single character change. The Welford accumulator is an independent
numerical-stability improvement that produces the same result at N = 2000.

---

## 3. D2 — RNG isolation

Each call to `ms28_run_family` initialises `var rng = ms28_rng_new(seed)`
independently with `seed = 1729`. There is no cross-family RNG state sharing.
Intra-process and inter-process determinism are both intact for both harnesses.

**Gate:** `MC_PBPK28_DETERMINISM_RNG_ISOLATED_PASS`

---

## 4. D4 — Compiler determinism guarantees

Sounio on x86-64 Linux:
- Emits `addsd / subsd / mulsd / divsd` — IEEE 754-2008 round-to-nearest-even
- No fast-math, no implicit FMA contraction for scalar code
- No optimization-flag divergence between CI and local development
- The E1/E4 discrepancy was entirely in user-space math helpers, not compiler output

The self-implemented `exp` function (post-fix) has < 10⁻¹² relative error for
`|x| ≤ 40`. The `ln` function has < 2 × 10⁻⁸ relative error.

**Gate:** `MC_PBPK28_COMPILER_DETERMINISM_PASS` (via `mc_determinism_probe.sh --post-fix`)

> **Engine dependency (verified 2026-08-17).** `scripts/audit/mc_determinism_probe.sh`
> hardcodes `SOUC="./bin/souc"` (line 26) with no engine override, and under `set -euo
> pipefail` it compiles *and executes* `pbpk28_mc_cross_validation.sio` and
> `pbpk28_mc_prior_family_sweep.sio`. Both of those harnesses **crash at runtime with `rc=182`**
> (`madaros: handles full`, a resource-ceiling abort) under default Madaros — so this script
> currently aborts before it can emit `MC_PBPK28_COMPILER_DETERMINISM_PASS` at all when run
> under the project's default engine. The "Sounio on x86-64 Linux" codegen claim above is true
> of both engines in principle (same `addsd`/`subsd`/`mulsd`/`divsd` scalar codegen), but this
> specific gate marker has only ever been produced under `SOUNIO_SOUC_ENGINE=lean_single`.

---

## 5. Canonical final numbers

**rel_Hess(LogNormal) = 0.155073**

Verified reproducible across:

| Configuration | u_MC (mg·h/L) | Source |
|---|---|---|
| E1 standalone, pre-fix, 3× intra-process | 0.549197 | v1 binary |
| E1 standalone, post-fix (Welford), 3× inter-process | 0.549197 | v2 binary |
| E4 sweep LogNormal, post-fix, 3× inter-process | 0.549197 | v2 binary |
| Probe binary, correct exp, 3× intra-process | 0.549197 (CL_hep SD; AUC distribution is same) | probe binary |

**Statement of confidence:**

> rel_Hess(LogNormal) = 0.155, u_MC = 0.549 mg·h/L, verified reproducible across
> E1 standalone, E4 sweep, intra-process (3×), inter-process (3×), and
> Welford/two-pass variance equivalence. The single-digit precision of these numbers
> is limited by Monte Carlo sampling variance at N = 2000 (SE ≈ 0.5%), not by
> numerical algorithm.

---

## 6. Scientific correction to dissertation narrative

The previously cited "LogNormal wins at rel_Hess = 0.026" must not be used.

The corrected conclusion for §4.12–4.13:

- No prior family achieves rel_Hess ≤ 0.10
- TruncNormal is marginally better than LogNormal (14.2% vs 15.5%) but neither converges
- The dominant factor is CL_hep CV = 58% (strongly nonlinear regime)
- Prior distributional choice is secondary; Bayesian posterior narrowing (CV ≤ ~20%)
  would be required for Hessian convergence

This is a **stronger** scientific conclusion than v1 — it shows the nonlinearity
problem is prior-family independent, not merely a Gaussian-vs-LogNormal distinction.

---

## 7. Gate markers — complete list

```
PBPK28_MC_DETERMINISM_PROBE_PASS
MC_PBPK28_DETERMINISM_RNG_ISOLATED_PASS
MC_PBPK28_VARIANCE_NUMERICAL_STABILITY_PASS
MC_PBPK28_COMPILER_DETERMINISM_PASS
MC_PBPK28_DETERMINISTIC_RESULTS_PASS
MC_CROSS_VALIDATION_PBPK28_LOGNORMAL_OUTPUT  (unchanged from v1 — E1 was always correct)
MC_PRIOR_FAMILY_SWEEP_OUTPUT                 (CORRECTED from v1 MC_PRIOR_FAMILY_SWEEP_PASS)
```

Pre-existing PBPK28 parity gates (all 7 green after D3 changes):
```
PBPK28_PARITY_PASS
PBPK28_MASS_CONSERVATION_PASS
PBPK28_TMDD_PARITY_PASS
PBPK28_PD_PARITY_PASS
PBPK28_SEMAGLUTIDE_PARITY_PASS
PBPK28_SEMA_TMDD_PARITY_PASS
PBPK28_SEMA_PD_PARITY_PASS
```
