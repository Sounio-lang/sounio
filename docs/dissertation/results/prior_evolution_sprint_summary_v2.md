<!-- docs:meta
topic_id: repo.docs.dissertation.results.prior-evolution-sprint-summary-v2
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.prior-evolution-sprint-summary-v2
-->

# PBPK28 Prior Evolution Sprint — Summary v2

**Date:** 2026-05-13  
**Replaces:** `prior_evolution_sprint_summary_v1.md`  
**Branch:** `codex/pbpk28-numerical-determinism-audit`

This document supersedes v1. The v1 summary reported a rel_Hess(LogNormal) range
of [0.026, 0.155] across two harnesses and attributed the spread to "JIT
compilation context." That attribution was incorrect. The spread was caused by a
Taylor-series coding defect in `ms28_exp` that has now been identified, corrected,
and verified to be eliminated.

> **Engine dependency (verified 2026-08-17).** Both harnesses behind this document's numbers
> (`pbpk28_mc_cross_validation.sio` and `pbpk28_mc_prior_family_sweep.sio`) compile clean but
> **crash at runtime with `rc=182`** (`madaros: handles full`, a resource ceiling) partway
> through the N=2000 loop under default Madaros (`bin/souc`). Both run to completion (`rc=0`,
> `PASS`) under `SOUNIO_SOUC_ENGINE=lean_single`. The "Verification command" below
> (`mc_determinism_probe.sh`) hardcodes `SOUC="./bin/souc"` with no engine override and compiles
> *and executes* both harnesses under `set -euo pipefail` — under default Madaros it aborts at
> that crash before reaching the gate markers this document cites. Every canonical number here
> was produced under lean_single.

---

## Canonical numbers — single verified value per metric

All values from `mc_cross_validation_lognormal_v2.md` and
`mc_prior_family_sweep_v2.md`. Verified reproducible across:
- E1 standalone and E4 sweep (same u_MC)
- Three independent process runs (same u_MC)
- With/without Welford accumulator (same u_MC at N = 2000)

### E1: MC cross-validation (LogNormal, N=2000, seed=1729)

| Quantity | Value | Units |
|---|---|---|
| u_GUM (first-order) | 0.317093 | mg·h/L |
| u_Hessian (second-order) | 0.464032 | mg·h/L |
| **u_MC (canonical)** | **0.549197** | **mg·h/L** |
| rel_GUM | 0.422624 | — |
| **rel_Hess** | **0.155073** | **—** |
| MC mean AUC | 1.204004 | mg·h/L |

### E4: Prior-family sweep (N=2000, seed=1729)

| Family | u_MC (mg·h/L) | rel_Hess | Hess criterion? |
|---|---|---|---|
| Gaussian (positive) | 1.511514 | 0.693002 | NO |
| **LogNormal** | **0.549197** | **0.155072** | **NO** |
| TruncNormal | 0.540790 | 0.141938 | NO |

---

## Gate markers (corrected)

```
MC_CROSS_VALIDATION_PBPK28_LOGNORMAL_OUTPUT    (neither criterion met — unchanged from v1)
MC_PRIOR_FAMILY_SWEEP_OUTPUT                   (CORRECTED: was MC_PRIOR_FAMILY_SWEEP_PASS)
PBPK28_MC_DETERMINISM_PROBE_PASS
MC_PBPK28_DETERMINISM_RNG_ISOLATED_PASS
MC_PBPK28_VARIANCE_NUMERICAL_STABILITY_PASS
MC_PBPK28_COMPILER_DETERMINISM_PASS
MC_PBPK28_DETERMINISTIC_RESULTS_PASS
```

---

## Correction narrative

### What v1 reported (WRONG)
- E1 rel_Hess = 0.155 (correct)
- E4 LogNormal rel_Hess = 0.026 (wrong — from buggy ms28_exp)
- Anomaly A1: "E1 vs E4 discrepancy attributed to JIT context"
- Gate: MC_PRIOR_FAMILY_SWEEP_PASS (wrong — LogNormal appeared to win)

### What v2 reports (CORRECT)
- E1 rel_Hess = 0.155073 (unchanged)
- E4 LogNormal rel_Hess = 0.155072 (identical to E1, as expected)
- Root cause: `var t = rx` in ms28_exp (line 81 of prior-family sweep) instead of
  `var t: f64 = 1.0`; 18.3% exp() error per call
- Fix: one-line change; no scientific parameters altered
- Gate: MC_PRIOR_FAMILY_SWEEP_OUTPUT (corrected — no family meets ≤ 10%)

---

## Scientific conclusion (headline for writing thread)

**rel_Hess(LogNormal) = 0.155** — single canonical value, verified reproducible.

No prior family resolves the GUM/MC discord for rapamycin PBPK28. CL_hep CV = 58%
places the model firmly in the moderately nonlinear regime (Hessian second-order
correction reduces rel to 15.5% from 42.3%, but does not achieve the ≤ 10% target).
TruncNormal with physiological bounds is marginally better (14.2%) but also does not
pass. The §4.13 hypothesis remains unconfirmed.

The correct dissertation narrative: "The distributional choice (LogNormal vs
TruncNormal) is secondary to the dominant nonlinearity imposed by CL_hep CV = 58%.
Second-order GUM (Hessian correction) is necessary but not sufficient for convergence
under these uncertainty inputs."

---

## Verification command

```bash
# Confirm determinism post-fix:
bash scripts/audit/mc_determinism_probe.sh --post-fix
# Expected output includes:
#   MC_PBPK28_COMPILER_DETERMINISM_PASS
#   MC_PBPK28_DETERMINISTIC_RESULTS_PASS
```
