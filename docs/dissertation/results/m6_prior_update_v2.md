<!-- docs:meta
topic_id: repo.docs.dissertation.results.m6-prior-update-v2
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.m6-prior-update-v2
-->

---
docs:meta: true
topic: dissertation-results
kind: quantitative-output
drug: rapamycin
model: PBPK28
status: regenerated
version: m6-v2
date: 2026-09-26
---

# M6 Prior Update: CL_hep Variability — v2 (numbers regenerated on the mass-conserving kernel)

**Supersedes:** the numerical sections (§6–§9) of `m6_prior_update_v1.md`. §1–§5 of v1 (background,
literature evidence for CV(CL_hep) = 0.38, the Julia fu_plasma reconciliation and the source edit
`v[0]: 51.85 → 22.202944`) are prior-definition work that does not touch the PBPK28 stepper. They
stand unchanged and are not repeated here.

**Why v2:** every v1 PBPK28 number came through `pbpk28_full_cn_step`, whose negativity floors
injected mass into the 5 mg bolus. At the nominal parameters (audit, single trajectory) AUC_blood
was +28% at dt = 0.05 h (first-order GUM grid), +52% at 0.1 h (Hessian grid) and +140% at 0.5 h
(MC grid). For sampled parameters the amount varies, so MC means do not show these exact ratios
(`docs/audit/PBPK28_CN_RANNACHER_MASS_BALANCE_2026-09-26.md`). The stepper is now the
Rannacher-started clamp-free Crank–Nicolson of `stdlib/darwin_pbpk/theta_pbpk28.sio`. Both prior
generations were rerun on it. The legacy prior is reproduced by changing only `v[0]` back to
51.85 in a scratch stdlib copy.

## 6. Canonical numbers (regenerated)

Command (lean_single; Madaros aborts this harness with `rc=182`, before and after the fix):

```bash
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio
```

MC settings: N = 2000, seed = 1729, LogNormal; grids MC 0.5 h, GUM 0.05 h, Hessian 0.1 h.
Convention: rel_X = |u_X − u_MC| / u_MC (computed from unrounded values); gates rel_GUM ≤ 0.05,
rel_Hess ≤ 0.10. Legacy prior = `v[0]` 51.85 (σ/CL = 0.5807). The v1 columns are the May 2026
runs; the base-commit rerun reproduced the M6 v1 column exactly, while the legacy column was not
re-run at the base commit.

| Quantity | M6 v2 | Legacy v2 | M6 v1 (floored) | Legacy v1 (floored) |
|---|---:|---:|---:|---:|
| AUC_ref (GUM) | **0.403226** | 0.403226 | 0.516790 | 0.516790 |
| u_GUM | **0.183456** | 0.254969 | 0.228175 | 0.317093 |
| u_Hessian | **0.207808** | 0.326661 | 0.295160 | 0.464032 |
| MC mean AUC | **0.476478** | 0.549660 | 1.086296 | 1.204004 |
| u_MC | **0.211790** | 0.331812 | 0.357945 | 0.549197 |
| rel_GUM | **0.133783** | 0.231589 | 0.362543 | 0.422624 |
| rel_Hess(LogNormal) | **0.018801** | 0.015524 | 0.175405 | 0.155073 |
| ρ̃_normalized(CL_hep) = ½ρ_literal² ¹ | **0.072200** | 0.168607 | 0.072365 | 0.169 (doc) |
| ρ_literal(CL_hep) | **0.380000** | 0.580701 | 0.380433 | 0.581 (doc) |
| brain/blood ratio at 24 h | 0.016454 | 0.016454 | 0.016587 | 0.016587 |

¹ ρ̃ = ½ρ_literal² is not a variance ratio. CL_hep's second-order-to-first-order variance ratio is
2ρ_literal² = 0.289 (M6) / 0.674 (legacy); see `pbpk28_epistemic_v2.md` §4.10.4.

ρ_literal(CL_hep) = σ/CL exactly (0.38 and 0.5807). The closed form of u_GUM,
Dose/CL²·√(v_CLhep + v_CLren + v_fu·CL²/fu_ref²) at CL = 12.4, is 0.183454 (M6) and 0.254961
(legacy); the harness's finite-difference budget prints 0.183456 and 0.254969, 1.2×10⁻⁵ and
3.1×10⁻⁵ relative above it (finite-difference and time-discretisation error).

First-order sensitivity shares (M6, `epistemic_pbpk28.sio` at dt = 0.05 h; v1 values from the
floored Hessian CSV in brackets):

| Parameter | Share | v1 |
|---|---:|---:|
| CL_hep | 0.697609 | 0.697243 |
| CL_renal | 0.000452 | 0.000452 |
| fu_plasma | 0.301938 | 0.301780 |
| Kp_brain | 2.1e-25 | 0.000001 |
| Kp_liver | 5.2e-24 | 0.000498 |
| Kp_kidney | 2.8e-24 | 0.000024 |
| Kp_adipose | 2.5e-25 | 0.000003 |

**Caveat (pre-existing, unchanged):** the MC centres total clearance at CL_hep + CL_renal =
12.7 L/h, while the GUM/Hessian budgets linearise at `cl_central = 12.4 L/h`. The same closed
form evaluated at CL = 12.7 gives u_GUM = 0.1762 (M6, 4.0% lower; the fu term grows with CL², so
it is not a plain 1/CL² rescaling). The Hessian budget was not recomputed there. Also, for
N = 2000 the MC's own resolution on u is ≈ 1.6% relative (normal approximation, no error bar
printed). See `mc_cross_validation_lognormal_v3.md`.

## 7. Comparison and interpretation (regenerated)

M6 still lowers absolute uncertainty: u_MC 0.331812 → 0.211790 mg·h/L (ratio 0.638; v1: 0.652).

v1's central negative result, that "the Hessian/MC residual does not improve (0.155 → 0.175)
… M5 fourth-order closure remains essential", **does not survive**. On the mass-conserving
kernel, as run (limits in §6), the Hessian residual is 1.55% (legacy) and 1.88% (M6): inside the
≤ 10% gate under both priors, and of the order of the MC resolution. For M5 see
`m5_gum_4th_order_v2.md`.

What changes between the priors is the first-order residual: rel_GUM 23.2% (legacy) → 13.4%
(M6). The first-order GUM does not meet 5% under either prior. *Interpretation (the author's to
confirm):* the smaller residual is consistent with the weaker curvature of AUC ∝ 1/CL at lower
CV(CL_hep).

Canonical update statement (regenerated):

```text
The canonical M6 prior is CV(CL_hep) = 0.38.
The legacy CV = 0.58 result is retained only as historical comparison.
Under the M6 prior, on the mass-conserving PBPK28 kernel, the second-order (Hessian) GUM
differs from an N = 2000 Monte Carlo by 1.9% (criterion <= 10%; MC resolution ~1.6%;
budgets at CL = 12.4 L/h, MC centred at 12.7 L/h); the first-order GUM differs by 13.4%
(criterion <= 5%).
```

## 8. §4.7.2 prose draft — regenerated closing paragraph

The prior-evidence paragraphs of v1 §8 stand. Its closing paragraph is replaced by:

> The dissertation reports downstream GUM–Hessian and Monte Carlo results under the M6-updated
> prior. On the mass-conserving PBPK28 integrator, the updated prior reduces the Monte Carlo
> standard uncertainty of AUC_blood from 0.332 to 0.212 mg·h/L, and the relative first-order
> GUM/MC residual from 23.2% to 13.4%. Under both priors the second-order (Hessian) GUM
> residual is below 2% (1.55%, 1.88%), of the order of the Monte Carlo's own resolution at
> N = 2000. The budgets are linearised at CL = 12.4 L/h and the Monte Carlo is centred at
> 12.7 L/h.

(The Sobol/PCE results are not cited here: see `pbpk28_epistemic_v2.md` §4.10.5 for why the
Saltelli output cannot currently be used.)

## 9. Validation (regenerated)

| Artifact | Status |
|---|---|
| `runs/m6_epistemic_pbpk28_v2.txt` | 9/9, `ALL 9 TESTS PASSED` |
| `runs/m6_hessian_pbpk28_v2.txt` | 7/7 + 5/5, `HESSIAN_PBPK28_DUAL_RHO_PASS` |
| `runs/m6_full_stack_v2.txt` | `MC_CROSS_VALIDATION_PBPK28_LOGNORMAL_HESSIAN_PASS`, `M1_COPULA_CHOLESKY_PASS` |
| `runs/legacy_*_v2.txt` | legacy-prior counterparts |
| `m6_literature_access_v1.txt`, `m6_julia_reconciliation_v1.txt` | unaffected (no PBPK28 stepping) |
| `m6_dissertation_pbpk28_parity_gate_v1.txt` | unaffected: the parity refs carry their own kernel copies at dt = 0.001 h, where the floors never fire; `pbpk28_full_cn_step` is bit-identical |
| `m6_dissertation_pbpk_suite_gate_v1.txt`, `m6_dissertation_pbpk_hessian_gate_v1.txt` | not rerun here; the suite's PBPK28 members were rerun individually above |

Offload review: v1 recorded "WAIVED (keys absent)". For v2 see the review record appended to
`docs/audit/PBPK28_CN_RANNACHER_MASS_BALANCE_2026-09-26.md`.

Gate marker: `M6_PRIOR_UPDATED_PASS` (unchanged; it marks the prior edit, not a numerical
threshold).

## Provenance

Regenerated by Claude Code (Claude Opus 5.5) on 2026-09-26 (`SOUNIO_SOUC_ENGINE=lean_single`,
branch `claude/pbpk28-cn-rannacher`). Reviewed for arithmetic and overclaiming by xai grok-4.6
and qwen3-235b (`bin/llm-offload --raw`); findings applied. Scientific interpretation is the
author's.
