<!-- docs:meta
topic_id: repo.docs.dissertation.results.m5-gum-4th-order-v2
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.m5-gum-4th-order-v2
-->

---
docs:meta: true
topic: dissertation-results
kind: quantitative-output
drug: rapamycin
model: PBPK28
status: regenerated
version: m5-v2
date: 2026-09-26
---

# M5 GUM Fourth-Order Cumulant Budget - v2 (regenerated on the mass-conserving kernel)

**Supersedes:** `m5_gum_4th_order_v1.md`. Every PBPK28 number in v1 was computed through
`pbpk28_full_cn_step`, whose negativity floors injected mass into the bolus. The audit measured
this at the nominal parameters (single 5 mg trajectory): AUC_blood +52% at dt = 0.1 h (the
Hessian and derivative grid) and +140% at dt = 0.5 h (the MC run that supplied the canonical
u_MC). For sampled parameters the amount varies
(`docs/audit/PBPK28_CN_RANNACHER_MASS_BALANCE_2026-09-26.md`). The stepper is now the
Rannacher-started clamp-free Crank–Nicolson of `stdlib/darwin_pbpk/theta_pbpk28.sio`.

**Run log:** `runs/m5_gum_4th_order_v2.txt` (lean_single). The numbers come from a probe that
calls `m5_pbpk28_convergence_budget()` directly. The run-pass test
`tests/run-pass/pbpk28_m5_gum_4th_order.sio` is not the source: it does not compile on main
(pre-existing E035).
**Canonical u_MC:** 0.211790 mg·h/L, the post-fix M6 run (`runs/m6_full_stack_v2.txt`), now
the value of `m5_pbpk28_u_mc_canonical()` in `stdlib/darwin_pbpk/cumulants.sio`.
**Convention:** rel_X = |u_X − u_MC| / u_MC.

## Headline (measured)

Under the M6 prior, on the mass-conserving kernel, against this N = 2000 MC:
rel_Hess = 1.88% and rel_fourth = 25.8%. The v1 residuals (17.5% second order, 5.79% fourth
order) are not reproduced, and the fourth-order budget is further from the MC than the
second-order budget. This document does not establish whether any higher-moment closure is ever
needed for this model.

## 4.14.6 Convergence study (regenerated)

| Quantity | v1 (floored kernel) | v2 (corrected kernel) |
|---|---:|---:|
| canonical `u_MC` | 0.357945 | **0.211790** |
| `u_1st` on the Hessian grid | 0.260506 | 0.183456 |
| full `u_Hessian` | 0.295160 | 0.207808 |
| M5 `u_total` (fourth order) | 0.378674 | 0.266505 |
| `rel_Hess` | 0.175404 | **0.018801** |
| `rel_fourth` | 0.057910 | **0.258345** |
| dominant correction index | 0 (`CL_hep`) | 0 (`CL_hep`) |
| first-order variance u_1st² | 0.067863 | 0.033656 |
| second-order (Hessian) increment u_Hessian² − u_1st² | 0.019256 | 0.009528 |
| skewness contribution (signed) | −0.050860 | −0.025212 |
| excess-fourth-cumulant contribution | 0.019447 | 0.009629 |
| cubic-derivative contribution | 0.087687 | 0.043423 |
| total fourth-order variance u_total² | 0.143394 | 0.071025 |
| \|u_total − u_MC\| < \|u_Hessian − u_MC\|? | yes | **no** |

The variance rows add up: u_total² = u_1st² + Hessian increment + skewness + excess-κ₄ + cubic
(v2: 0.033656 + 0.009528 − 0.025212 + 0.009629 + 0.043423 = 0.071024). The v1 column is the v1
page, plus the two variance rows it did not print, which come from the same probe run on the
base commit.

Analytic anchor: u_1st = Dose/CL²·√(v_CLhep + v_CLren + v_fu·CL²/fu_ref²) = 0.183456 at
CL = 12.4 L/h, exact to the printed digits. The v1 derivative validation table (§4.14.4) and the
lognormal cumulants (§4.14.3, §4.14.5) do not involve the PBPK28 stepper and stand unchanged.

Gate marker: the M5 claim assertion now evaluates to `M5_GUM_FOURTH_ORDER_CUMULANT_BUDGET_OUTPUT`
(v1: `..._PASS`).

## Limits of the comparison

- **MC resolution.** For N = 2000 the MC standard deviation is itself uncertain by roughly
  1/√(2N) ≈ 1.6% relative (normal approximation; the harness prints no error bar). rel_Hess =
  1.88% is therefore of the order of the MC's own resolution, while rel_fourth = 25.8% is not.
- **Linearisation centre.** The MC centres total clearance at CL_hep + CL_renal = 12.7 L/h
  (`mc28_params_from_sample`). The GUM/Hessian/M5 budgets linearise at `cl_central = 12.4 L/h`.
  This is pre-existing and unchanged here. The same analytic formula evaluated at 12.7 L/h gives
  u_1st = 0.1762 (4.0% lower). The Hessian and M5 budgets were not recomputed there.
- **Grids.** Budgets at dt = 0.1 h, MC at dt = 0.5 h. Both are mass-conserving, but they are
  different discretisations.
- **Fourth-order overshoot.** The cubic-derivative contribution (0.043) is larger than the whole
  Hessian increment (0.0095). Why the truncated expansion overshoots was not analysed.

## Wording

**Safe to cite** (all three qualifiers belong in the sentence): "Under the M6 prior, on the
mass-conserving PBPK28 kernel, the second-order (Hessian) standard uncertainty 0.2078 mg·h/L
differs from an N = 2000 Monte Carlo estimate of 0.2118 mg·h/L by 1.9%, which is of the same
order as the Monte Carlo's own ~1.6% resolution. The budgets are linearised at CL = 12.4 L/h while the MC is
centred at 12.7 L/h. The fourth-order cumulant extension gives 0.2665 mg·h/L (25.8% above the
MC)."

**Withdrawn (v1):** "The fourth-order cumulant budget improves the residual from 17.54% to
5.79%" and "M5 higher-moment closure remains essential". Both depended on the floored kernel.

## Provenance

Regenerated by Claude Code (Claude Opus 5.5) on 2026-09-26 from the run log above
(`SOUNIO_SOUC_ENGINE=lean_single`, branch `claude/pbpk28-cn-rannacher`). Reviewed for arithmetic
and overclaiming by xai grok-4.6 and qwen3-235b (`bin/llm-offload --raw`); findings applied.
Scientific interpretation is the author's.
