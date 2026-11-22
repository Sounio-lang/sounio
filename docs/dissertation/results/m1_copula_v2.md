<!-- docs:meta
topic_id: repo.docs.dissertation.results.m1-copula-v2
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.m1-copula-v2
-->

---
docs:meta: true
topic: dissertation-results
kind: quantitative-output
drug: rapamycin
model: PBPK28
status: regenerated
version: m1-v2
date: 2026-09-26
---

# M1 Copula Sweep v2 (mass-conserving kernel)

**Supersedes:** `m1_copula_v1.md`. The Cholesky/copula implementation, its self-audit and its
regression test (§ Self-Audit in v1) are unchanged. Two behaviours changed under every MC sample
and under the GUM/Hessian budgets:
1. **Stepper.** v1 used `pbpk28_full_cn_step`, whose negativity floors injected mass. At the
   nominal parameters (audit) that inflated AUC_blood by +140% at the MC's dt = 0.5 h, +28% at the
   GUM's dt = 0.05 h and +52% at the Hessian's dt = 0.1 h, and the amount varied across samples
   (`docs/audit/PBPK28_CN_RANNACHER_MASS_BALANCE_2026-09-26.md`). All paths now use the
   Rannacher-started clamp-free Crank–Nicolson of `theta_pbpk28.sio`.
2. **Fail-closed samples.** A sample whose run is non-finite, runs away, or books negative mass
   of magnitude above 5×10⁻⁷ mg is now rejected (`pbpk28_bolus_run_mut`). The floored kernel
   never rejected a sample.

**Convention:** rel_X = |u_X − u_MC| / u_MC (computed from unrounded values); u and mean AUC in
mg·h/L. Row gate (`M1_COPULA_SWEEP_PASS`): every copula row keeps all 2000 draws (n_valid = 2000)
with u_MC > 0, and the ρ = 0 row reproduces the independent sampler. A failed row gate ends the run
with rc = 1 and no M1 PASS marker.

**Harness:** `stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio` (the copula sweep follows
the independent baseline). N = 2000, seed = 1729, LogNormal marginals, rapamycin 5 mg.
**Engine:** lean_single. **Run logs:** `runs/mc_pbpk28_rapamycin_lognormal_v3.txt` (legacy prior, the v1
configuration) and `runs/m6_full_stack_v2.txt` (M6 prior).

## Results — legacy prior (`v[0]` = 51.85, σ/CL = 0.5807; the v1 configuration, v1 = May 2026 run)

u_GUM = 0.254969 (v1: 0.317093); u_Hessian = 0.326661 (v1: 0.464032) mg·h/L.

| Scenario | ρ(CL_hep, fu) | ρ(CL_hep, CL_ren) | n_valid | mean AUC (mg·h/L) | u_MC | rel_GUM | rel_Hess | rel_Hess v1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| independent baseline | 0.0 | 0.0 | 2000 | 0.549660 | 0.331812 | 0.231589 | **0.015524** | 0.155073 |
| sweep_1 | −0.7 | 0.0 | 2000 | 0.503252 | 0.199799 | 0.276123 | 0.634948 | 0.376618 |
| sweep_2 | −0.5 | 0.0 | 2000 | 0.516344 | 0.239014 | 0.066753 | 0.366706 | 0.159906 |
| sweep_3 | −0.3 | 0.0 | 2000 | 0.529547 | 0.276629 | 0.078303 | 0.180863 | 0.007934 |
| sweep_4 | 0.0 | 0.0 | 2000 | 0.549660 | 0.331812 | 0.231589 | 0.015524 | 0.155073 |
| sweep_5 | +0.3 | 0.0 | **1998** | 0.570682 | 0.386567 | 0.340429 | 0.154968 | 0.272936 |
| combined | −0.5 | +0.3 | 2000 | 0.518316 | 0.243826 | 0.045700 | 0.339733 | 0.138421 |

## Results — M6 canonical prior (CV(CL_hep) = 0.38)

u_GUM = 0.183456; u_Hessian = 0.207808 mg·h/L.

| Scenario | ρ(CL_hep, fu) | ρ(CL_hep, CL_ren) | n_valid | mean AUC (mg·h/L) | u_MC | rel_GUM | rel_Hess |
|---|---:|---:|---:|---:|---:|---:|---:|
| independent baseline | 0.0 | 0.0 | 2000 | 0.476478 | 0.211790 | 0.133783 | **0.018801** |
| sweep_1 | −0.7 | 0.0 | 2000 | 0.448293 | 0.114802 | 0.598028 | 0.810152 |
| sweep_2 | −0.5 | 0.0 | 2000 | 0.456390 | 0.145841 | 0.257916 | 0.424894 |
| sweep_3 | −0.3 | 0.0 | 2000 | 0.464428 | 0.173501 | **0.057376** | 0.197734 |
| sweep_4 | 0.0 | 0.0 | 2000 | 0.476478 | 0.211790 | 0.133783 | 0.018801 |
| sweep_5 | +0.3 | 0.0 | 2000 | 0.488564 | 0.248060 | 0.260437 | 0.162267 |
| combined | −0.5 | +0.3 | 2000 | 0.457316 | 0.148202 | 0.237882 | 0.402200 |

Under both priors the ρ = 0 copula row reproduces the independent sampler exactly
(`delta_mean = 0`, `delta_u_MC = 0`). Gates: under the M6 prior `M1_COPULA_SWEEP_PASS` and
`M1_COPULA_CHOLESKY_PASS`; under the legacy prior the row gate **fails** (sweep_5, below), the run
prints `M1_COPULA_SWEEP_OUTPUT` and exits with rc = 1.

**n_valid = 1998 (legacy, ρ = +0.3).** Two samples were rejected by the fail-closed checks
(item 2 above), so the legacy sweep_5 row fails the row gate. Rejection depends on the sampled
parameters, so the 1998 retained draws are not a sample of the stated prior: this row's mean and
u_MC are printed for the record and **must not be cited** as the ρ = +0.3 estimate. Which check
fired for the two draws was not diagnosed.

## What changed in the reading

v1's §4.10 paragraph argued that moderate negative CL_hep–fu dependence (ρ = −0.3) moves PBPK28
*across* the Hessian/MC threshold (rel_Hess 0.0079) while independence does not (0.155). On
the mass-conserving kernel the pattern inverts:

- **independence gives the best Hessian agreement** (1.6% legacy, 1.9% M6), and every non-zero
  correlation swept makes it worse. This is expected of a GUM/Hessian budget that is itself
  computed under independent priors;
- negative dependence still compresses u_MC monotonically (M6: 0.212 → 0.115 from ρ = 0 to
  −0.7), and at ρ = −0.3 (M6: 5.7%) or ρ = −0.5 (legacy: 6.7%) the *first-order* GUM lands
  close to the MC. The legacy combined row (ρ_fu = −0.5, ρ_ren = +0.3) is the only scenario whose
  u_MC lies within 5% of the independence-based first-order u_GUM (4.6%). That is a numerical
  coincidence of two different quantities, not a validation of the GUM under that copula;
- no correlated scenario meets the 10% Hessian criterion under either prior.

The v1 paragraph's citation of Krauss et al. (2015) as methodological precedent for testing
dependence is unaffected. Its numerical sentences are withdrawn. A regenerated paragraph based on
the measured values above would read:

> In the M1 sensitivity sweep, the independent LogNormal prior gives the closest second-order
> GUM/MC agreement (rel_Hess = 1.9% under the M6 prior). Introducing negative CL_hep–fu
> dependence compresses the Monte Carlo spread (ρ = −0.7: u_MC = 0.115 mg·h/L), positive
> dependence widens it (ρ = +0.3: u_MC = 0.248 mg·h/L), and both degrade the agreement
> with the independence-based Hessian budget (rel_Hess 16–81% for |ρ| ≥ 0.3), while at ρ = −0.3
> the first-order GUM comes within 5.7% of MC. Posterior updating that estimates a non-zero joint
> prior structure would therefore need a correlation-aware second-order budget.

(The last sentence is an inference for the author to accept or reject; it is not a measured
result.)

## Provenance

Regenerated by Claude Code (Claude Opus 5.5) on 2026-09-26 from the run logs above
(`SOUNIO_SOUC_ENGINE=lean_single`). The legacy prior was reproduced by changing only `v[0]` to
51.85 in a scratch stdlib copy; the legacy configuration was not re-run at the base commit. The v1
compiler-pin section refers to the v1 run and does not apply to v2. Limits as in
`mc_cross_validation_lognormal_v3.md` (MC resolution ≈ 1.6%, 12.4 vs 12.7 L/h centre, mixed
grids). Reviewed for arithmetic and overclaiming by xai grok-4.6 and qwen3-235b
(`bin/llm-offload --raw`); findings applied. Scientific interpretation is the author's.
