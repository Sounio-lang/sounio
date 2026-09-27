<!-- docs:meta
topic_id: repo.docs.dissertation.results.prior-evolution-sprint-summary-v3
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.prior-evolution-sprint-summary-v3
-->

---
docs:meta: true
topic: dissertation-results
kind: quantitative-output
drug: rapamycin
model: PBPK28
status: regenerated
version: v3
date: 2026-09-26
---

# PBPK28 Prior Evolution Sprint — Summary v3 (mass-conserving kernel)

**Date:** 2026-09-26
**Replaces:** `prior_evolution_sprint_summary_v2.md` (and v1).

v2 corrected v1's `ms28_exp` Taylor defect; that correction stands. v3 corrects a second defect
underneath every number in v1 and v2: all PBPK28 simulations stepped with `pbpk28_full_cn_step`,
whose negativity floors injected mass into the 5 mg bolus. At the nominal parameters (audit,
single trajectory) AUC_blood was +28% at the GUM's dt = 0.05 h, +52% at the Hessian's 0.1 h and
+140% at the MC's 0.5 h. For sampled parameters the amount varies, which is why the MC means below
do not show these exact ratios (`docs/audit/PBPK28_CN_RANNACHER_MASS_BALANCE_2026-09-26.md`). All
harnesses now step with the Rannacher-started clamp-free Crank–Nicolson of `theta_pbpk28.sio`.
The audit's per-consumer mass-balance gates measure its conservation residual at
0.17–0.32 × 10⁻¹² relative.

Engine: lean_single. E1 and E4 still abort under Madaros with `rc=182` (`handles full`),
unchanged by this fix.

**Convention:** u = standard deviation of AUC_blood(0–168 h) in mg·h/L; rel_X =
|u_X − u_MC| / u_MC, computed by the harnesses from unrounded values, so recomputing from the
rounded u's here can differ in the last digit. Gates: rel_GUM ≤ 0.05, rel_Hess ≤ 0.10 (the only
tolerances used below).

---

## Legacy prior (`v[0]` = 51.85, σ/CL = 0.5807), v2 kernel vs v3 kernel

### E1: MC cross-validation (LogNormal, N = 2000, seed = 1729)

| Quantity | v2 (May 2026, floored) | v3 |
|---|---:|---:|
| u_GUM | 0.317093 | 0.254969 |
| u_Hessian | 0.464032 | 0.326661 |
| u_MC | 0.549197 | 0.331812 |
| rel_GUM | 0.422624 | 0.231589 |
| rel_Hess | 0.155073 | 0.015524 |
| MC mean AUC (mg·h/L) | 1.204004 | 0.549660 |

### E4: Prior-family sweep (N = 2000, seed = 1729)

| Family | u_MC v3 | rel_Hess v2 | rel_Hess v3 | rel_Hess ≤ 0.10? |
|---|---:|---:|---:|---|
| Gaussian (positive) | 0.914158 | 0.693002 | 0.642664 | NO |
| LogNormal | 0.331812 | 0.155072 | 0.015523 | YES |
| TruncNormal | 0.322173 | 0.141938 | 0.013932 | YES |

E1 and E4 give identical u_MC for LogNormal (0.331812); their rel_Hess values differ by 1×10⁻⁶
from independent rounding in the two harnesses.

## M6 prior (`v[0]` = 22.202944, CV(CL_hep) = 0.38)

E1: u_GUM 0.183456, u_Hessian 0.207808, u_MC 0.211790, rel_GUM 0.133783, rel_Hess 0.018801,
MC mean 0.476478. E4: LogNormal rel_Hess 0.018800 (YES); TruncNormal 0.279486 (NO); Gaussian
0.632187 (NO).

## Module markers (v3)

```
MC_CROSS_VALIDATION_PBPK28_LOGNORMAL_HESSIAN_PASS   (v2: ..._OUTPUT) — both priors; LogNormal rel_Hess <= 0.10
MC_PRIOR_FAMILY_SWEEP_PASS                          (v2: ..._OUTPUT) — both priors; at least one family meets rel_Hess <= 0.10,
                                                      Gaussian fails under both, TruncNormal fails under M6
M1_COPULA_CHOLESKY_PASS, M2_HIERARCHICAL_PRIOR_PASS (v1 M2: ..._OUTPUT)
```

Determinism: the M6 E1 and E4 runs printed byte-identical results in two independent processes
(PR verification capture and regeneration run). `scripts/audit/mc_determinism_probe.sh` was not
rerun: it builds its ELFs under `mktemp -d`, and the workspace's /tmp is noexec (exit 126).

---

## Findings for the writing thread (measured, with their limits)

- **LogNormal family:** rel_Hess = 0.0155 (legacy) and 0.0188 (M6), inside the 0.10 gate. The
  first-order residual is 0.232 (legacy) and 0.134 (M6), outside the 0.05 gate.
- **TruncNormal:** rel_Hess 0.0139 (legacy, inside) and 0.279 (M6, outside).
  **Gaussian (positive):** 0.643 / 0.632, outside under both.
- **Limits:** the MC resolves u only to ≈ 1.6% relative at N = 2000 (normal approximation). The
  GUM/Hessian budgets linearise at CL = 12.4 L/h while the MC centres total CL at 12.7 L/h, a
  2.4% difference in the clearance centre. The methods use different dt grids. Differences of
  about 2% in rel_Hess are therefore not resolved by these runs.

v2's narrative, "the distributional choice is secondary to the dominant nonlinearity imposed by
CL_hep CV = 58%; second-order GUM is necessary but not sufficient", **is withdrawn**. Its
quantitative basis (rel_Hess 0.155, rel_GUM 0.423) came from the floored integrator.

Candidate replacement sentence (the author's to accept): "On the mass-conserving integrator, the
second-order (Hessian) GUM with LogNormal priors meets the 10% criterion against N = 2000 Monte
Carlo under both hepatic-clearance priors (rel_Hess 0.016 and 0.019, of the order of the Monte
Carlo resolution; budgets and Monte Carlo centred at CL = 12.4 and 12.7 L/h respectively). The
first-order GUM does not meet the 5% criterion (0.23 and 0.13). The truncated-normal family meets
the 10% criterion only under the legacy prior."

Caveat (pre-existing): see `mc_cross_validation_lognormal_v3.md` for the clearance-centre
mismatch.

## Verification commands

```bash
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run stdlib/darwin_pbpk/validation/pbpk28_mc_prior_family_sweep.sio
```

For the legacy prior: set `v[0]` in `ep28_rapamycin_priors()` to 51.85 in a scratch copy of
`stdlib/`, and run from a directory whose `stdlib/` is that copy. Measured on 2026-09-26: with the
worktree's own `stdlib/` present in the working directory, the compiler resolved imports there
even when `SOUNIO_STDLIB_PATH` pointed at the copy.

## Provenance

Regenerated by Claude Code (Claude Opus 5.5) on 2026-09-26. Reviewed for arithmetic and
overclaiming by xai grok-4.6 and qwen3-235b (`bin/llm-offload --raw`); findings applied.
Scientific interpretation is the author's.
