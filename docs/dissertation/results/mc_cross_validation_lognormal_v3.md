<!-- docs:meta
topic_id: repo.docs.dissertation.results.mc-cross-validation-lognormal-v3
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.mc-cross-validation-lognormal-v3
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

# PBPK28 MC Cross-Validation — LogNormal Prior — v3 (mass-conserving kernel)

**Date:** 2026-09-26
**Replaces:** `mc_cross_validation_lognormal_v2.md` (and v1).
**Harness:** `stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio` (`mc28_selftest_main`).
**Configuration:** rapamycin 5 mg IV bolus, N = 2000, seed = 1729, LogNormal on all 7 parameters,
0–168 h. Grids: MC dt = 0.5 h, GUM dt = 0.05 h, Hessian dt = 0.1 h.
**Why v3:** v1/v2 stepped every simulation with `pbpk28_full_cn_step`, whose negativity floors
injected mass. At the nominal parameters (audit, single trajectory) that inflated AUC_blood by
+140% at dt = 0.5 h, +52% at 0.1 h and +28% at 0.05 h, and the amount varied across samples. The
three methods were therefore compared on differently biased maps. They now all use the
Rannacher-started clamp-free Crank–Nicolson of `theta_pbpk28.sio`
(`docs/audit/PBPK28_CN_RANNACHER_MASS_BALANCE_2026-09-26.md`).
**Engine:** lean_single. Under Madaros this harness aborts with `rc=182` (`handles full`), before
and after the fix.
**Run logs (post-fix):** `runs/m6_full_stack_v2.txt` (M6 prior; the run-log series number
continues v1 of that file, not this document's version),
`runs/mc_pbpk28_rapamycin_lognormal_v3.txt` (legacy prior).
**Convention:** rel_X = |u_X − u_MC| / u_MC, computed by the harness from unrounded values; u is
the standard deviation of AUC_blood in mg·h/L. Gates: rel_GUM ≤ 0.05, rel_Hess ≤ 0.10.

---

## Results

### M6 canonical prior (`v[0]` = 22.202944, CV(CL_hep) = 0.38)

| Method | u (mg·h/L) | rel. to u_MC | Gate | Verdict |
|---|---:|---:|---|---|
| GUM first-order | 0.183456 | 0.133783 | ≤ 0.05 | NOT MET |
| Hessian second-order | 0.207808 | 0.018801 | ≤ 0.10 | MET |
| **MC** | **0.211790** | — | — | — |

MC mean AUC 0.476478 mg·h/L; n_valid 2000/2000. The base-commit run (before the kernel
switch) reproduced the earlier M6 values (u_MC 0.357945, rel_Hess 0.175405) exactly. Marker:
`MC_CROSS_VALIDATION_PBPK28_LOGNORMAL_HESSIAN_PASS`.

### Legacy prior (`v[0]` = 51.85, σ/CL = 0.5807; "CV 0.58" in v1/v2)

| Method | u v2 (May 2026 run, floored) | u v3 | rel. to u_MC v2 | rel. to u_MC v3 | Verdict v3 |
|---|---:|---:|---:|---:|---|
| GUM first-order | 0.317093 | 0.254969 | 0.422624 | 0.231589 | NOT MET |
| Hessian second-order | 0.464032 | 0.326661 | 0.155073 | 0.015524 | MET |
| **MC** | **0.549197** | **0.331812** | — | — | — |

MC mean AUC 1.204004 → 0.549660 mg·h/L; n_valid 2000/2000. Marker: `…_OUTPUT` →
`…_LOGNORMAL_HESSIAN_PASS`. The legacy run then exits with rc = 1: its M1 copula row ρ(CL, fu) =
+0.3 keeps 1998/2000 draws, which the row gate rejects (`m1_copula_v2.md`). The numbers in this
section come from the independent baseline, which keeps all 2000. The legacy configuration was not
re-run at the base commit.

Analytic anchors: AUC_ref = Dose/CL = 5/12.4 = 0.403226. The closed-form first-order u_GUM =
Dose/CL²·√(v_CLhep + v_CLren + v_fu·CL²/fu_ref²) at CL = 12.4 is 0.183454 (M6) and 0.254961
(legacy); the harness's finite-difference values in the tables, 0.183456 and 0.254969, sit
1.2×10⁻⁵ and 3.1×10⁻⁵ relative above them (finite-difference and time-discretisation error).

---

## Limits of the comparison

1. **MC resolution.** For N = 2000 the MC standard deviation is uncertain by roughly
   1/√(2N) ≈ 1.6% relative (normal approximation; the harness prints no error bar). The Hessian
   residuals 0.0188 / 0.0155 are of that order; the first-order residuals 0.134 / 0.232 are not.
2. **Linearisation centre (pre-existing, unchanged).** `mc28_params_from_sample` sets
   `cl_central = (CL_hep + CL_renal)·fu_scale`, centring the MC's total clearance at 12.7 L/h.
   The GUM/Hessian budgets linearise at `cl_central = 12.4 L/h`, with CL_renal entering only as a
   perturbation. The same analytic first-order formula at 12.7 L/h gives u_GUM = 0.1762 (M6,
   4.0% lower) and 0.2440 (legacy, 4.3% lower). The Hessian budget was not recomputed at 12.7,
   so the effect on rel_Hess is not established.
3. **Grids.** GUM, Hessian and MC run at dt = 0.05, 0.1 and 0.5 h. All are mass-conserving now,
   but they are different discretisations.

## What changed in the reading

v2 read its 42% / 15.5% residuals as a "moderately-to-strongly nonlinear regime" and as the
motivation for the §4.13 truncated-prior analysis. On the mass-conserving kernel, as run
(limits above):

- the Hessian residual is 1.9% (M6) and 1.6% (legacy): inside the 10% gate, and of the order of
  the MC's own resolution;
- the first-order residual is 13.4% (M6) and 23.2% (legacy): outside the 5% gate;
- the MC mean exceeds Dose/(12.4 L/h) by 18.2% (M6) and 36.3% (legacy). That excess mixes the
  convexity of AUC ∝ 1/CL with the 12.4 vs 12.7 L/h centre difference, and is not a pure Jensen
  measure. v2's reading of MC mean = 1.204 as "Jensen upward bias" is withdrawn: at the nominal
  parameters the floored kernel already gave AUC = 0.9666 at the MC's dt.

## Dissertation wording (§4.12)

**Safe to cite (M6; the qualifiers belong in the sentence):** "On the mass-conserving PBPK28
integrator, the second-order (Hessian) GUM standard uncertainty (0.208 mg·h/L) differs from an
N = 2000 Monte Carlo estimate (0.212 mg·h/L) by 1.9%, within the 10% criterion and of the order
of the Monte Carlo's own ~1.6% resolution. The first-order GUM (0.183 mg·h/L) differs by 13.4%
and does not meet the 5% criterion. The budgets are linearised at CL = 12.4 L/h, while the Monte
Carlo is centred at CL_hep + CL_renal = 12.7 L/h."

**Withdrawn (v1/v2):** u_MC = 0.549; rel_Hess = 15.5%; "CL_hep CV = 58% places the model in the
moderately-to-strongly nonlinear regime"; "the second-order correction reduces the gap … but does
not fully resolve it"; "MC mean 1.204 = Jensen bias".

**Still forbidden:** claiming first-order GUM adequacy.

## Determinism

The M6 E1 run printed byte-identical results in two independent processes (the PR verification
capture and the regeneration run). E1 and E4 agree on u_MC for the LogNormal family under both
priors (0.211790 M6, 0.331812 legacy). `scripts/audit/mc_determinism_probe.sh` was not rerun: it
builds its ELFs under `mktemp -d`, and the workspace's /tmp is noexec (exit 126).

## Provenance

Regenerated by Claude Code (Claude Opus 5.5) on 2026-09-26 (`SOUNIO_SOUC_ENGINE=lean_single`,
branch `claude/pbpk28-cn-rannacher`). The legacy prior was reproduced by changing only `v[0]`
in a scratch stdlib copy, run from a directory whose `stdlib/` is that copy. Reviewed for
arithmetic and overclaiming by xai grok-4.6 and qwen3-235b (`bin/llm-offload --raw`); findings
applied. Scientific interpretation is the author's.
