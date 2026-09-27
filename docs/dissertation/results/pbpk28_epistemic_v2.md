<!-- docs:meta
topic_id: repo.docs.dissertation.results.pbpk28-epistemic-v2
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.pbpk28-epistemic-v2
-->

---
docs:meta: true
topic: dissertation-results
kind: quantitative-output
drug: rapamycin
model: PBPK28
status: regenerated
version: v2
date: 2026-09-26
---

# PBPK28 Epistemic Uncertainty Budget — v2 Results (mass-conserving kernel)

**Supersedes:** `pbpk28_epistemic_v1.md` and the run bundle `pbpk28_epistemic_runs_v1.txt`.
**Drug**: Rapamycin (Sirolimus), IV bolus 5 mg, window 0–168 h.
**Model**: 28-state permeability-limited PBPK (`PBPKState28`: 14 organs × {C_v, C_t}); the only
elimination is central, with clearance `cl_central`. fu_plasma acts through it:
CL_eff = CL·fu/fu_ref (`ep28_perturb`).
**Kernel**: clamp-free Crank–Nicolson with Rannacher start-up (`stdlib/darwin_pbpk/theta_pbpk28.sio`).
The first two steps after the bolus are backward-Euler half-step pairs, and AUCs use each
sub-step's own quadrature. v1 used `pbpk28_full_cn_step`, whose negativity floors injected mass.
At the nominal parameters (audit, single 5 mg trajectory) that was 1.41 mg at dt = 0.05 h and
2.59 mg at dt = 0.1 h (`docs/audit/PBPK28_CN_RANNACHER_MASS_BALANCE_2026-09-26.md`).
**Prior**: M6 canonical, `v[0] = 22.202944` (CV(CL_hep) = 0.38). The *legacy* prior is
`v[0] = 51.85` (σ/CL = 0.5807, called "CV 0.58" in v1), reproduced by changing only that constant
in a scratch stdlib copy.
**Run logs** (lean_single): `runs/m6_epistemic_pbpk28_v2.txt`, `runs/m6_hessian_pbpk28_v2.txt`,
`runs/pbpk28_sobol_pce_v2.txt`, `runs/legacy_epistemic_pbpk28_v2.txt`,
`runs/legacy_hessian_pbpk28_v2.txt`.

---

## §4.10.3 — ISO GUM Table H.1: First-Order Uncertainty Budget

**Endpoint**: AUC_blood(0→168 h), mg·h/L. Central-difference Jacobian (JCGM 100:2008 §5.1.3),
h_i = max(1×10⁻⁶|μ_i|, 1×10⁻²σ_i), dt = 0.05 h.

| # | Parameter | μ | σ² | Fraction of Var(AUC_blood) |
|---|-----------|---|-----|---------:|
| 0 | CL_hepatic (L/h) | 12.4 | 22.202944 | **0.697609** |
| 1 | CL_renal (L/h)   | 0.30 | 0.0144 | 0.000452 |
| 2 | fu_plasma        | 0.08 | 0.0004 | **0.301938** |
| 3 | Kp_brain         | 0.10 | 0.000625 | 2.1e-25 |
| 4 | Kp_liver         | 5.40 | 1.8225 | 5.2e-24 |
| 5 | Kp_kidney        | 4.20 | 1.1025 | 2.8e-24 |
| 6 | Kp_adipose       | 0.30 | 0.0144 | 2.5e-25 |

| Output | v1 (floored) | v2 |
|---|---:|---:|
| AUC_blood mean (mg·h/L) | 0.516790 | **0.403226** (= Dose/CL) |
| AUC_blood SD (u_GUM, mg·h/L) | 0.228175 | 0.183456 |
| CV(AUC_blood) | 0.441523 | 0.454971 |
| AUC_liver (mg·h/L) | 2.284939 | 1.804839 |
| C_brain at 24 h (mg/L) | 5.1e-5 | 3.9e-5 |
| C_brain/C_blood ratio (approx.) | 0.016587 | 0.016454 |
| evidence-weighted confidence Σ ε_i·s_i | 0.671038 | 0.671068 |
| remaining body mass M(168 h)/Dose | — | ≈ 4×10⁻¹⁸ |
| mass-identity residual \|M(T)+CL·AUC−Dose\|/Dose | not checked (1.41 mg injected) | 3.2×10⁻¹³ |
| self-test | 9/9 | 9/9 |

**Analytic check.** The integrator conserves mass exactly, so AUC_blood(0–168 h) =
(Dose − M(168 h))/CL_eff, and M(168 h)/Dose ≈ 4×10⁻¹⁸. CL_hep and CL_renal enter CL_eff
additively, and fu multiplies it. The first-order fractions therefore follow from the priors:
v_CLhep : v_CLren : v_fu·CL²/fu_ref² = 22.202944 : 0.0144 : 9.61, giving 0.697606 / 0.000452 /
0.301942. The measured finite differences match to ≈ 1×10⁻⁵ (FD truncation), and
u_GUM = Dose/CL²·√31.827344 = 0.183456. The Kp parameters have no route to AUC_blood except
through M(168 h).

**Change from v1.** v1's narrative table *expected* Kp_liver at ~5–15% and Kp_kidney at ~3–8%.
The v1 run itself gave 2.2×10⁻⁴ and 1.1×10⁻⁴. On the mass-conserving kernel all four Kp fractions
are ≤ 5.2×10⁻²⁴, i.e. rounding. On the floored kernel the Kp fractions were ~10⁻⁴; they vanish
once the mass injection is removed. Kp still matters for organ endpoints. The v2 self-test prints
each Kp's share of its own organ endpoint (runs/m6_epistemic_pbpk28_v2.txt, TEST 5):

| Kp | endpoint | share of that endpoint's variance |
|---|---|---:|
| kp_liver | Var(AUC_liver) | 0.215235 |
| kp_kidney | Var(AUC_kidney) | 0.216488 |
| kp_brain | Var(C_brain, 24 h) | 0.005374 |

*Legacy prior:* fractions CL_hep / CL_ren / fu = 0.843448 / 0.000234 / 0.156318 (analytic
0.843441 / — / 0.156325); u_GUM = 0.254969; CV(AUC) = 0.632322; AUC mean unchanged at 0.403226
(`runs/legacy_epistemic_pbpk28_v2.txt`).

**Measured facts (regenerated)**:
- CL_hepatic carries 69.8% and fu_plasma 30.2% of Var(AUC_blood); together > 99.9%.
- C_brain/C_blood ≈ 0.016 at 24 h (Kp_brain = 0.10 in the parameter set).
- AUC_liver = 1.80 and AUC_blood = 0.40 mg·h/L (ratio 4.48). Kp_liver = 5.40 is a model input;
  the ratio is not equal to it.
- Evidence-weighted confidence (Σ over parameters of prior ε_i times variance fraction s_i) =
  0.671. It barely moves from v1 because the CL/fu fractions barely move.

---

## §4.10.4 — Hessian-Corrected Budget (Second-Order GUM)

`epistemic_pbpk28_hessian.sio`, dt = 0.1 h, 3+4-point central FD stencil. Definitions (module
`h28_nonlinearity_ratio_*`): ρ_literal,i = |½H_ii σ_i²| / |c_i σ_i|; ρ̃_i = ½ρ_literal,i².
**ρ̃ is not a variance ratio.** For a normal input, the per-parameter second-order-to-first-order
variance ratio is ½H_ii²σ_i⁴/(c_i²σ_i²) = 2ρ_literal,i² = 4ρ̃_i (last column). The module's
earlier comments and v1 called ρ̃ that ratio, which is a factor-of-4 error (PR #2696 review). The
editorial threshold "ρ̃ < 0.20 = weakly nonlinear" was set on ρ̃ as computed. Whether the
dissertation should use ρ̃ or the variance ratio 2ρ² (CL_hep: 0.289, above 0.20) is the
author's decision; it is not changed here.

| Parameter | ρ_literal | ρ̃ = ½ρ² | v1 ρ_literal / ρ̃ | variance ratio 2ρ² = 4ρ̃ |
|---|---:|---:|---|---:|
| CL_hepatic | **0.380000** | **0.072200** | 0.380433 / 0.072365 | 0.2888 |
| fu_plasma | 0.250000 | 0.031250 | 0.249911 / 0.031228 | 0.1250 |
| CL_renal | 0.009677 | 0.000047 | 0.009795 / 0.000048 | 0.0002 |
| Kp_brain | 0 (no resolvable effect) | 0 | 0.349828 / 0.061190 | 0 |
| Kp_adipose | 0 | 0 | 0.333576 / 0.055636 | 0 |
| Kp_kidney | 0 | 0 | 0.138407 / 0.009578 | 0 |
| Kp_liver | 0 | 0 | 0.066835 / 0.002233 | 0 |

For AUC ∝ 1/CL_eff with CL_eff linear in CL_hep and in fu, ρ_literal equals that parameter's
σ/μ: 0.38 for CL_hep and 0.25 for fu, as measured. The Kp rows are 0 because |c_i|σ_i lies below
the finite-difference rounding floor (`h28_first_order_resolvable`, 1×10⁻⁹·AUC_ref). Without that
guard, noise divided by noise printed values up to ρ̃ = 1.1×10⁶. The v1 Kp values (0.35, 0.33, …)
were computed on the floored kernel and do not recur on the mass-conserving one.

| Budget total | v1 | v2 |
|---|---:|---:|
| AUC_ref (first-order mean, mg·h/L) | 0.611694 | **0.403226** |
| Hessian mean-corrected (mg·h/L) | 0.729467 | 0.486692 |
| mean shift (Hessian − first order)/first order | +19.25% | **+20.70%** |
| u₁ = √var₁ (mg·h/L) | 0.2605 | **0.1835** |
| u₂ = √var₂ (mg·h/L) | 0.2952 | **0.2078** |
| var₂/var₁ | 1.284 | **1.283** |
| ∂AUC/∂CL (analytic −Dose/CL² = −0.032518) | −0.046164 | −0.032519 |
| H₀₀ (analytic 2·Dose/CL³ = 0.005245) | 0.007454 | 0.005245 |

**§4.9 / §4.10.4 wording (regenerated).** "For CL_hepatic, ρ_literal = 0.380 (= σ/CL): its
diagonal Hessian term adds variance equal to 2ρ² = 29% of its first-order variance (normal-input
formula ½H²σ⁴). fu_plasma
follows (ρ_literal = 0.250, 12.5%). The Kp parameters have no resolvable first-order effect on
AUC_blood." Withdrawn from v1: "only marginally ahead of Kp_brain (ρ̃ = 0.061)", and any wording
that equates ρ̃ = 0.072 with "~7% additional variance".

v1's headline sentence becomes: "the second-order (Hessian) mean is **20.7%** above the
first-order mean, and the second-order variance is **28%** larger than the first-order variance
(standard uncertainty 13%)." The relative figures barely move from v1 because, for
AUC ∝ 1/CL_eff, they depend on the parameters' CVs rather than on the mean. The v1 rule stands:
ρ_literal = 0.380 is CL_hep's ratio, not a "model nonlinearity".

*Legacy prior* (`runs/legacy_hessian_pbpk28_v2.txt`): ρ_literal(CL_hep) = 0.580701 (= σ/CL),
ρ̃ = 0.168607; var₁ = 0.065009, var₂ = 0.106708 (u₁ = 0.2550, u₂ = 0.3267, ratio 1.641);
Hessian mean-corrected AUC 0.564443 (+39.98%). 7/7 + 5/5 pass.

---

## §4.10.5 — Sobol' indices (rapamycin) — estimator output, not validated

`validation/pbpk28_sobol_pce.sio`, Saltelli N = 512, dt = 0.5 h. Two pre-existing problems stop
these numbers from being used as Sobol' indices:
1. The module maps CL_hep with its own hard-coded CV = 0.58 (`sp28_cv(0)`). It never adopted the
   M6 prior.
2. The estimator's output is not a consistent Sobol' decomposition. It reports S_i[CL_hep] =
   0.000000 with S_Ti[CL_hep] = 1.000000, and in the semaglutide block a first-order index above
   its total-order index (S_i(fu) = 0.9505 > S_Ti(fu) = 0.5557). True indices satisfy
   S_i ≤ S_Ti. Both features are present in v1 and v2 alike. They point to
   `stdlib/epistemic/sobol.sio`, which was not examined here. The module's TEST 7 now checks
   S_i ≤ S_Ti for all 7 parameters and fails in both blocks. In this one it is violated for
   CL_renal (0.018363 > 0.001414) and, at noise level, for Kp_brain and Kp_kidney.

Raw estimator output, for the record only:

| Parameter | S_i v1 run | S_i v2 | S_Ti v2 |
|---|---:|---:|---:|
| CL_hepatic | 0.000000 | 0.000000 | 1.000000 |
| CL_renal | 0.029295 | 0.018363 | 0.001414 (TEST 7) |
| fu_plasma | 0.049450 | 0.094698 | not printed |
| Kp_brain | 0.000000 | 9.7e-11 | 4.0e-20 (TEST 7) |
| Kp_liver | 0.000003 | ~0 | 1.3e-15 |
| Kp_kidney | 0.000052 | 1.1e-9 | 5.1e-18 (TEST 7) |
| Kp_adipose | 0.000000 | ~0 | not printed |

v1's dissertation claims for this section ("quasi-additive … ρ_add ∈ [0.85, 0.99]", "CL_hepatic
alone accounts for ≥ 60% of AUC variance" as a Sobol' result) are **withdrawn**. The first-order
GUM fractions in §4.10.3 are the verified variance attribution.

## §4.10.6 — PCE (bivariate {CL_hep, fu})

| PCE output | v2 |
|---|---:|
| S_CL (PCE) | 0.801029 |
| S_fu (PCE) | 0.148907 |
| S_ij (PCE) | 0.050064 |
| PCE CL fraction S_CL/(S_CL+S_fu) | 0.843246 |

The PCE block is a separate bivariate product model that does not call the PBPK28 stepper. It is
unchanged, and it does not validate the Saltelli numbers above. v1's "Saltelli/PCE agree within
20%" is withdrawn (the Saltelli CL fraction is 0). **Build caveat (pre-existing):** lean_single
compiles this module with 36 "tuple index out of bounds" errors in `stdlib/epistemic/pce.sio`
(lines 332–520) and still emits an ELF, on the base commit as well. Whether the PCE values pass
through that code was not verified; treat them as unverified until the build is clean.

## §4.10.7 — Semaglutide

See `sobol_pce_semaglutide_v2.md`. v1's semaglutide prior table is a prior definition, not a
simulation output, and stands.

---

## Implementation status (v2)

| Deliverable | File | Self-tests |
|---|---|---|
| First-order GUM | `epistemic_pbpk28.sio` | 9/9 (TEST 5, 7, 9 re-derived 2026-09-26) |

TEST 9 now gates the convergence order on C_brain(24 h), a point value that carries the
transient discretisation error. Every AUC here is dt-independent under exact conservation, so AUCs
cannot show order. Halving dt from 0.05 to 0.025 and 0.0125 h changes C_brain(24 h) by 1.02×10⁻¹⁰
and 2.55×10⁻¹¹ mg/L, a ratio of 3.9998, i.e. second order. One fix was needed: the old rule
`t ≥ 24` sampled one step late at the finer dts, because float accumulation left `t` a few ulp
below 24. The sample is now taken at the step nearest 24 h. The production dt = 0.05 h value is
unchanged.
| Hessian correction | `epistemic_pbpk28_hessian.sio` | 7/7 + 5/5 dual-ρ |
| Sobol + PCE | `validation/pbpk28_sobol_pce.sio` | 6/7 + 6/7: TEST 7 (S_i ≤ S_Ti) fails in both; rc = 2, no PASS marker (see §4.10.5) |
| Mass-balance gates | `tests/run-pass/pbpk28_consumer_mass_balance.sio` | PASS |

## Provenance

Regenerated by Claude Code (Claude Opus 5.5) on 2026-09-26 from the run logs above
(`SOUNIO_SOUC_ENGINE=lean_single`, branch `claude/pbpk28-cn-rannacher`). Reviewed for arithmetic
and overclaiming by xai grok-4.6 and qwen3-235b (`bin/llm-offload --raw`); findings applied.
Scientific interpretation is the author's.
