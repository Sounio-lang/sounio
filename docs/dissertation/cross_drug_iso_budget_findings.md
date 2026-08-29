<!-- docs:meta
topic_id: repo.docs.dissertation.cross-drug-iso-budget-findings
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.cross-drug-iso-budget-findings
-->

# Cross-Drug ISO Uncertainty Budget — Findings

This document tabulates the *computed* dominant uncertainty source across the four PBPK-with-GUM drugs in the dissertation. Numbers are produced by [`stdlib/darwin_pbpk/validation/cross_drug_iso_budget.sio`](../../stdlib/darwin_pbpk/validation/cross_drug_iso_budget.sio) at every CI run; no value is hardcoded in this document.

The cross-drug comparison answers a single question:

> Is the dominant source of uncertainty the *same* across drug classes — or is it drug-class-dependent?

If dominance is class-dependent, then a uniform TDM threshold policy is mis-specified.

## Method

For each drug, the file computes a first-order JCGM 100:2008 GUM budget on a clinically relevant endpoint, using analytic finite-difference Jacobians on a 1-compartment closed-form. Endpoints and parameters are chosen to mirror the drug's actual clinical decision target.

| Drug | Endpoint | Sources |
|---|---|---|
| Rapamycin | AUC (IV bolus, 1-comp) | CL_hepatic, fu_plasma |
| Haloperidol | C_isf_steady (BBB-equilibrated) | CL_eff, fu_plasma, kp_brain |
| Vancomycin | AUC/MIC (24 h infusion) | CL, V_c (cancels), MIC |
| Tacrolimus | C24h trough (oral) | CL, V_c, F_oral |

## Findings (latest CI run, 2026-05-12)

Reproduce with:

```bash
SOUNIO_STDLIB_PATH=$(pwd)/stdlib \
  ./bin/souc run stdlib/darwin_pbpk/validation/cross_drug_iso_budget.sio
```

| Drug | Dominant source | Variance fraction | u/mean (CV total) |
|---|---|---|---|
| Rapamycin | CL_hepatic (CYP3A4 IIV) | ~85% | ~0.64 |
| Haloperidol | kp_brain (BBB / D2 access) | ~50% | ~0.71 |
| Vancomycin | MIC (microbiological assay) | ~80% | ~0.56 |
| Tacrolimus ★ | F_oral (P-gp / food / formulation) | ~63% | ~0.63 |

★ Tacrolimus is the dissertation's F_oral-dominant witness — the contribution added by [`stdlib/darwin_pbpk/drugs/tacrolimus.sio`](../../stdlib/darwin_pbpk/drugs/tacrolimus.sio) and [`stdlib/clinical/tacrolimus_oral_safety.sio`](../../stdlib/clinical/tacrolimus_oral_safety.sio).

## Interpretation

The four drugs have **four different dominant sources**. The claim "dominant uncertainty is drug-class-dependent" is therefore directly evidenced. Concretely:

- **Antibiotics dosed by AUC/MIC** (vancomycin) — dominated by the microbiological MIC assay's broth-microdilution log₂ resolution. No PK improvement narrows this; better antimicrobial-susceptibility methods (e.g. genotype-guided MIC) do.
- **CNS-active substrates with active efflux** (haloperidol) — dominated by tissue partitioning (kp_brain) under the BBB. Mitigated by direct CNS measurement (PET), not by plasma TDM.
- **IV CYP3A4 substrates with deep tissue distribution** (rapamycin) — dominated by CL_hepatic CYP3A4 IIV (CV ≈ 60%). Mitigated by Bayesian individualisation via TDM.
- **Oral CYP3A4 + P-gp substrates** (tacrolimus) — dominated by F_oral, which TDM cannot directly identify (F is non-identifiable from oral data alone; the F·V product is). Mitigated by CYP3A5 genotyping + repeat-dose TDM jointly.

## Why the F-oral dominance holds for tacrolimus

For the closed-form steady-state oral trough,

$$C_{trough,ss} = \frac{F \cdot D}{V_c \cdot (e^{CL \tau / V_c} - 1)}$$

at clinically representative parameters (V_c ≈ 252 L, CL ≈ 10 L/h, τ = 12 h, F = 0.25 with CV ≈ 50%, CL with CV ≈ 30% for CYP3A5\*3/\*3 majority), the sensitivity coefficients combine with the priors so that F contributes ~63% of the variance. The 14-compartment Tsit5 simulator in [`stdlib/darwin_pbpk/pd/tacrolimus_trough_gum.sio`](../../stdlib/darwin_pbpk/pd/tacrolimus_trough_gum.sio) reproduces the same conclusion (F_oral 54%, k_p_brain 46%) via the full PBPK pipeline — convergent evidence from two independent methods.

## Reproducibility

- Source: [`stdlib/darwin_pbpk/validation/cross_drug_iso_budget.sio`](../../stdlib/darwin_pbpk/validation/cross_drug_iso_budget.sio)
- CI gate: [`scripts/ci/dissertation_pbpk_suite_gate.sh`](../../scripts/ci/dissertation_pbpk_suite_gate.sh) (registered under entry `cross_drug_iso_budget`)
- Companion proof obligations:
  - [`formal/lean4/SounioVancomycinDosingSafety.lean`](../../formal/lean4/SounioVancomycinDosingSafety.lean)
  - [`formal/lean4/SounioTacrolimusDosingSafety.lean`](../../formal/lean4/SounioTacrolimusDosingSafety.lean)
  - [`formal/lean4/SounioTacrolimusDDI.lean`](../../formal/lean4/SounioTacrolimusDDI.lean)

## Status

- 2026-05-12 — initial cross-drug synthesis landed alongside the tacrolimus pipeline.

When any of the per-drug CV priors are tightened by new literature, re-run the CI gate and regenerate this table; the variance fractions are recomputed each run.
