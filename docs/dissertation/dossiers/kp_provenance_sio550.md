<!-- docs:meta
topic_id: repo.docs.dissertation.dossiers.kp-provenance-sio550
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.dossiers.kp-provenance-sio550
-->

# Kp provenance audit — 15 scenarios (SIO-550)

Contract: every Kp used in the 15 EPIC scenarios is classified as measured,
predicted by a named method, calibrated (with fit objective and data), or
assumed, with source, unit/convention, and domain. Calibrated Kp are never
presented as independent prediction. Uncertainty budgets must not double
count one source. Computational coverage and predictive evaluation are
reported separately; no scenario is called validated by parity alone.

Convention: PBPK28 Kp are tissue:vascular at equilibrium (organ order
0=blood, 1=liver, 2=kidney, 3=brain, 4=heart, 5=lung, 6=muscle, 7=adipose,
8=gut, 9=skin, 10=bone, 11=spleen, 12=pancreas, 13=other; blood index 0 is
1.00 everywhere). PBPK14 Kp are tissue:plasma, well-stirred. Physiology is
the 70 kg reference (PBPK28: shared v_ref; PBPK14 haloperidol/midazolam:
Davies and Morris 1993 volumes in-file).

## 1. Master inventory (peripheral organs 1-12,13 as liver..other)

| Scenario (model) | Kp liver..other | Class | Source / fit objective |
| --- | --- | --- | --- |
| rapamycin PBPK28 core (parity ref consumer) | 5.40, 4.20, 0.10, 2.30, 3.10, 0.50, 0.30, 2.60, 0.80, 0.20, 2.10, 1.80, 0.40 | historical fixed, provenance NOT established | Valoteau 1996 cited in code but reference not located; brain 0.10 cites Lampen 1998 which treats intestine, not brain. Explicit hypothesis only. PENDING. |
| rapamycin PBPK14 ep14 (oral_rapamycin_bbb consumer) | 8.0, 4.0, 0.08, 3.5, 10.0, 3.5, 0.3, 5.0, 2.0, 1.5, 6.0, 4.0, 0.5 | literature-attributed | Ferron 1997 / Schreiber 1991 per scenario header; brain range 0.07-0.12 per epistemic module. |
| venlafaxine parent (core) | 4.20, 3.50, 1.20, 2.00, 2.80, 1.80, 1.20, 2.40, 1.50, 0.80, 2.00, 1.60, 0.90 | predicted (named method) | Rodgers-Rowland class estimates in Wang 2022 joint PPK context. |
| venlafaxine ODV (core) | 3.50, 3.00, 1.00, 1.60, 2.20, 1.40, 0.90, 2.00, 1.20, 0.70, 1.60, 1.30, 0.70 | predicted (named method) | Same as parent; ODV/V phenotype ratios UNSOURCED, previously miscited. PENDING. |
| semaglutide (core) | 0.50, 0.60, 0.05, 0.40, 0.30, 0.20, 0.10, 0.80, 0.30, 0.20, 0.40, 0.70, 0.30 | empirical (peptide rationale) | Vascular-confined peptide; R-R declared NOT applicable (peptide class). Overgaard 2019 popPK context. |
| tirzepatide (scenario-owned) | 0.50, 0.60, 0.05, 0.40, 0.30, 0.20, 0.10, 0.80, 0.30, 0.20, 0.40, 0.70, 0.30 | assumed (analogue) | Mirrors semaglutide peptide vector (MW 4813 vs 4114 Da); drug-specific refinement open. |
| everolimus (scenario-owned) | 5.40, 4.20, 0.10, 2.30, 3.10, 0.50, 0.30, 2.60, 0.80, 0.20, 2.10, 1.80, 0.40 | assumed (analogue) | Mirrors rapamycin vector (same mTOR class, MW 958 vs 914 Da); declared assumed in scenario header. Inherits the rapamycin gap. |
| tacrolimus (scenario-owned) | 20.0, 6.0, 0.10, 4.0, 8.0, 3.0, 5.0, 6.0, 2.0, 1.5, 8.0, 4.0, 2.0 | literature-anchored | Hepatic Kp about 20 per CYP3A4 metabolism review; brain 0.10 P-gp-restrained. |
| olanzapine (scenario-owned; PBPK14 twin in drug file) | 12.0, 5.0, 10.0, 4.0, 8.0, 15.0, 8.0, 5.0, 5.0, 4.0, 5.0, 3.5, 3.5 | calibrated to label Vd | Fit objective: Vd about 700 L (10 L/kg, Bergstrom 2006); R-R pattern named as structural prior. |
| clozapine (scenario-owned) | 10.0, 4.0, 8.0, 3.0, 6.0, 8.0, 6.0, 4.0, 3.0, 2.0, 4.0, 3.0, 3.0 | estimated (declared) | R-R pattern for basic lipophilic amine; Vd target about 400 L (5.7 L/kg); own values, distinct from olanzapine. |
| quetiapine (scenario-owned) | 8.0, 4.0, 3.0, 3.0, 6.0, 6.0, 5.0, 4.0, 3.0, 2.0, 4.0, 3.0, 3.0 | estimated (declared) | R-R pattern; Vd target about 700 L; own values; adipose capped per window note. Full-Vd variant is follow-up. |
| lithium (scenario-owned) | 0.6, 0.8, 0.4, 0.6, 0.6, 0.8, 0.2, 0.6, 0.5, 0.5, 0.6, 0.6, 0.6 | measured plus estimated (rubric) | ECF physiology plus measured anchors (brain 0.4; Vd 45 L, Amdisen range); explicit confidence rubric in drug file. |
| vancomycin (scenario-owned) | 0.6, 1.0, 0.1, 0.6, 0.6, 0.9, 0.2, 0.6, 0.5, 0.3, 0.6, 0.6, 0.9 | estimated with measured anchors | ECF rationale plus measured poor BBB (brain 0.1); Vd about 45 L vs Roberts 2-comp 49 L (residual uncharacterized). |
| risperidone parent (scenario-owned) | 4.0, 3.0, 3.0, 2.0, 3.0, 2.0, 1.0, 3.0, 1.0, 0.5, 2.5, 2.0, 2.0 | assumed (declared) | Pattern targets label Vd (parent about 115 L); declared ASSUMED in drug file. |
| risperidone 9-OH (scenario-owned) | 15.0, 10.0, 1.0, 6.0, 12.0, 10.0, 4.0, 10.0, 4.0, 1.0, 10.0, 8.0, 8.0 | assumed (declared) | Same declaration as parent. |
| aripiprazole (scenario-owned; PR 3460 queued) | 12.0, 8.0, 6.0, 5.0, 8.0, 8.0, 4.0, 8.0, 3.0, 1.5, 8.0, 6.0, 6.0 | assumed (declared) | Pattern targets label Vd (about 410 vs 404 L steady-state); declared ASSUMED in drug file. |
| haloperidol PBPK14 (drug file; oral_haloperidol_bbb consumer) | 18.0, 8.0, 15.0, 6.0, 12.0, 35.0, 12.0, 6.0, 8.0, 6.0, 8.0, 5.0, 5.0 | calibrated to label Vd (RESOLVED this lane) | Fit objective: Vd 1400 L (20 L/kg, Forsman and Ohman 1977); sum(V x Kp) = 1468 L (+4.9%). See section 2. |
| midazolam PBPK14 (drug file; oral_midazolam_ddi consumer) | 2.0, 1.5, 1.5, 1.2, 1.5, 1.0, 2.0, 1.5, 1.2, 0.6, 1.2, 1.2, 1.0 | assumed / nominal (declared) | Nominal midazolam-plausible values (Vss about 1.4 L/kg); header-explicit. F/CLh/AUCR independent of distribution. |

## 2. Haloperidol header/code contradiction — resolved

The drug-file header claimed a Rodgers-Rowland derivation listing muscle 8
and adipose 3, while the code (and its own Vd comment) used muscle 35 and
adipose 12. Discriminant: Forsman 1977 label Vd is 20 L/kg = 1400 L/70kg.
The code set implies sum(V x Kp) + V_blood = 1468 L (+4.9%, PASS); the
header set implies about 521 L (63% under, FAIL). The header now declares
Vd-calibrated status and records the R-R muscle-8/adipose-3 values as the
superseded prediction attempt, not model values.

Executable guard: tests/run-pass/dissertation_pbpk14_haloperidol_kp_vd.sio
computes Vss from the LIVE haloperidol_pbpk_params() and requires 15%
agreement with 1400 L. Verified fail-before (stale 8/3 set: 521.17 L, FAIL,
exit 1) and pass-after (code set: 1467.97 L, PASS, exit 0).

## 3. Uncertainty budget (no double counting)

Calibrated Kp (haloperidol, olanzapine, and Vd-targeted estimated sets) carry
Type A uncertainty from the Vd residual only where a stated adjustment model
exists; they do NOT inherit predictor-input (logP/fu) GUM terms for the same
tissues, and predictor-based GUM terms (core/rodgers_rowland.sio,
RRScaledUnc) apply only where Kp are genuinely predicted (venlafaxine) or as
documented sensitivity structure, never summed with a fit residual over the
same parameter. See tests/run-pass/dissertation_pbpk14_model_form_uc.sio for
the one-term model-form precedent.

## 4. Pending provenance items (unchanged semantics, explicit owners)

- Rapamycin PBPK28 table: Valoteau 1996 full reference; brain 0.10 cerebral
  measurement (Lampen 1998 is intestinal); tissue:vascular vs tissue:plasma
  denominator reconciliation before importing new values.
- Venlafaxine ODV/V phenotype ratios: source or reclassify as assumption.
- Everolimus/tirzepatide analogue Kp: drug-specific refinement or keep the
  declared assumed status with the rapamycin/semaglutide gap inherited.
- Quetiapine full-Vd variant; clozapine smoking/DDI extensions: follow-ups,
  not blockers.

Versions: main b0e0a0cdbc; aripiprazole values from PR 3460 branch
(pbpk28-aripiprazole, queued at audit time); R-R module
core/rodgers_rowland.sio untouched (still uncalled by drug files — the
prediction path remains to be wired, out of scope here).
