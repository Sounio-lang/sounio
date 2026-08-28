<!-- docs:meta
topic_id: repo.docs.research.irb-protocol-draft
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.irb-protocol-draft
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# IRB Protocol Draft — Vancomycin Verified CDS Retrospective Cohort

**Status**: M0 deliverable. Drafted for institutional review submission. Final submission pending PI institutional details.

**Title**: Retrospective validation of a formally verified clinical decision support tool for vancomycin dosing in critically ill patients under Knightian uncertainty.

**Principal Investigator**: Demetrios Chiuratto Agourakis, MD
**Institution**: [TO FILL]
**Anticipated submission**: M0 (immediate; lead-time governs M4 start).

## 1. Background and Significance

Vancomycin is among the most prescribed antibiotics in intensive care. Therapeutic drug monitoring (TDM) is mandated in most settings to balance efficacy (AUC₂₄/MIC ≥ 400 mg·h/L) against acute kidney injury risk (AKI; observed 7–19% in modern cohorts). Current decision support tools (BestDose, InsightRX, PrecisePK) report point estimates with narrow confidence intervals and do not represent **Knightian uncertainty** — uncertainty about the underlying probability distribution itself, arising from assay bias, model misspecification, and population heterogeneity.

This study evaluates a Sounio-based CDS that propagates Knightian uncertainty from input (weight, CrCl, trough assay) through a 2-compartment PBPK model to AUC and Cmin predictions, with **Lean 4 formal proofs** of safety bounds. It is, to our knowledge, the first formally verified clinical decision support system to handle non-Bayesian uncertainty.

## 2. Specific Aims

**Aim 1**: Validate that Sounio CDS predictions of Cmin and AUC₂₄ are non-inferior to current SOTA (Bayesian forecasting via NPAG/NONMEM) when measured against observed TDM values, MAE as primary outcome, in a retrospective cohort of ICU vancomycin patients.

**Aim 2**: Validate that Sounio's Lean 4 safety theorem — under Knightian Cmin estimate, the recommended dose satisfies `AUC₂₄/MIC ≥ 400 ∧ P(AKI) < 0.10` — corresponds to observed clinical outcomes (efficacy and AKI rates) at non-inferior levels.

**Aim 3**: Characterize cases where Knightian uncertainty bands are **wider** than Bayesian credible intervals (expected in early-treatment, low-data regimes) and assess whether the wider bands prevent unsafe recommendations.

## 3. Methods

### 3.1 Study design

Retrospective cohort, single-center (PI institution), with optional MIMIC-IV public-data validation arm if institutional data acquisition delayed.

### 3.2 Population

**Inclusion**:
- Adult ICU patients (≥ 18 yr) admitted between [DATE_START] and [DATE_END]
- Received intravenous vancomycin for ≥ 48 hours
- ≥ 2 trough levels measured during therapy
- Baseline serum creatinine and weight available

**Exclusion**:
- Renal replacement therapy at any point during vancomycin course
- Pregnancy
- Burn injury > 20% TBSA (altered Vd)

**Target**: 100 patients (Aim 1 powered for MAE non-inferiority margin of 1.5 mg/L on Cmin).

### 3.3 Data extraction

For each patient:
- Demographics: age, sex, weight, height
- Renal: serum creatinine over time, calculated CrCl (Cockcroft-Gault), eGFR
- Drug: dose history (mg, interval, infusion duration), trough levels
- Microbiology: organism, MIC if available
- Outcomes: clinical cure (yes/no), AKI per KDIGO criteria, ICU LOS, mortality
- Concomitant nephrotoxics: NSAIDs, piperacillin-tazobactam, contrast, aminoglycosides

### 3.4 Comparator (SOTA)

Bayesian forecasting using two-compartment population PK model (Roberts 2011 ICU vancomycin parameters), implemented in NONMEM or pmetrics. Posterior Cmin/AUC predictions with 95% credible intervals.

### 3.5 Sounio CDS pipeline

For each patient at each TDM time-point:
1. Lift inputs to `Knightian<f64>` (PBox), preserving assay/model bias bounds
2. Run `stdlib/clinical/vancomycin_pbpk.sio` PBPK forward simulation
3. Output: Cmin and AUC₂₄ p-boxes
4. Lean 4 safety theorem checked against output before dose recommendation
5. If theorem fails to close, refuse to recommend (BLOCKED state per Sounio refusal fixtures)

### 3.6 Outcomes

**Primary**: MAE of predicted Cmin vs measured trough, Sounio CDS vs Bayesian SOTA, paired t-test or Wilcoxon as appropriate.

**Secondary**:
- Coverage rate of Knightian bands (% of measurements within p-box)
- Clinical efficacy rate (cure, no relapse) within Sounio-recommended doses
- AKI incidence within Sounio-recommended doses
- Refusal rate (cases where Lean theorem failed) and clinical disposition of refused cases

### 3.7 Sample size justification

Primary: MAE non-inferiority margin Δ = 1.5 mg/L (clinically meaningful given target trough range 10–20 mg/L). Assuming SOTA MAE σ ≈ 3 mg/L (Roberts 2011), n = 100 gives 80% power at α = 0.05 (one-sided non-inferiority test, paired).

### 3.8 Analysis plan

Primary analysis intent-to-treat. Subgroup analyses by:
- Renal status (CrCl < 30, 30–60, > 60 mL/min)
- Time on vancomycin (< 72 h vs ≥ 72 h — Knightian bands expected wider in early treatment)
- Critical illness severity (SOFA tertiles)

## 4. Risks and Benefits

**Risk**: Minimal — retrospective, no patient contact, no intervention.
**Benefit (population-level)**: First validation of formally verified CDS for vancomycin under deep uncertainty. Potential foundational work for future prospective trial and regulatory submission.

## 5. Data Security

De-identified data only. HIPAA Safe Harbor identifiers removed. Data stored on institutional secure server, encrypted at rest. No cloud upload of identified data. Sounio CDS pipeline runs on institutional hardware; no PHI leaves the institution.

## 6. Funding and Conflicts of Interest

**Funding**: [TO FILL]
**Conflicts**: PI is creator of Sounio language. Mitigation: pre-registered analysis plan (M0 deliverable), independent statistician for primary outcome analysis, full code/data availability post-publication.

## 7. Pre-registration

Analysis plan pre-registered at OSF prior to data extraction (M3 milestone). Deviation from pre-registered plan documented in publication.

## 8. Publication

Primary publication target: *Clinical Pharmacokinetics* or *Journal of the American Medical Informatics Association* (JAMIA), 2027 cycle.

## 9. Timeline

| Milestone | Activity | Date |
|---|---|---|
| M0 | IRB submission | This session |
| M1-M2 | Sounio infrastructure (no patient data) | Months 1-2 |
| M3 | Pre-registered analysis plan finalized; IRB approval expected | Month 3 |
| M4 | Data extraction; cohort analysis | Month 4 |
| M5 | Manuscript drafting | Month 5 |
| M6 | Submission | Month 6 |

## 10. References (key)

- Rybak MJ et al. (2020) "Therapeutic monitoring of vancomycin..." *Am J Health-Syst Pharm* 77(11): 835-864. (ASHP 2020 Level 1A)
- Roberts JA et al. (2011) "Vancomycin dosing in critically ill patients." *Antimicrob Agents Chemother* 55(6).
- Ferson S et al. (2003) Sandia SAND2002-4015 (p-box methodology).
- Walley P (1991) *Statistical Reasoning with Imprecise Probabilities*.

## Status

**Draft for institutional review.** Cannot submit until institutional details + sponsor + biostatistician identified. Sounio infrastructure work (M1-M3) proceeds in parallel without data access.
