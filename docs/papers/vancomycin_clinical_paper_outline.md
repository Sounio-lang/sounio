<!-- docs:meta
topic_id: repo.docs.papers.vancomycin-clinical-paper-outline
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.vancomycin-clinical-paper-outline
-->

# Clinical Paper Draft Outline — *Clinical Pharmacokinetics* / JAMIA target

**Working title**: *Formally Verified Vancomycin Dosing under Knightian Uncertainty: A Retrospective Cohort Study*

**Target venue**: *Clinical Pharmacokinetics* (Springer, IF ~5) or *JAMIA* (Oxford, IF ~6)
**Target page count**: 4500-6000 words
**Authors**: D. C. Agourakis (PI), institutional pharmacist co-author (TBD), ICU physician co-author (TBD), independent biostatistician (per IRB protocol).
**Status**: M5 outline. Filled prose deferred to M4 cohort completion.

## Structured abstract (250 words; CP format)

**Background**: Therapeutic drug monitoring (TDM) for vancomycin balances efficacy (AUC₂₄/MIC ≥ 400 mg·h/L) against acute kidney injury (AKI) risk. Current decision support tools (BestDose, InsightRX) handle Bayesian uncertainty but not Knightian uncertainty — the doubt about the underlying probability distribution itself, arising from assay bias, model misspecification, and inter-individual variability beyond what the population PK can quantify.

**Objective**: Validate a Sounio-based, formally verified clinical decision support (CDS) tool that propagates Knightian uncertainty through a 2-compartment population PK model, and compare predictive accuracy and clinical outcomes against current Bayesian SOTA.

**Methods**: Retrospective cohort of 100 ICU adults receiving intravenous vancomycin with ≥ 2 trough measurements. Primary outcome: MAE of predicted Cmin vs measured trough, Sounio CDS vs Bayesian forecasting (Roberts 2011 popPK). Secondary outcomes: Knightian band coverage, refusal rate, observed AKI / cure correlations.

**Results**: [TO COMPLETE post-cohort.]

**Conclusions**: [TO COMPLETE.] Provisional thesis: Knightian-conservative refusals correlate with subsequent AKI events at non-inferior rates; the Sounio CDS provides a pre-TDM "do not prescribe yet" signal that conventional Bayesian point estimates miss.

## 1. Introduction

- Vancomycin TDM landscape (Rybak 2020).
- AKI epidemiology (7-19% incidence in ICU).
- The gap: Bayesian forecasting handles parameter uncertainty but not distributional uncertainty.
- The Sounio approach (concise: types, effects, formal verification — keep PL detail for the supplement).
- Specific aims (mirror IRB Aims 1-3).

## 2. Methods

### 2.1 Study design and population
Retrospective cohort, single-center. Inclusion / exclusion per `irb_protocol_draft.md`.

### 2.2 Data extraction
Demographics, renal function, dosing, troughs, microbiology, outcomes, concomitant nephrotoxics. EHR query language: [TBD per institution].

### 2.3 Sounio CDS pipeline
Brief: Knightian PBox over Vc, CL inputs (band width parametric on TDM samples observed); 2-compartment forward simulation; Lean-verified safety gate. Detailed PL machinery in supplement and `vancomycin_pl_paper_outline.md`.

**Joint-dependence treatment of (Vc, CL).** Vancomycin Vc and CL are correlated in population PK literature (typical r ≈ 0.3–0.7, varying with renal/obesity/critical-illness state). The Sounio CDS does NOT assume independence and does NOT fix a copula. Instead, it relies on the Fréchet (point-mass) outer enclosure for monotone-in-each-arg functions: because Cmin is strictly increasing in Vc and strictly decreasing in CL on the physiological domain, the corner-enumeration band `[Cmin(Vc_lo, CL_hi), Cmin(Vc_hi, CL_lo)]` contains the true Cmin almost surely for *any* joint distribution of (Vc, CL) supported on the marginal Knightian rectangle. The enclosure is sharp under the counter-monotonic copula and conservative (factor ≈ 2.6× the true 95%-CI width) under co-monotonic `r ≈ +0.7`. Theorem and Lean discharge: `formal/lean4/SounioFrechet.lean`. Empirical verification across 5 correlation values (n = 250 samples): `tests/stdlib/clinical/test_vancomycin_correlation_sensitivity.sio`. Mathematical refs: Moore (1966), Williamson & Downs (1990), Ferson et al. (2003).

### 2.4 Bayesian SOTA comparator
NONMEM or pmetrics implementation of Roberts 2011 popPK with Bayesian forecasting. Posterior mean Cmin / AUC₂₄ with 95% credible intervals.

### 2.5 Primary analysis
Paired one-sided non-inferiority test of MAE; Δ = 1.5 mg/L; α = 0.05; n = 100 powered at 80%.

### 2.6 Secondary analyses
Knightian coverage, refusal rate, AKI / cure stratification.

### 2.7 Pre-registration
OSF link [to add at submission].

## 3. Results

[TO COMPLETE.]

Subsections planned:
- 3.1 Cohort description (Table 1)
- 3.2 Primary outcome — MAE comparison (Figure 1: Bland-Altman; Table 2)
- 3.3 Knightian coverage rate (Figure 2: coverage vs band width)
- 3.4 Refusal rate stratified by TDM samples (Figure 3)
- 3.5 Clinical correlates: refusal → AKI / cure
- 3.6 Subgroup: pre-TDM vs post-TDM cohort split

## 4. Discussion

- Headline finding (TBD; provisional: Knightian non-inferior to Bayesian on MAE; superior on refusal-AKI correlation).
- Why Knightian matters clinically: pre-TDM, you can't trust the point estimate; the band width quantifies what TDM is buying you.
- Comparison with prior CDS validation: most BestDose / InsightRX validations report MAE without uncertainty granularity.
- Limitations: single-center, retrospective, vancomycin-specific (substrate generalises but generalisation is unproven).
- Future work: prospective trial; broader antibiotic panel.

## 5. Conclusions

[TO COMPLETE.]

## Tables and figures (planned)

- T1: Cohort demographics
- T2: Primary outcome MAE comparison + secondary outcomes
- T3: Refusal-event correlations
- F1: Bland-Altman of predictions vs measured
- F2: Coverage rate as function of TDM samples
- F3: Refusal rate by clinical strata
- F4: Cmin Knightian band evolution per patient (longitudinal)

## Supplement

- S1: Sounio PL machinery (one-paragraph summary; references PL paper).
- S2: Sounio code listings (`stdlib/clinical/vancomycin_pbpk.sio`, full).
- S3: Lean theorem statements (one per safety obligation).
- S4: Cohort extraction queries.
- S5: NONMEM control file for SOTA comparator.

## Authorship and contributions (CRediT)

- Conceptualization: PI
- Methodology: PI + co-authors
- Software: PI (Sounio); independent statistician for analysis pipeline.
- Investigation: PI + clinical co-author.
- Formal analysis: independent statistician (primary outcome).
- Writing — original draft: PI.
- Writing — review and editing: all authors.

## Conflict of interest

PI is the creator of Sounio. Mitigation per IRB protocol § 6: pre-registered plan, independent statistician, full code/data release post-publication.

## Reviewer-pre-empt notes (internal)

- "Why not just MC?" — Knightian / p-box subsumes MC under choice of distribution; MC alone cannot represent distributional uncertainty.
- "Why is this not a Bayesian model with hyper-priors?" — Klibanoff-style smooth ambiguity is one alternative; we discuss in `knightian_operator_choice.md` why p-box was chosen.
- "How do you justify the band widths (±15%/±10%)?" — pre-TDM bands match Roberts 2011 inter-individual CV; post-TDM bands match the typical posterior CV after 2-3 observations. Sensitivity analysis included in supplement.
- "You are assuming Vc and CL are independent" — explicitly NO; § 2.3 details the Fréchet outer enclosure that holds copula-free for monotone-in-each-arg PBPK functions. Lean theorem + 250-sample empirical verification across r ∈ [-0.7, 0.7]. The enclosure trades tightness for soundness; the conservatism factor is disclosed.
