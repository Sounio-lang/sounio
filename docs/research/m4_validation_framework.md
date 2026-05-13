<!-- docs:meta
topic_id: repo.docs.research.m4-validation-framework
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.m4-validation-framework
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# M4 Validation Framework — Cohort Analysis Plan

**Status**: M4 deliverable. Pre-registered analysis plan.
**Date**: 2026-04-30 (locked at M4 start; deviations documented post-hoc.)
**PI**: Demetrios Chiuratto Agourakis

## Inputs

- **Real cohort**: institutional retrospective TDM cohort (lead: PI; awaiting IRB approval per `docs/research/irb_protocol_draft.md`).
- **Fallback cohort**: MIMIC-IV (PhysioNet, ~382k ICU stays, vancomycin TDM available). MIMIC-IV access requires CITI training + DUA; pre-extraction queries staged in `scripts/clinical/mimic_iv_extract.sql` (TBD).
- **Synthetic skeleton**: `scripts/clinical/data_synthetic/tdm_cohort_synthetic_v1.csv` (20 patients) — pipeline plumbing only; **no inferential analysis**.

## Primary outcome (Aim 1)

**Measure**: Mean Absolute Error (MAE) of predicted Cmin vs measured trough.

**Comparator**: Bayesian forecasting via population PK posterior update (NONMEM or pmetrics implementation of the Roberts 2011 popPK model).

**Hypothesis**: Sounio Knightian-CDS MAE is non-inferior to Bayesian SOTA, with non-inferiority margin Δ = 1.5 mg/L.

**Statistical test**: paired one-sided non-inferiority test (Wilcoxon if non-normal residuals, paired t-test if normal). α = 0.05.

**Sample size**: n = 100 (powered at 80% under σ = 3 mg/L per Roberts 2011).

## Secondary outcomes (Aim 2)

1. **Knightian coverage rate**. Proportion of measured Cmin values within the predicted Knightian p-box. Target: ≥ 95% under correctly-specified bands.
2. **Refusal rate**. Proportion of cases where the Lean safety theorem refused to close (BLOCKED). Stratified by:
   - pre-TDM (0 samples) vs post-TDM (≥ 3 samples)
   - SOFA tertile
   - CrCl strata
3. **Clinical safety correlation**. Among Sounio-recommended doses, compute observed AKI incidence and clinical cure rate. Compare to Bayesian-recommended doses.

## Tertiary analyses (Aim 3)

Width comparison: **Sounio Knightian band width** vs **Bayesian 95% CrI width**. Hypothesis: pre-TDM, Sounio bands are wider; post-TDM, the bands converge. This is the operational signature of Knightian conservativeness.

## Refusal handling

For every BLOCKED case (Lean theorem refused to close), document:

- the input that triggered refusal (which contract or which p-box-crossing-boundary)
- the disposition the clinician chose (TDM, dose adjustment, alternate antibiotic, ignored)
- 30-day outcome

This is a **scientifically novel measurement**: how often does a formally verified system *refuse* to recommend, and what do clinicians do when they get refusal?

## Pre-specified pivots

- If **MAE non-inferiority fails**: re-position the paper around Knightian conservativeness (refusal correlates with safety) rather than predictive accuracy.
- If **refusal rate > 50%** in pre-TDM: the band width parameters in `vp_band_*` are too conservative; tighten to ±15% / ±10% (CL / Vc) and re-run with pre-registered amendment.
- If **MIMIC-IV TDM data unusable** (sparse sampling, missing covariates): scope to a single-cohort report and flag external validity as a limitation.

## Software / data versioning

Each analysis run snapshots:

- Sounio commit hash (`git rev-parse HEAD`)
- Lean toolchain version (`lean --version`)
- Cohort CSV file SHA-256
- Per-patient prediction CSV (output of `process_tdm_cohort.sh`)

into `scripts/clinical/runs/<timestamp>/manifest.json`.

## Reproducibility

Post-publication, the following are released under a permissive licence:

- All Sounio code (`stdlib/epistemic/`, `stdlib/clinical/`)
- All Lean modules
- Pre-registered analysis plan (this document, frozen via OSF)
- Per-patient predictions (deidentified)

Raw cohort data is not released (institutional restrictions); MIMIC-IV is publicly available.

## Status of this document

**Pre-registered for the M4 milestone.** Locked when IRB approval lands or MIMIC-IV ETL completes (whichever is first). Deviations from this plan must be documented in the eventual publication.
