<!-- docs:meta
topic_id: repo.docs.research.cpc2026-orc-group-preregistration-2026-07-11
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cpc2026-orc-group-preregistration-2026-07-11
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# CPC 2026 ORC Group Contrast Preregistration

Frozen: 2026-07-11 UTC, before inspecting subject-level ORC values or any group-comparison p-value in the selected files.

## Eligible Dataset

- Repository: `/workspace/hyperbolic-semantic-networks`
- Primary input: `results/fmri/abide_orc_sinkhorn_t0.40.json`
- Population: every record with `dx_group` 1 (ASD) or 2 (control); no post-result exclusions.
- Unit: one CC200 resting-state functional-connectivity graph per ABIDE-I subject.
- Graph pipeline: CC200 time series, Pearson correlation, Fisher z transform, fixed threshold 0.40, largest connected component if needed, Ollivier-Ricci curvature with alpha 0.5 using the repository's Sinkhorn pipeline.

## Primary Analysis

- Sole endpoint: subject-level mean edge Ollivier-Ricci curvature (`kappa_mean`).
- Contrast: mean(ASD) minus mean(control); direction is determined by the observed sign.
- Test: 100,000 random label permutations, two-sided, seed 20260711.
- Monte Carlo permutation p-value: `(extreme + 1) / (100000 + 1)`.
- Effect: small-sample corrected Hedges g, oriented ASD minus control.
- Effect interval: 95% percentile interval from 20,000 within-group bootstrap resamples using the same seed stream.
- Alpha: 0.05. A null result remains null.

## Mandatory Variant

- Repeat the identical analysis for the already-existing threshold 0.50 file, `results/fmri/abide_orc_sinkhorn_t0.50.json`.
- This sensitivity result is reported regardless of direction or p-value and cannot replace the primary result.
- No other thresholds, alpha values, curvature summaries, subgroups, covariate models, or feature-selection steps will be computed for this poster decision.

## Power And Decision Rule

- Estimate two-sided two-sample design power at alpha 0.05 for a fixed benchmark effect |g| = 0.5 using the noncentral t approximation.
- Report the minimum detectable |g| for 80% power at the observed group sizes.
- Label the design underpowered for a medium effect when benchmark power is below 0.80.
- `PRINTABLE-PRELIMINARY` requires primary p below 0.05 and adequate design sensitivity; a primary p at or above 0.05 is `NULL`.

Power amendment, 2026-07-11 UTC after orthogonal review: the initial text proposed post-hoc power at the observed effect. That circular quantity was replaced by benchmark design power and MDE. The endpoint, groups, contrast, permutation test, effect estimate, confidence interval, seed, and decision remain unchanged.

## Interpretation Limits

- This is an unadjusted ASD-versus-control association, not a diagnostic classifier or causal effect.
- Site, motion, age, sex, medication, and threshold sensitivity are not resolved by the primary test.
- The ORC values were computed previously; this preregistration governs the group-separation test and poster decision, not the historical graph preprocessing.
