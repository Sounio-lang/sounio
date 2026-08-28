<!-- docs:meta
topic_id: repo.docs.research.brain-ossm.acceptance-criteria
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.brain-ossm.acceptance-criteria
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Brain O-SSM Acceptance Criteria

This file defines what would count as a meaningful win, a conditional result,
or a negative result for the ABIDE validation program.

## Pre-registered interpretation rule

Do not decide after the run what success means.
Judge the study against these buckets.

## Strong positive result

Classify the result as strong if all of the following are true:

1. O-SSM beats the parameter-matched H-SSM by at least `2.0` percentage points
   in balanced accuracy on at least one site-aware evaluation setting.
2. The sign of the gain is stable across at least `5` seeds.
3. The gain is not explained away by a larger parameter budget, looser
   regularization, or data leakage.
4. At least one mechanism-side diagnostic supports the interpretation that the
   gain comes from order-sensitive composition rather than incidental training
   effects.

## Conditional positive result

Classify the result as conditional if any of the following are true:

1. O-SSM does not win headline balanced accuracy, but wins on calibration,
   variance, or robustness under missing/noisy ROI channels.
2. O-SSM wins only on cross-site transfer, low-data, or subgroup regimes.
3. O-SSM ties H-SSM overall but shows a clean and interpretable gain on
   transition-heavy segments or order-sensitive auxiliary tasks.

This is still publishable if the paper is framed honestly around regime
dependence.

## Negative result

Classify the result as negative if all of the following are true:

1. O-SSM fails to beat H-SSM in any practically relevant metric or regime.
2. O-SSM gains are unstable across seeds or disappear on site-aware splits.
3. Mechanism diagnostics do not show a plausible association between
   non-associative activity and informative temporal structure.

This does not invalidate the whole research line.
It narrows the claim and strengthens the benchmark story.

## Mandatory metrics

Every ABIDE result bundle should report at least:

- balanced accuracy
- macro F1
- AUROC
- calibration metric such as ECE or Brier-like score
- mean and standard deviation across seeds
- per-site or grouped-site breakdown

## Mandatory baselines

Do not call the result strong unless all of these are present:

- O-SSM
- H-SSM parameter-matched
- diagonal / naive baseline

If compute allows, add one stronger non-hypercomplex temporal baseline, but do
not weaken the core matched comparison.

## Mandatory robustness checks

Before claiming a real win, verify:

- site-aware split integrity
- no leakage from preprocessing or subject overlap
- seed stability
- parameter count comparability
- identical input pipeline across models

## Mechanism evidence required for the strongest claim

To claim that non-associativity itself matters, not just the architecture label,
include at least two of:

- associator norm trajectory analysis
- ablation with associator-dependent path removed or frozen
- transition-window analysis where O-SSM wins disproportionately
- correlation between associator activity and confidence or error reduction

## Figure threshold

The work is not yet "killer application" level until there is at least one
figure that a reviewer can understand in under thirty seconds:

- x-axis: site-aware regime, sample size, or corruption level
- y-axis: balanced accuracy or equivalent primary metric
- O-SSM visibly above H-SSM in a reproducible setting

If that figure does not exist, the claim remains promising but unproven.
