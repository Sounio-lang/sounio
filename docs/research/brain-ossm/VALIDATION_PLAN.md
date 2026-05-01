<!-- docs:meta
topic_id: repo.docs.research.brain-ossm.validation-plan
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.brain-ossm.validation-plan
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Brain O-SSM ABIDE Validation Plan

## Core claim

The practical thesis is not that octonions are mathematically exotic.
The practical thesis is that non-associative state composition can represent
temporal cascades whose meaning depends on grouping order, and that this matters
for real neurodynamic sequences.

For this project, the strongest real-data test is ABIDE-scale fMRI-derived
regional time series with cross-site evaluation.

## Main hypothesis

An O-SSM temporal model operating on ROI time series derived from ABIDE will
outperform parameter-matched associative baselines in at least one practically
meaningful regime:

- cross-site generalization
- low-data regime / sample efficiency
- calibration / uncertainty quality
- robustness to missing or noisy ROI channels
- transient order-sensitive segments

The intended interpretation is:

- O-SSM wins where order-dependent composition carries signal
- H-SSM and diagonal baselines should remain competitive where order dependence
  is not informative

## Null hypothesis

Non-associative composition does not provide measurable practical benefit on
real temporal neurodynamics relative to parameter-matched associative baselines.

## Why ABIDE is the right next dataset

ABIDE is not useful merely because it is larger than the toy benchmarks.
It is useful because it introduces the exact ingredients needed for a stronger
claim:

- real temporal sequences rather than synthetic supervision only
- multi-site heterogeneity
- clinically meaningful label structure
- opportunity for subgroup and transfer analysis
- plausible directed cascade phenomena across distributed brain systems

## Experimental ladder

The program should advance in three explicit stages.

### Stage 1: Synthetic-to-real bridge

Build a controlled task from ROI-like time series where the ground truth depends
on ordered composition over time.

Purpose:

- prove that O-SSM can exploit order-sensitive signal when the signal is known
  to exist
- avoid the jump from algebraic toy tasks directly into messy clinical data

### Stage 2: ABIDE MVP

Run the smallest paper-legible real benchmark:

- input: preprocessed fMRI -> parcellated ROI time series
- task: ASD vs control
- splits:
  - within-site
  - grouped site split / leave-site-out style split
- seeds: start with 3, then scale to 5+
- models:
  - O-SSM
  - H-SSM parameter-matched
  - diagonal / naive baseline

Deliverable:

- one clean result table and one clean generalization figure

### Stage 3: Mechanism validation

Show that any gain is actually tied to non-associative dynamics.

Required analyses:

- associator norm over time
- ablation of associator-dependent path
- comparison against H-SSM with matched parameter budget
- site-wise breakdowns
- error slices for difficult segments

## Primary success modes

The study is strong if any one of these is true and replicable:

1. O-SSM improves balanced accuracy by a meaningful margin on grouped site
   splits.
2. O-SSM matches headline accuracy but has lower cross-seed variance or better
   calibration.
3. O-SSM is clearly better in low-data or partial-observation regimes.
4. O-SSM wins specifically on transition-heavy or order-sensitive windows, with
   mechanistic evidence from associator-aware diagnostics.

## Primary failure modes

These outcomes would weaken the central practical claim:

1. H-SSM or diagonal baselines dominate across all real-data settings.
2. O-SSM gains are inconsistent and disappear under site-aware splits.
3. Any gain vanishes after parameter matching or regularization matching.
4. Associator diagnostics fail to align with informative temporal segments.

## What makes the paper strong even if the headline claim softens

This program can still produce a strong paper if the result is conditional
rather than triumphant.

Examples:

- O-SSM helps only under distribution shift
- O-SSM helps only on selected temporal motifs
- O-SSM is not more accurate, but is better calibrated or more stable
- O-SSM does not help on ABIDE, but the mechanism study shows why and narrows
  the domain of validity

That would support a more careful conclusion:

> non-associativity is expressive and mechanistically meaningful, but its
> practical benefit depends on the temporal regime and task structure

## Minimum viable paper artifact

Before scaling, the project should be able to produce this exact package:

- one frozen ABIDE preprocessing path
- one frozen model comparison table
- one cross-site figure
- one mechanism figure
- one explicit criteria document for interpreting the outcome

If that package exists, the line is paper-ready even before full cluster scale.
