<!-- docs:meta
topic_id: repo.docs.research.neurodyn-next-pivot-2026-07-04
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.neurodyn-next-pivot-2026-07-04
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# NeuroDyn next pivot after ABIDE O-SSM negative run

Date: 2026-07-04 UTC

## Current Evidence

The ABIDE O-SSM checkpoint-persistence campaign completed and produced a verified trained checkpoint, but the aggregate result was near chance:

- Run ID: `brain-ossm-abide-oassoc-stable-identity-ckpt-r740-20260704T214534Z`
- O-SSM aggregate balanced accuracy: `50.18 +/- 1.104355`
- H-SSM aggregate balanced accuracy: `49.94 +/- 1.335814`
- Best selected fold checkpoint replay: `76.923076 -> 76.923076`, delta `0.000000pp`
- Trained ROI associator rows: `32000`
- Holm-significant ROI biomarkers: `0`

Receipt:

`artifacts/research/abide/brain-ossm-abide-oassoc-stable-identity-ckpt-r740-20260704T214534Z/README.md`

## What This Means

This is a useful engineering and assay result, not a positive clinical result.

The useful part:

- Checkpoint persistence now has a hard receipt.
- The associator extraction can be forced to use trained native hidden states.
- The gate correctly rejects model-free extraction and replay mismatches.
- Native replay is sensitive enough to expose source/compiler drift.

The negative part:

- ABIDE, with this manifest and `o_assoc_stable + identity`, does not support a positive O-SSM ASD classification claim.
- The trained ROI associator does not yield Holm-surviving ROI effects.
- The selected best fold is not valid as an aggregate-performance claim.

## Immediate Blockers

1. OrangeFS capacity

   `/orangefs/training` is full: `983G/983G`, `0` available. This blocked copying the trained associator directory back to OrangeFS.

2. Compiler/source reproducibility

   The verifier that passed in the campaign was tied to an exact generated source/ELF. Rebuilding equivalent native extractors with nearby source/compiler combinations can change replay from `76.923076` to `50.000000` or `46.153846`. The extractor was completed only after reusing the verified forward-pass prefix and the older Madaros binary available on the r740 worker.

3. Scientific signal

   ABIDE aggregate performance is near chance. The next positive claim cannot be ABIDE ASD classification from this run.

## Recommended Pivot

Prioritize a real utility question where O-SSM is an assay/observable, not a classifier leaderboard:

1. MDD `ds002748` mechanistic assay lane

   Use O-SSM as a trained dynamical perturbation/associator assay over fMRIPrep-derived ROI sequences. The claim target should be methodological: whether trained non-associative state perturbations expose reproducible condition-sensitive dynamics under strict negative controls.

2. ADHD200 control lane

   Use ADHD200 as a stress test for site leakage and label-shuffle controls. Do not make ADHD clinical claims until site-balanced and temporal-shuffle controls pass.

3. ORC / non-O-SSM fallback

   If O-SSM continues to stay near chance, keep the WPA abstract ORC-only or NeuroDyn-methods-only. Do not include positive O-SSM claims unless a replicated, controlled, non-leaky result exists.

## Next Concrete Run

Run a small MDD pilot with these acceptance gates:

- fMRIPrep pilot artifact exists for a minimal subject set.
- Manifest quality gate passes.
- O-SSM checkpoint persists.
- Fresh-process replay delta is `<= 0.5pp`.
- Label-shuffle or temporal-shuffle negative control is run before any positive wording.
- ROI associator extraction is only allowed after trained replay passes.
- Promotion requires replicated nonzero effect after Holm/FDR or a pre-registered descriptive assay with no biomarker claim.

## Claim Boundary

Allowed now:

- "We built a checkpoint-persistent O-SSM/associator assay pipeline and found ABIDE near chance under this configuration."
- "The trained associator extraction is now guarded against model-free false positives."

Not allowed now:

- "O-SSM detects ASD on ABIDE."
- "O-SSM has an ABIDE biomarker."
- "The selected 76.9% fold is the model's campaign performance."
- "The associator ROI table is clinically meaningful."

