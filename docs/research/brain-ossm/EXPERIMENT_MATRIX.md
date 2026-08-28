<!-- docs:meta
topic_id: repo.docs.research.brain-ossm.experiment-matrix
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.brain-ossm.experiment-matrix
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Brain O-SSM Experiment Matrix

## Experimental slices

| Slice | Purpose | Data regime | Must have |
|---|---|---|---|
| Synthetic bridge | Prove order-sensitive signal can be exploited | controlled synthetic ROI-like series | O-SSM vs H-SSM vs diagonal |
| ABIDE MVP | Establish real-data baseline | ASD vs control, 3 seeds minimum | site-aware splits |
| Cross-site stress | Test transfer claim | grouped site split / leave-site-out | balanced accuracy + variance |
| Low-data regime | Test sample efficiency | 10%, 25%, 50%, 100% train fractions | fixed preprocessing |
| Missing-channel stress | Test robustness | dropped ROI channels or masked windows | same corruption across models |
| Mechanism slice | Link gain to non-associativity | associator-aware diagnostics | ablation and temporal traces |

## Model comparison table

| Model | Role | Parameter policy | Expected use |
|---|---|---|---|
| O-SSM | target model | reference budget | non-associative temporal composition |
| H-SSM | primary baseline | parameter-matched | associative hypercomplex control |
| Diagonal SSM | weak structure baseline | matched where possible | no coupling baseline |
| Optional stronger baseline | realism check | documented separately | modern temporal baseline if compute allows |

## Dataset pipeline

| Step | Output | Notes |
|---|---|---|
| ABIDE acquisition | subject list + metadata | freeze version and provenance |
| preprocessing import | harmonized subject tensor set | do not vary by model |
| parcellation | ROI time series | freeze atlas choice |
| split generation | train/val/test subject partitions | grouped by site where required |
| feature packaging | Sounio-readable serialized inputs | versioned and checksummed |

## Primary tasks

| Task | Labels | Priority | Why |
|---|---|---|---|
| ASD vs control | binary | highest | simplest clinically legible benchmark |
| ASD/ADHD/control | three-way | medium | richer but noisier decision boundary |
| auxiliary transition task | synthetic or derived | high | tests order-sensitive temporal structure directly |

## Metrics table

| Metric | Role | Required |
|---|---|---|
| Balanced accuracy | primary | yes |
| Macro F1 | class-balance support | yes |
| AUROC | threshold-independent check | yes |
| Calibration | practical trustworthiness | yes |
| Seed variance | stability | yes |
| Per-site breakdown | transfer honesty | yes |

## Mechanism analyses

| Analysis | Question answered |
|---|---|
| Associator norm over time | when does non-associativity activate? |
| Associator ablation | does the gain disappear when the mechanism is removed? |
| Transition-window slicing | does O-SSM help most on order-sensitive intervals? |
| Error-conditioned tracing | are high associator windows aligned with corrected mistakes? |

## Minimal compute ladder

| Phase | Seeds | Scope | Goal |
|---|---|---|---|
| smoke | 1 | tiny subset | pipeline correctness |
| MVP | 3 | binary task, site-aware split | first paper-legible signal |
| confirmation | 5 | same as MVP | stability |
| expansion | 5+ | low-data + robustness + subgroup | practical claim strengthening |

## Stop / continue rules

Continue to larger scale if:

- preprocessing is frozen
- 3-seed MVP runs cleanly
- at least one signal favors O-SSM or motivates mechanism analysis

Pause and reassess if:

- site-aware splits are not reproducible
- baselines are not parameter-matched
- O-SSM is unstable even before transfer stress
- diagnostics show no relationship between associator behavior and informative
  windows
