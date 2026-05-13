<!-- docs:meta
topic_id: repo.docs.research.brain-ossm.readme
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.brain-ossm.readme
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Brain O-SSM ABIDE Package

This directory is the execution-facing validation package for turning the
current O-SSM non-associative benchmark line into a paper-grade real-data study.

It exists to answer one question cleanly:

> What would count as strong evidence that non-associative state composition is
> useful on real temporal neurodynamics rather than only on synthetic algebraic
> probes?

That question now has an empirical answer on ABIDE. The current verified
full-train winner is documented in:

- `RESULTS_2026-04-11.md`

Use these files in order:

1. `VALIDATION_PLAN.md`
2. `ACCEPTANCE_CRITERIA.md`
3. `EXPERIMENT_MATRIX.md`
4. `../../papers/temporality-psychiatry/abide-abstract.md`

Execution surfaces now linked to this package:

- benchmark:
  - `examples/brain_ossm_abide.sio`
- temporal manifest builder:
  - `scripts/research/build_abide_temporal_manifest.py`
- structured parse:
  - `scripts/research/parse_brain_ossm_abide_output.py`
- external baseline suite:
  - `scripts/research/abide_external_baselines.py`
- campaign aggregation:
  - `scripts/research/aggregate_brain_ossm_campaign.py`
- cluster wrappers:
  - `slurm-jobs/brain-ossm/submit-abide-gpu.sh`
  - `slurm-jobs/brain-ossm/submit-abide-external-baselines-gpu.sh`
  - `slurm-jobs/brain-ossm/submit-abide-campaign-gpu.sh`

Current stance:

- the synthetic and controlled benchmark line is already scientifically useful
- ABIDE-scale temporal validation now produced a verified O-SSM winner
- the next decisive step is consolidation, write-up, and promotion of the
  winning branch rather than proving basic trainability
- the infrastructure now supports both the legacy flat `8x8` ABIDE manifest and
  richer temporal flat manifests with variable sequence length
- the temporal line now has two concrete recipes:
  - `v3`: contiguous ROI-group window means
  - `v4`: global-PCA window means
- `v4` is materially better than `v3`, but the best current smoke still comes
  from the legacy `8x8` manifest
- the primary remaining question is no longer whether O-SSM works, but which
  algebra/projection family gives the strongest full-train headline
