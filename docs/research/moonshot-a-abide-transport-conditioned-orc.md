<!-- docs:meta
topic_id: repo.docs.research.moonshot-a-abide-transport-conditioned-orc
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.moonshot-a-abide-transport-conditioned-orc
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

---
docs:meta:
  status: active
  owner: codex
  updated: 2026-05-25
status-note: "ABIDE 1034-subject scalar/f32 ORC joined to 168/ZD transport-prototype class shifts."
---

# Moonshot A ABIDE Transport-Conditioned ORC

## Evidence

This package is the first ABIDE x 168/ZD bridge after the f32-epistemic
full-cohort run. It binds:

- accepted scalar ABIDE ORC cohort: `1034` subjects
- accepted f32-epistemic ABIDE cohort: `1034` subjects
- transport-168 runtime prototype: `168` runtime slots
- selected transport classes: `8`

Gate:

```bash
SOUNIO_MOONSHOT_A_F32_COHORT_SUMMARY=/tmp/moonshot_a_abide_f32_full_summary.tsv \
SOUNIO_MOONSHOT_A_F32_COHORT_ARTIFACT=/tmp/moonshot_a_abide_f32_full_artifact.json \
bash scripts/ci/moonshot_a_abide_transport_conditioned_orc_gate.sh
```

Accepted artifact:

```text
/tmp/moonshot-a-abide-transport-conditioned-final/moonshot_a_abide_transport_conditioned_orc.v1.json
```

Accepted feature TSV:

```text
/tmp/moonshot-a-abide-transport-conditioned-final/moonshot_a_abide_transport_conditioned_orc.tsv
```

## Results

- status: `PASS_TRANSPORT_CONDITIONED_ORC_READY`
- joined subjects: `1034`
- selected 168/ZD runtime classes: `8`
- feature rows: `8272`
- label counts: `ASD=249`, `TD=250`, `UNKNOWN=535`
- scalar vs f32 mean-ORC correlation: `0.9999976563086985`
- scalar vs f32 mean-delta absolute mean: `0.000996001180064754`
- scalar vs f32 mean-delta absolute max: `0.0020095770759479548`
- transport-conditioned delta absolute max: `0.001085675688928034`
- transport-conditioned delta absolute mean: `0.0009623429927998473`
- transport-delta / scalar-f32 mean-absolute-drift ratio:
  `1.0900345407798109`
- transport-delta / scalar-f32 max-absolute-drift ratio:
  `0.5402508328355112`
- selected runtime class indices: `75`, `76`, `85`, `86`, `141`, `142`,
  `151`, `152`
- selected fiber labels: `10`, `10`, `10`, `10`, `11`, `11`, `11`, `11`

The gate requires the transport-conditioned delta to be nonzero and above the
scalar-vs-f32 mean absolute drift. It does not require it to exceed the worst
single-subject scalar-vs-f32 drift.

## Feature Columns

The TSV contains one row per subject and selected class:

- `subject`
- `label`
- `site`
- `class_168_probe_id`
- `fiber_label`
- `baseline_orc`
- `f32_epistemic_orc`
- `scalar_f32_delta`
- `transport_conditioned_orc_delta`
- `transport_conditioned_orc`
- `residual_vs_scalar_f32_delta`
- `variance_bound`
- `claim_scope`

## Boundaries

- This is a descriptive transport-conditioned feature package, not a
  confirmatory biomarker claim.
- Cohort scope is the accepted 1034-subject scalar baseline, not the raw
  1035-file directory.
- The 168/ZD input is the runtime transport prototype, not final ZD surgery
  semantics and not the Lean census itself.
- The transport delta is a global class-probe feature, not a subject-specific
  CUDA rerun of every ABIDE graph under every 168/ZD modulation.
- f32-epistemic variance remains a finite overflow-bound sensitivity
  diagnostic, not calibrated coverage.
- This does not claim clinical utility, diagnosis, or external validation.
