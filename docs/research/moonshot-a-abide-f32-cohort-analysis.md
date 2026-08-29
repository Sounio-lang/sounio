<!-- docs:meta
topic_id: repo.docs.research.moonshot-a-abide-f32-cohort-analysis
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.moonshot-a-abide-f32-cohort-analysis
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
status-note: "Downstream descriptive analysis for Moonshot A ABIDE f32-epistemic full-cohort evidence."
---

# Moonshot A ABIDE f32-Epistemic Cohort Analysis

## Evidence

This analysis binds the accepted scalar ABIDE ORC cohort to the accepted
f32-epistemic full-cohort Slurm artifact.

Scalar baseline:

- `artifacts/research/abide_orc/cohort_summary.tsv`
- subjects: `1034`
- excludes malformed raw input `UM_1_0050284`

f32-epistemic full cohort:

- run directory:
  `/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-abide-epi-full-cohort-run-20260525T010745`
- artifact:
  `/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-abide-epi-full-cohort-run-20260525T010745/moonshot_a_abide_epistemic_cohort_slurm.v1.json`
- summary:
  `/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-abide-epi-full-cohort-run-20260525T010745/abide_f32_epistemic_cohort_summary.tsv`
- status: `pass`
- limit: `0`
- subjects reported: `1034`
- expected subjects: `1034`
- ORC `.npy` artifacts: `1034`
- f32-epistemic diagnostic TSV artifacts: `1034`

Analysis gate:

```bash
SOUNIO_MOONSHOT_A_F32_COHORT_SUMMARY=/tmp/moonshot_a_abide_f32_full_summary.tsv \
SOUNIO_MOONSHOT_A_F32_COHORT_ARTIFACT=/tmp/moonshot_a_abide_f32_full_artifact.json \
bash scripts/ci/moonshot_a_abide_f32_cohort_analysis_gate.sh
```

Claim discipline gate:

```bash
SOUNIO_MOONSHOT_A_ABIDE_ANALYSIS_JSON=/tmp/moonshot-a-abide-f32-analysis.wp2mid/moonshot_a_abide_f32_cohort_analysis.v1.json \
SOUNIO_MOONSHOT_A_F32_COHORT_ARTIFACT=/tmp/moonshot_a_abide_f32_full_artifact.json \
bash scripts/ci/moonshot_a_abide_claim_discipline_gate.sh
```

The claim gate reports `PASS_CLAIM_DISCIPLINE` and checks:

- f32 full-cohort artifact status and `limit=0`
- `subjects_reported == expected_subjects == 1034`
- downstream analysis status and clean variance diagnostics
- required scope phrases across the Moonshot A docs
- absence of broad raw-1035, confirmatory-biomarker, clinical, diagnostic, or
  calibrated-coverage claims

Accepted analysis artifact:

```text
/tmp/moonshot-a-abide-f32-analysis.4XuXSX/moonshot_a_abide_f32_cohort_analysis.v1.json
```

## Results

- status: `PASS_COHORT_ANALYSIS_READY`
- joined subjects: `1034`
- scalar-only subjects: `0`
- f32-only subjects: `0`
- label counts: `ASD=249`, `TD=250`, `UNKNOWN=535`
- scalar vs f32 mean-ORC correlation: `0.9999976563086985`
- mean ORC delta mean: `0.000996001180064754`
- mean ORC delta absolute max: `0.0020095770759479548`
- std ORC delta mean: `-0.0001076741630807397`
- std ORC delta absolute max: `0.0006806028466551928`
- scalar mean-ORC ASD/TD Cohen's d: `0.01124003105667986`
- f32 mean-ORC ASD/TD Cohen's d: `0.011385455451838606`

Variance diagnostics:

- `inf_count=0`
- `nan_count=0`
- `negative_count=0`
- `bad_first_index_count=0`
- `max_output_var=3.40282e+38`

## Boundaries

- This is descriptive alignment analysis, not a confirmatory biomarker claim.
- The f32-epistemic variance output is a sensitivity diagnostic / finite
  overflow-bound lane, not calibrated coverage.
- Cohort scope is the accepted 1034-subject scalar baseline, not the raw
  1035-file directory.
- This does not claim clinical utility, diagnosis, or external validation.
