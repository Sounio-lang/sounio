<!-- docs:meta
topic_id: repo.docs.handoff.neurodyn-adhd200-pcp-pilot24-2026-07-07
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.neurodyn-adhd200-pcp-pilot24-2026-07-07
-->

# ADHD-200 PCP Pilot-24 Execution Report

Date: 2026-07-07

Decision: `UNDERCONTROLLED_LOW_POWER`

Claim boundary: this is a computational-psychiatry framework and data-path
pilot only. It is not a diagnostic, biomarker, treatment-response,
biological-mechanism, clinical-decision, or O-SSM superiority claim.

## Why This Exists

The active objective is to build "a controlled framework for testing
path-dependent neural dynamics in computational psychiatry." The first
psychiatric lane therefore paused Algebra-C priority and asked for:

1. a psychiatric preregistration;
2. one primary dataset and one dimensional phenotype family;
3. baselines and controls;
4. O-SSM dynamic feature export rather than classification only;
5. a small site-aware pilot;
6. a decision about whether a clinical story exists only after the pilot.

This report records the first real-data, small, site-aware ADHD-200 pipeline
execution. Its result is underpowered under the current pilot gate.

## Data Source

Primary source: public FCP-INDI / Preprocessed Connectomes Project ADHD-200
S3 bucket.

Source anchors:

- `https://registry.opendata.aws/fcp-indi/`
- `https://fcon_1000.projects.nitrc.org/indi/s3/index.html`
- `https://preprocessed-connectomes-project.org/adhd200/`
- `https://raw.githubusercontent.com/preprocessed-connectomes-project/adhd200/gh-pages/download.html`

The workspace could list and fetch from the public S3 bucket:
`s3://fcp-indi/data/Projects/ADHD200/`.

## Executed Cache Bootstrap

Command:

```bash
python3 scripts/research/adhd200_s3_bootstrap.py \
  --output-dir /tmp/adhd200_pcp_pilot24_bootstrap \
  --max-subjects 24 \
  --overwrite
```

Result:

- status: `ready`;
- selected subjects: 24;
- available C-PAC subjects in selected pipeline: 162;
- downloaded CC200 ROI files: 24;
- sites: `KKI=16`, `NYU=8`;
- labels: `ADHD=12`, `TD=12`;
- source derivative: C-PAC benchmark with frequency filter, CC200 ROI
  timeseries, no global signal regression.

Primary files:

- `/tmp/adhd200_pcp_pilot24_bootstrap/adhd200_phenotypic.csv`;
- `/tmp/adhd200_pcp_pilot24_bootstrap/rois/`;
- `/tmp/adhd200_pcp_pilot24_bootstrap/adhd200_s3_bootstrap_summary.json`.

## Access Audit And Readiness

Access audit command:

```bash
python3 scripts/research/adhd200_data_access_audit.py \
  --phenotypic-csv /tmp/adhd200_pcp_pilot24_bootstrap/adhd200_phenotypic.csv \
  --roi-dir /tmp/adhd200_pcp_pilot24_bootstrap/rois \
  --output-dir /tmp/adhd200_pcp_pilot24_bootstrap/access_audit \
  --overwrite
```

Result: `ready`.

Manifest readiness result from the pilot runner:

- status: `pass`;
- rows: 24;
- sites: 2;
- labels: `ADHD=12`, `TD=12`;
- primary phenotypes present: `inattention`,
  `hyperactivity_impulsivity`, `adhd_total`;
- feature layout: `8x8_temporal_roi_block`;
- finite feature values: 1536 / 1536;
- feature nonzero fraction: 1.0.

## Pilot Command

Command:

```bash
PHENOTYPIC_CSV=/tmp/adhd200_pcp_pilot24_bootstrap/adhd200_phenotypic.csv \
ROI_DIR=/tmp/adhd200_pcp_pilot24_bootstrap/rois \
OUTPUT_DIR=/tmp/adhd200_pcp_pilot24_bootstrap/pilot_real24 \
NULL_PERMUTATIONS=1 \
GENERIC_BASELINE_MODELS=gru_reservoir,trained_rnn \
GENERIC_BASELINE_SEEDS=55555,11111 \
GENERIC_TRAINED_EPOCHS=4 \
GLOBAL_TRAIN_EPOCHS=2 \
OCT_TRAIN_EPOCHS=2 \
H_TRAIN_EPOCHS=2 \
SOUNIO_SOUC_ENGINE=lean_single \
scripts/research/adhd200_dimensional_pilot_smoke.sh
```

Result: exit code 0.

Primary artifacts:

- `/tmp/adhd200_pcp_pilot24_bootstrap/pilot_real24/PILOT_SUMMARY.md`;
- `/tmp/adhd200_pcp_pilot24_bootstrap/pilot_real24/SHA256SUMS`;
- `/tmp/adhd200_pcp_pilot24_bootstrap/pilot_real24/manifest/readiness_gate.json`;
- `/tmp/adhd200_pcp_pilot24_bootstrap/pilot_real24/ossm_run/brain_ossm_abide.raw.txt`;
- `/tmp/adhd200_pcp_pilot24_bootstrap/pilot_real24/dimensional_probe/adhd_dimensional_dynamic_features.tsv`;
- `/tmp/adhd200_pcp_pilot24_bootstrap/pilot_real24/generic_recurrent_baseline/adhd_generic_recurrent_baseline_summary.tsv`;
- `/tmp/adhd200_pcp_pilot24_bootstrap/pilot_real24/pilot_decision/adhd_dimensional_pilot_decision.md`.

Dynamic feature export:

- `STATE_TRACE` rows: 960;
- dynamic feature TSV lines: 961;
- trained generic prediction TSV lines: 151.

## Decision

Overall verdict: `UNDERCONTROLLED_LOW_POWER`.

Low-power reasons:

- `row_count 24 < min_decision_subjects 50`;
- `min null_permutations_mean 1 < min_decision_null_permutations 20`.

Per phenotype:

| phenotype | verdict | pilot metric verdict | O-SSM hidden Spearman | best control |
|---|---|---|---:|---|
| inattention | `UNDERCONTROLLED_LOW_POWER` | `NEGATIVE_GENERIC_RECURRENT_COMPETITIVE` | -0.049117 | H-SSM hidden 0.169187 |
| hyperactivity_impulsivity | `UNDERCONTROLLED_LOW_POWER` | `NEGATIVE_GENERIC_RECURRENT_COMPETITIVE` | -0.062728 | H-SSM hidden 0.141717 |
| adhd_total | `UNDERCONTROLLED_LOW_POWER` | `NEGATIVE_GENERIC_RECURRENT_COMPETITIVE` | -0.032160 | H-SSM hidden 0.156969 |

Interpretation: under this small real-data pilot, O-SSM did not expose an
interpretable dimensional ADHD lead beyond controls, but the run is too small
and uses too few null permutations to support a negative scientific claim. The
honest follow-up is not a clinical story; it is a larger controlled
replication attempt or a clearly labeled underpowered smoke result.

## Limits

- `n=24` is a smoke-scale pilot, not a claim-scale cohort.
- Only two sites were present in the pilot cache.
- Null permutations were intentionally small (`NULL_PERMUTATIONS=1`) to keep
  the workspace run lightweight.
- Generic controls included a reservoir and a small trained NumPy RNN, not a
  full GRU/S4/Transformer suite.
- `lean_single` was used because the Madaros f64 ABI blocker remains a
  compiler-lane issue.
- The old ABIDE-named runner remains the execution surface, but the semantic
  source of truth is the ADHD-200 rich manifest.

## Next Action

Do not promote a clinical claim. The next useful action is a larger ADHD-200
controlled run with:

- more sites and subjects from the same PCP/S3 source;
- higher null count;
- trained generic recurrent controls with more seeds;
- leakage and nuisance-control review;
- external offload review after the scaled artifact is produced.
