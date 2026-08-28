<!-- docs:meta
topic_id: repo.docs.research.brain-ossm.abide-runbook
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.brain-ossm.abide-runbook
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Brain O-SSM ABIDE Runbook

This runbook defines the shortest honest path from the current synthetic benchmark suite to an ABIDE-scale temporal validation.

## Current State

- The Slurm lane is working for Sounio benchmark jobs.
- The synthetic benchmark suite is running on cluster GPU nodes.
- ABIDE training source now exists at `examples/brain_ossm_abide.sio`.
- The benchmark expects a frozen TSV manifest at `/orangefs/training/sounio/abide-data/abide_roi_manifest.tsv`.
- The shared Slurm-visible ABIDE path is `/orangefs/training/sounio/abide-data`.
- The Sounio benchmark now emits structured prediction traces (`PRED<TAB>...`) so post-processing can derive AUROC, Brier, and ECE.
- The external deep baseline suite now exists at `scripts/research/abide_external_baselines.py`.
- The unified aggregation step now exists at `scripts/research/aggregate_brain_ossm_campaign.py`.

## Goal

Run an ABIDE cross-site benchmark where:

- input is a frozen per-subject ROI-feature manifest derived from preprocessed fMRI
- task is at least ASD vs control
- split is leave-one-site-out by acquisition site
- result bundle reports multi-seed mean, spread, per-site holdout summaries, and machine-readable leaderboard artifacts

## Required Preconditions

1. Dataset present in OrangeFS at a stable path.
2. Frozen atlas choice, default `CC200`.
3. Frozen subject manifest and metadata snapshot.
4. Runnable benchmark source, expected path:
   - `examples/brain_ossm_abide.sio`

## Paths

- Expected dataset root:
  - `/orangefs/training/sounio/abide-data`
- Expected manifest path:
  - `/orangefs/training/sounio/abide-data/abide_roi_manifest.tsv`
- Preflight output:
  - `/orangefs/training/sounio/abide-preflight/<RUN_ID>/preflight.txt`
- Future training outputs:
  - `/orangefs/training/sounio/abide-runs/<RUN_ID>/`
- External baseline outputs:
  - `/orangefs/training/sounio/abide-baselines/<RUN_ID>/`
- Full campaign outputs:
  - `/orangefs/training/sounio/abide-campaign/<RUN_ID>/`

## What To Run Now

From the control plane:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-preflight-gpu.sh
```

This checks:

- whether the dataset root is visible from the Slurm login surface
- whether the frozen ABIDE manifest is visible from the Slurm login surface
- whether the benchmark source exists locally
- where the preflight artifact was written

Core benchmark:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-gpu.sh
```

External deep-sequence suite:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-external-baselines-gpu.sh
```

Unified campaign:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-campaign-gpu.sh
```

Smaller external-lane smoke:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
MAX_SITES=2 LIMIT_SUBJECTS=64 SEED_COUNT=1 EPOCHS=1 BASELINE_MODELS=lstm \
  lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-external-baselines-gpu.sh
```

Smaller unified campaign smoke:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
MAX_SITES=2 LIMIT_SUBJECTS=64 SEED_COUNT=1 EPOCHS=1 BASELINE_MODELS=lstm \
  lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-campaign-gpu.sh
```

The external wrappers default to `JOB_MEM=8G` because the live `gpu-orangefs`
partition currently rejects `16G` requests with `Requested node configuration is
not available`.

## What Still Needs To Be Implemented

### Data contract

Freeze and document:

- atlas
- manifest schema
- subject inclusion criteria
- train / val / test split logic
- seed schedule

Current manifest schema:

- `subject_id`
- `label`
- `site`
- `f0` .. `f63`

Interpretation:

- one row = one subject
- `label` is binary ASD vs control
- `site` drives the hash-based holdout split
- `f0..f63` are frozen ROI-derived subject features arranged as 8 temporal steps × 8 dimensions

The manifest may now include optional leading metadata comments:

- `# schema=brain_ossm.abide.v2`
- `# seq_len=8`
- `# input_dim=8`
- `# feature_layout=flat`
- `# label_space=asd_vs_control`
- `# split_policy=leave_one_site_out`

Use `scripts/research/normalize_abide_manifest.py` when you want to rewrite an
older flat manifest into that explicit versioned contract without changing the
underlying data.

The cluster wrappers now use that same normalizer to materialize a run-local
manifest inside the staged snapshot. This matters for smoke runs: if you set
`MAX_SITES` or `LIMIT_SUBJECTS`, both the compiled Sounio benchmark and the
external deep baselines read the same filtered manifest file, so leaderboard
comparisons stay apples-to-apples.

Current benchmark semantics:

- benchmark file:
  - `examples/brain_ossm_abide.sio`
- default manifest path:
  - `/orangefs/training/sounio/abide-data/abide_roi_manifest.tsv`
- site-aware split:
  - grouped leave-one-site-out cross-validation over the real site labels
- seeds:
  - fixed 20-seed schedule inside the benchmark
- model scope:
  - frozen recurrent core + trained linear readout
  - O-SSM vs H-SSM, balanced-accuracy headline metric
- output:
  - per-seed headline lines
  - final multi-seed summary
  - per-site holdout table with `N`, `O-bal`, `H-bal`, `Gap`, and `O-assoc`
  - structured prediction trace lines consumed by `parse_brain_ossm_abide_output.py`
  - machine-readable `overall_metrics.*`, `per_seed_metrics.tsv`, `per_site_metrics.tsv`, and `prediction_rows.tsv`

External baseline semantics:

- runner file:
  - `scripts/research/abide_external_baselines.py`
- models:
  - `lstm`
  - `gru`
  - `transformer`
  - `tcn`
- metrics:
  - balanced accuracy
  - macro F1
  - AUROC
  - Brier
  - ECE
- stress knobs:
  - `TRAIN_FRACTION`
  - `DROP_CHANNEL_FRAC`
  - `NOISE_STD`
  - `MAX_SITES`
  - `LIMIT_SUBJECTS`

Python runtime resolution for the external wrappers:

- use `BASELINE_PYTHON` if provided and `import torch` works
- otherwise use system `python3` if it already has `torch`
- otherwise reuse `PYTORCH_VENV_DIR` if already provisioned
- otherwise bootstrap a cached user-base install under `PYTORCH_USERBASE_DIR`

The cold-start bootstrap now installs `torch==2.6.0` and `numpy==2.2.6` into
that cached runtime. Expect the first run on a fresh OrangeFS cache to spend
extra time downloading wheels; later runs reuse the cache.

## Recommended Execution Ladder

- smoke:
  - `SEED_COUNT=3`
  - `TRAIN_FRACTION=1.0`
  - `DROP_CHANNEL_FRAC=0.0`
  - `NOISE_STD=0.0`
- first serious run:
  - `SEED_COUNT=10`
- paper-grade run:
  - `SEED_COUNT=20`
- consolidation:
  - promote only the headline slice to `50` seeds

## Interpretation Rule

Do not headline ABIDE from a single seed.
The first real claim should come from a grouped cross-site multi-seed result bundle, with the per-site table inspected before any global conclusion.
