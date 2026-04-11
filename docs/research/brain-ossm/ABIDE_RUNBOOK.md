# Brain O-SSM ABIDE Runbook

This runbook defines the shortest honest path from the current synthetic benchmark suite to an ABIDE-scale temporal validation.

## Current State

- The Slurm lane is working for Sounio benchmark jobs.
- The ABIDE campaign now has verified low-data and full-train winners.
- The synthetic benchmark suite is running on cluster GPU nodes.
- ABIDE training source now exists at `examples/brain_ossm_abide.sio`.
- The benchmark expects a frozen TSV manifest at `/orangefs/training/sounio/abide-data/abide_roi_manifest.tsv`.
- The benchmark now also accepts richer flat temporal manifests at alternate paths, as long as the feature count is divisible by `8`.
- The shared Slurm-visible ABIDE path is `/orangefs/training/sounio/abide-data`.
- The Sounio benchmark now emits structured prediction traces (`PRED<TAB>...`) so post-processing can derive AUROC, Brier, and ECE.
- The external deep baseline suite now exists at `scripts/research/abide_external_baselines.py`.
- The unified aggregation step now exists at `scripts/research/aggregate_brain_ossm_campaign.py`.
- The temporal manifest builder now exists at `scripts/research/build_abide_temporal_manifest.py`.

Current best verified results:

- full-train winner:
  - `o_alg_v1_mandel + mandelbrot_d2_hybrid`
  - `O-SSM 52.297928`
  - `H-SSM 52.205751`
  - best external `transformer 50.803891`
  - artifact:
    - `/home/devsounio/sounio/artifacts/research/abide/brain-ossm-abide-campaign-20260410T231806-3519612`
- conservative full-train rerun:
  - `o_assoc_stable + identity`
  - `O-SSM 52.261573`
  - `H-SSM 52.253517`
  - artifact:
    - `/home/devsounio/sounio/artifacts/research/abide/brain-ossm-abide-campaign-20260411T010210-3663479`
- canonical human-readable summary:
  - `/home/devsounio/sounio/docs/research/brain-ossm/RESULTS_2026-04-11.md`

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
- Alternate temporal manifest path:
  - `/orangefs/training/sounio/abide-data/abide_roi_temporal_manifest_v3.tsv`
  - `/orangefs/training/sounio/abide-data/abide_roi_temporal_manifest_v4_pca.tsv`
  - `/orangefs/training/sounio/abide-data/abide_roi_temporal_manifest_v5_pca_delta.tsv`
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

Current reliable production mode:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
PERSIST_MODE=worker_local \
  lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-campaign-gpu.sh

/home/devsounio/sounio/scripts/gpu/fetch_abide_campaign_by_run_id.sh \
  --run-id <RUN_ID> \
  --dest-dir /home/devsounio/sounio/artifacts/research/abide/<RUN_ID>
```

Current orangefs-first production fetch:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
PERSIST_MODE=orangefs \
  lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-campaign-gpu.sh

/home/devsounio/sounio/scripts/gpu/fetch_abide_campaign_from_orangefs.sh \
  --run-id <RUN_ID> \
  --dest-dir /home/devsounio/sounio/artifacts/research/abide/<RUN_ID>
```

Experimental payload-distribution lane:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
PERSIST_MODE=orangefs PAYLOAD_TRANSFER_MODE=sbcast \
  lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-campaign-gpu.sh
```

Robustness campaign, defaulting to the best temporal candidate so far (`v5`):

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
PROFILE=lowdata TEMPORAL_PROFILE=v5 \
  lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-robustness-gpu.sh
```

Example missing-channel stress:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
PROFILE=missing TEMPORAL_PROFILE=v5 DROP_CHANNEL_FRAC=0.25 \
  lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-robustness-gpu.sh
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

Current infra status:

- the Slurm `gpuorangefs` worker pool is gated by the node label
  `sounio.dev/slurm-worker-gpuorangefs=true`
- `r770-proxmox` is admitted in that pool
- `r740-proxmox` remains quarantined again after renewed dataplane instability
  during later full-train reruns
- the platform-side all-in-one validation gate for this worker lane is now:
  `/home/devsounio/beagle/k8s/hpc-sota/slurm-pilot/scripts/66-gpuorangefs-gate.sh`
- the canonical worker admission helper is:
  `/home/devsounio/beagle/k8s/hpc-sota/slurm-pilot/scripts/68-manage-gpuorangefs-worker.sh`
- the same helper now also owns the canonical `repair` path for admitted
  workers that regress into `NOT_RESPONDING`
- a safe no-op health/autoheal wrapper also exists for admitted workers:
  `/home/devsounio/beagle/k8s/hpc-sota/slurm-pilot/scripts/69-autoheal-gpuorangefs-worker.sh`
- the pool-level periodic autoheal timer is also live on `t560`:
  `slurm-pilot-gpuorangefs-autoheal.timer`
- that timer now emits textfile metrics and Prometheus alerts for stale runs,
  repaired workers, deferred repairs, and hard autoheal failures
- the latest observed ABIDE campaign submit is also exported as host-local
  node-exporter metrics on `t560`, including submit age plus the resolved
  `PAYLOAD_TRANSFER_MODE` and `PERSIST_MODE`
- the gate now waits briefly for `cilium-health` to converge during agent
  restart windows instead of failing immediately on transient `0/0` output
- the gate now also waits for the matching Slurm node to register before
  submitting its smoke job, which removes the transient `Invalid node name
  specified` failure during fresh `r740` admission
- the canonical admit/quarantine helper now serializes admission per node and
  stamps an admission ID annotation, so an older failed attempt cannot tear
  down a newer successful admission
- direct end-of-job persistence into OrangeFS is working again on healthy
  `gpuorangefs` runs, and that is now the default campaign path
- `worker_local + fetch` remains the fallback if OrangeFS degrades again
- when `PERSIST_MODE=orangefs` is used, treat the persisted
  `abide_campaign_bundle.tgz` archive as the canonical shared artifact rather
  than assuming the extracted `results/` tree will already exist on the shared
  mount
- use
  [`fetch_abide_campaign_from_orangefs.sh`](/home/devsounio/sounio/scripts/gpu/fetch_abide_campaign_from_orangefs.sh)
  to pull and extract that bundle locally by `RUN_ID`
- `PAYLOAD_TRANSFER_MODE=sbcast` exists for ABIDE campaign payload
  distribution inside the Slurm allocation
  - the underlying Slurm `stepmgr` rendering bug was fixed by promoting the
    patched operator image, and a standalone `sbcast` smoke now succeeds
  - an ABIDE campaign smoke now completes end-to-end with
    `PAYLOAD_TRANSFER_MODE=sbcast` on the `worker_local + fetch` lane
  - after OrangeFS persistence hardening, `sbcast + orangefs` also validates
    end-to-end on both admitted workers:
    - `job 186` on `r770`
    - `job 183` on `r740`
  - `worker_local + fetch` remains the cleanest fallback if you want the most
    failure-isolated proof path
  - `embedded` remains the conservative default because it has the deepest
    production history on this lane
  - use `sbcast` when you explicitly want to exercise the recovered transfer
    path

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
- `f0` .. `fN`

Interpretation:

- one row = one subject
- `label` is binary ASD vs control
- `site` drives the hash-based holdout split
- `f0..fN` is a flat sequence layout whose total feature count must be divisible by `8`
- the current v2 baseline manifest is `8` temporal steps × `8` dimensions
- the current v3 temporal experiment manifest is `32` temporal steps × `8` dimensions

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

Use `scripts/research/build_abide_temporal_manifest.py` when you want to build
the richer temporal v3 contract from the cached CC200 `.1D` time-series files.
The current builder:

- z-scores each ROI time series per subject
- splits the time axis into `seq_len` windows
- pools the `200` ROI channels into `8` contiguous groups
- preserves the current `1023`-subject canonical cohort by trimming the same
  `12` control subjects that are already absent from the historical manifest

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
  - trainable recurrent core + trained linear readout
  - O-SSM vs H-SSM, balanced-accuracy headline metric
  - O-SSM now also supports named training profiles:
    - `o_default`
    - `o_g2_transfer`
    - `o_assoc_stable`
    - `o_alg_v1`
    - `o_alg_v1_mandel`
  - input projection modes:
    - `identity`
    - `learned_linear`
    - `residual_linear`
    - `mandelbrot_d2_residual`
    - `mandelbrot_d2_hybrid`
- output:
  - per-seed headline lines
  - final multi-seed summary
  - per-site holdout table with `N`, `O-bal`, `H-bal`, `Gap`, and `O-assoc`
  - structured prediction trace lines consumed by `parse_brain_ossm_abide_output.py`
  - machine-readable `overall_metrics.*`, `per_seed_metrics.tsv`, `per_site_metrics.tsv`, and `prediction_rows.tsv`

Current temporal-contract status:

- v2 flat `8x8` manifest:
  - strongest current smoke result for O-SSM
- v3 temporal `32x8` manifest:
  - contract validated end-to-end
  - current windowed ROI-group pooling recipe is scientifically honest but weak
  - first smoke on `2 sites / 64 subjects` regressed badly for O-SSM, so the
    next iteration should improve feature construction rather than merely adding
    more steps
- v4 temporal `32x8` PCA manifest:
  - built from a global CC200 PCA basis with temporal window means
  - first smoke recovered most of the damage introduced by the v3 contiguous
    pooling recipe
  - current smoke ordering is:
    - `external:gru 60.51`
    - `sounio:H-SSM 52.05`
    - `external:lstm 51.17`
    - `sounio:O-SSM 49.99`
  - this is still below the best `v2 flat 8x8` smoke for O-SSM (`58.71`), but
    it is a much stronger temporal starting point than v3
- v5 temporal `32x8` PCA + delta manifest:
  - built from a global CC200 PCA basis with per-window projected means plus
    projected temporal deltas
  - this is the best temporal O-SSM candidate so far
  - first smoke on `2 sites / 64 subjects` landed at:
    - `external:gru 54.82`
    - `external:lstm 53.93`
    - `sounio:H-SSM 51.09`
    - `sounio:O-SSM 50.00`
  - it still trails the legacy `v2 flat 8x8` smoke, but it is the right base
    for robustness sweeps because it keeps temporal structure without the
    severe collapse seen in v3

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

Current recommended experimental control surface for O-SSM:

- `OCT_TRAIN_PROFILE=o_assoc_stable`
- `OCT_INPUT_PROJ_MODE=identity`
- `H_INPUT_PROJ_MODE=identity`
- `residual_linear` remains an explicit experiment branch; do not treat it as the default champion until it beats the current identity-path O-SSM smoke.
- `o_alg_v1` is the new algebra-aware trainer branch:
  - associator schedule
  - drift regularization
  - headwise trust region
  - head renorm
- `o_alg_v1_mandel` is the structured-fractal branch:
  - same algebra-aware trainer family
  - intended for `mandelbrot_d2_residual` / `mandelbrot_d2_hybrid`

The wrappers resolve named profiles into concrete scalar knobs before staging,
so the benchmark log prints both the named profile and the resolved numeric
settings that were actually used.
- stress knobs:
  - `TRAIN_FRACTION`
  - `DROP_CHANNEL_FRAC`
  - `NOISE_STD`
  - `MAX_SITES`
  - `LIMIT_SUBJECTS`

Sounio robustness semantics:

- the benchmark now reads an optional local `abide_run_config.tsv`
- supported keys:
  - `train_fraction`
  - `drop_channel_frac`
  - `noise_std`
  - `oct_train_mode`
  - `oct_assoc_schedule_mode`
  - `oct_assoc_schedule_start`
  - `oct_assoc_schedule_mid`
  - `oct_assoc_schedule_end`
  - `oct_assoc_drift_reg`
  - `oct_head_trust_region`
  - `oct_head_renorm`
  - `oct_input_proj_mode`
  - `oct_proj_lr_scale`
  - `oct_proj_structured_scale`
  - `oct_proj_delta_scale`
  - `oct_proj_hybrid_scale`
- the campaign wrapper materializes that config automatically in the staged repo
- this keeps Sounio and the external suite aligned under the same robustness
  regime instead of perturbing only one side

The campaign snapshot is intentionally minimal now:

- upload one validated payload tarball to OrangeFS
- extract it locally on the worker under `/tmp`
- keep OrangeFS for manifests, logs, and result artifacts only

This avoids the earlier tree-copy failures that showed up when trying to
materialize a whole repo snapshot directly inside OrangeFS before `sbatch`.

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
