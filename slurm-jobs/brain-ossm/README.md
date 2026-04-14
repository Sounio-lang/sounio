# Fractal-G2 O-SSM GPU Jobs

## Overview

GPU batch jobs for the Fractal-G2 O-SSM research campaign.
17 experiments, 10-seed validation, first documented non-associative training.

Current verified ABIDE headline:

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

## Job Scripts

| Script | Purpose | Time | GPU |
|--------|---------|------|-----|
| `submit-fractal-g2-gpu.sh` | Full benchmark suite (probe + brain + supporting) | ~5 min | 1× L4 |
| `submit-fractal-g2-20seed-gpu.sh` | Fractal-G2 robust replay with 20-seed probe snapshot | ~6 min | 1× L4 |
| `submit-native-cuda-smoke-gpu.sh` | Native CUDA runtime smoke for `souc run --gpu-runtime` | ~1-2 min | 1× L4 |
| `submit-abide-preflight-gpu.sh` | ABIDE path + benchmark preflight, no training yet | ~10 sec | 0× |
| `submit-abide-gpu.sh` | Sounio ABIDE benchmark + structured metric export | ~5-10 min | 1× L4 |
| `submit-abide-external-baselines-gpu.sh` | External deep-sequence suite (LSTM/GRU/Transformer/TCN) | ~15-30 min | 1× L4 |
| `submit-abide-campaign-gpu.sh` | Full campaign bundle: Sounio + external baselines + leaderboard | ~20-40 min | 1× L4 |
| `submit-abide-robustness-gpu.sh` | Robustness campaign wrapper (`lowdata`, `missing`, `noise`) | ~20-40 min | 1× L4 |

## Submit (from control plane)

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-fractal-g2-gpu.sh
```

20-seed Fractal-G2 replay:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-fractal-g2-20seed-gpu.sh
```

ABIDE preflight:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-preflight-gpu.sh
```

Native CUDA smoke:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-native-cuda-smoke-gpu.sh
```

Current validated status:

- successful end-to-end cluster run:
  - job `244`
- worker:
  - `gpuorangefs-r770-proxmox`
- staged compiler artifact hash:
  - `7e94c4090feb5b2f8724b2fd8f37dc46`
- expected log lines:
  - `PASS: GPU vec_add`
  - `CUDA_SMOKE_OK`

If the login deployment name changed on the control plane, override it directly:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
LOGIN_POD_NAME=<live-login-pod> \
  lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-native-cuda-smoke-gpu.sh
```

Or use a selector fallback instead of the default `slurm-pilot-login-slinky` deployment:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
LOGIN_SELECTOR='app.kubernetes.io/name=login' \
  lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-native-cuda-smoke-gpu.sh
```

Alternate fixture:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
FIXTURE_REL=tests/run-pass/gpu_launch_vec_slices.sio \
  lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-native-cuda-smoke-gpu.sh
```

ABIDE benchmark only:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-gpu.sh
```

ABIDE external baselines:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-external-baselines-gpu.sh
```

Smaller external smoke:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
MAX_SITES=2 LIMIT_SUBJECTS=64 SEED_COUNT=1 EPOCHS=1 BASELINE_MODELS=lstm \
  lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-external-baselines-gpu.sh
```

Smaller unified campaign smoke with apples-to-apples manifest slicing:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
MAX_SITES=2 LIMIT_SUBJECTS=64 SEED_COUNT=1 EPOCHS=1 BASELINE_MODELS=lstm \
  lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-campaign-gpu.sh
```

The `gpu-orangefs` partition currently rejects `--mem=16G` with
`Requested node configuration is not available`, so the ABIDE external wrappers
default to `JOB_MEM=8G`. Override only if the partition contract changes.

Full ABIDE campaign:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-campaign-gpu.sh
```

Current safe production mode:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
PERSIST_MODE=orangefs \
  lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-campaign-gpu.sh

/home/devsounio/sounio/scripts/gpu/fetch_abide_campaign_from_orangefs.sh \
  --run-id <RUN_ID> \
  --dest-dir /home/devsounio/sounio/artifacts/research/abide/<RUN_ID>
```

If the run used `PERSIST_MODE=worker_local`, use instead:

```bash
/home/devsounio/sounio/scripts/gpu/fetch_abide_campaign_by_run_id.sh \
  --run-id <RUN_ID> \
  --dest-dir /home/devsounio/sounio/artifacts/research/abide/<RUN_ID>
```

Experimental payload lane with `sbcast`:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
PERSIST_MODE=orangefs PAYLOAD_TRANSFER_MODE=sbcast \
  lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-campaign-gpu.sh
```

Robustness campaign examples:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
PROFILE=lowdata TEMPORAL_PROFILE=v5 \
  lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-robustness-gpu.sh
```

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
PROFILE=missing TEMPORAL_PROFILE=v5 DROP_CHANNEL_FRAC=0.25 \
  lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-abide-robustness-gpu.sh
```

## Results

Output goes to:
- Run staging + logs: `/orangefs/training/sounio/brain-ossm-runs/<RUN_ID>/`
- Stable result copies: `/orangefs/training/sounio/ossm-results/`

Specialized wrappers can override `ORANGEFS_RESULTS_DIR` to write into subfolders
such as `/orangefs/training/sounio/ossm-results/fractal20/`.

The submit wrapper stages a minimal Sounio snapshot into OrangeFS before
calling `sbatch`, so the worker does not depend on `/home/devsounio/sounio`
being mounted.

The ABIDE wrappers also materialize a run-local `abide_roi_manifest.tsv` inside
the staged snapshot. When `MAX_SITES` or `LIMIT_SUBJECTS` is set, both the
compiled Sounio benchmark and the external baseline suite consume that same
filtered manifest, so campaign smokes stay scientifically apples-to-apples.

The campaign wrapper now also materializes `abide_run_config.tsv` inside the
staged snapshot. This means the Sounio benchmark sees the same robustness knobs
that the external suite sees:

- `train_fraction`
- `drop_channel_frac`
- `noise_std`

The ABIDE wrappers also accept a first-class O-SSM training surface:

- `OCT_TRAIN_PROFILE=o_default|o_g2_transfer|o_assoc_stable|o_alg_v1|o_alg_v1_mandel`
- `OCT_INPUT_PROJ_MODE=identity|learned_linear|residual_linear|mandelbrot_d2_residual|mandelbrot_d2_hybrid`
- `H_INPUT_PROJ_MODE=identity|learned_linear|residual_linear|mandelbrot_d2_residual|mandelbrot_d2_hybrid`

The algebra-aware O-SSM surface is now explicit:

- `o_assoc_stable`: current stable branch built around `g2preset=1` and a fixed associator target
- `o_alg_v1`: first explicit algebraic trainer with associator schedule, drift regularization, trust-region clipping, and head renorm
- `o_alg_v1_mandel`: same algebraic trainer family, but intended for the structured Mandelbrot-d2 projection branches

Current recommended O-SSM branch:

- `OCT_TRAIN_PROFILE=o_assoc_stable`
- `OCT_INPUT_PROJ_MODE=identity`
- `H_INPUT_PROJ_MODE=identity`
- `residual_linear` preserved more signal than `learned_linear`, but the new default search direction is algebra-aware training before shallow free projections.
- `mandelbrot_d2_residual` and `mandelbrot_d2_hybrid` are now first-class experimental branches for structured fractal projection.

The ABIDE wrappers now emit machine-readable artifacts alongside raw stdout:

- `overall_metrics.json`
- `overall_metrics.tsv`
- `per_seed_metrics.tsv`
- `per_site_metrics.tsv`
- `prediction_rows.tsv`
- `leaderboard.tsv` and `per_site_leaderboard.tsv` for the full campaign wrapper

If you rerun the same `RUN_ID`, the wrapper reuses the staged payload already
present under `/orangefs/training/sounio/abide-campaign-runs/<RUN_ID>/repo-payload.tgz`.
The worker now extracts the repo snapshot locally under `/tmp` and only writes
results back to OrangeFS, which is much more stable than copying a full repo
tree into OrangeFS before `sbatch`. Set `FORCE_RESTAGE=1` to rebuild and upload
the payload again.

Current cluster note:

- the Slurm `gpuorangefs` worker pool is gated by the node label
  `sounio.dev/slurm-worker-gpuorangefs=true`
- `r770-proxmox` is admitted in that pool
- `r740-proxmox` is currently quarantined again after later worker churn
  invalidated one of the full-train reruns
- `orangefs` is the current default again after the cleanup / validation pass
- `worker_local + fetch` remains available as the safe fallback if OrangeFS
  degrades again
- the persisted `abide_campaign_bundle.tgz` archive remains the canonical
  shared artifact for run recovery
- use
  [`fetch_abide_campaign_from_orangefs.sh`](/home/devsounio/sounio/scripts/gpu/fetch_abide_campaign_from_orangefs.sh)
  to pull and extract that bundle locally by `RUN_ID`
- `PAYLOAD_TRANSFER_MODE=sbcast` is available as an experimental lane for
  payload distribution inside the Slurm allocation
  - the previous Slurm-side `stepmgr` rendering bug was fixed via the patched
    operator image
  - standalone `sbcast` smoke now succeeds on the healthy lane
  - ABIDE campaign smoke completes successfully through `sbcast` on the
    `worker_local + fetch` lane
  - after OrangeFS persistence hardening, `sbcast + orangefs` also validates
    cleanly on both admitted workers:
    - `job 186` on `r770`
    - `job 183` on `r740`
  - `worker_local + fetch` remains the cleanest recovery path if you want the
    least coupling to shared storage state
  - `embedded` remains the stable default because it has more production
    mileage on this campaign path

Recommended platform-side gate before trusting a repaired worker node:

```bash
K8S_NODE_NAME=r740-proxmox \
  /home/devsounio/beagle/k8s/hpc-sota/slurm-pilot/scripts/66-gpuorangefs-gate.sh
```

Canonical self-heal path when an admitted worker regresses:

```bash
/home/devsounio/beagle/k8s/hpc-sota/slurm-pilot/scripts/68-manage-gpuorangefs-worker.sh \
  r740-proxmox repair
```

Safe preflight check before launching long jobs on an admitted worker:

```bash
/home/devsounio/beagle/k8s/hpc-sota/slurm-pilot/scripts/69-autoheal-gpuorangefs-worker.sh \
  r740-proxmox
```

The platform also now runs a periodic safe autoheal loop on `t560` via
`slurm-pilot-gpuorangefs-autoheal.timer`, so transient worker regressions are
more likely to be repaired before they block campaign jobs.
That loop now emits node-exporter textfile metrics and Prometheus alerts for
stale timer runs, repaired workers, deferred repairs, and hard autoheal
failures.

The gate now waits briefly for `cilium-health` to converge during agent restart
windows instead of failing immediately on transient `0/0` output.

The external baseline wrappers resolve Python conservatively:
- `BASELINE_PYTHON` if it already imports `torch`
- otherwise system `python3` if `torch` is already installed there
- otherwise cached venv under `PYTORCH_VENV_DIR`
- otherwise cached user-base bootstrap under `PYTORCH_USERBASE_DIR`

On the first cold run, the wrapper may spend a minute bootstrapping
`torch==2.6.0` and `numpy==2.2.6` into the cached runtime in OrangeFS. After
that, later runs reuse the cached runtime and start much faster.

## Key Files

| File | What |
|------|------|
| `fractal_g2_ossm_v3.sio` | 10-seed probe + ListOps, λ annealing, Fano curriculum |
| `brain_ossm_classifier.sio` | Brain connectome ASD/ADHD/Control classifier |
| `hssm_native_algebra.sio` | Parameter-matched Cayley vs Hamilton vs diagonal |
| `multihead_unit_oct_benchmark.sio` | Composition algebra mechanism validation |
| `associativity_probe_benchmark.sio` | Fano plane ground truth |

## Expected Results (from CPU runs)

```
O-SSM Fractal-G2 v3 (10 seeds):
  Overall:  12.4% ± 0.5% (2.0× random)
  NonAssoc:  4.0% ± 0.7%
  AssocNorm: 0.076 (alive, stable)
  Gap vs H-SSM: +1.7pp overall (p≈0.02)
```

## Future: ABIDE-Scale

For real clinical data (1000+ subjects), the current execution ladder is:
1. Freeze dataset visibility and metadata with `submit-abide-preflight-gpu.sh`
2. Run the Sounio benchmark with `submit-abide-gpu.sh`
3. Run the external suite with `submit-abide-external-baselines-gpu.sh`
4. Use `submit-abide-campaign-gpu.sh` for the unified leaderboard bundle
5. Sweep `TRAIN_FRACTION`, `DROP_CHANNEL_FRAC`, and `NOISE_STD` when stress-testing robustness

See:
- `docs/research/brain-ossm/ABIDE_RUNBOOK.md`
- `docs/research/brain-ossm/ROBUSTNESS_PLAN.md`
