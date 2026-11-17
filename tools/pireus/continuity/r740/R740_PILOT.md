# R740 admitted-lowering pilot

The Spark pair stays on its own launcher (`benchmark_pair.py`): host fence, `pireus-spark-pair` lease, `check_pair`, and `gpu:gb10:1`. This directory is the other machine. It does not import those controls and it does not submit a job while `r740-runtime-lock.json` says the amd64 image is `UNBUILT`.

Node: `gpuorangefs-multi-r740-proxmox` (x86_64). Partition: `gpu-orangefs`. GPUs, one per step:

| GPU | architecture | gres |
|---|---|---|
| RTX A5000 | sm_86 | `gpu:a5000:1` |
| Quadro RTX 8000 | sm_75 | `gpu:quadro_rtx_8000:1` |

Admission still accepts only target identity `701202`. `ptx` still emits `.target sm_121` with the bytes frozen before this change. `ptx-sm86` and `ptx-sm75` are materialization modes selected after those same checks. The PTX ISA line stays `.version 9.0`. Whether driver 595 / CUDA 13.2 JIT-loads that ISA for sm_75 and sm_86 was not run; the host has no `nvcc`, and the observer loads PTX with `cuModuleLoadData`.

A gain is per architecture. Both resident-layout controls (direct and shuffle) on that GPU must clear median gain >= 50000 ppm and a 95% bootstrap lower bound > 0. The bootstrap is the Spark one: seed 20260906, multiplier 48271, modulus 2147483647, 4000 samples, order statistics 99 and 3899. The source hash is `arch_gain_rule.lock.json`, frozen on 2026-11-17 with `r740_observation_files` = 0. Cross-architecture numbers are descriptive. They do not promote.

Observation directories are `observed-gpu-<uuid>-sm<N>` and `timings-gpu-<uuid>-sm<N>`. The hostname is recorded and is not the key. Each step refuses unless exactly one GPU is visible and its compute capability is the one the gres asked for.

## What the launcher will execute

`r740_launch.py run` calls `srun`. It does not call `sbatch`. The sbatch text below is the same allocation with `--no-requeue`, rendered for review and marked `submitted: false`. There is no retry loop. A failed stage is recorded and a second call refuses it. A second execution does not reuse the run directory.

Today `run` raises `r740 amd64 image is not pinned; refusing to submit` before `srun`, because `derived_sif_sha256` is null. The lines below are the argv it builds for review name `20261117T120000Z-1-0123456789abcdef`, scratch `/scratch/pireus/r740`, `--mem=16G` (the default; 64G is the ceiling), `--time=30`. `UNPINNED` is replaced by the 64-hex digest only after the lock is pinned. Memory is always passed. The step is not `--exclusive`.

Probe A5000, then probe Quadro, then timing A5000, then timing Quadro:

```
srun -p gpu-orangefs -w gpuorangefs-multi-r740-proxmox --nodes=1 --ntasks=1 --ntasks-per-node=1 -c8 --gres=gpu:a5000:1 --mem=16G --propagate=NONE --kill-on-bad-exit=1 --time=30 --job-name=pireus-r740-probe-sm86 bash -c 'set -euo pipefail
export PIREUS_R740_SIF_SHA256=UNPINNED
echo "UNPINNED  /scratch/pireus/images/pireus-r740-amd64.sif" | sha256sum -c -
exec /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef/bin/run_in_container_r740.sh python3 /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef/bin/r740_observe.py --phase probe --run /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef --expect-sm 86
'
```

```
srun -p gpu-orangefs -w gpuorangefs-multi-r740-proxmox --nodes=1 --ntasks=1 --ntasks-per-node=1 -c8 --gres=gpu:quadro_rtx_8000:1 --mem=16G --propagate=NONE --kill-on-bad-exit=1 --time=30 --job-name=pireus-r740-probe-sm75 bash -c 'set -euo pipefail
export PIREUS_R740_SIF_SHA256=UNPINNED
echo "UNPINNED  /scratch/pireus/images/pireus-r740-amd64.sif" | sha256sum -c -
exec /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef/bin/run_in_container_r740.sh python3 /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef/bin/r740_observe.py --phase probe --run /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef --expect-sm 75
'
```

```
srun -p gpu-orangefs -w gpuorangefs-multi-r740-proxmox --nodes=1 --ntasks=1 --ntasks-per-node=1 -c8 --gres=gpu:a5000:1 --mem=16G --propagate=NONE --kill-on-bad-exit=1 --time=30 --job-name=pireus-r740-timing-sm86 bash -c 'set -euo pipefail
export PIREUS_R740_SIF_SHA256=UNPINNED
echo "UNPINNED  /scratch/pireus/images/pireus-r740-amd64.sif" | sha256sum -c -
exec /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef/bin/run_in_container_r740.sh python3 /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef/bin/r740_observe.py --phase timing --run /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef --expect-sm 86
'
```

```
srun -p gpu-orangefs -w gpuorangefs-multi-r740-proxmox --nodes=1 --ntasks=1 --ntasks-per-node=1 -c8 --gres=gpu:quadro_rtx_8000:1 --mem=16G --propagate=NONE --kill-on-bad-exit=1 --time=30 --job-name=pireus-r740-timing-sm75 bash -c 'set -euo pipefail
export PIREUS_R740_SIF_SHA256=UNPINNED
echo "UNPINNED  /scratch/pireus/images/pireus-r740-amd64.sif" | sha256sum -c -
exec /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef/bin/run_in_container_r740.sh python3 /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef/bin/r740_observe.py --phase timing --run /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef --expect-sm 75
'
```

Reviewed sbatch for the first step, not executed:

```
#!/bin/bash
#SBATCH -p gpu-orangefs
#SBATCH -w gpuorangefs-multi-r740-proxmox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=16G
#SBATCH --propagate=NONE
#SBATCH --kill-on-bad-exit=1
#SBATCH --time=30
#SBATCH --job-name=pireus-r740-probe-sm86
#SBATCH --no-requeue
# submitted: false. The launcher executes srun, not sbatch.
set -euo pipefail
export PIREUS_R740_SIF_SHA256=UNPINNED
echo "UNPINNED  /scratch/pireus/images/pireus-r740-amd64.sif" | sha256sum -c -
exec /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef/bin/run_in_container_r740.sh python3 /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef/bin/r740_observe.py --phase probe --run /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef --expect-sm 86
```

The other three steps use the same header with their own `--gres` and `--job-name`.

## Image gap

`runtime/runtime-lock.json` pins an arm64 SIF. This pilot needs `linux/amd64`. CI did not build one, so no digest is invented here. `run_in_container_r740.sh` checks a 64-hex `PIREUS_R740_SIF_SHA256` and then execs `/scratch/pireus/r740/apptainer-1.5.3/usr/bin/apptainer`. That prefix is not installed by this change. The Ubuntu amd64 package hash `c85138e99d2a3d7fabfb2b93e415f326bdc6e4ded357f74c147069b4865b2ccf` is the hash already recorded for Apptainer 1.5.3; it is not a hash of an image or of an installed prefix.

## Phases

Phase A is the deterministic condition, kind 1, round 0, indices 0..7. That is the generator in `cycle.py` (`lane_stride=1+2*(index%8)`, `lane_offset=(index//8+round)%16`, `load=(index//2)%2`, `layout=index%2`, `unroll` in 1,2,4,8,16). Read from `codex/pireus-inkling-cycle-20260906` at `validation/deterministic-live-baseline-20260907`, the manifest is `condition=deterministic`, `round=0`, `budget=8`, and the eight `*.proposal.json` lane plans match the table below. Those files are not in this checkout. The context hash differs per run; the lane plans do not.

| index | stride | offset | load | layout | unroll |
|---|---|---|---|---|---|
| 0 | 1 | 0 | 0 | 0 | 1 |
| 1 | 3 | 0 | 0 | 1 | 2 |
| 2 | 5 | 0 | 1 | 0 | 4 |
| 3 | 7 | 0 | 1 | 1 | 8 |
| 4 | 9 | 0 | 0 | 0 | 16 |
| 5 | 11 | 0 | 0 | 1 | 1 |
| 6 | 13 | 0 | 1 | 0 | 2 |
| 7 | 15 | 0 | 1 | 1 | 4 |

Each admitted plan is emitted for sm_86 and sm_75. Controls are direct (`load=0`) and shuffle (`load=1`), as on the Spark path. The decision for each GPU reads that GPU's two timing files only.

Phase B is 32 further deterministic plans, not a resubmit of Phase A. `cycle.py` still accepts only budget 8 or 32 and one round per run. The kept pairs, which need a budget-32 invocation and then a filter, are:

- round 0, indices 8..17 (10)
- round 1, indices 0..10 (11)
- round 2, indices 0..10 (11)

They are disjoint from Phase A and use the same lane formula. Each round is a fresh run directory. A failed stage is not resubmitted inside it.

Local materialization, on the submit host, is `admission.elf context proposal ptx-sm86` or `ptx-sm75`. `ptx` remains the sm_121 spelling used by `cycle.py materialize`. Parity stays `material_parity.sio`. Neither of those is an srun.
