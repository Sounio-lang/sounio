# R740 admitted-lowering pilot

The Spark pair stays on its own launcher (`benchmark_pair.py`): host fence, `pireus-spark-pair` lease, `check_pair`, and `gpu:gb10:1`. This directory is the other machine. It does not import those controls. On r740-proxmox, `/scratch` is `/mnt/hpc-local-nvme/slurm-scratch`. The job lines keep `/scratch` because that is the path inside the allocation.

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

Run this from inside the pod `slurm-pilot-worker-gpuorangefs-multi-*`, where `/scratch` is the r740 NVMe. Do not run it from the r770 login.

`bundle --out DIR` copies the imported Phase A proposals, the baseline fixtures, and the vendored sm_86/sm_75 PTX. It writes one manifest pair per architecture. It does not need `souc`. `bundle --engine admission.elf` regenerates those PTX files and refuses if they differ from the vendored bytes or from the sm_121 baseline hashes. `prepare --name NAME --inputs DIR` allocates the run directory and stages both architectures. It does not call `srun`.

`run --name NAME --step STEP` is one `srun`, with the argv below, on that already prepared directory. It hashes the SIF first. `STEP` is `probe-sm86`, `probe-sm75`, `timing-sm86`, or `timing-sm75`. A failed probe blocks only that GPU's timing. Timing refuses unless the same GPU's probe succeeded in the same run directory.

Pause the LLM services after `prepare` and before the two probes. Resume them after both probes. Pause them again before the two timings, and resume them after the timings. Parity between those windows is local and is not an `srun`.

```
python3 tools/pireus/continuity/r740/r740_launch.py bundle --out /scratch/pireus/r740/bundle-phase-a
python3 tools/pireus/continuity/r740/r740_launch.py prepare --name 20261117T120000Z-1-0123456789abcdef --inputs /scratch/pireus/r740/bundle-phase-a
# PAUSE LLM services
python3 tools/pireus/continuity/r740/r740_launch.py run --name 20261117T120000Z-1-0123456789abcdef --step probe-sm86
python3 tools/pireus/continuity/r740/r740_launch.py run --name 20261117T120000Z-1-0123456789abcdef --step probe-sm75
# RESUME LLM services
# PAUSE LLM services
python3 tools/pireus/continuity/r740/r740_launch.py run --name 20261117T120000Z-1-0123456789abcdef --step timing-sm86
python3 tools/pireus/continuity/r740/r740_launch.py run --name 20261117T120000Z-1-0123456789abcdef --step timing-sm75
# RESUME LLM services
```

The lines below are what each `--step` emits for review name `20261117T120000Z-1-0123456789abcdef`. A real name is a fresh directory under `/scratch/pireus/r740/runs/`.

`run` hashes `/scratch/pireus/images/pireus-r740-amd64.sif` before `srun` and refuses if the size or sha256 differs from the lock. The lines below are the argv for review name `20261117T120000Z-1-0123456789abcdef`, scratch `/scratch/pireus/r740`, `--mem=16G` (the default; 64G is the ceiling), `--time=30`. The digest is the one Infra Sounio declared. Memory is always passed. The step is not `--exclusive`.

Probe A5000, then probe Quadro, then timing A5000, then timing Quadro:

```
srun -p gpu-orangefs -w gpuorangefs-multi-r740-proxmox --nodes=1 --ntasks=1 --ntasks-per-node=1 -c8 --gres=gpu:a5000:1 --mem=16G --propagate=NONE --kill-on-bad-exit=1 --time=30 --job-name=pireus-r740-probe-sm86 bash -c 'set -euo pipefail
export PIREUS_R740_SIF_SHA256=e68b2794b7718bd0f8943ffbe48c06ca5dcccebebf19e751dbf4186c6ddbbf25
echo "e68b2794b7718bd0f8943ffbe48c06ca5dcccebebf19e751dbf4186c6ddbbf25  /scratch/pireus/images/pireus-r740-amd64.sif" | sha256sum -c -
exec /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef/bin/run_in_container_r740.sh python3 /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef/bin/r740_observe.py --phase probe --run /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef --expect-sm 86
'
```

```
srun -p gpu-orangefs -w gpuorangefs-multi-r740-proxmox --nodes=1 --ntasks=1 --ntasks-per-node=1 -c8 --gres=gpu:quadro_rtx_8000:1 --mem=16G --propagate=NONE --kill-on-bad-exit=1 --time=30 --job-name=pireus-r740-probe-sm75 bash -c 'set -euo pipefail
export PIREUS_R740_SIF_SHA256=e68b2794b7718bd0f8943ffbe48c06ca5dcccebebf19e751dbf4186c6ddbbf25
echo "e68b2794b7718bd0f8943ffbe48c06ca5dcccebebf19e751dbf4186c6ddbbf25  /scratch/pireus/images/pireus-r740-amd64.sif" | sha256sum -c -
exec /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef/bin/run_in_container_r740.sh python3 /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef/bin/r740_observe.py --phase probe --run /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef --expect-sm 75
'
```

```
srun -p gpu-orangefs -w gpuorangefs-multi-r740-proxmox --nodes=1 --ntasks=1 --ntasks-per-node=1 -c8 --gres=gpu:a5000:1 --mem=16G --propagate=NONE --kill-on-bad-exit=1 --time=30 --job-name=pireus-r740-timing-sm86 bash -c 'set -euo pipefail
export PIREUS_R740_SIF_SHA256=e68b2794b7718bd0f8943ffbe48c06ca5dcccebebf19e751dbf4186c6ddbbf25
echo "e68b2794b7718bd0f8943ffbe48c06ca5dcccebebf19e751dbf4186c6ddbbf25  /scratch/pireus/images/pireus-r740-amd64.sif" | sha256sum -c -
exec /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef/bin/run_in_container_r740.sh python3 /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef/bin/r740_observe.py --phase timing --run /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef --expect-sm 86
'
```

```
srun -p gpu-orangefs -w gpuorangefs-multi-r740-proxmox --nodes=1 --ntasks=1 --ntasks-per-node=1 -c8 --gres=gpu:quadro_rtx_8000:1 --mem=16G --propagate=NONE --kill-on-bad-exit=1 --time=30 --job-name=pireus-r740-timing-sm75 bash -c 'set -euo pipefail
export PIREUS_R740_SIF_SHA256=e68b2794b7718bd0f8943ffbe48c06ca5dcccebebf19e751dbf4186c6ddbbf25
echo "e68b2794b7718bd0f8943ffbe48c06ca5dcccebebf19e751dbf4186c6ddbbf25  /scratch/pireus/images/pireus-r740-amd64.sif" | sha256sum -c -
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
export PIREUS_R740_SIF_SHA256=e68b2794b7718bd0f8943ffbe48c06ca5dcccebebf19e751dbf4186c6ddbbf25
echo "e68b2794b7718bd0f8943ffbe48c06ca5dcccebebf19e751dbf4186c6ddbbf25  /scratch/pireus/images/pireus-r740-amd64.sif" | sha256sum -c -
exec /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef/bin/run_in_container_r740.sh python3 /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef/bin/r740_observe.py --phase probe --run /scratch/pireus/r740/runs/20261117T120000Z-1-0123456789abcdef --expect-sm 86
```

The other three steps use the same header with their own `--gres` and `--job-name`.

## Image

`pireus-r740-amd64.def` is the definition Infra built. Its sha256 is `b8c273f00197c0046e802b86ef17ea016e0216c2f214c00e8c043a9f25f06f9a`. The base is `pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime` at `sha256:924707fa828f821c9a82d987b405675df6369eee5ce368aece6757b025ffcde5`. The SIF at `/scratch/pireus/images/pireus-r740-amd64.sif` is 3270336512 bytes, sha256 `e68b2794b7718bd0f8943ffbe48c06ca5dcccebebf19e751dbf4186c6ddbbf25`, declared by Infra Sounio on 2026-11-17 11:28 BRT. The host mtime is 2026-11-17 11:24:55 -0300 on r740-proxmox. This tree has not re-hashed that file.

The Apptainer prefix `/scratch/pireus/r740/apptainer-1.5.3` reports `apptainer version 1.5.3-3.el8`. It came from `install-unprivileged.sh` for tag v1.5.3 (sha256 `a097956eafa6ab3dd843ee429d0b774ed029753bfadf013b8046556228fac6f2`), which fetches `apptainer-1.5.3-3.el8.x86_64.rpm` (sha256 `db786cb78a8bba81347e46ca676b1ad7409e4cd7225f2ee48e2915bdfedd535c`, 50171128 bytes). The old noble PPA package is not this install. `run_in_container_r740.sh` still checks the SIF digest inside the job.

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
