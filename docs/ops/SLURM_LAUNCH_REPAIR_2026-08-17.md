<!-- docs:meta
topic_id: repo.docs.ops.slurm-launch-repair-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: grok-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ops.slurm-launch-repair-2026-08-17
-->

# Slurm launch repair — diagnosis + working recipe (2026-08-17)

**Author:** grok-cli1  
**Worktree:** `/workspace/.wt/slurm-launch-repair` (not shared `/workspace/sounio`)  
**Instrument:** `SLURM_CONF=/tmp/slurm-direct.conf`  
**Partition used for proof:** `cpu-ops` (1 node, idle)  
**Refute control:** a broken recipe must fail or show missing `/workspace` paths; a working recipe must reach `status=pass` with `workspace_visible=no` and a receipt under `/orangefs/...`.

## Supported path today (read this first)

| Path | Status | Who can fix it |
|---|---|---|
| **`srun`** with scrubbed env + absolute `/bin/bash` + `--chdir=/tmp` | **Supported.** Proven COMPLETED (job **10113**). Use `scripts/dev/slurm_srun_minimal.sh`. | Lane / agent |
| **`sbatch`** for submitter `openvscode-server` | **Not repaired.** Still fails with `user_env_retrieval_failed_requeued_held` even under `--export=NONE` and `--chdir=/tmp` (jobs 10106, 10108; same class as held 9668). | **Admin / Slurm controller** — out of reach from an agent lane |

**Plain statement:** this deliverable did **not** repair `sbatch`. The `user_env_retrieval` failure is a controller-side issue for this submitter (uid / home resolution on the batch path). From a fleet lane the only honest supported path today is **`srun`** via the recipe and helper below. Do not tell downstream agents (e.g. grok-cli2 dissertation-gate re-runs) that batch is fixed.

## Held jobs (pre-existing)

| Job | Partition | Reason |
|---|---|---|
| 9635–9637 | gpu-orangefs | `launch failed requeued held` |
| 9668 | gpu-orangefs | `user env retrieval failed requeued held` |

9668 SubmitLine included `--export=ALL`. 9635–9637 also used `--export=ALL`.  
Hardware is up (`sinfo`: all partitions idle). **Commands were wrong / batch env retrieval broken** — not missing nodes.

## Culprits (measured)

### A. `sbatch` → `user_env_retrieval_failed_requeued_held` (primary for held batch jobs)

Even with `--export=NONE` and `--chdir=/tmp`, **sbatch** jobs for `openvscode-server` stay PENDING with:

```text
Reason=user_env_retrieval_failed_requeued_held
BatchFlag=2
```

Reproduced: jobs **10106**, **10108**. Never left PENDING in 40s of polling; no receipt written.

**srun** of the same logical command on the same partition **succeeds**.

**Conclusion:** the batch launch path tries to retrieve a user environment for the submitter and fails (uid mapping / no real shell home on the cluster). This is **independent** of scrubbing `/workspace` vars once `--export=NONE` is set. It is an **admin/controller problem**, not something a lane can patch in-repo. Fleet heavy jobs **must use `srun`** until an admin fixes sbatch user-env for this submitter (or until jobs are submitted as a principal slurmd can resolve).

### B. `--export=ALL` (or default inheritance) pulls ~40 `/workspace/*` paths onto the node

Measured on node with `--export=ALL`:

```text
Unable to create TMPDIR [/workspace/.tmp]: No such file or directory
HOME=/workspace/.home/openvscode-server/.agents/grok-cli1
workspace_env_count=42
home_missing
```

Node **cannot** see `/workspace`. TMPDIR is then force-rewritten to `/tmp` (noise + fragile). HOME points at a path that does not exist on the node. Matches founder note that ~26+ workspace-rooted vars are invisible on compute.

### C. Relative `bash` with `--export=NONE` → `execve(): bash: No such file or directory`

```text
srun ... --export=NONE bash -lc '...'
→ execve(): bash: No such file or directory  (exit 2)
```

Without PATH, `bash` is not found. **Use absolute `/bin/bash`.**

### D. `--chdir` / submitter cwd under `/workspace`

```text
couldn't chdir to `/workspace/...': No such file or directory: going to /tmp instead
```

srun often falls back to `/tmp`, so this alone may not kill the job, but it is not reliable. **Always set `--chdir=/tmp`** (or a path on `/orangefs`).

## Side-by-side: failing vs passing

### FAIL A — bare `bash` + `--export=NONE`

```bash
export SLURM_CONF=/tmp/slurm-direct.conf
srun --partition=cpu-ops --nodes=1 --ntasks=1 --time=00:01:00 \
  --chdir=/tmp --export=NONE \
  bash -lc 'echo no'
```

**Observed (job 10111):**

```text
srun: error: ... task 0: Exited with exit code 2
error: execve(): bash: No such file or directory
```

Log: `docs/ops/slurm_failA_export_none_bare_bash.txt`

### FAIL C — `--export=ALL` (workspace env)

```bash
export SLURM_CONF=/tmp/slurm-direct.conf
srun --partition=cpu-ops --nodes=1 --ntasks=1 --time=00:01:00 \
  --export=ALL \
  /bin/bash -lc 'echo TMPDIR=$TMPDIR; echo HOME=$HOME; ...'
```

**Observed (job 10112):**

```text
Unable to create TMPDIR [/workspace/.tmp]: No such file or directory
HOME=/workspace/.home/.../grok-cli1
workspace_env_count=42
home_missing
```

Log: `docs/ops/slurm_failC_export_all_workspace_env.txt`

### FAIL B — `sbatch` (any export scrub we tried)

```bash
sbatch --partition=cpu-ops --chdir=/tmp --export=NONE,PATH=/usr/bin:/bin,TMPDIR=/tmp \
  --wrap='...'
```

**Observed (jobs 10106, 10108):** stuck `PENDING (user env retrieval failed requeued held)` — same class as job 9668.

### PASS — scrubbed `srun` + absolute bash + chdir=/tmp

```bash
export SLURM_CONF=/tmp/slurm-direct.conf
srun --partition=cpu-ops --nodes=1 --ntasks=1 --time=00:05:00 \
  --chdir=/tmp \
  --export=NONE,PATH=/usr/bin:/bin,TMPDIR=/tmp,HOME=/tmp \
  /bin/bash -lc 'set -euo pipefail
    R=/orangefs/training/sounio/slurm-launch-repair/slurm_launch_receipt_YYYYMMDDTHHMMSSZ.txt
    mkdir -p /orangefs/training/sounio/slurm-launch-repair
    {
      echo status=pass
      echo mode=srun
      echo hostname=$(hostname)
      echo pwd=$(pwd)
      echo TMPDIR=$TMPDIR
      echo date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
      echo workspace_visible=$([ -d /workspace ] && echo yes || echo no)
      echo orangefs_visible=$([ -d /orangefs ] && echo yes || echo no)
    } | tee "$R"
  '
```

**Observed (job 10113, COMPLETED via interactive srun):**

```text
status=pass
mode=srun
hostname=cpuops-t560-proxmox
pwd=/tmp
TMPDIR=/tmp
date_utc=2026-08-17T14:50:42Z
workspace_visible=no
orangefs_visible=yes
```

Log: `docs/ops/slurm_pass_srun_scrubbed.txt`  
Earlier staged receipt on OrangeFS:  
`/orangefs/training/sounio/slurm-launch-repair/slurm_launch_receipt_20260817T144823Z.txt`  
(fetch with the same srun recipe + `cat`).

## Minimal working recipe (copy-paste)

```bash
export SLURM_CONF=/tmp/slurm-direct.conf

# Prefer partition `all` or `cpu-ops` for CPU; `gpu-orangefs` only when you need GPU.
# Do NOT use sbatch until user_env_retrieval is fixed for this submitter.

srun --partition=cpu-ops --nodes=1 --ntasks=1 --time=00:30:00 \
  --chdir=/tmp \
  --export=NONE,PATH=/usr/bin:/bin:/usr/local/bin,TMPDIR=/tmp,HOME=/tmp \
  /bin/bash -lc '
    set -euo pipefail
    # clone repo on node — /workspace is invisible
    # git clone ... /tmp/sounio && cd /tmp/sounio
    hostname
    # heavy work here
  '
```

**Rules:**

1. `SLURM_CONF=/tmp/slurm-direct.conf`
2. **`srun`, not `sbatch`** (until batch user-env works)
3. **`--chdir=/tmp`** (or `/orangefs/...`)
4. **`--export=NONE,...`** minimal allowlist — never bare `--export=ALL`
5. **Absolute `/bin/bash`**
6. **TMPDIR=/tmp** (or `/scratch` if present on node)
7. Clone on node; never assume `/workspace`
8. Stage outputs under **`/orangefs/training/...`** (visible on compute, not on login pod)

Helper script: `scripts/dev/slurm_srun_minimal.sh`

## Instrument validation (not boilerplate)

| Control | Expectation | Result |
|---|---|---|
| Broken recipe (bare bash) | fail | exit 2, execve bash missing |
| Broken recipe (export=ALL) | missing HOME, TMPDIR noise | home_missing, 42 workspace vars |
| Broken recipe (sbatch) | held | user_env_retrieval_failed |
| Working recipe | status=pass, workspace_visible=no | pass, receipt on orangefs |

A recipe that only “worked” without a demonstrated fail would not identify the culprit. Here: **sbatch user-env retrieval** is the fleet hold; **export=ALL + workspace paths** is the srun footgun; **absolute bash + chdir + export=NONE** is the pass.

## What we did not fix (and will not claim)

- **`sbatch` was not repaired.** Controller-side user-env retrieval for `openvscode-server` remains broken. Only an admin can fix that. This PR documents the failure and routes traffic to `srun`.
- Releasing historical held jobs 9635–9668 (left for owners; `scancel` if desired).
- GPU partition smoke (cpu-ops was sufficient to prove launch; GPU adds gres, same env rules).
