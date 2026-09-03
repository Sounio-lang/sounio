# Cluster Agent Request: Enable Native CUDA Smoke Submission

## Goal

Enable this workspace to submit and verify the native CUDA runtime smoke job for Sounio:

- submitter: [submit-native-cuda-smoke-gpu.sh](/workspace/sounio/slurm-jobs/brain-ossm/submit-native-cuda-smoke-gpu.sh)
- smoke runner: [run_native_cuda_smoke.sh](/workspace/sounio/scripts/gpu/run_native_cuda_smoke.sh)
- default fixture: [gpu_vec_add.sio](/workspace/sounio/tests/run-pass/gpu_vec_add.sio)
- alternate fixture: [gpu_launch_vec_slices.sio](/workspace/sounio/tests/run-pass/gpu_launch_vec_slices.sio)

Success means:

1. the job submits through the Slurm/Kubernetes control plane
2. it lands on a GPU node
3. `souc run --gpu-runtime` executes inside the allocation
4. the smoke output contains `CUDA_SMOKE_OK`

## Repo State

The repo-side work is already done:

- `bin/souc` supports `run --gpu-runtime`
- `scripts/gpu/run_native_cuda_smoke.sh` detects `libcuda.so.1` from cluster-style mounts and fails hard in strict mode
- `submit-native-cuda-smoke-gpu.sh` stages the minimal repo snapshot to OrangeFS and submits an `sbatch` job
- the submitter now supports control-plane overrides:
  - `KUBECTL_BIN`
  - `LOGIN_POD_NAME`
  - `LOGIN_SELECTOR`
- `kubectl` is installed in this workspace at:
  - `/workspace/.home/openvscode-server/.local/bin/kubectl`

## What This Workspace Can Reach

- This workspace is running inside Kubernetes
- the in-cluster API is reachable
- the service account here is:
  - `system:serviceaccount:beagle:default`
- namespace of this pod:
  - `beagle`

## Current Blockers

The original access/RBAC blocker is resolved. This workspace can now:

- use `kubectl` directly
- see the live login deployment/pod in `slurm-pilot`
- `exec` into the login pod
- submit Slurm jobs from the workspace

The current blocker is no longer submission, staging, or `libcuda.so.1`. It is the PTX JIT userspace on the GPU worker.

### 1. No control-plane helper mount in this workspace

These documented control-plane paths are not mounted here:

- `/home/devsounio/beagle/k8s/hpc-sota`
- `ops/lab-ops.sh`
- `lab_copy_and_run`

That is no longer blocking direct submission from this workspace; it only means the documented helper wrapper is unavailable here.

### 2. RBAC into `slurm-pilot` is resolved

The workspace service account can now access the Slurm namespace resources required by the submitter.

### 3. Worker runtime is missing the PTX JIT compiler library

Current confirmed state on the allocated GPU worker `gpuorangefs-r770-proxmox`:

- `CUDA_VISIBLE_DEVICES=0`
- `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia-modeset` exist
- `nvidia-smi` is present and reports `NVIDIA L4`
- `ldconfig -p` reports `libcuda.so.1`
- CUDA Driver API base init works
- `libnvidia-ptxjitcompiler.so*` is **not** present under `/usr`, `/lib`, `/run`, or `/opt`
- `ldconfig -p` does **not** report the PTX JIT compiler library

This is now the active blocker.

The Sounio workspace-side runtime no longer segfaults and no longer fails at `cuModuleLoadData(...)=200`.
The current worker-visible runtime result is:

- `CUDA rc 221`
- `221 == CUDA_ERROR_JIT_COMPILER_NOT_FOUND`

That strongly indicates the node exposes `libcuda.so.1` but not the NVIDIA PTX JIT compiler userspace required to JIT-load PTX modules.

### 4. Staging model is corrected

The submitter previously extracted the repo snapshot directly onto OrangeFS, which corrupted text payload files into zero-filled content. That has been fixed in repo code:

- OrangeFS now stores only `payload.tgz`
- the Slurm batch job extracts into worker-local `/tmp`

This is no longer the active blocker.

## What The Cluster Agent Should Do

Preferred path: run the submitter from the real control plane.

### Option A: Submit from control plane

From the environment that has `kubectl` plus `lab_copy_and_run`:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
LOGIN_SELECTOR='app.kubernetes.io/name=login' \
  lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-native-cuda-smoke-gpu.sh
```

If the live login pod is already known, use the more explicit form:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
LOGIN_POD_NAME=<live-login-pod> \
  lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-native-cuda-smoke-gpu.sh
```

Alternate fixture:

```bash
cd /home/devsounio/beagle/k8s/hpc-sota
source ops/lab-ops.sh
LOGIN_SELECTOR='app.kubernetes.io/name=login' \
FIXTURE_REL=tests/run-pass/gpu_launch_vec_slices.sio \
  lab_copy_and_run /home/devsounio/sounio/slurm-jobs/brain-ossm/submit-native-cuda-smoke-gpu.sh
```

### Option B: Allow this workspace to submit directly

If the intent is to let this workspace submit without the external control plane, grant the `beagle:default` service account the minimum RBAC it needs in `slurm-pilot`:

- `pods`: `get`, `list`
- `pods/exec`: `create`
- `deployments.apps`: `get`

Optional but useful:

- `deployments.apps`: `list`
- `namespaces`: `get`

This is enough for the current submitter, which does not create Kubernetes resources in `slurm-pilot`; it only:

1. resolves a login pod
2. `exec`s into that pod
3. stages the tarball
4. runs `sbatch` inside the login pod

## What Needs To Be Confirmed By The Cluster Agent

1. Expose `libnvidia-ptxjitcompiler.so*` into the GPU job environment.
   This is now the main blocker.

   Acceptable fixes include:
   - mount the NVIDIA PTX JIT userspace into the job environment
   - install/publish the PTX JIT userspace on the worker image
   - expose a valid `libnvidia-ptxjitcompiler.so*` path under a standard loader path or a documented cluster path

2. Keep `libcuda.so.1` and `nvidia-smi` visible.
   Those are already present now and should remain part of the admission gate.

3. Does the login pod still expose:
   - `/run/slurm/sack.socket`
   - working `scontrol ping`
   - working `sbatch`

4. Is the GPU queue still:
   - partition `gpu-orangefs`
   - qos `gpuorangefs`
   - account `plruntime`
   - `--gres=gpu:1`

5. Is the node pin still valid?
   - current default in the submitter is:
     - `gpuorangefs-r770-proxmox`

6. Is OrangeFS staging still writable from the login pod?
   - `/orangefs/training/sounio/native-cuda-smoke/`

## Confirmed Current Submission State

The workspace successfully submitted:

- job `199`: failed with `126` due corrupted staged payload before the staging fix
- job `200`: submitted and executed, but failed with missing `libcuda.so.1`
- job `235`: after the workspace-side compiler/runtime fixes, no more segfault; runtime fell back after a CUDA driver failure
- job `238`: current state with fresh fixed-point artifact and runtime diagnostics; worker-side direct rerun prints:
  - `GPU unavailable: libnvidia-ptxjitcompiler.so missing (CUDA rc 221)`
  - `PASS: GPU vec_add`

This means:

- submission path works
- Slurm path works
- GPU allocation works
- payload integrity is fixed
- `libcuda.so.1` is visible
- the remaining blocker is specifically the missing PTX JIT compiler userspace on the worker image

## Quick Discovery Commands For The Cluster Agent

Use one of these from the control plane:

```bash
kubectl -n slurm-pilot get deploy,pods -l 'app.kubernetes.io/name=login' -o wide
```

```bash
kubectl -n slurm-pilot get pods -A | grep login
```

```bash
kubectl -n slurm-pilot exec <login-pod> -- bash -lc 'test -S /run/slurm/sack.socket && scontrol ping && which sbatch'
```

Worker-side probe:

```bash
kubectl -n slurm-pilot exec <login-pod> -- bash -lc '
srun -p gpu-orangefs -A plruntime --qos=gpuorangefs --gres=gpu:1 -N1 -n1 -c1 --mem=2G --time=00:03:00 -w gpuorangefs-r770-proxmox bash -lc "
  echo [host] \$(hostname)
  echo [cuda-visible] \${CUDA_VISIBLE_DEVICES:-unset}
  ls -l /dev/nvidia* 2>/dev/null || true
  grep -i nvidia /proc/modules || true
  find /usr /lib /run /opt -name \"libcuda.so*\" 2>/dev/null | sort
"
'
```

## Expected Output Once Fixed

The submitter should print:

- `Submitted batch job <JOB_ID>`
- `RUN_ID: ...`
- `Stage root: /orangefs/training/sounio/native-cuda-smoke/<RUN_ID>`

The job log should include:

- `Native CUDA Smoke`
- optional `nvidia-smi -L`
- `CUDA_SMOKE_OK`

If the smoke fails, the most important artifact is:

- `${STAGE_ROOT}/results/native_cuda_smoke.txt`

## Bottom Line

This is not blocked on Sounio compiler code anymore. The remaining work is cluster-side:

- expose `libcuda.so.1` into the GPU worker job environment
- optionally restore `nvidia-smi`
- keep the current submission path intact
