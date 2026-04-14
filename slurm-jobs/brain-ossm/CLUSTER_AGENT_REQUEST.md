# Cluster Status: Native CUDA Smoke Unblocked

## Current State

The original cluster-side blockers for the native CUDA smoke are resolved.

Validated end-to-end state:

- this workspace can submit the smoke job directly into `slurm-pilot`
- the login pod path is live and `sbatch` works
- the GPU worker exposes:
  - `libcuda.so.1`
  - `libnvidia-ml.so.1`
  - `nvidia-smi`
- the worker CUDA Driver API initializes successfully
- the active workspace compiler/runtime fixes removed the previous GPU dispatch crash and invalid `kernelParams` launch layout
- the canonical active artifact hash is:
  - `7e94c4090feb5b2f8724b2fd8f37dc46`

## Final Validation

Successful cluster validation was performed with the usual submitter:

- submitter:
  - [submit-native-cuda-smoke-gpu.sh](/workspace/sounio/slurm-jobs/brain-ossm/submit-native-cuda-smoke-gpu.sh)
- smoke runner:
  - [run_native_cuda_smoke.sh](/workspace/sounio/scripts/gpu/run_native_cuda_smoke.sh)
- default fixture:
  - [gpu_vec_add.sio](/workspace/sounio/tests/run-pass/gpu_vec_add.sio)

Confirmed successful run:

- job `244`
- worker:
  - `gpuorangefs-r770-proxmox`
- staged artifact hash:
  - `7e94c4090feb5b2f8724b2fd8f37dc46`

Expected log lines were present:

- `PASS: GPU vec_add`
- `CUDA_SMOKE_OK`

## What Was Fixed

Cluster-side:

- GPU worker image/runtime now exposes the NVIDIA userspace required for the smoke path
- the worker admission/validation path was tightened to catch missing CUDA driver runtime earlier

Workspace-side:

- GPU runtime wide FFI call path now preserves the callee before reusing `rax`
- `cuLaunchKernel` `kernelParams` pointer-array layout now matches downward-growing local slots
- the active self-hosted compiler artifact was rebuilt and synced in the workspace

## Operational Guidance

Use the submitter as documented in the README. The smoke path is expected to work as long as:

- the login pod remains reachable
- the Slurm GPU queue remains available
- the worker image continues to expose the NVIDIA driver userspace

## Remaining Work

No immediate cluster-side action is required for the native CUDA smoke path.

Any future failure in this lane should be treated as a regression and triaged in this order:

1. confirm the staged artifact hash
2. confirm worker visibility of `libcuda.so.1` and `nvidia-smi`
3. rerun the smoke submitter
4. inspect `${STAGE_ROOT}/results/native_cuda_smoke.txt`
