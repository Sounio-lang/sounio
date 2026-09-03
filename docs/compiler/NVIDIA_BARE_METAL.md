<!-- docs:meta
topic_id: repo.docs.compiler.nvidia-bare-metal
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.nvidia-bare-metal
-->

# Sounio NVIDIA Bare GPU Backend

This page defines the first Ω-Metal lane for NVIDIA GPUs.

Terminology:

- **Apple Metal** means the Apple GPU API backend that emits or validates Metal
  Shading Language and dispatches through Apple's Metal runtime.
- **Ω-Metal** means Sounio direct GPU machine-code ownership. For NVIDIA, that
  means Sounio emits the CUDA binary object bytes that the public CUDA Driver
  API can load.

Current backend identity:

- backend: `nvidia_bare`
- target: `GpuBareTarget::NvidiaSm80`
- format: `GpuBareFormat::CubinElf64`
- first kernel: `GpuBareKernelKind::ExitOnly`
- highest checked kernel: `GpuBareKernelKind::VecAddF32`
- highest runtime rung: `epistemic_dual_lane_f32`
- proof mode: `GpuBareProofMode::Structural`, with optional runtime admission
  and launch rungs on NVIDIA hosts

The structural proof path is intentionally fenced from PTX, MSL, SPIR-V, LLVM,
`ptxas`, `nvcc`, and CUDA PTX compiler APIs. Inspection tools such as `file`,
`readelf`, `nvdisasm`, and `cuobjdump` are allowed only after Sounio has emitted
the artifact bytes.

The existing synthetic and seeded CUBIN scripts under `scripts/omega/` remain
legacy comparison lanes. They are useful for specimen cartography and runtime
comparison, but they are not proof that Sounio owns NVIDIA machine-code
emission.

## Current Gate

Run:

```bash
bash scripts/ci/native_v2_nvidia_bare_metal_gate.sh
```

Expected result in the current container is a StoreU32Const structural pass with
runtime reported as `not_run` when NVIDIA driver/tooling is not visible. The
default gate emits:

- `artifacts/omega/sounio_bare_store_u32_const_sm80.cubin`
- `artifacts/omega/native_v2_nvidia_bare_metal_gate.v1.json`
- `artifacts/omega/nvidia_bare_cubin_cartography.v1.json`

Run the current VecAddF32 structural rung with:

```bash
SOUNIO_NVIDIA_BARE_RUNTIME_RUNG=vec_add_f32 \
  bash scripts/ci/native_v2_nvidia_bare_metal_gate.sh
```

That mode emits:

- `artifacts/omega/sounio_bare_vec_add_f32_sm80.cubin`
- `artifacts/omega/native_v2_nvidia_bare_metal_gate.v1.json`
- `artifacts/omega/nvidia_bare_cubin_cartography.v1.json`

Runtime evidence, as of 2026-04-28:

- L4 Slurm worker `gpuorangefs-r770-proxmox`
- latest job `1083`
- CUDA driver version `13020`
- CUDA device `NVIDIA_L4`, ordinal `0`, compute capability `8.9`
- CUDA Driver API `cuModuleLoadData` accepted the Sounio-emitted CUBIN
- `cuModuleGetFunction` resolved `sounio_bare_vec_add_f32_sm80`
- two `cuLaunchKernel` calls returned `CUDA_SUCCESS`: one value-lane launch and
  one uncertainty-lane launch
- the runtime harness allocated separate device output buffers, passed real
  kernel parameters, copied both results back with `cuMemcpyDtoH`, and matched
  both CPU oracles for 64 elements with nonzero uncertainty epsilon input
- runtime reason: `runtime_epistemic_dual_lane_f32_pass`
- value-lane max absolute error: `0`
- uncertainty-lane max absolute error: `0`
- artifact SHA-256:
  `1e11cdd111076f41f9c5e82d06bc281af3df2ab0bf58c395225f1ff8450ed58b`

This is a checked dual-lane runtime proof for a Sounio-owned SM80 CUBIN envelope
and SASS body. The current proof uses two launches of the same Sounio-emitted
kernel to verify value and uncertainty outputs independently. It is still a
narrow kernel proof, not a broad public NVIDIA bare runtime support claim. The
next rung is a single dedicated CUBIN kernel that writes both value and
uncertainty outputs in one launch.

Optional runtime admission mode:

```bash
SOUNIO_NVIDIA_BARE_RUNTIME=1 \
SOUNIO_NVIDIA_BARE_RUNTIME_RUNG=admission \
bash scripts/ci/native_v2_nvidia_bare_metal_gate.sh
```

Runtime rungs:

- `admission`: compile the loader, call `cuModuleLoadData`, then
  `cuModuleGetFunction`.
- `launch`: additionally call `cuLaunchKernel`.
- `store_u32_const`: allocate device memory, pass `(seed, out, n)` kernel
  parameters, call `cuLaunchKernel`, synchronize, copy the result back with
  `cuMemcpyDtoH`, and assert `observed_u32 == 42`.
- `vec_add_f32`: allocate device buffers, pass `(x, y, eps, out, n)` kernel
  parameters, call `cuLaunchKernel`, synchronize, copy the result back with
  `cuMemcpyDtoH`, and assert the output matches the CPU VecAddF32 oracle.
- `epistemic_elementwise_f32`: run the VecAddF32 CUBIN with a larger vector and
  nonzero epsilon lane, then assert exact CPU-oracle parity for
  `x + y * (1 + eps)`.
- `epistemic_dual_lane_f32`: run the VecAddF32 CUBIN twice with separate output
  buffers, once for the value lane and once for the uncertainty lane, then
  assert both CPU oracles.

The runtime classifier records exact reasons such as `cuda_driver_missing`,
`cuInit_failed`, `cuModuleLoadData_rejected`, `cuModuleGetFunction_missing`,
`cuLaunchKernel_failed`, `runtime_admission_pass`, or
`runtime_store_u32_const_pass`, `runtime_vec_add_f32_pass`, or
`runtime_epistemic_elementwise_f32_pass`, or
`runtime_epistemic_dual_lane_f32_pass`.
