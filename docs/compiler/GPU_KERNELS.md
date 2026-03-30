<!-- docs:meta
topic_id: repo.docs.compiler.gpu-kernels
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.gpu-kernels
-->

# Sounio GPU Kernel Guide

This guide describes the GPU kernel surface that is actually backed by the
checked GPU artifact and the current self-hosted implementation tree.

## Current contributor summary

There are two separate truths you need to hold at once:

- public contract: the checked GPU artifact accepts kernel syntax, checks launch
  surfaces, and emits PTX through `build --backend gpu`
- implementation breadth: `self-hosted/gpu/` already contains PTX, SPIR-V,
  Metal, runtime, tensor, autotuning, epistemic, and multi-GPU work

Contributor docs should always label which layer they are talking about.

For the canonical support classes and repo-local gate entrypoints, use
`docs/implementation/GPU_CAPABILITY_MODEL.md`. That document is now the source
of truth for how GPU capability is classified as surface-only, lowering-backed,
compile-proof, simulator-runtime, hardware-runtime, or explicitly unsupported.

## Public kernel contract

Use the checked GPU artifact:

```bash
export SOUC_GPU_BIN="$(pwd)/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
```

Verified commands:

```bash
"$SOUC_GPU_BIN" check examples/gpu.sio
"$SOUC_GPU_BIN" check examples/kernel_vec_add.sio
"$SOUC_GPU_BIN" check examples/kernel_matmul.sio
"$SOUC_GPU_BIN" check examples/kernel_epistemic_vec_add.sio
"$SOUC_GPU_BIN" check tests/run-pass/gpu_launch_surface.sio

"$SOUC_GPU_BIN" build examples/kernel_vec_add.sio --backend gpu -o /tmp/kernel_vec_add.ptx
"$SOUC_GPU_BIN" build examples/kernel_matmul.sio --backend gpu -o /tmp/kernel_matmul.ptx
```

That verified public contract currently includes:

- `kernel fn`
- `with GPU`
- `perform GPU.launch(...)` with explicit grid/block 3-tuples
- `perform GPU.sync()`
- PTX emission via `build --backend gpu`

Wave 5 adds checked and deterministic evidence for a non-unit multidimensional
launch tuple surface through `tests/run-pass/gpu_launch_multidim_surface.sio`.
Contributors should still keep the layered truth straight:

- public checked surface: explicit 3-tuple launch syntax is accepted
- source-tree PTX helper generation: the convenience `n`-based path remains
  1D-default unless an explicit descriptor path is used
- deterministic sim/reference lane: explicit multidimensional tuple carriage is
  exercised without turning that into a hardware-runtime claim

It does **not** currently include checked-artifact support for:

- `gpu.thread_id.*`
- `gpu.block_id.*`
- `gpu.block_dim.*`
- `gpu.alloc<T>(...)`

Each fenced surface now has a dedicated negative fixture under
`tests/gpu/fixtures/`, which keeps the public contract honest per builtin
instead of relying on a single combined rejection example.

If you want to document those surfaces, do it as implementation work or future
public contract work, not as default checked-artifact syntax.

## Kernel implementation map

Important files in the current self-hosted GPU pipeline:

- `self-hosted/gpu/hlir_to_gpu.sio` — frontend/HLIR bridge into GPU IR
- `self-hosted/gpu/kernel_ir.sio` — GPU opcode model and resource metadata
- `self-hosted/gpu/ptx.sio` and `self-hosted/gpu/ptx_emitter.sio` — PTX lowering
  and text emission
- `self-hosted/gpu/spirv.sio` — SPIR-V backend work
- `self-hosted/gpu/metal.sio` — Metal backend work
- `self-hosted/gpu/portable.sio` — target capability and portability logic
- `self-hosted/gpu/runtime/` — launch/runtime bridge work

Important evidence artifacts:

- `artifacts/omega/hlir_gpu_cross_coverage.v1.json`
- `artifacts/omega/gpu_codegen_parity.v1.json`
- `artifacts/omega/gpu_binary_attestation.v1.json`
- `artifacts/omega/gpu_runtime_attest_gate.v1.json`
- `artifacts/omega/gpu_public_contract.v1.json`

## Backend status

Current evidence-backed compute lanes:

- CUDA target lane: `cuda-sm80`
- ROCm target lane: `rocm-gfx942`

Current source-tree backend breadth beyond those attested lanes:

- PTX/CUDA
- ROCm packing path through parity + binary attestation
- SPIR-V
- Metal
- WGSL/render-oriented code under the broader GPU tree

When you need to talk about “stable” versus “implementation present,” use the
artifacts above rather than directory names alone.

## Examples and tests

Use these as the main orientation set:

- `examples/gpu.sio` — public launch-surface example
- `examples/kernel_vec_add.sio` — minimal kernel surface
- `examples/kernel_matmul.sio` — multi-kernel PTX emission smoke
- `examples/kernel_epistemic_vec_add.sio` — epistemic kernel acceptance + PTX
  emission
- `tests/run-pass/gpu_launch_surface.sio` — checked public launch contract
- `tests/run-pass/kernel_fn_gpu_effect.sio` — effect acceptance
- `tests/run-pass/kernel_multi_backend.sio` — multi-kernel acceptance

## Documentation rule

If a kernel doc claims a command is public, it must be backed by:

1. `souc info` from the checked GPU artifact
2. a passing `check` or `build --backend gpu` command
3. the committed attestation artifacts when backend/runtime claims are involved
