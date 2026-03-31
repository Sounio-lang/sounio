<!-- docs:meta
topic_id: repo.docs.implementation.gpu-capability-wave2
authority: repo_only
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.implementation.gpu-capability-wave2
-->

# GPU Capability Wave 2

This document records the Wave 2 technical truth update for the governed GPU
capability track. Wave 1 established taxonomy, gates, and promotion
discipline. Wave 2 narrows the gap between syntax acceptance and runtime truth
without making the institutional selfhost baseline depend on GPU hardware.

## Canonical truth planes

Wave 2 keeps four distinct truth planes, but treats only one of them as the
deterministic runtime source of truth:

1. checked public GPU artifact
   - binary: `artifacts/omega/souc-bin/souc-linux-x86_64-gpu`
   - purpose: public surface acceptance plus PTX lowering
2. self-hosted compiler source lane
   - implementation: `self-hosted/compiler/lean_single.sio`
   - purpose: compile-proof acceptance and focused frontend/lowering repairs
3. deterministic sim-runtime lane
   - gate: `scripts/gpu/gpu_sim_runtime_gate.sh`
   - runner: `./bin/souc run`
   - purpose: canonical runtime truth without physical GPU hardware
4. hardware runtime lane
   - gate: `scripts/gpu/gpu_hardware_runtime_gate.sh`
   - purpose: supplemental hardware evidence only
   - blocking status: informational in `auto` mode

## Surface -> lowering -> sim-runtime map

| Surface or case | Evidence | Current class | Wave 2 decision |
| --- | --- | --- | --- |
| `kernel fn` + `with GPU` | `tests/run-pass/gpu_launch_surface.sio`, `examples/gpu.sio` | `gpu-surface-supported` | unchanged |
| `perform GPU.launch(...)` | `tests/run-pass/gpu_launch_surface.sio` | `gpu-surface-supported` + `gpu-sim-runtime-supported` | promoted in sim-runtime because it executes under `./bin/souc run` |
| `perform GPU.sync()` | `tests/run-pass/gpu_launch_surface.sio`, `examples/gpu.sio`, `tests/gpu/compile_proof/gpu_sync_takes_no_args.sio` | `gpu-surface-supported` + `gpu-sim-runtime-supported` | runtime-covered and compile-fail regression-backed |
| PTX emission via `build --backend gpu` | `examples/gpu.sio`, `examples/kernel_vec_add.sio`, `examples/kernel_matmul.sio`, `examples/kernel_epistemic_vec_add.sio` | `gpu-lowering-supported` | unchanged |
| Kernel source cases that compile but do not exercise runtime dispatch | `examples/kernel_source_level.sio`, `tests/run-pass/kernel_multi_backend.sio`, `tests/run-pass/kernel_ptx_emit.sio`, `tests/run-pass/epistemic_kernel_shadow.sio`, `tests/run-pass/epistemic_kernel_unary.sio` | `gpu-compile-proof` | not promoted; runtime evidence is too weak |
| CPU fallback/reference execution | `tests/run-pass/gpu_kernel_basic.sio`, `tests/gpu/test_gpu_pipeline.sio`, `tests/stdlib/gpu/test_gpu.sio`, `examples/gpu.sio`, `examples/gpu_hypercomplex.sio`, `tests/run-pass/gpu_launch_surface.sio` | `gpu-sim-runtime-supported` | canonical deterministic runtime truth plane |
| Hardware-attested runtime | `artifacts/omega/gpu_runtime_attest_gate.v1.json` through the hardware wrapper | `gpu-hardware-runtime-supported` | still informational |
| `gpu.thread_id.*`, `gpu.block_id.*`, `gpu.block_dim.*`, `gpu.alloc<T>(...)` on the checked public artifact | `tests/gpu/fixtures/gpu_public_builtin_not_yet_supported.sio` | `gpu-explicit-unsupported` | unchanged |

## Wave 2 compiler correction

Wave 2 lands one real selfhost compiler repair in
`self-hosted/compiler/lean_single.sio`:

- `perform GPU.<ident>` now enters the explicit GPU statement checker even when
  the method name is unknown
- unknown GPU perform methods now fail closed with `error: unknown GPU perform`
  instead of escaping as a generic `undefined variable`

Focused regression:

- `tests/gpu/compile_proof/gpu_unknown_perform_method.sio`

This keeps the GPU surface fail-closed and makes diagnostics match the
governed capability model.

## Canonical sim-runtime model

Wave 2 treats `scripts/gpu/gpu_sim_runtime_gate.sh` as the only canonical
repo-local runtime truth plane for GPU capability work. A case is only
eligible for `gpu-sim-runtime-supported` if:

1. it runs deterministically under `./bin/souc run`
2. it exercises meaningful runtime behavior rather than only source
   acceptance or backend emission
3. it does not require physical GPU hardware

That rule is why `tests/run-pass/gpu_launch_surface.sio` is promoted in this
wave, while the pure kernel source and PTX cases remain in compile-proof.

## Non-promotions preserved on purpose

Wave 2 explicitly does not promote these classes of evidence:

- source-only kernel examples that define GPU kernels but do not demonstrate
  dispatch/runtime behavior
- PTX emission cases whose proof is lowering truth rather than runtime truth
- hardware runtime evidence from the omega lane, because the runner is still
  informational and not stable enough to gate baseline mergeability

## Baseline discipline

Wave 2 does not change the baseline authority model:

- selfhost fixed-point remains mandatory when `lean_single.sio` changes
- source-to-artifact parity and provenance gates remain authoritative
- no required selfhost check is weakened or renamed
- GPU hardware availability remains non-blocking
