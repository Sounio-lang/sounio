<!-- docs:meta
topic_id: repo.docs.implementation.gpu-capability-model
authority: historical
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.implementation.gpu-capability-model
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# GPU Capability Model

This document defines the canonical GPU capability taxonomy for the repository,
the observable surface that currently exists, and the repo-local gates that
measure each lane without destabilizing the institutional self-hosted baseline.

## Capability inventory

The current repo has four distinct GPU truth planes:

1. checked public GPU artifact
   - binary: `artifacts/omega/souc-bin/souc-linux-x86_64-gpu`
   - evidence: `info`, `check`, and `build --backend gpu`
   - scope: public source acceptance plus PTX emission
2. self-hosted compiler source lane
   - implementation: `self-hosted/compiler/lean_single.sio`
   - scope: compile-proof acceptance and frontend correctness fixes that must
     still preserve selfhost fixed-point
3. deterministic simulator/reference runtime lane
   - entrypoint: `./bin/souc run`
   - scope: CPU fallback/reference execution for GPU-oriented programs and tests
     that do not require physical GPU hardware
4. hardware-attested runtime lane
   - entrypoint: `scripts/omega/omega_gpu_runtime_attest_gate.sh`
   - scope: committed remote/runtime evidence on provisioned hardware
   - blocking semantics: informational by default until the runner is stable and
     repo-local enough to avoid making baseline mergeability depend on hardware

## Syntax to lowering to runtime mapping

| Surface | Primary evidence | Current class | Notes |
| --- | --- | --- | --- |
| `kernel fn` + `with GPU` | `tests/run-pass/gpu_launch_surface.sio`, `tests/run-pass/kernel_fn_gpu_effect.sio` | `gpu-surface-supported` | Checked GPU artifact accepts the syntax. |
| `perform GPU.launch(...)` | `tests/run-pass/gpu_launch_surface.sio` | `gpu-surface-supported` | Checked via public GPU artifact for the baseline 1D-default tuple shape. The multidimensional-tuple fixture this row used to cite, `tests/run-pass/gpu_launch_multidim_surface.sio`, was never committed to this repository, so explicit non-unit multidimensional tuples have no evidence here. |
| `perform GPU.sync()` | `tests/run-pass/gpu_launch_surface.sio` plus selfhost compile-fail regression | `gpu-surface-supported` | Public contract is zero-argument sync; selfhost source now rejects argumented sync. |
| PTX emission via `build --backend gpu` | `examples/gpu.sio`, `examples/kernel_vec_add.sio`, `examples/kernel_matmul.sio`, `examples/kernel_epistemic_vec_add.sio` | `gpu-lowering-supported` | Deterministic lowering truth via checked GPU artifact. |
| Kernel examples that compile in selfhost source but are not runtime-promoted here | `examples/kernel_source_level.sio`, `tests/run-pass/kernel_multi_backend.sio`, `tests/run-pass/kernel_ptx_emit.sio`, epistemic kernel smokes | `gpu-compile-proof` | Kept honest until runtime evidence exists. |
| CPU fallback/reference execution | `tests/run-pass/gpu_kernel_basic.sio`, `tests/gpu/test_gpu_pipeline.sio`, `tests/stdlib/gpu/test_gpu.sio`, `examples/gpu.sio`, `examples/gpu_hypercomplex.sio`, `tests/gpu/sim_runtime/gpu_launch_marshaled_count_reference.sio` | `gpu-sim-runtime-supported` | Deterministic repo-local runtime truth plane covering fail-closed marshaling for overflow-prone element counts without hardware dependency. The multidimensional-tuple, thread-budget, and nonpositive-count reference fixtures this row used to cite (`tests/run-pass/gpu_launch_multidim_surface.sio`, `tests/gpu/sim_runtime/gpu_launch_descriptor_thread_budget_reference.sio`, `tests/gpu/sim_runtime/gpu_launch_nonpositive_count_reference.sio`) were never committed, so multidimensional launch tuple carriage, the 1024-thread block-budget contract, and zero/nonpositive marshaling are unevidenced here. |
| Remote GPU attestation lane | `artifacts/omega/gpu_runtime_attest_gate.v1.json` via canonical wrapper | `gpu-hardware-runtime-supported` | Informational by default; not merge-blocking. |
| `gpu.thread_id.*`, `gpu.block_id.*`, `gpu.block_dim.*`, `gpu.alloc<T>(...)` on the checked public artifact | none in this repository | `gpu-explicit-unsupported` | Implementation breadth exists in the source tree, but both the checked public artifact and the current selfhost front-end still reject these names at the source surface. The six `tests/gpu/fixtures/gpu_public_*_not_yet_supported.sio` fixtures this row used to cite were never committed — the `tests/gpu/fixtures/` directory does not exist — so the fenced status here rests on no checked-in fixture. |

## Capability taxonomy

The repository now uses these support classes:

- `gpu-surface-supported`
  - syntax and semantic surface accepted in deterministic check mode
- `gpu-lowering-supported`
  - checked artifact lowers accepted input into backend output deterministically
- `gpu-compile-proof`
  - selfhost source accepts the program, but this wave does not claim runtime
    evidence for it
- `gpu-sim-runtime-supported`
  - deterministic repo-local runtime/reference execution exists without
    requiring GPU hardware
- `gpu-hardware-runtime-supported`
  - real hardware runtime evidence exists, but the lane remains informational
    until it is stable enough for broader blocking use
- `gpu-explicit-unsupported`
  - the surface is named, fenced, and regression-covered as unsupported

Support-class promotion rules:

1. parse success alone is never enough to call a surface runtime-supported
2. simulator/reference execution is sufficient only for
   `gpu-sim-runtime-supported`
3. physical or remote hardware evidence is required for
   `gpu-hardware-runtime-supported`
4. unsupported surfaces stay explicit until the checked artifact or source lane
   actually accepts them with evidence

## Canonical GPU gates

None of the five `scripts/gpu/gpu_*_gate.sh` entrypoints this section once
specified — `gpu_surface_lowering_gate.sh`, `gpu_compile_proof_gate.sh`,
`gpu_sim_runtime_gate.sh`, `gpu_hardware_runtime_gate.sh`, and the
`gpu_capability_gate.sh` aggregate — was ever committed to this repository, so
the capability classes above have never had the dedicated blocking gates
described here. The design intent is retained below for lineage; the paths are
not runnable.

- `gpu_surface_lowering_gate.sh` (never implemented)
  - checked public artifact
  - would validate `gpu-surface-supported`, `gpu-lowering-supported`, and
    `gpu-explicit-unsupported`
- `gpu_compile_proof_gate.sh` (never implemented)
  - self-hosted compiler lane
  - would validate `gpu-compile-proof` and focused surface regressions
- `gpu_sim_runtime_gate.sh` (never implemented)
  - deterministic CPU fallback/reference runtime
  - would validate `gpu-sim-runtime-supported`
- `gpu_hardware_runtime_gate.sh` (never implemented)
  - intended as a canonical wrapper over the omega runtime attestation lane,
    informational in `auto` mode
- `gpu_capability_gate.sh` (never implemented) — the aggregate entrypoint

The GPU gates that do exist in the tree today are
`scripts/omega/omega_gpu_runtime_attest_gate.sh` (the hardware attestation lane
this document already cites above),
`scripts/omega/omega_gpu_public_contract_gate.sh`,
`scripts/omega/omega_gpu_codegen_parity_gate.sh`,
`scripts/omega/omega_gpu_binary_attest_gate.sh`, `tests/gpu/gate_ptx_codegen.sh`,
and `tests/gpu/gate_public_gpu_cfg_build.sh`.

## Relation to the selfhost authority program

The GPU capability program is subordinate to the institutional selfhost
baseline:

- touching `lean_single.sio` still requires selfhost fixed-point discipline
- GPU changes must not weaken selfhost parity, provenance, promotion, drift, or
  release gates
- hardware GPU evidence may enrich trust, but baseline mergeability must remain
  possible without live hardware
