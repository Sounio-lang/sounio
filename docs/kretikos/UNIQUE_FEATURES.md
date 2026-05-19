<!-- docs:meta
topic_id: repo.docs.kretikos.unique-features
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.kretikos.unique-features
-->

# Kretikos Unique GPU Compiler Features

This document is a repo-grounded roadmap for Kretikos, the Sounio GPU compiler
surface. It is not a publication claim by itself. Treat every row as a
claim-control entry: a feature is only externally claimable when its evidence
and gate columns point to committed, reproducible artifacts in the current
branch.

## Claim Levels

| Level | Meaning |
|---|---|
| Demonstrated | The repo has committed source plus a gate or golden artifact that exercises the feature. |
| Infrastructure present | The repo has source modules, sketches, or partial gates, but the end-to-end claim still needs a stronger artifact. |
| Design target | The feature is the next coherent compiler objective, not yet a completed result. |

## Feature Map

| Feature | Current evidence | Key files | Gate or verifier | Maturity | Next experiment |
|---|---|---|---|---|---|
| Evidence-carrying CUBIN | Kretikos now has hardware-backed evidence gates for selected CUDA rungs. Each case emits one JSON envelope binding source hash, K-AXI hash, PTX hash, CUBIN hash, SM target, `ptxas` version, NVIDIA L4 runtime identity, and validation digest. | `scripts/ci/kretikos_cubin_evidence_gate.sh`; `scripts/ci/kretikos_cubin_evidence_matrix_gate.sh`; `self-hosted/gpu/kretikos_emit_cubin.sio`; `self-hosted/gpu/kretikos_cubin_validate.sio`; `self-hosted/gpu/kretikos_json_emit.sio`; `tests/golden/kaxi_ptx/` | `bash scripts/ci/kretikos_cubin_evidence_gate.sh` and `bash scripts/ci/kretikos_cubin_evidence_matrix_gate.sh` on a CUDA Driver API host. Latest local matrix proof wrote `artifacts/omega/kretikos_cubin_evidence_matrix/kretikos_cubin_evidence_matrix.v1.json` with `status=pass`, `case_count=11`, and runtime pass on NVIDIA L4 for f32 and f64 vector add, sub, mul, div, FMA, plus the `epistemic_dual_output_f32` value/uncertainty rung. | Demonstrated | Require the evidence JSON in future GPU benchmark bundles, then extend the matrix to additional epistemic kernels with richer provenance lanes. |
| Algebra-aware tiler | K-AXI already has octonion and sedenion emission patterns. Recent benchmark work outside this doc showed that direct GPU matmul is correct but leaves performance on the table versus tiled/shared-memory CUDA-style kernels. The compiler feature should encode algebraic structure instead of relying on handwritten kernel shape. | `self-hosted/gpu/kretikos_emit_kaxi.sio`; `tests/golden/kaxi_ptx/*/octonion_mul.ptx`; `tests/golden/kaxi_ptx/*/sedenion_mul.ptx`; `tests/gpu/test_hmma_octonion.sio` | Existing emit drift check: `scripts/ci/kaxi_ptx_golden_gate.sh`. Needed: an octonion matmul tiling gate with CPU oracle, direct-kernel baseline, tiled-kernel baseline, and zero-mismatch validation. | Design target | Implement a K-AXI tiled octonion matmul pattern using shared memory/threadgroup memory, compare against the current direct pattern and a CPU oracle, and record batch-size scaling. |
| Epistemic tensor fragments | The GPU guide documents `Knowledge<T>` lowering as value, epsilon, validity, and provenance lanes; tensor-core-oriented epistemic modules already exist. Kretikos now has a narrow scalar proof that value, epsilon, validity, and provenance lanes can be emitted together and runtime-validated on CUDA. The missing feature is a first-class fragment abstraction that carries these lanes through tensor operations instead of treating epistemic state as side-channel code. | `docs/architecture/GPU_PROGRAMMING_GUIDE.md`; `self-hosted/gpu/tensor_epistemic.sio`; `self-hosted/gpu/epistemic_tensor_core.sio`; `self-hosted/gpu/epistemic_tensor_core_optimized.sio`; `self-hosted/gpu/epistemic_mma_reference.ptx`; `scripts/ci/kretikos_cross_backend_cuda_runtime_gate.sh`; `scripts/gpu/kretikos_cross_backend_cuda_runner.c` | Existing related gates: `scripts/ci/native_v2_epistemic_accel_spine_gate.sh`; `scripts/ci/epistemic_witness_gate.sh`. Current scalar lane proof: `bash scripts/ci/kretikos_cross_backend_cuda_runtime_gate.sh` validates `value`, `eps`, `valid`, and `provenance` outputs on NVIDIA L4 for `epistemic_dual_output_f32`. Needed: a tensor-fragment gate that validates the same four lanes against a CPU oracle for tensor-shaped work. | Infrastructure present, scalar runtime proof | Define `EpistemicFragment` at the GPU IR/K-AXI boundary and lower one 16x16 tensor kernel with value plus uncertainty/provenance validation. |
| Uncertainty-aware autotune | Optimisation modules already name autotuning, epistemic fusion, covariance shadows, tiled covariance, warp-vote fast paths, second-order GUM, and entropy dispatch. The unique Kretikos angle is that the search objective should include epistemic stability, not only throughput. | `self-hosted/gpu/opt/autotune.sio`; `self-hosted/gpu/opt/epistemic_fusion.sio`; `self-hosted/gpu/opt/covariance_shadow.sio`; `self-hosted/gpu/opt/tiled_covariance.sio`; `self-hosted/gpu/opt/warp_vote_fastpath.sio`; `self-hosted/gpu/opt/second_order_gum.sio`; `self-hosted/gpu/opt/entropy_dispatch.sio` | Needed: `scripts/ci/kretikos_uncertainty_autotune_gate.sh`, with throughput, IQR, error budget, and uncertainty-budget acceptance criteria. | Design target | Run two or more legal kernel variants for the same epistemic workload and choose by a Pareto objective: median time, timing IQR, max value error, and propagated uncertainty stability. |
| Compile-time confidence gates | Phase J already demonstrates K-AXI confidence-budget enforcement at GPU code-emission time: a kernel can declare `conf_budget=<N>`, and the K-AXI to PTX driver can refuse lowering when `--min-conf=<N>` is not satisfied. | `self-hosted/gpu/kretikos_emit_kaxi.sio`; `scripts/ci/kretikos_kaxi_phase_j_gate.sh`; `tests/golden/kaxi_ptx/f64_epistemic_gate/` | `scripts/ci/kretikos_kaxi_phase_j_gate.sh` | Demonstrated | Extend the gate from demo kernels to real K-AXI scientific kernels with `min_conf`, `max_error_budget`, and `provenance_required` policies. |
| Cross-backend semantic triangulation | The documented GPU pipeline targets PTX, Metal, and SPIR-V from the same GPU IR surface. Kretikos now has a structural triangulation gate for `vec_add_f32` and `epistemic_dual_output_f32`. The gate emits PTX, Metal, and SPIR-V artifacts, compiles the PTX artifacts with `ptxas`, records hashes, checks backend-specific value/uncertainty/validity/provenance markers, and binds them to scalar oracles. The PTX-side artifacts also have a CUDA Driver API runtime gate on NVIDIA L4 using global thread IDs across multiple blocks. This is not yet runtime equivalence across Metal and SPIR-V. | `docs/architecture/GPU_PROGRAMMING_GUIDE.md`; `self-hosted/gpu/ptx.sio`; `self-hosted/gpu/metal.sio`; `self-hosted/gpu/kretikos_emit_spirv.sio`; `self-hosted/gpu/kaxi_to_ptx.sio`; `self-hosted/gpu/kaxi_to_c.sio`; `self-hosted/gpu/metal_render.sio`; `self-hosted/gpu/spirv_render.sio`; `self-hosted/gpu/wgsl_render.sio`; `self-hosted/gpu/runtime/cpu_ir_interpreter.sio`; `scripts/ci/kretikos_cross_backend_semantic_gate.sh`; `scripts/ci/kretikos_cross_backend_cuda_runtime_gate.sh`; `scripts/gpu/kretikos_cross_backend_cuda_runner.c` | `bash scripts/ci/kretikos_cross_backend_semantic_gate.sh` writes `artifacts/omega/kretikos_cross_backend_semantic/kretikos_cross_backend_semantic.v1.json` with PTX, PTXAS CUBIN, Metal, and SPIR-V hashes plus marker checks for global ID calculation, value add, epsilon square-sum, validity AND, and provenance XOR. `bash scripts/ci/kretikos_cross_backend_cuda_runtime_gate.sh` writes `artifacts/omega/kretikos_cross_backend_cuda_runtime/kretikos_cross_backend_cuda_runtime.v1.json` after real CUDA launches over 4096 elements with zero value/epsilon/validity/provenance mismatch. Existing related check: `scripts/ci/kaxi_ptx_golden_gate.sh`. | Demonstrated for PTX runtime; structural for Metal/SPIR-V | Extend the gate from PTX-side CUDA runtime to Metal and SPIR-V runtime comparison where backend runtimes are available, then promote the scalar four-lane proof into tensor-fragment kernels. |

## Current Hard Boundaries

- Do not describe Kretikos as a finished optimizing GPU compiler until tiled
  matmul/shared-memory and runtime gates exist for the specific workload being
  claimed.
- Do not cite local or uncommitted benchmark bundles as repo evidence. First
  commit the result bundle, provenance, verifier, and report.
- Do not collapse PTX, Metal, and SPIR-V capability envelopes. Metal lacks
  compute `f64` and exposed tensor cores; PTX has SM-specific features; SPIR-V
  capability availability is driver-dependent.
- Do not treat a generated PTX/CUBIN file as evidence unless there is an oracle
  or digest validation path for the relevant semantics.

## Recommended Ordering

1. Require evidence-carrying CUBIN JSON in future GPU benchmark bundles, then
   extend the current f32/f64/epistemic CUDA matrix to richer epistemic rungs.
2. Add algebra-aware tiling for octonion matmul and compare it against the
   direct Kretikos path plus a CPU oracle.
3. Promote compile-time confidence gates from demo kernels to one real
   scientific or algebraic kernel.
4. Introduce epistemic tensor fragments for one small tensor-core-shaped
   workload.
5. Add uncertainty-aware autotuning once there are at least two legal variants
   for the same epistemic kernel.
6. Extend cross-backend semantic triangulation from PTX-side runtime plus
   structural Metal/SPIR-V markers to Metal and SPIR-V runtime outputs where
   those backend runtimes are available.

## Publication-Safe Summary

Kretikos is strongest when framed as an evidence-oriented GPU compiler for
scientific and epistemic workloads. Its distinctive direction is not raw GPU
code generation alone, but the combination of language-level GPU effects,
auditable K-AXI lowering, confidence-gated emission, hypercomplex algebra
targets, and uncertainty/provenance-aware GPU execution.
