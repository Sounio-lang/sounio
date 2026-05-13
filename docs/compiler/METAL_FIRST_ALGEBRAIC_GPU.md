<!-- docs:meta
topic_id: repo.docs.compiler.metal-first-algebraic-gpu
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.metal-first-algebraic-gpu
-->

# Metal-First Algebraic GPU Spine

Snapshot: 2026-04-28

This lane promotes Apple Metal as the next primary accelerator proof surface for
Sounio's algebraic GPU work. The goal is not to clone CUDA/Triton/MLIR. The goal
is to make Sounio's compiler own the algebra laws that generic GPU compilers do
not see: octonion alternativity, sedenion Cayley-Dickson structure, zero-divisor
sensitivity, associator mass, and epistemic uncertainty shadows.

## Current Contract

- Metal Shading Language is treated as a no-native-f64 target.
- Source-level f64 kernels must lower through an explicit policy:
  - `f32_checked` for tolerance-bounded witnesses.
  - `float2_compensated` as a named future policy.
  - `reject` for generic f64 kernels that have no checked lowering.
- Existing O-SSM and S-SSM f64 parity programs remain CPU oracles.
- Checked Metal witnesses live under `tests/gpu/metal_first/`.
- `scripts/ci/native_v2_metal_algebra_gate.sh` records structural Metal proof on
  Linux and runs Apple Metal dispatch only when `xcrun`, `swiftc`, and an Apple
  GPU host are available.

## Boundary

This is not a public claim of full GPU f64, tensor-core performance, ROCm,
WebGPU, DDC, or full end-to-end Sounio source-to-Metal runtime parity. It is the
first honest compiler-owned Metal algebra spine: no fake `double` MSL emission,
law-profile markers in emitted kernels, and runtime-ready Metal witnesses for
O-SSM/S-SSM.
