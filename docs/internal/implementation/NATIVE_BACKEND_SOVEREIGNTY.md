<!-- docs:meta
topic_id: repo.docs.internal.implementation.native-backend-sovereignty
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.implementation.native-backend-sovereignty
-->

# Native Backend Sovereignty

Phase 1 of the native backend sovereignty bridge is now in-tree as a real preview contract instead of prose-only planning.

## What Landed

- `native-v2` is now recognized in the self-hosted shell at `self-hosted/main.sio`; the old `native-v2-shadow` alias has been retired.
- The native backend owns explicit `v2` contract helpers in `self-hosted/native/codegen.sio` for:
  - language-visible `RuntimeContext` layout
  - preview Machine IR module summary
  - target policy summaries for `x86-64` and `AArch64`
  - machine-readable contract emission
- The active native x86-64 path now stores heap/process/runtime state in `RuntimeContext` memory instead of pinning `r12`/`r13` as hidden runtime-state registers.
- The active `native-v2` x86-64 preview lane now publishes real stack-map and deopt metadata into `.data`, emits v2 stack-map records with explicit root-temp/root-spill counts plus deopt-id/tier/OSR-eligibility fields, emits a concrete `gc_state` block plus a managed-object descriptor table, stores those pointers in `RuntimeContext.stack_maps` / `RuntimeContext.deopt_state` / `RuntimeContext.gc_state`, reserves a fixed-capacity handle table in runtime memory for native-v2 heap objects, compiles allocation overflow into a real runtime slow-path trap, and ships an executable descriptor-driven mark/compact GC model that proves precise pointer-slot scanning, relocation, and pin-aware stability rules.
- The checked JIT artifact can stage its native runtime C shim through `scripts/lib/stage_native_runtime_bundle.sh` before `build --backend native`.
- Compiler self-tests T57-T61, T72-T85, T154-T156, T166-T168, and T210-T213 pin the new contract, Machine IR, regalloc, frame-policy, runtime metadata publication, GC-state decoding, alloc slow-path behavior, and the descriptor-driven mark/compact GC model in `self-hosted/compiler/main.sio`.
- The Omega gate at `scripts/omega/omega_native_v2_shadow_gate.sh` validates the wiring, emitted artifact, and x86 preview smoke ELF.

## Current Contract

The preview backend is intentionally architecture-first, but `x86-64` is no longer contract-only:

- `RuntimeContext` is modeled as language-visible and owns GC, process I/O, instrumentation, deopt/OSR, capability, heterogeneous, and provenance groups. In the current x86 slice it also carries explicit handle-table, descriptor-table, and pin-registry slots for managed native-v2 heap objects.
- Machine IR is represented as a preview summary contract with mandatory stack maps, safepoints, deopt, OSR, and two execution tiers. On `x86-64`, the preview lane now emits concrete stack-map/deopt/gc-state/descriptor blobs, including the root-map-capable v2 record schema, publishes them through `RuntimeContext` at runtime, routes `alloc`/field/index object access through handle resolution, uses the GC state from the alloc slow path, and carries an executable descriptor-driven mark/compact collector model that exercises precise pointer-slot scanning and pin-aware relocation in self-tests; on `AArch64`, those fields remain contract-level until runtime bring-up exists.
- `x86-64` policy currently declares:
  - reserved: `rsp`, `rbp`, `rbx`
  - allocatable callee-saved pool: `r12`, `r13`, `r14`, `r15`
  - explicit allocation order for regalloc/lowering: `r15`, `r14`, `r13`, `r12`
  - active preview emitter mode: real `machine-ir-scalar-core`, fail-closed outside the supported subset
- `AArch64` policy currently declares:
  - reserved: `sp`, `fp`, `lr`, `x18`
  - allocatable callee-saved pool: `x19` through `x28`
  - real scalar-core Mach-O preview emission, but not yet runtime-attested or metadata-published like x86

## What Has Not Landed Yet

This patch does **not** claim that the full sovereignty program is complete.

Not landed yet:

- runtime-triggered tracing GC execution beyond the current handle-table, slow-path recording, and descriptor-driven collection model
- full Machine IR coverage beyond the current scalar-core subset
- deopt/OSR execution machinery
- AArch64 native emitter graduation
- unified CPU/GPU/render lowering on the new substrate

The current state is a real x86 self-hosted preview lane with a narrow Machine IR subset, plus a broader contract layer that makes the rest of the sovereignty commitments explicit in code and artifacts without breaking the existing public shell.

## Verification

Primary checks:

- `./bin/souc run self-hosted/compiler/main.sio -- --self-test`
- `bash scripts/omega/omega_native_v2_shadow_gate.sh`

The gate emits:

- `artifacts/omega/native_backend_v2_contract.v1.json`
- `artifacts/omega/native_backend_v2_gate.v1.json`
- `artifacts/omega/native_backend_v2_scalar_smoke.selftest.bin`
- compatibility copies at `artifacts/omega/native_backend_v2_shadow_contract.v1.json` and `artifacts/omega/native_backend_v2_shadow_gate.v1.json`
