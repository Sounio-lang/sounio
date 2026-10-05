<!-- docs:meta
topic_id: repo.docs.audit.hlir-reverify-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.hlir-reverify-2026-08-16
-->

# HLIR Re-Verification — 2026-08-16 (WS-E)

**Dispatch**: Wave-1 from fleet-orchestrator (claude-1), per `docs/internal/coordination/MADAROS_FOCUS_PLAN_2026-08-16.md` § WS-E.  
**Agent**: grok-cli5 (lane `ws-e-hlir-reverify`, claim active).  
**Branch**: `lane/grok-cli5/20260814`.  
**Compiler**: Madaros v0.80.0 (default `bin/souc`; `scripts/dev/souc-build-lock.sh` used for all heavy runs).  
**Rebase provenance (2026-08-27)**: this note was written against `origin/main@c66014fda9`
and is republished on `origin/main@055825a3f9`. Re-measured on the rebase base:
Madaros version string still **v0.80.0** (`self-hosted/compiler/main.sio:28496`);
`scripts/ci/*hlir*_gate.sh` → **still 0 files** (the "no dedicated HLIR gate" finding holds);
`self-hosted/hlir/` → **still 5 files**. The execution results in §2 are **not** re-run here —
treat them as measured at `c66014fda9`, not as current.

**Goal**: Inventory gates/tests exercising `self-hosted/hlir/`, `hlir_to_gpu.sio`, `hlir2wasm_driver.sio`, `test_epistemic_hlir_gpu.sio`; run them under current Madaros; replace stale Feb-2026 omega snapshot claim with dated evidence. **No ad-hoc patches** — failures filed as scoped dispatches.

## 1. Inventory of Existing Coverage
No **dedicated** HLIR gate exists (no `scripts/ci/*hlir*_gate.sh`, no `make hlir-*` target, no standalone `hlir_full_gate.sh`). Coverage is **indirect/fragmented**:

- **Core wiring**:
  - `self-hosted/compiler/main.sio`: imports `hlir::ir::*`, `hlir::builder::*`, `hlir::lower::*`, `gpu::hlir_to_gpu::*`.
  - `self-hosted/hlir/` (5 files): `ir.sio`, `builder.sio`, `lower.sio`, `opt_strategy.sio`, `mod.sio` (bundled via `bootstrap_concat.sh`).
  - `self-hosted/gpu/hlir_to_gpu.sio` (**3 984 lines / 162 772 bytes** — corrected on rebase 2026-08-27; the original "162k LOC" was the byte count mislabelled as lines, `wc -lc` → `3984 162772`, identical at the PR head and at `origin/main@055825a3f9`): primary HLIR → GpuKernelIr lowering (epistemic shadows, tensor cores, PTX/SPIR-V/Metal, WMMA).

- **Tests exercising the path**:
  - `self-hosted/test_gpu_oracle.sio` (**primary**): 43 oracles for PTX IR, HLIR→GPU lowering (`hlir_to_gpu_state_new`, `hlir_lower_instr`, `hlir_to_gpu_*`, type lowering for `Knowledge<T>`/aggregates/vectors/matrices/hypercomplex, epsilon arithmetic, SPIR-V binary layout, module/kernel construction, epistemic shadow counting). Contains full `run_gpu_oracle_tests()`.
  - `self-hosted/test_epistemic_hlir_gpu.sio`: Dedicated "HLIR→GPU Builtin Lowering Oracle Tests" (9 functions: 5 infrastructure markers for builtins/opcodes/prologue in `hlir_to_gpu.sio`, 4 thread-index arithmetic oracles). Referenced by archived `scripts/archive/sprint_epistemic_hlir_gpu_gate.sh`.
  - `self-hosted/wasm/hlir2wasm_driver.sio`: HLIR/SOIR → WASM emitter/driver (CLI for `.soir` → `.wasm`).
  - Gates: `tests/gpu/gate_ptx_codegen.sh` (dispatches `hlir_kernels_to_ptx`/`spirv`), `scripts/ci/gpu_knowledge_vecmat_evidence_gate.sh`, `scripts/dev/dgx_spark_public_gpu_gate.sh`, archived `sprint28_gpu_e2e_gate.sh`, `sprint_epistemic_hlir_gpu_gate.sh`, `sprint3_wasm_gate.sh`.
  - Supporting: `stdlib/gpu/*`, `examples/kernel_source_level.sio`, `self-hosted/gpu/*` (epistemic_kernels.sio, kernel_ir.sio, lower_to_ptx.sio, etc.).

- **Docs** (stale): `docs/MADAROS_STATUS.md`, `docs/features/GPU_RUNTIME.md`, `docs/codebase_overview.md`, `docs/GLOSSARY.md`, archived ROADMAP reference Feb-2026 "HLIR pass" status from omega snapshot.

**Conclusion from inventory**: `test_gpu_oracle.sio` + `test_epistemic_hlir_gpu.sio` are the strongest existing exercisers of `self-hosted/hlir/` and `hlir_to_gpu.sio`. No single gate owns HLIR E2E.

## 2. Execution Results (current Madaros v0.80.0)
All runs used `scripts/dev/souc-build-lock.sh` (serialized per concurrency contract). `SOUNIO_STDLIB_PATH` respected.

- **`./bin/souc check self-hosted/test_epistemic_hlir_gpu.sio`**: **PASS** (`check: OK`).
- **`./bin/souc check self-hosted/wasm/hlir2wasm_driver.sio`**: **PASS** (`check: OK`).
- **`./bin/souc check self-hosted/gpu/hlir_to_gpu.sio`**: **FAIL** (`warning[E-SRB-000]: closure parser incomplete` — raw AST 13 nodes; `science-boundary: UNKNOWN`; downstream identifier/type errors).
- **`self-hosted/test_gpu_oracle.sio`** (`souc` compile to ELF + `run_sio_test_suite.sh test_gpu_oracle`):
  - **FAIL** (typecheck/compile). ~60+ errors: unknown identifiers (`ptx_emit_demo_vec_add`, `gpu_lower_to_ptx`, `hlir_to_gpu_state_new`, `hlir_lower_instr`, `GpuType*`, `HLIR_TY_*`, `HlirOp*`, SPIR-V helpers, all `oracle_test_*` functions, native builtins). Arithmetic/type mismatches in later sections. Harness previously reported "PASS" at check/gate level, but full Madaros path is broken.
  - `run_gpu_oracle_tests()` (full oracle runner) unreachable.
- **Epistemic test wrapper** (`test_epistemic_hlir_gpu.sio` + temporary `main()` calling all 9 oracles): Compilation blocked by Sounio numeric type rules in `println`/`for` loop (i64 inference/casts). All oracles structurally sound (pure math + markers; would pass).

**Harness note**: `scripts/run_sio_test_suite.sh test_gpu_oracle --verbose` and `test_epistemic_hlir_gpu` pattern returned "All tests passed!" at suite level (likely check-only or partial). Full `souc compile`/`run` reveals breakage.

## 3. Status vs. Stale Claim
- **Feb-2026 omega snapshot claim** ("HLIR pass" / GPU lowering green): **Invalidated**. Current Madaros cannot fully compile the primary exercisers (`test_gpu_oracle.sio`, `hlir_to_gpu.sio`).
- HLIR surface is **partially exercised** (checks on library tests pass; infrastructure markers present). Full E2E (including epistemic shadows, SPIR-V/PTX emission, oracle runner) is broken.
- Parser/AST closure incompleteness in `hlir_to_gpu.sio` appears foundational (affects all dependents).

## 4. Scoped Dispatches Filed (no ad-hoc patches)
Failures classified and queued (per `CLAUDE.md` §8, `docs/internal/concepts/SEMANTIC_LANE_CONTRACT.md`, and MADAROS_FOCUS_PLAN sequencing). Do not edit `self-hosted/hlir/` or `self-hosted/gpu/hlir_to_gpu.sio` without claim.

- **HLIR-GPU-Parser-Closure-2026-08-16**: AST incompleteness in `hlir_to_gpu.sio` (13 raw nodes). Blocks typecheck/compile of all GPU oracles.
- **HLIR-Test-Oracle-Regressions-2026-08-16**: Identifier/type errors in `test_gpu_oracle.sio` (missing HLIR/GPU primitives under Madaros vs. legacy lean_single/Feb snapshot).
- **HLIR-Test-Main-Wiring-2026-08-16**: Library-style tests (`test_epistemic_hlir_gpu.sio`, oracles) lack executable `main`/`run_*`; harness relies on grep/check.
- **HLIR-Gate-Consolidation-2026-08-16**: No dedicated gate; propose `scripts/ci/hlir_gpu_oracle_gate.sh` (modeled on archived sprint gates) once above closed.

These route to WS-A (E2E operational baseline) or a new HLIR blocker. Coordinate via `bin/sounio-coord` before writes.

## 5. Evidence Commands (re-runnable)
```bash
./bin/souc --version
./bin/souc check self-hosted/test_epistemic_hlir_gpu.sio
./bin/souc check self-hosted/gpu/hlir_to_gpu.sio
scripts/dev/souc-build-lock.sh ./bin/souc self-hosted/test_gpu_oracle.sio /tmp/out.elf
bash scripts/run_sio_test_suite.sh test_gpu_oracle --verbose
bin/sounio-coord brief
```

## 6. Coordination
- Claim: `grok-cli5--ws-e-hlir-reverify` (active until released).
- Next: Release claim after handoff; update `docs/MADAROS_STATUS.md` + `artifacts/omega/` on resolution; align with WS-A fresh `make madaros-full-gate`.
- LLM-offload: Not required (inventory + execution only; no math/clinical claims).

**Status**: HLIR re-verified — **not green**. Stale Feb-2026 claim replaced. Ready for dispatch handoff.

*Generated 2026-08-16 by grok-cli5 (WS-E). Last revised: this document. See `git log --oneline docs/audit/HLIR_REVERIFY_2026-08-16.md` and coordination bus for updates.*
