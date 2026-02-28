# Canonical Execution Structure (2026-02-26)

This file defines the current source-of-truth plan structure used on `cutover/no-rust-v1`.

## Why this file exists

The `.claude/` folder contains historical plans from multiple phases. Some are still useful context, but not all are canonical for current execution. This document prevents scope drift.

## Source-of-truth precedence

1. `PLAN_ORIGINAL.md`
2. `.claude/offload-specs/*.md`
3. `artifacts/omega/selfhost_compiler_progress.v1.json`
4. `artifacts/omega/parallel_cutover_status.v1.json`

Historical context docs (not canonical for ordering):
- `.claude/plan.md`
- `.claude/pending.md`
- `.claude/session-context.md`
- `.claude/decisions/2026-02-13-rustless-cutover-unified-plan.md`

Operational entrypoint docs:
- `.claude/OPERATIONAL_CANONICAL_INDEX.md`
- `.claude/PROMPT_EXECUTION_CONTRACT.md`

## Active structure

### Track A: No-Rust repo-hard cutover

- Pinned `SOUC_BIN` provenance and runtime resolution.
- No `cargo/rustc` dependency on blocking runtime/gate surfaces.
- `crates/` removed from mainline execution path.
- Governance and execution-surface artifacts must pass.

### Track B: Self-hosted compiler completion (strict order)

1. `data_structures.md`
2. `gpu_ir_expansion.md`
3. `hlir_lowering.md`
4. `metal_msl_codegen.md`
5. `ptx_regalloc_expansion.md`

No reorder and no scope drift from `PLAN_ORIGINAL.md`.

### Composite synchronization gate

`PARALLEL_SELFHOST_CUTOVER_PASS` requires:

- `REPO_HARD_NO_RUST_PASS`
- `DATA_STRUCTURES_PASS`
- `GPU_IR_EXPANSION_PASS`
- `HLIR_LOWERING_PASS`
- `METAL_MSL_CODEGEN_PASS`
- `PTX_REGALLOC_EXPANSION_PASS`
- `HLIR_GPU_CROSS_COVERAGE_PASS`
- `GPU_OPCODE_SMOKE_PASS`

## Current snapshot (from artifacts)

- Track A: pass
- Track B: pass
- Track B order: pass
- Composite status: pass

See:
- `artifacts/omega/selfhost_compiler_progress.v1.json`
- `artifacts/omega/parallel_cutover_status.v1.json`

## Working rule for all next changes

Any new work must:

1. Preserve Track B order.
2. Keep Track A non-regressing.
3. Update evidence artifacts and gate markers in the same change-set when behavior changes.
