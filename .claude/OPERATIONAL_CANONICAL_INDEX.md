# Operational Canonical Index (cutover/no-rust-v1)

Status: active operational contract for multi-agent execution.

This file is the required entrypoint before executing any `.claude/prompts/*.md`.

## Canonical Precedence (strict)

Canonical precedence is owned by `.claude/PLAN_CANONICAL_EXECUTION.md`
(contract fingerprint: `sounio.canonical.precedence.v1`).
This index must not redefine the precedence list.

Historical context only (non-canonical):

- `.claude/plan.md`
- `.claude/pending.md`
- `.claude/session-context.md`
- `.claude/decisions/2026-02-13-rustless-cutover-unified-plan.md`

## Current Operational Snapshot

- Track A (no-rust cutover): pass
- Track B (self-host sequence): pass
- Track B order: pass
- Composite cutover status: pass

Source of truth:

- `artifacts/omega/selfhost_compiler_progress.v1.json`
- `artifacts/omega/parallel_cutover_status.v1.json`

Parallel blocker discipline:

- `.claude/PARALLEL_BLOCKER_CONTRACT.md`

## Track B Execution Order (locked)

1. `data_structures.md`
2. `gpu_ir_expansion.md`
3. `hlir_lowering.md`
4. `metal_msl_codegen.md`
5. `ptx_regalloc_expansion.md`

No reorder and no scope drift.

## Operational Rules

1. Any behavior change must update gate evidence in the same change-set.
2. Any prompt execution must declare dependencies and target files first.
3. `self-hosted/check/check.sio` is a sensitive merge surface: only one call-path prompt modifies it at a time.
4. No-rust policy is fail-closed; never silently fallback to local rebuild paths.
5. Before/after `.claude` governance changes, run `bash scripts/ci/claude_operational_contract_gate.sh` and refresh `artifacts/omega/claude_operational_contract_status.v1.json`.
6. The active serialized `check.sio` merge window is declared in `.claude/check_sio_integration_window.v1.json` and validated by `bash scripts/check_check_sio_integration_window.sh`.
7. Any remaining blocker in a parallel lane must use `.claude/PARALLEL_BLOCKER_CONTRACT.md` severity, class, evidence level, ownership, and handoff fields.
