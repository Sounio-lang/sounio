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
- `docs/internal/coordination/MADAROS_FOCUS_PLAN_2026-08-16.md` — active operational plan (wave dispatch, sequencing)

Attention governance (`5 = 1 + 2`):

- `.claude/ATTENTION_CHARTER.md` — binding P0 ranking (compiler + epistemic honesty)
- `.claude/attention_p0.v1.json` — machine P0 snapshot
- `bash scripts/dev/attention_brief.sh` — shepherd daily ritual
- MCP `sounio-coord` (`scripts/mcp/sounio_coord_mcp.py`) — claims + agent inbox over MCP

Parallel blocker discipline:

- `.claude/PARALLEL_BLOCKER_CONTRACT.md`

## Track B Execution Order (retired 2026-08-16)

Status: **retired — superseded by `docs/internal/coordination/MADAROS_FOCUS_PLAN_2026-08-16.md`.**

The five `.claude/prompts/*.md` files this section referenced
(`data_structures.md`, `gpu_ir_expansion.md`, `hlir_lowering.md`,
`metal_msl_codegen.md`, `ptx_regalloc_expansion.md`) no longer exist in
`.claude/prompts/`. That directory now carries a different work-breakdown
scheme (`door1_break_array_wall.md`, `door2_gpu_compute.md`,
`door3_octonion_ssm.md`, `garden.md`) unrelated to the historical Track B
ordering.

All current multi-agent execution sequencing, workstream decomposition,
writer-ceiling rules, and wave dispatch live in the Madaros Focus Plan,
which is the active operational plan under the same governance
(`.claude/ATTENTION_CHARTER.md`, `CLAUDE.md §4`,
`docs/internal/coordination/COMPILER_LANE_CONTRACT.md`). This index does
not duplicate the plan; refer agents there for sequencing.

## Operational Rules

1. Any behavior change must update gate evidence in the same change-set.
2. Any prompt execution must declare dependencies and target files first.
3. `self-hosted/check/check.sio` is a sensitive merge surface: only one call-path prompt modifies it at a time.
4. No-rust policy is fail-closed; never silently fallback to local rebuild paths.
5. Before/after `.claude` governance changes, run `bash scripts/ci/claude_operational_contract_gate.sh` and refresh `artifacts/omega/claude_operational_contract_status.v1.json`.
6. The active serialized `check.sio` merge window is declared in `.claude/check_sio_integration_window.v1.json` and validated by `bash scripts/check_check_sio_integration_window.sh`.
7. Any remaining blocker in a parallel lane must use `.claude/PARALLEL_BLOCKER_CONTRACT.md` severity, class, evidence level, ownership, and handoff fields.
