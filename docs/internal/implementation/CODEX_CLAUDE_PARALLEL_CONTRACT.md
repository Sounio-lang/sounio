<!-- docs:meta
topic_id: repo.docs.internal.implementation.codex-claude-parallel-contract
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.implementation.codex-claude-parallel-contract
-->

# Codex x Claude Code Parallel Contract

Status: active
Branch: `cutover/no-rust-v1`
Scope: parallel execution without merge collisions

## 1. Objective

Keep both agents productive in parallel while preserving:

- no-rust cutover integrity (Track A),
- self-hosted compiler sequence integrity (Track B),
- zero silent file ownership conflicts.

## 2. Canonical Plan References

All work must remain aligned with:

1. `PLAN_ORIGINAL.md`
2. `.claude/offload-specs/*.md`
3. `artifacts/omega/selfhost_compiler_progress.v1.json`
4. `artifacts/omega/parallel_cutover_status.v1.json`

## 3. File Ownership (Current Session)

### Claude-owned zone (Codex does not touch)

- `self-hosted/check/**`
- `self-hosted/lexer/**`
- `self-hosted/parser/**`
- `tests/run-pass/unit_*`
- `tests/compile-fail/unit_*`

### Codex-owned zone (Claude should avoid touching)

- `.github/workflows/**`
- `scripts/**`
- `artifacts/omega/**`
- `docs/**`

### Shared zone (requires explicit handoff note)

- `self-hosted/gpu/**`
- `self-hosted/hlir/**`
- `hardware/**`
- `bootstrap/policies/**`

## 4. Locking and Handoff Protocol

Before changing a shared-zone file:

1. Append a lock entry to `artifacts/omega/agent_handoff.log.md`.
2. Include: agent, UTC timestamp, target files, intent.
3. Release lock with commit hash + validation commands executed.

Required handoff fields:

- `agent`
- `time_utc`
- `files`
- `intent`
- `checks`
- `commit`
- `status`

## 5. Conflict Rules

If either agent detects unexpected modifications outside its zone:

1. stop edits in that area immediately,
2. write a handoff log entry,
3. continue only inside owned zone until acknowledged.

## 6. Merge Gate Rule

No merge unless both are true in the same branch tip:

- `REPO_HARD_NO_RUST_PASS`
- `PARALLEL_SELFHOST_CUTOVER_PASS`

And no unresolved ownership overlap in handoff log.

## 7. Validation Minimum Per Change

Each commit must list at least one executed check:

- `python3 -m py_compile ...` (for Python changes)
- `bash scripts/dev/check_workflow_script_refs.sh` (for workflow/script changes)
- relevant gate or smoke command for behavior changes

## 8. Session Note

This contract is additive and does not change language semantics, codegen semantics, or governance schemas. It only defines collaboration safety for parallel agent execution.
