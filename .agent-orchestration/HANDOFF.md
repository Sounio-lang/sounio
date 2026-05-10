# Sounio Parallel Agent Handoff

Status: active
Updated: 2026-05-10
Workspace: `/workspace/sounio`
Safe branch: `integration/sounio-dev-ready-base`

## ACTIVE: 6-Agent Lane Assignment (2026-05-10)

A 6-agent coordination overlay is active. See
`.agent-orchestration/coordination/6_lane_assignment.md` for the full
matrix, file-set ownership, build targets, and merge order.

| # | Lane                     | Owner       | Branch                              | Worktree                                |
|---|--------------------------|-------------|-------------------------------------|-----------------------------------------|
| 1 | golden-recapture         | Claude #1   | `coord/lane-1-golden-recapture`     | `/workspace/sounio-lane-1-goldens`      |
| 2 | dissertation-evidence    | Codex #1    | `coord/lane-2-dissertation-evidence`| `/workspace/sounio-lane-2-dissertation` |
| 3 | paper-168-cohomological  | Claude #2   | `coord/lane-3-paper-168`            | `/workspace/sounio-lane-3-paper168`     |
| 4 | nv2-compiler-hardening   | Codex #2    | `coord/lane-4-nv2-hardening`        | `/workspace/sounio-lane-4-nv2`          |
| 5 | python-extermination Φ5  | Codex #3    | `coord/lane-5-phase5-recognizer`    | `/workspace/sounio-lane-5-phase5`       |
| 6 | integration-shepherd     | Claude A    | `main`                              | `/workspace/sounio` (canonical)         |

Merge order when multiple lanes are PR-ready: 1 → 4 → 5 → 2 → 3.
Lane 4 must serialize against Lane 1 on `bin/souc-linux-x86_64`.

**Out of scope** (continue independent, not part of the 6):
`garden/above-stars`, `cursor/quaternionic-ssm-88c0`,
`worktree-agent-a04d29d914b22568f`.

Live CLAIMs: `artifacts/omega/agent_handoff.log.md`.



This handoff is the shared startup packet for Codex, Claude Code, Roo, Cursor,
Kimi, and Beagle-context agents working in the Sounio repository.

## Non-negotiable startup

1. Confirm the current branch with `git branch --show-current`.
2. Read `CLAUDE_HANDOFF.md`, then `AGENTS.md`, then this file.
3. Treat `/workspace/sounio` as the active development surface.
4. Do not trust stale VM paths such as `/home/demetrios/RustroverProjects/sounio`
   as the current workspace.
5. Do not trust `.beagle/context/*` branch fields without Git verification;
   some Beagle context still says `main`.

## Current coordination facts

- Active branch verified during the sweep: `integration/sounio-dev-ready-base`.
- During orchestration setup, a local modification was observed in
  `self-hosted/compiler/lean_single.sio`. Verify current `git status` before
  touching it. If it is modified, treat it as active work owned by whoever
  started it; do not overwrite, reformat, or revert it without explicit user
  direction.
- `.claude/check_sio_integration_window.v1.json` declares an active serialized
  window for `self-hosted/check/check.sio`.
- `artifacts/omega/agent_handoff.log.md` contains the shared lock/handoff log.
  Check it before editing shared surfaces, and append entries for new locks.

## Source-of-truth order

Use this order when documents disagree:

1. Actual repo files and executable scripts.
2. `CLAUDE_HANDOFF.md`.
3. `AGENTS.md`.
4. `CLAUDE.md`.
5. `.agent-orchestration/HANDOFF.md`.
6. Tool-specific hidden-folder notes.
7. Historical docs and memory.

## Parallel ownership protocol

Before editing:

1. Run `git status --short`.
2. Identify whether target files are already modified.
3. If touching shared or high-risk files, append a lock entry to
   `artifacts/omega/agent_handoff.log.md`.
4. Keep edits inside your lane unless the user explicitly asks for cross-lane
   integration.
5. Release the lock with checks and status before handing off.

Lock entry template:

```text
---
agent: codex|claude|roo|cursor|kimi|beagle|human
time_utc: YYYY-MM-DDTHH:MM:SSZ
files:
  - path/to/file
intent: short description
checks:
  - command or "not run: reason"
commit: <sha-or-pending>
status: lock-open|lock-released|blocked
```

## High-risk shared files

Serialize edits to these files:

- `CLAUDE.md`
- `AGENTS.md`
- `.claude/settings.json`
- `.claude/settings.local.json`
- `.claude/check_sio_integration_window.v1.json`
- `.claude/PROMPT_EXECUTION_CONTRACT.md`
- `.claude/PLAN_CANONICAL_EXECUTION.md`
- `artifacts/omega/agent_handoff.log.md`
- `scripts/lib/resolve_souc.sh`
- `scripts/run_sio_test_suite.sh`
- `scripts/ci/build_native_souc.sh`
- `.github/workflows/ci.yml`
- `bin/souc`
- `self-hosted/check/check.sio`
- `self-hosted/compiler/lean_single.sio`

## Suggested lanes

- Codex: implementation, scripts, tests, small reversible commits, validation
  wiring, repo-local orchestration.
- Claude Code: read-only survey, secondary review, memory lookup, broad context
  synthesis, documentation lookup.
- Roo compiler mode: self-hosted compiler implementation within its file rules.
- Roo validation mode: tests, scripts, benchmark/gate evidence.
- Cursor: interactive editing with `.cursor/rules/*` syntax guardrails.
- Kimi: large-context implementation brief execution after a precise target file
  list and tests are supplied.
- Beagle: workspace/subagent metadata only; use Git and repo docs for authority.

## Memory routing

Do not bulk-load all memory. Use `.claude/MEMORY_LANES.md`.

Important live memory path:

`/workspace/.home/openvscode-server/.claude/projects/-workspace-sounio/memory/`

Useful Codex summaries:

`/workspace/.home/openvscode-server/.codex/memories/rollout_summaries/`

Repo Kimi material:

- `kimi-export-952daaaf-20260422-011625.md`
- `artifacts/kimi-briefing/BUG3_IMPL_DISPATCH_FIX.md`
- `artifacts/kimi-briefing/PHASE1_MODULE_RESOLVER.md`

## Validation discipline

Use repo scripts before ad hoc commands:

- Compiler resolution: `scripts/lib/resolve_souc.sh`
- Test suite: `scripts/run_sio_test_suite.sh`
- Ontology validation:
  - `scripts/ci/build_ontology_validation_souc.sh`
  - `scripts/ci/run_ontology_validation.sh`

For direct Sounio work, remember Sounio is not Rust:

- `var`, not `let mut`
- `&!`, not `&mut`
- no semicolons
- no Rust macros
- effects are explicit

## Reporting

Each substantial handoff should report:

1. Files touched.
2. What changed.
3. Commands run.
4. Results.
5. Path used: default, rebuilt wrapper, or fallback.
6. Remaining blockers.
7. Legacy paths intentionally kept.
