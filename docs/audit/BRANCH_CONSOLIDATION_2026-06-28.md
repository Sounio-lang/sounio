# Sounio branch consolidation sweep - 2026-06-28

Status: active cleanup lane
Worktree: `/workspace/sounio-compiler-consolidation`
Branch: `integration/compiler-consolidation-20260628`
Base: `origin/main` at `63644b6b0`

## What changed in this sweep

- Created the compiler consolidation branch from `origin/main`.
- Pruned remote refs with `git fetch --prune`.
- Deleted 28 local branches that were already contained in `origin/main` and had no active worktree.
- Deleted 48 additional local branches with no patch-id-exclusive commits versus `origin/main` and no active worktree.
- Deleted 26 merged `origin/*` branches that had no active local worktree.
- Deleted 9 stale local remote-tracking refs under `refs/remotes/local/*`.

Branch counts observed in this session:

- Before cleanup after fetch: 204 local branches, 135 remote refs.
- After cleanup: 129 local branches, 100 remote refs.
- Remaining local branch audit: 128 named branches plus the detached `/workspace/sounio` checkout context.

## Current ownership split

Do not use `/workspace/sounio` as the integration surface. It is detached at
`d7befc9d1` and should remain coordination/read-only until explicitly moved.

The consolidation surface is:

```text
Lane: compiler consolidation
Owner: Codex
Base: origin/main
Worktree: /workspace/sounio-compiler-consolidation
Branch: integration/compiler-consolidation-20260628
Write-Set: docs/audit/BRANCH_CONSOLIDATION_2026-06-28.md, future curated compiler merges
Read-Set: all branches/worktrees
Required-Gates: bin/souc info; bash scripts/run_sio_test_suite.sh hello --verbose before any compiler-code merge
Merge-Target: main, after conflicts and gates are explicit
Known-Blockers: many dirty worktrees remain; no broad compiler merge without lane owner transfer
```

## Remaining branch classes

### Active worktree branches

There are 68 local branches attached to worktrees; 54 of those worktrees are
dirty. They are not deletion candidates until each owner either lands, archives,
or explicitly abandons the worktree.

High-priority dirty compiler worktrees include:

- `/workspace/sounio-project-spine` on `codex/project-spine-madaros`, dirty=10.
- `/workspace/sounio-source-elf-proof` on `codex/source-elf-on-madaros-proof`, dirty=5.
- `/workspace/sounio-madaros-check-segv` on `codex/madaros-full-functioning`, dirty=1.
- `/workspace/sounio-ir` on `claude/ir-heap-indirect`, dirty=7.
- `/workspace/sounio-effects` on `claude/effects-enforcement`, dirty=8.
- `/workspace/sounio-gpu-kernel` on `feat/gpu-thread-intrinsics`, dirty=4.
- `/workspace/sounio-solver-sota` on `research/solver-sota-class`, dirty=1 and had a live Slurm solver job during recovery.

### Review before integration

These branches still have patch-exclusive commits and compiler-surface diffs,
but are not safe for automatic merge:

- `fix/madaros-tuple-let-desugar`: broad compiler + GPU/solver/docs merge train; direct merge conflicts in `check`, `module_frontend`, `lower`, `codegen`, and witness scripts.
- `docs/pbpk-session-notes-2026-06-28`: same broad train plus audit/docs content; review separately from compiler code.
- `codex/madaros-retire-lean-single-20260627` and `codex/madaros-close-20260627`: overlapping Madaros closeout trains; likely supersede some older branches, but need gate-by-gate merge.
- `claude/solver-gpu-native-path` and `claude/gpu-e137-fix`: solver/GPU path branches; keep out of compiler core until ownership is assigned.
- `fix/binop-literal-float-478b`: debug-only codegen tracing branch; do not merge into the clean compiler line without stripping or gating the trace.
- `codex/tuple-signature-types-20260626`: likely important compiler work, but broad enough to deserve a focused merge.
- `codex/madaros-import-stdlib-lowering-current`: older imported-stdlib lowering branch; conflicts with the newer imported-lowering path already on `origin/main`.
- `codex/madaros-boxnew-clean` and `codex/madaros-boxnew-append-fix`: older Box::new trains; need comparison against current `origin/main` before any merge.

### Archive/quarantine

These are too broad or old to merge automatically:

- `fix/flatparse-and-scan-operator`
- `codex/gpu-semantic-profile`
- `codex/gpu-modular-bridge`
- `claude/kw-demote-module`
- `integration/effects-kwdemote`
- `research/erdos-compiler-wip`

Keep them as archaeological branches until a specific missing feature is named.

## Merge attempts made

Attempted `git merge --no-ff fix/madaros-tuple-let-desugar` in the consolidation
worktree. It conflicted in:

- `.claude/llm_offload_log.md`
- `scripts/ci/madaros_multimodule_witness.sh`
- `self-hosted/check/check.sio`
- `self-hosted/check/mod.sio`
- `self-hosted/compiler/main.sio`
- `self-hosted/compiler/module_frontend.sio`
- `self-hosted/ir/lower.sio`
- `self-hosted/native/codegen_x86_linux.sio`

The merge was aborted because it mixed compiler, GPU/solver, docs, and audit-log
state in one step.

Attempted focused cherry-picks:

- `4dd66d8e7` tuple-let desugar: skipped as redundant; `origin/main` already has tuple-let plus newer struct-let handling.
- `2bd304962` println f64 dispatch: skipped as redundant/equivalent; current `origin/main` already has f64 dispatch and tests.
- `36a3b1d2b`, `3158a46b3`, `4eb67966c`, `95b16f24b`: skipped as already represented by current `origin/main`.
- `220f5b8f1` imported runtime lowering: skipped after resolving showed the remaining conflict would downgrade current `with_externs`/call-target remap behavior.
- `542a91e31` imported stdlib lowering: aborted; too old and broad against current imported-lowering architecture.

## Next clean consolidation path

1. Keep `origin/main` as the compiler baseline, not any old WIP train.
2. Merge only one lane at a time into `integration/compiler-consolidation-20260628`.
3. First real candidates:
   - `codex/tuple-signature-types-20260626`
   - `codex/madaros-retire-lean-single-20260627`
   - `fix/binop-literal-float-478b` only after debug trace is converted to gated instrumentation or dropped.
4. For each candidate, require:
   - no active worktree owner collision,
   - named conflict files,
   - `bin/souc info`,
   - `bash scripts/run_sio_test_suite.sh hello --verbose`,
   - the candidate-specific gate named in its audit doc.

## Current blocker record

```text
Blocker-ID: BLK-20260628-compiler-consolidation-dirty-worktrees
Status: classified
Severity: B1
Class: ownership-conflict
Owner: Codex coordination until lane owners transfer or close worktrees
Lane: compiler consolidation
Worktree: /workspace/sounio-compiler-consolidation
Branch: integration/compiler-consolidation-20260628
Files-Owned: docs/audit/BRANCH_CONSOLIDATION_2026-06-28.md
Files-Read-Only: remaining dirty worktrees and compiler branches
Do-Not-Touch: /workspace/sounio detached checkout; dirty compiler worktrees listed above
Repro: git worktree list --porcelain && git status --short --branch in each worktree
Observed: 68 worktree branches remain; 54 are dirty
Expected: one owner per dirty lane before compiler-code integration
Acceptance-Gate: active owner map plus green focused gate after each merge
Evidence-Level: E2
Evidence: /tmp/sounio_remaining_branch_audit_20260628.tsv
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: not-required
Next-Action: choose one candidate branch and assign exclusive ownership for conflict resolution
```
