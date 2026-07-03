<!-- docs:meta
topic_id: repo.docs.audit.madaros-worktree-cleanup-ledger-2026-07-03
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-worktree-cleanup-ledger-2026-07-03
-->

# Madaros Worktree Cleanup Ledger - 2026-07-03

## Scope

This ledger is the non-destructive disposition record for the remaining
Madaros readiness cleanup work after the protected greenline landed.

Completed before this ledger:

- PR #586 merged into `canon/madaros-greenline` at
  `bf46bda919596ce71b8fc35dc29cb3a31ff01d7b`.
- PR #587 merged into `canon/madaros-greenline` at
  `4419f2097913de65e36665d442a65c6f64d5f3ca`.
- `canon/madaros-greenline` is protected, force-push/delete are disabled,
  admins are enforced, strict required checks are enabled, and
  `Madaros Greenline Gate` is required.
- `scripts/dev/madaros_readiness_status.sh --production-ready` passed after
  refreshing `origin/main` and `origin/canon/madaros-greenline`.

This ledger does **not** approve deletion. It exists so the eventual cleanup can
be reviewed and run with a push-before-delete rule instead of ad hoc `rm -rf`.

The reproducible dry-run planner is:

```bash
scripts/dev/madaros_worktree_cleanup_plan.sh
```

It writes:

- `worktree-audit.tsv` — raw `scripts/dev/worktree_branch_audit.sh` inventory.
- `madaros-cleanup-plan.tsv` — classified unallowed critical dirty worktrees,
  including `unique_commits_origin_main`, `unique_commits_upstream`,
  tracked/untracked dirty file counts, tracked diff numstat,
  `critical_vs_base`, and a suggested `salvage_ref`.
- `madaros-cleanup-plan.commands.sh` — inspection and salvage commands, with
  mutating push/remove commands commented out.

The planner is intentionally non-destructive. It never runs `git push`,
`git reset`, `git clean`, `git branch -D`, or `git worktree remove`.

`scripts/dev/madaros_readiness_status.sh --strict` prints a
`cleanup_plan_command=...` line after the worktree audit section, using the
exact audit TSV path from that run. That is the operator handoff from red
strict-audit output to this cleanup planner.

## Current Blocker To `--strict`

Readiness production proof is green, but the stricter worktree audit still
reports dirty critical worktrees:

```text
total=56
dirty=36
critical_dirty=21
unallowed_critical_dirty=19
```

Repro command:

```bash
SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_RE='^(/workspace/sounio|/workspace/sounio-effects|/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b|/tmp/sounio-madaros-cleanup-ledger)$' \
  scripts/dev/worktree_branch_audit.sh --check /tmp/madaros-worktree-audit.tsv
```

## Cleanup Policy

Do not delete or reset any row below until the operator explicitly approves the
cleanup phase.

For every non-detached branch before deletion:

1. If the branch has useful commits or local-only dirty changes, push a salvage
   branch first.
2. Record the salvage branch name in the ledger or PR.
3. Only then remove the worktree and local branch.

For every detached worktree before deletion:

1. Create a salvage branch from its current `HEAD`.
2. Commit or archive a patch of dirty changes.
3. Push the salvage branch.
4. Only then remove the worktree.

## Disposition Table

| Path | Branch | Remote/PR | Dirty critical files | Proposed disposition |
|---|---|---|---|---|
| `/tmp/sounio-madaros-greenline-codex` | `work/madaros-greenline-codex` | remote exists; PR #586 merged | `self-hosted/ir/lower.sio`, `self-hosted/native/codegen_x86_linux.sio` | Highest-confidence cleanup candidate. Salvage dirty patch or explicitly discard after operator approval; then remove worktree. The pushed branch is already merged to canon. |
| `/tmp/sounio-madaros-plan-mainline-20260702` | `codex/madaros-plan-mainline-20260702` | no remote, no PR | wrappers/gates plus compiler/lower/codegen | Likely superseded by PR #586/#587 for wrapper/gate work, but contains compiler WIP. Create salvage branch before any deletion. |
| `/tmp/sounio-madaros-main-port-20260702` | detached | detached | `bin/madaros`, `scripts/ci/madaros_full_gate.sh`, `scripts/lib/resolve_madaros.sh`, `self-hosted/ir/lower.sio`, `scripts/dev/madaros_two_gate.sh` | Likely superseded for wrapper/gate files; detached so requires salvage branch or patch before cleanup. |
| `/tmp/sounio-phase03-step5-clean` | detached | detached | `bin/madaros`, `scripts/ci/madaros_full_gate.sh`, `scripts/lib/resolve_madaros.sh`, `self-hosted/ir/lower.sio`, `scripts/dev/madaros_two_gate.sh` | Phase03 duplicate; detached. Salvage first, then cleanup only with approval. |
| `/tmp/sounio-phase03-step5-lowerfix-min` | `codex/phase03-step5-lowerfix-min-20260702` | no remote, no PR | wrappers/gates plus `self-hosted/ir/lower.sio` | Phase03 minimal lowerfix branch. Needs owner decision: promote as future Lane V/F64 work or salvage and retire. |
| `/tmp/sounio-phase03-step5-fix` | `codex/phase03-step5-box-variance-20260702` | no remote, no PR | `self-hosted/ir/lower.sio` | Possible Lane V/variance WIP. Do not discard without owner approval; push salvage if retiring. |
| `/tmp/sounio-phase03-step5-lower-revert` | `codex/phase03-step5-lower-revert-probe-20260702` | no remote, no PR | `self-hosted/ir/lower.sio` | Possible revert/probe WIP. Salvage or classify as duplicate before deletion. |
| `/tmp/sounio-phase03-4e68` | `recover/green-first-phase03-step5` | no remote, no PR | release gates plus `self-hosted/compiler/main.sio`, `self-hosted/ir/lower.sio` | Active-looking Phase03 branch. Do not delete without explicit owner handoff. |
| `/workspace/sounio-greenfirst` | `recover/green-first` | no remote, no PR | `bin/madaros`, `self-hosted/compiler/module_frontend.sio` | Green-first source base; likely historically important. Preserve or push archival branch before any cleanup. |
| `/tmp/sounio-active-compact-ir-20260702` | `codex/active-compact-ir-20260702` | no remote, no PR | `module_frontend`, `module_native_driver`, `lower` | Imported-simple/compact-IR lane. Compare against PR #586 before deciding; salvage if any unique work remains. |
| `/tmp/sounio-active-lowerfix` | `codex/active-lowerfix-20260702` | no remote, no PR | `module_native_driver`, `lower` | Lowerfix branch; likely duplicate or investigation residue. Needs patch review before cleanup. |
| `/tmp/sounio-abide-madaros-rebuild-20260630` | `codex/abide-madaros-rebuild-20260630` | no remote, no PR | `module_native_driver` | Older single-file WIP. Salvage branch or patch before deletion. |
| `/tmp/sounio-abide-madaros-singlemodule-20260630` | `codex/abide-madaros-singlemodule-20260630` | no remote, no PR | `module_native_driver`, `lower`, `codegen_x86_linux`, `reloc` | Older broad WIP. Preserve until reviewed; high conflict surface. |
| `/tmp/sounio-madaros-fncount-20260701` | `fix/madaros-singlemodule-fncount-lowering-139` | no remote, no PR | `module_frontend`, `ir.sio`, `lower`, `codegen_x86_linux` | Function-count/lowering investigation. Salvage before retirement. |
| `/tmp/sounio-madaros-lower-known-test-20260702` | detached | detached | many compiler/IR/native files | Detached broad lower-known-test WIP. Must branch/archive before any deletion. |
| `/tmp/sounio-madaros-retire-lean-single-20260627` | `codex/madaros-retire-lean-single-20260628` | no remote, no PR | wrappers, AGENTS/CLAUDE, lean_single, compiler, IR, native | Large policy/architecture branch. Do not delete as part of greenline cleanup without separate review. |
| `/tmp/sounio-project-spine-slice-20260630` | `codex/project-spine-slice-20260630` | no remote, no PR | `module_native_driver` | Small project-spine slice. Salvage or compare to project-spine branch before cleanup. |
| `/workspace/sounio-gc-fix-20260701` | detached | detached | `self-hosted/ir/lower.sio` | Detached GC/lower WIP. Requires salvage branch before deletion. |
| `/tmp/sounio-bdf64-bridge` | `fix/bdf64-bridge` | no remote, no PR | `lower`, `codegen.sio`, `codegen_x86_linux` | F64 bridge work. Likely belongs to future F64 lane, not greenline cleanup. Preserve unless superseded by a new F64 branch. |

## Suggested Cleanup Order After Approval

1. Retire the already-merged greenline worktree:

   ```bash
   scripts/dev/madaros_worktree_cleanup_plan.sh --out-dir /tmp/madaros-cleanup-plan
   git -C /tmp/sounio-madaros-greenline-codex diff > /tmp/madaros-greenline-codex-leftover.patch
   git worktree remove /tmp/sounio-madaros-greenline-codex
   git branch -D work/madaros-greenline-codex
   ```

2. For detached wrapper/gate duplicates, create archival branches before removal:

   ```bash
   git -C /tmp/sounio-madaros-main-port-20260702 switch -c archive/madaros-main-port-20260702
   git -C /tmp/sounio-phase03-step5-clean switch -c archive/madaros-phase03-step5-clean-20260702
   git push origin archive/madaros-main-port-20260702 archive/madaros-phase03-step5-clean-20260702
   ```

3. For non-detached local branches with no remote, push explicit archive refs or
   move unique work into a new active lane before worktree removal.

4. Rerun:

   ```bash
   scripts/dev/madaros_readiness_status.sh --strict
   ```

## Blocker Contract

```text
Blocker-ID: BLK-20260703-codex-madaros-worktree-cleanup
Status: open
Severity: B3
Class: coordination-governance
Owner: operator + current cleanup agent after explicit approval
Lane: Madaros cleanup / worktree disposition
Worktree: multiple, see disposition table
Branch: multiple local branches and detached worktrees
Observed: `scripts/dev/madaros_readiness_status.sh --production-ready` passes, but
  `scripts/dev/madaros_readiness_status.sh --strict` reports
  `unallowed_critical_dirty=19`.
Expected: `--strict` has zero unallowed critical dirty worktrees, or every
  remaining critical dirty worktree is explicitly allowed and has an owner,
  branch, purpose, and next command.
Acceptance-Gate: `scripts/dev/madaros_readiness_status.sh --strict` exits with no
  worktree audit violation.
Evidence-Level: E3
Fallback-Path: none
Legacy-Kept: yes, until operator approves cleanup and archival push.
LLM-Offload: not-required
Next-Action: obtain explicit approval for the cleanup phase, then salvage/push
  before deleting any worktree or branch.
```
