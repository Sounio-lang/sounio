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

Key completed PRs in this ledger series:

- PR #586 merged into `canon/madaros-greenline` at
  `bf46bda919596ce71b8fc35dc29cb3a31ff01d7b`.
- PR #587 merged into `canon/madaros-greenline` at
  `4419f2097913de65e36665d442a65c6f64d5f3ca`.
- PR #588 merged the first cleanup ledger at
  `2bfe06b4cea88e86c0875e4c9069526c2292e9fc`.
- PR #589 merged the non-destructive cleanup planner at
  `95d689462126b34f57fc25d83b45dc8dbf3c01cc`.
- PR #590 merged readiness handoff wiring at
  `0fdaa2b1c740c6e04b533eee0b90e335b3bf728f`.
- PR #591 merged planner evidence columns plus live-audit shape validation at
  `3df24a04af6b35e4f710663accb03f9f97825b0a`.
- PR #592 synchronized the status docs after PR #591 at
  `246cd44e59026b8ca6fecd72336e631036015f07`.
- PR #593 removed stale live-tip wording from the status docs at
  `6f0d9ec8c5289f717d668f773d2e54a8d8d9bbdb`.
- Later status-only PRs may advance `canon/madaros-greenline`; use
  `git log --oneline --merges origin/canon/madaros-greenline` for the live
  merge list instead of treating this section as exhaustive.
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
As of PR #591, `scripts/ci/madaros_operational_contract_gate.sh` also exercises
the planner on a deterministic fixture and on the live worktree audit stream,
checking the 22-column TSV shape, evidence-comment coverage, and absence of
uncommented mutating commands.

`scripts/dev/madaros_readiness_status.sh --strict` prints a
`cleanup_plan_command=...` line after the worktree audit section, using the
exact audit TSV path from that run. That is the operator handoff from red
strict-audit output to this cleanup planner.

For a portable non-destructive evidence bundle before the approval decision,
run:

```bash
scripts/dev/madaros_worktree_salvage_packet.sh --tarball
```

That wraps the cleanup planner and writes per-worktree status, tracked/staged
binary diffs, short logs, untracked-file manifests, size indexes, and an
optional `DIR.tar.gz` + sha256. It does **not** push branches, remove worktrees,
delete files, reset, or clean; it is evidence for the operator decision, not the
cleanup itself.

After reviewing the packet, create the machine-readable approval template:

```bash
scripts/dev/madaros_worktree_cleanup_approval.sh template \
  --plan-tsv /tmp/madaros-cleanup-plan/madaros-cleanup-plan.tsv \
  --out-dir /tmp/madaros-cleanup-approval
```

The template defaults every row to `decision=hold`. The operator must edit rows
to one of `approve_salvage_remove`, `approve_discard_remove`, or
`approve_remove_clean`, and must fill `approver`, `approved_utc`, and
`approval_id`. Validate and render the reviewed commands with:

```bash
scripts/dev/madaros_worktree_cleanup_approval.sh validate \
  --manifest-tsv /tmp/madaros-cleanup-approval/madaros-cleanup-approval.tsv
scripts/dev/madaros_worktree_cleanup_approval.sh render \
  --manifest-tsv /tmp/madaros-cleanup-approval/madaros-cleanup-approval.tsv \
  --out-dir /tmp/madaros-cleanup-approved
```

Rendered mutating commands stay commented unless the renderer is invoked with
`--allow-mutating-output` and
`SOUNIO_MADAROS_CLEANUP_APPROVAL=I_ACCEPT_PUSH_BEFORE_DELETE`. This preserves
the explicit approval boundary while making the post-approval command stream
reviewable and reproducible.

## Current Blocker To `--strict`

Readiness production proof is green, but the stricter worktree audit still
reports dirty critical worktrees. The exact `total` and `critical_vs_base`
counts drift as clean coordination worktrees are created or removed; the live
authority is the repro command below. The stable blocker is still
`unallowed_critical_dirty=19`.

Sample observed after PR #595 while generating the approval packet:

```text
total=64
dirty=37
critical_dirty=21
unallowed_critical_dirty=19
critical_vs_base=48
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

## Approval Packet Snapshot - post-PR #595

Generated non-destructively with:

```bash
scripts/dev/madaros_worktree_cleanup_plan.sh --out-dir /tmp/madaros-cleanup-approval-packet-plan
```

Planner categories:

```text
planned_worktrees=19
active_other_lane_wip=8
detached_risky=4
greenline_leftover=2
stale_local_temp=5
```

This snapshot is an approval aid, **not** approval. Some clean coordination
companions are listed even though they are not part of the 19 strict-blocker
rows, because removing them after approval would reduce local clutter. The
approval decision can be split into these buckets:

| Bucket | Items in packet | Operator decision needed |
|---|---:|---|
| Clean status/coordination companions already represented by merged PRs | 4 | After explicit approval, remove local worktrees/branches only if `git status --short` is still clean. |
| Dirty greenline/status leftovers | 2 strict rows + 1 companion | Review patches; either salvage/commit/push an archive branch or explicitly discard before removal. |
| Detached wrapper/gate duplicates | 2 | Archive branch plus patch required before removal (`madaros-main-port`, `phase03-step5-clean`). |
| Detached broad/lower/GC probes | 2 | Preserve or archive as future lowering/GC evidence; do not treat as wrapper cleanup. |
| Active/future F64/Lane V/variance work | 5 audited paths | Preserve as active lanes or salvage under explicit branch names; do not delete as cleanup residue. |
| Stale local temp branches | 5 | Review unique commits and dirty patches; push archive/salvage refs before any retirement. |

Read-only subagent audit (2026-07-03) refined the first cleanup pass:

- Clean local status/coordination worktrees: `/tmp/sounio-madaros-cleanup-evidence`,
  `/tmp/sounio-madaros-status-current`, `/tmp/sounio-madaros-post-593-status`,
  `/tmp/sounio-madaros-status-no-pr-range`. These were clean at audit time and
  are likely removal candidates after explicit approval.
- Dirty greenline/status leftovers: `/tmp/sounio-madaros-greenline-codex`
  (2 compiler files, 426 insertions / 34 deletions) and
  `/tmp/sounio-madaros-greenline-status` (`worktree_audit.tsv` untracked).
- Detached wrapper/gate duplicates: `/tmp/sounio-madaros-main-port-20260702`
  and `/tmp/sounio-phase03-step5-clean` overlap PR #586/#587 wrapper/gate paths
  (`bin/madaros`, `scripts/ci/madaros_full_gate.sh`,
  `scripts/lib/resolve_madaros.sh`, `scripts/dev/madaros_two_gate.sh`) and need
  archive/patch before cleanup.
- Preserve/salvage future-lane work: `/workspace/sounio-greenfirst`,
  `/tmp/sounio-phase03-4e68`, `/tmp/sounio-phase03-step5-fix`,
  `/tmp/sounio-phase03-step5-lowerfix-min`, and `/tmp/sounio-bdf64-bridge`.
  These hold green-first, Phase03, F64, or Lane V/variance-looking compiler work.
- Detached broad/lower probes: `/tmp/sounio-madaros-lower-known-test-20260702`
  and `/workspace/sounio-gc-fix-20260701` do not overlap the wrapper/gate PRs
  directly, but contain lowering/GC evidence and should be archived if retired.

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
