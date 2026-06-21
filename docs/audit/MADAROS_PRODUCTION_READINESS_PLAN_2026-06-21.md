<!-- docs:meta
topic_id: repo.docs.audit.madaros-production-readiness-plan-2026-06-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-production-readiness-plan-2026-06-21
-->

# Madaros Production Readiness Plan — 2026-06-21

## Scope

This is the execution plan for turning the current Madaros audit into a
production-readiness program. It is intentionally not a compiler patch.

The plan coordinates:

- primary-checkout reconciliation,
- the active Claude compiler lane,
- Madaros Root 2/SRET/archive evidence,
- Slurm/foundry validation,
- merge and release gates.

Current baseline:

- `origin/main`: `91551953cedefb780cba9fe7ebd61c8a8a5b301d`
  (`Merge pull request #355 from Sounio-lang/codex/madaros-root2-gate-target`).
- `main` CI run `27892902187`: success.
- Canonical live blocker: GitHub issue #356.
- Protected dirty primary checkout: `/workspace/sounio`.
- Current planning worktree: create a fresh isolated worktree from `origin/main`
  for each narrow readiness change.

The goal is not to make every old worktree disappear immediately. The goal is
to ensure that only current, gated, ownership-clear work can influence Madaros
production behavior.

## Production-Ready Definition

Madaros is production-ready only when all of these are true:

1. `origin/main` is the only production baseline.
2. The primary checkout is no longer treated as source truth while it is stale
   or dirty.
3. No compiler file has two active writers.
4. `bin/madaros` and `bin/souc` resolve through the canonical resolver paths.
5. Root 2/SRET/enum/method-call evidence is refreshed on a current branch, not
   replayed from archived notes.
6. Focused local gates pass without environment contamination.
7. Heavy validation runs through foundry/Slurm rather than the live workspace.
8. Main CI is green after every merge.
9. Any remaining blocker has a `.claude/PARALLEL_BLOCKER_CONTRACT.md` record.

## Already Closed

| Area | Result | Evidence |
|---|---|---|
| Worktree governance | CI check mode is on `main` | #346 |
| Codex docs contract | `ctx7`/AGENTS contract is on `main` | #347 |
| Primary checkout reconciliation | Archive buckets are classified | #348 |
| `examples/hello.sio` | Prints and exits cleanly | #349 plus post-merge `main` CI |
| Archived Madaros docs | Raw notes are triaged, not promoted | #350 plus post-merge `main` CI |
| Bucket D scripts | Raw local scripts are blocked from promotion | #351 plus post-merge `main` CI |
| Root 2 operator gate | Acceptance probes are packaged but intentionally red while blocker is open | #354 plus post-merge `main` CI |
| Root 2 target-worktree gate | Current `main` can run the gate against older active compiler lanes via `--root` | #355 plus post-merge `main` CI |

## Phase 0 — Freeze Source Of Truth

Owner: Codex, governance lane.

Write set:

- docs and audit files only.

Actions:

1. Treat `/workspace/sounio` as protected dirty state until explicitly cleaned
   by the human author.
2. Require all new production work to start from `origin/main` in a clean
   isolated worktree.
3. Keep the worktree audit gate green.
4. Keep a written disposition for every archive bucket.

Gate:

```bash
SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_RE='^(/workspace/sounio|/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b|<current-lane>)$' \
  scripts/dev/worktree_branch_audit.sh --check
```

Exit:

- No undocumented local artifact is eligible for direct replay.

## Phase 1 — Serialize Compiler Ownership

Owner: compiler lane, currently Claude for active compiler files.

Codex write set:

- none in compiler files.

Claude-owned active compiler files:

- `self-hosted/compiler/main.sio`
- `self-hosted/native/codegen.sio`
- `self-hosted/native/codegen_x86_linux.sio`
- `self-hosted/native/lower_ir.sio`
- `self-hosted/native/suite.sio`

Actions:

1. Claude continues the active Root 2/SRET compiler repair.
2. Codex may inspect and run read-only gates, but must not edit the owned files.
3. Any proposed Codex compiler work must first become an
   `ownership-conflict` or ownership-transfer record.
4. The archive-derived blocker remains:

```text
Blocker-ID: MADAROS-ROOT2-SRET-ARCHIVE-TRIAGE-2026-06-21
Status: reproduced
Severity: B1
Class: compiler-semantics
Owner: compiler lane
Lane: Madaros Root 2/SRET/enum/method-call repair
Canonical-Issue: https://github.com/Sounio-lang/sounio/issues/356
Files-Owned: self-hosted/compiler/main.sio, self-hosted/native/codegen.sio, self-hosted/native/codegen_x86_linux.sio, self-hosted/native/lower_ir.sio, self-hosted/native/suite.sio
Acceptance-Gate: scripts/ops/madaros_root2_acceptance_gate.sh --root <compiler-worktree> passes without --allow-fail from the branch that changes compiler code
Evidence-Level: E1
Next-Action: compiler owner closes native_v2_enum_match and sret_forwarding segfaults, then reruns the gate without --allow-fail before any PR
```

Exit:

- One compiler owner, one write set, one acceptance gate.

## Phase 2 — Refresh Root 2 Evidence

Owner: compiler lane.

Actions:

1. Rebuild or select the current branch compiler using canonical resolver
   routing.
2. Run fresh repros for:
   - enum variant registration,
   - method-call lowering,
   - SRET/tail-array return,
   - Root 2 function-count behavior.
3. Record whether each failure is:
   - fixed by the active Root 2/SRET patch,
   - still failing as compiler semantics,
   - a harness-routing issue,
   - stale archive evidence.

Required command hygiene:

```bash
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bash scripts/run_sio_test_suite.sh <focused-suite> --verbose
```

Packaged operator gate:

```bash
scripts/ops/madaros_root2_acceptance_gate.sh
```

From a current checkout, the same gate can target an older active compiler lane
without copying files into that lane:

```bash
scripts/ops/madaros_root2_acceptance_gate.sh \
  --root /workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b
```

Use `--allow-fail` only for diagnostic snapshots while the blocker is still
open; a merge-ready compiler lane must run the gate without `--allow-fail`.

Exit:

- Archive notes are replaced by current evidence or explicitly closed as stale.

## Phase 3 — Land Compiler Repair

Owner: compiler lane, with one integration shepherd.

Actions:

1. Keep the patch narrow to compiler/runtime surfaces needed for the failing
   repros.
2. Add focused regression tests for the exact Root 2/SRET/enum/method cases.
3. Run focused gates locally.
4. Run source-bootstrap/native gates as required by the touched surfaces.
5. Open one PR with the blocker record in the PR body.

Minimum local gates:

```bash
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bash scripts/run_sio_test_suite.sh <new-focused-tests> --verbose

scripts/ops/madaros_root2_acceptance_gate.sh

bash scripts/ci/build_native_souc.sh
```

Remote gate:

- PR CI green.
- Post-merge `main` CI green.

Exit:

- Root 2 blocker closed at E3/E4 or narrowed with a new blocker record.

## Phase 4 — Prove Madaros As Default Path

Owner: integration shepherd after compiler PR lands.

Actions:

1. Verify `scripts/lib/resolve_madaros.sh`, `scripts/lib/resolve_souc.sh`,
   `bin/madaros`, and `bin/souc` agree on the intended default.
2. Run the standard suite without environment overrides.
3. Re-run with environment variables explicitly unset to catch contamination.
4. Submit heavy validation through foundry/Slurm.

Workspace-safe gates:

```bash
bin/souc info
bash scripts/run_sio_test_suite.sh hello --verbose
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bash scripts/run_sio_test_suite.sh hello --verbose
```

Heavy validation:

```bash
/home/devsounio/projects/sounio/sounio-forge submit full-compiler --source wip --gpu auto
```

If the host path is not mounted in the workspace, leave a handoff for a
host/control-plane agent instead of running stress locally.

Exit:

- Madaros default routing is proven by clean local gates plus remote/foundry
  evidence.

## Phase 5 — Clean Or Archive Remaining Worktrees

Owner: integration shepherd plus human author for destructive cleanup.

Actions:

1. List all worktrees and classify each as:
   - active lane,
   - merged and removable,
   - stale but preserved,
   - unknown owner.
2. Remove only worktrees owned by the current lane and already merged.
3. Convert unknown or stale compiler lanes into blocker/investigation records.
4. Do not reset or clean `/workspace/sounio` without explicit human approval.

Inspection commands:

```bash
git worktree list --porcelain
gh pr list --state open --limit 50 --json number,title,headRefName,baseRefName,isDraft,mergeable,url
gh run list --branch main --limit 10 --json databaseId,workflowName,headSha,status,conclusion,event,createdAt,url
```

Exit:

- Every remaining lane has an owner, a purpose, and a next command.

## Immediate Next Commands

For the compiler owner:

```bash
cd /workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b
git status --short --branch
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bash scripts/run_sio_test_suite.sh <enum-method-root2-focused-suite> --verbose
```

For Codex/integration:

```bash
cd /tmp/sounio-madaros-production-plan
git status --short --branch
bash scripts/dev/check_docs_registry.sh
bash scripts/dev/check_docs_consistency.sh
SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_RE='^(/workspace/sounio|/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b|/tmp/sounio-madaros-production-plan)$' \
  scripts/dev/worktree_branch_audit.sh --check
```

## Stop Rules

Stop and create a blocker if:

- a Codex lane needs to edit a Claude-owned compiler file,
- Root 2/SRET acceptance diverges from canonical issue #356,
- a gate passes only because of `SOUC_BIN`, `MADAROS_BIN`, or stdlib path
  contamination,
- an archived Madaros note contradicts fresh current-branch evidence,
- `/workspace/sounio` is needed as source truth before it is explicitly
  reconciled,
- a post-merge `main` CI run goes red.

## Result

This converts the audit into a resolution program:

- docs/tooling buckets are closed or blocked from raw promotion,
- compiler repair is serialized to the active compiler owner,
- current evidence replaces stale archive evidence,
- production readiness is gated by resolver, local, CI, and foundry/Slurm
  proof.
