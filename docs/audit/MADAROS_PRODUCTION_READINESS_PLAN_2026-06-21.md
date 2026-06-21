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
- Madaros source-to-ELF, Root 2/SRET, and archive evidence,
- Slurm/foundry validation,
- merge and release gates.

Current baseline:

- `origin/main`: `302d1bc9766f6e185775f8fb5869f23d66e1e297`
  (`ci(governance): audit governance control surfaces (#365)`).
- `main` CI run `27901532987`: success.
- Canonical live blocker: GitHub issue #356.
- Protected dirty primary checkout: `/workspace/sounio`.
- `/workspace/sounio` is stale relative to `origin/main` and must not be used
  as evidence for current `main` behavior until it is explicitly reconciled.
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
| Root 2/BSS lowerer floor | BSS globals lower through the stable mut path; native_v2/build global witnesses are healthy | #362 plus post-merge `main` CI |
| Open blocker probe | `scripts/ci/madaros_open_blockers_probe.sh` keeps known-open direct-call/BSS witnesses executable without promoting them to the source-to-ELF manifest | #363 plus post-merge `main` CI |
| Governance control surfaces | Worktree audit treats agent contracts and governance scripts as critical surfaces | #365 plus post-merge `main` CI |

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
SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_RE='^(/workspace/sounio|/workspace/sounio-effects|/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b|<current-lane>)$' \
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

1. Claude continues the active compiler/codegen repair.
2. Codex may inspect and run read-only gates, but must not edit the owned files.
3. Any proposed Codex compiler work must first become an
   `ownership-conflict` or ownership-transfer record.
4. The active Claude lane must become check-clean before any source-to-ELF
   blocker result from that lane is treated as semantic evidence. Read-only
   checks on 2026-06-21 still returned `verdict=1` for:
   - `self-hosted/native/lower_ir.sio`
   - `self-hosted/native/codegen.sio`
   - `self-hosted/native/codegen_x86_linux.sio`
5. The old archive-derived Root 2/SRET blocker is superseded by current
   source-to-ELF readiness blockers in #356. Keep Root 2/SRET gates as
   regression guards, but do not treat them as the only production blocker.

Current primary compiler blocker:

```text
Blocker-ID: BLK-20260621-codex-source-elf-direct-call-arg
Status: owned
Severity: B1
Class: compiler-semantics
Owner: Claude compiler/codegen lane unless ownership transfers explicitly
Lane: Madaros source-to-ELF direct-call argument ABI
Canonical-Issue: https://github.com/Sounio-lang/sounio/issues/356
Files-Owned: self-hosted/compiler/main.sio, self-hosted/native/codegen.sio, self-hosted/native/codegen_x86_linux.sio, self-hosted/native/lower_ir.sio, self-hosted/native/suite.sio
Observed: call_arg_id_exit42 exits 0 in normal/native_v2/build on current evidence artifacts
Expected: call_arg_id_exit42 exits 42 in normal/native_v2/build
Acceptance-Gate: scripts/ci/madaros_source_to_elf_gate.sh direct_call passes and scripts/ci/madaros_open_blockers_probe.sh no longer reports the direct-call witness as still_open
Evidence-Level: E4
Next-Action: make the Claude lane check-clean, then inspect/fix the core source-to-ELF codegen path around parameter spill/load/store
```

Exit:

- One compiler owner, one write set, check-clean compiler files, and one
  acceptance gate for the direct-call argument ABI blocker.

## Phase 2 — Close Current Source-To-ELF Blockers

Owner: compiler lane.

Actions:

1. Do not start from archive notes. Start from `origin/main` and the current
   #356 blocker records.
2. Fix `BLK-20260621-codex-source-elf-direct-call-arg` first. It fails across
   normal, native_v2, and build modes and blocks the official source-to-ELF
   gate.
3. Only after direct-call argument ABI passes, fix
   `BLK-20260621-codex-source-elf-normal-bss`. Current evidence shows Root 2
   global read/store is healthy in native_v2/build, while normal mode still
   segfaults for global read.
4. Keep failing witnesses out of `tests/madaros/source_to_elf/manifest.tsv`
   until the open-blocker probe reports changed behavior and the blocker record
   is updated or closed.

Required command hygiene:

```bash
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bash scripts/run_sio_test_suite.sh <focused-suite> --verbose
```

Known-open blocker probe:

```bash
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  scripts/ci/madaros_open_blockers_probe.sh
```

Official source-to-ELF gate:

```bash
bash scripts/ci/madaros_source_to_elf_gate.sh
```

Legacy Root 2/SRET acceptance remains useful as a regression guard, but it is
no longer the only production-readiness gate:

```bash
scripts/ops/madaros_root2_acceptance_gate.sh
```

Exit:

- `call_arg_id_exit42` exits 42 in normal/native_v2/build.
- `direct_call` passes in `scripts/ci/madaros_source_to_elf_gate.sh`.
- `global_read_exit4` exits 4 in normal/native_v2/build.
- The open-blocker probe is updated or removed because the witnesses no longer
  match known-open behavior.

## Phase 3 — Land Compiler Repair

Owner: compiler lane, with one integration shepherd.

Actions:

1. Keep the patch narrow to compiler/runtime surfaces needed for the failing
   repros.
2. Add focused regression tests for the exact source-to-ELF direct-call
   argument and normal-mode BSS cases.
3. Run focused gates locally.
4. Run source-bootstrap/native gates as required by the touched surfaces.
5. Open one PR with the blocker record in the PR body.

Minimum local gates:

```bash
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bash scripts/run_sio_test_suite.sh <new-focused-tests> --verbose

env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  scripts/ci/madaros_open_blockers_probe.sh

bash scripts/ci/madaros_source_to_elf_gate.sh

bash scripts/ci/build_native_souc.sh
```

Remote gate:

- PR CI green.
- Post-merge `main` CI green.

Exit:

- Direct-call and normal-BSS blockers closed at E3/E4 or narrowed with new
  blocker records.

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
  bin/souc check self-hosted/native/lower_ir.sio
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bin/souc check self-hosted/native/codegen.sio
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bin/souc check self-hosted/native/codegen_x86_linux.sio

# After the lane is check-clean:
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  scripts/ci/madaros_open_blockers_probe.sh
```

For Codex/integration:

```bash
cd /tmp/sounio-madaros-prod-plan
git status --short --branch
bash scripts/dev/check_docs_registry.sh
bash scripts/dev/check_docs_consistency.sh
SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_RE='^(/workspace/sounio|/workspace/sounio-effects|/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b|/tmp/sounio-madaros-prod-plan)$' \
  scripts/dev/worktree_branch_audit.sh --check
```

## Stop Rules

Stop and create a blocker if:

- a Codex lane needs to edit a Claude-owned compiler file,
- source-to-ELF blocker behavior diverges from canonical issue #356,
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
