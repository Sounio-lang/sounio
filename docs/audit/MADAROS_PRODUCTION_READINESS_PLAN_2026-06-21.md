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

- `origin/main`: `0d714587f46af0529b9f85010d62583550bef66d`
  (`tools(madaros): surface lowering diagnostics in readiness status`, #377).
- `main` CI after #377: success
  (<https://github.com/Sounio-lang/sounio/actions/runs/27909711839>).
- `Madaros Prebuilt Refresh` on `1d0dc6baa`: remote seed-decoupled build and
  `madaros_full_gate` succeeded. The promoted workspace local self-build still
  segfaults and is tracked separately as a workspace parity blocker.
- Canonical live blocker: GitHub issue #356.
- Protected dirty primary checkout: `/workspace/sounio`.
- `/workspace/sounio` is stale relative to `origin/main` and must not be used
  as evidence for current `main` behavior until it is explicitly reconciled.
- Current planning worktree rule: create a fresh isolated worktree from
  `origin/main` for each narrow readiness change.

The goal is not to make every old worktree disappear immediately. The goal is
to ensure that only current, gated, ownership-clear work can influence Madaros
production behavior.

## Resolution Plan

The audit resolves into this execution order:

1. Keep `origin/main` as the only production baseline and keep `/workspace/sounio`
   out of evidence while it is stale or dirty.
2. Keep Codex and Claude write sets serialized. Codex owns governance, probes,
   issue hygiene, and validation. Claude owns the active compiler/codegen lane
   unless ownership transfers explicitly.
3. Close `BLK-20260621-codex-source-elf-normal-bss` before returning to larger
   Root 2/SRET witnesses. The first passing proof must be the minimal current
   global read/write witnesses, not archived Root 2 notes.
4. Treat `BLK-20260621-codex-madaros-build-segfault` as promoted-workspace
   parity unless a compiler owner proves a semantic root. GitHub
   `Madaros Prebuilt Refresh` remains the authoritative seed-decoupled
   production build path while it is green.
5. After the BSS/global blocker changes behavior, update
   `scripts/ci/madaros_open_blockers_probe.sh` from known-open expectations to
   closed expectations and only then promote the witnesses into the regular
   source-to-ELF gate.
6. Merge only after the focused local gates and post-merge `main` CI agree with
   the blocker records.

The next actionable compiler command for the owning compiler lane is:

```bash
env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bash scripts/ci/madaros_open_blockers_probe.sh
```

The next actionable integration command for Codex or the release shepherd is:

```bash
scripts/dev/madaros_readiness_status.sh --strict
```

To include the active Claude compiler lane's read-only check-clean threshold:

```bash
scripts/dev/madaros_readiness_status.sh --check-compiler-lane
```

The required BSS closure evidence is:

- `global_read_exit4` exits `4` in `normal`, `native_v2`, and `build`.
- `global_store_exit7` exits `7` at least in `build`, then in broader modes
  only when the compiler owner promotes them.
- `scripts/ci/madaros_source_to_elf_gate.sh` remains green.
- `main` CI and the relevant Madaros remote build path remain green after merge.

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
| Open blocker probe | `scripts/ci/madaros_open_blockers_probe.sh` keeps known-open direct-call, BSS, and local workspace self-build parity witnesses executable without promoting them to production manifests | #363/#367/#369 plus post-merge `main` CI |
| Governance control surfaces | Worktree audit treats agent contracts and governance scripts as critical surfaces | #365 plus post-merge `main` CI |
| Production readiness plan | Current-main readiness plan is committed and merged | #366 plus post-merge local verification |
| Worktree disposition queue | Current worktree/branch/PR disposition queue is committed and merged | #372 plus post-merge `main` CI |
| Readiness status command | `scripts/dev/madaros_readiness_status.sh` prints the current baseline, GitHub state, audit gate, blockers, and next gates | #373 plus post-merge `main` CI |
| Compiler-lane status command | `scripts/dev/madaros_readiness_status.sh --check-compiler-lane` inspects the active Claude lane and runs read-only check-clean probes with summarized logs | #374 plus post-merge `main` CI |
| Lowering-diagnostic status pointer | `scripts/dev/madaros_readiness_status.sh` prints the `--diagnose-lowering` command needed before editing the BSS/global compiler path | #377 plus post-merge `main` CI |
| Direct-call argument ABI | `call_arg_id_exit42` exits 42 in normal/native_v2/build and is now a regression control, not an open blocker | #367 plus post-merge local verification |

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
   - `self-hosted/compiler/main.sio`
   - `self-hosted/native/lower_ir.sio`
   - `self-hosted/native/codegen.sio`
   - `self-hosted/native/codegen_x86_linux.sio`
5. The old archive-derived Root 2/SRET blocker is superseded by current
   source-to-ELF readiness blockers in #356. Keep Root 2/SRET gates as
   regression guards, but do not treat them as the only production blocker.

Current source-to-ELF compiler blocker:

```text
Blocker-ID: BLK-20260621-codex-source-elf-normal-bss
Status: classified
Severity: B1
Class: compiler-semantics
Owner: Claude compiler/codegen lane unless ownership transfers explicitly
Lane: Madaros source-to-ELF global/BSS lowering
Worktree: /workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b
Branch: worktree-agent-adc1cd8b9d52ba53b
Canonical-Issue: https://github.com/Sounio-lang/sounio/issues/356
Files-Owned: self-hosted/compiler/main.sio, self-hosted/native/codegen.sio, self-hosted/native/codegen_x86_linux.sio, self-hosted/native/lower_ir.sio, self-hosted/native/suite.sio
Files-Read-Only: scripts/ci/madaros_open_blockers_probe.sh, scripts/ci/madaros_source_to_elf_gate.sh, scripts/dev/madaros_readiness_status.sh
Do-Not-Touch: self-hosted/compiler/main.sio, self-hosted/native/codegen.sio, self-hosted/native/codegen_x86_linux.sio, self-hosted/native/lower_ir.sio, self-hosted/native/suite.sio unless ownership transfers explicitly
Repro: env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN bash scripts/ci/madaros_open_blockers_probe.sh --diagnose-lowering
Observed: global_read_exit4 normal/native_v2/build => compile_rc_139; global_store_exit7 build => compile_rc_139
Expected: global_read_exit4 exits 4 and global_store_exit7 exits 7 from emitted ELF, without raw Madaros compile segfault
Acceptance-Gate: scripts/ci/madaros_open_blockers_probe.sh is updated from known-open to closed-BSS expectations and passes; scripts/ci/madaros_source_to_elf_gate.sh also passes
Evidence-Level: E3
Evidence: https://github.com/Sounio-lang/sounio/issues/356#issuecomment-4762337445
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: not-required
Next-Action: inspect the standard source-to-ELF/native emission path after successful typecheck and merged-IR capture, especially global/BSS symbol allocation or emission
```

Current promoted-workspace parity blocker:

```text
Blocker-ID: BLK-20260621-codex-madaros-build-segfault
Status: classified
Severity: B2
Class: platform-resource
Owner: integration shepherd / workspace-runtime lane unless compiler owner proves semantic root
Lane: promoted workspace local self-build parity
Worktree: /workspace/sounio
Branch: main
Canonical-Issue: https://github.com/Sounio-lang/sounio/issues/356
Files-Owned: none by Codex
Files-Read-Only: scripts/ci/build_modular_madaros.sh, scripts/dev/souc-build-lock.sh, self-hosted/compiler/main.sio
Do-Not-Touch: bin/madaros, bin/souc, self-hosted/compiler/main.sio unless a focused parity lane is opened
Repro: env -u SOUC_BIN -u SOUNIO_SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN bash scripts/ci/build_modular_madaros.sh /tmp/sounio-madaros-self-build-check/madaros
Observed: clean promoted-workspace build exits rc=139 while GitHub Madaros Prebuilt Refresh build+gate succeeds on the same commit
Expected: local promoted workspace build agrees with remote seed-decoupled build, or docs/gates explicitly mark local workspace self-build as non-authoritative for production readiness
Acceptance-Gate: local build_modular_madaros.sh passes in a clean workspace worktree, or release/readiness docs classify GitHub prebuilt refresh as authoritative and local workspace self-build as a platform/parity blocker
Evidence-Level: E4
Evidence: https://github.com/Sounio-lang/sounio/issues/356#issuecomment-4763092558
Fallback-Path: GitHub Madaros Prebuilt Refresh remains authoritative while green because local promoted workspace self-build is classified as platform/parity
Legacy-Kept: yes
LLM-Offload: not-required
Next-Action: investigate workspace/runtime parity, not broad compiler bootstrap failure, unless a compiler owner proves a semantic root
```

Exit:

- One compiler owner, one write set, check-clean compiler files, and one
  acceptance gate for each active compiler/bootstrap blocker.

## Phase 2 — Close Current Source-To-ELF Blockers

Owner: compiler lane.

Actions:

1. Do not start from archive notes. Start from `origin/main` and the current
   #356 blocker records.
2. Treat `BLK-20260621-codex-source-elf-direct-call-arg` as closed by current
   `main` evidence. It is now a passing control in
   `scripts/ci/madaros_open_blockers_probe.sh`.
3. Fix `BLK-20260621-codex-source-elf-normal-bss`. Current evidence shows raw
   Madaros compile segfault before ELF production for global read/store in
   normal/native_v2/build coverage.
4. Treat `BLK-20260621-codex-madaros-build-segfault` as a promoted-workspace
   parity blocker. Current remote `Madaros Prebuilt Refresh` evidence proves
   that the seed-decoupled build and `madaros_full_gate` can succeed on GitHub
   for `1d0dc6baa`; current local workspace evidence still reproduces
   `build_rc_139`.
5. Keep failing witnesses out of `tests/madaros/source_to_elf/manifest.tsv`
   until the open-blocker probe reports changed behavior and the blocker record
   is updated or closed.

Current localization for the BSS/global blocker:

- Direct raw-ELF execution of `bin/madaros-linux-x86_64` reproduces `rc=139`
  for `global_read_exit4.sio` and `global_store_exit7.sio`; the wrapper is not
  required to trigger the crash.
- The raw-ELF runs reach `Type check complete for module 0` before crashing.
- `--probe-frontend`, `--probe-compile-capture`, and the earlier
  `--probe-load-ir-trace` path pass for the BSS witnesses.
- `--probe-machine-ir` is not discriminating because it also fails for controls.
- `--probe-native-streaming` is not authoritative for this blocker because it
  follows a different probe path: it segfaults on a control and reports E137 on
  globals even though the standard check/typecheck path succeeds.
- Next code owner should inspect the standard source-to-ELF/native emission path
  after typecheck / IR capture. Do not spend the first fix pass in parser,
  typecheck, load-IR, or the native-streaming probe.

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
- `global_store_exit7` exits 7 at least in build mode, then in any broader
  source-to-ELF mode promoted by the compiler owner.
- GitHub `Madaros Prebuilt Refresh` build+gate remains green for the production
  build path.
- The local workspace self-build parity blocker is either fixed, explicitly
  scoped as non-authoritative for production readiness, or tracked outside the
  compiler-semantics lane.
- The open-blocker probe is updated or removed because the witnesses no longer
  match known-open behavior.

## Phase 3 — Land Compiler Repair

Owner: compiler lane, with one integration shepherd.

Actions:

1. Keep the patch narrow to compiler/runtime surfaces needed for the failing
   repros.
2. Add focused regression tests for the exact source-to-ELF BSS/global read and
   write cases. Keep direct-call argument as a regression control.
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
- Local workspace self-build parity blocker closed or explicitly separated from
  the production build path with GitHub prebuilt refresh named as authoritative.

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

env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
  bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros
```

For Codex/integration:

```bash
cd <current-readiness-worktree>
git status --short --branch
scripts/dev/madaros_readiness_status.sh --strict
scripts/dev/madaros_readiness_status.sh --check-compiler-lane
bash scripts/dev/check_docs_registry.sh
bash scripts/dev/check_docs_consistency.sh
SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_RE='^(/workspace/sounio|/workspace/sounio-effects|/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b|<current-readiness-worktree>)$' \
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
