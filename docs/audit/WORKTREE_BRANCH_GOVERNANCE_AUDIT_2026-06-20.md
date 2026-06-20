<!-- docs:meta
topic_id: repo.docs.audit.worktree-branch-governance-audit-2026-06-20
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.worktree-branch-governance-audit-2026-06-20
-->

# Worktree / Branch Governance Audit — 2026-06-20

## Scope

Read-only audit of the current Sounio workspace/worktree/branch state after the
Madaros #328 merge and during the Root2 compiler lane. No branches were checked
out, reset, cleaned, rebased, or deleted.

The audit target is governance, not semantic correctness: identify where active
compiler evidence can be confused by dirty worktrees, stale branches, duplicate
compiler artifacts, and PRs that overlap serialized files.

## Current Status After Follow-Through

This audit has been acted on. The original snapshot below is preserved because
it explains the failure mode, but the current authoritative cleanup status is:

- Current `origin/main`: `b73a11f5838f138f3d06b591f832e98d501214f3`
- #331 merged and production default refreshed:
  - #331 merge commit: `c5d0366d40490c18b927ef0d03cf98705697b6c2`
  - prebuilt refresh commit: `b73a11f5838f138f3d06b591f832e98d501214f3`
  - refreshed `bin/madaros-linux-x86_64` SHA256:
    `04a6982414a13c17b52a7df1f9b7e75ae33193f26dca11b50a3754d9c7b4216d`
  - GitHub runs:
    - #331 PR CI `27879637381`: pass
    - post-#331 main CI `27879826808`: pass
    - #332 main CI `27880006182`: pass
    - Madaros prebuilt refresh `27880089845`: pass
  - Clean `origin/main` smoke through default `bin/madaros`:
    `madaros_enum_gate.sh`: pass; `madaros_loop_gate.sh`: pass.
- Stale/conflicting compiler-adjacent PRs were quarantined with comments:
  - #232: <https://github.com/Sounio-lang/sounio/pull/232#issuecomment-4759602432>
  - #296: <https://github.com/Sounio-lang/sounio/pull/296#issuecomment-4759602426>
  - #313: <https://github.com/Sounio-lang/sounio/pull/313#issuecomment-4759602427>
- Safe cleanup completed:
  - `git worktree prune --verbose` removed 7 stale records whose gitdir files
    pointed to missing locations.
  - Clean worktrees for already-merged PRs #328, #330, and #332 were removed:
    `/tmp/sounio-madaros-boxnew-review`,
    `/tmp/sounio-mcp-contract-cleanup`,
    `/tmp/sounio-whereami-repair`.
  - Two additional clean worktrees whose HEADs are already ancestors of
    `origin/main` were removed:
    `/workspace/sounio-madaros-operational-contract`,
    `/workspace/sounio-website-ll`.
- Post-cleanup inventory:
  - raw TSV: `/tmp/sounio-worktree-audit-after-clean-ancestors-20260620.tsv`
  - total worktrees: 73
  - dirty worktrees: 58
  - prunable worktree records: 0
  - dirty critical worktrees: 8
  - critical-diff vs `origin/main`: 51

The remaining blocker is therefore narrower: owner-by-owner disposition of the
8 dirty critical worktrees. It is no longer a generic unclassified PR/prebuilt
blocker.

## Original Snapshot

- Primary path: `/workspace/sounio`
- Primary local branch: `main`
- Primary local HEAD: `1689e2d7a389162f898356e4ef7bfc613eec2926`
- Remote `origin/main`: `aafe4b7bb99b530dcfe2ff4ce78756dfe72ffbcc`
- Primary status: dirty and behind `origin/main` by 5 commits
- Raw inventory TSV: `/tmp/sounio-worktree-audit-20260620T175105.tsv`
- Reusable audit script added: `scripts/dev/worktree_branch_audit.sh`

## Headline Findings

1. `/workspace/sounio` must not be treated as clean `main`.

   The primary checkout is on local `main`, but it is behind `origin/main` and
   has local modifications, including compiler-adjacent files:

   - `artifacts/omega/agent_handoff.log.md`
   - `bin/madaros`
   - `self-hosted/native/machine_ir.sio`

   Any claim like "main's Madaros is broken" or "#328 regressed the compiler"
   must be reproduced from a clean `origin/main` worktree or from an identified
   artifact SHA, not from `/workspace/sounio`.

2. Worktree sprawl is now a governance risk.

   Inventory totals:

   - total worktrees: 84
   - dirty worktrees: 65
   - prunable worktrees: 7
   - worktrees with dirty critical compiler/CI files: 8
   - worktrees whose branch differs from `origin/main` in critical compiler/CI
     paths: 52
   - open PRs represented by current worktrees: 5

3. Compiler evidence is fragmented across multiple Madaros identities.

   The current environment has at least these identity surfaces:

   - `bin/madaros` launcher
   - `bin/madaros-linux-x86_64` checked-in prebuilt
   - `artifacts/self-hosted/madaros` local raw build artifact
   - GitHub Actions uploaded artifacts, for example `madaros-built`
   - stale raw binaries in old worktrees

   A run ID alone is not evidence. Compiler reports must include source ref,
   source SHA, binary path, binary SHA256, execution mode (`launcher` or `raw`),
   stdlib path, gate name, and run ID.

4. The Root2 lane is active and partially green, but not merge-clean yet.

   PR #331:

   - title: `fix(madaros): native_v2 codegen — print-int, for-loops, enum registration, break/continue (+ regression gates)`
   - branch: `fix/root2-enum-inplace`
   - head: `5e60acc6f3e9a51d3127d7638bc543b33a69252b`
   - mergeability: `MERGEABLE`
   - changed compiler paths include `self-hosted/ir/lower.sio`
   - CI status at audit time: `Contracts` failed at `Docs registry`; compiler,
     full-test, website, macOS/Linux self-host, and Lean jobs later showed
     green in the same rollup, but the red Contracts job remained in the PR
     status.
   - separate production-path validation run `27878982663` passed for the
     rebased PR artifact.

   This PR must not be merged until the Contracts failure is repaired or
   explicitly classified.

5. Several older compiler PRs are stale/conflicting and should be quarantined
   before new compiler integration.

   Open compiler-adjacent PRs observed:

   - #313 `fix(codegen): restore SRET + refactor module_frontend Box-based merge helpers`
     - state: open
     - mergeability: `CONFLICTING`
     - touches `self-hosted/compiler/main.sio`,
       `self-hosted/compiler/module_frontend.sio`,
       `self-hosted/compiler/module_native_driver.sio`
   - #296 `[codex] cover Madaros tmp directory check in full gate`
     - state: open draft
     - mergeability: `CONFLICTING`
     - touches `scripts/ci/madaros_full_gate.sh`
   - #232 `Consolidated modular compiler -> main...`
     - state: open draft
     - mergeability: `CONFLICTING`
     - very large historical compiler/archive surface

   These should not remain visually equivalent to active landing candidates.

## Dirty Critical Worktrees

These worktrees have uncommitted changes in critical compiler/CI paths:

| Worktree | Branch | Critical dirty files |
|---|---|---|
| `/workspace/sounio` | `main` | `artifacts/omega/agent_handoff.log.md`, `bin/madaros`, `self-hosted/native/machine_ir.sio` |
| `/workspace/sounio-codex` | `codex/calls-5-6-args` | `self-hosted/compiler/native_compile_driver.sio`, `self-hosted/native/codegen_x86_linux.sio`, `self-hosted/native/encode.sio`, new native-v2 CI gates |
| `/workspace/sounio-language-reality-gate` | `codex/madaros-language-reality-gate` | broad dirty set across `self-hosted/ir/*`, `self-hosted/native/*`, `render_native_compile_driver_lean.sio`, `scripts/ci/madaros_wide_int_gate.sh` |
| `/workspace/sounio-language-showcase` | `codex/language-showcase` | new foundry/language CI gates |
| `/workspace/sounio-madaros-source-elf-main` | `codex/madaros-source-to-elf-main` | `self-hosted/compiler/module_frontend.sio` |
| `/workspace/sounio-scientific-workbench` | `codex/scientific-workbench` | new `scripts/ci/scientific_workbench_e2e_gate.sh` |
| `/workspace/sounio-sret` | `claude/sret-builtins` | new `scripts/ci/native_v2_sret_builtins_gate.sh` |
| `/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b` | `worktree-agent-adc1cd8b9d52ba53b` | `self-hosted/compiler/main.sio`, `self-hosted/native/codegen.sio`, `self-hosted/native/codegen_x86_linux.sio`, `self-hosted/native/lower_ir.sio`, `self-hosted/native/suite.sio` |

## Critical Worktree Disposition Plan

This table is the actionable cleanup plan after #331/#332 and the Madaros
prebuilt refresh. It was derived from `origin/main` at
`b73a11f5838f138f3d06b591f832e98d501214f3`, the audit TSV
`/tmp/sounio-worktree-audit-critical-lanes-20260620.tsv`, and direct per-worktree
status/diff inspection.

| Worktree | Evidence | Classification | Required Action |
|---|---|---|---|
| `/workspace/sounio` | local `main` is 20 behind `origin/main`; dirty critical files are `artifacts/omega/agent_handoff.log.md`, `bin/madaros`, `self-hosted/native/machine_ir.sio`; `./sounio-whereami --quick` is missing in this stale checkout | protected primary surface, not evidence-clean | Do not reset. Reconcile only in an isolated branch/worktree: first stage/version the governance audit + handoff, then separately decide whether local `bin/madaros` and `machine_ir.sio` are salvageable or stale WIP. |
| `/workspace/sounio-codex` | branch `codex/calls-5-6-args`; no PR; 340 behind and 0 ahead vs `origin/main`; dirty compiler/native patch `+142/-40`; untracked full fixture set: `examples/native/*call*`, `examples/native/branch_exit.sio`, `scripts/ci/native_v2_*_gate.sh`, `self-hosted/ci/native_v2_calls_arity_ir_driver.sio` | stale local WIP with salvageable native-v2 calls/branch fixture bundle | Do not merge from this worktree. If still desired, create a fresh branch from `origin/main` and port the full fixture+gate bundle plus the minimal compiler changes, then run native-v2 focused gates. Otherwise archive patch and remove. |
| `/workspace/sounio-language-reality-gate` | branch `codex/madaros-language-reality-gate`; upstream exists; 141 behind and 3 ahead; no PR; dirty IR/native patch is large (`+1969/-480`) across `self-hosted/ir/*`, `self-hosted/native/*`, and `scripts/ci/madaros_wide_int_gate.sh` | active-or-stale high-risk compiler lane, not cleanup-safe | Requires owner decision. If active, rebase into an isolated current-main worktree and split into IR, native, and gate commits. If inactive, archive patch before removal. |
| `/workspace/sounio-language-showcase` | branch `codex/language-showcase`; no PR; 306 behind and 60 ahead; untracked full product/example set under `docs/pitches/`, `examples/clinical_foundry/`, `examples/evidence_foundry/`, `examples/language_tour/`, plus three CI gates | stale large showcase/foundry lane, not compiler-core | Salvage as one or more non-compiler PRs from current `origin/main`, with offload review if external-facing clinical/teaching artifacts are kept. Do not let this lane block compiler cleanup. |
| `/workspace/sounio-madaros-source-elf-main` | branch `codex/madaros-source-to-elf-main`; upstream incorrectly points at `origin/main`; 172 behind and 9 ahead; dirty critical change is only four lines in `module_frontend.sio`; branch diff touches source-to-ELF gates and compiler/IR/native files | potentially valuable but stale source-to-ELF lane | Owner should rebase/split. First test whether current `origin/main` already satisfies the source-to-ELF gate; if yes, archive/remove. If not, salvage the smallest patch only. |
| `/workspace/sounio-scientific-workbench` | branch `codex/scientific-workbench`; no PR; 306 behind and 48 ahead; untracked `examples/scientific_workbench/` plus `scripts/ci/scientific_workbench_e2e_gate.sh`; depends on native visual frontend gates in branch diff | stale scientific-workbench product lane | Salvage as a fresh non-compiler PR only if examples and gate still run on current main. Otherwise archive/remove. Do not mix with compiler production-readiness. |
| `/workspace/sounio-sret` | branch `claude/sret-builtins`; no PR; 340 behind and 0 ahead; full local bundle includes `docs/audit/NATIVE_V2_SRET_BUILTINS_AUDIT_2026-06-06.md`, `tests/native_v2_sret_builtins/`, and `scripts/ci/native_v2_sret_builtins_gate.sh` | stale local SRET gate bundle with no commits ahead | Salvage bundle onto current main only if the SRET witness gap is still relevant after #331/#332. Otherwise archive patch and remove. |
| `/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b` | branch `worktree-agent-adc1cd8b9d52ba53b`; no PR; 76 behind and 0 ahead; dirty native/compiler patch `+238/-234`; handoff identifies this as active Claude compiler lane | active Claude lane, do not touch | Coordinate with Claude. It must rebase/refresh against current `origin/main` before any merge candidate is considered. Codex should not edit these files concurrently. |

The next governance gate should fail only after these dispositions are recorded
as explicit lane states. Until then, these eight are not equivalent: some are
active compiler lanes, some are stale local fixture bundles, and some are
product/showcase work that should not block Madaros production readiness.

## Prunable Worktree Records

Status: cleaned.

These git worktree records point to missing locations and should be cleaned up
only after owner confirmation. They were subsequently removed by
`git worktree prune --verbose` because all seven were Git-reported stale records
whose gitdir files pointed to missing locations:

- `/tmp/lean-tier2`
- `/tmp/seq-a64`
- `/tmp/seq-tier2`
- `/tmp/sounio-website-pr`
- `/tmp/wt-affine`
- `/tmp/wt-hygiene`
- `/tmp/wtw2`

## Immediate Governance Rules Recommended

1. Freeze compiler merges except a single named landing lane.

   Historical note: PR #331 was that lane and is now merged, CI-green, and
   shipped through a refreshed default Madaros prebuilt. The next compiler lane
   must be named explicitly before touching serialized compiler files.

2. Do not use `/workspace/sounio` primary for compiler evidence.

   It is dirty and behind `origin/main`. Use a clean `/tmp` worktree or a
   branch-specific active worktree with exact SHA.

3. Every compiler gate must emit provenance.

   Required minimum:

   ```text
   source_ref:
   source_sha:
   binary_path:
   binary_sha256:
   execution_mode: launcher|raw
   stdlib_path:
   gate:
   run_id:
   worktree:
   ```

4. GitHub Actions `workflow_dispatch` runs must print checked-out identity.

   `headSha` can show default-branch workflow identity while the job checks out
   `inputs.ref`. The workflow must print and upload the actual `git rev-parse
   HEAD` after checkout, plus artifact SHA256.

5. Old conflicting compiler PRs must be closed, converted to draft archive, or
   explicitly marked superseded.

   #232, #296, and #313 have been marked/quarantined with exit criteria, but
   they are still open as of this follow-through.

6. Add and use the versioned audit script.

   `scripts/dev/worktree_branch_audit.sh` now reproduces the worktree inventory
   as TSV and prints the high-risk dirty compiler worktrees. Follow-up should
   wire a stricter gate mode that can fail on:

   - dirty critical files in more than one active compiler lane
   - PRs touching serialized files without a CLAIM in the handoff log
   - compiler PR without source SHA + binary SHA provenance
   - prunable worktree records older than a threshold

## Blocker Classification

Blocker-ID: `GOV-WORKTREE-SPRAWL-20260620`

- severity: high
- class: governance / multi-agent coordination
- evidence level: repo inventory + GitHub PR metadata
- owner: integration shepherd
- affected surfaces: `self-hosted/compiler/**`, `self-hosted/ir/**`,
  `self-hosted/native/**`, `bin/madaros*`, `bin/souc*`, `scripts/ci/**`,
  `.github/workflows/**`
- acceptance gate:
  - one active compiler landing lane named in `artifacts/omega/agent_handoff.log.md`
  - all stale conflicting compiler PRs classified
  - dirty critical worktrees either claimed, parked, or cleaned by owner
  - workflow/gate provenance emitted for Madaros builds
- next action:
  - classify each of the remaining 8 dirty critical worktrees by owner and
    action: salvage, close/remove, or active lane
  - evolve `scripts/dev/worktree_branch_audit.sh` into a failing governance gate
  - add durable Madaros build provenance artifacts/checks beyond the current
    refresh workflow logs
