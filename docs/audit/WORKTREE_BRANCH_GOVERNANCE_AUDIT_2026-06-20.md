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

## Current Status After Post-#336 Follow-Through

This audit has been acted on. The original snapshot below is preserved because
it explains the failure mode, but the current authoritative cleanup status is:

- Audit refresh baseline: `origin/main` at
  `06fd5de4f7f2e5d7cebf1a4c244ce568657b8735`
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
- #335 repaired the Madaros native-v2 source-to-ELF bridge:
  - #335 merge commit: `6ac1fc041d0b4a08e712bdb4e972be8eb2d87f2f`
  - branch fix commit: `44270a22e14da8fd2ebe468d60b9f4305483dd58`
  - GitHub PR CI run `27882127595`: pass
  - source-to-ELF gate on the built artifact: pass
- The Madaros checked-in prebuilt was refreshed again after #335:
  - refresh commit: `45ed56fa1e9cc01a5b119efbc4bf9e7d65289d0c`
  - refreshed `bin/madaros-linux-x86_64` SHA256:
    `506d24c47e6a735340b0f8ced2072fa1baf485bb8d65461857b5a8d5565b0cef`
  - GitHub prebuilt refresh run `27882373784`: pass
- #336 repaired the full gate for clean worktrees without local raw artifacts:
  - #336 merge commit: `02ad4473dcff9fd2b42b2135e47e17b43abbb304`
  - branch fix commit: `98167ef6797c0ec6487c3c1d94d54418c7b1ecbf`
  - GitHub PR CI run `27882751094`: pass
  - clean `origin/main` gates after #336:
    `madaros_full_gate.sh`, `madaros_source_to_elf_gate.sh`,
    `madaros_enum_gate.sh`, `madaros_loop_gate.sh`, and
    `madaros_operational_contract_gate.sh`: pass.
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
  - Four temporary clean Madaros proof/probe worktrees superseded by #335/#336
    were removed:
    `/tmp/sounio-madaros-d228-revert-probe`,
    `/tmp/sounio-madaros-source-elf-fix`,
    `/tmp/sounio-main-post335`,
    `/tmp/sounio-madaros-hardening`.
  - The stale source-to-ELF worktree was archived and removed:
    `/workspace/sounio-madaros-source-elf-main`.
    Its local branch commits and dirty patch were preserved under
    `/workspace/sounio-worktree-archives/madaros-source-to-elf-main-20260620/`.
  - The stale Scientific Workbench product/demo worktree was archived and
    removed: `/workspace/sounio-scientific-workbench`. Its local branch commits,
    dirty patch, and untracked example/gate files were preserved under
    `/workspace/sounio-worktree-archives/scientific-workbench-20260620/`.
  - The stale Language Showcase product/foundry worktree was archived and
    removed: `/workspace/sounio-language-showcase`. Its local branch commits,
    dirty patch, and untracked pitch/clinical/evidence/language-tour files were
    preserved under
    `/workspace/sounio-worktree-archives/language-showcase-20260620/`.
  - The stale SRET builtins worktree was archived and removed:
    `/workspace/sounio-sret`. Its dirty Slurm patch plus untracked SRET audit,
    gate, and test-bundle files were preserved under
    `/workspace/sounio-worktree-archives/sret-builtins-20260620/`.
  - The stale native-v2 calls/branch arity worktree was archived and removed:
    `/workspace/sounio-codex`. Its local compiler/native patch plus untracked
    native-v2 examples, gates, and IR driver were preserved under
    `/workspace/sounio-worktree-archives/calls-5-6-args-20260620/`.
  - The stale Madaros language-reality gate worktree was archived and removed:
    `/workspace/sounio-language-reality-gate`. Its three branch commits, broad
    dirty IR/native/checker patch, and archival bundle were preserved under
    `/workspace/sounio-worktree-archives/language-reality-gate-20260620/`.
    The remote branch `origin/codex/madaros-language-reality-gate` was not
    deleted.
  - The dirty protected primary checkout was archived in place, without reset,
    clean, stash, branch switch, or file removal:
    `/workspace/sounio-worktree-archives/primary-main-dirty-20260620/`.
    This preserves the primary `main` patch, untracked inventory, small
    untracked files, and checksums before any future reconciliation. The large
    untracked `data/processed/expansion` tree was inventoried but not copied.
- Post-cleanup inventory:
  - raw TSV: `/tmp/sounio-worktree-audit-20260620-post-primary-archive.tsv`
  - total worktrees: 70
  - dirty worktrees: 52
  - prunable worktree records: 0
  - dirty critical worktrees: 2
  - critical-diff vs `origin/main`: 48
  - open PRs represented by current worktrees: 0

The remaining blocker is therefore narrower: owner-by-owner disposition of the
2 dirty critical worktrees. It is no longer a generic unclassified PR/prebuilt
blocker; it is now specifically primary-checkout reconciliation plus Claude-lane
coordination. The primary checkout is now evidence-preserved but intentionally
not cleaned.

## Post-#371 Refresh — 2026-06-21

The audit remains active. A fresh current-main pass was run from clean worktree
`/tmp/sounio-madaros-worktree-triage` at
`c70075772db4265e4f14790c423711f9e6a02d63`.

Commands:

```bash
SOUNIO_AUDIT_INCLUDE_PRS=1 \
  scripts/dev/worktree_branch_audit.sh /tmp/sounio-worktree-audit-current.tsv

gh pr list --repo Sounio-lang/sounio --state open --limit 40 \
  --json number,title,headRefName,baseRefName,isDraft,mergeable,updatedAt,url
```

Current counts:

- total worktrees: 72
- dirty worktrees: 52
- prunable worktree records: 0
- dirty critical worktrees: 3
- unallowed dirty critical worktrees under strict local check mode: 3
- critical diffs versus `origin/main`: 49
- open PRs represented by current worktrees: 2
- open GitHub PRs: 8
- open GitHub PRs conflicting with current `main`: 7

Critical dirty worktrees requiring disposition:

| Worktree | Branch | Status | Required disposition |
|---|---|---|---|
| `/workspace/sounio` | `main` | dirty, behind `origin/main` by 78 commits | Keep protected; do not use as production evidence; reconcile only from archive plus fresh `origin/main` worktree. |
| `/workspace/sounio-effects` | `claude/effects-enforcement` | dirty, touches critical scripts/compiler surfaces versus `origin/main` | Treat as separate Claude/effects lane; do not mix into Madaros production readiness without owner handoff and current-main replay. |
| `/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b` | `worktree-agent-adc1cd8b9d52ba53b` | dirty active compiler/codegen lane | Keep Claude-owned; Codex may inspect only; must become check-clean before its compiler results are semantic evidence. |

Open PR disposition for Madaros production readiness:

| PR | Branch | Base | Mergeability | Disposition |
|---|---|---|---|---|
| #329 | `codex/pl-command-center` | `main` | conflicting, draft | Keep draft/reference-only until rebased onto current `main`; docs lane, not compiler evidence. |
| #313 | `fix/native-codegen-sret-regression-v2` | `main` | conflicting | Treat as stale compiler/codegen lane; do not merge over active Claude compiler ownership without explicit transfer and current gates. |
| #308 | `chore/repo-hygiene` | `main` | conflicting | Repo hygiene lane; keep out of Madaros production path until conflicts are resolved in a fresh worktree. |
| #297 | `qual/pbpk28-tissue-composition` | `main` | conflicting | Scientific/PBPK lane; requires science/offload discipline and must not be used as compiler-readiness evidence. |
| #287 | `feat/affine-octonion-correlation` | `main` | conflicting | Math/research lane; requires math/offload discipline and separate current-main replay. |
| #241 | `nl-castle/native-orc-audit` | `main` | conflicting | Historical/research lane; reference only for Madaros readiness unless replayed on current `main`. |
| #232 | `codegen/nested-mut-write-fix` | `main` | conflicting, draft | Stale broad compiler integration; keep quarantined unless ownership transfers and the patch is rebuilt from current `main`. |
| #226 | `feat/erdos-straus-gpu-sieve` | `integration/sounio-dev-ready-base` | mergeable to non-main base | Not part of current `main` production readiness; handle separately against its integration base. |

The next governance objective is not "delete every worktree." It is to reduce
the Madaros production influence set to:

1. clean `origin/main`,
2. issue #356 blocker records,
3. `scripts/ci/madaros_open_blockers_probe.sh`,
4. `scripts/ci/madaros_source_to_elf_gate.sh`,
5. one active compiler owner for the BSS/global blocker.

Any PR or worktree outside that set is evidence only after a current-main replay
and a named owner/gate.

Follow-up governance gate:

- `scripts/dev/worktree_branch_audit.sh --check` now fails when unallowed
  critical dirty worktrees or prunable worktree records are present.
- The CI `Contracts` job runs that gate in strict mode. Fresh CI clones should
  have zero prunable records and zero unallowed dirty critical worktrees.
- Local operation may temporarily allow known active lanes by setting
  `SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_RE`, but that exception must name exact
  worktree paths and keep `unallowed_critical_dirty=0`.
- The current known active exceptions are the protected primary checkout and
  the Claude-owned compiler lane:
  - `/workspace/sounio`
  - `/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b`

## Post-#336 Resolution Plan

This is the active plan for turning the audit into cleanup. It is intentionally
ordered so that compiler truth stays stable while stale or product lanes are
parked outside the Madaros production-readiness lane.

1. Keep `origin/main` as the only Madaros production baseline.

   Required evidence before any new compiler claim:

   - clean worktree at the named `origin/main` SHA
   - `env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN`
     on all compiler gates
   - `bin/madaros-linux-x86_64` SHA256 in the report
   - `madaros_full_gate.sh` plus the focused gate for the touched surface

2. Retire source-to-ELF stale lanes first.

   #335 and #336 prove that the cleanup baseline already passes the
   source-to-ELF, enum, loop, and full Madaros gates. Therefore
   `/workspace/sounio-madaros-source-elf-main` is no longer an active fix lane
   and has been removed from the active worktree set. Its archived evidence is:

   - `codex-madaros-source-to-elf-main.bundle`
   - `local-commits.txt`
   - `uncommitted.patch`
   - `SHA256SUMS`

   under
   `/workspace/sounio-worktree-archives/madaros-source-to-elf-main-20260620/`.

3. Split compiler-core lanes from product/demo lanes.

   Product/showcase worktrees must not block compiler production-readiness:
   `/workspace/sounio-scientific-workbench` and
   `/workspace/sounio-language-showcase` have now both been archived and
   removed. Their contents remain available in the archive tree for a future
   fresh non-compiler PR, with repository offload review required before any
   clinical or external-facing artifacts are published.

4. Put every remaining compiler WIP behind an owner and a current-main replay.

   `/workspace/sounio-language-reality-gate` was not eligible for direct merge
   from its stale worktree. It has now been archived and removed from the active
   worktree set. Future salvage must start from current `origin/main` and use
   the archive as evidence only. The preserved archive contains:

   - `language-reality-gate.bundle`
   - `branch.patch`
   - `uncommitted.patch`
   - `SHA256SUMS`

5. Do not touch the active Claude compiler lane from Codex.

   `/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b` remains an
   active Claude-owned lane. It must refresh at least against cleanup baseline
   `06fd5de4f7f2e5d7cebf1a4c244ce568657b8735` before review, but Codex must
   not edit its files concurrently.

6. Restore the primary checkout only after all salvage decisions are recorded.

   `/workspace/sounio` is still dirty, behind `origin/main`, and contaminated by
   environment variables that point compiler selection back to itself. It should
   remain protected. Its current dirty state has been archived under
   `/workspace/sounio-worktree-archives/primary-main-dirty-20260620/`; future
   reconciliation should now be done from that archive plus a fresh
   `origin/main` worktree, not by editing or resetting the primary checkout in
   place. Salvage decisions should be split into:

   - MCP client/test cleanup (`tools/mcp/**`)
   - local compiler binary/machine-IR artifacts (`bin/madaros`,
     `self-hosted/native/machine_ir.sio`)
   - workspace context hydration (`.beagle/context/**`)
   - audit/handoff notes and Slurm helper scripts
   - large generated data inventory (`data/processed/expansion`)

7. Enforce the audit as a gate, not a report.

   The audit script now supports `--check`. In check mode the default contract is:

   - `SOUNIO_AUDIT_MAX_PRUNABLE=0`
   - `SOUNIO_AUDIT_MAX_CRITICAL_DIRTY=0`
   - no implicit allowance for dirty compiler/native/CI paths

   Local operators may set `SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_RE` only for
   explicitly claimed active lanes. The useful invariant is not "no work can be
   dirty"; it is "no dirty critical worktree is invisible or ownerless."

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

These worktrees have uncommitted changes in critical compiler/CI paths after
the stale language-reality gate worktree was archived and removed and the
primary checkout state was evidence-preserved:

| Worktree | Branch | Critical dirty files |
|---|---|---|
| `/workspace/sounio` | `main` | `artifacts/omega/agent_handoff.log.md`, `bin/madaros`, `self-hosted/native/machine_ir.sio` |
| `/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b` | `worktree-agent-adc1cd8b9d52ba53b` | `self-hosted/compiler/main.sio`, `self-hosted/native/codegen.sio`, `self-hosted/native/codegen_x86_linux.sio`, `self-hosted/native/lower_ir.sio`, `self-hosted/native/suite.sio` |

## Critical Worktree Disposition Plan

This table is the actionable cleanup plan after #331/#332/#335/#336/#338/#339/#340/#341/#342/#343/#344
and the Madaros prebuilt refreshes. It was refreshed from `origin/main` at
`bf55bb190367228782993bf882eb1a255d8f9362`, the audit TSV
`/tmp/sounio-worktree-audit-20260620-post-primary-archive.tsv`, and direct
per-worktree status/diff inspection.

| Worktree | Evidence | Classification | Required Action |
|---|---|---|---|
| `/workspace/sounio` | local `main` is 37 behind `origin/main`; dirty critical files are `artifacts/omega/agent_handoff.log.md`, `bin/madaros`, `self-hosted/native/machine_ir.sio`; additional dirty files include `.beagle/context/**`, `AGENTS.md`, `examples/hello.sio`, and `tools/mcp/**`; untracked files include Madaros audit notes, helper scripts, `bin/madaros.bak-20260619`, and `data/processed/expansion`; state preserved under `/workspace/sounio-worktree-archives/primary-main-dirty-20260620/` | protected primary surface; evidence-preserved but not clean | Do not reset. Reconcile from a fresh `origin/main` worktree using the archive as evidence. Split salvage into MCP, compiler artifact, workspace-context, audit/handoff, Slurm-helper, and generated-data decisions. |
| `/workspace/sounio-codex` | branch `codex/calls-5-6-args`; no PR; 352 behind and 0 ahead of current `origin/main`; dirty compiler/native patch plus native-v2 calls/branch fixture bundle archived under `/workspace/sounio-worktree-archives/calls-5-6-args-20260620/`; local worktree and branch removed | stale local native-v2 calls/branch fixture bundle; parked outside active worktree set | Future salvage must start from current `origin/main` and use the archive as evidence only. |
| `/workspace/sounio-language-reality-gate` | branch `codex/madaros-language-reality-gate`; upstream existed; no PR; 154 behind and 3 ahead of current `origin/main`; broad dirty IR/native/checker patch archived under `/workspace/sounio-worktree-archives/language-reality-gate-20260620/`; local worktree and branch removed; remote branch preserved | stale high-risk compiler lane; parked outside active worktree set | Future salvage must start from current `origin/main`, split the gate from IR/native/compiler changes, and use the archive as evidence only. |
| `/workspace/sounio-language-showcase` | branch `codex/language-showcase`; no PR; stale product/example/foundry set plus three untracked CI gates; archived under `/workspace/sounio-worktree-archives/language-showcase-20260620/`; local worktree and branch removed | stale large showcase/foundry lane, not compiler-core; parked outside active worktree set | Future salvage must start from current `origin/main` and use the archive as evidence only. Offload review is required before publishing any clinical or external-facing artifacts. |
| `/workspace/sounio-sret` | branch `claude/sret-builtins`; no PR; 351 behind and 0 ahead of current `origin/main`; stale local bundle archived under `/workspace/sounio-worktree-archives/sret-builtins-20260620/`; local worktree and branch removed | stale local SRET gate bundle with no commits ahead; parked outside active worktree set | Future salvage must start from current `origin/main` and use the archive as evidence only. |
| `/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b` | branch `worktree-agent-adc1cd8b9d52ba53b`; no PR; 93 behind current `origin/main`; dirty native/compiler patch remains in `main.sio`, `codegen.sio`, `codegen_x86_linux.sio`, `lower_ir.sio`, and `suite.sio`; handoff identifies this as active Claude compiler lane | active Claude lane, do not touch | Coordinate with Claude. It must refresh against current `origin/main` before any merge candidate is considered. Codex must not edit these files concurrently. |

The next governance gate should fail only after these dispositions are recorded
as explicit lane states. Until then, these remaining lanes are not equivalent: some are
active compiler lanes, some are stale local fixture bundles, and some are
product/showcase work that has been parked outside the Madaros production
readiness path.

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
  - reconcile the evidence-preserved primary checkout from a fresh
    `origin/main` worktree, split by MCP, compiler-artifact, workspace-context,
    audit/handoff, Slurm-helper, and generated-data categories
  - coordinate the active Claude compiler lane refresh from current `origin/main`
    without Codex editing its five dirty compiler/native files
  - evolve `scripts/dev/worktree_branch_audit.sh` into a failing governance gate
  - add durable Madaros build provenance artifacts/checks beyond the current
    refresh workflow logs
