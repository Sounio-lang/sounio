<!-- docs:meta
topic_id: repo.docs.audit.madaros-post-merge-closeout-2026-06-22
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-post-merge-closeout-2026-06-22
-->

# Madaros Post-Merge Closeout - 2026-06-22

## Scope

This note closes the 2026-06-21 Madaros blocker cluster that was tracked in
GitHub issue #356. It records what is closed, what proved it, and what should
not be reopened from stale evidence.

This is a status/audit artifact, not a new compiler patch.

## Closed Items

| Item | Final status | Evidence |
|---|---|---|
| `BLK-20260621-codex-source-elf-normal-bss` | Closed; now a regression control | `scripts/ci/madaros_open_blockers_probe.sh --diagnose-lowering` returned `rc=0`; `global_read_exit4` exits `4`; `global_store_exit7` exits `7`; lowering diagnostics reached `bodies_done` with markers present |
| PR #313 ownership disposition | Closed | PR #313 was closed on 2026-06-21T22:32:52Z as stale/conflicting compiler-owner overlap |
| `BLK-20260621-codex-madaros-build-segfault` | Closed; now a regression control | Default promoted-workspace self-build returns `build_rc_0`; `self_build_madaros` is a control row in `scripts/ci/madaros_open_blockers_probe.sh` |
| GitHub issue #356 | Closed | Closed on 2026-06-22T01:17:18Z after PR #395 merged and main CI completed successfully |

## Merge Evidence

- PR: #395, `fix(madaros): resolve workspace self-build parity`.
- Merge commit: `4c452498c156c0ce143ae48763abfaf0fb2c7b5d`.
- Main CI: run `27923484270`, completed `success`.
- Green main jobs:
  - `Contracts`
  - `Sounio Lint`
  - `Source-Bootstrap Self-Host (Linux x86_64)`
  - `Native Self-Host (Linux x86_64)`
  - `Native Self-Host (macOS arm64)`
  - `Full Test Suite`
  - `Lean Proofs`
  - `Website`

## Local Final Evidence

Final local checks were run in an isolated worktree derived from current
`origin/main`, not in the protected dirty primary checkout.

```bash
scripts/ci/madaros_open_blockers_probe.sh --diagnose-lowering
# rc=0

scripts/ci/madaros_source_to_elf_gate.sh
# rc=0

git diff --check
# rc=0
```

The open-blocker probe reported these resolved rows:

```text
control  global_read_exit4    normal      4          4          ok
control  global_read_exit4    native_v2   4          4          ok
control  global_read_exit4    build       4          4          ok
control  global_store_exit7   build       7          7          ok
control  self_build_madaros   self_build  build_rc_0 build_rc_0 ok
```

## Root Cause

`lean_single` import resolution could fall through to bogus base-directory
`mod.sio` candidates for module-qualified imports encountered inside imported
files. The observed bad shape was:

```text
self-hosted/parser/parser/types/mod.sio
```

The intended module candidate was:

```text
self-hosted/parser/types.sio
```

The final fix is intentionally narrow:

- keep loaded-path lookup before duplicate imports hit filesystem fallback,
- avoid base-directory `mod.sio` fallback for module-qualified raw paths after
  the base path read fails,
- preserve `stdlib` precedence, so imports such as `tensor::ops` continue to
  resolve to `stdlib/tensor/ops.sio` rather than `self-hosted/tensor/ops.sio`.

The broad self-hosted-first variant was rejected because it made tensor/PINN
tests resolve the wrong module family.

## What Not To Reopen From

Do not reopen #356 from any of these alone:

- a stale `bin/madaros` or `artifacts/self-hosted/madaros` ELF,
- a stale worktree that predates `main@4c452498c`,
- warnings emitted during successful self-builds,
- the historical PR #313 branch,
- old `build_rc_139` logs from before PR #395.

If a new failure appears, create a new blocker with fresh current-main evidence
and a new acceptance gate.

## Residual Warning Debt

The closeout does not claim all Madaros warning debt is gone. The following
families still deserve separate classification work, but they are not #356
blockers:

- assignment-type warnings in GPU/SPIR-V modules,
- borrow diagnostics in `stdlib/tensor/ops.sio` that do not currently fail the
  native test run,
- stack-frame-too-large warnings in large checker/native modules,
- parser expression match-arm warnings.

Recommended next lane: create a warning-debt tracker that classifies each family
as harmless, stale false positive, type-system bug, or runtime-risk predictor.
