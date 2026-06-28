# Sounio compiler branch cleanup matrix - 2026-06-28

Status: active cleanup matrix for `integration/compiler-consolidation-20260628`

Purpose: convert the branch audit into operational actions. This file is not a
semantic proof that every branch claim is correct; it is the ownership and merge
safety map for reducing the branch set without losing compiler work.

## Current integration lane

- Keep: `integration/compiler-consolidation-20260628`
- Worktree: `/workspace/sounio-compiler-consolidation`
- Remote: `origin/integration/compiler-consolidation-20260628`
- Latest checked commit when this matrix was created: `e2d7b16e2`
- State: clean and synced with origin before this file was added

## Safe local deletion scan

Checked local branches without an attached worktree against the consolidation
branch.

Result:

- No non-`main` local branch without a worktree was both ancestor-contained and
  patch-id-empty versus `integration/compiler-consolidation-20260628`.
- Therefore no additional local branch was deleted in this pass.

Rationale: branches attached to worktrees are treated as owned lanes, even when
their branch tip looks old, because dirty worktree state may contain the real
handoff.

## Already absorbed or covered by current consolidation

These should not be merged again. If a remote/local branch is later removed,
its compiler claim is represented by the consolidation branch and audit.

- `fix/madaros-for-loop-lowering`: patch-equivalent / loop-print gate covers
  the current behavior.
- `fix/madaros-print-int-dispatch`: current scalar-kind dispatch is broader;
  `tests/native_v2_loop_print_gate/run.sh bin/souc` passed 7/7.
- `origin/fix/test-suite-epistemic-failures`: already contained by ancestry.
- `origin/feat/madaros-bump-arena`: already ancestor-contained.
- `origin/fix/root2-enum-inplace`: already ancestor-contained.
- `origin/fix/assoc-variance-273-consolidated`: already ancestor-contained.
- `codex/tuple-signature-types-20260626`: core parser/checker/native fixes are
  equivalent in consolidation; remaining history is GPU/script/test
  bookkeeping.
- `codex/release-conformance-spine-fix`: compiler-core validated-call
  hardening was extracted; remaining commits are release/tooling conformance.
- BoxNew branches: `codex/madaros-boxnew-clean` and
  `codex/madaros-boxnew-append-fix` are covered by current BoxNew/source gates;
  remaining branch-only material is debug markers, bridge rewrites, or binary
  refresh.

## Keep as owner lane, not merge parent

These have live worktrees or active ownership. Do not delete or direct-merge
without owner transfer.

- `/workspace/sounio` -> `feat/hyper-epistemic-mul`
- `/workspace/sounio-effects` -> `claude/effects-enforcement`
- `/workspace/sounio-ir` -> `claude/ir-heap-indirect`
- `/workspace/sounio-m2-firewall` -> `m2/effect-firewall`
- `/workspace/sounio-parser` -> `claude/parser-traits-iflet-enumdata`
- `/workspace/sounio-parser-integ` -> `claude/parser-integ`
- `/workspace/sounio-checker` -> `claude/checker-singlemodule-crashes`
- `/workspace/sounio-mmh` -> `claude/mm-hardening`
- `/workspace/sounio-stdlibpath` -> `claude/stdlib-path`
- `/workspace/sounio-gates` -> `claude/gate-recovery`
- `/workspace/sounio-lsp-work` -> `claude/lsp-revival`
- `/workspace/sounio-nv2-consolidate` -> `integration/native-v2-onto-exact-orc`
- `/workspace/sounio-nestedfix` -> `integration/modular-checker-e008`
- `/workspace/sounio-source-elf-proof` -> `codex/source-elf-on-madaros-proof`
- `/workspace/sounio-project-spine` -> `codex/project-spine-madaros`
- `/workspace/sounio-madaros-source-elf-consolidated` ->
  `codex/madaros-source-elf-consolidated`
- `/workspace/sounio-madaros-main-proof` -> `codex/madaros-main-proof-17d115`
- `/workspace/sounio-no-caveats` -> `codex/no-caveats-warning-zero`
- `/workspace/sounio-real-runner` -> `codex/real-language-runner`
- `/workspace/sounio-gpu-kernel` -> `feat/gpu-thread-intrinsics`
- `/workspace/sounio-affine` -> `feat/affine-nonassoc-uncertainty`
- `/workspace/sounio-affine-pg` -> `feat/affine-octonion-correlation`

## Archive/quarantine after owner ack

These branches should not be merged as branches. They may be deleted only after
their owners agree the audit is sufficient or after a tag/archive ref is made.

- E008/checker trains: `origin/check/*-e008`,
  `origin/parser/fn-type-effects-list-e008`,
  `origin/integration/e008-nested-store-complete`,
  `origin/integration/consolidate-modular`.
- Effects/kw-demote trains: `origin/claude/effects-enforcement`,
  `origin/claude/kw-demote-module`, `origin/integration/effects-kwdemote`,
  `origin/integrate/kw-demote-landing`,
  `origin/campaign/mc-frontend-fixes`.
- Native-v2/source bridge trains: `origin/feat/native-v2-source-bridge`,
  `origin/feat/native-v2-bridge-sret`, `origin/feat/mc-v2-opcodes`,
  `origin/modular/native-v2-e2e-gate`,
  `origin/modular/native-v2-source-to-elf`.
- G1/codegen trains: `origin/codegen/byval-arg-crasher`,
  `origin/codegen/deref-nested-store`, `origin/codegen/nested-mut-write-fix`,
  `origin/g1/qualify-bare-patterns`.
- GPU/research/backend trains: `origin/feat/gpu-thread-intrinsics`,
  `origin/gpu/epistemic-tensor-core-gum-sm75`,
  `origin/feat/exact-orc-machinery`,
  `origin/feat/erdos-straus-gpu-sieve`,
  `nl-castle/native-orc-audit`.
- Release/docs/tooling trains: `chore/repo-hygiene`,
  `codex/main-release-ci-repair-20260623`,
  `codex/release-install-visibility-20260626`,
  `codex/website-docs-support-gate-20260626`,
  `codex/baseline-docs-registry`, `fix/docs-registry-sync`,
  `claude/release-apparatus`, `claude/release-e2e-eval`,
  `claude/release-real-mc`.

## Extract-only residuals

These branches contain possible compiler ideas, but branch-level merge is
explicitly unsafe. Any future work must extract a small source hunk plus a
focused gate.

- `fix/ocp-locals-cap`: only the OCP splits are absorbed. Wide-int/SRET/class-2
  wall material is residual.
- `fix/silent-typecheck-diag`: A64/native-runtime train; keep branch history out
  of compiler consolidation.
- `origin/codex/direct-call-param-slot`: witness still exits 139; do not port
  until native-v2 runtime witness is green.
- `codex/madaros-import-stdlib-lowering-current` and
  `codex/imported-full-lowerer-20260627`: imported multimodule witness remains
  red at `lower_array: seed_begin`.
- `origin/codex/madaros-raw-async-gates-20260627`: raw async remains red
  (`E012`, `E137` surfaces).
- `feat/hyper-epistemic-mul`: backend/Metal prototype; no normal producer/gate
  and audit/help wording still conflicts with the accepted v2 scope.

## Next cleanup commands, when owner approval exists

Do not run these as a batch without confirming owner transfer and archive
policy. Use this shape one branch at a time:

```bash
git tag archive/<branch-name>/<date> <branch>
git branch -d <branch>
git push origin :<branch>
```

For branches with dirty worktrees, archive the dirty diff first:

```bash
git -C <worktree> status --short
git -C <worktree> diff > /tmp/<branch-name>.dirty.patch
```

## Verification used for this matrix

- `git status --short --branch` on `/workspace/sounio-compiler-consolidation`
- `git branch --format='%(refname:short)'`
- `git branch -r --format='%(refname:short)'`
- `git worktree list --porcelain`
- `git merge-base --is-ancestor <branch> HEAD`
- `git cherry -v HEAD <branch>`
- `git diff --shortstat HEAD..<branch>`
- Existing focused gates recorded in
  `docs/audit/BRANCH_CONSOLIDATION_2026-06-28.md`
