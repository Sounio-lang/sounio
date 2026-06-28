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

## Local cleanup executed

Archived and deleted 24 local-only branches that had no attached worktree and
no matching `origin/*` branch. These branches still had patch-exclusive history,
so each branch tip was first preserved as a pushed archive tag under
`archive/local/*/2026-06-28`.

Local branch count after deletion: 105.

Archived local branch tips:

- `archive/local/claude-gpu-e137-fix/2026-06-28` -> `150f72aaf`
- `archive/local/claude-solver-gpu-native-path/2026-06-28` -> `a8fe4756e`
- `archive/local/codex-card-a-parser/2026-06-28` -> `617204fdc`
- `archive/local/codex-card-b-ir-bodies/2026-06-28` -> `617204fdc`
- `archive/local/codex-gpu-modular-bridge/2026-06-28` -> `b3dc49ef2`
- `archive/local/codex-gpu-semantic-profile/2026-06-28` -> `2202eded4`
- `archive/local/codex-madaros-check-segv-sret/2026-06-28` -> `b021c6720`
- `archive/local/codex-madaros-rebuild-probe/2026-06-28` -> `e4284b656`
- `archive/local/codex-madaros-retire-lean-single-20260627/2026-06-28` ->
  `933da3c4a`
- `archive/local/codex-release-conformance-spine-fix/2026-06-28` ->
  `0e0845447`
- `archive/local/codex-release-install-visibility-20260626/2026-06-28` ->
  `2091589c2`
- `archive/local/codex-tuple-signature-types-20260626/2026-06-28` ->
  `c40517e2b`
- `archive/local/codex-website-docs-support-gate-20260626/2026-06-28` ->
  `f2d28ca06`
- `archive/local/debug-root2-probe/2026-06-28` -> `7bd6ac02e`
- `archive/local/f64-cast-field-type-isolate/2026-06-28` -> `717349576`
- `archive/local/feat-affine-octonion-clean/2026-06-28` -> `29159d660`
- `archive/local/feat-assoc-variance-wiring/2026-06-28` -> `01148b616`
- `archive/local/feat-lean-tier2/2026-06-28` -> `b5dac8017`
- `archive/local/feat-m2-effect-firewall/2026-06-28` -> `c3a2f7c8b`
- `archive/local/fix-docs-registry-sync/2026-06-28` -> `cac90b8db`
- `archive/local/fix-native-codegen-sret-regression-rebase/2026-06-28` ->
  `6312f1194`
- `archive/local/fix-silent-typecheck-diag/2026-06-28` -> `8875efa9e`
- `archive/local/research-affect-curvature-depression/2026-06-28` ->
  `aa10da5c5`
- `archive/local/research-erdos-compiler-wip/2026-06-28` -> `b8828063d`

## Clean worktree cleanup executed

Archived and removed three clean worktrees whose branch tips were already
covered by the consolidation audit. For each branch, the local and remote tips
matched, the worktree had zero `git status --short` lines, and an archive tag
was pushed before deleting the worktree, local branch, and remote branch.

Removed worktrees:

- `/workspace/sounio-forloop` -> `fix/madaros-for-loop-lowering`
- `/workspace/sounio-printint` -> `fix/madaros-print-int-dispatch`
- `/workspace/sounio-arena` -> `feat/madaros-bump-arena`

Archive tags:

- `archive/worktree/fix-madaros-for-loop-lowering/2026-06-28` ->
  `6dde85913`
- `archive/worktree/fix-madaros-print-int-dispatch/2026-06-28` ->
  `c152ac3b4`
- `archive/worktree/feat-madaros-bump-arena/2026-06-28` -> `59dd2bc8f`

Verification after deletion:

- Worktree paths absent: `/workspace/sounio-forloop`,
  `/workspace/sounio-printint`, `/workspace/sounio-arena`.
- Local branches absent: `fix/madaros-for-loop-lowering`,
  `fix/madaros-print-int-dispatch`, `feat/madaros-bump-arena`.
- Remote branches absent after `git fetch --prune origin`.
- Counts after this cleanup: 102 local branches, 96 remote refs, 70 worktrees.

## Remote cleanup executed

After `git fetch --prune origin`, checked `refs/remotes/origin/*` for remote
branches with no attached local worktree and no patch-id-exclusive commits
against `integration/compiler-consolidation-20260628`.

Archive tags created and pushed:

- `archive/codex-gum-variance-sota-20260626/2026-06-28` ->
  `36a3b1d2b`
- `archive/codex-parser-strict-20260627/2026-06-28` -> `95b16f24b`
- `archive/fix-stdlib-e2e-sret-workarounds/2026-06-28` -> `57255a289`

Remote branches deleted after archive tags were pushed:

- `origin/codex/gum-variance-sota-20260626`: `git cherry -v HEAD` reported the
  branch's exclusive fix as patch-equivalent (`- 36a3b1d2b...`). The variance
  tests are already recorded as passing in the consolidation audit.
- `origin/codex/parser-strict-20260627`: `git cherry -v HEAD` reported the
  branch's Box::new path-call checker fix as patch-equivalent
  (`- 95b16f24b...`).
- `origin/fix/stdlib-e2e-sret-workarounds`: both branch commits are
  patch-equivalent (`- a4c5db9de...`, `- 57255a289...`). Deleting the branch
  does not claim the broader native-runtime/SRET residual is solved; it only
  removes a branch with no remaining patch-exclusive source.

Verification after deletion:

- `git fetch --prune origin`
- `refs/remotes/origin/codex/gum-variance-sota-20260626`: absent
- `refs/remotes/origin/codex/parser-strict-20260627`: absent
- `refs/remotes/origin/fix/stdlib-e2e-sret-workarounds`: absent
- Remote ref count after prune: 99

Commands executed:

```bash
git tag archive/codex-gum-variance-sota-20260626/2026-06-28 origin/codex/gum-variance-sota-20260626
git tag archive/codex-parser-strict-20260627/2026-06-28 origin/codex/parser-strict-20260627
git tag archive/fix-stdlib-e2e-sret-workarounds/2026-06-28 origin/fix/stdlib-e2e-sret-workarounds
git push origin archive/codex-gum-variance-sota-20260626/2026-06-28 archive/codex-parser-strict-20260627/2026-06-28 archive/fix-stdlib-e2e-sret-workarounds/2026-06-28
git push origin :codex/gum-variance-sota-20260626 :codex/parser-strict-20260627 :fix/stdlib-e2e-sret-workarounds
```

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
