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

Archived and removed clean worktrees whose branch tips were already covered by
the consolidation audit. For each branch, the worktree had zero
`git status --short` lines and an archive tag was pushed before deleting the
worktree and local branch. Remote branches were deleted when a matching remote
branch existed.

Removed worktrees, batch 1:

- `/workspace/sounio-forloop` -> `fix/madaros-for-loop-lowering`
- `/workspace/sounio-printint` -> `fix/madaros-print-int-dispatch`
- `/workspace/sounio-arena` -> `feat/madaros-bump-arena`

Archive tags, batch 1:

- `archive/worktree/fix-madaros-for-loop-lowering/2026-06-28` ->
  `6dde85913`
- `archive/worktree/fix-madaros-print-int-dispatch/2026-06-28` ->
  `c152ac3b4`
- `archive/worktree/feat-madaros-bump-arena/2026-06-28` -> `59dd2bc8f`

Removed worktrees, batch 2:

- `/workspace/sounio/.claude/worktrees/wf_932ba1b4-006-1` ->
  `backlog/f64-compare`
- `/workspace/sounio/.claude/worktrees/wf_932ba1b4-006-3` ->
  `backlog/sret-return`
- `/workspace/sounio/.claude/worktrees/wf_5d5668eb-0cd-1` ->
  `honest/composer-strcat`
- `/workspace/sounio/.claude/worktrees/wf_5d5668eb-0cd-2` ->
  `honest/sret-builtins`

Archive tags, batch 2:

- `archive/worktree/backlog-f64-compare/2026-06-28` -> `54ecab937`
- `archive/worktree/backlog-sret-return/2026-06-28` -> `dce0ea1ca`
- `archive/worktree/honest-composer-strcat/2026-06-28` -> `7d260ebd5`
- `archive/worktree/honest-sret-builtins/2026-06-28` -> `27da861f9`

Removed worktrees, batch 3:

- `/workspace/sounio-no-caveats` -> `codex/no-caveats-warning-zero`
- `/workspace/sounio-erdos-canonical` -> `research/erdos-canonical`
- `/workspace/sounio-zd-surgery` -> `research/sedenion-zd-chromatic`
- `/workspace/sounio/.claude/worktrees/wf_80ff6027-621-4` ->
  `worktree-wf_80ff6027-621-4`
- `/workspace/sounio-frame-revert` -> `fix/revert-frame-noop`

Archive tags, batch 3:

- `archive/worktree/codex-no-caveats-warning-zero/2026-06-28` ->
  `aca1f4e19`
- `archive/worktree/research-erdos-canonical/2026-06-28` -> `e649f5b16`
- `archive/worktree/research-sedenion-zd-chromatic/2026-06-28` ->
  `ad1ac127a`
- `archive/worktree/worktree-wf_80ff6027-621-4/2026-06-28` ->
  `9f796a78a`
- `archive/worktree/fix-revert-frame-noop/2026-06-28` -> `683170964`

Verification after deletion:

- Batch-1 worktree paths absent: `/workspace/sounio-forloop`,
  `/workspace/sounio-printint`, `/workspace/sounio-arena`.
- Batch-2 worktree paths absent:
  `/workspace/sounio/.claude/worktrees/wf_932ba1b4-006-1`,
  `/workspace/sounio/.claude/worktrees/wf_932ba1b4-006-3`,
  `/workspace/sounio/.claude/worktrees/wf_5d5668eb-0cd-1`,
  `/workspace/sounio/.claude/worktrees/wf_5d5668eb-0cd-2`.
- Batch-3 worktree paths absent: `/workspace/sounio-no-caveats`,
  `/workspace/sounio-erdos-canonical`, `/workspace/sounio-zd-surgery`,
  `/workspace/sounio/.claude/worktrees/wf_80ff6027-621-4`,
  `/workspace/sounio-frame-revert`.
- Local branches absent for all twelve removed worktrees.
- Batch-1 and batch-3 remote branches absent after `git fetch --prune origin`;
  batch-2 had no matching remote branch.
- Counts after batch 3: 93 local branches, 95 remote refs, 61 worktrees.

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

## Residual branch archive/delete executed

Archived and deleted branches that had no attached worktree and were already
classified in this audit as non-merge residuals:

- `codex/imported-full-lowerer-20260627`: archived as
  `archive/branch/codex-imported-full-lowerer-20260627/2026-06-28` ->
  `40cbd8af8`, then local/remote refs were deleted. Its only patch-exclusive
  commit wires the currently-red multimodule witness into the Madaros prebuilt
  refresh workflow; the witness still fails at `lower_array: seed_begin`.
- `codex/madaros-import-stdlib-lowering-current`: archived as
  `archive/branch/codex-madaros-import-stdlib-lowering-current/2026-06-28` ->
  `542a91e31`, then local/remote refs were deleted. The branch remains an
  older imported-stdlib lowering residual that conflicts with the newer
  imported-lowering path.
- `recover/m2-effect-firewall`: archived as
  `archive/branch/recover-m2-effect-firewall/2026-06-28` -> `e2cdefc51`,
  then local/remote refs were deleted. The accepted M2 effect firewall work was
  already extracted from `m2/effect-firewall` into consolidation with the
  10/10 `native_v2_effects_gate`; this recovery branch is redundant provenance.
- `codex/main-release-ci-repair-20260623`: archived as
  `archive/branch/codex-main-release-ci-repair-20260623/2026-06-28` ->
  `19cde59a7`, then local/remote refs were deleted. This was already classified
  as release/docs/tooling rather than compiler-core merge material.
- `feat/generics-struct-instantiation`: archived as
  `archive/branch/feat-generics-struct-instantiation/2026-06-28` ->
  `f68d688c9`, then local/remote refs were deleted. The compiler-code portion
  is already present in consolidation; its branch-only generic struct witness
  still parses red on current consolidation and remains residual.
- `fix/binop-literal-float-478b`: archived as
  `archive/branch/fix-binop-literal-float-478b/2026-06-28` -> `3fd937bd1`,
  then local/remote refs were deleted. All three exclusive commits were
  `DEBUG:` IR tracing changes in `self-hosted/native/codegen_x86_linux.sio`,
  already rejected as not being a semantic #478 fix.
- `codex/madaros-pr-disposition-20260621`: archived as
  `archive/branch/codex-madaros-pr-disposition-20260621/2026-06-28` ->
  `3d522f3a2`, then local/remote refs were deleted. Its primary audit doc is
  byte-identical in consolidation; the remaining branch-only governance
  registry/report edits are stale relative to the current, much larger
  registry state and were not ported.
- `fix/main-ci-red-2026-06-18`: archived as
  `archive/branch/fix-main-ci-red-2026-06-18/2026-06-28` -> `db9239fb2`,
  then local/remote refs were deleted. This branch was a stale mixed CI repair
  lane. The only compiler-core hunk still missing from consolidation was
  extracted separately: `lean_single` no longer treats `bool` as compatible
  with `i64`, and the two formerly-known-failure compile-fail fixtures were
  promoted back to ordinary compile-fail tests. The old website locale and docs
  registry edits were not ported because they would overwrite newer website and
  governance state.
- `codex/pl-command-center`: archived as
  `archive/branch/codex-pl-command-center/2026-06-28` -> `efc852596`,
  then local/remote refs were deleted. Its exclusive work is a
  `docs/serious-language/command-center.md` public-claim operating board plus
  stale governance registry/report edits. It was not ported into the compiler
  consolidation lane because it is not compiler-core and should only land via a
  documentation/public-surface lane with the required external-artifact review.
- `codex/solver-pilot-blockers-clean-20260626`: archived as
  `archive/branch/codex-solver-pilot-blockers-clean-20260626/2026-06-28` ->
  `4af962057`, then local/remote refs were deleted. Its exclusive work is a
  solver external-corpus pilot harness and Slurm submitter under
  `scripts/research/` plus a benchmark manifest. It was not ported into the
  compiler consolidation lane because it is research/benchmark infrastructure,
  not compiler-core behavior.
- `codex/solver-pilot-blockers-20260626`: archived as
  `archive/branch/codex-solver-pilot-blockers-20260626/2026-06-28` ->
  `c6c6598ab`, then local/remote refs were deleted. This was the older sibling
  of the solver external-corpus pilot lane; its tuple-destructuring commit was
  patch-equivalent in consolidation and the remaining solver harness work is
  research infrastructure already preserved by archive tags.
- `fix/website-quality-check`: archived as
  `archive/branch/fix-website-quality-check/2026-06-28` -> `1e5e96d6c`,
  then local/remote refs were deleted. This was a website/docs CI repair branch;
  its `bool`/`i64` compiler hunk had already been extracted from
  `fix/main-ci-red-2026-06-18`, and the remaining website/governance edits are
  stale relative to the current website state.
- `codex/madaros-boxnew-clean`: archived as
  `archive/branch/codex-madaros-boxnew-clean/2026-06-28` -> `49434fc7b`,
  then local/remote refs were deleted. Current consolidation already contains
  the non-debug Box::new/flat-lowering fixes in newer form; remaining branch
  material is debug markers, stale bridge rewrites, or binary refresh history.
- `codex/madaros-boxnew-append-fix`: archived as
  `archive/branch/codex-madaros-boxnew-append-fix/2026-06-28` ->
  `de2a6e213`, then local/remote refs were deleted. It was covered by the same
  BoxNew triage: no direct merge because the branch would reintroduce stale
  broad compiler/script rewrites already superseded by the consolidation lane.
- `codex/root2-global-lookup-probe`: archived as
  `archive/branch/codex-root2-global-lookup-probe/2026-06-28` ->
  `d2ffa34ad`, then local/remote refs were deleted. Current consolidation
  already contains the non-debug global function lookup fix plus the
  `native_v2_global_bss_gate`; the branch-only residue is probe tracing.
- `codex/root2-global-preload-probe`: archived as
  `archive/branch/codex-root2-global-preload-probe/2026-06-28` ->
  `70c29abbd`, then local/remote refs were deleted. Current consolidation
  already contains the stable BSS global preload/store-back path; the branch
  only preserved old probe tracing history.
- `chore/repo-hygiene`: archived as
  `archive/branch/chore-repo-hygiene/2026-06-28` -> `4302652f3`, then
  local/remote refs were deleted. It is release/docs/repository-hygiene
  material, not a compiler-core merge source for this consolidation lane.
- `integration/madaros-fixes-trace`: archived as
  `archive/branch/integration-madaros-fixes-trace/2026-06-28` ->
  `cb77ab17b`, then local/remote refs were deleted. Its for-loop and
  print-int-literal fixes are already represented in consolidation by the
  native-v2 loop/print gate and newer lowering code; the direct branch diff
  would revert newer lowerer fields and docs metadata.
- `feat/future-work-first-slices`: archived as
  `archive/branch/feat-future-work-first-slices/2026-06-28` -> `cf1123f6b`,
  then local/remote refs were deleted. The branch is future-work/research/docs
  material covering Lean/GPU/PBPK slices, not compiler-core consolidation
  material; its tip remains recoverable by archive tag.
- `d3-a/missing-imports-sweep`: archived as
  `archive/branch/d3-a-missing-imports-sweep/2026-06-28` -> `728e4ca39`,
  then local/remote refs were deleted. It is a Lean/website/math-provenance
  branch rather than a compiler-core merge source; the archive tag preserves
  the offload log and theorem edits for any future math lane.
- `docs/pbpk-session-notes-2026-06-28`: archived as
  `archive/branch/docs-pbpk-session-notes-2026-06-28/2026-06-28` ->
  `37b7741b9`, then local/remote refs were deleted. It is PBPK dissertation
  notes plus broad Madaros/solver history, not a compiler-core merge branch;
  the clinical/external-facing notes remain recoverable by archive tag.
- `gpu/epistemic-tensor-core-gum-sm75`: archived as
  `archive/branch/gpu-epistemic-tensor-core-gum-sm75/2026-06-28` ->
  `a88b0c6b8`, then local/remote refs were deleted. It is GPU tensor-core GUM
  design/provenance material and a CUDA reference file, not part of this
  compiler-core consolidation pass.
- `nl-castle/native-orc-audit`: archived as
  `archive/branch/nl-castle-native-orc-audit/2026-06-28` -> `da34c57d6`,
  then local/remote refs were deleted. It is ORC definition/audit provenance,
  not an active compiler consolidation source.
- `parser/algebra-keyword-e008`: archived as
  `archive/branch/parser-algebra-keyword-e008/2026-06-28` -> `ff0bc1166`,
  then local/remote refs were deleted. Current consolidation already carries
  lexer/parser support for `algebra`, `study`, and scientific notation; the
  branch-level diff is an old E008 rollback/provenance shape and is not a safe
  merge source.

Post-delete counts: 60 local branches, 67 remote refs, 52 worktrees.

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
  `codex/madaros-boxnew-append-fix` are covered by current BoxNew/source gates
  and were archived/deleted above; remaining branch-only material is debug
  markers, bridge rewrites, or binary refresh.

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

## Newly absorbed clean-worktree compiler lanes

- `m2/effect-firewall`: extracted commit `52689f11d` into
  `integration/compiler-consolidation-20260628` instead of merging the branch
  wholesale. The port keeps M2 effect enforcement, adds the `LLM` effect kind,
  adds the build-mode codegen firewall, and installs
  `tests/native_v2_effects_gate/`. During conflict resolution the consolidation
  lane kept the existing chaotic/equiv and large-frame checker warnings, removed
  a duplicate `call_expr_is_box_new` recognizer, and made the inplace
  `Box::new` handler require `Alloc` like the by-value handler.
  Validation: `bash scripts/ci/build_modular_madaros.sh
  /tmp/madaros-effects-port` produced a fresh Madaros ELF; `bash
  tests/native_v2_effects_gate/run.sh /tmp/madaros-effects-port` passed 10/10;
  `bash scripts/run_sio_test_suite.sh hello --verbose` passed 2/2. The
  consolidation commit `90b147169` was pushed, the original branch tip was
  archived as `archive/worktree/m2-effect-firewall/2026-06-28` ->
  `8c34a11a8`, `/workspace/sounio-m2-firewall` was removed, and local/remote
  `m2/effect-firewall` refs were deleted. Post-delete counts: 92 local
  branches, 94 remote refs, 60 worktrees.

## Dirty-but-absorbed worktree cleanup executed

- `/workspace/sounio-integ` -> `fix/root2-enum-inplace`: the branch had no
  patch-exclusive commits versus `integration/compiler-consolidation-20260628`
  (`git cherry -v HEAD fix/root2-enum-inplace` emitted no `+` lines), and the
  only dirty state was one untracked audit note:
  `docs/audit/MADAROS_328_LOCAL_REGRESSION_2026-06-20.md`. That note was
  recovered into this consolidation branch before cleanup. The original branch
  tip was archived as `archive/worktree/fix-root2-enum-inplace/2026-06-28` ->
  `55c440b23`, `/workspace/sounio-integ` was removed, and local/remote
  `fix/root2-enum-inplace` refs were deleted. Post-delete counts: 91 local
  branches, 93 remote refs, 59 worktrees.

- Slurm-only Madaros proof lanes:
  `/workspace/sounio-madaros-check-segv` (`codex/madaros-full-functioning`),
  `/workspace/sounio-madaros-source-elf-consolidated`
  (`codex/madaros-source-elf-consolidated`), and
  `/workspace/sounio-madaros-main-proof`
  (`codex/madaros-main-proof-17d115`) had no remaining patch-exclusive compiler
  commits versus consolidation (`git cherry -v HEAD` empty for the first two;
  only `- 0b3624dcc...` for the third). The only dirty state in each worktree
  was the same local Slurm runner tweak, not a compiler change:

```diff
diff --git a/slurm-jobs/erdos90/run_on_cluster.sh b/slurm-jobs/erdos90/run_on_cluster.sh
@@
-base64 -w0 "$ELF" | srun --partition="$PART" --time=00:10:00 --chdir=/orangefs/training bash -c '
+base64 -w0 "$ELF" | srun --partition="$PART" --gres=gpu:0 --time=00:10:00 --chdir=/orangefs/training bash -c '
```

  This WIP patch was archived here rather than ported into compiler
  consolidation. Original branch tips were archived as:
  `archive/worktree/codex-madaros-full-functioning/2026-06-28` ->
  `17d1157be`, `archive/worktree/codex-madaros-source-elf-consolidated/2026-06-28`
  -> `14f984e26`, and
  `archive/worktree/codex-madaros-main-proof-17d115/2026-06-28` ->
  `0b3624dcc`. The three worktrees were removed and the three local branches
  were deleted. No matching remote branches existed at cleanup time. Post-delete
  counts: 88 local branches, 93 remote refs, 56 worktrees.

- `/workspace/sounio-tests-fix` -> `fix/test-suite-epistemic-failures`: the
  branch had no patch-exclusive commits versus consolidation and was already
  listed as absorbed by ancestry. The only dirty state was generated
  `test-results.xml` output (`tests="1079" failures="12" skipped="69"`), so it
  was not ported into the consolidation branch. The original branch tip was
  archived as
  `archive/worktree/fix-test-suite-epistemic-failures/2026-06-28` ->
  `de63733c9`, `/workspace/sounio-tests-fix` was removed, and local/remote
  `fix/test-suite-epistemic-failures` refs were deleted. Post-delete counts:
  87 local branches, 92 remote refs, 55 worktrees.

- `/workspace/sounio/.claude/worktrees/madaros-default` ->
  `fix/assoc-variance-273-consolidated`: the branch had no patch-exclusive
  commits versus consolidation and was already ancestor-contained. The dirty
  state was documentation-only: two audit docs had `docs:meta` headers already
  present in consolidation, and one untracked forensic audit,
  `docs/audit/FRAME_FIX_7fa3c3524_DEAD_CODE_2026-06-16.md`, was recovered into
  this branch before cleanup. The original branch tip was archived as
  `archive/worktree/fix-assoc-variance-273-consolidated/2026-06-28` ->
  `b595ac4a2`, `/workspace/sounio/.claude/worktrees/madaros-default` was
  removed, and local/remote `fix/assoc-variance-273-consolidated` refs were
  deleted. Post-delete counts: 86 local branches, 91 remote refs, 54 worktrees.

- `/workspace/sounio-f64cmp` -> `minimax/f64-compare`: the branch had no
  patch-exclusive commits versus consolidation and no matching remote branch.
  Dirty state was not compiler-core: an executable-bit-only change to
  `artifacts/sprint72/check_single.elf`, an untracked backup of
  `slurm-jobs/erdos90/run_on_cluster.sh`, and a Slurm runner WIP that defaulted
  the partition to `all`, selected a `souc` with a `compile` subcommand, exported
  `SOUNIO_STDLIB_PATH`, and added `--gres=gpu:0` to `srun`. This patch was
  archived in the audit rather than ported into compiler consolidation. The
  original branch tip was archived as
  `archive/worktree/minimax-f64-compare/2026-06-28` -> `8fcf23d18`,
  `/workspace/sounio-f64cmp` was removed, and local `minimax/f64-compare` was
  deleted. No matching remote branch existed at cleanup time. Post-delete
  counts: 85 local branches, 91 remote refs, 53 worktrees.

## Current zero-exclusive boundary

After the cleanup above, only two attached worktrees still have no
patch-exclusive commits versus consolidation. They are **not** automatic
deletion candidates:

- `/workspace/sounio-metrics` (`metrics/fregni-profile`): untracked PPCR /
  survival-validation docs and tests. Clinical/external-facing policy applies;
  keep as owner lane until reviewed.
- `/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b`
  (`worktree-agent-adc1cd8b9d52ba53b`): dirty native/compiler WIP in
  `self-hosted/compiler/main.sio`, native codegen, lowerer, and suite. Needs
  extraction/gates before cleanup.

Closed since the previous boundary entry:

- `/workspace/sounio-source-elf-proof`
  (`codex/source-elf-on-madaros-proof`): re-reviewed against current
  consolidation HEAD. Its committed history was patch-equivalent, and its dirty
  diff would downgrade current checker/parser architecture: it reverts
  heap-indirect struct/enum/fn tables, removes newer ontology verdict and
  visibility plumbing, and removes parser newline guards for `[`/`(`/prefix
  operators. The raw dirty diff was archived at
  `docs/audit/archived_wip/source-elf-proof-dirty-2026-06-28.patch`; do not
  port it wholesale. The original branch tip was archived as
  `archive/worktree/codex-source-elf-on-madaros-proof/2026-06-28` ->
  `8951073c1`, `/workspace/sounio-source-elf-proof` was removed, and local
  `codex/source-elf-on-madaros-proof` was deleted. No matching remote branch
  existed at cleanup time. Post-delete counts: 84 local branches, 91 remote
  refs, 52 worktrees.

- `feat/assoc-variance-clean`: direct merge rejected. Its first two commits
  (`3c7fa879a`, `d75ce1c7`) were patch-equivalent to consolidation, while the
  remaining diff had a stale rollback shape across docs/audit and offload-log
  surfaces. The useful compiler/stdlib hunk from `2e93f7d0` was extracted
  manually instead: `pg_leaf`/`pg_mul` now return `-1` before overflowing the
  64-node perturbation arena, and `pg_mul` plus `UncertainOct.mul3` clamp total
  variance at zero for negative-kappa / floating-point drift cases. Focused
  assertions were added to the existing run-pass witnesses. The later
  `e3174a35` module split to `epistemic::propagate_nonassoc` was not ported:
  it changes the public module path and is not required for the defensive hunk.
  Validation: `git diff --check` passed; `bash scripts/ci/build_modular_madaros.sh
  /tmp/madaros-assoc-variance-guard` produced `Madaros ready`; the rebuilt
  Madaros passed `check` for `stdlib/epistemic/perturbation_graph.sio`,
  `stdlib/epistemic/uncertain_octonion.sio`, and
  `tests/run-pass/perturbation_graph_order_safe.sio`. The focused run gates for
  `perturbation_graph_order_safe`, `uncertain_octonion_auto`, and
  `propagate_nonassoc_variance` remain known-failure territory:
  `tests/known_failures/hardened_diagnostics_full_suite.txt` already lists all
  three; baseline `HEAD` reproduces the same `perturbation_graph` run-139 and
  the same `uncertain` / `propagate` E137 typecheck failures. LLM-offload policy:
  `bin/llm-offload -t math-review -p xai -i
  stdlib/epistemic/uncertain_octonion.sio` returned OK / no mathematical errors;
  the file/diff reviews for the perturbation graph returned
  `NO MATHEMATICAL CONTENT TO REVIEW`, so they are recorded as non-approval
  context rather than semantic proof. The original branch tip was archived as
  `archive/branch/feat-assoc-variance-clean/2026-06-28` -> `e3174a35e`;
  local and remote `feat/assoc-variance-clean` refs were deleted. Post-delete
  counts: 59 local branches, 66 remote refs, 52 worktrees.

- `fix/flatparse-and-scan-operator`: no attached worktree, but not a narrow
  parser branch in current consolidation terms. `git cherry -v HEAD
  fix/flatparse-and-scan-operator` showed a long native-v2 / source-bridge /
  release train, and `git diff --stat HEAD..fix/flatparse-and-scan-operator`
  spanned 3022 files with 296362 insertions and 672707 deletions, including
  mass deletions across docs/audit, tests, website, and release surfaces. Direct
  merge was rejected as a stale rollback / provenance bundle. No hunk was
  extracted in this pass; native-v2/source-bridge work remains represented by
  active owner lanes and separate residual branches. The original branch tip was
  archived as
  `archive/branch/fix-flatparse-and-scan-operator/2026-06-28` -> `aa39c356d`;
  local and remote `fix/flatparse-and-scan-operator` refs were deleted.
  Post-delete counts: 58 local branches, 65 remote refs, 52 worktrees.

- `fix/madaros-tuple-let-desugar`: no attached worktree, and not a tuple-let
  branch in current consolidation terms. `git cherry -v HEAD
  fix/madaros-tuple-let-desugar` showed a mixed solver / GPU / PBPK /
  bootstrap substrate train, while `git diff --stat
  HEAD..fix/madaros-tuple-let-desugar` spanned 266 files with 9868 insertions
  and 10228 deletions. The branch would delete the active consolidation audit
  docs and several native-v2/effects gates already consolidated elsewhere, while
  adding unrelated solver/GPU research material and a PBPK dissertation note.
  Direct merge was rejected as a stale provenance bundle. No hunk was extracted
  in this pass; PBPK/external-facing material was not ported because it would
  require its own owner/offload review. The original branch tip was archived as
  `archive/branch/fix-madaros-tuple-let-desugar/2026-06-28` -> `d1f43914f`;
  local and remote `fix/madaros-tuple-let-desugar` refs were deleted.
  Post-delete counts: 57 local branches, 64 remote refs, 52 worktrees.

- `fix/native-codegen-sret-regression`: no attached worktree. The core SRET fix
  commit (`8f537aac1`, struct-shorthand + nested-array return after SRET
  refactor) was patch-equivalent to consolidation (`git cherry -v HEAD` marked
  it `-`). The only patch-exclusive commit was governance/docs registry churn
  (`6312f1194`), while the full branch diff still had stale rollback shape:
  1591 files, 4859 insertions, 231696 deletions, including deletion of the
  active consolidation audit docs and many already-consolidated gates. Direct
  merge was rejected; no hunk was extracted. Archive before deleting as
  `archive/branch/fix-native-codegen-sret-regression/2026-06-28` -> `6312f1194`.

- `fix/native-codegen-sret-regression-v2`: no attached worktree. Despite the
  SRET name, patch-exclusive commits were unrelated/mixed: `6aa3ef236`
  (ROCm/HIP backend work) and `90f13011a` (module_frontend compile-to-file API
  refactor). The full branch diff had stale rollback shape: 1581 files, 5909
  insertions, 231373 deletions, including deletion of active consolidation audit
  docs and native-v2/effects gates. Direct merge was rejected; no hunk was
  extracted. Archive before deleting as
  `archive/branch/fix-native-codegen-sret-regression-v2/2026-06-28` -> `90f13011a`.
  Local and remote refs for both SRET branches were deleted. Post-delete counts:
  55 local branches, 62 remote refs, 52 worktrees.

- `claude/kw-demote-module`: no attached worktree. The branch is not a narrow
  keyword-demotion branch in current consolidation terms; it is a historical
  native-v2/self-hosting train with many already-superseded changes. The full
  diff versus consolidation spanned 2974 files, 293894 insertions, and 672649
  deletions, including removal of the active branch-consolidation audit docs,
  workflows, demo/docs surfaces, and consolidated gates. Direct merge was
  rejected. The relevant M2/effect-firewall work has already been ported and
  validated in this consolidation branch; no additional hunk was extracted.
  The original branch tip was archived as
  `archive/branch/claude-kw-demote-module/2026-06-28` -> `d731cc3ce`;
  local and remote `claude/kw-demote-module` refs were deleted.

- `integration/effects-kwdemote`: no attached worktree. The branch shares the
  same historical effects/kwdemote/native-v2 ancestry, with a full diff of 2984
  files, 296305 insertions, and 672599 deletions versus consolidation. It would
  delete active consolidation audits and many already-consolidated gates while
  reintroducing old effects measurement artifacts. Direct merge was rejected.
  The useful effect-enforcement path was already extracted through the M2
  effect-firewall port and validated separately; no hunk was extracted here.
  The original branch tip was archived as
  `archive/branch/integration-effects-kwdemote/2026-06-28` -> `89a467baf`;
  local and remote `integration/effects-kwdemote` refs were deleted.
  Post-delete counts: 53 local branches, 60 remote refs, 52 worktrees.

- `fix/ocp-locals-cap`: no attached worktree. The first two OCP-local-cap
  commits were already patch-equivalent to consolidation (`git cherry -v HEAD`
  marked `a8005666b` and `abcd3077f` as `-`). The remaining exclusive commit
  (`ba02961ed`) mixed a rebuilt `bin/souc`, bool/i64 `ty_eq`, wide-int/SRET
  ports, and class-2 wall cleanup. The full branch diff still had rollback
  shape: 2742 files, 280135 insertions, 671534 deletions, including deletion of
  active consolidation audit docs and many historical docs/gates. Direct merge
  was rejected; no hunk was extracted. The bool/i64 scalar rule was already
  handled separately in consolidation, and wide-int/SRET material needs a named
  owner lane rather than wholesale branch import. Archive before deleting as
  `archive/branch/fix-ocp-locals-cap/2026-06-28` -> `ba02961ed`; local and
  remote `fix/ocp-locals-cap` refs were deleted. Post-delete counts: 52 local
  branches, 59 remote refs, 52 worktrees.

- `feat/native-v2-source-bridge`: no attached worktree. This is an important
  native-v2/wide-int/source-bridge provenance train, but not a direct merge
  source for consolidation. `git cherry -v HEAD` showed the whole source-bridge
  and wide-int sequence as patch-exclusive, while `git diff --shortstat
  HEAD..feat/native-v2-source-bridge` showed 2860 files, 284826 insertions, and
  718501 deletions, including removal of active consolidation audit docs,
  historical artifacts, datasets, demos, and gates. Direct merge was rejected.
  No hunk was extracted in this pass; future source-bridge/wide-int work should
  be reintroduced from named commits with focused gates, not via wholesale
  branch import. Note: local tip `55c8c94f9` was ahead of remote
  `origin/feat/native-v2-source-bridge` (`9c2097c33`), so archive the local tip
  before deleting as
  `archive/branch/feat-native-v2-source-bridge/2026-06-28` -> `55c8c94f9`;
  local and remote `feat/native-v2-source-bridge` refs were deleted.

- `codex/madaros-close-20260627`: no attached worktree. The branch contains a
  mixed Madaros solver/runtime closeout train: imported-SMT lowering substrate,
  theorem/solver readiness docs and scripts, gate deflaking, and seed/binary
  resync. The full branch diff still had rollback shape relative to
  consolidation: 256 files, 9018 insertions, 10304 deletions, including deletion
  of active branch-consolidation audit docs and several already-consolidated
  gates. Direct merge was rejected. No hunk was extracted; any remaining Madaros
  closeout work should be pulled by named patch and gate, not by branch merge.
  Archive before deleting as
  `archive/branch/codex-madaros-close-20260627/2026-06-28` -> `c64fbd8ad`;
  local and remote `codex/madaros-close-20260627` refs were deleted.
  Post-delete counts: 50 local branches, 57 remote refs, 52 worktrees.

- `qual/pbpk28-tissue-composition`: no attached worktree. The visible tip adds
  PBPK28 tissue-composition / peptide-partitioning / QE approximation modules
  and dissertation claim-truth-table updates, but the full diff versus
  consolidation had stale rollback shape: 2407 files, 10585 insertions, 253602
  deletions, including deletion of the active consolidation audit docs, PPCR
  docs, solver research docs, and many historical audit surfaces. Direct merge
  was rejected. No clinical/PBPK or external-facing hunk was extracted here
  because it would require a dedicated owner lane plus mandatory offload review.
  Current PBPK ownership remains with the attached
  `/workspace/sounio-pbpk-integration`
  (`integration/pbpk-sprints-28-70-onto-main`) lane, not this stale residual
  branch. Archive before deleting as
  `archive/branch/qual-pbpk28-tissue-composition/2026-06-28` -> `8269a9a80`;
  local and remote `qual/pbpk28-tissue-composition` refs were deleted.
  Post-delete counts: 49 local branches, 56 remote refs, 52 worktrees.

## Residual local no-worktree sweep complete

After the cleanup entries above, the local branch set has no remaining
non-`main`, non-consolidation branch without an attached worktree:

```bash
git for-each-ref --format='%(refname:short)' refs/heads | while read b; do
  if ! git worktree list --porcelain | rg -q "^branch refs/heads/${b}$"; then
    if [ "$b" != "integration/compiler-consolidation-20260628" ] && [ "$b" != "main" ]; then
      echo "$b"
    fi
  fi
done
```

Result: no output. Current counts are 49 local branches, 56 remote refs, and 52
worktrees. The remaining local branches are attached owner/worktree lanes or
`main`/consolidation and should be handled as an ownership/worktree phase rather
than as orphan-branch cleanup.

## Attached worktree ownership phase

Initial attached-worktree census after the no-worktree sweep:

- `git worktree list --porcelain`: 52 worktrees.
- All attached worktrees reported `git status --short` count `0` in this pass.
- Remaining branches are attached lanes, so deletion is no longer a branch-only
  hygiene operation. Each needs owner/role disposition first.

Ownership buckets by branch/path naming:

- Active primary / special lanes: `/workspace/sounio`
  (`feat/hyper-epistemic-mul`), `/workspace/sounio-compiler-consolidation`
  (`integration/compiler-consolidation-20260628`), `main` detached verifier
  worktrees.
- Compiler integration lanes: `integration/native-v2-honest`,
  `integration/native-v2-onto-exact-orc`, `integration/modular-checker-e008`,
  `claude/checker-singlemodule-crashes`, `claude/codegen-largestruct-fix`,
  `claude/ir-heap-indirect`, `claude/mm-hardening`, parser lanes, source/enum
  wall lanes, and `integrate/kw-demote-landing`.
- Release/docs/tooling lanes: release worktrees, docs honesty, gate recovery,
  stdlib path, LSP revival, project spine, real language runner, package/docs
  registry, and website/fixes-from-main.
- Scientific/clinical/research lanes: PBPK integration, dissertation /
  vancomycin, epistemic tensor, affine/octonion work, solver SOTA, Fregni
  metrics, GPU-thread intrinsics, and viz/molecule authoring. Clinical and
  external-facing lanes require the repo offload policy before any content
  extraction or claim promotion.
- Scratch/detached lanes: detached verification/repro/scopeprobe worktrees and
  `.claude/worktrees/*` agent lanes. These should be removed only after explicit
  owner disposition or after proving they are disposable temp worktrees with no
  branch-only state.

### Detached temp cleanup

Three clean detached temporary worktrees were archived by commit tag and removed:

- `/tmp/sounio-origin-main-verify` at `63644b6b0`, archived as
  `archive/worktree-detached/sounio-origin-main-verify/2026-06-28`.
- `/workspace/sounio-mm-repro` at `590af4641`, archived as
  `archive/worktree-detached/sounio-mm-repro/2026-06-28`.
- `/workspace/sounio-scopeprobe` at `090767311`, archived as
  `archive/worktree-detached/sounio-scopeprobe/2026-06-28`.

The remaining detached worktree is
`/workspace/.home/openvscode-server/.cursor/worktrees/DMH2026-e3d71e55/sounio-c48641ba187e`
at `4aab38cd8`. It was left intact because it is under the Cursor-owned
worktree area rather than an obvious temporary repo path.

Post-cleanup counts: 49 local branches, 56 remote refs, 49 worktrees.

### Scratch settings-only worktree cleanup

The first `.claude/worktrees/*` dirty pass found six attached scratch lanes
whose only local dirt was `.claude/settings.local.json`. Because that file is a
local Claude settings surface rather than compiler/language WIP, each branch
tip was archived by tag, the dirty local settings diff was saved under
`docs/audit/archived_wip/`, and then the worktree plus local branch were
removed:

- `backlog/enum-ctor-check` at `810c7abf8`, archived as
  `archive/worktree/backlog-enum-ctor-check/2026-06-28`.
- `backlog/strconcat-emit-fix` at `d5bf9dd9e`, archived as
  `archive/worktree/backlog-strconcat-emit-fix/2026-06-28`.
- `honest/codex-calls` at `eae0d5134`, archived as
  `archive/worktree/honest-codex-calls/2026-06-28`.
- `wall/source-to-elf` at `0f0cc9df9`, archived as
  `archive/worktree/wall-source-to-elf/2026-06-28`.
- `wall/check-enumctor` at `a595ad29f`, archived as
  `archive/worktree/wall-check-enumctor/2026-06-28`.
- `worktree-agent-a203887be9ace9526` at `171a9f708`, archived as
  `archive/worktree/worktree-agent-a203887be9ace9526/2026-06-28`.

The preserved local WIP archives for this pass are:

- `docs/audit/archived_wip/backlog-enum-ctor-check-*2026-06-28.*`
- `docs/audit/archived_wip/backlog-strconcat-emit-fix-*2026-06-28.*`
- `docs/audit/archived_wip/honest-codex-calls-*2026-06-28.*`
- `docs/audit/archived_wip/wall-source-to-elf-*2026-06-28.*`
- `docs/audit/archived_wip/wall-check-enumctor-*2026-06-28.*`
- `docs/audit/archived_wip/worktree-agent-a203887be9ace9526-*2026-06-28.*`

Two dirty scratch lanes were intentionally preserved because they contain
non-settings WIP:

- `worktree-agent-adc1cd8b9d52ba53b` in
  `/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b` modifies
  `self-hosted/compiler/main.sio` and four `self-hosted/native/*` files.
  Status and binary diff were archived in
  `docs/audit/archived_wip/worktree-agent-adc1cd8b9d52ba53b-*2026-06-28.*`,
  but the worktree and branch remain live pending compiler-owner disposition.
- `xai/conversa` in `/workspace/sounio/.claude/worktrees/xai-conversa`
  modifies `README.md` and has three untracked audit/hyper-epistemic notes.
  Status, tracked diff, and copied untracked docs were archived under
  `docs/audit/archived_wip/xai-conversa-*2026-06-28.*` and
  `docs/audit/archived_wip/xai-conversa-untracked-2026-06-28/`, but the
  worktree and branch remain live pending owner disposition.

Post-cleanup counts: 44 local branches, 56 remote refs, 44 worktrees.

### Remote-only stale ref cleanup

After local no-worktree and scratch cleanup, 34 `origin/*` branches had no
matching local branch or active local worktree and showed stale rollback-shaped
diffs against consolidation. Each remote tip was archived as
`archive/remote/<branch-slug>/2026-06-28`, the archive tag was pushed, and then
the remote branch was deleted:

- `check/closure-hof-triple-e008`
- `check/e014-int-index-e008`
- `check/f32-field-narrowing-e008`
- `check/field-deref-ref-e008`
- `check/fn-type-lower-e008`
- `check/int-cross-width-e008`
- `check/linear-double-consume-e039`
- `check/ref-param-lower-e008`
- `check/refinement-types-e008`
- `claude/madaros-substrate-review`
- `claude/refine-local-plan-XoG73`
- `codegen/byval-arg-crasher`
- `codegen/deref-nested-store`
- `codegen/nested-mut-write-fix`
- `codex/direct-call-param-slot`
- `codex/madaros-language-reality-gate`
- `codex/madaros-raw-async-gates-20260627`
- `feat/erdos-straus-gpu-sieve`
- `feat/exact-orc-machinery`
- `feat/mc-v2-opcodes`
- `feat/native-v2-bridge-sret`
- `fix/frame-slot-recycling`
- `fix/native-selfhost-prebuilt`
- `integration/canonical-souc-gate-shepherd`
- `integration/consolidate-modular`
- `integration/e008-nested-store-complete`
- `modular/native-v2-e2e-gate`
- `modular/native-v2-source-to-elf`
- `parser/const-decls`
- `parser/extern-blocks`
- `parser/fn-type-effects-list-e008`
- `parser/kernel-fn`
- `parser/sci-notation-float`
- `parser/sci-notation-modular-e008`

`integration/sounio-dev-ready-base` was explicitly excluded from deletion
because `AGENTS.md` records it as the historical safe base branch for the
recovered workspace. `pr/232`, `pr/296`, and `pr/313` were also excluded
because they are PR refs, not normal `origin/*` branch heads.

Post-prune counts: 44 local branches, 22 remote refs, 44 worktrees.

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
