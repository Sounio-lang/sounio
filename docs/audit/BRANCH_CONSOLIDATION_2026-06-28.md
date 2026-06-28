# Sounio branch consolidation sweep - 2026-06-28

Status: active cleanup lane
Worktree: `/workspace/sounio-compiler-consolidation`
Branch: `integration/compiler-consolidation-20260628`
Base: `origin/main` at `63644b6b0`

## What changed in this sweep

- Created the compiler consolidation branch from `origin/main`.
- Pruned remote refs with `git fetch --prune`.
- Deleted 28 local branches that were already contained in `origin/main` and had no active worktree.
- Deleted 48 additional local branches with no patch-id-exclusive commits versus `origin/main` and no active worktree.
- Deleted 26 merged `origin/*` branches that had no active local worktree.
- Deleted 9 stale local remote-tracking refs under `refs/remotes/local/*`.

Branch counts observed in this session:

- Before cleanup after fetch: 204 local branches, 135 remote refs.
- After cleanup: 129 local branches, 100 remote refs.
- Remaining local branch audit: 128 named branches plus the detached `/workspace/sounio` checkout context.

## Current ownership split

Do not use `/workspace/sounio` as the integration surface. As of the 2026-06-28
refresh it is on `feat/hyper-epistemic-mul` at `2973e2130` and should remain a
coordination/review surface for that experimental branch, not the compiler
consolidation landing lane.

The consolidation surface is:

```text
Lane: compiler consolidation
Owner: Codex
Base: origin/main
Worktree: /workspace/sounio-compiler-consolidation
Branch: integration/compiler-consolidation-20260628
Write-Set: docs/audit/BRANCH_CONSOLIDATION_2026-06-28.md, future curated compiler merges
Read-Set: all branches/worktrees
Required-Gates: bin/souc info; bash scripts/run_sio_test_suite.sh hello --verbose before any compiler-code merge
Merge-Target: main, after conflicts and gates are explicit
Known-Blockers: many dirty worktrees remain; no broad compiler merge without lane owner transfer
```

## Remaining branch classes

### Active worktree branches

There are 68 local branches attached to worktrees; 54 of those worktrees are
dirty. They are not deletion candidates until each owner either lands, archives,
or explicitly abandons the worktree.

High-priority dirty compiler worktrees include:

- `/workspace/sounio-project-spine` on `codex/project-spine-madaros`, dirty=10.
- `/workspace/sounio-source-elf-proof` on `codex/source-elf-on-madaros-proof`, dirty=5.
- `/workspace/sounio-madaros-check-segv` on `codex/madaros-full-functioning`, dirty=1.
- `/workspace/sounio-ir` on `claude/ir-heap-indirect`, dirty=7.
- `/workspace/sounio-effects` on `claude/effects-enforcement`, dirty=8.
- `/workspace/sounio-gpu-kernel` on `feat/gpu-thread-intrinsics`, dirty=4.
- `/workspace/sounio-solver-sota` on `research/solver-sota-class`, dirty=1 and had a live Slurm solver job during recovery.

### Review before integration

These branches still have patch-exclusive commits and compiler-surface diffs,
but are not safe for automatic merge:

- `fix/madaros-tuple-let-desugar`: broad compiler + GPU/solver/docs merge train; direct merge conflicts in `check`, `module_frontend`, `lower`, `codegen`, and witness scripts.
- `docs/pbpk-session-notes-2026-06-28`: same broad train plus audit/docs content; review separately from compiler code.
- `codex/madaros-retire-lean-single-20260627` and `codex/madaros-close-20260627`: overlapping Madaros closeout trains; likely supersede some older branches, but need gate-by-gate merge.
- `claude/solver-gpu-native-path` and `claude/gpu-e137-fix`: solver/GPU path branches; keep out of compiler core until ownership is assigned.
- `fix/binop-literal-float-478b`: reviewed and rejected as a compiler merge candidate. It contains only `DEBUG:` commits in `self-hosted/native/codegen_x86_linux.sio` that reshape `SOUNIO_NV2_IR_TRACE` printing; it does not contain a semantic fix for binop literal floats. Keep only as forensic trace history unless a real #478 fix is identified.
- `feat/hyper-epistemic-mul`: reviewed and rejected for immediate consolidation. It adds `IrHyperEpistemicMul` backend/Metal pieces, but the opcode is not produced by the normal compiler path, no `ir_hyper_epistemic_*` constructor or focused gate was found, and the committed audit/help text still describes the rejected associator-correction variant. Keep as an experimental/review lane until the audit doc is corrected, the user-visible help text matches the v2 formula, and either a real producer plus gate lands or the branch is explicitly scoped as backend-only prototype.
- `codex/tuple-signature-types-20260626`: reviewed. The parser, checker, native implicit-unit/assert, and native bool-literal fixes are already equivalent in the consolidation branch; remaining exclusive commits are GPU/script/test-bookkeeping and should not drive the compiler-core merge.
- `codex/madaros-import-stdlib-lowering-current`: older imported-stdlib lowering branch; conflicts with the newer imported-lowering path already on `origin/main`.
- `codex/imported-full-lowerer-20260627`: reviewed. Its only patch-exclusive commit adds `scripts/ci/madaros_multimodule_witness.sh` to the Madaros prebuilt refresh gate, but that witness is currently red on consolidation (`thin_single` exits 139 at `lower_array: seed_begin`). Do not wire it into prebuilt refresh until the imported multimodule lowerer residual is actually fixed.
- `codex/madaros-boxnew-clean` and `codex/madaros-boxnew-append-fix`: reviewed
  again after the substrate ultrareview micro-port. Current consolidation already
  has the non-debug Box::new/flat-lowering/validated-patching fixes in newer
  form; remaining branch-exclusive commits are debug markers, stale bridge
  rewrites, and bootstrap binary refreshes.

### Archive/quarantine

These are too broad or old to merge automatically:

- `fix/flatparse-and-scan-operator`
- `codex/gpu-semantic-profile`
- `codex/gpu-modular-bridge`
- `claude/kw-demote-module`
- `integration/effects-kwdemote`
- `research/erdos-compiler-wip`

Keep them as archaeological branches until a specific missing feature is named.

### Effects / kw-demote / legacy self-host trains

Reviewed:

- `origin/claude/effects-enforcement`
- `origin/claude/kw-demote-module`
- `origin/integration/effects-kwdemote`
- `origin/integrate/kw-demote-landing`
- `origin/campaign/mc-frontend-fixes`
- `origin/claude/ir-heap-indirect`
- `origin/claude/codegen-largestruct-fix`

Do not merge these branches as branches. They are all non-ancestors of current
consolidation and their raw diffs are old-base rollback shapes:

- effects/kw-demote lanes: about 2,976-2,986 files, about 294k-296k insertions
  and about 670k-671k deletions.
- kw-demote landing / mc frontend campaign: 2,678 files, about 280k insertions
  and about 660k deletions.
- `ir-heap-indirect`: 2,397 files, 11,391 insertions and 254,795 deletions.
- `codegen-largestruct-fix`: 2,887 files, 30,745 insertions and 338,597
  deletions.

Current consolidation already carries several central ideas from these trains
in newer form: `IR_MAX_FUNCS=2048`, `local_cap() = 2048`, checker `slice_len`
support, `load_multimodule_ir_into`, raw-ELF wrapper routing, effect violation
checks/effect-polymorphic call propagation, and `a64_preview_record_call_patch`
marked `with Mut` in the active native codegen path.

Classification:

- Treat effects/kw-demote and mc-frontend branches as historical provenance,
  not merge parents.
- Keep uncommitted worktrees `/workspace/sounio-effects` and
  `/workspace/sounio-ir` as owner lanes until their dirty diffs are explicitly
  archived or transferred.
- If one of these branches contains a still-missing fix, it must be extracted
  as a named single-purpose patch with a focused acceptance gate on
  `integration/compiler-consolidation-20260628`.

### Epistemic / affine / proof / future-work lanes

Reviewed:

- `origin/feat/affine-nonassoc-uncertainty`
- `origin/feat/affine-octonion-correlation`
- `feat/affine-octonion-clean`
- `feat/assoc-variance-clean`
- `feat/assoc-variance-wiring`
- `origin/feat/assoc-variance-clean`
- `origin/fix/assoc-variance-273-consolidated`
- `origin/m2/effect-firewall`
- `origin/recover/m2-effect-firewall`
- `feat/lean-tier2`
- `feat/future-work-first-slices`
- `origin/feat/future-work-first-slices`

Do not merge these branches as compiler-core branches. They are semantic,
epistemic, proof, effect-policy, GPU, or future-work lanes rather than cleanup
sources for the single compiler consolidation branch. Their raw diffs are also
old-base rollback shapes: affine/assoc branches range from ~1,373 to ~2,894
files; M2/effect-firewall branches are ~1,368-1,379 files; Lean/future-work
branches are ~2,390-2,411 files. Several would roll back hundreds of thousands
of lines relative to the current consolidation state.

Status notes:

- `origin/fix/assoc-variance-273-consolidated` is already an ancestor of the
  current consolidation branch; no branch-tip integration remains.
- `feat/assoc-variance-wiring` is build-lock/pod policy history, not a compiler
  source merge.
- `m2/effect-firewall` has a live worktree and should stay an owner lane until
  explicitly transferred.
- `lean-tier2` and `future-work-first-slices` include proof/GPU/future-work
  material. They require their own proof/offload policy lane if revived; this
  consolidation audit is not a validation of their mathematical claims.

Classification: keep these as research/semantics/proof provenance, not as
merge parents for `integration/compiler-consolidation-20260628`.

### Codegen / G1 / GPU / CI residue triage

Reviewed:

- `origin/codegen/byval-arg-crasher`
- `origin/codegen/deref-nested-store`
- `origin/codegen/nested-mut-write-fix`
- `origin/d3-a/missing-imports-sweep`
- `origin/g1/qualify-bare-patterns`
- `origin/gpu/epistemic-tensor-core-gum-sm75`
- `origin/feat/gpu-thread-intrinsics`
- `origin/feat/exact-orc-machinery`
- `origin/feat/erdos-straus-gpu-sieve`
- `origin/fix/flatparse-and-scan-operator`
- `origin/fix/main-ci-red-2026-06-18`
- `origin/fix/website-quality-check`

Do not merge these branches as branches. Their raw diffs range from old G1
rollback shapes (~2,812 files / ~716k-718k deletions), to GPU/archive trains
(`feat/erdos-straus-gpu-sieve`: 7,330 files and 3.2M deletions), to CI/website
repair branches that would roll back large parts of the current repo. They mix
compiler code, docs, GPU research, generated artifacts, website content, and
historical audits.

Status notes:

- `origin/feat/madaros-bump-arena` and `origin/fix/root2-enum-inplace` are
  already ancestors of the current consolidation branch; no integration action
  remains for their branch tips.
- `origin/codegen/*` and `origin/g1/qualify-bare-patterns` are G1/codegen
  provenance. Current consolidation already documents the relevant SRET,
  nested-store, and boundary-check surfaces elsewhere in this audit; revive only
  with a fresh failing witness.
- `origin/feat/gpu-thread-intrinsics`, `origin/gpu/epistemic-tensor-core-gum-sm75`,
  `origin/feat/erdos-straus-gpu-sieve`, and
  `origin/feat/exact-orc-machinery` are GPU/research/backend lanes, not
  compiler-core cleanup branches.
- `origin/fix/main-ci-red-2026-06-18` and `origin/fix/website-quality-check`
  are CI/website repair history, not compiler consolidation sources.

Validation after this triage:

- `bash scripts/run_sio_test_suite.sh hello --verbose`: PASS 2/2.

## Merge attempts made

Attempted `git merge --no-ff fix/madaros-tuple-let-desugar` in the consolidation
worktree. It conflicted in:

- `.claude/llm_offload_log.md`
- `scripts/ci/madaros_multimodule_witness.sh`
- `self-hosted/check/check.sio`
- `self-hosted/check/mod.sio`
- `self-hosted/compiler/main.sio`
- `self-hosted/compiler/module_frontend.sio`
- `self-hosted/ir/lower.sio`
- `self-hosted/native/codegen_x86_linux.sio`

The merge was aborted because it mixed compiler, GPU/solver, docs, and audit-log
state in one step.

Attempted focused cherry-picks:

- `4dd66d8e7` tuple-let desugar: skipped as redundant; `origin/main` already has tuple-let plus newer struct-let handling.
- `2bd304962` println f64 dispatch: skipped as redundant/equivalent; current `origin/main` already has f64 dispatch and tests.
- `36a3b1d2b`, `3158a46b3`, `4eb67966c`, `95b16f24b`: skipped as already represented by current `origin/main`.
- `220f5b8f1` imported runtime lowering: skipped after resolving showed the remaining conflict would downgrade current `with_externs`/call-target remap behavior.
- `542a91e31` imported stdlib lowering: aborted; too old and broad against current imported-lowering architecture.
- `9726be6f4` parser compound-assignment RHS: skipped as empty/equivalent in the consolidation branch.
- `07ecfa0ff` contextual int array literals: conflict only in tests already marked `//@ requires: madaros`; after resolving to keep the marker, the patch was empty/equivalent.
- `ac60fc11f` native implicit unit returns and assert builtin: conflict only in `tests/madaros/source_to_elf/manifest.tsv`, whose entries were already present; after resolving, the patch was empty/equivalent.
- `eaf803935` native bool literals: skipped as empty/equivalent in the consolidation branch.
- `71387df9a` raw async runtime gates: ported as compiler-only code from `codex/madaros-retire-lean-single-20260627`. Conflicts were resolved by preserving the current Box::new, println/unobserved, and configurable multimodule visibility behavior while adding async runtime call typing, TaskHandle/spawn/await lowering, raw-array native lowering, and `IrSyscall6` native emission. The branch merge commit `933da3c4a` was not merged because it also carries solver/governance/history and a bootstrap binary update.
- `85aaadccd`, `442031ed9`, `3fd937bd1` from `fix/binop-literal-float-478b`: not ported. All three are debug instrumentation for native-v2 IR tracing, including direct deep-field reads for investigation of #478, but no compiler semantic change or focused acceptance gate.
- `2286fb6d5` native-v2 memory-operand float arithmetic: ported in adapted form from `integration/native-v2-honest`/related branches. The old commit's `LowerLocalStack.is_float` model was not copied because the consolidation branch already has newer `scalar_kind` and `array_elem_float` tracking. The port adds `ir_binop_float`, tags real float binary expressions via existing `expr_result_is_float_ref`, and routes `IrBinOp` with imm_flags bit 1 to the SSE path. The old `scripts/ci/release_gate.sh` edit was intentionally not restored. The f32 testcase from the old gate was not ported because current Madaros rejects implicit f64-to-f32 literal arguments and explicit f32 casts still exit 0; f32 remains a separate residual, while the f64 memory-operand bug is covered.
- `c2a783f27` struct-field float arithmetic: compiler code was already absorbed by newer consolidation architecture (`StructFieldEntry.is_float`, struct-type local tracking, field float lookup, and `ir_binop_float` tagging). Only the focused gate was ported/adapted. The old f32 struct-field case was not ported because current Madaros rejects implicit f64 literals in f32 struct fields; f32 remains a separate residual. The old `scripts/ci/release_gate.sh` edit was intentionally not restored.

Dirty WIP extraction attempts:

- `/workspace/sounio-source-elf-proof`: reviewed dirty `check`, `defs`, `mod`, and `parser/exprs` changes. The valuable checker capacity/import/runtime-builtin and parser control-expression newline handling are already present in the consolidation branch. Copying this worktree would downgrade newer parser guards for newline `[`/`(`/prefix operators and newer checker architecture, so no code was ported.
- `/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b`: reviewed dirty native-codegen changes. `self-hosted/native/lower_ir.sio` is byte-identical to the consolidation branch, and all inspected call sites already use `&! NativeCompiler`. The only remaining difference changes a rodata constant in `self-hosted/native/codegen.sio` from `1e6` to another value without a clear acceptance gate, so it was not ported.

Validation after porting `71387df9a`:

- `bin/souc info`: PASS, selected Madares v0.80.0.
- `bash scripts/run_sio_test_suite.sh hello --verbose`: PASS, 2 passed / 0 failed.
- `bash scripts/ci/madaros_source_to_elf_gate.sh`: PASS, including check, trace, normal/native-v2 compile, ELF execution, and exit-code semantics.

Validation after porting adapted `2286fb6d5`:

- `bash tests/native_v2_float_arith_gate/run.sh bin/souc`: PASS, 10 passed / 0 failed. Covered f64 param +,-,*,/, f64 array +,*, memory+literal control, and integer controls.
- `bin/souc info`: PASS, selected Madares v0.80.0.
- `bash scripts/run_sio_test_suite.sh hello --verbose`: PASS, 2 passed / 0 failed.
- `bash scripts/ci/madaros_source_to_elf_gate.sh`: PASS, including check, trace, normal/native-v2 compile, ELF execution, and exit-code semantics.

Validation after absorbing/adapting `c2a783f27`:

- `bash tests/native_v2_struct_field_float_gate/run.sh bin/souc`: PASS, 7 passed / 0 failed. Covered f64 struct-field +/*, let annotated/inferred struct locals, mixed int/float struct controls, and integer field controls.
- `bin/souc info`: PASS, selected Madares v0.80.0.
- `bash scripts/run_sio_test_suite.sh hello --verbose`: PASS, 2 passed / 0 failed.
- `bash scripts/ci/madaros_source_to_elf_gate.sh`: PASS, including check, trace, normal/native-v2 compile, ELF execution, and exit-code semantics.

## Next clean consolidation path

1. Keep `origin/main` as the compiler baseline, not any old WIP train.
2. Merge only one lane at a time into `integration/compiler-consolidation-20260628`.
3. First real candidates:
   - remaining compiler-core commits from `codex/madaros-retire-lean-single-20260627` only if a missing gate is named; the raw async compiler patch has been extracted.
   - remaining compiler-core branches with patch-exclusive code not yet classified; prioritize branches with focused gates over broad solver/GPU trains.
4. For each candidate, require:
   - no active worktree owner collision,
   - named conflict files,
   - `bin/souc info`,
   - `bash scripts/run_sio_test_suite.sh hello --verbose`,
   - the candidate-specific gate named in its audit doc.

## Current blocker record

```text
Blocker-ID: BLK-20260628-compiler-consolidation-dirty-worktrees
Status: classified
Severity: B1
Class: ownership-conflict
Owner: Codex coordination until lane owners transfer or close worktrees
Lane: compiler consolidation
Worktree: /workspace/sounio-compiler-consolidation
Branch: integration/compiler-consolidation-20260628
Files-Owned: docs/audit/BRANCH_CONSOLIDATION_2026-06-28.md
Files-Read-Only: remaining dirty worktrees and compiler branches
Do-Not-Touch: /workspace/sounio detached checkout; dirty compiler worktrees listed above
Repro: git worktree list --porcelain && git status --short --branch in each worktree
Observed: 68 worktree branches remain; 54 are dirty
Expected: one owner per dirty lane before compiler-code integration
Acceptance-Gate: active owner map plus green focused gate after each merge
Evidence-Level: E2
Evidence: /tmp/sounio_remaining_branch_audit_20260628.tsv
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: not-required
Next-Action: choose one candidate branch and assign exclusive ownership for conflict resolution
```

## Second-pass dirty worktree triage

Follow-up command:

```bash
for d in /workspace/sounio-* /workspace/.cnp-phaseB /workspace/tmp/* /workspace/sounio/.claude/worktrees/*; do
  git -C "$d" status --short
done
```

Live-process check found active processes in `/workspace/sounio` and
`/workspace/sounio-solver-sota`; the solver lane also had Slurm job `4487`
running as `q13k11` on `cpu-ops`.

### Do not touch while live

- `/workspace/sounio-solver-sota`
  - Branch: `research/solver-sota-class`
  - Dirty: `slurm-jobs/queen-sat-assault/`
  - Reason: live Slurm solver job observed.

### Dirty mechanical noise repeated across many worktrees

Many worktrees are dirty only because of:

- `M slurm-jobs/erdos90/run_on_cluster.sh`
- `?? slurm-jobs/erdos90/run_on_cluster.sh.bak-cpuopsfix`

These should be handled by one coordinated policy, not lane-by-lane guessing:

1. Decide whether the CPU ops fix belongs on `main`.
2. If yes, land it once in the consolidation branch.
3. If no, restore/remove the repeated local noise only after the lane owner agrees.

Affected examples include:

- `/workspace/sounio-affine`
- `/workspace/sounio-affine-pg`
- `/workspace/sounio-baseline-contracts`
- `/workspace/sounio-checker`
- `/workspace/sounio-docs`
- `/workspace/sounio-fnptr-integ`
- `/workspace/sounio-gates`
- `/workspace/sounio-lsp-work`
- `/workspace/sounio-mmh`
- `/workspace/sounio-parser`
- `/workspace/sounio-parser-integ`
- `/workspace/sounio-pbpk-integration`
- `/workspace/sounio-real-runner`
- `/workspace/sounio-rel-eval`
- `/workspace/sounio-relmc`
- `/workspace/sounio-showcase`
- `/workspace/sounio-stdlibpath`
- `/workspace/sounio-strcat`
- `/workspace/sounio-viz-molecule-authoring`

### High-value compiler WIP, preserve until reviewed

These worktrees have actual compiler-core modifications or artifacts that may
matter:

- `/workspace/sounio-project-spine`
  - Branch: `codex/project-spine-madaros`
  - Dirty core files: `self-hosted/check/check.sio`, `self-hosted/check/defs.sio`, `self-hosted/parser/stmts.sio`
  - Notes: likely the highest-value local WIP, but needs owner review before extraction.

- `/workspace/sounio-source-elf-proof`
  - Branch: `codex/source-elf-on-madaros-proof`
  - Dirty core files: `self-hosted/check/check.sio`, `self-hosted/check/defs.sio`, `self-hosted/check/mod.sio`, `self-hosted/parser/exprs.sio`
  - Notes: `patch_plus=0` against `origin/main`, so committed history is redundant, but dirty files may contain uncommitted work.

- `/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b`
  - Branch: `worktree-agent-adc1cd8b9d52ba53b`
  - Dirty core files: `self-hosted/compiler/main.sio`, `self-hosted/native/codegen.sio`, `self-hosted/native/codegen_x86_linux.sio`, `self-hosted/native/lower_ir.sio`, `self-hosted/native/suite.sio`
  - Notes: `patch_plus=0`, but dirty native-codegen edits need diff review.

- `/workspace/sounio-gpu-kernel`
  - Branch: `feat/gpu-thread-intrinsics`
  - Dirty artifacts: `artifacts/omega/vadd.ptx`, `artifacts/self-hosted/madaros-*`
  - Notes: artifacts only in dirty state; committed branch has GPU intrinsic work.

- `/workspace/sounio-ir`
  - Branch: `claude/ir-heap-indirect`
  - Dirty docs/artifacts: heap-indirect plans and `test_fill_repro.sio`
  - Notes: likely design/repro material more than immediate compiler merge.

- `/workspace/sounio-effects`
  - Branch: `claude/effects-enforcement`
  - Dirty artifacts/scripts: effects reports, ELFs, `scripts/dev/effects_annotate.sh`
  - Notes: preserve as evidence lane.

### Likely cleanup/close candidates after owner ack

These have `patch_plus=0` and dirty content that appears artifact/report-only or
mechanical:

- `/workspace/sounio-f64cmp`
- `/workspace/sounio-integ`
- `/workspace/sounio-madaros-check-segv`
- `/workspace/sounio-madaros-main-proof`
- `/workspace/sounio-madaros-source-elf-consolidated`
- `/workspace/sounio-tests-fix`
- `/workspace/sounio/.claude/worktrees/madaros-default`

Do not delete them yet. Next step is to archive their dirty diff or get explicit
owner approval to drop it.

### Recommended next manual action

Start with `/workspace/sounio-project-spine` because it has real compiler-core
dirty files and patch-exclusive branch history. The first extraction should be
read-only:

```bash
git -C /workspace/sounio-project-spine diff -- self-hosted/check/check.sio self-hosted/check/defs.sio self-hosted/parser/stmts.sio
```

Then either:

- cherry-pick the branch commits into `integration/compiler-consolidation-20260628`, or
- manually port only the still-missing hunks from the dirty diff.

### Project-spine extraction result

Reviewed `/workspace/sounio-project-spine` dirty core diff against
`integration/compiler-consolidation-20260628`.

Ported:

- `self-hosted/parser/stmts.sio`: removed the assignment-parser one-token
  lookahead fallback. The branch-local note is correct: only an assignment
  operator at the current parser position should bind the parsed expression;
  looking one token ahead can consume the next assignment as if the previous
  expression were the assignment target.

Not ported:

- `self-hosted/check/check.sio`: the useful ideas from the dirty diff are
  already present in the consolidation branch in newer form: `TypeFn` lowering,
  builtin `print_int`/`print_char` handling, and function-signature support.
- `self-hosted/check/defs.sio`: `fn_sig_table_get` already bounds-checks and
  returns `empty_fn_sig()` in the current chunked table implementation.
- `artifacts/omega/*.selftest.bin`: binary artifacts, not compiler source.
- `slurm-jobs/erdos90/run_on_cluster.sh` and `.bak-cpuopsfix`: repeated
  mechanical Slurm noise; keep out of compiler consolidation until that policy
  is decided globally.

Validation after port:

- `bin/souc info`: OK, `Madares v0.80.0`.
- `bash scripts/run_sio_test_suite.sh hello --verbose`: PASS 2 / FAIL 0.
- `bash scripts/run_sio_test_suite.sh assignment --verbose`: PASS 4 / FAIL 2.
  The same two failures reproduce on a clean `origin/main` worktree
  (`gtt_reassignment_wrong_channel.sio` expected compile-fail but passed;
  `gtt_reassignment_topology.sio` exited 139), so this is baseline noise rather
  than a regression from the parser hunk.

## Absorbed compiler fixes with focused gates

Reviewed two small non-ancestor compiler branches:

- `fix/madaros-for-loop-lowering` (`6dde85913`): current consolidation already
  contains `lower_for_in_expr_ref` and dispatches `ExprForIn` through
  `lower_expr_ref`.
- `fix/madaros-print-int-dispatch` (`c152ac3b4`): current consolidation already
  contains the older int-literal redirect and a stronger `scalar_kind`-based
  `println_dispatch_name` covering int variables and f64 values.

No old hunks were cherry-picked. Instead, added
`tests/native_v2_loop_print_gate/` to lock the absorbed behavior with native-v2
ELF execution: half-open range, inclusive range, `continue`, `break`,
`println(42)`, `print(7)`, `println(i64 variable)`, `print(i64 variable)`, and
`println/print(f64 variable)`.

Follow-up triage on the same small-fix cluster:

- `fix/madaros-for-loop-lowering`: `git cherry -v HEAD` marks
  `6dde85913` as patch-equivalent (`-`), so there is no remaining branch-owned
  delta to port.
- `fix/madaros-print-int-dispatch`: `c152ac3b4` is not patch-id equivalent, but
  the current `scalar_kind`-based dispatch is strictly broader than its
  int-literal redirect. Validation: `bash tests/native_v2_loop_print_gate/run.sh
  bin/souc` passed 7/7 (`p01_for_range_sum`, `p02_for_inclusive_range_sum`,
  `p03_for_continue`, `p04_for_break`, `p05_print_int_literal`,
  `p06_print_int_variable`, `p07_print_float_variable`).
- `origin/fix/test-suite-epistemic-failures`: already contained by ancestry.
  `git merge-base --is-ancestor origin/fix/test-suite-epistemic-failures HEAD`
  returned rc 0, and `git cherry -v HEAD
  origin/fix/test-suite-epistemic-failures` was empty.
- `fix/ocp-locals-cap`: the first two OCP split commits are
  patch-equivalent/already absorbed (`ocp_const_fold_pass_a1`,
  `ocp_const_fold_pass_a2`, `ocp_const_fold_pass_b`). The remaining top commit
  (`ba02961ed`) is not cleanup-safe: `git diff --stat HEAD..fix/ocp-locals-cap`
  spans 2,796 files with 297,647 insertions and 686,982 deletions, including
  current audit/gate deletion and broad binary/docs churn. Do not merge this
  branch into consolidation. Residual ownership, if revived, should be a named
  wide-int/SRET/class-2-wall lane with a focused gate; the bool/i64 `ty_eq`
  compatibility idea is already represented in the earlier consolidation port.
- `fix/silent-typecheck-diag`: not an ancestor and not patch-equivalent. The
  branch is really a merged A64 residual train (`fix/a64-emit`,
  `fix(lean_single)`, rebuilt self-hosted binaries, debug docs, and three new
  run-pass witnesses), not a silent-diag leaf. `git diff --stat
  HEAD..fix/silent-typecheck-diag` spans 2,581 files with 282,256 insertions
  and 650,018 deletions, including deletion of current consolidation audit/docs
  and broad website/test/artifact churn. Do not merge into the compiler
  consolidation branch. Residual ownership should stay with an A64/native
  runtime lane and a host-specific acceptance gate; the three branch-only
  witnesses are `f64_gt_method_a64.sio`, `knowledge_arith_a64.sio`, and
  `struct_shorthand_and_nested_array_store.sio`.

## Imported stdlib lowering branch triage

Reviewed `codex/madaros-import-stdlib-lowering-current` (`542a91e31`).

Not ported yet: the large `self-hosted/compiler/module_frontend.sio` and
`self-hosted/ir/lower.sio` imported-lowering machinery. It is a broad subsystem
change, not a cleanup cherry-pick, and should land only with a dedicated
multimodule/stdlib gate proving the exact residual it fixes.

Attempted the branch's apparently low-risk test-only pieces, but did not keep
them because current consolidation does not yet pass them:

- Adapting `tests/stdlib/core/test_result_e2e.sio` from unsupported method calls
  to free functions moves the test past checker errors but hits a current
  Madaros SIGSEGV during imported IR merge (`lower_array: seed_begin`, rc 139).
- Adding `tests/stdlib/csv/test_csv_parse.sio` exposes current stdlib/import
  parse/type failures before native-v2 ELF emission.

Treat this branch as a residual owner lane, not as absorbed cleanup.

## SRET/codegen branch triage

Reviewed `fix/native-codegen-sret-regression` (`8f537aac1`).

Current consolidation already contains the relevant `lean_single.sio` fixes:

- struct-literal shorthand rewind via `shorthand_ep`
- `compile_value_field_field_array_store_x86`
- `compile_stmt` dispatch for `stmt_is_field_field_array_store_shape(EP)` with
  pointer and value-struct roots

No cherry-pick needed. The sibling `fix/native-codegen-sret-regression-v2`
diff is not a small SRET follow-up; it includes a broad AMD HIP/ROCm GPU train
(`bin/kretikos`, `scripts/amd/*`, `self-hosted/gpu/*`, `tests/amd/*`) plus
compiler driver edits. Keep that as a GPU/backend owner lane, not part of this
compiler-core consolidation pass.

## BoxNew branch triage

Reviewed `codex/madaros-boxnew-clean` and
`codex/madaros-boxnew-append-fix`.

Current consolidation already contains the important non-debug BoxNew fixes in
newer form:

- `Box::new(...)` allocates the destination register before lowering its value
  argument.
- `let b = Box::new(...)` does not misclassify `Box` as the inner call return
  struct; the local is tagged as `Box` so dereference lowering can use field 0.
- flat summary/body lowering no longer applies validated-call patching in the
  flat body path.
- validated-call patching mutates functions through boxed module references
  rather than detached local copies.

Not ported:

- branch-local debug marker writes under `/tmp/lower_*`
- `bin/madaros-linux-x86_64` prebuilt refresh
- broad native bridge rewrites from the append branch; current consolidation
  already routes native-v2 through `compile_native_v2_preview_to_file`.

Validation on current consolidation:

- `bin/souc --native-v2-compile tests/run-pass/door1_box_new_array_65536.sio
  -o /tmp/sounio_box_array.elf` emitted an ELF; running it exited 0 and printed
  `PASS: box array`.
- `bin/souc --native-v2-compile tests/run-pass/type_hash_3level_nesting.sio
  -o /tmp/sounio_type_hash_box.elf` emitted an ELF; running it exited 0 and
  printed `type-hash 3-level PASS`.
- Re-run after `e04d3ec44`: `door1_box_new_array_65536.sio` emitted
  `/tmp/sounio_box_array_current.elf`; after `chmod +x`, running it printed
  `PASS: box array` and exited 0.
- Re-run after `e04d3ec44`: `type_hash_3level_nesting.sio` emitted
  `/tmp/sounio_type_hash_current.elf`; after `chmod +x`, running it printed
  `type-hash 3-level PASS` and exited 0.
- `bash scripts/run_sio_test_suite.sh door1_box_new_array_65536 --verbose`:
  PASS 1/1.
- `bin/madaros --self-test | rg 'T10|T11|validated|FAIL'` is not a BoxNew
  acceptance gate. It currently stops earlier with `FAIL: T08 ir_opt dispatch
  and merge layout`, `FAIL: T09 ir_opt remaps cloned branch targets`, and a
  raw-Madaros segmentation fault before the validated-chain lines are reached.
  Keep that as a separate `ir_opt` self-test residual, not evidence against the
  BoxNew branches.

## Frame noop and Root2 global branch triage

Reviewed `fix/revert-frame-noop` (`b33ef5593`).

Current consolidation already has the reverted/no-op state: the vestigial
`native_v2_core_begin_function_from_ir_into` body emits fixed `sub rsp, 512`,
and `rg "native_v2_core_begin_function_from_ir_into\\("` finds only the
definition. No code change needed.

Reviewed `codex/root2-global-lookup-probe` and
`codex/root2-global-preload-probe`.

Current consolidation already contains the non-debug global lowering fixes:

- `lowerer_lookup_fn_id_by_name_ref`
- BSS global metadata on `LowerLocalStack`
- `preload_bss_globals_as_locals`
- global-local store-back through `IrStoreGlobal`

Added `tests/native_v2_global_bss_gate/` to lock the absorbed behavior with
native-v2 ELF execution for scalar global read/write.

Residual found while trying to broaden the gate: global array element
read/write (`var GLOBAL_VALUES: [i64; 4]`) compiles to an ELF but the ELF exits
139 when writing/reading elements. That case is not marked absorbed.

## Tuple-signature branch triage

Reviewed `codex/tuple-signature-types-20260626`.

Current consolidation already contains the compiler fixes for the branch's core
cases:

- contextual int array literal bindings in the checker
- implicit unit returns in IR lowering
- native-v2 assert builtin and bool literal emission
- compound assignment RHS parsing for blocks, if-expressions, and nested parens

Added `tests/native_v2_tuple_signature_absorbed_gate/` to lock those absorbed
behaviors. The gate uses native-v2 ELF execution for assert/bool/unit cases and
`--check` for checker/parser cases, including the negative i8 array-repeat
diagnostic.

Not ported in this pass: the branch-local Metal opcode smoke script adjustment
and known-failures list edits. Those are GPU/test-harness ownership, not core
compiler consolidation.

## Semcall/HOF branch triage

Reviewed `codex/semcall-hof-main`.

Patch-id showed four commits already equivalent to consolidation:

- `960e68aa` source return semantics
- `4f8fc8e` higher-order source execution
- `dce38a6f` / `e613b648` `opt_cleanup` bootstrap split churn

The remaining semcall surface also passes on current consolidation without
porting old broad hunks. Added `tests/native_v2_semcall_absorbed_gate/` with
native-v2 ELF execution for:

- literal return `42`
- direct call `double(21)`
- returned function reference call through `let f = choose(0); f(21)`
- if-without-else followed by assignment, guarding the newline/assignment parse
  shape used in the old selfhost native runtime manifest

Not ported in this pass: the old `scripts/ci/madaros_source_to_elf_gate.sh`
replacement and broad `opt_cleanup.sio` re-splits; the current consolidation
already has the canonical source-to-ELF gate passing.

## Hyper-epistemic multiplication branch triage

Reviewed `feat/hyper-epistemic-mul` (`2973e2130`).

Do not absorb into `integration/compiler-consolidation-20260628` yet.

Findings:

- The branch adds `IrHyperEpistemicMul` to the IR enum, native dispatch/lowering,
  and a Metal emission path, but no normal compiler producer was found. Searches
  for `IrHyperEpistemicMul`, `ir_hyper_epistemic`, `lower_hyper_epistemic`, and
  `hyper_epistemic_mul` found enum/backend/docs/tool-pattern references only;
  unlike older hyper opcodes, there is no constructor analogous to
  `ir_hyper_mul_o` and no source-to-IR lowering path.
- The audit document committed on the branch still describes the earlier
  associator-correction design even though the commit message and code comments
  say that formula was rejected and removed. That makes the branch unsafe as an
  external-facing math/audit artifact.
- `self-hosted/gpu/kretikos_emit_metal.sio` still has user-visible help text
  claiming associator correction, which conflicts with the v2 GUM variance
  implementation.
- No focused gate was found for either the native-v2 opcode path or the Metal
  pattern emission path.

Validation attempted on the branch:

- `bin/souc info`: PASS, identifies `/workspace/sounio/bin/madaros` as the
  selected wrapper and `Madares v0.80.0` as the compiler identity.
- `bash scripts/run_sio_test_suite.sh hello --verbose`: PASS 2/2 on the branch.
- `bin/souc --check self-hosted/gpu/metal.sio`: FAIL on the current branch
  surface with local type/undeclared-variable diagnostics. This makes the new
  Metal emitter path unfit for absorption without a focused repair/golden gate.
- `bin/souc --check self-hosted/gpu/kretikos_emit_metal.sio`: FAIL on the
  current branch surface. The failure was not attributed to this commit because
  it was not parent-compared and appears mixed with existing broad check noise.
- `bin/souc --check self-hosted/native/lower_ir.sio`: FAIL on the current branch
  surface. Likewise, not attributed to this commit without parent comparison.

Required before reconsidering for consolidation:

- Rewrite the audit doc so it contains only the accepted v2 formula and honest
  backend scope.
- Fix the Metal/kretikos help text to remove associator-correction claims.
- Add either a real source/IR producer plus a focused native-v2 gate, or mark the
  branch explicitly as a backend-only prototype with a manual IR/codegen gate.
- Add a Metal emission smoke/golden test if the Metal path remains in scope.

## Imported full lowerer CI-gate triage

Reviewed `codex/imported-full-lowerer-20260627` (`40cbd8af8`).

The branch's only patch-exclusive commit modifies
`.github/workflows/madaros-prebuilt-refresh.yml` to require
`bash scripts/ci/madaros_multimodule_witness.sh` before refreshing the checked-in
Madaros prebuilt.

Do not port this workflow change yet. On current consolidation,
`bash scripts/ci/madaros_multimodule_witness.sh` fails:

```text
[madaros-mm-witness] FAIL: thin_single expected_exit=7 actual_exit=139
lower_array: seed_begin
Segmentation fault
```

That makes the workflow patch a useful desired gate, but not an absorptive
cleanup change. It belongs with the imported stdlib/multimodule lowerer residual
until `thin_single` and the rest of the witness manifest pass on the branch that
would enable the prebuilt refresh.

## Tuple-let/PBPK broad-train refresh

Re-reviewed `fix/madaros-tuple-let-desugar` and
`docs/pbpk-session-notes-2026-06-28` after the consolidation branch reached
`939725590`.

Do not merge either branch as a unit.

Evidence:

- `fix/madaros-tuple-let-desugar` still has a broad right-side history including
  solver/GPU Blackwell receipt work, KAXI scorer kernels, dissertation notes,
  bootstrap binary refresh, imported SMT lowering, and raw async closeout.
- Direct branch diff against consolidation touches 241 files, including
  `.claude/llm_offload_log.md`, `.github/workflows/ci.yml`, `bin/souc`,
  `bin/madaros-linux-x86_64`, solver benchmarks/scripts, GPU emitters,
  theorem/stdlib files, parser/checker/IR/native compiler surfaces, docs, and
  many test deletions.
- The raw diff would delete the consolidation audit doc and the focused gates
  added by this sweep (`native_v2_float_arith_gate`,
  `native_v2_struct_field_float_gate`, `native_v2_loop_print_gate`,
  `native_v2_global_bss_gate`, `native_v2_tuple_signature_absorbed_gate`, and
  `native_v2_semcall_absorbed_gate`).
- `docs/pbpk-session-notes-2026-06-28` has the same broad-train shape plus
  additional handoff/docs content. It is documentation/audit material, not a
  compiler-core merge branch.

Classification:

- Keep both as broad provenance branches.
- Extract only named compiler fixes with a small diff and a focused gate.
- Do not use either branch as the base for the unified compiler branch.

## Madaros rebuild probe extraction

Reviewed `codex/madaros-rebuild-probe` (`e4284b656`).

Ported the complete patch because it is a small source-only compiler fix:

- `self-hosted/compiler/lean_single.sio`: extends the legacy `ty_eq` escape hatch
  for integer scalar values flowing into enum/bool struct fields by adding the
  bool case (`k1 == 4 && k2 == 1`).
- `self-hosted/ir/opt_cleanup.sio`: restores two malformed unary-op tracking
  guards by turning dangling `&& ...` fragments into explicit `if
  result.instrs[i].op == IrUnaryOp && ...` conditions and by reading through a
  local instruction value before populating `is_bnot`/`bnot_src`.

Validation after port:

- `bin/souc info`: PASS, selected Madares v0.80.0.
- `bash scripts/run_sio_test_suite.sh hello --verbose`: PASS 2 / FAIL 0.
- `bash scripts/ci/madaros_source_to_elf_gate.sh`: PASS, including check,
  trace, normal/native-v2 compile, ELF execution, and exit-code semantics.
- `bash tests/native_v2_semcall_absorbed_gate/run.sh bin/souc`: PASS 4 / FAIL 0.

Additional probe:

- `bin/souc --check self-hosted/ir/opt_cleanup.sio`: FAILS on broad existing
  parse/check noise in this module surface, so it is not used as proof for this
  extraction.

## Release conformance spine compiler-core extraction

Reviewed `codex/release-conformance-spine-fix`.

Ported only the compiler-core commit `0e0845447`
(`Harden validated-call patching for multimodule lower`). The other exclusive
commits in the branch are local editor/package/install/docs conformance-gate
work and were not pulled into the compiler consolidation lane.

Ported/adapted:

- `self-hosted/ir/lower.sio`: added `lowerer_lower_program_items_ref_mut` so
  summary-based imported body lowering mutates a single `Lowerer` rather than
  lowering through a by-value return path.
- Added bounds guards for validated-call patching over `fn_id`, module
  functions, and function instruction counts (`2048` and `IR_MAX_INSTRS`).
- Resolved the branch conflict by preserving the newer consolidation signature
  that passes `callee_strategies` and `callee_param_counts` by reference.

Validation after port:

- `bin/souc info`: PASS, selected Madares v0.80.0.
- `bash scripts/run_sio_test_suite.sh hello --verbose`: PASS 2 / FAIL 0.
- `bash scripts/ci/madaros_source_to_elf_gate.sh`: PASS, including check,
  trace, normal/native-v2 compile, ELF execution, and exit-code semantics.
- `bash scripts/ci/madaros_multimodule_witness.sh`: still FAILS with
  `thin_single expected_exit=7 actual_exit=139` at `lower_array: seed_begin`.

Status:

- The validated-call hardening is absorbed.
- The imported multimodule witness remains a residual blocker and still must not
  be wired into the prebuilt-refresh workflow.

## Release / docs / repo tooling branch triage

Reviewed:

- `origin/chore/repo-hygiene` and local `chore/repo-hygiene`
- `origin/codex/main-release-ci-repair-20260623` and local
  `codex/main-release-ci-repair-20260623`
- local `codex/release-install-visibility-20260626`
- local `codex/release-conformance-spine-fix`
- local `codex/website-docs-support-gate-20260626`
- local `codex/baseline-docs-registry`
- `claude/release-apparatus`
- `claude/release-e2e-eval`
- `claude/release-real-mc`
- local `fix/docs-registry-sync`

Do not merge these branches as compiler-core branches. They are release,
install, website/docs, governance, package/editor, Slurm/rebootstrap, or repo
hygiene trains. Raw diffs against the consolidation branch still include old
worktree rollback shapes, from smaller install-support branches (~176-177
files) up to release/docs trains with ~2,862-2,932 files and ~671k deletions.

Compiler-core status:

- `codex/release-conformance-spine-fix`: the only compiler-core commit needed
  for this sweep, `0e0845447` validated-call patch hardening, was already
  ported in the previous section.
- `claude/release-apparatus`, `claude/release-e2e-eval`, and
  `claude/release-real-mc` contain release packaging and some checker/native
  history, but their branch-level diffs are too broad for consolidation. Any
  remaining compiler claim must be extracted with a named gate.
- `main-release-ci-repair`, `release-install-visibility`,
  `website-docs-support-gate`, `baseline-docs-registry`, `docs-registry-sync`,
  and `repo-hygiene` are not merge sources for the compiler lane.

## Direct-call param-slot branch triage

Reviewed `origin/codex/direct-call-param-slot` (`9174e3fc5`).

The branch is a small native-v2 patch:

- initialize `MachineFunction.temp_count` from `IrFunction.reg_count` in
  `native_v2_lower_function_to_machine`
- start legalizer temps above `max(base_slot_count, temp_count)`
- add `tests/native-v2/param_slot_legalize_boundary_witness.sio`

Do not port yet. The patch applies cleanly and the witness passes `--check`, but
the real execution surface is red on current consolidation:

```text
bin/souc --native-v2-compile tests/native-v2/param_slot_legalize_boundary_witness.sio -o /tmp/sounio_param_slot_witness.elf
/tmp/sounio_param_slot_witness.elf
=> exit 139

bin/madaros run tests/native-v2/param_slot_legalize_boundary_witness.sio
=> lower_array: seed_begin; Segmentation fault
```

Related probe: existing
`tests/native-v2/large_value_legalize_boundary_witness.sio` also compiles to an
ELF and then exits 139. Treat this as a native-v2 witness/runtime residual, not
as an absorbed compiler-core change.

## Generic struct instantiation branch triage

Reviewed `feat/generics-struct-instantiation` (`f68d688c9`).

The compiler-code portion is already present in current consolidation:

- `mono_register_struct` exists in `self-hosted/compiler/lean_single.sio`.
- the generic-function rewrite loop already collapses `GenStruct<TYPE>` into a
  registered monomorphic struct token.

The branch's new witness was not kept because it is still red on current
consolidation:

```text
struct Box<T> { val: T }
fn wrap<T>(x: T) -> Box<T> { Box<T>{ val: x } }

bin/souc --native-v2-compile tests/generics/box_generic_struct.sio -o /tmp/box_generic_struct.elf
=> native_v2_compile: front-half failed: parse_failed
```

Classification: code absorbed already; runnable witness remains residual.

## Legacy native-v2 backlog branch triage

Reviewed the top commits of the older native-v2 backlog/worktree branches:

- `wall/source-to-elf` (`0f0cc9df9`)
- `wall/check-enumctor` (`a595ad29f`)
- `backlog/f64-compare` (`54ecab937`)
- `backlog/sret-return` (`dce0ea1ca`)
- `backlog/strconcat-emit-fix` (`d5bf9dd9e`)
- `honest/codex-calls` (`eae0d5134`)
- `claude/fn-pointers-integ` (`999b6633b`)

Do not merge these branches as branches. Their branch bases are very old and a
raw diff against consolidation spans thousands of files, including massive
deletions unrelated to the top commit being inspected.

Absorbed/equivalent evidence on current consolidation:

- `wall/source-to-elf`: current `self-hosted/compiler/main.sio` documents and
  implements the second positional output path for `--native-v2-compile`; the
  canonical `bash scripts/ci/madaros_source_to_elf_gate.sh` remains PASS.
- `honest/codex-calls`: `bin/madaros --native-v2-emit-call5` emits an ELF that
  exits 31; `--native-v2-emit-call6` emits an ELF that exits 63.
- `backlog/sret-return`: `bin/madaros --native-v2-emit-sret` emits an ELF that
  exits 14.
- `backlog/strconcat-emit-fix`: `--native-v2-emit-strconcatlen` emits an ELF
  that exits 5; `--native-v2-emit-strconcatcharat` emits an ELF that exits 100.
- `claude/fn-pointers-integ`: `--native-v2-emit-fnptr` emits an ELF that exits
  110.
- `backlog/f64-compare`: `--native-v2-emit-f64cmp` is present and emits/runs
  the full case matrix. Observed results on consolidation:
  `lt-true=1`, `lt-false=0`, `lt-nan=1`, `le-true=1`, `le-false=0`,
  `le-nan=1`, `gt-true=1`, `gt-false=0`, `gt-nan=0`, `ge-true=1`,
  `ge-false=0`, `ge-nan=0`, `eq-true=1`, `eq-false=0`, `eq-nan=1`,
  `ne-true=1`, `ne-false=0`, `ne-nan=0`.
- `wall/check-enumctor`: kept as source-absorbed/needs-no-branch-merge for now;
  current consolidation already carries enum constructor checker work in newer
  branches, and this top commit is not safe to replay through the old branch
  history.

Additional validation:

- `bash scripts/ci/madaros_full_gate.sh`: PASS on current consolidation,
  including public/raw check CLI, source build/run, native-v2 ABI/backend
  witnesses, multimodule visibility diagnostics, and package manager self-test.

Classification:

- Treat these legacy backlog branches as provenance, not merge sources.
- If one of their claims regresses, add a fresh focused gate on
  `integration/compiler-consolidation-20260628` instead of reviving the old
  branch history.

## Parser/lexer small-branch triage

Reviewed the small parser/lexer branches:

- `parser/extern-blocks` (`74368b22b`)
- `parser/kernel-fn` (`f9c189397`)
- `parser/const-decls` (`0b3918faa`)
- `parser/sci-notation-float` (`f77ae77b0`)
- `parser/algebra-keyword-e008` (`ff0bc1166`)
- `parser/sci-notation-modular-e008` (`c39750e5d`)

Do not merge these branches as branches. The current consolidation source
already contains most of the parser/lexer surface, and the remaining red surface
needs fresh focused repair rather than old branch history.

Absorbed/equivalent evidence on current consolidation:

- `parser/kernel-fn`: `kernel fn k() -> () { }` plus a simple `main` checks OK.
  The same probe without explicit `-> ()` still reports `E072`, so keep the
  accepted surface to the explicit-unit form.
- `parser/const-decls`: a top-level `const ANSWER: i64 = 42;` plus a simple
  `main` checks OK.
- `parser/sci-notation-float` and `parser/sci-notation-modular-e008`:
  `let x: f64 = 1.25e-3;` checks OK.
- `parser/algebra-keyword-e008`: current source already carries parser/AST
  support for `algebra`/`study`; the inspected branch is docs/provenance only
  for this consolidation pass.

Residual:

- `parser/extern-blocks`: current source has `parse_extern_fn_item` and brace
  form handling, but `extern "C" { fn puts(s: *const i8); }` still reports
  `E072` ("kernel function must return unit type") on `bin/souc --check`.
  Treat this as parser-present/checker-surface residual, not a green absorbed
  feature.

Validation:

- `bash scripts/run_sio_test_suite.sh hello --verbose`: PASS 2/2.

## E008 checker train triage

Reviewed the older E008/checker stack:

- `origin/check/e014-int-index-e008`
- `origin/check/int-cross-width-e008`
- `origin/check/ref-param-lower-e008`
- `origin/check/fn-type-lower-e008`
- `origin/check/f32-field-narrowing-e008`
- `origin/check/field-deref-ref-e008`
- `origin/check/closure-hof-triple-e008`
- `origin/check/linear-double-consume-e039`
- `origin/check/refinement-types-e008`
- `origin/parser/fn-type-effects-list-e008`
- `origin/integration/e008-nested-store-complete`
- `origin/integration/consolidate-modular`

Do not merge these branches as branches. None is an ancestor of the current
consolidation branch, and each raw branch diff is an old-worktree rollback
shape: roughly 2,808-2,815 files, about 283k insertions and about 716k-717k
deletions. The diffs include broad docs, website, binary, and audit churn, so
they are not safe compiler cleanup sources.

Current consolidation already carries the main surfaces these lanes were
tracking:

- lexer/parser support for `algebra`, `study`, `kernel`, `const`, extern
  function items, scientific notation, refinement types, and `fn(...) -> ...`
  types
- `lean_single.sio` refinement checks, fn-type signature scanning, effect
  propagation for higher-order calls, reference compatibility in call arguments,
  and linear-consume path checks

Classification:

- Treat the E008/checker stack as provenance and residual ownership history.
- If a specific claim regresses, create or revive a fresh focused witness on the
  consolidation branch and port only the minimal source hunk needed for that
  witness.
- Do not use any of the E008 branch histories as direct merge parents for
  `integration/compiler-consolidation-20260628`.

Validation:

- `bash scripts/run_sio_test_suite.sh hello --verbose`: PASS 2/2 after this
  triage.

## Native-v2 source bridge / prebuilt branch triage

Reviewed the old native-v2/source-bridge and prebuilt-refresh branches:

- `origin/feat/native-v2-source-bridge`
- `origin/feat/native-v2-bridge-sret`
- `origin/feat/mc-v2-opcodes`
- `origin/fix/frame-slot-recycling`
- `origin/fix/native-selfhost-prebuilt`
- `origin/fix/stdlib-e2e-sret-workarounds`
- `origin/modular/native-v2-e2e-gate`
- `origin/modular/native-v2-source-to-elf`
- `origin/integration/canonical-souc-gate-shepherd`

Do not merge these branches as branches. None is an ancestor of the current
consolidation branch. Raw diffs range from 2,722 to 2,898 files and include
roughly 282k-284k insertions with 702k-729k deletions, which is old-base
rollback shape rather than focused compiler work.

Current consolidation already proves the main current native-v2/source-to-ELF
surfaces through canonical gates:

- `bin/souc info`: selected `Madares v0.80.0`.
- `bash scripts/ci/madaros_source_to_elf_gate.sh`: PASS, including check,
  trace, normal/native-v2 compile, ELF execution, and exit-code semantics.
- `bash scripts/ci/madaros_full_gate.sh`: PASS, including public/raw check CLI,
  source build/run, native-v2 ABI/backend witnesses, multimodule visibility
  diagnostics, and package manager self-test.

Classification:

- Treat `native-v2-source-bridge`, `native-v2-source-to-elf`, `mc-v2-opcodes`,
  and `native-v2-e2e-gate` as provenance covered by the current source-to-ELF
  and full Madaros gates, not direct merge sources.
- Treat `native-selfhost-prebuilt` and `canonical-souc-gate-shepherd` as
  binary/canonical-wrapper provenance. Do not refresh prebuilts from old branch
  state during compiler consolidation.
- Treat `native-v2-bridge-sret`, `frame-slot-recycling`, and
  `stdlib-e2e-sret-workarounds` as residual native-runtime/SRET ownership lanes
  unless a fresh focused gate identifies a missing source hunk.

## Madaros closeout / variance / async branch triage

Reviewed:

- `origin/codex/gum-variance-sota-20260626` (`36a3b1d2b`)
- `origin/codex/parser-strict-20260627` (`95b16f24b`)
- `origin/codex/madaros-raw-async-gates-20260627` (`c25ccdc6f`)
- `codex/madaros-close-20260627` (`c64fbd8ad` tip, including
  `ae4b63943` ultrareview substrate patch)
- `origin/claude/madaros-substrate-review` (`e0d265ef2`)

Absorbed/equivalent:

- `origin/codex/gum-variance-sota-20260626`: `git cherry` reports the single
  patch as patch-equivalent on consolidation. The focused tests exist and pass:
  `variance_of_literal` and `variance_of_measure_sum`.
- `origin/codex/parser-strict-20260627`: `git cherry` reports the single
  `Box::new` path-call checker patch as patch-equivalent on consolidation.
  Current `self-hosted/check/check.sio` has `call_expr_is_box_new` and both
  in-place/by-value `Box::new` call handlers.

Ported from `codex/madaros-close-20260627`:

- `ae4b63943` partial ultrareview substrate fix, limited to the two
  compiler-core changes that were still missing and did not require the broad
  solver/bootstrap train:
  - `self-hosted/ir/opt_strategy.sio`: `ir_opt_append` now writes through the
    full `[IrInstr; 1024]` output buffer instead of silently dropping writes
    after index 127 while continuing to advance the returned count.
  - `self-hosted/native/codegen_x86_linux.sio`: ET_REL and ET_DYN finalizers now
    read `.text` through `NATIVE_V2_TEXT_BUF` when available, avoiding narrow
    `CodeBuffer.bytes[131072]` reads when the native-v2 text mirror grows past
    128 KiB.

Not merged as branches:

- `origin/codex/madaros-raw-async-gates-20260627`: the compiler patch was
  previously extracted, but current consolidation does not prove async closed.
  `async_basic` fails typecheck with `E012` on async/await shapes, and
  `async_spawn_syscall_pid` fails with `E137` plus `E012`. Treat as residual
  language/checker surface, not absorbed-green.
- `codex/madaros-close-20260627`: direct branch diff is still broad and
  destructive relative to consolidation (`self-hosted/compiler/main.sio`,
  `module_frontend.sio`, `ir/lower.sio`, native codegen, multimodule witness
  scripts, solver/theorem docs, and bootstrap binary refresh). Keep it as
  provenance and extract only named, reviewable compiler patches.
- `origin/claude/madaros-substrate-review`: its only patch-exclusive commit is
  another copy of the broad imported-SMT substrate train. `git diff
  HEAD..origin/claude/madaros-substrate-review` spans 196 files and would delete
  the consolidation audit plus the focused gates added in this sweep
  (`native_v2_float_arith_gate`, `native_v2_global_bss_gate`,
  `native_v2_loop_print_gate`, `native_v2_semcall_absorbed_gate`,
  `native_v2_struct_field_float_gate`, and
  `native_v2_tuple_signature_absorbed_gate`). Do not merge it as a branch; the
  only small substrate review fix identified so far was the `ae4b63943`
  ultrareview bounds/mirror-read patch ported above.

Validation:

- `bash scripts/run_sio_test_suite.sh variance_of_literal --verbose`: PASS 1/1.
- `bash scripts/run_sio_test_suite.sh variance_of_measure_sum --verbose`: PASS
  1/1.
- `bash scripts/run_sio_test_suite.sh async_basic --verbose`: FAIL 1/1
  (`E012`, type has no field; async/await checker surface still red).
- `bash scripts/run_sio_test_suite.sh async_spawn_syscall_pid --verbose`: FAIL
  1/1 (`E137` undeclared variable plus `E012`).
- After the ultrareview micro-port:
  - `git diff --check`: PASS.
  - `bash scripts/run_sio_test_suite.sh hello --verbose`: PASS 2/2.
  - `bash scripts/ci/madaros_source_to_elf_gate.sh`: PASS.
  - `bash scripts/run_sio_test_suite.sh variance_of_measure_sum --verbose`:
    PASS 1/1.
  - `bin/souc --check self-hosted/ir/opt_strategy.sio`: FAIL on existing
    multimodule visibility noise (`E175`) and large-frame warnings; not used as
    acceptance evidence for this micro-port.
  - `bin/souc --check self-hosted/native/codegen_x86_linux.sio`: FAIL on
    existing broad native-codegen check noise (`E137`, `E035`, `E012`, `E002`,
    `E175`, large-frame warnings); not used as acceptance evidence for this
    micro-port.
