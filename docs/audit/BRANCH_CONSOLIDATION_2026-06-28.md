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

Do not use `/workspace/sounio` as the integration surface. It is detached at
`d7befc9d1` and should remain coordination/read-only until explicitly moved.

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
- `codex/tuple-signature-types-20260626`: reviewed. The parser, checker, native implicit-unit/assert, and native bool-literal fixes are already equivalent in the consolidation branch; remaining exclusive commits are GPU/script/test-bookkeeping and should not drive the compiler-core merge.
- `codex/madaros-import-stdlib-lowering-current`: older imported-stdlib lowering branch; conflicts with the newer imported-lowering path already on `origin/main`.
- `codex/madaros-boxnew-clean` and `codex/madaros-boxnew-append-fix`: older Box::new trains; need comparison against current `origin/main` before any merge.

### Archive/quarantine

These are too broad or old to merge automatically:

- `fix/flatparse-and-scan-operator`
- `codex/gpu-semantic-profile`
- `codex/gpu-modular-bridge`
- `claude/kw-demote-module`
- `integration/effects-kwdemote`
- `research/erdos-compiler-wip`

Keep them as archaeological branches until a specific missing feature is named.

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

## Next clean consolidation path

1. Keep `origin/main` as the compiler baseline, not any old WIP train.
2. Merge only one lane at a time into `integration/compiler-consolidation-20260628`.
3. First real candidates:
   - remaining compiler-core commits from `codex/madaros-retire-lean-single-20260627` only if a missing gate is named; the raw async compiler patch has been extracted.
   - `c2a783f27` struct-field float arithmetic, if it can be adapted without downgrading the newer struct-layout/field metadata path.
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
