<!-- docs:meta
topic_id: repo.docs.audit.madaros-gen2-pr2307-isolation-2026-08-31
authority: repo_only
audience: users
last_validated: 2026-09-01
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-gen2-pr2307-isolation-2026-08-31
-->

# Madaros Gen2 PR #2307 isolation (2026-08-31)

## Result

PR #2307 is not the demonstrated cause of the observed
`into_acc_done=40 -> 0` change.

A valid fixed-Madaros-seed experiment produced the same result on current
source and on a current-compatible whole-PR inverse:

```text
into_acc_done=0
first function=compiler_try_default_native_v2_single
lower.sio site=9712
cause=cannot safely lower print/println argument with unresolved scalar kind
```

Both rows typechecked `self-hosted/compiler/main.sio` with zero errors and
compiled/ran `examples/hello.sio`. There is therefore no PR-local causal delta
to bisect or repair from this comparison.

The actual current-source defect was a scalar-kind inference failure at direct
`CompilerOptions.output_file` field uses in
`compiler_try_default_native_v2_single`. Anchoring that field in an explicitly
typed `string` local restored the fixed-seed floor from 0 to 40. The canonical
fixed-point gate now passes its default progress floor; the remaining 40-to-122
work is a separate failure in `module_frontend_compile_imported_to_file`.

## Fixed experiment contract

- Source anchor: `a1769a1eb545cbef638f7bdac5d7c9cc679c25cd`.
- Fixed Madaros ELF SHA-256:
  `2080defa2f1042ef0b0c3d6796e77de0226e2b840856af5da7d2e36ee911e253`.
- Fixed Madaros bytes: `103142391`.
- Fixed lean seed used only to derive that ELF:
  `e594a9ab7985f5c906bf472c1f588b454d4b1529641beb3885edf9df86916712`.
- Foundry node: `gpuorangefs-r770-proxmox`, 32 CPUs.
- Slurm environment: `--export=NONE` plus a fixed `PATH`.
- Slurm job name: `sounio-gen2-fixed-env`.
- Hello, self-check, and direct Gen2 measurement ran as separate jobs so each
  compiler witness was process ordinal one.
- The lane's fixed-point gate was appended as the final tar entry, keeping the
  measurement harness fixed when the source root pointed at a snapshot.
- All Madaros ELFs stayed in the noncanonical OrangeFS task directory.

The fixed ELF was rebuilt from the clean current compiler source and reproduced
the previously observed SHA-256 `2080defa...`. Independent validation reported:

```text
REMOTE: hello compile_rc=0 run_rc=0 output=Hello, Sounio
REMOTE: fpcheck rc=0 errors=0
REMOTE: run_check_mode: verdict=0
REMOTE: check: OK
```

## Valid comparison

### Current source

Direct single-process self-build measurement:

```text
REMOTE: gen2_measure rc=1 into_acc_done=0 minimum=0
REMOTE: gen2_measure first_failure=IR lowering failed during merge: ir_bodies_failed
REMOTE: gen2_measure context=  lowering-error record: total=1 hard=1
REMOTE: gen2_measure context=  lowering errors: 1, first while lowering function `compiler_try_default_native_v2_single`
REMOTE: gen2_measure context=  raised at lower.sio lines: 9712
REMOTE: gen2_measure context=  cause: cannot safely lower print/println argument with unresolved scalar kind in function `compiler_try_default_native_v2_single`
```

### Current-compatible whole-PR inverse

The inverse was constructed in `/tmp/sounio-pr2307-synthetic-inverse` from the
same `a1769a1e...` source anchor with:

```text
git revert --no-commit -m 1 503eba5217b05f26003eafd60c7ba6e52f269a7d
```

The revert auto-applied across the PR write set. Two conflict files needed
compatibility resolution because later changes overlapped the reverted storage
representation:

- `self-hosted/ir/ir.sio`: remove the PR's function-table violation kinds while
  retaining the later instruction-pool violation kind.
- `self-hosted/ir/lower.sio`: restore the pre-PR inline local-stack fields while
  retaining the later `is_unsigned_int` field and the later FO bind helper.

This is a current-compatible semantic inverse, not a historical checkout. It
keeps post-PR APIs fixed and varies the PR #2307 storage migration. It passed:

```text
REMOTE: fpcheck rc=0 errors=0
REMOTE: run_check_mode: verdict=0
REMOTE: check: OK
REMOTE: hello compile_rc=0 run_rc=0 output=Hello, Sounio
```

Two consecutive direct Gen2 measurements matched current source exactly:

```text
REMOTE: gen2_measure rc=1 into_acc_done=0 minimum=0
REMOTE: gen2_measure first_failure=IR lowering failed during merge: ir_bodies_failed
REMOTE: gen2_measure context=  lowering errors: 1, first while lowering function `compiler_try_default_native_v2_single`
REMOTE: gen2_measure context=  raised at lower.sio lines: 9712
REMOTE: gen2_measure context=  cause: cannot safely lower print/println argument with unresolved scalar kind in function `compiler_try_default_native_v2_single`
```

## Source repair

`compiler_try_default_native_v2_single` now copies `opts.output_file` into an
explicitly typed `string` local before passing or printing it. Its success
message also uses separate `print` calls instead of constructing a string with
`+`. This is the smallest source change that supplies a concrete scalar kind at
all three uses rejected by the fixed Madaros seed.

The independent controls remained green:

```text
REMOTE: hello compile_rc=0 run_rc=0 output=Hello, Sounio
REMOTE: fpcheck rc=0 errors=0
REMOTE: run_check_mode: verdict=0
REMOTE: check: OK
```

The direct default-floor measurement passed:

```text
REMOTE: gen2_measure rc=1 into_acc_done=40 minimum=40
REMOTE: gen2_measure first_failure=IR lowering failed during merge: ir_into_acc_failed
REMOTE: gen2_measure context=  lowering errors: 1, first while lowering function `module_frontend_compile_imported_to_file`
REMOTE: gen2_measure context=  raised at lower.sio lines: 9712
REMOTE: gen2_measure context=  cause: cannot safely lower print/println argument with unresolved scalar kind in function `module_frontend_compile_imported_to_file`
```

The canonical fixed-point wrapper independently reproduced the same boundary:

```text
[rung check] rc=0 errors=0
[rung gen2] rc=1 merged_ir_functions=<none>
            into_acc_done=40 minimum=40
            first_failure=IR lowering failed during merge: ir_into_acc_failed
MADAROS_FIXED_POINT_OK: reached rung 'check' as recorded; the next wall is 'gen2'
REMOTE: gen2_progress rc=0 minimum=40
```

## Invalidated controls

The earlier matrix fixed the lean bootstrap seed but rebuilt Madaros from every
source snapshot. That varied both the compiler ELF and the source under test.
Its reported inverse progress (`22`, historically `40`) is not a fixed-Madaros
comparison and cannot establish PR #2307 as the cause of the current zero.

Two candidate fixed Madaros seeds were rejected:

| Candidate | SHA-256 | Rejection |
|---|---|---|
| Madaros built from historical parent `8ba8f0ed...` | `9aa665725890be0bf144c61e63cb57dc4a99c81f239c50c14c922f98e5c27823` | Segfaulted compiling hello at `lower_array: seed_begin` on its own source snapshot. |
| Older hello-stable Madaros | `b755dbb941cd16bd5789b0c7a629e9b17c445d8ceebb90cd42c0533f2c0a288a` | Could pass isolated hello/check, but direct Gen2 crashed at `lower_array: seed_begin` with `into_acc_done=0`. |

The historical parent checkout also failed current-seed preflight with `E010`
in `compiler/claim_executor::ce_run_gate`, so it was replaced by the
current-compatible inverse above.

## Environment sensitivity

The same Madaros ELF could pass self-check as the first compiler process in a
controlled Slurm step and segfault after another compiler invocation. Adding an
unrelated environment variable also shifted the outcome under the inherited
environment. This is a real layout-sensitive runtime defect, not evidence that
`SOUNIO_SPEC_TRACE` changes semantics.

The opt-in `SOUNIO_REMOTE_CLEAN_ENV=1` launcher mode and separate gate jobs
contain that defect for this experiment. They do not repair it. A combined
`--gate hello --gate check` invocation is not an authoritative self-check
control for this lane.

## Gate changes

`scripts/ci/madaros_fixed_point_gate.sh`:

- reports `into_acc_done`, first failure, and lowering context;
- supports `SOUNIO_MADAROS_FP_MIN_INTO_ACC_DONE` with default 40;
- rejects regressions below the restored 40 floor.

`scripts/dev/souc-build-remote.sh`:

- supports a source-root override and fixed Madaros ELF import/export with
  SHA-256 validation;
- validates the lean seed only when deriving a Madaros ELF;
- supports an opt-in clean Slurm environment;
- keeps a fixed gate script when measuring historical or synthetic sources;
- adds `gen2-measure`, a direct single-process progress receipt;
- reports the raw Madaros ELF SHA-256 and independent hello/self-check results.

One compiler source change is retained in the product worktree: the explicitly
typed output-path local and non-concatenating success output in
`self-hosted/compiler/main.sio`. PR #2307 remains intact.

## Closed blocker

`BLK-20260901-gen2-current-zero` is closed. Its acceptance gate passed with
`into_acc_done=40`, independent hello and self-check controls green, and the
fixed Madaros SHA-256 recorded above.

## Remaining blocker

```text
Blocker-ID: BLK-20260901-gen2-40-to-122
Status: reproduced
Severity: B1
Class: gate-regression
Owner: codex-2
Lane: gen2-pr2307-isolation-20260831
Worktree: /tmp/sounio-gen2-pr2307-20260831
Branch: codex/gen2-pr2307-isolation-20260831
Files-Owned: self-hosted/compiler/main.sio, scripts/ci/madaros_fixed_point_gate.sh, scripts/dev/souc-build-remote.sh, docs/audit/MADAROS_GEN2_PR2307_ISOLATION_2026-08-31.md
Files-Read-Only: canonical Madaros ELF outputs, origin/main
Do-Not-Touch: canonical Madaros ELF outputs
Repro: run the clean-environment gen2-progress or gen2-measure job with fixed Madaros SHA-256 2080defa2f1042ef0b0c3d6796e77de0226e2b840856af5da7d2e36ee911e253
Observed: hello passes; self-check exits 0 with zero errors; Gen2 reaches into_acc_done=40, then rejects an unresolved-scalar print argument in module_frontend_compile_imported_to_file at lower.sio site 9712
Expected: Gen2 reaches the independently established next floor of into_acc_done>=122 while hello and self-check remain green
Acceptance-Gate: SOUNIO_MADAROS_FP_MIN_INTO_ACC_DONE=122 with scripts/ci/madaros_fixed_point_gate.sh, plus independent --gate hello and --gate check jobs under the fixed clean environment
Evidence-Level: E4
Evidence: this audit, the fixed-seed direct receipt, and the canonical fixed-point Foundry receipt from 2026-09-01
Fallback-Path: none
Legacy-Kept: yes; PR #2307 remains intact because its causal attribution was falsified
LLM-Offload: not-required
Next-Action: isolate the unresolved scalar kind in module_frontend_compile_imported_to_file without weakening the restored default floor of 40
```

## Review disposition

The earlier Grok 4.6 review correctly warned that a crashing inverse was not an
acceptance oracle. The fixed-seed rerun sharpened that warning: rebuilding the
compiler for each row was the primary attribution confound. The repaired source
then passed both the direct floor and canonical fixed-point wrapper. No math
claim, clinical pathway, or external-facing artifact changed, so repository
policy did not require an additional LLM offload.
