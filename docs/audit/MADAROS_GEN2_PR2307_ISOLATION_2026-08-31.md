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
fixed-point gate now passes its default progress floor. A second explicitly
typed path local in `module_frontend_compile_imported_to_file` advances the same
fixed-seed measurement from 40 to 123, closing the separate 40-to-122 blocker.
The later terminal is outside this lane's blocker scope; its contract class is
`bootstrap-runtime` and its observed cause is native code-buffer overflow. A
stale gate diagnostic had instead called the 13,107-function merge a capacity
hit, but the source cap is 16,384 and the strict closure census retains 5,132
slots of headroom.

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

## Origin-main anchor 40-to-123 repair

The two isolation commits were transplanted without conflicts onto a fresh
worktree based at the user-pinned `origin/main` anchor
`a1590c1e98d18a43d8a46954fc981954fddefffb`:

```text
1df2215324 ci(madaros): isolate gen2 fixed-seed attribution
b06379d4cd fix(madaros): restore gen2 progress floor
```

Before the new change, the transplanted source independently reproduced the
40 boundary with self-check and hello green. The only hard lowering record was
the unresolved `print(file_path)` argument in
`module_frontend_compile_imported_to_file`. `file_path` came from the indexed
string-array field `work_imports.paths[dep_i]` without an explicit local type.

The repair is one line:

```sio
let file_path: string = work_imports.paths[dep_i]
```

It preserves the diagnostic and supplies positive scalar-kind evidence without
weakening fail-closed `print` lowering. The direct fixed-seed measurement passed
the elevated floor:

```text
REMOTE: gen2_measure rc=1 into_acc_done=123 minimum=122
REMOTE: gen2_measure tail=  error: multimodule native thin-link compilation failed
```

The canonical fixed-point wrapper reproduced that boundary and retained its
typecheck rung:

```text
ir_max_functions 16384
[rung check] rc=0 errors=0
[rung gen2] rc=1 merged_ir_functions=13107
            into_acc_done=123 minimum=122
            first_failure=Error: Failed to write native binary to /tmp/sounio-remote-899242/fixed-point/madaros.gen2 rc=19
MADAROS_FIXED_POINT_OK: reached rung 'check' as recorded; the next wall is 'gen2'
REMOTE: gen2_progress rc=0 minimum=122
```

The independent post-change executable control also remained green:

```text
REMOTE: hello compile_rc=0 run_rc=0 output=Hello, Sounio
```

The gate's reusable default floor remains 40. The existing CI invocation now
sets `SOUNIO_MADAROS_FP_MIN_INTO_ACC_DONE=122`, so the independently reproduced
progress is retained without changing the local/default experiment contract.

The gate output above originally printed a stale warning based on a hardcoded
2048 comparison. That warning is invalidated. `IR_MAX_FUNCS` is 16,384, and
`scripts/ci/madaros_ir_capacity_probe.sh` passed with 124 closure nodes, 11,252
function declarations, and 5,132 slots of headroom. The gate now reads the cap
from `self-hosted/ir/ir.sio`, fails closed if the observed merged count reaches
that cap, and the CI capacity probe is strict rather than report-only.

Foundry job `11464` reran the canonical gate with the fixed attribution
compiler. Foundry job `11471` rebuilt Madaros from this exact source snapshot,
then ran self-check, hello, and the canonical 122-floor gate in one job. Its
Madaros SHA-256 was
`20ad5d9a5bd99ee05541a55b70c80dd4d0da152b7654313108a239a40179d8e2`;
all three gates passed and it retained the terminal as:

```text
Error: Failed to write native binary to /tmp/sounio-remote-926813/fixed-point/madaros.gen2 rc=19
```

The complete job `11471` output is retained at
`artifacts/audit/madaros_gen2_40_to_123_current_source_retry_foundry_20260901.txt`.
The companion JSON receipt records the exact command, source and harness
content hashes, log digest, and separate evidence limits for older runs whose
raw logs were not retained.

In both native writers, rc=19 is the fail-closed `code_overflow` return in
`self-hosted/native/codegen_x86_linux.sio`. The later boundary's contract class
is `bootstrap-runtime`; its cause is native code-buffer overflow, not IR
capacity. The lane does not open a second blocker for that later boundary.

## Invalidated controls

The earlier matrix fixed the lean bootstrap seed but rebuilt Madaros from every
source snapshot. That varied both the compiler ELF and the source under test.
Its reported inverse progress (`22`, historically `40`) is not a fixed-Madaros
comparison and cannot establish PR #2307 as the cause of the current zero.

Foundry job `11470` is also invalidated as an ordered-gate control, not hidden.
Its standalone self-check and hello both passed, then `gen2-progress` check
segfaulted at rc=139 because the hello case had lowered the shared remote
shell's stack to 8 MiB. Removing that parent-shell mutation was the only harness
change before identical job `11471` passed all three gates. The full negative
control is retained at
`artifacts/audit/madaros_gen2_40_to_123_current_source_foundry_20260901.txt`.

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

Two compiler source changes are retained in the product worktree: the
explicitly typed output-path local and non-concatenating success output in
`self-hosted/compiler/main.sio`, plus the explicitly typed imported-module
`file_path` local in `self-hosted/compiler/module_frontend.sio`. PR #2307
remains intact.

## Live origin/main replay

The pinned `a1590c1e98` evidence above remains the attribution record. After
`origin/main` advanced by 30 commits to `bc6da34c11`, the four lane commits were
replayed in a separate clean worktree rather than rewriting that record. The
replay applied without conflicts, including across the intervening additions
to `self-hosted/compiler/main.sio` and `.github/workflows/ci.yml`.

Foundry job `11473` measured the intermediate replay commit `839ef8b023`, after
the two baseline transplants but before the `file_path` repair. With fixed
Madaros SHA-256
`2080defa2f1042ef0b0c3d6796e77de0226e2b840856af5da7d2e36ee911e253`, it
retained the same boundary:

```text
REMOTE: fpcheck rc=0 errors=0
REMOTE: hello compile_rc=0 run_rc=0 output=Hello, Sounio
into_acc_done=40 minimum=40
first_failure=IR lowering failed during merge: ir_into_acc_failed
cause: cannot safely lower print/println argument with unresolved scalar kind
```

Foundry job `11474` measured the complete replay at `ec98d9886c`. With the same
fixed Madaros and gate order, it passed the raised acceptance floor:

```text
REMOTE: fpcheck rc=0 errors=0
REMOTE: hello compile_rc=0 run_rc=0 output=Hello, Sounio
ir_max_functions 16384
merged_ir_functions=13140
into_acc_done=123 minimum=122
first_failure=Error: Failed to write native binary to /tmp/sounio-remote-927140/fixed-point/madaros.gen2 rc=19
```

Slurm recorded `11473` as `COMPLETED/0:0` in `00:12:17` and `11474` as
`COMPLETED/0:0` in `00:33:46`, both on `gpuorangefs-r770-proxmox`. The merged
function count rose from the anchored run's 13,107 to 13,140, but remained below
the 16,384 source cap. A fresh strict closure census also passed with 124 nodes,
11,285 function declarations, and 5,099 slots of headroom. The blocker criterion
did not drift: the default floor remains 40, the CI floor remains 122, and the
localized repair still advances the fixed compiler from 40 to 123 completed
imports.

The complete transcripts and hash-bound replay receipt are retained at:

- `artifacts/audit/madaros_gen2_40_to_123_live_main_baseline_foundry_20260901.txt`
- `artifacts/audit/madaros_gen2_40_to_123_live_main_postfix_foundry_20260901.txt`
- `artifacts/audit/madaros_gen2_40_to_123_live_main_replay_receipt_20260901.json`

## Closed blocker

`BLK-20260901-gen2-current-zero` is closed. Its acceptance gate passed with
`into_acc_done=40`, independent hello and self-check controls green, and the
fixed Madaros SHA-256 recorded above.

`BLK-20260901-gen2-40-to-122` is closed. Its acceptance gate passed with
`into_acc_done=123`, self-check `rc=0` with zero errors, independent hello
compile/run green, and the same fixed Madaros SHA-256.

## Remaining boundary

No same-lane B1 blocker is opened after closing
`BLK-20260901-gen2-40-to-122`. The 123-floor evidence answers this lane's
acceptance question. The later code-buffer boundary belongs in a separate lane
with its own owner and acceptance gate if pursued.

## Review disposition

The earlier Grok 4.6 review correctly warned that a crashing inverse was not an
acceptance oracle. The fixed-seed rerun sharpened that warning: rebuilding the
compiler for each row was the primary attribution confound. The repaired source
then passed the direct and canonical 40 floors, and the pinned-anchor follow-up
passed both 122-floor forms at `into_acc_done=123`. No math claim, clinical
pathway, or external-facing artifact changed, so repository policy did not
require an additional LLM offload.
