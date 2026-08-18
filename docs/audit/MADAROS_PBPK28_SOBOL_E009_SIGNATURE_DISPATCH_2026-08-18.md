# Madaros PBPK28 Sobol E009 Signature Dispatch

Date: 2026-08-18
Pre-fix baseline: `origin/main` at `41c087cba66a77b52afb0d18ce3f844de975b19f`
Worktree: `/tmp/sounio-e009`
Lane: `e009-pbpk28-sobol-20260818`
Source-built validation branch: `lane/codex-2/e009-pbpk28-sobol-20260818` at
`f1c94bd1a0526cd2c7c9345abcbb23df5346e206`

## Verdict

The primary E009 is a **stdlib API signature defect**, not a Madaros function
resolver selecting the wrong function. `saltelli_run` (and its two internal
callback boundaries) declared a pure callback type:

```sounio
fn([f64; 10]) -> f64
```

Both PBPK28 callbacks are explicitly effectful:

```sounio
fn([f64; 10]) -> f64 with Mut, Div, Panic
```

The callbacks perform mutable state updates and division, and call code whose
failure mode is `Panic`. An effectful function cannot be passed to a pure
function slot. The correct repair is to carry `with Mut, Div, Panic` through
`saltelli_eval_row`, `saltelli_analyze`, and `saltelli_run`. No numerical
formula, sample scheme, or scientific assertion changed.

The earlier report used `expected fn#165`; the current pre-fix run on the
newer `origin/main` lineage printed `expected fn#167`, `found fn#6` and
`found fn#11`. These numbers are not stable function names. They are local
`FnSigTable` indices (`TypeEntry.fn_sig_id`) printed by the diagnostic, so a
different module/signature collection order can change them. They do not, by
themselves, establish a resolver collision.

## Source Evidence

The pre-fix declarations were:

- `stdlib/epistemic/sobol.sio:1326`: `saltelli_eval_row` accepted a bare
  `fn([f64; 10]) -> f64`.
- `stdlib/epistemic/sobol.sio:1353-1356`: `saltelli_analyze` accepted the same
  bare callback type.
- `stdlib/epistemic/sobol.sio:1474-1479`: public `saltelli_run` accepted the
  same bare callback type.
- `stdlib/darwin_pbpk/validation/pbpk28_sobol_pce.sio:141`: the rapamycin
  callback is declared `with Mut, Div, Panic`.
- `stdlib/darwin_pbpk/validation/pbpk28_sobol_pce.sio:396`: the semaglutide
  callback is declared `with Mut, Div, Panic`.
- `stdlib/darwin_pbpk/validation/pbpk28_sobol_pce.sio:183` and `:432`: those
  callbacks are passed to `saltelli_run`.

The checker makes the intended rule explicit:

- `self-hosted/check/check.sio:14693-14704` accepts a provided effect row only
  when it is a subset of the expected row. A non-empty effect row therefore
  cannot flow into a pure slot.
- `self-hosted/check/check.sio:14752-14780` compares function signatures
  structurally, including the effect-row check; it does not compare a name
  hash or silently coerce a callback.
- `self-hosted/check/check.sio:8189-8194` resolves a bare function identifier
  to the selected `FnSigTable` entry.
- `self-hosted/check/check.sio:7349-7355` delegates lookup to
  `fn_sig_table_find_prefer_module`, whose implementation in
  `self-hosted/check/defs.sio:1490-1560` prefers the function defined in the
  module currently being checked before considering public imports.

The E009 is raised at the type boundary before IR lowering. The observed
`IR_NAME_POOL_LEN` and 8-byte name/storage collisions are therefore separate
collision classes; neither is evidence for this particular failure.

## Decisive Controls

The controls were deliberately arranged so that the instrument had to reject
one case and accept two others.

`/tmp/e009-fn-controls.sio` contains:

1. a pure callback passed to a pure slot;
2. an effectful callback passed to an effectful slot;
3. an effectful callback passed to a pure slot.

With the default Madaros launcher (the checked-in prebuilt ELF), the first two
forms were accepted and the third produced exactly one E009:

```text
error[E009] ... argument type does not match parameter
  = expected fn#2
  = found fn#1
```

Removing only the deliberately invalid third call produced `check: OK`.
The same invalid program was accepted by `SOUNIO_SOUC_ENGINE=lean_single`,
which is the expected legacy under-checking gap and not evidence that the
Madaros resolver is wrong.

Before the API repair, default Madaros checking of
`pbpk28_sobol_pce.sio` produced two E009s, one at each callback call site:

```text
expected fn#167 / found fn#6
expected fn#167 / found fn#11
```

After carrying the effect row through all three Sobol callback boundaries,
the same check produced **no E009**. It still produced one independent E035
on `darwin_pbpk/epistemic_pbpk28::main` (missing `Epistemic`), which is a
separate imported-module effect-inference issue and remains open.

## Change

`stdlib/epistemic/sobol.sio` now declares `with Mut, Div, Panic` on:

- the callback parameter and function result of `saltelli_eval_row`;
- the callback parameter and function result of `saltelli_analyze`;
- the callback parameter and function result of `saltelli_run`.

This is the smallest contract repair that makes the API describe the callback
it actually invokes. `pbpk28_sobol_pce.sio` did not need a workaround or a
callback rename.

## Validation Boundary

The focused checks in this worktree used the checked-in
`bin/madaros-linux-x86_64` through `./bin/souc`; that binary is prebuilt and
is labelled as such here. The source-built rerun was then completed through
the working Slurm path, so the compiler result below is current-source
evidence rather than an inference from the prebuilt ELF.

The operational blocker was narrower than "Slurm unavailable": `sbatch` is
unusable for the `openvscode-server` submitter and left held controller-side
job records, but `srun` is functional. The positive control
`bash scripts/dev/slurm_srun_minimal.sh "hostname; nproc; echo OK"` returned
`cpuops-t560-proxmox / 32 / OK / rc=0`. The rerun used the scrubbed wrapper
with `--partition=all`, `--chdir=/tmp`, and `/bin/bash`.

The node had no readable CA bundle at the standard Linux paths, so the clone
step required an explicit, one-command `git -c http.sslVerify=false` transfer
of the public branch. This is recorded as node plumbing, not as compiler
evidence. The clone identified the validation input as
`f1c94bd1a0526cd2c7c9345abcbb23df5346e206`. Building
`self-hosted/compiler/main.sio` from that tree completed and produced a
`100084269` byte Madaros ELF at
`artifacts/self-hosted/madaros`. Running that ELF on
`pbpk28_sobol_pce.sio` returned `rc=1` with one E035:

```text
error[E035] ... effect not declared in function signature (missing: Epistemic)
    -- required by `ep28_selftest_main`
```

The source-built output contained no E009. Thus the repaired Sobol callback
contract is accepted by the current-source compiler; the remaining E035 is
an independent imported-module effect-inference issue and is not evidence of
a resolver collision. The remote `origin/main` subsequently moved to
`743be30b260f44dc557de94ecf94fff1ab6921e5`; this receipt names the exact
published repair branch and SHA actually built, rather than attributing the
run to a later main commit it did not contain.

No compiler, IR, native-codegen, or name-pool file was changed in this lane.
