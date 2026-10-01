<!-- docs:meta
topic_id: repo.docs.audit.madaros-no-main-entry-fn0-dispatch-2026-09-26
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-no-main-entry-fn0-dispatch-2026-09-26
-->

# Madaros builds a program with no `main` into an ELF that jumps into function 0 — dispatch

**Date:** 2026-09-26
**Base:** `main` @ `2e8b76d31`
**Engines:** see "Measurements" for which binary produced each row. Source-built Madaros:
`make build-madaros` at `2e8b76d31` (content-cache key `4784102d…`, the key of this worktree's own tree),
`artifacts/self-hosted/madaros` 125 110 129 bytes, md5 `5764851f…`, invoked through `./bin/souc`. The committed
Madaros ELF was measured too; both agree on every row.
**Status:** fix applied 2026-09-26 on operator instruction ("use E221 to match lean_single, apply the
fix"). See "Fix (applied)". Everything above that section describes the unpatched compiler.
**Related:** [`ENGINE_DIVERGENCE_E221_REFINEMENTS_2026-08-30.md`](ENGINE_DIVERGENCE_E221_REFINEMENTS_2026-08-30.md) §1
recorded that Madaros accepts a program with no `main` (`rc=0`, writes an ELF) where lean_single
refuses with `error[E221]: no main`. It did not run the ELF. This dispatch runs it, explains
what executes, and explains why the body is missing.

## Reported symptom

```
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run stdlib/darwin_pbpk/scenarios/semaglutide_sc_depot.sio
NATIVE_REFUSAL kind=empty_stub_ud2 fn=0 name=pbpk28_state_zero reason=missing_lowered_body
... Illegal instruction  (rc=132)
```

The report framed this as "an imported function's body is not lowered, depending on how
`semaglutide_sc_depot` imports or reaches it", since other importers of `pbpk28_state_zero`
(for example `tests/run-pass/darwin_venlafaxine_xr_pgx_smoke.sio`) run fine.

**That framing does not hold.** `semaglutide_sc_depot.sio` has no `fn main`. It is a library.
Its only runnable caller is `tests/run-pass/darwin_sema_sc_depot_smoke.sio`. The importers that
work all have a `main`. The import shape does not matter: removing `main` from any importing
program reproduces the crash, and adding one fixes it.

> Operating principle 3 applies to the command line: `souc run` on a library file is a category
> error. The compiler defect is that Madaros does not **say** so. It emits an executable with no
> entry point, and that executable dies in a way that looks like a lowering bug.

The header of `semaglutide_sc_depot.sio` calls itself a "runnable scenario". It is not runnable.
`des_sirolimus.sio` and `venlafaxine_xr.sio` in the same directory each define a `main`.

## Minimal reproduction (19 lines across four files)

`docs/audit/repro/no_main_entry/`:

```sounio
// lib/nm/dep.sio
pub fn z_first_unused() -> i64 { 11 }
pub fn y_called() -> i64 { 22 }

// nomain.sio
use nm::dep::*
fn helper() -> i64 { z_first_unused() + y_called() }

// withmain.sio  (control)
use nm::dep::*
fn helper() -> i64 { z_first_unused() + y_called() }
fn main() with IO { println(helper()) }
```

```bash
cd docs/audit/repro/no_main_entry
SOUNIO_STDLIB_PATH=$PWD/lib ../../../../bin/souc compile nomain.sio   -o /tmp/nomain.elf;   /tmp/nomain.elf;   echo rc=$?
SOUNIO_STDLIB_PATH=$PWD/lib ../../../../bin/souc compile withmain.sio -o /tmp/withmain.elf; /tmp/withmain.elf; echo rc=$?
```

`stdlib_nomain.sio` is the reported symptom in two lines, against the real
`darwin_pbpk::core::pbpk28_params`. Run it from the repo root with
`SOUNIO_STDLIB_PATH=$PWD/stdlib`. The lean_single rows must also be run from the repo root: run
from the repro directory, lean_single resolves imports against `self-hosted/` relative to the
working directory, and stops at `error[E224]: unreadable import` before it reaches the
`main` check.

## Measurements

Workspace pod, worktree `/workspace/worktrees/claude-sema-nomain` at `2e8b76d31`.
"committed" = the Madaros ELF committed at that commit, reached through `./bin/souc` (fresh
worktree, no `artifacts/self-hosted/madaros`). "source" = Madaros built from that commit with
`make build-madaros` and reached through `./bin/souc`.

| program | engine | build log | ELF entry | exec |
|---|---|---|---|---|
| `nomain.sio` | Madaros committed | `seed_main_idx 0`, `NATIVE_REFUSAL kind=empty_stub_ud2 fn=0 name=z_first_unused` | `0x401000` (`.text`+0) | **SIGILL, rc=132** |
| `withmain.sio` | Madaros committed | `seed_main_idx 3`, no refusal | `0x40119a` | prints `33`, rc=0 |
| `stdlib_nomain.sio` | Madaros committed | `NATIVE_REFUSAL ... fn=0 name=pbpk28_state_zero` | `0x401000` | **SIGILL, rc=132** |
| same plus `fn main` that calls `init()` | Madaros committed | `seed_main_idx 2`, no refusal | `0x40165c` | rc=0 |
| `semaglutide_sc_depot.sio` | Madaros committed | `NATIVE_REFUSAL ... fn=0 name=pbpk28_state_zero` | — | **SIGILL, rc=132** |
| `fn helper() -> i64 { return 7 }` alone (no imports) | Madaros committed | no refusal | `0x401000` | **SIGSEGV, rc=139** |
| any of the no-`main` files above | lean_single | `error[E221]: no main` | no ELF | rc=1 |
| `nomain.sio` | **Madaros source** | `seed_main_idx 0`, `NATIVE_REFUSAL ... fn=0 name=z_first_unused` | `0x401000` | **SIGILL, rc=132** |
| `withmain.sio` | **Madaros source** | `seed_main_idx 3`, no refusal | `0x4011f8` | prints `33`, rc=0 |
| `stdlib_nomain.sio` | **Madaros source** | `NATIVE_REFUSAL ... fn=0 name=pbpk28_state_zero` | `0x401000` | **SIGILL, rc=132** |
| same plus `fn main` that calls `init()` | **Madaros source** | `seed_main_idx 2`, no refusal | `0x40165c` | rc=0 |
| `semaglutide_sc_depot.sio` | **Madaros source** | `seed_main_idx 0`, `NATIVE_REFUSAL ... fn=0 name=pbpk28_state_zero` | `0x401000` | **SIGILL, rc=132** |
| `fn helper() -> i64 { return 7 }` alone | **Madaros source** | no refusal | `0x401000` | **SIGSEGV, rc=139** |

One discrepancy with the report: under `./bin/souc run`, lean_single exits **rc=1**, not 0,
on `semaglutide_sc_depot.sio`. `run` swallows the compile output, so the refusal is silent there.
`SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile ...` prints `error[E221]: no main`.

## Root cause: two defects, both keyed on "no user `main`"

### D1: no entry point, but the executable is written anyway (this is what crashes)

`self-hosted/native/codegen_x86_linux.sio`, `compile_native_v2_preview_to_file`
(the writer both `compile_imported_to_file` routes call, `module_frontend.sio:6864` and `:6968`):

```sounio
let main_idx = find_main_index(module)          // :13180  -> -1
if main_idx >= 0 {
    emit_entry_trampoline_with_global_inits_into(&! nc, module, main_idx)
}
...
let entry = trampoline_offset(&nc)              // :13188  -> nc.entry_offset
```

`entry_offset` is set only by the trampoline emitters. Its initial value is `0`
(`compiler_new`, `:12263`). With no `main`, the ELF header's `e_entry` is `.text + 0`, which is
the first byte of merged function 0. The five overflow tiers checked between `:13188` and the
write at `:13254` all pass, and the binary is written.

The kernel then enters function 0 with no frame and no return address.
- If fn 0 has a body, it runs and its `ret` pops `argc` as a return address: **SIGSEGV** (the
  no-imports row).
- If fn 0 is an empty stub, the backend emits `ud2` for it: **SIGILL** (every importing row).
  D2 is why it is empty.

The same unguarded `main_idx >= 0` pattern appears at `:6883`, `:10820` and `:13276`. The
defect belongs to the writer contract, not to one path.

### D2: the "restore user `main`" step overwrites merged slot 0 with an empty snapshot (this is why the body is missing)

`self-hosted/compiler/module_frontend.sio`, multi-module lowering (`lower_array`, into-acc mode):

```sounio
var seed_main_idx = module_frontend_find_user_main_idx(&(*acc_box))   // :5702 -> -1
if seed_main_idx < 0 {
    seed_main_idx = 0                                                   // :5704  fallback
}
var seed_user_main = ir_function_deep_copy(ir_fn_get(&(*acc_box), seed_main_idx as usize))
... into-acc lowers every dep, filling dep bodies into their low fn ids ...
if seed_main_idx >= 0 && seed_main_idx < (*acc_box).fn_count {
    ir_fn_set(&! (*acc_box), seed_main_idx as usize, seed_user_main)   // :5998
}
```

The seed (main-module) lowering reserves slots for the imported functions, and the first function
declared in the first dependency takes slot 0. At snapshot time that slot is still body-less. The
into-acc pass fills it (`fn_before N fn_after N`: no new slots are appended). The restore then
writes the pre-dependency snapshot back over it. The body is lowered and then erased.

The comment directly above the fallback describes this hazard ("restoring functions[0] destroyed
those bodies → order_spread SEGV"). The `< 0 → 0` fallback reintroduces it exactly when there is
no `main` to find. The `>= 0` guard at `:5997` was written to make the restore optional, but the
fallback means it is never skipped.

**Discriminating evidence** (committed ELF, rows 1–2 above, plus two variants):
- The emptied function is always the one at slot 0. It is `z_first_unused` in `nomain.sio` and
  `pbpk28_state_zero` in the darwin repro, because each is declared first in its dependency.
  It is emptied whether or not it is called: `z_first_unused` is emptied both when only
  `y_called` is used and when both are used.
- `y_called`, the called function at slot 1, keeps its body in every variant.
- Reordering the functions in the *main* module (3 variants against `pbpk28_params`) never
  moves the refusal. `pbpk28_state_zero` is slot 0 because it is the first `pub fn` in
  `pbpk28_params.sio`.
- With a `main` present, the snapshot index points at `main`, the restore is correct, and the
  same dependency function keeps its body (control row).

On its own, D2 only corrupts no-`main` builds, because `find_user_main_idx` succeeds whenever a
`main` with a body exists. Fixing D1 so that no-`main` executables are refused makes D2
unreachable for executables. It is still worth closing, because the snapshot/restore should not
run at all when there is nothing to restore.

## Fix (applied)

The operator chose `E221` to match lean_single.

1. **D1, frontend refusal** (`module_frontend.sio`, `module_frontend_compile_imported_to_file`,
   right after `imported_compile: typecheck ok`). If the entry file's own AST declares no
   top-level `fn main`, print `error[E221]: no main` and a `note:` naming the file, then return 1
   before lowering. The check sits before the single-module/multi-module split, so one site
   covers both writer calls. It scans `programs[0].items` (new helper
   `module_frontend_items_declare_main`), not merged IR, so a dependency that declares its own
   `main` (`scenarios/venlafaxine_xr.sio` does) cannot make a library entry file look runnable.
   Type errors still report first, as in lean_single. `check` and `--emit-obj` do not take this
   route and still accept libraries.
2. **D1, writer backstop** (`codegen_x86_linux.sio`, `compile_native_v2_preview_to_file`).
   `main_idx < 0` now prints `error[E221]: no main -- refusing to write an executable with no
   entry point` and returns **rc=24**, instead of writing an ELF whose `e_entry` is `.text+0`.
   Every other caller of this writer (the hand-built witnesses in `compiler/main.sio`) names its
   entry function `main`. This was checked for all 34 `compiler_main_make_native_v2_*` builders.
3. **D2, lowering** (`module_frontend.sio`, `lower_array` seed). With no user `main`,
   `seed_main_idx` stays `-1`. The snapshot is not taken, and the existing `>= 0` guard skips the
   restore. The `< 0 → 0` fallback is gone.
4. **Witness:** `tests/compile-fail/madaros_no_main_imported_library.sio`
   (`//@ requires: madaros`, the importing-library shape).
   `tests/compile-fail/diagnostic_codes_no_main.sio` already covers the import-free shape on both
   engines.

Left alone: the header of `semaglutide_sc_depot.sio`, which still calls itself "runnable", and
the visibility finding below. Both are `stdlib/` changes outside this fix.

### Verification

Patched Madaros built from `2e8b76d31` plus this change: `make build-madaros`, md5 `123e3657…`,
invoked through `./bin/souc`.

| program | before | after |
|---|---|---|
| `nomain.sio` | ELF written, SIGILL rc=132 | `error[E221]: no main`, rc=1, no ELF |
| `stdlib_nomain.sio` | ELF written, SIGILL rc=132 | `error[E221]: no main`, rc=1, no ELF |
| `semaglutide_sc_depot.sio` (`compile` and `run`) | SIGILL rc=132 | `error[E221]: no main`, rc=1 |
| `fn helper() -> i64 { return 7 }` alone | ELF written, SIGSEGV rc=139 | `error[E221]: no main`, rc=1 |
| `withmain.sio` | `33`, rc=0 | `33`, rc=0 |
| `pbpk28_params` importer with `main` calling it / not calling it | rc=0 / rc=0 | rc=0 / rc=0 |
| `check` on `nomain.sio` and on `semaglutide_sc_depot.sio` | `check: OK` | `check: OK` |
| `--emit-obj nomain.sio`: offset of `y_called` (i.e. the size of slot 0) | `0x2`, `seed_main_idx 0` | `0x27`, `seed_main_idx -1` (same layout as with `main`) |

- `scripts/ci/native_v2_e2e_exit_code_gate.sh` (the writer path with a hand-built `main`): PASS.
- `scripts/ci/madaros_wide_int_gate.sh` fails with `wide-add4 rc=22` (IR arena contract, handle
  `-1`). The unpatched base fails identically, so this is pre-existing and not caused by this
  change.

**Compiled-output identity, base vs patched.** For a program that has a `main`, the change should
be a no-op. The check: compile the same file with the base Madaros (md5 `5764851f…`) and the
patched one (md5 `123e3657…`), one pair at a time with no timeout, and `cmp` the ELFs.
- The determinism control passes: the same binary and file compiled twice gives the same md5.
- The sample is 66 run-pass files: every tenth `*imported*` test (40), all `darwin*`,
  `*multimod*` and `*module*` tests, and the two CPC epistemic receipts.

| outcome | count | files |
|---|---:|---|
| byte-identical ELF | **61** | all the rest, including `order_spread_exact_n4` and every `darwin_*` that builds |
| ELFs differ | **0** | |
| refused by both, pre-existing | 3 | `darwin_sema_sc_depot_smoke` and `darwin_venlafaxine_xr_pgx_smoke` (privacy E175/E259, see below); `octonion_associator_gum_validation` (E245/E004 under Madaros; it is a lean_single receipt) |
| built by base, refused by patched | 2 | `imported_module_f64_const_a` and `_b`: `//@ ignore` leaf libraries with no `main`. Base wrote an ELF with entry `.text+0`; patched says `error[E221]: no main`. This is the intended change. |

- Harness: `tests/compile-fail/madaros_no_main_imported_library.sio` and
  `diagnostic_codes_no_main.sio` both pass under patched Madaros
  (`--filter no_main`, `SOUNIO_MADAROS_AVAILABLE=1`: 2 pass, 0 fail).
- `--filter darwin`: 8 pass / 3 fail on both binaries, with the same three names. All three are
  refused at preflight (type check/privacy), before the changed code runs.
- The full `--filter imported` run is **not evidence either way**. It was run on a shared 8-CPU
  pod at load average ~50, and all 319 of its failures are `run timed out`; none mentions E221.
  The run was stopped, and the identity comparison above replaced it.

`E221` remains overloaded in Madaros. `check.sio:16840` still uses it for "math function bound
but not emittable" (issue #2507, `docs/llm-guide/error-catalog.md:452`). This fix adds the
lean_single meaning alongside it, as the operator chose, and does not resolve the collision.

## Adjacent finding: the scenario's real caller does not build under source Madaros

`tests/run-pass/darwin_sema_sc_depot_smoke.sio` is the program that actually runs
`semaglutide_sc_depot`. Under the source-built Madaros above it is **refused**, `rc=1`:

```
error[E175] ... darwin_sema_sc_depot_smoke::main at 1310..1339: function is private in its defining module   (sema_scenario_init)
error[E175] ... darwin_sema_sc_depot_smoke::main at 1641..1687: function is private in its defining module   (sema_strang_step)
error[E259] ... darwin_sema_sc_depot_smoke::main at 1861..1871: struct field is private in its defining module (SemaglutideScenario.pbpk)
```

None of these three items is `pub` in `semaglutide_sc_depot.sio`. lean_single does not enforce
cross-module privacy, so CI (which runs lean_single) stays green. Today the dissertation's
semaglutide scenario therefore has **no path that runs under the default engine**: its own file
has no `main`, and its caller is refused. The fix is a visibility change in `stdlib/`, not in
`self-hosted/`. It has the same shape as the `pub fn vfx_scenario_init` change for
`venlafaxine_xr.sio` in `d89ba11c7`, which is on a lane branch and not yet on `main`. It is not applied here.

## Not established

- **Whether `--emit-obj` output is correct.** The object's `.text` bytes read back as zeros
  whether or not the file has a `main`, so the object path has problems of its own. The fix
  changes only the symbol layout, which now matches the with-`main` build (see Verification).
  That is not evidence that the object's contents are correct.
- **Who compiles a no-`main` file and depends on `rc=0`.** A survey found none:
  - the 11 `tests/run-pass` files without `main` are all `//@ ignore` or `check-only`;
  - the `scripts/*_gate.sh` compile targets are test programs with a `main`;
  - nothing references `semaglutide_sc_depot` in `scripts/` or `.github/`.
  The survey was a grep, not an exhaustive run.
- The general hazard that `NATIVE_REFUSAL kind=empty_stub_ud2` is a log line and not a refusal
  (see the note at `module_frontend.sio:1890`). Any empty stub still gets written. This is wider
  than the no-`main` case and is out of scope here.
