<!-- docs:meta
topic_id: repo.docs.audit.madaros-raw-elf-8mib-checker-stack-2026-09-26
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-raw-elf-8mib-checker-stack-2026-09-26
-->

# Madaros `check` SIGSEGV on `darwin_pbpk/release/matrix_er.sio` is stack exhaustion from calling the raw ELF

**Date:** 2026-09-26
**Base:** `main` at `98315edcdb`
**Status:** diagnosed. No `self-hosted/` change was made. The two follow-ups proposed below were not applied.

## Report under investigation

A Madaros built from `98315edcdb` with `make build-madaros` segfaulted (rc=139)
inside the checker on `stdlib/darwin_pbpk/release/matrix_er.sio`. The last line
printed was `run_check_mode: about to check 2 modules`. lean_single accepted the
file. Most of the `darwin_pbpk` test programs segfaulted in the same sweep. The
reporting session later confirmed it had invoked
`artifacts/self-hosted/madaros` directly.

## Minimal repro (5 lines)

```sounio
fn h(x: f64) -> f64 { return x }
fn main() -> i32 {
    let e = h(h(h(h(1.0))))
    return 0
}
```

| invocation (same ELF, sha256 `e2256828…c401d`, built from `98315edcdb`) | rc |
|---|---:|
| `artifacts/self-hosted/madaros check repro.sio` under the default `ulimit -s 8192` | 139 |
| the same with `h(h(h(1.0)))` (depth 3) | 0 |
| `ulimit -s 65536 && artifacts/self-hosted/madaros check repro.sio` | 0 |
| `bin/madaros check repro.sio` | 0 |
| `bin/souc check repro.sio` | 0 |
| `SOUNIO_SOUC_ENGINE=lean_single bin/souc check repro.sio` | 0 |

The committed prebuilt `bin/madaros-linux-x86_64.gz` (sha256 `31a15e31…`,
source `70fa3952`) gives the same results, so this is not a regression since
2026-09-14.

## How the reduction went

1. `matrix_er.sio` → item-level delta: every item removed, crash kept. The
   only thing left was `use darwin_pbpk::tsit5_pbpk14::{abs_f64, sqrt_f64}` plus
   an empty `main`. That is why the user's one-import probe crashed for
   `matrix_er` and passed for `tsit5_pbpk28`, `pbpk28_params` and the others:
   `matrix_er` is the only one that pulls in `tsit5_pbpk14`.
2. `tsit5_pbpk14.sio` checked on its own → rc=139. Item-level delta reduced
   it to `tsit5_step_pbpk`, whose RK stages nest `pbpk_state_add(...)` calls
   seven deep.
3. Nesting depth scan: depth 3 or less passes and depth 4 or more crashes. The
   argument position, the argument type (f64, i64 or a struct) and whether the
   callee is defined do not matter.
4. `ulimit -s` scan: the crash goes away above 8 MiB, so it is stack
   exhaustion, not a logic fault in the checker.

## Stack cost per nesting level

This is the smallest `ulimit -s` at which `check` exits 0. It was found by
binary search to within 16 KiB, using the raw ELF.

| construct nested `d` deep in one `let` | d=2 | d=6 | per level |
|---|---:|---:|---:|
| free-function call `h(…)` | 5885 KiB | 12106 KiB | **~1555 KiB** |
| method call `s.m(…)` | 5853 KiB | 11994 KiB | ~1535 KiB |
| enum constructor `Some(…)` | 4094 KiB | 9707 KiB | ~1403 KiB |
| block `{ … }` | 1743 KiB | 3022 KiB | ~319 KiB |
| parentheses `( … )` | 1007 KiB | 1007 KiB | 0 |
| binary `(… + 1.0)` (d=4 → d=8) | 4955 KiB | 5194 KiB | ~60 KiB |

Free-function calls, exactly (d = 0…8): 1007, 4331, 5882, 7448, 8999, 10549,
12100, —, 15217 KiB. The cost is linear at about 1.5 MiB per level of call
nesting, so 8 MiB is exhausted at depth 4. `tsit5_pbpk14`'s seven-deep RK stage
sums need about 13.5 MiB.

**Where the frame is (hypothesis, not measured).** The ELF has no symbol table
and there is no debugger on the workspace pod, so the ~1.5 MiB has not been
attributed to one function. Every call-like form pays about the same amount.
That points at a frame shared by call checking, not at the expression spine,
since blocks and parentheses are cheap. The leading candidate is
`checker_check_call_expr_inplace`
(`self-hosted/check/check.sio`, ~10594). It takes up to five by-value
snapshots, `let call_start_borrows = (*c).borrows`, of a `BorrowEnv`. Each
snapshot holds 128 × `BorrowEntry{ name: Name{ buf: [i8; 384] } … }` plus
`branch_snap: [bool; 4096]`, and each is live across the argument recursion.
The in-source comment at `checker_check_one_call_arg_inplace` ("~12MB+ held
across this fn's nested-call recursion") records the same family of problem
from an earlier round. Confirming it needs a build that removes the snapshots
and re-runs the table above.

## This is the documented contract, not a new defect

`bin/madaros` reserves the stack before exec (`MADAROS_STACK_KB`, default
512 MiB, Lane A2-lite of `MADAROS_HARDENING_PLAN_2026-08-12.md`). The comment
there says "Raw Madaros needs more than the default 8 MiB Linux stack".
`bin/souc` routes through that wrapper. The CI gates raise the limit
themselves: `aggregate_field_identity_gate.sh` sets 512 MiB and
`canonical_compiler_gate.sh` sets 1 GiB.
`docs/audit/SWEEP_STACK_LIMIT_CORRECTION_2026-09-09.md` already found that an
8 MiB sweep produced 429–514 spurious SIGSEGVs, and all of them went away
under the project limit. The 2026-09-26 report was the same trap, reached by
calling `artifacts/self-hosted/madaros` directly.

The wrapper's reservation is best effort, not a guarantee. `bin/madaros`
ignores a failed `ulimit -s` (`|| true`) and carries on at whatever limit it
inherited, and a caller can set `MADAROS_STACK_KB` below the default. On a
host whose hard stack limit is below what the checker needs, the wrapper
therefore hits the same SIGSEGV. Every wrapper result in this record was
measured on the sounio-workspace pod, where `ulimit -Hs` is `unlimited`, so
the raise succeeded there.

## Re-measured through the wrapper

The compiler is the same ELF, built from unmodified `98315edcdb` (sha256
`e2256828…`) before any source change. The source tree differed from
`98315edcdb` in one input: `stdlib/darwin_pbpk/scenarios/venlafaxine_xr.sio`
carried the one-line `pub fn vfx_scenario_init` change, which does not affect
the ELF. Each fixture was first compiled with the raw ELF at 8 MiB and then run
with `bin/souc run`.

| fixture | raw ELF @ 8 MiB | `bin/souc run` |
|---|---:|---|
| `stdlib/darwin_pbpk/release/matrix_er.sio` | 139 | 0, PASS |
| `stdlib/darwin_pbpk/scenarios/venlafaxine_xr.sio` | 139 | 0, PASS |
| `tests/run-pass/darwin_venlafaxine_xr_pgx_smoke.sio` | 139 | 0, `VENLAFAXINE_XR_PGX_SMOKE_PASS` |
| `tests/run-pass/darwin_venlafaxine_xr_matrix_smoke.sio` | 139 | 0, sentinel OK |
| `tests/run-pass/dissertation_midazolam_scenario_smoke.sio` | 139 | 0, `MDZ_SCENARIO_SMOKE_PASS` |
| `tests/run-pass/dissertation_pbpk14_hessian.sio` | 139 | 0, sentinel OK |
| `tests/run-pass/dissertation_pbpk28_parity_ref_venlafaxine.sio` | 139 | 0, `…_PARITY_DONE` |
| `tests/run-pass/dissertation_pbpk28_parity_ref_semaglutide.sio` | 139 | 0, `…_PARITY_DONE` |
| `tests/stdlib/darwin_pbpk/bbb/test_bbb_{kpuu_steady,mass_balance,transient,validation_rapamycin}.sio` | 139 | 0, sentinel OK |

Without the `pub` change, the pgx smoke fails under the wrapper with
`error[E175] … function is private in its defining module` at both
`vfx_scenario_init` call sites. lean_single does not enforce E175 across
modules. That one-line change is deliberately left out of this record's
commit. It already exists as `0f2558e87` on `feat/w1-qd128-transcend` and its
descendants, and as `d89ba11c7` on `claude/competent-mcclintock-b137de` and
`claude/determined-kilby-3fc1ca`. None of those branches is on `main` yet.

### Failures that remain under the wrapper (separate defects, not diagnosed here)

These fail under a correctly launched Madaros, and a sample of each ran clean
under lean_single. None of them is stack-related:

- `stdlib/darwin_pbpk/scenarios/semaglutide_sc_depot.sio`: rc=132 at run
  time. The build prints `NATIVE_REFUSAL kind=empty_stub_ud2
  name=pbpk28_state_zero reason=missing_lowered_body`.
- `test_bbb_gum_budget`, `test_bbb_hdmr_7d`, `test_bbb_pce2d_sobol`,
  `test_bbb_pce_vs_gum` and `test_des_bbb_coupled`: the build succeeds, then
  the run exits 182 with `madaros: handles full`.
- `test_bbb_gate` and `test_bbb_voi`: `error[E137] use of undeclared variable`
  in `bbb_gate_print_verdict` and `bbb_voi_print`.

Two of these were diagnosed later the same day by other sessions. The
findings below are as reported over `sounio-coord`; I have not re-measured
them:

- **E137 is correct, and lean_single is the engine that is wrong.** The
  undeclared name is `print_i64`. `bbb_gate.sio` and `bbb_voi.sio` neither
  declare nor import it. Other, unrelated modules define their own local
  `print_i64`, and the builtin is `print_int`. lean_single passes because it never fails the build on a type
  error inside an imported function. It compiles the call to `xor eax,eax`,
  and the integers silently drop out of the output. See
  `docs/audit/LEAN_SINGLE_IMPORTED_TYPE_ERROR_FAIL_OPEN_2026-09-26.md`
  (branch `claude/confident-kilby-8d4142`). Madaros built from `2e8b76d` also
  reports E259 on these tests, because the test `main` reads private struct
  fields.
- **rc=182 is the unreclaimed handle table** already described in the
  2026-08-17 dispatch. Every construction of a struct larger than 16 bytes
  takes a handle that is never freed. The required table size ranges from
  1.95× to 10.3× the 2^22 cap, so raising the cap cannot fix it. The
  checked-in dispatch is
  `docs/audit/MADAROS_HANDLE_TABLE_182_LIFETIME_DISPATCH_2026-08-17.md`. The
  per-construction repro and the 1.95×–10.3× range come from the
  sleepy-easley session and are not in this tree yet. Treat that range as
  reported, not reproducible from `main`.

## Proposed follow-ups (dispatch, not applied)

1. **Fail closed at the raw entry point.** When the raw ELF starts under a
   soft `RLIMIT_STACK` below what the checker needs, it should either raise the
   limit itself with `setrlimit` and re-exec, or refuse with a named
   diagnostic. At present a bare SIGSEGV after `about to check N modules` looks
   exactly like a checker crash. This change belongs in the driver
   (`self-hosted/compiler/`), not in `check/`. `bin/madaros` should fail
   closed as well: if the soft stack limit cannot be raised to
   `MADAROS_STACK_KB`, it should refuse with a named diagnostic instead of
   ignoring the `ulimit` failure. The follow-up should also verify that
   refusal under a lowered hard limit, for example
   `( ulimit -Ss 8192 && ulimit -Hs 8192 && bin/madaros check repro.sio )`. The
   soft limit is lowered first, each step gates the next, and the
   irreversible hard-limit change stays inside the subshell.
2. **Shrink the call-checking frame.** First confirm the attribution above by
   building with the `call_start_borrows` snapshots moved out of the recursive
   frame (heap, or one snapshot slot on the `Checker`) and re-running the
   per-level table. The target is a per-level cost small enough that
   stdlib-shaped code, meaning RK stage sums about 7 deep, checks inside
   8 MiB. This is part of the "IR-arena root fix" that the wrapper comment
   defers to.

## Reproduce

```bash
make build-madaros                       # bare; never wrap it in souc-build-lock.sh
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
printf 'fn h(x: f64) -> f64 { return x }\nfn main() -> i32 {\n    let e = h(h(h(h(1.0))))\n    return 0\n}\n' > /tmp/repro.sio
( ulimit -s 8192 && artifacts/self-hosted/madaros check /tmp/repro.sio ); echo rc=$?   # 139
( ulimit -s 65536 && artifacts/self-hosted/madaros check /tmp/repro.sio ); echo rc=$?   # 0; a failed raise returns ulimit's error, not 139
./bin/souc check /tmp/repro.sio; echo rc=$?   # 0 only where bin/madaros could raise the limit (check ulimit -Hs)
```
