<!-- docs:meta
topic_id: repo.docs.audit.madaros-struct-reassign-array-alias-dispatch-2026-09-26
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-struct-reassign-array-alias-dispatch-2026-09-26
-->

# Madaros aggregate assignment stores the source handle — dispatch

**Date:** 2026-09-26
**Base:** `origin/main` @ `2e8b76d31` (workspace worktree `/workspace/worktrees/claude-struct-reassign-alias`, branch `claude/madaros-struct-reassign-alias`)
**Engines:** Madaros, cross-checked against `SOUNIO_SOUC_ENGINE=lean_single`
**Owner:** unassigned (`self-hosted/ir/lower.sio`, `Lowerer::lower_assign_stmt_ref`)
**Status:** evidence recorded; source-built baseline **reproduces**; fix for identifier sources proposed and measured ([Patched build](#patched-build)); non-identifier sources measured and **left open** ([Boundary](#boundary-still-open-after-the-fix)).

## Why this dispatch

Found while measuring PBPK28 transport-fix candidates. A saved Runge–Kutta
stage (`xg = x; step(&!x)`) silently changed a rapamycin bolus AUC under Madaros
(2.821332 → 2.711950) while every state metric matched lean_single. See
"Observations deliberately not pursued" in
`docs/audit/PBPK28_CN_FLOOR_CLAMP_MASS_INJECTION_DISPATCH_2026-09-26.md` on
branch `audit/pbpk28-cn-floor-clamp-dispatch`.

The first report was narrow: a struct with an **array field**, and the two
forms `z = x` and `*dst = src`. Measurement shows the defect is wider. On
Madaros, **every aggregate copy except `let y = <identifier>` shares storage
with its source.** Scalar-only structs are affected, plain fixed arrays are
affected, and the alias works in both directions. lean_single copies in every
form measured here.

Prior fixes #1475, #1479, #1487 and #1497
(`tests/run-pass/aggregate_nested_field_deep_copy.sio`,
`tests/run-pass/perturb_struct_array_field_no_alias.sio`) cover copy on `let`
initialisation from an identifier. That path is correct. Nothing in
`docs/audit/` records the assignment or store forms.
`docs/audit/REF_ALIAS_BIND_AS_REF_MISS_2026-09-22.md` is adjacent but concerns
the `is_ref` bit, not value copies.

## Repro (as reported)

```sounio
struct S {
    a: [f64; 14],
}
fn bump(s: &!S) with Mut { (*s).a[0] = (*s).a[0] + 1.0 }
fn store(dst: &!S, src: S) with Mut { *dst = src }
fn store_elem(dst: &!S, src: S) with Mut {
    var i: i32 = 0
    while i < 14 {
        (*dst).a[i] = src.a[i]
        i = i + 1
    }
}
fn main() -> i32 with IO, Mut, Div, Panic {
    var x = S { a: [1.0; 14] }
    var z = S { a: [0.0; 14] }
    var w = S { a: [0.0; 14] }
    var e = S { a: [0.0; 14] }
    let y = x
    z = x
    store(&!w, x)
    store_elem(&!e, x)
    bump(&!x)
    println(y.a[0])
    println(z.a[0])
    println(w.a[0])
    println(e.a[0])
    0
}
```

| Compiler | Command | `y` `z` `w` `e` |
|---|---|---|
| `bin/souc-lean-single-x86_64` (committed) | `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile repro.sio -o l.elf` | `1 1 1 1` |
| committed `bin/madaros-linux-x86_64.gz`, unpacked (md5 `57c015c1…`) | `madaros build repro.sio r.elf` under `ulimit -s 524288` | `1 **2** **2** 1` |
| Madaros built from `2e8b76d31` with `make build-madaros` (md5 `5764851f…`) | same | `1 **2** **2** 1` |

The source-built row settles principle 15: the defect is in current source,
not only in a stale shipped ELF. Every probe below gave identical output on the
committed and the source-built Madaros.

Environment for every Madaros row: `SOUC_BIN`, `SOUNIO_SOUC_BIN`,
`MADAROS_RAW_BIN` and `SOUNIO_MADAROS_BIN` unset, `SOUNIO_STDLIB_PATH` pointed at
the worktree's `stdlib/`, and `ulimit -s 524288` before running the raw ELF.

## How wide it is

Each probe has its own program. Values are what the program printed. "Correct"
is lean_single's answer, which is also the value-semantics answer by hand.

| # | Form | lean_single | Madaros (committed and source-built) |
|---|---|---|---|
| 1 | `let y = x`, then write `x` | correct | correct |
| 2 | `z = x` (z already declared), then write `x` through `&!x` | correct | **aliased** |
| 3 | `*dst = src` in a callee, then write `x` | correct | **aliased** |
| 4 | element-wise `(*dst).a[i] = src.a[i]` | correct | correct |
| 5 | scalar-only `struct P { u: f64, v: i64 }`: `z = x`, then `x.u`, `x.v` written | correct | **aliased**, both fields |
| 6 | same, through `*dst = src` | correct | **aliased**, both fields |
| 7 | `z = x`, then write **`z`**; read `x` | correct | **aliased** (the source changes) |
| 8 | `*dst = src`, then write the destination; read the source | correct | **aliased** |
| 9 | `xg = x` inside a `while`, `x.arr[2] += 1` after it (the saved-stage shape) | `xg.arr[2] = 2` | **3** |
| 10 | nested field: `y = x`, then `x.inner.q = 7` | correct | **aliased** |
| 11 | fixed array: `b = a`, then `a[0] = 9` | correct | **aliased** |
| 12 | field store from a local: `o.inner = i1`, `o.arr = c`, then write `i1`, `c` | correct | **aliased** |
| 13 | `let v = *p`, then `(*p).arr[0] = 9` | correct | **aliased** |
| 14 | `v = *p`, then `(*p).arr[1] = 9` | correct | **aliased** |
| 15 | `let li = o.inner`, then `o.inner.q = 9` | correct | **aliased** |
| 16 | `zi = o.inner`, then `o.inner.q = 5` | correct | **aliased** |
| 17 | `za = o.arr` (array field), then `o.arr[2] = 9` | correct | **aliased** |
| 18 | by-value param copied with `var t = s` and written | correct | correct |

Rows 5–8 are why the title's "array" is too narrow. The destination does not
share an array **field** with the source. It shares the whole object. Row 9 is
the PBPK28 shape. In a loop of 100,000 iterations that accumulates `xg.arr[0]`
after `xg = x; x.arr[0] += 1`, Madaros prints `5000150000` and lean_single
`5000050000`: every saved stage is one step ahead.

## Root cause (source)

`Lowerer::lower_let_stmt_after_expr_ref` (`self-hosted/ir/lower.sio`) is the
only place that copies an aggregate. When the initialiser is a bare
`ExprIdent`, it calls `emit_fixed_array_value_copy` for a word-scalar array
local and `emit_struct_value_copy_deep` (#1479) for a value-copyable struct
local. Every other binding shape (rows 13, 15) and every assignment binds or
stores the 64-bit **handle** it was given.

`Lowerer::lower_assign_stmt_ref` lowers the right-hand side with
`lower_opt_expr_ref`. For an identifier that returns the source local's
register, which holds a handle to arena storage. The target arms then store
that register unchanged:

| Target | Emitted | Effect |
|---|---|---|
| local `z` | `ir_copy(dst, rhs)` | z's register now holds x's handle |
| `*dst` (non-Box) | `ir_store_ptr(ptr_reg, rhs)` | `&!w` is the LEA of w's handle slot (see the native-v2 OpRef notes in `lower.sio`), so w's slot now holds x's handle |
| `o.f` | `ir_field_set(base, fidx, rhs, …)` | the field slot holds x's handle |
| `a[i]` | `ir_index_set(base, idx, rhs)` | the element slot holds x's handle |

A struct, a fixed array and a nested aggregate field are all one handle in the
Madaros layout. `count_struct_fields * 8`, with aggregate fields held as
handles, as the #1479 comment in `emit_struct_value_copy_deep` explains. So
each row stores a pointer where the language means a value. lean_single inlines
aggregates and copies `nslots` words (`copy_struct_into_local_slot_a64`), so it
never had the defect.

Rows 13–17 are the same mechanism on the `let` side. `let v = *p` and
`let v = o.f` do not reach the identifier-only copy, and they bind the loaded
handle.

## Prior art and the constraint that shapes the fix

- **E232** (`25a6cf9c8`, #2163) refuses `S { u: z, v: z }`, one array local
  initialising two fields, because "the value copy is shallow". Its commit
  message states that a deep-copying fix was deferred on purpose. The Madaros
  allocator is a bump cursor with no reclamation.
- **Handle table.** `native_v2_handle_table_capacity_default` is **4,194,304**
  (`self-hosted/native/gc.sio:64`). It is never reclaimed, and exhaustion exits
  182 (`madaros: handles full`). The #1479 witness header still quotes
  1,048,576, which is stale. Aggregates of at most 16 B take no handle (#919).
  Measured here on the base Madaros: a loop doing `var xg = x` (the `let` copy
  that is already in tree) with `x` carrying one 24 B array field completes at
  2,000,000 iterations and exits 182 at 8,000,000. That is consistent with one
  handle per copy.
- A concurrent lane (sleepy-easley, 2026-09-26) measured that a >16 B struct
  **construction** costs one handle while reassignment costs none, and that one
  `tsit5_step_pbpk` costs 63 handles. It is working on handle reclamation in
  `self-hosted/native/gc.sio`.

So a correct copy is not free on Madaros today. The design choice below is
about where to pay.

## Proposed fix

**Scope: identifier sources in assignment.** This covers the four reported
forms and rows 2–12. It is one hunk in `lower_assign_stmt_ref` plus one helper
shared with the `let` path.

1. Factor the budgeted struct copy out of the `let` path into
   `emit_struct_value_copy_budgeted` (deep copy when the projected instruction
   cost fits under `IR_MAX_INSTRS`, flat copy otherwise). The `let` path calls
   it unchanged.
2. Add `emit_ident_aggregate_value_copy(name, reg)`, which returns a fresh copy
   of the named local for a word-scalar fixed array or a value-copyable struct,
   and `reg` otherwise. It keeps the `let` path's carve-outs: `[Struct; N]`
   arrays, Box, Knowledge, reference locals and BSS globals keep the handle.
3. In `lower_assign_stmt_ref`, when the right-hand side is an `ExprIdent` (and
   the target is not an f128 slot, which already copies limbs), replace `rhs`
   with that copy before the target arms run. Every target shape then stores a
   fresh handle, and no target arm changes.

**Why a fresh copy, not a copy into the destination's storage.** Writing the
source's contents into the object the destination already holds would cost no
handle in steady state, and it is what lean_single's `memcpy` does. It is
correct only if the destination is the sole owner of that object. Madaros does
not guarantee that today: rows 13–17 and shallow struct literals (E232's
territory) all bind shared storage. After `var v = o.inner; v = x`, an in-place
copy would overwrite `o.inner`, a new miscompile in place of the old one. A
fresh copy only ever rebinds the destination, so it cannot write into another
owner. Its cost is one handle per >16 B aggregate per executed assignment. When
the table runs out, the program exits 182 instead of printing a wrong number.
The in-place variant becomes the better choice once the non-identifier sources
below copy too, or once handles are reclaimed.

The patch lands as its own commit after this evidence, so it can be reviewed
or dropped on its own. The evidence commit adds the witness with a
`//@ known-failure:` line citing this dispatch; the fix commit removes it.

## Witness

`tests/run-pass/madaros_struct_reassign_store_no_alias.sio`
(`//@ requires: madaros`) asserts the value of every observable in rows 1–12,
tagged 1–13 in the file, and prints `STRUCT_REASSIGN_STORE_NO_ALIAS_OK` only if
all hold. Tags 1–4 are the reported shape, with the store done in a callee
exactly as reported.

| Compiler | Output |
|---|---|
| lean_single (committed) | `STRUCT_REASSIGN_STORE_NO_ALIAS_OK` |
| Madaros from `2e8b76d31` (md5 `5764851f…`) | `BAD 2`, `BAD 3`, `BAD 5` ×2, `BAD 6` ×2, `BAD 7` ×2, `BAD 8` ×2, `BAD 9`, `BAD 10`, `BAD 11`, `BAD 12` ×2 |
| Madaros from `2e8b76d31` + fix (md5 `cc0c6e49…`) | `STRUCT_REASSIGN_STORE_NO_ALIAS_OK` |

Tags 1, 4 and 13 (`let y = x`, the element-wise store, and the source's own
value) pass on the unpatched build, as rows 1 and 4 predict.

## Patched build

The patch was applied to `2e8b76d31` and Madaros was rebuilt with
`make build-madaros` (md5 `cc0c6e49…`). The control is the unpatched build of
the same commit (md5 `5764851f…`). Both build logs carry the same two
pre-existing `tuple index out of bounds` lines from
`self-hosted/parser/types.sio`.

**Witness and probes.**

| Probe | Control | Patched | lean_single |
|---|---|---|---|
| reported repro, `y z w e` | `1 2 2 1` | `1 1 1 1` | `1 1 1 1` |
| witness | 15 `BAD` lines | `STRUCT_REASSIGN_STORE_NO_ALIAS_OK` | `STRUCT_REASSIGN_STORE_NO_ALIAS_OK` |
| rows 2–12 | aliased | correct | correct |
| rows 13–17 (non-identifier sources) | aliased | **aliased, unchanged** (out of scope) | correct |
| saved-stage loop, 100,000 iterations | `5000150000` | `5000050000` | `5000050000` |

**Handle cost.** Each program is a loop that copies a struct holding one
24-byte array field and one ≤16 B nested struct (no handle, #919), then writes
the source.

| Form, iterations | Control | Patched |
|---|---|---|
| `var xg = x` (the in-tree `let` copy), 2,000,000 | correct | correct |
| `var xg = x`, 8,000,000 | exit 182, `handles full` | exit 182 |
| `xg = x`, 2,000,000 | wrong sum, exit 0 | correct |
| `xg = x`, 8,000,000 | `32000012000000`, exit 0 (wrong: value semantics give 32,000,004,000,000) | exit 182, `handles full` |

The patched assignment costs exactly what `let` already costs. Where the
table runs out, the patched build exits 182 in place of printing a wrong number
and exiting 0.

**A/B over affected and at-risk tests.** The list has 55 run-pass files: the 15
that `aggregate_copy_alias_sweep.py . tests/run-pass` finds with an aggregate
`a = b` or `*p = b`, every `dissertation_*`, `*pbpk*`, `aggregate_*`,
`perturb_*` and `*alias*` test, and the witness. Each was built with both
compilers and run from its own directory. Stdout plus exit code was compared
byte for byte.

| | Result |
|---|---|
| Identical stdout and exit code | 54 |
| Different | 1: the witness, `BAD`×15 → `STRUCT_REASSIGN_STORE_NO_ALIAS_OK` |
| Compile failure on both (pre-existing type-check preflight) | 6: `connectome_laplacian_eigenvectors`, `g2_bridge_pipeline`, `pbpk28_deriv_zero`, `pbpk28_extract_only`, `pbpk28_m2_hierarchical_prior`, `pbpk28_m5_gum_4th_order` |
| Non-zero exit on both | 1: `dissertation_pbpk14_model_form_uc`, rc 5 |
| ELF changed by the patch | 12, all with identical output except the witness: `bignat_selftest`, `bignat_selftest_divmod_rem`, `dissertation_midazolam_scenario_smoke`, `dissertation_pbpk14_hessian`, `dissertation_pbpk14_model_form_uc`, `gresnigt_family_s3`, `optimization_nelder_mead`, `pbpk28_struct_return`, `sedenion_ratbig_channel_case1`–`3`, and the witness |

The seven `dissertation_pbpk28_parity_*` ELFs are byte-identical under both
compilers, so the patch does not touch them, and it adds no handle pressure to
the 30,000-step rapamycin reference that the #1479 header discusses.
`scripts/dev/run_sio_test_suite.sh --test-list` over the same list, with
`SOUNIO_MADAROS_AVAILABLE=1` and `SOUNIO_TEST_SOUC_BIN` set to each compiler,
agrees: 48 pass / 6 fail on the control, 49 / 5 on the patched build, and the
witness is the only status that differs.

**Not run:** the full 3,286-test suite. The change touches only assignments
whose right-hand side is an aggregate identifier. The list above covers every
run-pass file the sweep finds with that shape, plus the dissertation surface.
A full-suite A/B is the remaining acceptance step before landing.

## Boundary still open after the fix

Rows 13–17, where the source is `*p` or `o.f` rather than an identifier, are
**not** changed by this patch, in either `let` or assignment. Fixing them needs
the static aggregate type of an arbitrary place expression at the copy site,
which the identifier path gets for free from the local table. They are the same
defect and should be the next change. The witness does not assert them, so it
can pass on a fixed identifier path without claiming the rest. Until then, copy
from a pointer or a field element-wise.

## stdlib sweep

**Method.** `python3 docs/audit/repro/aggregate_copy_alias_sweep.py . stdlib`,
a line-oriented sweep over `git ls-files 'stdlib/*.sio'`. It finds
every `a = b` and `*p = b` statement whose two sides are bare identifiers. It
resolves the static type of each side from its declaration in the enclosing
`fn`: an annotation, a struct literal, an array literal, a callee's declared
return type, or another identifier. It keeps rows where that type is a fixed
array or a struct declared somewhere in `stdlib/`. For each row it then looks
for an in-place write to either side: `name.f… =`, `name[…] =`, compound forms,
or `&!name`. The search runs to the end of the function, starting at the copy,
or at the head of the outermost enclosing loop when the copy is inside one,
because a back-edge makes earlier writes later. It also checks for method calls on either name (none found). The
script is heuristic. It misses multi-line statements and interprocedural
mutation, so every row with a write was read by hand.

**Result: 217 whole-aggregate copies.**

| Class | Rows | Wrong on Madaros today? |
|---|---|---|
| Pireus negative-oracle batteries (`bad = canonical`, `bad = good`, `request = base`, then `bad.f = …`) | 154 | **Wrong, but unreachable today** (see below) |
| Inside a loop: 41 with no in-place write to either side anywhere from the loop head to the end of the function, 12 where the only write is to a fresh per-iteration source (read by hand, below) | 53 | No: the alias is never observed |
| Store of a fresh, dead-after local through `*out` (`stdlib/eisa/asm.sio:153`, `stdlib/eisa/backend.sio:312`, `:834`) | 3 | No |
| Handle swaps through a `let` temporary (`stdlib/math/softfloat_f128.sio:376-377`, `softfloat_f256.sio:449-450`) | 4 | No: a swap of handles is a correct swap when neither side is written afterwards |
| Other one-shot copies with no later write (`stdlib/compiler/effects/compile_confidence.sio:170`, `stdlib/hardware/pireus/operator_discovery_engine.sio:1265-1266`) | 3 | No |

**The Pireus batteries.** Each negative oracle is written as
`bad = canonical; bad.<field> = <illegal>; check(&bad)`. After the first
reassignment, `bad` **is** `canonical`, so each case inherits every earlier
case's illegal fields. `canonical` itself is corrupted as a side effect. A
battery that counts "rejected" would then still pass, but vacuously, because
most cases are rejected for an earlier case's reason. A battery that checks for
a specific error code can flip. Sites, all in `stdlib/hardware/pireus/`:

| File | Sites |
|---|---|
| `cubic_operator_forge.sio` (`cof_exercise_negatives`, from L1158) | 34 |
| `quotient_novelty_forge.sio` | 30 |
| `operator_genome.sio` | 24 |
| `operator_morphogenesis.sio` | 23 |
| `operator_novelty_feedback.sio` (`bad = good`, `bad_challenge = challenge`) | 22 |
| `operator_lowering_forge.sio` | 16 |
| `operator_seed_kernel.sio` (`osk_build_negative_cases`, `request = base`, checks exact error codes) | 5 |

None of them runs on Madaros today. All seven `examples/pireus_*.sio` drivers
are refused at `check` by the source-built Madaros. Each has six E232 errors,
inherited through `operator_genome`, and some add E035 or E137. lean_single
builds and runs them, and it is correct: `examples/pireus_cubic_operator_forge.sio`
prints `PIREUS_CUBIC_NEGATIVES passed=35 total=35`. So these sites are
**latent**. They would become silent vacuous passes on Madaros the day E232 is
resolved without this fix. With the fix, they are value-correct.

**Loop rows read by hand**, including all 12 where the sweep saw a write:
`stdlib/darwin_pbpk/epistemic_pbpk14_hessian.sio:133`,
`stdlib/math/bignat.sio:196` and `stdlib/data/bigrat.sio:161`,
`stdlib/hardware/pireus/operator_morphogenesis.sio:705-706` and `:1067-1068`,
`stdlib/optimize/levenberg_marquardt.sio:496-497`,
`stdlib/quantum/epistemic_vqe.sio:458,465`, and
`stdlib/stats/timeseries/arima.sio:130`. In every one, the written name is the
source, which is re-declared (`var st_next = st`, a literal, or `var … = params`,
all of them copies) at the top of the next iteration before any read. The
destination is never written in place before its next reassignment. The
dissertation-path integrators
(`stdlib/darwin_pbpk/tsit5_pbpk28.sio:219`, `epistemic_pbpk28.sio:154`,
`epistemic_pbpk28_hessian.sio:102`, `cumulants.sio:229`,
`validation/pbpk28_*.sio`) are all `st = st_new` with `st_new` a fresh
per-step result. **No stdlib site found by this sweep is silently wrong on
Madaros today.** The PBPK28 AUC shift that started this dispatch came from a
candidate implementation outside `main`, which is exactly the hazard: the first
new `mid = x; step(&!x)` gets a wrong answer.

Not swept, because the request covered only `a = b` and `*p = b`: the `let`-side
rows 13–17 (`let v = *p`, `let v = o.f`) and field or element stores
(`o.f = x`, `a[i] = x`).
