<!-- docs:meta
topic_id: repo.docs.audit.madaros-let-field-source-alias-dispatch-2026-09-26
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-let-field-source-alias-dispatch-2026-09-26
-->

# Madaros `let v = <place>` binds the aggregate handle — dispatch

**Date:** 2026-09-26
**Base:** `origin/main` @ `2e8b76d31` (workspace worktree `/workspace/worktrees/claude-let-field-alias`, branch `claude/madaros-let-field-alias`)
**Engines:** Madaros, cross-checked against `SOUNIO_SOUC_ENGINE=lean_single`
**Owner:** unassigned (`self-hosted/ir/lower.sio`, `Lowerer::lower_let_stmt_after_expr_ref`)
**Status:** evidence recorded. The source-built baseline **reproduces**. A fix is proposed and measured on a scratch build ([Patched build](#patched-build)). It is **not** applied to `self-hosted/` in this change. Two stdlib functions give wrong answers on Madaros today ([stdlib sweep](#stdlib-sweep)).

## Why this dispatch

Found on 2026-09-26 while auditing `stdlib/darwin_pbpk/scenarios/venlafaxine_xr.sio`
(commit `bba219013` on `claude/elegant-borg-a14bdf`, which works around it with
`vfx_state_copy`). A steady-state periodicity check took a snapshot with
`let a_start = s.a`, advanced `s`, and compared the two. On Madaros the snapshot
moved with `s`. As that lane recorded it (not re-measured here), the error term
was silently zero, a certified-interval check then failed on Madaros only, and
every AUC in the same run matched lean_single to 12 digits, so the output looked
healthy.

This is rows 13–17 of
[`MADAROS_STRUCT_REASSIGN_ARRAY_ALIAS_DISPATCH_2026-09-26.md`](MADAROS_STRUCT_REASSIGN_ARRAY_ALIAS_DISPATCH_2026-09-26.md)
(Sounio-lang/sounio#2700, branch `claude/madaros-struct-reassign-alias`, open and not on `main` at the time of writing).
That dispatch fixes the **assignment** forms (`z = x`, `*p = x`, `o.f = x`) for
identifier sources, and it leaves the **binding** forms with a place source
open. This dispatch covers those binding forms. The two fixes touch different
arms of `lower.sio` and compose; see [Relation to the assignment dispatch](#relation-to-the-assignment-dispatch).

## Repro

As reported (lean_single prints `REPRO_OK`, Madaros prints `REPRO_ALIASED`):

```sounio
struct In { v: [f64; 4] }
struct Out { a: In, b: In, k: f64 }
fn bump(o: Out) -> Out with Mut {
    var a = o.a
    a.v[0] = a.v[0] + 1.0
    Out { a: a, b: o.b, k: o.k }
}
fn main() -> i32 with IO, Mut {
    var s = Out { a: In { v: [1.0; 4] }, b: In { v: [2.0; 4] }, k: 0.0 }
    let a_start = s.a
    s = bump(s)
    if a_start.v[0] == 1.0 { println("REPRO_OK") } else { println("REPRO_ALIASED") }
    return 0
}
```

The report has **two** place bindings in it, not one: `let a_start = s.a` in
`main` and `var a = o.a` in `bump`. `s = bump(s)` only rebinds `s`. It does not
write the old object. The write that `a_start` observes is `a.v[0] = …` inside
`bump`, which lands in the caller's `s.a` because `var a = o.a` did not copy.
Either binding copying would have been enough to hide the defect here.

Reduced (8 lines; lean_single `1 9`, Madaros `9 9`):

```sounio
struct In { v: [f64; 4] }
struct Out { a: In, k: f64 }
fn main() -> i32 with IO, Mut {
    var s = Out { a: In { v: [1.0; 4] }, k: 0.0 }
    let a0 = s.a
    s.a.v[0] = 9.0
    println(a0.v[0])
    println(s.a.v[0])
    return 0
}
```

## Measurements

Each row is its own program. The value columns are what the program printed.
"Correct" is lean_single's output, and it is also the value-semantics answer by
hand.

| Compiler | Identity |
|---|---|
| lean_single | `bin/souc-lean-single-x86_64`, md5 `0cb08380…`, via `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile` |
| Madaros, committed | `bin/madaros-linux-x86_64.gz` unpacked, md5 `57c015c1…` |
| Madaros, from source | `make build-madaros` on `2e8b76d31`, md5 `5764851f…` (the same md5 the assignment dispatch records for its control) |

Environment for every Madaros row: `SOUC_BIN`, `SOUNIO_SOUC_BIN`,
`MADAROS_RAW_BIN` and `SOUNIO_MADAROS_BIN` unset, `SOUNIO_STDLIB_PATH` set to
the worktree's `stdlib/`, and `ulimit -s 524288` before `madaros build` and
before running the ELF. The committed and source-built Madaros gave identical
output on every probe run on both (the reported repro, the witness, and the two
stdlib drivers below). So the defect is in current source (principle 15), not
only in a stale shipped ELF.

| # | Form | lean_single | Madaros from source |
|---|---|---|---|
| 1 | reported repro | `REPRO_OK` | **`REPRO_ALIASED`** |
| 2 | `let a0 = s.a` (struct field), then `s.a.v[0] = 9` | `1 9` | **`9 9`** |
| 3 | `let r = bump(s)` with `s` an immutable `let`; read `s.a.v[0]` | `1` | **`2`**: the callee changed the caller's immutable argument |
| 4 | `let a0 = s.arr` (`[f64; 4]` field), then `s.arr[0] = 9` | `1` | **`4621256167635550208`** (aliased, and printed as raw bits: see 12) |
| 5 | `let snap = *p` (`p: &!In`), then `(*p).v[0] = 9` | `1` | **`9`** |
| 6 | `let deep = o.l1.l2`, `let mid = o.l1`, then write `o.l1.l2.v[0]` and `o.l1.k` | `1 1 0` | **`9 9 5`** |
| 7 | scalar-only field structs, 24 B (`P3`) and 16 B (`P2`), then write `h.p3.u`, `h.p2.u` | `1 1` | **`9 9`** |
| 8 | `var a = s.a`, then write `a.v[0]`; read `s.a.v[0]` | `1` | **`9`**: the reverse direction writes the owner |
| 9 | `let e = xs[0]`, `xs` an array of structs, then `xs[0].v[0] = 9` | `1` | **`9`** |
| 10 | `let a0 = get_a(&s)` where `fn get_a(o: &Out) -> In { (*o).a }` | `1` | **`9`** |
| 11 | controls: `let whole = s` (identifier), and an element-wise copy helper `in_copy(s.a)` | `1`, `1` | `1`, `1` |
| 12 | `let a0 = s.arr` with no later write; `println(a0[0])`, `a0[0] + 1.0`, `a0[0] == 1.5` | `1.5 2.5 EQ_OK` | **`4609434218613702656`** `2.5 EQ_OK` |

Rows 2–9 are the forms this dispatch proposes to fix. Row 7 shows the struct does
not need an array field, and a 16-byte struct is not exempt: a struct-typed
**field** is a handle in the Madaros layout whatever its size. Row 3 is the
worst case: a function that only reads its by-value parameter, by the language's
rules, writes the caller's value. Row 12 is a second, smaller defect on the same
line. The new binding never gets the float-element mark, so `println` prints the
IEEE bits; arithmetic and comparison are unaffected.

### A separate engine divergence: E035

The reported `bump` omits `with Mut` in its original form. lean_single refuses
it with `error[E035]: effect not declared in function signature`; Madaros
accepts it and runs it (`2`). The same divergence makes lean_single refuse
`stdlib/epistemic/proptest.sio` and `examples/cybernetic_demo.sio` outright
while Madaros compiles them. That is an effect-checking divergence, not part of
this defect, and it is recorded here only because it decides which engine can
run the stdlib witnesses below.

## Root cause (source)

`Lowerer::lower_let_stmt_after_expr_ref` (`self-hosted/ir/lower.sio:16960`)
lowers the initialiser first. For a place (`s.a`, `*p`, `xs[i]`) the result is
the 64-bit value stored at that place. For an aggregate that value is a
**handle**: Madaros lays a struct out as `count_struct_fields * 8`, with every
aggregate field held as a handle (the #1479 comment on
`emit_struct_value_copy_deep`, `lower.sio:8146`).

The only copy in the function is guarded on the initialiser's kind
(`lower.sio:17101–17166`):

```sounio
match s.expr {
    Some(rhs_ident) => {
        if (*rhs_ident).kind == ExprKind::ExprIdent {
            // fixed array  -> emit_fixed_array_value_copy   (#1475)
            // struct       -> emit_struct_value_copy_deep   (#1479)
            // scalar       -> fresh register
        }
        // every other kind: bind_reg = reg, the handle itself
    }
    _ => {}
}
lo = lo.bind_local(s.name, bind_reg, is_mut)
```

`ExprFieldAccess`, `ExprUnary(OpDeref)` and `ExprIndex` fall through, so
`bind_local` binds the owner's handle. That is the whole defect for rows 2–9.

The type information the copy needs is not missing. The same function, a few
calls later in `lower_let_stmt_type_metadata_ref`, already resolves the field's
struct type for `let c = x.field` with `expr_receiver_struct_type_ref` and
`struct_layout_field_named_type` (the "`let c = __r.checker`" arm, added so
method calls on the new local resolve). It binds that type to the new local.
It is simply never consulted for the copy, which runs first and asks only
whether the initialiser is an identifier.

Row 12 has the same origin. For an identifier source,
`rhs_array_elem_is_float` is taken from the source local's record. For a field
source nothing sets it, so the new local is never marked float.

Row 10 is a different path. The copy would have to happen at the `return` of a
place (`(*o).a`) or at the call site. It is not a `let` of a place. It is
recorded as a boundary below.

lean_single has none of this because it inlines aggregates and copies
`nslots` words on binding.

## Stdlib impact found

Two stdlib functions give wrong answers on Madaros today. Both were found by
the [sweep](#stdlib-sweep) and then measured.

**`stdlib/epistemic/proptest.sio`, `ILLEGAL_narrow_uncertainty`.** It builds the
narrowed result from `var new_uncert = ev.uncert` and writes
`new_uncert.std_u` / `interval_lo` / `interval_hi`. On Madaros those writes land
in the caller's `ev.uncert`. The negative oracle in
`run_property_illegal_detected` then evaluates
`law_uncertainty_non_contraction(ev, narrowed)` on two values that share one
uncertainty. The widths are equal, the law holds, and the battery records that
the illegal narrowing was **not detected**. `transform_scale` has the same
shape, but its laws never read `uncert`, so its battery is unaffected.

Driver: the full `stdlib/epistemic/proptest.sio` source followed by a `main`
that prints `run_property_scale_preserves_laws(100, 12345)` and
`run_property_illegal_detected(100, 14345)`. (`examples/epistemic/proptest_demo.sio`
is the in-tree caller, but it imports nothing and does not build on its own.)

| Compiler | scale laws passed / failed | illegal detected passed / failed |
|---|---|---|
| lean_single | refused, E035 (see above) | refused |
| Madaros, committed and from source | 100 / 0 | **36 / 64** |
| Madaros from source + proposed fix | 100 / 0 | 100 / 0 |

With lean_single unable to build the file, no existing run shows the correct
answer. The fixed build's 100 / 0 is the value-semantics answer: every generated
value with a non-zero width has its width halved.

**`stdlib/cybernetic/autopoiesis.sio`, `produce_cycle`.** It is written as a
Jacobi update: `var new_values: [f64; 8] = s.component_values`, then every new
value is computed from `s.component_values`. On Madaros `new_values` **is**
`s.component_values`, even with the `[f64; 8]` annotation, because the copy
arm tests the initialiser's kind, not the annotation. Each write is therefore
visible to later reads in the same cycle, and the update becomes Gauss–Seidel.
The variances (`new_vars`) have the same defect.

Probe: a chain 0 → 1 → 2 with values 1, 3, 5, then one `produce_cycle`.

| Compiler | `component_values[1]` | `component_values[2]` |
|---|---|---|
| lean_single | `2.0` | `4.0` (= (3 + 5) / 2, Jacobi) |
| Madaros, committed and from source | `2.0` | **`3.5`** (= (2 + 5) / 2) |
| Madaros from source + proposed fix | `2.0` | `4.0` |

`tests/run-pass/multi_agent.sio` calls `produce_cycle` and gives byte-identical
output on lean_single and Madaros. Its ring topology never reads a component
after it was written in the same cycle, so it does not exercise the difference.
`test_autopoiesis_stdlib.sio` passes on both for the same reason.

## Proposed fix

Scope: `let` and `var` bindings whose initialiser is a place of aggregate type.
One arm in `lower_let_stmt_after_expr_ref` and three small helpers. The full
diff is [`repro/madaros_let_place_source_copy.patch`](repro/madaros_let_place_source_copy.patch)
(132 added lines, applies to `2e8b76d31`).

1. `struct_layout_field_pos_by_name(type, field)`: the by-name sibling of
   `struct_layout_field_pos`, same first-registration rule.
2. `let_place_source_word_scalar_array_pos_ref(e)`: for `o.f` where `f` is a
   word-scalar fixed-array field, the field's layout position, else -1.
3. `let_place_source_struct_type_ref(e)`: the static struct type of `o.f`,
   `*p` or `xs[i]`, taken from `expr_receiver_struct_type_ref` and
   `field_named_type_for_struct`. It returns empty, meaning keep the handle, for
   a tuple slot, a Box field, a Box deref, a deref of anything but an
   identifier, an index into anything but an identifier, and a `Seq` element.
4. In the `Some(rhs)` arm, when the initialiser is **not** an `ExprIdent`: if (2)
   finds an array field, copy with `emit_fixed_array_value_copy` and record the
   fixed-array length, word-scalar bit and float-element bit (this also fixes
   row 12). Otherwise, if `nested_struct_field_slot_count` of (3) is positive,
   copy with the same budgeted deep copy the identifier arm uses. That function
   already refuses Box, Knowledge and unregistered names. A zero handle, from a
   field a partial struct literal omitted, is bound unchanged, not dereferenced.

Why a fresh copy, not a shared one with copy-on-write or an escape analysis:
it is the #1475/#1479 identifier arm applied to one more source shape, so it
inherits their reviewed carve-outs and budget. Its cost is the one the
identifier arm already pays (below).

## Patched build

The patch was applied to `2e8b76d31` in the worktree above and Madaros was
rebuilt with `make build-madaros` (md5 `60322816…`). The build log carries the
same two pre-existing `tuple index out of bounds` errors from
`self-hosted/parser/types.sio` that the assignment dispatch records for its
builds.

| Probe | Control (`5764851f`) | Patched (`60322816`) | lean_single |
|---|---|---|---|
| rows 1–8 | aliased | **correct** | correct |
| row 9, `var xs: [In; 2] = [...]` (annotated) | aliased | **correct** | correct |
| row 9, `var xs = [In {..}, In {..}]` (unannotated) | aliased | aliased, unchanged | correct |
| row 10, return of a place | aliased | aliased, unchanged | correct |
| row 12, `println` of an element | raw bits | `1.500000` | `1.500000` |
| controls (row 11) | correct | correct | correct |
| witness | 12 `BAD` lines | `LET_FIELD_SOURCE_NO_ALIAS_OK` | `LET_FIELD_SOURCE_NO_ALIAS_OK` |
| proptest illegal battery | 36 / 64 | 100 / 0 | refused (E035) |
| `produce_cycle` probe | `3.5` | `4.0` | `4.0` |

**Handle cost.** A loop that does `let a = s.a` (one `[f64; 4]` field, so one
32-byte array per copy) and then writes `s.a.v[0]`:

| Iterations | Control | Patched | lean_single |
|---|---|---|---|
| 2,000,000 | `2000003000000`, exit 0 (**wrong**) | `2000001000000`, exit 0 | `2000001000000` |
| 8,000,000 | `32000012000000`, exit 0 (**wrong**) | `madaros: handles full`, exit **182** | `32000004000000` |

So the patched binding costs one handle per executed copy of an aggregate over
16 B, the same as `var y = x` already costs (the assignment dispatch measures
that form exiting 182 at the same 8,000,000). The handle table holds 4,194,304
entries (`native_v2_handle_table_capacity_default`, `self-hosted/native/gc.sio`)
and is never reclaimed. Where the table runs out, the patched build stops with
exit 182 instead of printing a wrong number and exiting 0. Handle reclamation
is being worked on separately (sleepy-easley lane, `self-hosted/native/gc.sio`).

**A/B over the at-risk surface.** 132 files: every `tests/run-pass` file whose
name matches `dissertation_`, `pbpk`, `aggregate_`, `alias`, `perturb_`,
`deep_copy`, `struct_`, `cybernetic`, `multi_agent`, `vqe`, `causal`,
`venlafaxine` or `sema_`, plus every file under `tests/run-pass/` and
`tests/stdlib/` that imports `darwin_pbpk`, `cybernetic`,
`quantum::epistemic_vqe`, `causal` or `epistemic::proptest`. Each was built with
both compilers and run from its own directory. Stdout and exit code were
compared byte for byte.

| | Result |
|---|---|
| Built and ran on both, identical stdout and exit code | **114 of 114** |
| Different | 0 |
| Build failure on both (pre-existing, same set on both) | 18 |
| Non-zero exit on both, identical output | 10: eight `darwin_pbpk` tests at rc 182 (`handles full`, the handle-table lane's acceptance list), `dissertation_pbpk14_model_form_uc` rc 5, `global_struct_bss_full_size` rc 139 |
| ELF changed by the patch, output identical | 9: `dissertation_pbpk28_parity_ref_{haloperidol,midazolam,venlafaxine}`, `f128_v0e55_language_struct_fields`, `madaros_wide_struct_variance_field`, `multi_agent`, `test_autopoiesis_stdlib`, `darwin_pbpk/test_epistemic_pbpk`, `darwin_pbpk/test_pipeline_real_e2e` |

The build failures include `darwin_sema_sc_depot_smoke` and
`tests/stdlib/quantum/test_epistemic_vqe.sio`, so two of the sweep's benign
sites are not compiled by Madaros today at all. None of the eight rc 182 tests
moved: they already exhaust the table on the control.

**Not run:** the full suite. The change is confined to `let`/`var` bindings
whose initialiser is an aggregate-typed field, deref or index place. A full
A/B is the remaining acceptance step before landing.

## Witness

`tests/run-pass/madaros_let_field_source_no_alias.sio` (`//@ requires: madaros`)
asserts rows 1–9 as tags 1–9, each against a value the program itself wrote, and
prints `LET_FIELD_SOURCE_NO_ALIAS_OK` only if all hold. Tag 9 uses the annotated
array, which is what the proposed fix covers. It carries
`//@ known-failure:` citing this dispatch. The fix commit should remove that
line.

| Compiler | Output |
|---|---|
| lean_single | `LET_FIELD_SOURCE_NO_ALIAS_OK` |
| Madaros, committed (`57c015c1`) and from source (`5764851f`) | `BAD 1`, `BAD 2`, `BAD 3`, `BAD 4`, `BAD 5`, `BAD 6` ×3, `BAD 7` ×2, `BAD 8`, `BAD 9`, `…_FAIL` |
| Madaros from source + proposed fix (`60322816`) | `LET_FIELD_SOURCE_NO_ALIAS_OK` |

`//@ requires: madaros` tests run only with `SOUNIO_MADAROS_AVAILABLE=1`. A
harness run that reports `Skip` for this file checked nothing.

## Boundary still open after the fix

- **Unannotated array of structs** (row 9 unannotated). The element type of
  `var xs = [In {..}, In {..}]` is not recorded on the local; only the
  `[T; N]` annotation records it (the "An ARRAY local records its ELEMENT
  type" arm in `lower_let_stmt_type_metadata_ref`). With no type, the
  fix keeps the handle. Recording the element type of an array literal would
  close it.
- **Returning a place** (row 10). `fn get_a(o: &Out) -> In { (*o).a }` returns
  the owner's handle. The copy belongs at the return, and it is not a `let`.
  A concurrent lane (determined-leavitt, 2026-09-26) reports it is fixing
  call results that return a by-value parameter or a place, callee-side at
  the return sites of `lower.sio`. Its hunks do not overlap this patch.
- **Not measured:** `let v = *p` for `p: &[T; N]` (the deref resolves no struct
  type, so the fix keeps the handle), tuple slots of aggregate type, and
  aggregate places reached through a method call.

## Relation to the assignment dispatch

The assignment dispatch changes `lower_assign_stmt_ref` and factors the
identifier arm's budgeted struct copy into `emit_struct_value_copy_budgeted`.
This patch touches only the `let` path. It was written against `main`, so it
repeats the budget check inline. Rebased onto the assignment fix, step 4 should
call `emit_struct_value_copy_budgeted` rather than repeat it. The two witnesses
are disjoint: theirs asserts assignment forms, this one asserts binding forms.

Assignment **from** a place (`z = o.f`, rows 16–17 there) is not fixed by
either patch. The assignment fix copies only identifier sources. The helpers in
this patch answer the static type of a place, so the same arm there could reuse
them.

## stdlib sweep

**Method.** `python3 docs/audit/repro/let_place_source_alias_sweep.py . stdlib`,
over `git ls-files 'stdlib/*.sio'` at `2e8b76d31`. It reads every struct
declaration in `stdlib/`, `self-hosted/`, `examples/` and `tests/`, and every
declared return type. For each `fn` it resolves the static type of each
`let`/`var` whose initialiser is a place (`a.f…`, `(*p).f…`, `*p`, `xs[i]`),
from parameter and `let` annotations, struct literals, callee return types and
earlier place bindings. It keeps bindings whose type is a fixed array or a
struct. It then classifies each by what could observe the alias:

- **A**: the source place is written in place (`a.f.x = …`, `a.f[i] = …`,
  `&!a`, `&!a.f`) while the new binding is still read afterwards. This is the
  reported snapshot shape.
- **B**: the new binding is written in place (`v.x = …`, `v[i] = …`, `&!v`),
  which writes the owner. **B-param** when the owner is a by-value parameter,
  so the write reaches the caller.

A write that rebinds (`a = f(a)`, `a.f = x`) does not touch the shared storage
and is not flagged. Inside a loop, the search starts at the loop head, because a
back-edge makes earlier writes later.

The sweep is heuristic and line-oriented. Of 1,310 place bindings in `stdlib/`,
it resolves 1,017 to a scalar type and 79 to an aggregate. **228 (17 %) stay
unresolved**, mostly because the root's type comes from a construct the sweep
does not model (method returns, generic structs, multi-line statements). It
does not follow mutation into a callee except through the B-param class. Every
flagged row was read by hand, together with its callers.

**Result: 79 aggregate place bindings; 21 flagged, all class B; no class A.**

| Class | Rows | Wrong on Madaros today? |
|---|---|---|
| A (snapshot, then source written in place) | 0 | — |
| B-param, and a caller reads the argument after the call | 2 (`stdlib/epistemic/proptest.sio:218`, `:304`) | **Yes** for `:304`, measured above. `:218` is observable, but its battery reads no field that changes |
| B within one function, the owner read after the write | 2 (`stdlib/cybernetic/autopoiesis.sio:68`, `:69`) | **Yes**, measured above |
| B-param, every caller rebinds (`x = f(x, …)`), so the old value is dead | 6 (`stdlib/causal/do_calculus.sio:159`; `stdlib/compiler/transform/partial_eval.sio:181`; `stdlib/darwin_pbpk/scenarios/semaglutide_sc_depot.sio:99`, `:104`, `:109`; `stdlib/quantum/epistemic_vqe.sio:509`) | No |
| B-param with no caller in the repository | 5 (`stdlib/compiler/bootstrap/verify.sio:266`, `compiler/effects/compile.sio:93`, `compiler/effects/compile_confidence.sio:281`, `compiler/handlers/selfhost.sio:188`, `compiler/transform/partial_eval.sio:196`) | No, until someone calls them |
| B on a function-local owner that is dead after the write (`var inv = run.parser.inventory; inv.x = …; inv` with `run` a local; `var s = evt.context` with `evt` returned only on the path that does not write) | 6 (`stdlib/compiler/handlers/selfhost.sio:306`; `stdlib/cybernetic/bateson.sio:195`; `stdlib/hardware/pireus/aarchmrs_import.sio:831`, `apple_a64_tbl_lowering.sio:963`, `apple_metal_import.sio:1164`, `ptx_import.sio:927`) | No |

The dissertation-path rows are the three in `semaglutide_sc_depot.sio`'s
`sema_strang_step` (`var s_brain = scen.tmdd_brain; s_brain.ct_mass = …`).
They write into the input scenario's TMDD states. Both callers
(`sema_integrate_to` and `tests/run-pass/darwin_sema_sc_depot_smoke.sio`)
rebind `s = sema_strang_step(s, …)`, and `sema_integrate_to` starts from
`var s = scen`, a deep copy. So the caller's scenario is never written and the
result is unaffected. A new caller that keeps the input scenario, for example
to compare two step sizes from one state, would see it change.

`epistemic_vqe.sio:509` also has the Jacobi shape of `produce_cycle`, but each
index pair `(i, i | stride)` is read before it is written and visited once, so
the in-place update computes the same numbers.

**Handle exposure of the fix.** 29 of the 79 bindings sit inside a loop, where
the fix would add one handle per iteration for an aggregate over 16 B (the sweep
prints them under `LOOP`). The dissertation-path ones are
`stdlib/darwin_pbpk/simulation_real.sio:96` (`let st_next = step.state_new`,
one per integration step) and
`stdlib/darwin_pbpk/scenarios/steady_state_runner.sio:246`
(`let tr = res.trace`, one per dosing interval). Neither is near the table
size on its own. Both add to the per-step handle budget that the handle-table
work measures (63 handles per `tsit5_step_pbpk`).

**Not swept:** `tests/` and `examples/`, assignment from a place (`z = o.f`),
and bindings the sweep could not type (above).
