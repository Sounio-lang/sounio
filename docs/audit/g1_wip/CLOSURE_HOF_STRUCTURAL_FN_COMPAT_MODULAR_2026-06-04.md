# Closure / HOF unlock — structural fn-type compatibility (modular *mut checker)

> ⚠️ **SUPERSEDED (effect model only), 2026-06-04.** The "Effect subtyping" section below
> describes effect-SUBSUMPTION (a value's effects must be ⊆ the param's) and presents the
> `eff1`/GPU/Async *rejections* as verified-sound. That model was **wrong** — it only ever
> "passed" because closure literals crashed before reaching the check. Branch
> `parser/fn-type-effects-list-e008` replaced it with **effect-POLYMORPHISM**: bare `fn(T)->U`
> params accept any-effect value by signature, and a fn/closure argument's effects propagate to
> the HOF **call site** (the enclosing function must declare them). Under the correct model the
> `eff1`/GPU/Async cases PASS (their caller declares the effect); rejection happens only when the
> *caller* lacks it. The structural-compatibility mechanism (sig-id → structural) below remains
> correct; only the effect-comparison part is superseded.

**Date:** 2026-06-04
**Branch:** `check/closure-hof-triple-e008` (off `check/field-deref-ref-e008` @ `7c18aae54`)
**Commits (atomic, stacked):**
- `cac7f6bf6` structural fn-type compatibility in the inplace mismatch reporter (Block B)
- `c3fc1fdbf` lower `fn(T)->U` types in the *mut checker (Block A, re-land of campaign fix #9)
- `944bd4aff` bounds-guard `FnSigTable.get` against overflow OOB

**Scope:** modular checker only (`self-hosted/check/check.sio`, `self-hosted/check/defs.sio`).
`lean_single.sio` UNTOUCHED → `bin/souc` unchanged, canonical gate unaffected.

## Result (measured, same-session baseline)

| binary | PASS | CRASH(rc=139) | corpus |
|---|---:|---:|---|
| baseline `7c18aae54` (`/tmp/mc_base.elf`) | 262 | 72 | tests/run-pass (504) |
| **sound unit (A + B-with-effects + ambient + boundscheck)** (`/tmp/mc_unit4.elf`) | **269** | 72 | tests/run-pass (504) |

**+7 PASS, 0 PASS→FAIL, 0 FAIL→CRASH, crash set unchanged (72, identical members).**

The 7 FAIL→PASS transitions (first-class-function / closure / HOF programs):
- `closure_fn_ref.sio` — named functions as first-class values
- `closure_higher_order.sio`
- `closure_lambda_lift.sio`
- `closure_sort_by.sio`
- `hof_mut_struct_min.sio`
- `ode_generic_solver.sio` — generic ODE solver taking a `fn` RHS (scientific)
- `root_finding.sio` — root-finder taking a `fn` (scientific)

### Effect subtyping — the iteration that made it sound

A FIRST build of this guard ignored effects entirely and measured +7, but it was UNSOUND: it
accepted an effectful `fn(i64)->i64 with IO` where a pure `fn(i64)->i64` was required. The
authoritative semantics (`tests/compile-fail/effects_closure_escape.sio`: "pure HOF — f must be
pure") rejects that. The fix landed in two commits:
- `119836e2d` directional effect subsumption `got.effects ⊆ expected.effects` + linearity equality
  → restores soundness, but rejected 3 run-pass HOFs whose passed functions over-declare pervasive
  effects (`fn f_quadratic(x: f64) -> f64 with Mut, Panic, Div`) against a bare-pure param.
- ambient set: `{Mut=1, Alloc=2, Panic=3, Div=4}` are skipped in fn-type subtyping (Mut is
  signalled at the type level by `&!` refs; Panic/Div/Alloc are total-program effects), so only
  OBSERVABLE effects (`IO=0, GPU=5, Async=6, Prob=7, Observe=13, …`) gate. Recovers the 3 HOFs.

**Observable-effect soundness verified** (base rejects → must stay rejected): `with IO`, `with GPU`,
and `with Async` functions passed where a pure fn-type is required all REJECT, as does
`effects_closure_escape`. Caveat recorded: Sounio has mutable globals (top-level `var`), so treating
`Mut` as ambient has a narrow theoretical gap (a fn truly mutating a global, passed as pure); the
run-pass corpus over-declares `Mut` spuriously (pure math), so no corpus program exercises it. This
ambient cut was an explicit operator decision.

## Diagnosis — why it was a multi-block, and the necessary+sufficient trace

A named function used as a first-class value carries **its own** FnSig id; a `fn(T)->U`
annotation registers a **separate** anonymous FnSig. So the two `ty_fn` types always have
different `sig_id`. There were TWO independent blocks, surfaced by minimal probes:

1. **Block B (visible, E007/E008/E009):** `types_compatible` (compat.sio:127) compares `ty_fn`
   by `a.fn_sig_id == b.fn_sig_id` → spurious mismatch. Surfaced as `error[E007] = expected fn#N
   / found fn#M`. The mismatch flows through one inplace reporter, `checker_report_mismatch_inplace`
   — guarding its top (where `(*c).fn_sigs` is in scope) rescues all report codes (7/8/9/1) at once.
2. **Block A (silent):** the *mut `checker_lower_type_expr_mut` had `TypeFn` fall to
   `_ => checker_note_type_error_mut` (a SILENT had_error, no print) in the bind/check pass. The
   collect pass lowers fn-types (producing `fn#N`, hence the E007), but the bind/check pass
   re-lowering hit the silent setter. This is why Block B alone was **+0** (silent error remained)
   and Block A alone was **+0** (E007 remained) — they are jointly necessary.

Minimal repros (in `/tmp/probes/`):
- `p1` named fn → fn-type param: baseline rc=1 silent → unit rc=0
- `p2` `let f = square; f(7)` (no annotation): rc=0 both (already worked — isolates the annotation as the trigger)
- `p3` named fn returned as fn-type: baseline rc=1 silent → unit rc=0

3. **Bounds-check (regression guard for Block A):** Block A registers a new FnSig per fn-type
   annotation; `FnSigTable` capacity is 64 and `add()` silently no-ops on overflow, returning an
   out-of-range idx. `get(idx)` then read `entries[idx]` OOB → SIGSEGV on >64-sig programs
   (quadrature-class). `get()` now returns `empty_fn_sig()` on out-of-range idx → graceful FAIL,
   not crash. (Enlarging the table was rejected: `FnSigTable` lives inside the ~164 KB `Checker`
   that is copied per-expr in by-value paths; growing it risks corpus-wide layout crashes.)

## Soundness (the guard suppresses errors — verified it still rejects)

- `neg` (`fn(i64)->i64` passed where `fn(f64)->f64` expected): **rejected** (param types recurse
  through `types_compatible`; i64 vs f64 incompatible).
- `neg2` (arity 2 vs 1): **rejected** (`param_count` mismatch).
- Negative suite `tests/compile-fail` (250 files): **zero** files that the baseline rejected are
  newly accepted by the unit. (see `/tmp/cf_base.txt` vs `/tmp/cf_unit.txt`)

## Measurement caveat (recorded in memory)

The documented campaign baseline "PASS 322 / CRASH 0" does NOT reproduce: a fresh build of the
exact tip `7c18aae54` with the committed `bin/souc` (a `c634b38f`-family `mini_native`, known
span-sensitive) yields 262/72. The 72 rc=139 crashes are bootstrap-compiler miscompile artifacts,
layout-sensitive across build instances but deterministic within a binary. **The +7 result is
measured against a same-session baseline via per-file non-crash transitions** (FAIL rc=1 → PASS),
which is the only reliable signal; raw PASS-count is not stable across build instances.
