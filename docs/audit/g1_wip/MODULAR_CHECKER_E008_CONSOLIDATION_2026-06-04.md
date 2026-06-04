# Modular checker — consolidated feature stack (2026-06-04)

Consolidation of four stacked levers landed this session on the modular `*mut` checker, on top of
the campaign tip `check/field-deref-ref-e008` (`7c18aae54`). Branch
`integration/modular-checker-e008` (`ffe950ef7`) contains all four.

## Cumulative result (accurate census, `timeout 25`, `</dev/null`, 0 timeouts)

| binary | PASS | CRASH | corpus |
|---|---:|---:|---|
| campaign base `7c18aae54` | 262 | 72 | tests/run-pass (504) |
| **stack tip `ffe950ef7`** | **297** | **65** | tests/run-pass (504) |

**+35 PASS (28 FAIL→PASS + 7 CRASH→PASS), −7 crashers, ZERO PASS→FAIL / FAIL→CRASH / TIMEOUT.**

`self-hosted/compiler/lean_single.sio` is **untouched** across the entire stack → `bin/souc` and the
canonical bootstrap gate are unaffected. All changes are confined to the modular checker/parser:
`check/check.sio` (+345), `check/borrow.sio`, `check/defs.sio`, `parser/ast.sio`, `parser/types.sio`.
Orthogonal to the concurrent i128 (`check/types.sio`, `compat.sio`, native) and SRET (`ir/*`) lanes.

## The four levers (suggested merge order = stack order)

1. **#235 — closures / HOF** (`check/closure-hof-triple-e008`, +7): structural fn-type compatibility
   guard in the inplace mismatch reporter + re-landed fn-type lowering + `FnSigTable.get` bounds-guard.
2. **#237 — effect-polymorphic HOFs + parser** (`parser/fn-type-effects-list-e008`, +14): closure-literal
   SRET dodge (`*mut`-transcribed closure helpers — any `|x|` literal SIGSEGV'd) + the effect-polymorphic
   HOF model (closure-arg effects propagate to the call site) + effect-list parser fix. **Supersedes
   #235's effect-subsumption model** (see that PR's doc banner).
3. **#242 — linear E039** (`check/linear-double-consume-e039`, +7): removed redundant double-consume in
   call-args + branch-isolated linear borrows in `if`-expr (128-byte consumed-flag merge, dodging the
   large-`BorrowEnv`-copy miscompile).
4. **#243 — refinement types** (`check/refinement-types-e008`, +7): `*mut` `TypeRefinement` lowering arm
   + `!=` op + definite-violation→error completion + compound predicates (`&&`/`||`).

## Soundness (verified, not just census)

Every error-suppressing/lowering change was checked against the negative suite (`tests/compile-fail`):
- Effect-polymorphism: IO/GPU/Async functions passed where pure is required REJECT; `effects_closure_escape` rejects.
- Linear: `linear_double_use`/`call_double_use`/`not_consumed`/`implicit_drop` reject.
- Refinement: all compile-fail refinement violations reject, incl. `refinement_compound_violation`.

## Measurement methodology (load-bearing)

`rc` is **deterministic per-binary** (verified 5× repeat). The full **census is authoritative**;
per-test loops produced spurious `rc=0` under load and must not be trusted over the census. Census with
`timeout 25` + `</dev/null` + treat `rc=124` as TIMEOUT (not FAIL); never `pkill -f <binary>` from a
command that references the binary (self-kill). The documented campaign "322/0" baseline does not
reproduce with the committed `bin/souc` (a `c634b38f` `mini_native`); deltas are measured against
same-session baselines, which is sound.

## Known follow-ups (out of scope, documented)

The clean single-mechanism well is dry; remaining levers expose incomplete downstream error-detection
(a structural property of the maturing checker). Tracked: `let _ =` underscore binding + completing
violation-detection at all positions (would flip `guard_and`/`guard_or` etc. but regresses 3 green-by-
accident negative tests until the downstream checks are completed); `type` alias keyword; arithmetic-RHS
refinement predicates; units (dimensional arithmetic + Kelvin/generic-param collision); generic methods;
async (parser + checker semantics).
