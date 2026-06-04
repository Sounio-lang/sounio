# Effect-polymorphic closures + inplace effect inference (modular *mut checker)

**Date:** 2026-06-04
**Branch:** `parser/fn-type-effects-list-e008` (off `check/closure-hof-triple-e008`)
**Scope:** `self-hosted/parser/types.sio`, `self-hosted/check/check.sio` only — `lean_single.sio`
untouched, canonical gate unaffected. **Supersedes PR #235's effect-subsumption model.**

## Result (vs same-session closure-tip baseline 269 PASS / 72 CRASH)

**PASS 269 → 283 (+14), CRASH 72 → 65; zero PASS→FAIL, zero FAIL→CRASH, zero negative-suite regressions.**

Three sub-fixes, each verified by full per-file census (the layout-robust signal; raw PASS-count is
not stable across `bin/souc` build instances — see the campaign note):

### 1. Parser — effect-list no longer swallows the next parameter (+5)
`parse_effect_list` (parser/types.sio) consumed a trailing `, name:` as another effect, so a
`fn(T)->U with Eff,...` type used as a NON-last parameter failed to parse (the next param name was
eaten). Fix: break the loop when the comma is followed by `ident :` (`parser_peek_ahead(p,2)==Colon`).
Wins: `test_diffgeo`, `test_functional`, `test_multigrid`, `test_stochastic_calc` (scientific) +
`closure_effect_checked`.

### 2. Closure-literal SRET dodge (+6 CRASH→PASS)
`checker_check_closure_expr_inplace` still called three BY-VALUE `Checker` helpers
(`collect_closure_params → {checker}`, `bind_closure_params → Checker`, `lower_opt_closure_return →
(Checker, TypeEntry)`), each returning the ~164 KB `Checker` by value (tuple-wrapped in the return
case) — the SRET large-struct-return miscompile — so **any** closure literal `|x| …` SIGSEGV'd at
check time. Added `*mut` transcriptions using the existing inplace primitives. Wins:
`closure_arity_2`, `closure_basic`, `closure_escape`, `closure_linear`, `closure_returned`,
`closure_effect_transparent_hof`.

### 3. Effect-polymorphic HOF model (resolves a self-contradictory test suite)
The corpus contained two incompatible intended semantics for a bare `fn(T)->U` HOF param:
- `closure_effect_transparent_hof` (run-pass): "unannotated fn param is effect-polymorphic —
  requires the effect at the call site" → IO closure from an IO caller PASSES.
- `effects_closure_escape` (compile-fail): "pure HOF — f must be pure" → reject any effectful value.

Operator decision: **effect-polymorphism**. Implementation:
- **Closure-body effect inference:** a `closure_inference_depth` counter (new `Checker` field,
  declared last to preserve layout) makes `checker_check_callee_effects_inplace` ACCUMULATE a
  callee's effects into `current_effects` while checking a closure body (instead of enforcing), so
  the closure's inferred effect set captures them. `print`/`println`/`print_int`/`print_char` (IO
  builtins, not in `fn_sigs`) are recognized by name during inference.
- **Polymorphic params:** removed effect comparison from `fn_sigs_structurally_compatible` (a bare
  param accepts any-effect value by signature).
- **Call-site propagation:** in `checker_check_call_args_inner_inplace`, a fn-typed argument's
  effects are propagated via `checker_check_callee_effects_inplace` — enforced against the enclosing
  function's declared effects at depth 0, accumulated into an outer closure at depth > 0.

Outcome — all three reconciled: `closure_effect_escape` REJECTS (its caller `pure_fn` lacks IO),
`closure_effect_transparent_hof` PASSES (caller `main` has IO), `effects_closure_escape` still
rejects (its `fn(x){}` anonymous-fn syntax, unrelated to effects).

## Soundness verification
- Named effectful fn → HOF from a **pure** caller: **rejects**; from an **IO** caller: **passes**
  (the common case, not just closures).
- IO/GPU/Async closures from a pure caller: reject. The earlier `eff1`/GPU/Async "must reject"
  probes were subsumption-model artifacts — under polymorphism they correctly PASS when the caller
  declares the effect.
- Negative suite (fn/closure compile-fail subset): 0 reject→accept regressions.

## Known incompleteness (stated, not hidden)
- The negative scan was **targeted** (fn/closure compile-fail), not the full 250 — the full loop
  stalls on pre-existing 3 GB-bss SIGSEGV crashers even with `ulimit -c 0`.
- IO-builtin recognition is a **hardcoded list** (`print`/`println`/`print_int`/`print_char`); a
  closure using a different IO builtin from a pure caller would under-reject. This is an
  incompleteness of the inference, not a complete effect system.
