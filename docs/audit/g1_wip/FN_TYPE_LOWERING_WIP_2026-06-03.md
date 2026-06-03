# fn(T)->U type lowering — correct in isolation but NOT pushable (WIP, 2026-06-03)

Branch `check/fn-type-lower-e008` (commit `2a6339c25`), LOCAL / UNPUSHED off the PASS-322 tip.
Do NOT push as-is: +0 corpus PASS and introduces 1 crash.

## What it does (correct)
checker_lower_type_expr_mut had no TypeFn case -> fn-pointer/closure param & field types fell to
the silent `_ => checker_note_type_error_mut` arm. Ported lower_fn_type_expr + lower_fn_type_params
to *mut helpers (register FnSig via (*c).fn_sigs.add, return ty_fn(sig_id)). Minimal repro
`fn apply(f: fn(i64)->i64, x: i64) -> i64 { f(x) }` now check:OK; bad body still rejected; 12
fn-type params fine. Confirmed it LANDS: closure programs moved from SILENT "type checking failed"
to a proper E009 diagnostic.

## Why it is NOT a clean win (the closure cluster is TRIPLE-blocked)
1. fn-type lowering (THIS fix — done).
2. **E009 "argument type does not match parameter"**: a closure expr `g` gets ty_fn(sigA), the
   param expects ty_fn(sigB); types_compatible compares fn-types by SIG-ID IDENTITY, not by
   STRUCTURAL signature (param types + return). So `apply(g, 5)` -> E009. Needs a ty_fn structural
   compatibility case in types_compatible (compat.sio).
3. closure-expression typing (`|x| ...`) interactions.
Net: +0 PASS until all three clear.

## The crash (blocker)
`quadrature` (FAIL on PASS-322 tip) -> SEGFAULT (rc=139) with my fix. NOT a fn_sigs scaling issue
(12 fn-type params are fine). Triggered by something in quadrature beyond line 340 (head -340
truncation does not crash; the file is longer). Unpinned. A crash cannot ship regardless of the
+0/+block analysis above.

## Verdict
The clean single-mechanism levers are exhausted at PASS 322. This cluster (closures/HOF) is a
multi-fix unit (fn-type + ty_fn structural compat + closure-expr) with an unresolved crash — a
larger, riskier effort than the 8 landed fixes. Shelved as WIP; resume by (a) pinning the
quadrature crash, (b) adding ty_fn structural compatibility, (c) re-measuring the cluster flip.
