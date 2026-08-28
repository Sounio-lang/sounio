<!-- docs:meta
topic_id: repo.docs.audit.madaros-closures-step1-fix-2026-06-24
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-closures-step1-fix-2026-06-24
-->

# Madaros zero-capture closures — working end-to-end (step 1, 2026-06-24)

*Branch:* `wip/madaros-closures-step1` (off `main`). Resolves the closure triage
(`MADAROS_CLOSURES_TRIAGE_2026-06-23.md`) **step 1**: `(|x| x)(n)` and `let f = |x| …; f(n)`
now compile and run correctly. (Capture is step 2; still out of scope.)

## Summary

Closures were broken at five points, four of which are the **same systemic root**: the
Madaros native codegen (the lean_single-built seed) drops/partials a store when its target
is a **field-accessed Box dereferenced inline** (`(*lo.field).x = v` or
`(*lo.field).arr[i] = v`). The working idiom everywhere else in the lowerer is to **extract
the Box to a local first** (`var b = lo.field; (*b).x = v`), which `fresh_reg` and
`lowerer_flush_current_func_mut` already use. The closure path did not.

## The five fixes (all in `lower_closure_expr_ref` / `lower_call_expr_ref`)

1. **Compile crash (let-bound):** the closure used the *functional* `find_or_add_fn_id`,
   whose ADD path returns a 128 KB `IrFunction` by value (miscompiled) → switched to the
   `*mut` `lowerer_find_or_add_fn_id_mut` (as regular calls do).
2. **Inline callee:** `lower_call_expr_ref` resolved the callee by *name*; an inline closure
   `(|x| x)(n)` has no name, so it fell through to `find_or_add_fn_id(empty)` → a body-less
   call. Now an expression callee is lowered to a fn-ref reg + `ir_call_indirect`.
3. **`current_func_loaded`:** the functional lowering reads param info from `lo.current_func`
   only when `current_func_loaded` is true (else from the empty module placeholder); the
   closure never set it. Now saved/set/restored around the closure body.
4. **Param metadata dropped (extract-Box):** `(*lo.current_func).param_count = pc+1` (deref
   of a field-accessed Box) was dropped → `param_count` stayed 0 → no parameter prologue →
   SIGSEGV. Fixed by extracting the func Box to a local first (`var cf = lo.current_func;
   (*cf).param_count = …`). Verified: the param is then passed (`f(7) → 7`).
5. **Body instrs dropped (extract-Box):** the flush `(*lo.module).functions[id] = *func` (a
   field-accessed-Box two-level store) was **partial** — `param_count` copied but the body
   instr array did not, so the closure returned its parameter instead of the body result
   (`|x| 99 → 7`). Fixed by extracting the module Box first, as
   `lowerer_flush_current_func_mut` does.

## Verified (madaros built from this source, `ulimit -s unlimited` as `bin/madaros` sets)

- `(|x| x)(42) → 42`; `let f=|x| x; f(42) → 42`; `let f=|x| x+10; f(32) → 42`
- `|x| 99 → 99`; `|x| x+100; f(23) → 123`; `|a,b| a*b+2; f(8,5) → 42`
- two closures in one program: `f(10)+g(31) → 42`
- **No regression:** identical 18/40 run-pass exit-0 vs the prebuilt main madaros (0
  regressed); structs (`P{7,35}.a+.b → 42`), enums (`C::G → 2`), methods, fn-pointers, free
  fns all still run. madaros self-builds.

## Honest scope

- **Zero-capture only.** Capturing closures (`|x| x + k`) are step 2 — the body still wipes
  the enclosing locals (`empty_lower_local_stack()`); no free-variable analysis / environment
  yet. The triage's step 2/3 remain.
- The receiver/callee handling covers local + inline closures; `f().method()`-style call
  chains are unchanged.

## AI disclosure
Fix by AI agent (Claude) under human direction; every claim backed by re-runnable
`madaros compile/run` probes and the staged param_count traces used to localise each store.
