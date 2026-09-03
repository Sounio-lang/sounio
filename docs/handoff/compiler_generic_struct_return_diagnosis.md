<!-- docs:meta
topic_id: repo.docs.handoff.compiler-generic-struct-return-diagnosis
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.compiler-generic-struct-return-diagnosis
-->

# Diagnosis — generic-struct-return monomorphisation gap

**Date:** 2026-07-06
**Author:** codex (MiniMax-M3)
**Branch:** gpu/epistemic-tensor-core-next @ afc2878cb
**Lane:** doc-claim + compiler-semantics (compiler-internals, serialized)

## TL;DR

Both engines reject `fn make_wrapper<T>(v: T) -> Wrapper<T>` and the
`fn cd_add<F>(a: CDExact<F>, b: CDExact<F>) -> CDExact<F>` shape. **Root
causes are engine-specific** but both share a common shape: the function's
type-parameter bindings are never substituted into the parameter/return
type expressions at the call site. The minimum fix touches Madaros's
`check.sio` and requires a Foundry rebuild. The lean_single side has a
downstream tail-type-mismatch that requires a separate fixed-point rebuild.

## Repro evidence (this branch)

```
$ ./bin/souc run tests/run-pass/turbofish.sio
   = expected T
   = found i64
   ...
error: type checking preflight failed
EXIT=0   # false-green, see RC caveat in CLAIM

$ bin/souc-lean-single-x86_64 tests/run-pass/turbofish.sio /tmp/x.elf
generics: fns=3 structs=1
mono: 4 specializations, tokens=2608
...
error: tail type mismatch at <main>:17
typecheck: failed

$ ./bin/souc run docs/handoff/spike_generic_struct_return.sio
   = expected CDExact
   = found CDExact__T
   = expected [F; 4
   = found [i64; 4
   ...
```

## Madaros root cause (default engine, bin/souc)

Path: `self-hosted/check/check.sio`

### Where the type args are lost

- Parser sets turbofish into `e.type_args` at `self-hosted/parser/exprs.sio:1106`
  (ExprIdent) and `:1128` (ExprPath / simple ident).
- `check_call_expr` at `:18319` line `:18424` calls `check_opt_expr(e.left)` to
  resolve the callee. That dispatches to `check_expr` ExprIdent arm at
  `:17342` which resolves to `ty_fn(fn_sig_id)` and **drops `e.type_args` on
  the floor**. No record is kept that the call site said `::<i64>`.
- `check_call_expr` at `:18467` then asks for `sig = fn_sig_table_get(...)`
  and passes the *unsubstituted* `sig.params` / `sig.return_type` to
  `check_call_args`. The function's type params never get bound, so the
  declared parameter type still contains `F` (or `[F; 4]`) literally.
- Downstream, `call_arg_types_compatible_structural` compares `arg_ty` (an
  `i64` literal narrowing) to `param_ty` (a `TyArray` of `TyNamed("F")`)
  and emits E009.

### Why the mangle says `__T` not `__F`

`substitute_type_param` at `:637` is shallow: it only handles `TyNamed`
leafs. For `c: [F; 4]` (TyArray), the inner `F` is in a `Box<TypeEntry>`
that the substitution never recurses into. So even if the call site tried
to bind `F -> i64`, the substitution would miss the inner `F`.

The `__T` rather than `__F` is the fallback arm of
`mangle_append_type_name` at `:583-587`: when a type arg is NOT a
recognized primitive kind and not `TyNamed`, the mangle appends the
literal character `T`. That means the type arg's lowered kind is not
`TyNamed("F")` — most likely `TyError` (the `lower_named_type_with_args`
path on a bare ident that doesn't match any builtin/unit/refinement
falls through to `ty_named`, which IS TyNamed, so the leak is more
likely in the call-site substitution machinery that never gets reached
and the "F" the checker tries to print is coming from the FnSig template
side, not the call-site arg side).

### Minimum Madaros fix

1. **FnSig** (in `self-hosted/check/types.sio`): add
   `type_param_count: i64`, `type_param_names: [Name; 8]`.
2. **lower_fn** at `check.sio:13645`: populate the two new fields from the
   `tp_count` / `tp_names` already in scope (no extra lowering work).
3. **Initialize every FnSig literal at all ~12 add sites** (`fn_sig_table_add`
   is called from `lower_fn`, closures, ghost fn signatures, etc.) — at
   minimum: `type_param_count: 0, type_param_names: [Name { buf: [0; 128], len: 0 }; 8]`
   for non-generic cases.
4. **substitute_type_param** at `:637`: recurse into `TyArray.inner`,
   `TyRef.inner`, `TyRefMut.inner`, `TySlice.inner`, `TyTuple` elements,
   `TyFn` param + return (mirroring how `lower_*_type` builds them). Keep
   the primitive fast-path first.
5. **check_call_expr** at `:18467`: before `check_call_args`, check
   `e.left.type_args`. If `Some(args)` and `sig.type_param_count > 0`,
   lower the args via `lower_type_expr_list`, then build a *substituted*
   `FnSig` with `params`/`param_count`/`return_type` mapped through
   `substitute_type_param`, and pass it to `check_call_args`. Return the
   substituted return type.
6. **check_method_call** at `:18108`: same shape; turbofish on a method
   call (`s.method::<T>()`) needs the same treatment.
7. **mangle_append_type_name**: optionally strengthen the fallback to
   emit the actual type name if any, so that the diagnostic shows
   `__F` not `__T` (cosmetic but reduces future forensic time).

### Rebuild cost & risk

- Madaros rebuild: ~3 minutes per the 2026-06-26 handoff. Needs
  `ulimit -v unlimited` under `env -i`. Goes via Sounio Compiler
  Foundry, NOT workspace-local, per AGENTS.md.
- Risk: FnSig is shared by `fn_sig_table_get` at 61+ locations. Missing
  initialization = ABI mismatch = likely segfault at first call to a
  generic fn. Mitigation: initialize both fields conservatively at
  every FnSig literal site.
- Risk: `substitute_type_param` recursion could infinite-loop on
  self-referential types (TypeEntry has no hash consing in this codegen
  target). Mitigation: bound recursion depth (use existing
  `type_depth` counter from `check_type_depth`).

### Control cases (baseline, BEFORE the fix)

Default engine (`./bin/souc`):

| test                                          | status   | annotation                  |
|-----------------------------------------------|----------|-----------------------------|
| generic_struct_basic                          | PASS     |                             |
| generic_struct_instantiate                    | PASS     |                             |
| generic_struct_nested                         | PASS     |                             |
| generic_knowledge                             | PASS     |                             |
| generic_arg_infer                             | FAIL     | "feature in lean_single.sio only" |
| generics_multi_param                          | FAIL     | no annotation (baseline red) |
| closure_generic_hof                           | FAIL     | no annotation (baseline red) |

Lean_single (`bin/souc-lean-single-x86_64`):

| test                                          | status   | notes                       |
|-----------------------------------------------|----------|-----------------------------|
| generic_struct_basic                          | PASS     |                             |
| generic_struct_instantiate                    | PASS     |                             |
| generic_struct_nested                         | PASS     |                             |
| generic_arg_infer                             | PASS     |                             |
| generics_multi_param                          | PASS     |                             |
| closure_generic_hof                           | PASS     |                             |
| turbofish (annot: known-failure)              | FAIL     | the bug under study         |

So lean_single handles the non-struct-return generic cases fine; Madaros
already breaks on bare turbofish-without-explicit-type-args (covered by
generic_arg_infer), plus the generic-struct-return shape we are fixing.

## Lean_single root cause (bootstrap seed, bin/souc-lean-single-x86_64)

Path: `self-hosted/compiler/lean_single.sio` (~35k lines)

### Mono pass produces the right mangling

Pass 0d at line `24390` correctly registers instantiations from turbofish
sites: `make_wrapper::<i64>` → mono entry, mangle `make_wrapper_i64`.
Pass 0c/0d collapse rule at line `24684` correctly substitutes `Wrapper<T>`
inside fn bodies to `Wrapper_i64`.

The "Re-scan monomorphized structs" loop at line `24758` should register
the emitted `46 Wrapper_i64 { val : i64 }` declaration in ST, but the
emitted token stream also includes the mono-copied struct declarations.
The bug is downstream — at `compile_fn` tail-type check, line `26541`:

```
if ty_eq(CURRENT_RET_TY, CURRENT_RET_HASH, LAST_STMT_TY, LAST_STMT_TY_HASH) == false {
    ...
    print("error: tail type mismatch at ")
    ...
}
```

The emitted `Wrapper_i64` declaration's field type was set via
`scan_type` to whatever `i64` lowers to. CURRENT_RET_TY is `Wrapper_i64`
(a struct ref) and LAST_STMT_TY should be `Wrapper_i64` (the struct
literal). They should match. The fact that they don't means either:

  (a) `Wrapper_i64` is not registered in ST (rescan missed it), so
      `ty_eq` against an unknown type returns false; OR
  (b) `ty_eq`'s struct-hash comparison is comparing against the
      generic `Wrapper` (not the mangled `Wrapper_i64`).

Without instrumentation I cannot isolate which. Lean_single's fixed-point
ELF would need to be regenerated with a probe (`print` after the rescan
showing ST[Wrapper_i64] idx). That requires a Foundry rebuild, which is
expensive and outside this workspace.

### Minimum lean_single fix (DIFF ONLY, not built)

Two-line instrumentation probe to confirm (a) vs (b):

  1. After the rescan at `24758`, add:
     `if name_eq(rsn_ns, rsn_ne, "Wrapper"...)` → print `rsh`, `rsi`, and
     whether `st_find(rsh)` was already populated.

  2. Before `26541`, print `CURRENT_RET_TY`, `CURRENT_RET_HASH`, and the
     same for `LAST_STMT_TY`.

Then rebuild via Foundry and re-run `tests/run-pass/turbofish.sio`. The
fix is one of:
  - (a) → rescan has a missing `continue` or a wrong `<` vs `<=`; check
    the loop bounds and the `TK[p] == 9` skip at `24828`.
  - (b) → `ty_eq` needs the mangled hash; check `gl_name_hash(Wrapper_i64)`
    vs whatever `ty_named("Wrapper_i64")` hashes to in the LAST_STMT arm.

### Why I'm not patching lean_single in this session

The canonical-compiler-gate (`scripts/ci/canonical_compiler_gate.sh`)
verifies `bin/souc-lean-single-x86_64` is the byte-identical fixed point
of `lean_single.sio`. Any source edit without a Foundry rebuild breaks
the gate. Per AGENTS.md and the CLAIM, heavy rebuilds go via Foundry.

## Decision back to operator

Two paths forward, both safe under the parallel-work contract:

**Path A (recommended by lane doc):** hand the compiler-internals fix to
the heavy compiler lane (nv2-compiler-hardening or Lane 4 — the owner of
the `bin/souc` token per the handoff prompt). I document the diagnosis
(this file) and the Madaros fix shape, then release the CLAIM.

**Path B:** I implement the Madaros-side fix in a worktree, ship a
Foundry handoff for the rebuild, run the canonical gate + the new
run-pass fixtures (`generic_struct_return.sio`, `generic_struct_return_structf.sio`),
and document the lean_single diff as staged-for-next-campaign. This is
within the AGENTS.md Codex lane ("implementation, file creation,
surgical refactors, test harness wiring") but requires the operator's
go-ahead because the serialized-surface protocol calls for explicit
CLAIM and the handoff prompt asks for both engines to be green.

## Acceptance criteria (Madaros side)

1. `tests/run-pass/turbofish.sio` → run-pass: remove the `known-failure`
   annotation; all three `PASS` markers print.
2. `tests/run-pass/generic_struct_return.sio` (new, promoted from the
   spike) → run-pass: prints `1` (r.c[0] is 1 because the body is the
   copy stub, not real add) then `spike PASS`.
3. `tests/run-pass/generic_struct_return_structf.sio` (new) — instantiates
   a 2-field struct as the type param, e.g. `Rational { num: i64, den: i64 }`,
   and a fn returning `CDExact<Rational>`. Both compile and run.
4. Canonical compiler gate green (only if lean_single.sio is NOT touched;
   if it is, requires Foundry refresh of bin/souc-lean-single-x86_64).
5. No regression on `generic_struct_basic / nested / instantiate /
   knowledge` (Madaros side).
6. Madaros E004/E009 no longer emitted for the new fixtures; **output-
   verified** (assert printed values, not just rc).

## Out of scope (explicit)

- Do NOT touch the exact-algebra consumer files
  (`stdlib/algebra/cayley_dickson_exact.sio`,
  `stdlib/math/sedenion_verdict.sio`, `tests/run-pass/sedenion_zd_*`).
- Do NOT touch Lane 3 paper-168 files
  (`examples/cocycle_*`, `examples/*168*`, `docs/papers/main/168-*`).
- No trait system, no higher-kinded types, no generic-method
  monomorphisation beyond the minimum.
- No Lean / GPU / EISA / paper-lane work.
