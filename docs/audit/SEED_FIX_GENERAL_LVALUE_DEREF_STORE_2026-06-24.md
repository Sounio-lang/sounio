<!-- docs:meta
topic_id: repo.docs.audit.seed-fix-general-lvalue-deref-store-2026-06-24
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.seed-fix-general-lvalue-deref-store-2026-06-24
-->

# Seed fix: general lvalue base for deref indexed-field stores (2026-06-24)

*Second fix to the lean_single bootstrap seed. Closes the **complex-base** indexed-store class
that blocked heap-indirect data structures (the path to Madaros self-hosting).*

## The bug

A store whose LHS is a deref of an **arbitrary pointer expression** —
`(*(*c).w).data[i] = v`, and the exact shape the heap-indirect checker uses,
`(*(*c).fn_sigs.entries).data[i] = sig` — was **silently dropped**. A simple-name deref store
`(*p).field[i] = v` worked.

## Root cause

lean_single dispatches stores via ~16 shape-specific handlers, **every one of which hardcodes
a simple-variable root** (`var_find(ns, ne)`). There is **no general lvalue-address routine**.
When the base is a complex expression (`(*c).w`), no pattern matches, the LHS is parsed as a
read-only expression, the `=` is hit as an unrecognized token and skipped, and the RHS is
discarded.

## Fix (additive — existing handlers untouched)

1. `stmt_is_deref_complex_field_array_store` — detector for `(*<complex>).field[idx] = …`,
   discriminated from the simple-name form by `TK[p0+2]==6` (the inner is parenthesized);
   paren-matches the outer `(` to locate `.field[ … ] =`.
2. `compile_deref_complex_field_array_store_x86` — evaluates the **base pointer with
   `compile_or`** (so any nesting works), then stores: push base, push index, `rhs→rdx`,
   pop index→rcx, pop base→rax, `lea rax,[rax+foff]`, `mov [rax+rcx*esize], …`. Field offset
   and element size are resolved from the base pointer's pointee struct type.
3. Dispatch: route the complex shape ahead of the simple-name deref handlers (no conflict —
   they require a simple-name root).

This generalises the lvalue base for the indexed-field deref store, closing the whole
complex-base class in one handler rather than a shape at a time.

## Verified
- `(*(*c).w).data[i]: 0 → 42`; inline double-deref `→ 99`. No regression: simple
  `(*p).inner.arr[i] → 42`, the #427 value-nested `o.inner.b → 42`, 15/15 run-pass no-139.
- **Fixed point preserved**: `make build` gen2==gen3 bit-identical (CI self-host legs are the
  canonical gate).

## Why
This is the access shape heap-indirect Checker tables (the rustc-`TyCtxt` pattern) require.
With it, the scale fix that makes Madaros type-check its own 117-module source becomes
expressible without dodges. Scalar complex-base derefs and the a64 path are follow-ups.

## AI disclosure
Fix by AI agent (Claude) under human direction; the missing-general-lvalue root was located by
an Explore sub-agent over the seed; every claim is backed by a re-runnable compile+run and the
md5 fixed-point check.
