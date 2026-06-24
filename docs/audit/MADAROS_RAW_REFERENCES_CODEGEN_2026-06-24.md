<!-- docs:meta
topic_id: repo.docs.audit.madaros-raw-references-codegen-2026-06-24
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-raw-references-codegen-2026-06-24
-->

# Madaros raw references `&T` / `&!T` — codegen (2026-06-24)

*Branch off `main`. Raw references (true `&T`/`&!T`, distinct from `Box`) now lower **and**
codegen in native_v2; previously the whole path was a loud "native-v2 bridge compilation
failed".*

## What works now
- Pass a reference to a function: `fn rd(p: &i64) -> i64 { *p }`, `rd(&n)` → value.
- Read through a reference: `*p`, and `(*p).field` on `&Struct`.
- **Mutate through a reference**: `fn inc(p: &!i64) with Mut { *p = *p + 1 }`,
  `var n = 41; inc(&!n); n` → **42** (write-through-alias).

The lowering already emitted `IrUnaryOp(OpRef/OpRefMut/OpDeref)`; the gap was purely codegen
(the core_ir loop returned `false` for them — a deliberate loud-fail). The mutation path
needed a new store op.

## Changes
- **`codegen_x86_linux.sio`** core_ir `IrUnaryOp`: add
  - `OpDeref` → `mov rax,[p]; mov rax,[rax]` (raw pointer load),
  - `OpRef`/`OpRefMut` → `lea rax,[rbp+slot(src)]` (address of the operand's stack slot).
- **`*p = val`** (deref-store): new IR op `IrStorePtr` (`ir.sio`) + lowering case in
  `lower_assign_stmt_ref` (`lower.sio`) + codegen (`mov [p], val` via new helper
  `nc_emit_store_rbx_to_mem_rax`). A `Box` deref-store still routes to `IrFieldSet(0)`.

Aliasing works because native_v2 round-trips every temp through its rbp-relative slot, so
`&n` (the slot address) and a later read of `n` refer to the same memory.

## Escape safety — MEASURED, and weak (read this before relying on it)

References here are effectively **second-class with weak enforcement**. The only sound,
enforced guarantee is no-return; **store-escapes are largely unguarded and can silently
dangle.** Measured behaviour of address-of-a-local flowing into storage:

| Escape route | Observed |
|---|---|
| `return &x` | **rejected by E091** (checker, on `main` via #397) — sound |
| `h.p = &x` (direct field store) | prints an `error:` **but still emits an exit-0 ELF** (madaros leniency) — *loud, not blocked* |
| `arr[i] = &x` (array element store) | **compiles and runs — silent dangling** (reads a dead slot) |
| `*pp = &x` (store address through a deref) | **compiles and runs — silent dangling** |
| `&x` passed to a fn that stores it (interproc) | compiles, **crashes at runtime** (undefined) |

So: E091 enforces no-return; the lowering guard catches only the *direct* `obj.field = &x`
form and only as a printed warning; **all other store-escape routes are unguarded** and
range from silent-dangling to crash. The safe, intended use is **passing a reference
directly to a function and reading/writing through it within the callee's lifetime** — that
is sound (the referent outlives the call).

## Honest scope / known gaps
- The proper fix for store-escapes is to extend the escape analysis (E091) to **store sites**
  (provenance check on field/index/deref assignments and across calls), or to enforce true
  second-class references (no reference may be stored at all). Neither is done here.
- Reference fields in structs and arrays-of-references are mechanically possible but **not
  escape-safe** — see the table.
- Multi-level refs (`&&T`) and references to temporaries are not addressed.

## Verified (madaros from this source, `ulimit -s unlimited`)
- `rd(&n)*p`→42, `(*p).a` of `&P`→42, `mutref *p=*p+1`→42 (alias write-back).
- `escret`→E091; `dangle2` (direct field-store of `&x`)→loud error.
- No-regression: 26/50 run-pass = prebuilt main (0 regressed); madaros self-builds.

## AI disclosure
Implementation by AI agent (Claude) under human direction, on advisor guidance to gate the
ship on (a) the escape guard being live on `main` and (b) the write-through-alias test
(`mutref`) rather than the read-only test. Every claim backed by a re-runnable probe.
