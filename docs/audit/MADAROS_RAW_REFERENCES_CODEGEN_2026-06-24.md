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

## Escape safety — second-class references, enforced for the DIRECT forms (not fully sound)

References are treated as second-class: pass them to functions and read/write through them,
but do not return or store them. The escape analysis (E091) now covers **store sites** in
addition to return sites, but enforcement is **provenance/value-type based, not full
lifetime inference** — sound for the direct reference forms, with **measured gaps** for
references hidden inside aggregate values and for non-direct return tails. Read the table.

The store-site check (in `checker_check_assign_stmt_inplace`) rejects assigning into a struct
field, array element, or through a pointer when the value is a frame-local reference
(provenance) **or** its type is a reference (`TyRef`/`TyRefMut`, which catches a stored ref
*parameter* — the interproc case).

| Escape route | Status (measured) |
|---|---|
| `return &x` (direct) | **E091** (return site, #397) |
| `h.p = &x` / `arr[i] = &x` / `*pp = &x` | **E091, no ELF** |
| interproc: `&x` → fn that stores its ref param | **E091, no ELF** (value-type clause) |
| store/return `H{ p: &local }` (aggregate of a **local** ref) | **E091** (struct-literal provenance taint) |
| **store `H{ p: r }` where `r` is a ref PARAM** | ❌ **NOT caught — can silently dangle** |
| **`return …` via an `if`/block tail** that yields `&local` | ❌ **NOT caught** (E091's if/block-tail deferral) |

## Honest scope / known gaps (measured, not assumed)
- **Aggregate-embedded parameter reference, then stored** (`fn f(out:&!H, r:&i64){ (*out) =
  H{p:r} }`): the stored value's type is the struct (not `TyRef`) and its provenance is
  first-class (the embedded ref is a parameter), so neither clause fires → **silent dangling**.
  Closing it needs structural type analysis (a value whose type transitively contains a
  reference) or disallowing reference-typed struct fields outright.
- **`if`/block-tail returns** of a local reference are not caught — `checker_expr_provenance`
  does not recurse into `if`/block tail expressions (a pre-existing E091 deferral). Direct
  `return &local` and `return r` are caught.
- Conservative second-class rule otherwise: it rejects *all* direct reference stores, so a
  hypothetical safe store of a longer-lived reference is also rejected (sound-direction, not
  complete). A reference stored into a bare `ident` target is not separately checked.
- Multi-level refs (`&&T`) and references to temporaries are not addressed.

The honest one-line model: **direct reference store/return/interproc escapes are caught;
references smuggled through aggregate values or `if`/block-tail returns are not yet.**

## Verified (madaros from this source, `ulimit -s unlimited`)
- Functionality: `rd(&n)*p`→42, `(*p).a` of `&P`→42, `mutref *p=*p+1`→42 (alias write-back).
- Escape safety (all **E091, no ELF**): `return &x`, `h.p=&x`, `arr[i]=&x`, `*pp=&x`,
  interproc `store(h,&x)`.
- No over-rejection: the three functionality cases still compile and run.
- No-regression: 34/60 run-pass = prebuilt main +1 (a raw-ref test now passes), 0 regressed;
  madaros self-builds.

## AI disclosure
Implementation by AI agent (Claude) under human direction, on advisor guidance to gate the
ship on (a) the escape guard being live on `main` and (b) the write-through-alias test
(`mutref`) rather than the read-only test. Every claim backed by a re-runnable probe.
