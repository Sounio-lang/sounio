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

## Escape safety — second-class references, enforced (all measured routes caught)

References are second-class: pass them to functions and read/write through them, but they may
not be returned or stored. The escape analysis (E091) covers **return sites and store sites**,
including references hidden inside aggregates and yielded from `if`/`match`/block tails. Every
measured escape route is a hard compile error (no ELF), not a dangling pointer. The two
mechanisms:
- **Stores** (`checker_check_assign_stmt_inplace`): reject assigning into a struct field, array
  element, or through a pointer when the value **transitively contains a reference**
  (`checker_type_has_ref`: a ref, slice, array-of-ref, or struct with a ref field, recursively)
  **or** is a frame-local reference by provenance. The type clause catches a stored ref
  *parameter* (interproc) and references buried in nested structs.
- **Returns / value tails** (`checker_expr_provenance`): a frame-local reference yielded
  directly, through a struct literal, or via an `if`/`match`/block **tail** expression.

| Escape route | Status (measured) |
|---|---|
| `return &x` / `return r` (direct) | **E091** |
| `return` via `if`/`match`/block tail yielding `&local` | **E091** |
| `h.p = &x` / `arr[i] = &x` / `*pp = &x` | **E091, no ELF** |
| interproc: `&x` → fn that stores its ref param | **E091, no ELF** |
| store `H{ p: &local }` / `H{ p: <ref param> }` (aggregate) | **E091, no ELF** |
| store a nested `Outer{ Inner{ p: r } }` | **E091, no ELF** |
| no over-rejection: `rd(&n)*p`, `inc(&!n)`, `(*p).a`, **return a param ref** | compile + run / OK |

## Honest scope
- Enforcement is **provenance + structural-type** based, not full lifetime inference, so it is
  conservative: it rejects *all* stores of a reference-containing value (the second-class
  rule), including the hypothetical safe store of a longer-lived reference. Sound direction,
  not complete.
- `checker_type_has_ref` recursion is depth-bounded (16) and treats an unregistered named type
  as ref-free; tuple element types are not walked (a tuple literal embedding a reference
  loud-fails earlier as **E005**, so it is not a silent hole).
- A reference stored into a bare `ident` target is not separately checked, but a local
  `let r = &x` only escapes when `r` is later returned or stored (both covered), and Sounio
  has no first-class global reference variables.
- Multi-level refs (`&&T`) and references to temporaries are not addressed.

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
