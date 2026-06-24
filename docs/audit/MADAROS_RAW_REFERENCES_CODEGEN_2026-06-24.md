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

## Escape safety — second-class references, enforced at compile time

References are **second-class**: pass them to functions and read/write through them, but they
may **not be returned or stored**. The checker's escape analysis (E091) now enforces this at
**both return sites and store sites**, so every measured escape route is a hard compile error
(no ELF), not a dangling pointer. The store-site check (in `checker_check_assign_stmt_inplace`)
rejects assigning a reference into a struct field, an array element, or through a pointer,
when either the value is a frame-local reference (binary provenance) **or** the value's type
is a reference (`TyRef`/`TyRefMut`) — the latter covers a stored *reference parameter*, i.e.
the interproc case.

| Escape route | Before | Now |
|---|---|---|
| `return &x` | E091 (return site, #397) | E091 |
| `h.p = &x` (field store) | silent / printed warning | **E091, no ELF** |
| `arr[i] = &x` (array store) | silently dangled | **E091, no ELF** |
| `*pp = &x` (store through deref) | silently dangled | **E091, no ELF** |
| `&x` → fn that stores its ref param (interproc) | compiled, crashed | **E091, no ELF** |

The intended, sound use — passing a reference to a function and reading/mutating through it
within the callee — is unaffected.

## Honest scope
- Enforcement is by **value type / provenance at the assignment**, not full lifetime
  inference: it conservatively rejects *all* reference stores (the second-class rule), so a
  hypothetical safe pattern like storing a reference to a longer-lived global is also
  rejected. This is sound (never a dangling ref), not complete.
- Store-site check covers field/index/deref assignment targets. A reference stored into a
  bare `ident` target is not separately checked, but a *local* `let r = &x` only escapes when
  `r` is itself later returned or stored (both covered), and Sounio has no first-class global
  reference variables.
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
