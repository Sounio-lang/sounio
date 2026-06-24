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

## Escape safety
- **Return-escape** (`fn bad() -> &i64 { let x=5; &x }`) → rejected by the checker's escape
  analysis **E091** (already on `main`, PR #397).
- **Direct field-store escape** (`h.p = &x`, storing a ref-to-local into a struct field) →
  new **loud lowering error** ("storing a reference into a struct field is not supported").
  Before this, enabling `OpRef` would have let it compile to a dangling pointer.

## Honest scope / known gaps
- **Indirect field-store escapes** (`let r = &x; h.p = r`) are **not** caught — E091 tracks
  provenance at return sites, not store sites, and the lowering guard only catches a direct
  address-of. Such a program can still produce a dangling pointer. The proper fix is
  extending E091 to store sites (a follow-up); this PR ships the common, safe cases
  (references passed to functions, read/written through) plus guards for the two most common
  escape vectors.
- Reference fields in structs are otherwise minimally supported; multi-level refs
  (`&&T`) and references to temporaries are not addressed here.

## Verified (madaros from this source, `ulimit -s unlimited`)
- `rd(&n)*p`→42, `(*p).a` of `&P`→42, `mutref *p=*p+1`→42 (alias write-back).
- `escret`→E091; `dangle2` (direct field-store of `&x`)→loud error.
- No-regression: 26/50 run-pass = prebuilt main (0 regressed); madaros self-builds.

## AI disclosure
Implementation by AI agent (Claude) under human direction, on advisor guidance to gate the
ship on (a) the escape guard being live on `main` and (b) the write-through-alias test
(`mutref`) rather than the read-only test. Every claim backed by a re-runnable probe.
