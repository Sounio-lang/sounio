<!-- docs:meta
topic_id: repo.docs.audit.seed-fix-value-nested-field-store-2026-06-24
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.seed-fix-value-nested-field-store-2026-06-24
-->

# Seed fix: value-struct nested field store (2026-06-24)

*The first fix applied to the **lean_single bootstrap seed** itself, reversing the standing
"never edit the seed — dodge in modular source" rule. Justified because dodging had hit a hard
wall for self-hosting (see `MEMORY` / the self-hosting tier notes), and the fixed point is
**re-verifiable**, not lost.*

## The bug

`o.inner.b = v` — assignment whose LHS is a **two-level field access on a value struct**
(`o: Outer`, `Outer.inner: Inner`, `Inner.b: i64`) — was **silently dropped**: reading
`o.inner.b` afterward returned the old value. A one-level store (`i.b = v`) worked.

This is the root behind a whole class of session bugs (struct-field-index #413's
`table.entries[i].fields[fc]=…`, etc.), which were dodged with copy-modify-writeback.

## Root cause (located via Explore over the 1.7MB seed)

lean_single's codegen dispatches nested stores by **shape-specific handlers**. It had:
- `compile_autoderef_field_field_store_x86` — pointer root, scalar (`p.f1.f2 = v`) ✓
- `compile_value_field_field_array_store_x86` — value root, array (`o.f1.f2[i] = v`) ✓
- **MISSING: value root, scalar (`o.f1.f2 = v`)**

So a value-struct scalar nested store fell through the dispatcher (`compile_stmt`), the LHS was
parsed as a read-only expression, the `=` was hit as an unrecognized token and **skipped**
(`compile_primary` falls through to `EP+1; xor eax,eax`), and the RHS was discarded — no store
emitted.

## Fix (additive — lowest fixed-point risk)

1. Added `compile_value_field_field_store_x86` (`lean_single.sio`) — the pointer handler minus
   the pointer-type check: resolve the root struct from the var's own value-struct type via
   `resolve_field_struct_index`, sum the two field offsets, and store with
   `emit_store_to_pointer_offset_x86` (a value-struct local's slot holds a pointer to the
   struct, so the same store path is correct).
2. Dispatch (`compile_stmt`): the `name.f1.f2 = expr` branch now routes value-struct roots to
   the new handler instead of falling through.

Existing pointer/array paths are untouched.

## Verified
- Self-compile: the current seed compiles the fixed `lean_single.sio` cleanly (the new handler
  uses no value-struct nested stores, so it's compiled correctly by the old seed).
- Repro: `o.inner.b = 42 → 42` (was 0); `nested_a/nested_b → 42`; sibling field intact (5);
  one-level `i.b → 42`. 20/20 run-pass compile+run (no 139).
- **Fixed point preserved**: a compiler built from the fixed seed reproduces itself
  bit-identically (`md5(genA) == md5(genB)`), and the fix propagates through it
  (`genA: o.inner.b = 42`). Canonical `make build` gen2==gen3 gate: see CI self-host legs.

## Why this matters
This proves the **seed-fix + fixed-point-re-verify** workflow the project had avoided. It is
the path to retire the accumulated dodges and to unblock Madaros self-hosting (the remaining
nested-store shapes — pointer-deref-field-field, double-deref — can be fixed the same way).

## AI disclosure
Fix by AI agent (Claude) under human direction; bug located by an Explore sub-agent over the
seed, every claim backed by a re-runnable compile+run and the md5 fixed-point check.
