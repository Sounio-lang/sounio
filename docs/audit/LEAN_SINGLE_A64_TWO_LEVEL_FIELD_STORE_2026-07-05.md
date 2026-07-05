<!-- docs:meta
topic_id: repo.docs.audit.lean-single-a64-two-level-field-store-2026-07-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lean-single-a64-two-level-field-store-2026-07-05
-->

# lean_single forensic dispatch — aarch64 has no codegen for two-level chained field assignment

Date: 2026-07-05
Branch: `main` (post-PR #640, a64 struct-literal aggregate-copy fix)
Class: **missing aarch64 codegen — a whole class of assignment statements had
no dispatch entry at all**, not a subtle addressing bug like the previous
dispatch — closes 2 of the 4 aarch64 `native_runtime` failures left
out-of-scope by that fix
Status: fixed, verified — aarch64 native_runtime manifest 97/99 pass (up
from 95/99; the 2 remaining are the separately-diagnosed, unrelated
`epistemic_ops_42`/`epistemic_propagation` Knowledge<T>-arithmetic
segfaults), full x86-64 suite 1314/0/124/689 exact baseline (fix is
aarch64-only)

## Symptom

`s.inner.vals[0] = 10` (two chained field accesses — `inner`, then `vals` —
followed by an array index) silently does nothing on aarch64: no compile
error, no crash, the write simply never happens.

```sio
struct Inner { vals: [i64; 2] }
struct Outer { inner: Inner }

fn main() -> i32 with IO, Mut, Panic, Div {
    var s = Outer { inner: Inner { vals: [0; 2] } }
    s.inner.vals[0] = 10
    s.inner.vals[1] = 32
    println(s.inner.vals[0])  // pre-fix (aarch64): 0, should be 10
    println(s.inner.vals[1])  // pre-fix (aarch64): 0, should be 32
    0
}
```

The identical source compiles and runs correctly on x86-64. This affected
issue tracking as `abi_nested_array_local_only_42` and
`abi_return_nested_array_42` in the aarch64 `native_runtime` manifest —
both left explicitly out of scope by the previous dispatch
(`docs/audit/LEAN_SINGLE_A64_STRUCT_FIELD_AGGREGATE_COPY_2026-07-05.md`)
pending their own investigation.

## Root cause

`self-hosted/compiler/lean_single.sio`'s statement compiler recognizes
assignment-statement shapes by their exact token pattern
(`stmt_is_*_shape()` helpers, shared between backends) and dispatches each
shape to a dedicated codegen function. x86-64's dispatch chain
(`compile_stmt()`) has explicit entries for:

- `stmt_is_field_field_array_store_shape` (`x.f1.f2[idx] = expr`) →
  `compile_value_field_field_array_store_x86()` /
  `compile_autoderef_field_field_array_store_x86()`
- `stmt_is_field_field_store_shape` (`x.f1.f2 = expr`, no array) →
  `compile_value_field_field_store_x86()` /
  `compile_autoderef_field_field_store_x86()`

aarch64's dispatch chain (`compile_stmt_a64()`) had **no entries for either
shape at all** — it only handled the single-level cases (`x.field[idx] =
expr`, `x.field = expr`) and a different two-level shape where the array
index comes *before* the second field (`x.arr_field[idx].leaf = expr`, via
`stmt_is_indexed_field_field_array_store`). A statement matching the
two-plain-fields shape fell through every check in `compile_stmt_a64()` and
was silently swallowed with no diagnostic.

Confirmed via a minimal repro that this is aarch64-specific (the identical
source runs correctly on x86-64) and via `grep` that
`stmt_is_field_field_array_store_shape`/`stmt_is_field_field_store_shape`
were called only from `compile_stmt()` (x86-64), never from
`compile_stmt_a64()`, before this fix.

## Fix

Added the aarch64 twins of all four x86-64 functions, ported instruction-
for-instruction using the already-correct aarch64 encodings from the
sibling single-level function `compile_field_array_store_a64()` (register
plan: index in `x1`, value in `x10`, base pointer in `x0`, matching that
function's proven-working `str x10, [x0, x1, lsl #3]` /
`strb w10, [x0, x1]` store instructions verbatim) and the existing
`emit_store_to_pointer_offset_a64()` helper (already used by the
pointer/ref single-field store path) for the non-array scalar case:

- `compile_value_field_field_array_store_a64()` — `x.f1.f2[idx] = expr`,
  value-struct root (the struct's own slot holds a pointer to its field
  data, so the same "load slot, add combined field offset, indexed store"
  shape as the working single-level function applies directly).
- `compile_autoderef_field_field_array_store_a64()` — same shape through a
  `&!T`/pointer-typed root (auto-deref).
- `compile_value_field_field_store_a64()` / `compile_autoderef_field_field_store_a64()`
  — the no-array sibling (`x.f1.f2 = expr`), delegating to
  `emit_store_to_pointer_offset_a64()` for the actual store, exactly as
  x86-64's twins delegate to `emit_store_to_pointer_offset_x86()`.

Wired all four into `compile_stmt_a64()`'s dispatch, in the same relative
order x86-64 uses (field-field-array and field-field checked before the
existing single-level and indexed-field-field-array checks).

**Scope note**: the no-array sibling (`x.f1.f2 = expr`) was not one of the
4 originally-reported failing tests, but was discovered to have the
identical "no aarch64 dispatch entry at all" gap while implementing this
fix — confirmed via its own minimal repro (`s.inner.tag = 99` silently
no-oped on aarch64, worked on x86-64) — and fixed alongside the array
variant since it is the same missing-feature class, not a separate defect.

## Verification

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
scripts/dev/souc-build-lock.sh ./bin/souc-lean-single-x86_64 self-hosted/compiler/lean_single.sio /tmp/lean_fixed.elf

# Exact CI repros, cross-compiled + run under qemu-aarch64-static:
/tmp/lean_fixed.elf tests/selfhost/native_runtime/abi_nested_array_local_only_42.sio /tmp/t1.elf --target aarch64-linux
qemu-aarch64-static /tmp/t1.elf; echo $?   # 42 (was 0)
/tmp/lean_fixed.elf tests/selfhost/native_runtime/abi_return_nested_array_42.sio /tmp/t2.elf --target aarch64-linux
qemu-aarch64-static /tmp/t2.elf; echo $?   # 42 (was 0)

# Full aarch64 native_runtime manifest (99 cases) via aarch64-linux + qemu:
# 97 pass / 2 fail (was 95/4) — both new targets fixed; the 2 remaining are
# epistemic_ops_42/epistemic_propagation (unrelated, separately diagnosed).

# Full x86-64 suite (fix is aarch64-only, unaffected by construction):
bash scripts/run_sio_test_suite.sh --format junit --jobs 8
# Pass: 1314  Fail: 0  Known failures: 124  Skip: 689  Total: 2127 — exact baseline
```

Also confirmed directly:
- The no-array sibling (`s.inner.tag = 99`) now correctly prints `99` (was
  `0`).
- The autoderef (pointer-root) variant, via a `&!Outer` parameter mutating
  two chained fields (one array, one scalar) inside a called function:
  correct on both x86-64 and the aarch64 fix (`55`, `66`, `77`).
- Single-level (`x.field[idx] = expr`, already-working
  `compile_field_array_store_a64`) and nested-struct-literal-read
  (unrelated to assignment) both remain correct — this dispatch is purely
  additive, it does not modify any existing dispatch branch.

## Cross-references

- `docs/audit/LEAN_SINGLE_A64_STRUCT_FIELD_AGGREGATE_COPY_2026-07-05.md` —
  the immediately preceding dispatch, which found and fixed a different
  aarch64 bug (`copy_agg_into_struct_slots_a64`'s off-by-`(nslots-1)`
  addressing) and left this one, plus `epistemic_ops_42`/
  `epistemic_propagation`, explicitly out of scope for follow-up.
- `epistemic_ops_42`/`epistemic_propagation` remain open — both segfault on
  `Knowledge<f64>` binary arithmetic (`a + b`) on aarch64, confirmed via a
  minimal repro to be unrelated to field-chain assignment (isolated to the
  epistemic/uncertainty-propagation operator codegen, not investigated
  further here).
