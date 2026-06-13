# a64 nested-aggregate field-copy regression (2026-06-13)

## One-line

`copy_agg_into_struct_slots_a64` anchored its block-copy at the wrong slot for a
struct/array/Option field embedded **by value** inside another struct literal,
so the field's last word collided with the *next* field. a64-only; x86 correct.
This re-broke a fix (`47c1a4246`) that a later `origin/main` merge silently
reverted.

## Symptom cluster

- `tests/run-pass/autodiff_tape_basic` → **timeout** (a64). `TapeAndIdx{tape:
  MiniTape, idx}` returned `idx` = bit-pattern of `3.0` (= `tape.v0`); the garbage
  `out_idx` fed `while i>=0` → infinite loop.
- `tests/selfhost/native_runtime/abi_return_literal_array_field_42` → **SIGSEGV**
  (a64). This is `47c1a4246`'s *own* regression test.
- General: any `Outer { inner: Inner{…}, next: … }` — reads of fields after the
  embedded aggregate, and of the embedded aggregate's own trailing fields, return
  shifted/garbage values. x86 and a5ab are correct.

## Pin (minimal repro, i64-only)

```sounio
struct I { x: i64, y: i64 }
struct O { i: I, z: i64 }
fn main() -> i64 with IO { let o = O { i: I { x: 5, y: 7 }, z: 9 }; println(o.i.y); 0 }
```

- x86 / a5ab-a64: prints `7`.   HEAD-a64: prints `5`.

Disassembly of `main` (a64), building `O`:
- `I{5,7}` laid at `x29-0x28`(x), `x29-0x20`(y).
- copy of field `i` into `O`: HEAD sets dst `x11 = x29-0x10`, copies 2 words
  **upward** → writes `-0x10`(5), `-0x08`(7). a5ab sets dst `x11 = x29-0x18` →
  writes `-0x18`(5), `-0x10`(7).
- `z=9` then stores at `x29-0x08`. On HEAD that **overwrites `i.y`**; on a5ab `z`
  sits below `i` with no overlap.

The struct slot layout grows downward in address (higher slot = lower addr;
`dst_start = base_slot + total_slots - 1 - foff/8` is the field's highest slot =
lowest addr). `emit_copy_words_x10_x11_a64` copies with both pointers
**increasing**, so the copy must start at `lea(dst_start)`. HEAD started at
`lea(dst_start - (nslots-1))`, i.e. one-too-low, writing into the next field.

## Fix

`self-hosted/compiler/lean_single.sio`, `copy_agg_into_struct_slots_a64`: revert
the three anchors to `emit_lea_var_a64(dst_start)` (array, Option, struct paths).
This is byte-identical to `47c1a4246`. The regression entered via merge
`9b53bb8d4 Merge origin/main into fix/silent-typecheck-diag`, which restored the
pre-`47c1a4246` `dst_start - (nslots - 1)` form.

## Verification (required before declaring done)

1. Rebuild the x86 host from fixed source; re-emit a64 for the minimal repro,
   `abi_return_literal_array_field_42`, and `autodiff_tape_basic` → expect
   `7` / `42` / `ALL PASS`.
2. Confirm x86 target still correct for the same (no x86 regression — the edit is
   a64-only but verify).
3. Re-run the run-pass batch under qemu for collateral.

## Out of scope / still open

- `_diag_sobol` SIGILL is **pre-existing** (fails on a5ab too) — a different bug
  (field after a `[f64;5000]` array + huge by-value SRET), NOT covered by this fix.
- a64 field-**read** aggregate-detection (lean_single ~30409, ~30445) was missing
  `type_is_option_inline(EXPR_TY, EXPR_TY_HASH)` that the x86 path (~14675,
  ~14711) has — so reading a struct field of type `Option<T>` took the scalar
  (single-word) branch instead of leaving a pointer to the 2-slot Option, and the
  following `if let Some(..)` SIGSEGV'd. **RESOLVED in a follow-up commit** (added
  the predicate to both a64 read paths; regression test
  `tests/run-pass/option_struct_field_read.sio`). Was Option-only and orthogonal
  to the copy anchor, so kept as a separate atomic commit.
