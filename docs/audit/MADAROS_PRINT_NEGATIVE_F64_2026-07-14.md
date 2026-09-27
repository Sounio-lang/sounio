<!-- docs:meta
topic_id: repo.docs.audit.madaros-print-negative-f64-2026-07-14
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-print-negative-f64-2026-07-14
-->

# Madaros v0.80.0 — `print(f64)` drops the magnitude of negative floats

**Date:** 2026-07-14
**Status:** FIXED (2026-07-20) — `emit_builtin_print_f64` reloads abs bits from `xmm0`
  after the `'-'` write (rdi held stdout fd). Regression:
  `tests/run-pass/println_f64_negative.sio`.
  **Wave15 B residual closeout (2026-07-22):** dedicated `print_f64` witness
  `tests/run-pass/print_f64_negative.sio` (`-0.0`, `-2.0`, `-0.5`, `+2.0` +
  bits oracle) and gate `scripts/ci/madaros_print_f64_negative_gate.sh`
  (`MADAROS_PRINT_F64_NEGATIVE_GATE_OK`). Closes the stale Note in
  `MADAROS_NATIVE_V2_F64_REMAINING_BUGS_2026-07-20.md` and issue #890.
**Toolchain:** `./bin/souc` → Madaros v0.80.0
**Owner:** CODEX-2 (`self-hosted/` float formatting in the print builtin)
**Class:** compiler-semantics · **Severity:** B1 (any stdout of a negative number is wrong)
Forensic dispatch per CLAUDE.md §8.

## Symptom

The overloaded `print(f64)` / `println(f64)` builtins render **every negative float** as `-0.000000` —
the sign is kept, the magnitude is zeroed. Positive floats print correctly.

```sounio
fn main() -> i32 with IO, Mut, Div, Panic {
    println(0.0 - 0.2)    // prints -0.000000  (expected -0.200000)
    println(0.0 - 0.75)   // prints -0.000000  (expected -0.750000)
    println(0.0 - 1.5)    // prints -0.000000  (expected -1.500000)
    println(0.2)          // prints  0.200000  (correct)
    return 0
}
```

## The value is intact — this is print-only

The underlying `f64` is correct; only the formatting is wrong:
```sounio
let x = 0.0 - 0.2
if x < 0.0 { ... }                 // TRUE  — really negative
// |x - (-0.2)| < 1e-9             // TRUE  — really equals -0.2
```
This was found via `linalg::matnm`: `matnm_inv([[2,1],[1,3]])` computes the correct inverse
`[[0.6,-0.2],[-0.2,0.4]]` (verified because `A·A⁻¹ = I` exactly), but printing it showed the
off-diagonals as `-0.000000`. All matnm arithmetic (solve/inv/mul/det) is correct; only display was wrong.

## Impact

- Any program that prints a negative result mis-displays it. Affects display helpers already merged:
  `epistemic::gum::gum_report` and `units::lib::quantity_show` mis-print negative values (their shipped
  examples happen to use positive values, so they weren't caught).
- Numerical correctness is **not** affected (comparisons, arithmetic, and file-free computation are fine).

## Workaround (in use)

Print the sign manually and the positive magnitude:
```sounio
if v < 0.0 { print("-"); print(0.0 - v) } else { print(v) }
```
`linalg::matnm::matnm_show` uses this (inlined — a private helper called from a `pub` fn across a module
boundary segfaults at runtime, a separate multi-module quirk; inlining avoids it).

## Acceptance gate

`println(0.0 - 0.2)` prints `-0.200000`.

## Next-Action

Fix the sign/magnitude handling in the f64 formatting path of the print builtin.
