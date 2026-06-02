# bin/souc SRET-forwarding codegen bug — minimal repro (2026-06-02)

**A function that returns the struct result of another struct-returning function returns a
ZEROED struct.** Verified, minimal (6 lines, 1-field struct), size-independent. It provably
explains the olanzapine bug (commit `a3ea42082`). Whether it relates to the G1 lane's E008
work is an **open hypothesis with evidence against it** — see "Relation to E008" below;
treat this as a real, independently-worth-fixing codegen bug, not (yet) "the E008 root."

Repro: `docs/audit/g1_wip/SRET_FORWARDING_MINIMAL_REPRO_2026-06-02.sio`
```
struct S { f0: f64 }
fn ctor() -> S { var p: S = S { f0: 7.0 }; return p }
fn make() -> S { return ctor() }     // forwards a struct-returning call
fn main()       { let s = make(); print_f64(s.f0) }   // prints 0.000000; want 7.0
```
Verified on `bin/souc` md5 `9d4ef541` (the bootstrap compiler), `ulimit -s 1048576`.

## Isolation matrix (exact)
| `make()` body | result | |
|---|---|---|
| `return ctor()` | **0.0** | BUG |
| `var p = ctor(); return p` | **0.0** | BUG (= `olanzapine_pbpk_params_smoker` shape) |
| `let p = ctor(); return p` | **0.0** | BUG |
| `ctor()` (tail) | **0.0** | BUG |
| **`let p = ctor(); <read p in place>`** (no outer return) | **7.0** | **OK (control)** |

- **Size-independent**: triggers at a 1-field (8-byte) struct AND at 44 fields. Not a
  frame-size / large-struct-threshold effect.
- **Not the field write**: the no-reassign copy-and-forward (`var p = ctor(); return p`
  with ctor setting the field) already returns 0. It is the **return-forwarding** of a
  struct-returning call that loses the data.
- **Use-in-place is fine**: `let p = ctor(); p.f0` read in a non-returning caller gives 7.0.
  The bug only manifests when the caller's own return value IS the forwarded struct.

## Mechanism (hypothesis, for the codegen fix)
SRET (struct-return via hidden pointer): the outer fn's hidden return-slot pointer is **not
threaded into the inner struct-returning call**. The inner `ctor` writes into a discarded
temporary (or a fresh stack slot), and the outer fn returns its own untouched (zeroed)
return slot. Use-in-place works because the inner call's result is materialized into a real
local before being read. The fix is in the SRET calling-convention lowering: when a
struct-returning call's result is itself returned (directly, via tail, or via a local that
is then returned), forward the caller's sret pointer to the callee instead of allocating a
throwaway.

## Relation to E008 — HYPOTHESIS, with evidence AGAINST a direct match
Initial guess was "this is the E008 root." A cheap discriminator weakened it; stated
honestly so the G1 lane isn't misdirected:
- **TyUnit discriminant is 3, not 0** (TyI64=0, TyF64=1, TyBool=2, TyUnit=3, check/types.sio:19).
  So a zeroed struct/discriminant reads as **TyI64 (0)**, NOT TyUnit. The clean "zeroed →
  TyUnit → E008 'expected ()'" mechanism therefore does **not** hold — E008's "expected ()"
  is TyUnit(3), which zeroing would not produce.
- **Symptom mismatch**: this bug is **non-crashing** (silent 0.0) and **size-independent**
  (fires at N=1). The G1 lane's *current* blocker is a **crash** ("layout-sensitive sentinel
  deref", commits `cd4377838`/`7f8c4dac8`). Non-crashing + size-independent vs crashing +
  layout-sensitive ⇒ **likely a different (sibling) codegen bug**, not the same one.
- It *might* relate to the earlier `3f2591eeb` "fn_sigs nested-write not persisting" thread
  (a value-persistence flavor), but that link is unproven and the discriminant evidence
  above argues against the simplest mapping. **Do not treat as E008's root without
  demonstrating the E008 path actually forwards a struct-returning call AND that the
  zeroing maps to TyUnit (it doesn't, per the discriminant).**

## What IS established
- **Verified codegen bug**: SRET return-forwarding zeroes the returned struct. Real,
  reproducible, size-independent, worth fixing on its own.
- **Explains olanzapine**: `olanzapine_pbpk_params_smoker` (`var prm = ...(); ...; return
  prm`) was this exact pattern; fixed at source with a constant literal (`a3ea42082`).
- A clean small repro for whoever works the SRET/struct-return lowering — possibly a
  sibling of the G1 codegen issues; useful regardless of the E008 mapping.

## Handoff
This is a doc + repro (no `check.sio` / codegen edit) — the SRET lowering lives in the
codegen the G1 lane is actively working. Hand this minimal repro to whoever drives the
codegen fix; it should pin the bug in minutes vs gdb-ing the full Checker. Supporting:
`FRONT_HALF_LEVERAGE_HANDOFF_2026-06-02.md`, `EPISTEMIC_DEMO_SWEEP_2026-06-02.md`.
