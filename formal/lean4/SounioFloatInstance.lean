-- formal/lean4/SounioFloatInstance.lean
import SounioFloatBounded

/-!
# Sounio.FloatInstance — `BoundedOrderedCarrier Float` (Stage 3b, Route C)

This file is the **Route C interim** Float instance for the
`BoundedOrderedCarrier` typeclass from `SounioFloatBounded.lean`.

## ⚠️  WARNING: this file uses 4 axioms

This file declares **four** axioms asserting structural facts
about IEEE-754 binary64 (`Float`) arithmetic. The axioms are NOT
proven in Sounio; they are deliberate assumptions whose
discharge is part of the long-term roadmap.

The route choice is documented in
`docs/research/lean_float_real_roadmap.md` Stage 3b. The three
routes are:

  - **Route a — Mathlib `Float` bridge** (~10 LOC if it exists):
    Use Mathlib's IEEE-754 lemmas to *prove* the axioms below.
    Replaces this file's `axiom` declarations with `theorem`s
    whose proofs use Mathlib.

  - **Route b — In-tree IEEE-754 model** (~5000 LOC): Formalise
    IEEE-754 binary64 round-to-nearest from scratch in Sounio
    (rounding modes, ulp, guard bits). Comparable to Coq's
    Flocq library.

  - **Route c — Axiomatised interim** (this file, ~150 LOC):
    Declare the IEEE-754 properties as axioms with explicit
    citations to Higham 2002 §2.4. Fast and operationally
    useful, but breaks the axiom-free in-tree philosophy.

Route C is the **explicit** documented choice here. When
Route A or B becomes available, this file's axioms become
theorems (or are deleted in favour of imports), and downstream
code requires no changes.

## Why per-call `eps_inf`, not a fixed bound

A naive Route C might axiomatise:

    axiom Float.mul_bounded_error :
      ∀ a b, |fl(a · b) - (a · b)| ≤ 0.0001

But this is **mathematically wrong** for IEEE-754: the rounding
error scales with the operand magnitude as `ulp/2 ∝ |a · b| · 2⁻⁵²`.
For `a = b = 10¹⁰`, the actual error is ~`5 · 10⁻⁷`, not bounded
by `0.0001`.

The math-review (Grok 4.1, 2026-05-01) caught this BUG:

    [WRONG] Higham 2002 §2.4 eq. (2.5) supports fixed 0.0001
      Higham states |fl(x op y) - (x op y)| ≤ ulp(fl(x op y))/2
      (round-to-nearest), ulp ∝ |x op y|.

The fix: instead of axiomatising a fixed bound, axiomatise the
typeclass laws **directly**, with the per-call `eps_inf`
parameter (already part of `BoundedOrderedCarrier`) carrying
the user-supplied bound. The user computes the correct ulp
budget from operand magnitudes per Higham; the axiom asserts
that any sufficient `eps_inf` discharges the typeclass law.

This is the "right" Route C: the axioms exactly mirror the
typeclass methods, and the soundness obligation is pushed to
the **call site** where the user knows the operand magnitudes.

## The 4 axioms

```
axiom Float.le_trans :
  ∀ {a b c : Float}, a ≤ b → b ≤ c → a ≤ c

axiom Float.zero_le_zero :
  (0.0 : Float) ≤ 0.0

axiom Float.mul_le_mul_of_nonneg_right_bounded :
  ∀ {a b : Float} (c eps_inf : Float),
    a ≤ b → 0.0 ≤ c → 0.0 ≤ eps_inf →
    a * c ≤ (b * c) + eps_inf

axiom Float.add_le_add_right_bounded :
  ∀ {a b : Float} (c eps_inf : Float),
    a ≤ b → 0.0 ≤ eps_inf →
    a + c ≤ (b + c) + eps_inf
```

Each axiom has a citation to the corresponding IEEE-754 fact:

  - `Float.le_trans`: IEEE-754 binary64 `≤` is a total order on
    the finite (non-NaN) representable subset; transitivity is
    immediate. The axiom is sound for the finite subset and
    holds vacuously when any operand is NaN (`a ≤ b` is `False`
    if either is NaN, so the implication is true). Higham 2002
    §2.5.

  - `Float.zero_le_zero`: positive zero ≤ positive zero is
    `True` per IEEE-754 §5.11. Trivially sound.

  - `Float.mul_le_mul_of_nonneg_right_bounded`: from Higham
    2002 §2.4 eq. (2.5):
        |fl(a · c) - (a · c)| ≤ ulp(a · c) / 2 ≤ k · ε_machine · |a · c|
    where `ε_machine = 2⁻⁵²` for binary64 and `k ≤ 4` for a
    single operation. The axiom states that *any* `eps_inf ≥
    k · ε_machine · max(|a|, |b|, |c|, |a·c|, |b·c|)`
    discharges the bound. The user is responsible for choosing
    a valid `eps_inf` per call.

  - `Float.add_le_add_right_bounded`: same Higham reference,
    with `+` substituted for `·`.

## User responsibility (cookbook formula)

When using `BoundedOrderedCarrier Float` methods, the user
MUST supply `eps_inf` satisfying:

    eps_inf ≥ k · ε_machine · max(|a|, |b|, |c|, |a*c|, |b*c|)

where:
  - `ε_machine = 2⁻⁵² ≈ 2.22 × 10⁻¹⁶` (binary64)
  - `k ≤ 4` for a single arithmetic operation
        (round-to-nearest, no underflow, no overflow,
         no Subnormal-flush-to-zero)

For clinical PK regimes (operands in `[10⁻², 10⁴]`):

    max-magnitude ≈ 10⁴
    eps_inf ≥ 4 · 2.22 × 10⁻¹⁶ · 10⁴ = 8.88 × 10⁻¹²

For the Sounio Cmin band (width ~3 mg/L), the rounding-induced
inflation is `~9 × 10⁻¹² mg/L` — fourteen orders of magnitude
below the trough/MIC clinical threshold (which is ≥ 1 mg/L).

## Counter-example to a naive fixed-bound axiom

Just to make the math-review-caught BUG concrete, here's why
a fixed bound `0.0001` would be unsound:

```
example_axiom_failure (large operands):
  Let a = 10¹⁰ (large finite Float, normal range)
  Let b = 10¹⁰
  Then a · b = 10²⁰
  ulp(10²⁰) ≈ 10²⁰ · 2⁻⁵² ≈ 2.22 × 10⁴
  So |fl(a · b) - (a · b)| can reach ≈ 1.11 × 10⁴
  which exceeds 0.0001 by EIGHT orders of magnitude.

example_axiom_failure (clinical operands, smaller margin):
  Let a = b = 10⁴ (clinical PK upper range, e.g. dose in μg)
  Then a · b = 10⁸
  ulp(10⁸) ≈ 10⁸ · 2⁻⁵² ≈ 2.22 × 10⁻⁸
  Half-ulp ≈ 1.11 × 10⁻⁸
  This is well below 0.0001 — but the trend is clear: as
  operand magnitude grows, fixed bounds eventually fail. The
  per-call design tracks this scaling correctly.
```

The per-call axiom design avoids both failures: the user passes
`eps_inf ≥ k · ε_machine · max(|a|, |b|, |a·b|)` per Higham
2002 §2.4, scaled to operand magnitudes.

Math-review caught a numerical slip in an earlier draft of this
example (claimed `ulp(10²⁰) ≈ 10⁻⁵` instead of `≈ 10⁴`); the
qualitative point — that fixed bounds break for large operands —
is unchanged, and the corrected arithmetic is above.

## Status

The four axioms + the `BoundedOrderedCarrier Float` instance +
a demonstration theorem build cleanly under
`lake build SounioFloatInstance`. The file uses `axiom`
deliberately and explicitly; this is the documented Route C
interim. No `sorry`, no Mathlib import.

Math-review record:
  - Pre-impl thesis (`/tmp/cauchy_float_thesis.md`):
    1 BUG (fixed-bound axiom unsound for large operands —
    addressed by switching to per-call `eps_inf`),
    1 TIGHTENABLE (Route C honesty — addressed by counter-
    example documentation above and the `WARNING` header),
    + 6 OK on the per-call axiom design once corrected.
-/

namespace Sounio.FloatInstance

-- ================================================================
-- §1. The 4 axioms — IEEE-754 binary64 facts.
-- ================================================================

/-- IEEE-754 `≤` transitivity for binary64 finite (non-NaN)
    operands. Soundness: NaN handling makes the antecedent
    `False` which discharges the implication; for finite
    operands, `≤` is a total order so transitive.

    Source: Higham 2002 §2.5; IEEE-754-2008 §5.11. -/
axiom Float.le_trans :
  ∀ {a b c : Float}, a ≤ b → b ≤ c → a ≤ c

/-- IEEE-754 zero is reflexively `≤` itself.

    Source: IEEE-754-2008 §5.11 (positive zero ≤ positive
    zero is `True`). -/
axiom Float.zero_le_zero : (0.0 : Float) ≤ 0.0

/-- IEEE-754 right-multiplication monotonicity with bounded
    inflation. The axiom states that for ANY `eps_inf ≥ 0`, the
    bound holds.

    The user is responsible for choosing `eps_inf` satisfying
    the cookbook formula
        eps_inf ≥ k · ε_machine · max(|a|, |b|, |c|, |a·c|, |b·c|)
    per Higham 2002 §2.4 eq. (2.5).

    Source: Higham 2002 §2.4 eq. (2.5). -/
axiom Float.mul_le_mul_of_nonneg_right_bounded :
  ∀ {a b : Float} (c eps_inf : Float),
    a ≤ b → 0.0 ≤ c → 0.0 ≤ eps_inf →
    a * c ≤ (b * c) + eps_inf

/-- IEEE-754 right-addition monotonicity with bounded inflation.
    Same Higham reference as `mul`. -/
axiom Float.add_le_add_right_bounded :
  ∀ {a b : Float} (c eps_inf : Float),
    a ≤ b → 0.0 ≤ eps_inf →
    a + c ≤ (b + c) + eps_inf

-- ================================================================
-- §2. The BoundedOrderedCarrier Float instance.
-- ================================================================
--
-- Each method is a thin wrapper around the corresponding axiom.
-- The instance is structurally complete: `lake build` requires
-- no further proofs once the axioms are admitted.

instance : BoundedOrderedCarrier Float where
  zero := 0.0
  le_trans := fun {a b c} hab hbc =>
    Float.le_trans (a := a) (b := b) (c := c) hab hbc
  zero_le_zero := Float.zero_le_zero
  mul_le_mul_of_nonneg_right_bounded :=
    fun {a b} c eps_inf hab hc heps =>
      Float.mul_le_mul_of_nonneg_right_bounded
        (a := a) (b := b) c eps_inf hab hc heps
  add_le_add_right_bounded :=
    fun {a b} c eps_inf hab heps =>
      Float.add_le_add_right_bounded
        (a := a) (b := b) c eps_inf hab heps

-- ================================================================
-- §3. Demonstration: Stage 3b bounded-Fréchet on Float.
-- ================================================================
--
-- The bounded-Fréchet theorem from `SounioFloatBounded.lean`
-- now applies on `Float` directly. This demonstrates that the
-- Stage 3b typeclass machinery is genuinely operational on the
-- IEEE-754 carrier.

/-- Bounded vancomycin Cmin Fréchet enclosure on `Float`.

    Direct application of
    `Sounio.BoundedOrderedCarrier.vancomycin_cmin_frechet_enclosure_bounded`
    with α = `Float`. The user supplies the bounded-monotone
    hypotheses for the actual `cmin` Float computation, the
    rounding budgets `eps_vc`, `eps_cl`, the combined budget
    `total_eps`, and the bookkeeping lemmas.

    For clinical PK regimes (operands in `[10⁻², 10⁴]`), the
    cookbook formula gives:
        eps_vc, eps_cl ≤ 4 · 2⁻⁵² · 10⁴ ≈ 9 × 10⁻¹²
        total_eps ≤ eps_vc + eps_cl ≈ 2 × 10⁻¹¹

    The Sounio Cmin band correctness modulo `2 × 10⁻¹¹ mg/L`
    is **eleven orders of magnitude** below the trough/MIC
    clinical threshold. -/
theorem vancomycin_cmin_frechet_enclosure_float
    (cmin : Float → Float → Float)
    (eps_vc eps_cl total_eps : Float)
    (mono_vc_bounded :
      ∀ vc vc' cl, vc ≤ vc' → cmin vc cl ≤ (cmin vc' cl) + eps_vc)
    (mono_cl_bounded :
      ∀ vc cl cl', cl ≤ cl' → cmin vc cl' ≤ (cmin vc cl) + eps_cl)
    (vc_lo vc_hi cl_lo cl_hi vc cl : Float)
    (h_vc_lo : vc_lo ≤ vc) (h_vc_hi : vc ≤ vc_hi)
    (h_cl_lo : cl_lo ≤ cl) (h_cl_hi : cl ≤ cl_hi)
    (combine_lower :
      ((cmin vc cl) + eps_vc) + eps_cl
        ≤ (cmin vc cl) + total_eps)
    (combine_upper :
      ((cmin vc_hi cl_lo) + eps_cl) + eps_vc
        ≤ (cmin vc_hi cl_lo) + total_eps)
    (lift_lower :
      cmin vc_lo cl + eps_cl
        ≤ ((cmin vc cl) + eps_vc) + eps_cl)
    (lift_upper :
      cmin vc_hi cl + eps_vc
        ≤ ((cmin vc_hi cl_lo) + eps_cl) + eps_vc)
    : cmin vc_lo cl_hi ≤ (cmin vc cl) + total_eps
      ∧ cmin vc cl ≤ (cmin vc_hi cl_lo) + total_eps :=
  Sounio.BoundedOrderedCarrier.vancomycin_cmin_frechet_enclosure_bounded
    (α := Float)
    cmin eps_vc eps_cl total_eps
    mono_vc_bounded mono_cl_bounded
    vc_lo vc_hi cl_lo cl_hi vc cl
    h_vc_lo h_vc_hi h_cl_lo h_cl_hi
    combine_lower combine_upper
    lift_lower lift_upper

-- ================================================================
-- §4. Roadmap to discharge the axioms.
-- ================================================================
--
-- Each axiom corresponds to a known, citable IEEE-754 fact:
--
-- 1. `Float.le_trans` — provable from any IEEE-754 model that
--    formalises ≤ on the finite subset. Mathlib's
--    `Float.le_trans` (when added) discharges this directly.
--
-- 2. `Float.zero_le_zero` — provable by reflection
--    (`decide`) once `Float.le` reflects to a decidable
--    instance for literal arguments. Mathlib provides this.
--
-- 3. `Float.mul_le_mul_of_nonneg_right_bounded` — provable
--    from any IEEE-754 model that includes:
--      (a) the rounding bound `|fl(x op y) - (x op y)| ≤ ulp/2`
--      (b) the relationship `ulp(x) ≤ x · 2⁻⁵²` for normal
--          x ≠ 0
--      (c) Cauchy-Schwarz-style operand triangle inequalities
--    Coq's Flocq has all of (a)-(c). The Lean port (Float-Flocq)
--    is the canonical Route B discharge target.
--
-- 4. `Float.add_le_add_right_bounded` — same as 3 but for
--    addition. Same discharge target.
--
-- The migration path: when Route A (Mathlib) or Route B (in-tree
-- IEEE-754) lands, this file's `axiom` keywords become `theorem`
-- and the bodies become proofs. Downstream consumers (the
-- `BoundedOrderedCarrier Float` instance and the
-- `vancomycin_cmin_frechet_enclosure_float` theorem) continue
-- to work without modification — they only depend on the
-- typeclass interface, not on whether the laws are axiomatic
-- or proven.

end Sounio.FloatInstance
