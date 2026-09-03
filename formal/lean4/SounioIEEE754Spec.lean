-- formal/lean4/SounioIEEE754Spec.lean

/-!
# Sounio.IEEE754 — canonical IEEE-754 binary64 specification

This file is the **canonical IEEE-754 spec** for Sounio's
Float-bounded soundness work (Stage 3b-F, Phase 1). It declares
five axioms that capture the operationally relevant facts about
IEEE-754 binary64 arithmetic, drawn from Higham 2002 §2.1
(basic operation model) and IEEE-754-2008 §5.11 (total order on
finite operands).

These five axioms are the **only** Float-specific assumptions
in the Sounio proof base. The four `BoundedOrderedCarrier
Float` typeclass methods (`Float.le_trans`,
`Float.zero_le_zero`,
`Float.mul_le_mul_of_nonneg_right_bounded`,
`Float.add_le_add_right_bounded`) are **derived as theorems**
from this spec in `SounioFloatInstance.lean`.

## Why a separate spec file

The previous `SounioFloatInstance.lean` declared four
typeclass-shape axioms directly. This was honest but mixed
two concerns:
  - the typeclass interface (`BoundedOrderedCarrier`);
  - the IEEE-754 specification (Higham facts).

Phase 1 separates them:
  - `SounioIEEE754Spec.lean` (this file): 5 IEEE-754 axioms.
  - `SounioFloatInstance.lean`: 4 typeclass-method theorems,
    derived from the spec.

Migration to Route B (in-tree IEEE-754 model, ~5000 LOC, c.f.
Coq's Flocq) only requires proving the 5 axioms here; the 4
typeclass-method theorems remain unchanged.

## The 5 axioms (high-level)

  1. `Float.toRat : Float → Rat`
     Conversion to the rational subset of ℝ. In Phase 2 this
     becomes a function derived from `Float.toBits` + IEEE-754
     binary64 decoder.

  2. `Float.IsFiniteNormal : Float → Prop`
     Predicate for finite, non-NaN, non-Inf, non-subnormal
     operands. The operational domain for clinical PK regimes.

  3. `Float.toRat_le_iff_finite`
     IEEE-754 ≤ matches Rat ≤ on the finite-normal subset.

  4. `Float.mul_rne_bound`
     Higham 2002 §2.1 basic-operation model: for the
     finite-normal subset, the rounded product satisfies
       `|fl(a · b).toRat - (a.toRat · b.toRat)| ≤ u · |a.toRat · b.toRat|`
     where the unit roundoff `u = 2⁻⁵³` for binary64
     round-to-nearest. (Math-review note: `u = 2⁻⁵³`, not
     `2⁻⁵²`. The latter is `ulp(1)`, equivalent to `2u`.)

  5. `Float.add_rne_bound`
     Same Higham §2.1 model for addition.

## Math-review record

Pre-impl thesis (`/tmp/ieee754_spec_thesis.md`): Grok 4.1
review caught 3 critical bugs:
  - **WRONG**: ε_machine should be `2⁻⁵³` (unit roundoff u),
    not `2⁻⁵²` (ulp(1)). Higham §2.4 unit roundoff
    `u = 2⁻ᵖ⁻¹⁻¹ = 2⁻⁵³` for binary64 (p = 53 significand
    bits). Fixed: this file uses `2⁻⁵³`.
  - **WRONG**: Citation `Higham §2.4 eq. (2.5)` is for
    summation (γ_n bound). Basic operation model is §2.1.
    Fixed: cite §2.1.
  - **OVERREACH**: cookbook `eps_inf ≥ ε(|ac| + |bc|)` misses
    the add-rounding step `fl(bc + eps_inf)`. Coarse fix:
    `eps_inf ≥ 3u · max(|ac|, |bc|)`. Documented in the
    derivation (`SounioFloatInstance.lean`).
  - **WRONG**: `add_le_add` "no 0≤c needed" claim is for Rat,
    not for Float (which still needs the err budget). Fixed:
    cookbook is symmetric to mul.

Plus 5 OK (toRat_le_iff_finite, le_trans derivation sketch,
zero_le_zero derivation, two-sided if-then-else encoding,
dual-representation Phase 1 scope).

## Status

`lake build SounioIEEE754Spec` builds clean. Five axiom
declarations, no theorems, no Mathlib import.
-/

namespace Sounio.IEEE754

-- ================================================================
-- §1. Float.toRat — conversion to ℚ ⊆ ℝ.
-- ================================================================

/-- Convert a `Float` to its `Rat` representation. Axiomatised
    in Phase 1; in Phase 2 this becomes a function derived
    from `Float.toBits` + IEEE-754 binary64 decoder.

    For NaN inputs, the value is unspecified (any
    rational). All downstream theorems condition on
    `IsFiniteNormal`, so NaN inputs are excluded. -/
axiom Float.toRat : Float → Rat

-- ================================================================
-- §2. Float.IsFiniteNormal — operational subset predicate.
-- ================================================================

/-- Predicate for finite, non-NaN, non-Inf, non-subnormal
    operands. The operational domain for clinical PK regimes
    (operands in `[10⁻², 10⁴]` are well within the
    finite-normal range `[2⁻¹⁰²², 2¹⁰²³]`).

    Subnormals are excluded because their rounding error
    follows a different (denormalised) bound that doesn't
    match the §2.1 basic-operation model.

    Math-review note: tightened from `IsFinite` per Grok's
    TIGHTENABLE feedback: subnormals strictly excluded. -/
axiom Float.IsFiniteNormal : Float → Prop

-- ================================================================
-- §3. Float.toRat_le_iff_finite — ≤ matches Rat ≤.
-- ================================================================

/-- IEEE-754 `≤` matches `Rat ≤` on the finite-normal subset.
    Both directions of the iff hold.

    For finite-normal operands, IEEE-754 binary64 `≤` is the
    total order on the representable subset of ℝ, and `toRat`
    is order-preserving (and order-reflecting). NaN values are
    excluded by the finiteness hypotheses.

    Source: IEEE-754-2008 §5.11; Higham 2002 §2.5. -/
axiom Float.toRat_le_iff_finite :
  ∀ {a b : Float},
    Float.IsFiniteNormal a → Float.IsFiniteNormal b →
    (a ≤ b ↔ Float.toRat a ≤ Float.toRat b)

-- ================================================================
-- §4. Float.mul_rne_bound — Higham §2.1 multiplication.
-- ================================================================

/-- The unit roundoff for IEEE-754 binary64 round-to-nearest:
    `u = 2⁻⁵³`. Higham 2002 §2.4 defines `u` as the maximum
    relative error of the rounded result.

    Math-review note: `u = 2⁻⁵³`, not `2⁻⁵²` (the latter is
    `ulp(1) = 2u`). Earlier draft had `2⁻⁵²` from a citation
    error; corrected via Grok review. -/
def unit_roundoff : Rat := 1 / (2 ^ (53 : Nat) : Rat)

/-- The two-sided absolute value of a `Rat`, encoded without
    `Rat.abs` to avoid the absolute-value lemma chain. -/
def rat_abs (q : Rat) : Rat := if 0 ≤ q then q else 0 - q

/-- Higham 2002 §2.1 basic-operation model for binary64
    multiplication, restricted to finite-normal operands.

    For finite-normal `a`, `b` with `a * b` finite-normal:
        |(a * b).toRat - a.toRat · b.toRat| ≤ u · |a.toRat · b.toRat|
    where `u = 2⁻⁵³` is the unit roundoff for binary64
    round-to-nearest, no underflow, no overflow.

    Stated as a two-sided bound (avoiding `Rat.abs`):

    The first conjunct: `(a * b).toRat - a.toRat · b.toRat ≤
    u · |a.toRat · b.toRat|`
    (rounded-up-ness bound).

    The second conjunct: `a.toRat · b.toRat - (a * b).toRat ≤
    u · |a.toRat · b.toRat|`
    (rounded-down-ness bound).

    Source: Higham 2002 §2.1 eq. (2.4): `fl(x op y) = (x op y)
    (1 + δ)` with `|δ| ≤ u`. The two-sided bound above is the
    direct consequence.

    Math-review note: corrected from §2.5 (summation γ_n bound)
    to §2.1 (basic operation model) per Grok review. -/
axiom Float.mul_rne_bound :
  ∀ {a b : Float},
    Float.IsFiniteNormal a → Float.IsFiniteNormal b →
    Float.IsFiniteNormal (a * b) →
    Float.toRat (a * b) - Float.toRat a * Float.toRat b
      ≤ unit_roundoff * rat_abs (Float.toRat a * Float.toRat b)
    ∧
    Float.toRat a * Float.toRat b - Float.toRat (a * b)
      ≤ unit_roundoff * rat_abs (Float.toRat a * Float.toRat b)

-- ================================================================
-- §5. Float.add_rne_bound — Higham §2.1 addition.
-- ================================================================

/-- Higham 2002 §2.1 basic-operation model for binary64
    addition, restricted to finite-normal operands.

    For finite-normal `a`, `b` with `a + b` finite-normal:
        |(a + b).toRat - (a.toRat + b.toRat)| ≤ u · |a.toRat + b.toRat|

    Same Higham §2.1 reference as `mul_rne_bound`.

    Math-review note: previous draft claimed addition cookbook
    was "simpler — no 0≤c needed". That conflated Rat.add_le
    (no nonneg condition) with Float.add (which still has the
    error budget). Corrected: cookbook is symmetric to mul. -/
axiom Float.add_rne_bound :
  ∀ {a b : Float},
    Float.IsFiniteNormal a → Float.IsFiniteNormal b →
    Float.IsFiniteNormal (a + b) →
    Float.toRat (a + b) - (Float.toRat a + Float.toRat b)
      ≤ unit_roundoff * rat_abs (Float.toRat a + Float.toRat b)
    ∧
    (Float.toRat a + Float.toRat b) - Float.toRat (a + b)
      ≤ unit_roundoff * rat_abs (Float.toRat a + Float.toRat b)

-- ================================================================
-- §6. Float.div_rne_bound — Higham §2.1 division.
-- ================================================================

/-- Higham 2002 §2.1 basic-operation model for binary64
    division, restricted to finite-normal operands. Completes
    the {mul, add, div} rounding triad (§4–§6).

    For finite-normal `a`, `b` with `a / b` finite-normal:
        |(a / b).toRat - a.toRat / b.toRat| ≤ u · |a.toRat / b.toRat|
    where `u = 2⁻⁵³`. Same `fl(x op y) = (x op y)(1 + δ)`,
    `|δ| ≤ u` model as §4/§5 — division is a §2.1 basic operation.

    ⚠️ SCOPE: this is the **relative-error bound** on ONE rounded
    division. It is NOT a monotonicity oracle: it does NOT imply
    that a rounded *composition* such as `fl(fl(r·c)/fl(k+c))` is
    monotone in `c` (independent rounding of a non-decreasing
    numerator and denominator can dip sub-ulp near a binade
    boundary). See `f_boost_monotone_in_siro` for that obstruction.

    Source: Higham 2002 §2.1 eq. (2.4). Division `b ≠ 0` is
    implied by `IsFiniteNormal (a / b)` (a finite-normal quotient
    cannot arise from a zero divisor). -/
axiom Float.div_rne_bound :
  ∀ {a b : Float},
    Float.IsFiniteNormal a → Float.IsFiniteNormal b →
    Float.IsFiniteNormal (a / b) →
    Float.toRat (a / b) - Float.toRat a / Float.toRat b
      ≤ unit_roundoff * rat_abs (Float.toRat a / Float.toRat b)
    ∧
    Float.toRat a / Float.toRat b - Float.toRat (a / b)
      ≤ unit_roundoff * rat_abs (Float.toRat a / Float.toRat b)

end Sounio.IEEE754
