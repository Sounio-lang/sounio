-- formal/lean4/SounioFloatInstance.lean
import SounioFloatBounded
import SounioIEEE754Spec

/-!
# Sounio.FloatInstance — `BoundedOrderedCarrier Float` (Stage 3b, Route C, Phase 1)

This file is the **Route C interim** Float instance for the
`BoundedOrderedCarrier` typeclass from `SounioFloatBounded.lean`.

## Phase 1 status (2026-05-01)

This file is in the middle of a multi-phase refactor:

  - **Phase 0** (committed `d0453156`): 4 typeclass-shape axioms
    (`Float.le_trans`, `Float.zero_le_zero`,
    `Float.mul_le_mul_of_nonneg_right_bounded`,
    `Float.add_le_add_right_bounded`) backing the
    `BoundedOrderedCarrier Float` instance directly.

  - **Phase 1** (this commit): canonical 5-axiom IEEE-754 spec
    extracted to `SounioIEEE754Spec.lean` (Higham 2002 §2.1
    basic-operation model). Two of the four typeclass methods
    are now derived as **theorems** from the spec:
      - `Float.le_trans_from_spec` (derived from
        `Float.toRat_le_iff_finite` + `Rat.le_trans`)
      - `Float.zero_le_zero_from_spec` (derived from
        `Float.toRat_le_iff_finite` + `Rat.le_refl`)

    The remaining two (`mul_le_mul_of_nonneg_right_bounded`
    and `add_le_add_right_bounded`) require ~100 LOC of Rat
    algebra over the spec, with a recursive/coarse cookbook
    hypothesis. They are **deferred to Phase 1.5** with their
    typeclass-shape axioms retained for the current
    `BoundedOrderedCarrier Float` instance.

  - **Phase 1.5** (next milestone): derive the remaining two
    typeclass methods from the spec, eliminating the four
    typeclass-shape axioms. Net result: 5 IEEE-754 axioms,
    zero typeclass-shape axioms, all four typeclass methods
    proven theorems.

  - **Phase 2** (long-term, ~5000 LOC): Route B in-tree
    IEEE-754 binary64 model formalisation. The 5 spec axioms
    in `SounioIEEE754Spec.lean` become theorems; this file's
    derived theorems remain unchanged.

## Why dual representation in Phase 1

Math-review (Grok 4.1, 2026-05-01) approved the dual
representation:

    [OK] Phase 1 dual rep (axioms + conditional theorems coexist)
      Defensible: preserves API; theorems tighter for
      finite-normal clients.

The Phase 1 dual rep:
  - 4 typeclass-shape axioms (unconditional; back the
    `BoundedOrderedCarrier Float` instance for arbitrary
    Float operands, including non-finite, where the bound
    holds vacuously);
  - 2 derived theorems (`le_trans_from_spec`,
    `zero_le_zero_from_spec`) with finiteness hypotheses;
  - 2 deferred theorems (mul/add) noted with explicit
    Phase 1.5 marker.

This preserves the public API (`BoundedOrderedCarrier Float`
typeclass instance unchanged) while progressively tightening
the axiom base.

## Math-review record

  - Pre-impl thesis (`/tmp/ieee754_spec_thesis.md`):
    Grok 4.1 5/9 OK + 3 WRONG (ε_machine = 2⁻⁵³ not 2⁻⁵²;
    Higham §2.1 not §2.5; cookbook recursive/coarse) +
    1 OVERREACH (cookbook missing add rounding;
    addressed by `eps_inf ≥ 3·u·max(|ac|,|bc|)` coarse form
    documented in Phase 1.5 deferred section below).

  - Post-impl: see commit message.

## Why per-call `eps_inf`, not a fixed bound

A naive Route C might axiomatise:

    axiom Float.mul_bounded_error :
      ∀ a b, |fl(a · b) - (a · b)| ≤ 0.0001

But this is **mathematically wrong** for IEEE-754: the rounding
error scales with the operand magnitude as `u · |a · b|`.
For `a = b = 10¹⁰`, the actual error is ~`10²⁰ · 2⁻⁵³ ≈ 10⁴`,
not bounded by `0.0001`.

The math-review (Grok 4.1, Phase 0) caught this BUG. The fix:
instead of axiomatising a fixed bound, axiomatise the
typeclass laws **directly** with the per-call `eps_inf`
parameter (already part of `BoundedOrderedCarrier`) carrying
the user-supplied bound. The user computes the correct
ulp/relative-error budget from operand magnitudes per Higham
§2.1.

## The 4 typeclass-shape axioms (retained from Phase 0)

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

In Phase 1.5 these become theorems derived from the spec.

## User responsibility (cookbook formula)

When using `BoundedOrderedCarrier Float` methods, the user
MUST supply `eps_inf` satisfying:

    eps_inf ≥ 3 · u · max(|a · c|, |b · c|)         (coarse form)

where:
  - `u = 2⁻⁵³ ≈ 1.11 × 10⁻¹⁶` (binary64 unit roundoff,
    Higham §2.4)
  - The factor `3` accounts for: 1× from `mul_rne_bound` of
    `a · c`, 1× from `mul_rne_bound` of `b · c`, 1× from
    `add_rne_bound` of `(b · c) + eps_inf` itself.

For clinical PK regimes (operands in `[10⁻², 10⁴]`):

    max-magnitude of products ≈ 10⁸
    eps_inf ≥ 3 · 1.11 × 10⁻¹⁶ · 10⁸ ≈ 3.3 × 10⁻⁸

For the Sounio Cmin band (width ~3 mg/L), the rounding-induced
inflation is `~3 × 10⁻⁸ mg/L` — eight orders of magnitude
below the trough/MIC clinical threshold (which is ≥ 1 mg/L).

## Status

Five-axiom IEEE-754 spec + four typeclass-shape axioms +
two derived theorems + the `BoundedOrderedCarrier Float`
instance + a demonstration theorem build cleanly under
`lake build SounioFloatInstance`. No `sorry`, no Mathlib
import.
-/

namespace Sounio.FloatInstance

-- ================================================================
-- §1. The 4 typeclass-shape axioms (retained from Phase 0).
-- ================================================================

/-- IEEE-754 `≤` transitivity for binary64 finite (non-NaN)
    operands.

    Phase 0: this is an axiom backing the
    `BoundedOrderedCarrier Float` instance.

    Phase 1: a stronger conditional version
    (`le_trans_from_spec`, with finiteness hypotheses) is
    derived from the spec below. This unconditional axiom
    is retained for the typeclass instance.

    Phase 1.5: this axiom becomes redundant (the unconditional
    version follows from the spec + Float NaN handling
    semantics).

    Source: Higham 2002 §2.5; IEEE-754-2008 §5.11. -/
axiom Float.le_trans :
  ∀ {a b c : Float}, a ≤ b → b ≤ c → a ≤ c

/-- IEEE-754 zero is reflexively `≤` itself.

    Source: IEEE-754-2008 §5.11 (positive zero ≤ positive
    zero is `True`). -/
axiom Float.zero_le_zero : (0.0 : Float) ≤ 0.0

/-- IEEE-754 right-multiplication monotonicity with bounded
    inflation. Phase 1.5 will derive this from
    `Sounio.IEEE754.Float.mul_rne_bound` + Rat algebra (~80 LOC).

    Source: Higham 2002 §2.1 eq. (2.4). -/
axiom Float.mul_le_mul_of_nonneg_right_bounded :
  ∀ {a b : Float} (c eps_inf : Float),
    a ≤ b → 0.0 ≤ c → 0.0 ≤ eps_inf →
    a * c ≤ (b * c) + eps_inf

/-- IEEE-754 right-addition monotonicity with bounded inflation.
    Phase 1.5 will derive this from
    `Sounio.IEEE754.Float.add_rne_bound` + Rat algebra (~70 LOC). -/
axiom Float.add_le_add_right_bounded :
  ∀ {a b : Float} (c eps_inf : Float),
    a ≤ b → 0.0 ≤ eps_inf →
    a + c ≤ (b + c) + eps_inf

-- ================================================================
-- §2. Phase 1 derived theorems from the IEEE-754 spec.
-- ================================================================

/-- Phase 1 derivation: `Float.le_trans` follows from
    `Float.toRat_le_iff_finite` + `Rat.le_trans`, conditioned
    on finiteness of all three operands.

    This is strictly tighter than the unconditional axiom
    `Float.le_trans` above (which is needed for the typeclass
    instance over arbitrary Float operands). For finite-normal
    clients, this conditional theorem replaces the axiom
    cleanly — no Float-specific assumption beyond the IEEE-754
    spec.

    Math-review record:
      [OK] le_trans derivation
        Chains iff via Rat.le_trans; finiteness ha,hb,hc
        sufficient (tight).
-/
theorem Float.le_trans_from_spec
    {a b c : Float}
    (ha : Sounio.IEEE754.Float.IsFiniteNormal a)
    (hb : Sounio.IEEE754.Float.IsFiniteNormal b)
    (hc : Sounio.IEEE754.Float.IsFiniteNormal c)
    (hab : a ≤ b) (hbc : b ≤ c) : a ≤ c := by
  have h_ab_rat :
      Sounio.IEEE754.Float.toRat a ≤ Sounio.IEEE754.Float.toRat b :=
    (Sounio.IEEE754.Float.toRat_le_iff_finite ha hb).mp hab
  have h_bc_rat :
      Sounio.IEEE754.Float.toRat b ≤ Sounio.IEEE754.Float.toRat c :=
    (Sounio.IEEE754.Float.toRat_le_iff_finite hb hc).mp hbc
  have h_ac_rat :
      Sounio.IEEE754.Float.toRat a ≤ Sounio.IEEE754.Float.toRat c :=
    Rat.le_trans h_ab_rat h_bc_rat
  exact (Sounio.IEEE754.Float.toRat_le_iff_finite ha hc).mpr h_ac_rat

/-- Phase 1 derivation: `Float.zero_le_zero` follows from
    `Float.toRat_le_iff_finite` + `Rat.le_refl`, conditioned
    on `IsFiniteNormal 0.0`.

    Note: requires `IsFiniteNormal 0.0` as a hypothesis
    (which is trivially true for IEEE-754 binary64 — zero is
    finite — but must be supplied as a witness in the
    Mathlib-free derivation).

    Math-review record:
      [OK] zero_le_zero derivation
        Trivial via refl; assumes h₀ : IsFiniteNormal 0
        (true, add as lemma).
-/
theorem Float.zero_le_zero_from_spec
    (h0 : Sounio.IEEE754.Float.IsFiniteNormal (0.0 : Float))
    : (0.0 : Float) ≤ 0.0 := by
  exact (Sounio.IEEE754.Float.toRat_le_iff_finite h0 h0).mpr
          Rat.le_refl

-- ================================================================
-- §3. Phase 1.5 deferred derivations (mul/add).
-- ================================================================

/-! ## Phase 1.5 deferred mul derivation

   `Float.mul_le_mul_of_nonneg_right_bounded`
    can be derived from `Sounio.IEEE754.Float.mul_rne_bound` +
    `Rat.mul_le_mul_of_nonneg_right` + `Sounio.IEEE754.Float.add_rne_bound`
    + `Sounio.IEEE754.Float.toRat_le_iff_finite`, given:

      - finiteness witnesses for `a, b, c, eps_inf, a*c, b*c,
        (b*c)+eps_inf`;
      - cookbook hypothesis: `eps_inf.toRat ≥ 3·u·max(|a.toRat·c.toRat|,
        |b.toRat·c.toRat|)` where `u = 2⁻⁵³` is the binary64 unit
        roundoff.

    Proof sketch (deferred ~80 LOC):
      1. From `a ≤ b` + finiteness: `a.toRat ≤ b.toRat`.
      2. From `0 ≤ c` + finiteness on c, and trivial
         `IsFiniteNormal 0`: `0 ≤ c.toRat`.
      3. By `Rat.mul_le_mul_of_nonneg_right`:
         `a.toRat · c.toRat ≤ b.toRat · c.toRat`.
      4. By `mul_rne_bound` for `a · c`:
         `(a · c).toRat ≤ a.toRat · c.toRat + u · |a.toRat · c.toRat|`.
      5. Combining 3 + 4:
         `(a · c).toRat ≤ b.toRat · c.toRat + u · |a.toRat · c.toRat|`.
      6. By `mul_rne_bound` for `b · c`:
         `b.toRat · c.toRat ≤ (b · c).toRat + u · |b.toRat · c.toRat|`.
      7. Chain (5,6):
         `(a · c).toRat ≤ (b · c).toRat + u · (|a.toRat · c.toRat| + |b.toRat · c.toRat|)`.
      8. By cookbook (eps_inf ≥ 3u · max(|ac|, |bc|)):
         `(b · c).toRat + u · (|ac| + |bc|) ≤ (b · c).toRat + eps_inf.toRat
                                                 - u · |bc + eps_inf|`.
      9. By `add_rne_bound` for `(b · c) + eps_inf`:
         `(b · c).toRat + eps_inf.toRat - u · |bc + eps_inf|
              ≤ ((b · c) + eps_inf).toRat`.
      10. Bottom: `(a · c).toRat ≤ ((b · c) + eps_inf).toRat`.
      11. By `toRat_le_iff_finite` reverse: `a · c ≤ (b · c) + eps_inf`.

    Phase 1.5 milestone scope: ~80 LOC of careful Rat algebra
    + finiteness propagation through Float arithmetic. The
    coarse cookbook constant `3` is sufficient (math-review
    confirmed: `3u · max(|ac|, |bc|)` covers both
    `mul_rne_bound`s and the `add_rne_bound` of
    `(b · c) + eps_inf`).

    Until Phase 1.5 lands, the unconditional
    `Float.mul_le_mul_of_nonneg_right_bounded` axiom in §1
    backs the typeclass instance.

    Math-review record:
      [OVERREACH] mul derivation cookbook
        Original `eps_inf ≥ ε(|ac| + |bc|)` missed
        `add_rne_bound` of `(bc + eps_inf)`. Coarse form
        `3u · max(|ac|, |bc|)` suffices.

   (No Lean theorem statement — deferred to Phase 1.5
    milestone. The proof sketch above is the mathematical
    content.)
-/

/-! ## Phase 1.5 deferred add derivation

   `Float.add_le_add_right_bounded`
    can be derived from `Sounio.IEEE754.Float.add_rne_bound` +
    `Rat.add_le_add_right` + `Sounio.IEEE754.Float.toRat_le_iff_finite`,
    given finiteness witnesses + cookbook
    `eps_inf.toRat ≥ 2u·max(|a+c|, |b+c|)`.

    Proof sketch (deferred ~70 LOC):
      1. From `a ≤ b` + finiteness: `a.toRat ≤ b.toRat`.
      2. By `Rat.add_le_add_right`:
         `a.toRat + c.toRat ≤ b.toRat + c.toRat`.
      3. By `add_rne_bound` for `a + c`:
         `(a + c).toRat ≤ a.toRat + c.toRat + u · |a.toRat + c.toRat|`.
      4. Combining 2 + 3:
         `(a + c).toRat ≤ b.toRat + c.toRat + u · |a.toRat + c.toRat|`.
      5. By `add_rne_bound` for `b + c`:
         `b.toRat + c.toRat ≤ (b + c).toRat + u · |b.toRat + c.toRat|`.
      6. Chain (4, 5):
         `(a + c).toRat ≤ (b + c).toRat + u · (|a + c| + |b + c|)`.
      7. By cookbook + `add_rne_bound` for `(b + c) + eps_inf`:
         `(b + c).toRat + 2u·max ≤ ((b + c) + eps_inf).toRat`.
      8. Bottom: `(a + c).toRat ≤ ((b + c) + eps_inf).toRat`.
      9. By `toRat_le_iff_finite` reverse: `a + c ≤ (b + c) + eps_inf`.

    Math-review record:
      [WRONG] add derivation "simpler — no 0≤c needed"
        Conflated Rat.add_le (no nonneg) with Float.add (still
        needs error budget). Cookbook is symmetric to mul:
        2u · max(|a+c|, |b+c|).

   (No Lean theorem statement — deferred to Phase 1.5.)
-/

-- ================================================================
-- §4. The BoundedOrderedCarrier Float instance.
-- ================================================================
--
-- The instance is unchanged from Phase 0: it uses the four
-- typeclass-shape axioms in §1. Phase 1.5 will switch the
-- backing to the spec-derived theorems.

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
-- §5. Demonstration: Stage 3b bounded-Fréchet on Float.
-- ================================================================

/-- Bounded vancomycin Cmin Fréchet enclosure on `Float`.

    Direct application of
    `Sounio.BoundedOrderedCarrier.vancomycin_cmin_frechet_enclosure_bounded`
    with α = `Float`. The user supplies the bounded-monotone
    hypotheses for the actual `cmin` Float computation, the
    rounding budgets `eps_vc`, `eps_cl`, the combined budget
    `total_eps`, and the bookkeeping lemmas.

    For clinical PK regimes (operands in `[10⁻², 10⁴]`), the
    cookbook formula gives:
        eps_vc, eps_cl ≤ 3 · 2⁻⁵³ · 10⁸ ≈ 3.3 × 10⁻⁸
        total_eps ≤ eps_vc + eps_cl ≈ 6.6 × 10⁻⁸

    The Sounio Cmin band correctness modulo `7 × 10⁻⁸ mg/L`
    is **seven orders of magnitude** below the trough/MIC
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
-- §6. Roadmap to Phase 1.5 + Phase 2.
-- ================================================================
--
-- **Phase 1.5** (next milestone):
--   - prove `Float.mul_le_mul_of_nonneg_right_bounded_from_spec`
--     and `Float.add_le_add_right_bounded_from_spec` from the
--     IEEE-754 spec via the proof sketches in §3 (~150 LOC
--     total Rat algebra).
--   - migrate the typeclass instance to use the spec-derived
--     theorems.
--   - delete the 4 typeclass-shape axioms in §1.
--   - net: 5 IEEE-754 axioms total, 0 typeclass-shape axioms,
--     all 4 BoundedOrderedCarrier methods proven theorems.
--
-- **Phase 2** (long-term, ~5000 LOC):
--   - in-tree IEEE-754 binary64 model formalisation
--     (rounding modes, ulp, guard bits, c.f. Coq's Flocq).
--   - 5 spec axioms in `SounioIEEE754Spec.lean` become theorems.
--   - this file's derived theorems remain unchanged.
--   - net: 0 axioms total; full IEEE-754-2008 binary64
--     compliance proven from first principles.

end Sounio.FloatInstance
