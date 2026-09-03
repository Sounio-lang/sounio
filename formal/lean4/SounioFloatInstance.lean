-- formal/lean4/SounioFloatInstance.lean
import SounioFloatBounded
import SounioIEEE754Spec

/-!
# Sounio.FloatInstance — `BoundedOrderedCarrier Float` (Stage 3b, Route C, Phase 1.5)

This file is the **Route C interim** Float instance for the
`BoundedOrderedCarrier` typeclass from `SounioFloatBounded.lean`.

## Phase 1.5 status (2026-05-02)

This file completes the dual-representation refactor:

  - **Phase 0** (committed `d0453156`): 4 typeclass-shape axioms
    (`Float.le_trans`, `Float.zero_le_zero`,
    `Float.mul_le_mul_of_nonneg_right_bounded`,
    `Float.add_le_add_right_bounded`) backing the
    `BoundedOrderedCarrier Float` instance directly.

  - **Phase 1** (committed in approx-effect merge): canonical
    5-axiom IEEE-754 spec extracted to `SounioIEEE754Spec.lean`
    (Higham 2002 §2.1 basic-operation model). Two of the four
    typeclass methods derived as **theorems** from the spec:
      - `Float.le_trans_from_spec` (derived from
        `Float.toRat_le_iff_finite` + `Rat.le_trans`)
      - `Float.zero_le_zero_from_spec` (derived from
        `Float.toRat_le_iff_finite` + `Rat.le_refl`)

  - **Phase 1.5** (this commit, 2026-05-02): the remaining two
    typeclass methods are now derived as **conditional**
    theorems from the spec (~250 LOC of Rat algebra):
      - `Float.mul_le_mul_of_nonneg_right_bounded_from_spec`
      - `Float.add_le_add_right_bounded_from_spec`
    Both verified via `#print axioms` to depend only on the 5
    IEEE-754 spec axioms + Lean core (no typeclass-shape axioms).

    The 4 typeclass-shape axioms in §1 are **retained** (not
    deleted as the original Phase 1 §6 promised) — see §6
    below for the honest scope correction.

  - **Phase 2** (long-term, ~5000 LOC): Route B in-tree
    IEEE-754 binary64 model formalisation. The 5 spec axioms
    in `SounioIEEE754Spec.lean` become theorems; the 4 typeclass-
    shape axioms in §1 here become theorems too (via NaN/Inf
    model). This file's spec-derived theorems remain unchanged.

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

  - **Phase 1 pre-impl thesis** (`/tmp/ieee754_spec_thesis.md`,
    2026-05-01): Grok 4.1 5/9 OK + 3 WRONG (ε_machine = 2⁻⁵³
    not 2⁻⁵²; Higham §2.1 not §2.5; cookbook recursive/coarse)
    + 1 OVERREACH.
  - **Phase 1 post-impl**: NO_FINDINGS 7/7 OK.
  - **Phase 1.5 pre-impl thesis v1**
    (`/tmp/ieee754_phase15_thesis.md`, 2026-05-02): Grok 4.1
    CAUGHT_BUG ×2 + 1 OVERREACH + 5 OK. Bug: cookbook third
    term referenced exact Rat product `β·γ` but `add_rne_bound`
    operates on the rounded `(b·c).toRat`; `u·|β·γ|` symbolic
    gap. Fix: use `(b·c).toRat + ε` (rounded operand).
  - **Phase 1.5 pre-impl thesis v2** (corrected, 2026-05-02):
    Grok 4.1 NO_FINDINGS 9/9 OK. v2 cookbook is tight (no
    symbolic gap), 14-step mul + 12-step add proof structures
    derive symbolically from spec axioms + Lean core.
  - **Phase 1.5 post-impl**: see commit message.

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

These remain in Phase 1.5 to back the BoundedOrderedCarrier
instance over arbitrary Float (NaN/Inf/subnormal). Each has a
strictly tighter `*_from_spec` companion theorem proven from
the 5 IEEE-754 axioms (see §2 + §3). Full axiom elimination
requires Phase 2 (in-tree NaN/Inf model). See §6 for the honest
roadmap.

## User responsibility (cookbook formula, Phase 1.5 v2)

When using `BoundedOrderedCarrier Float` methods, the user
MUST supply `eps_inf` satisfying the **tight** cookbook
(equivalent to ~3·u·max coarse form, but stated as the exact
sum the proof chain consumes):

    u·|a.toRat · c.toRat|
    + u·|b.toRat · c.toRat|
    + u·|(b·c).toRat + eps_inf.toRat|        -- rounded operand
    ≤ eps_inf.toRat

where:
  - `u = 2⁻⁵³ ≈ 1.11 × 10⁻¹⁶` (binary64 unit roundoff,
    Higham §2.4)
  - Each term covers one rounding step: `mul_rne_bound` of
    `a · c`, `mul_rne_bound` of `b · c`, and `add_rne_bound` of
    `(b · c) + eps_inf` itself. The third term references the
    **rounded** `(b·c).toRat` (not the exact Rat product
    `b.toRat · c.toRat`), matching the operand of
    `add_rne_bound`. Math-review (Grok 4.1, 2026-05-02) caught
    and corrected this in the pre-impl thesis (CAUGHT_BUG).

For clinical PK regimes (operands in `[10⁻², 10⁴]`):

    max-magnitude of products ≈ 10⁸
    eps_inf ≥ 3 · 1.11 × 10⁻¹⁶ · 10⁸ ≈ 3.3 × 10⁻⁸

For the Sounio Cmin band (width ~3 mg/L), the rounding-induced
inflation is `~3 × 10⁻⁸ mg/L` — eight orders of magnitude
below the trough/MIC clinical threshold (which is ≥ 1 mg/L).

## Status (Phase 1.5)

Five-axiom IEEE-754 spec + four typeclass-shape axioms +
**four** spec-derived theorems (le_trans, zero_le_zero from
Phase 1; mul, add from Phase 1.5) + the
`BoundedOrderedCarrier Float` instance + a demonstration
theorem build cleanly under `lake build SounioFloatInstance`.
No `sorry`, no Mathlib import.

The four `*_from_spec` theorems are verified via `#print axioms`
to depend only on the 5 IEEE-754 axioms (`Float.toRat`,
`Float.IsFiniteNormal`, `Float.toRat_le_iff_finite`,
`Float.mul_rne_bound`, `Float.add_rne_bound`) plus Lean core
builtins (`propext`, `Classical.choice`, `Quot.sound`). No
typeclass-shape axiom from §1 is used in any spec-derived
theorem.
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
    inflation.

    Phase 1.5 (this commit) provides
    `Float.mul_le_mul_of_nonneg_right_bounded_from_spec` (§3),
    a strictly tighter conditional version derived from the
    IEEE-754 spec under finite-normal hypotheses + a per-call
    cookbook. This unconditional axiom is RETAINED to back the
    typeclass instance over arbitrary Float (including non-finite
    operands not modelled by the spec).

    Source: Higham 2002 §2.1 eq. (2.4). -/
axiom Float.mul_le_mul_of_nonneg_right_bounded :
  ∀ {a b : Float} (c eps_inf : Float),
    a ≤ b → 0.0 ≤ c → 0.0 ≤ eps_inf →
    a * c ≤ (b * c) + eps_inf

/-- IEEE-754 right-addition monotonicity with bounded inflation.

    Phase 1.5 (this commit) provides
    `Float.add_le_add_right_bounded_from_spec` (§3), a strictly
    tighter conditional version. This unconditional axiom is
    RETAINED for the typeclass instance over arbitrary Float. -/
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
-- §3. Phase 1.5 spec-derived mul/add (conditional theorems).
-- ================================================================
--
-- Phase 1.5 (this commit) completes the dual representation:
-- the remaining two BoundedOrderedCarrier methods (mul/add) are
-- now derived as **conditional theorems** from the IEEE-754 spec.
-- The four typeclass-shape axioms in §1 are RETAINED to back the
-- typeclass instance over arbitrary Float (including non-finite
-- operands not modelled by the spec). Full axiom elimination
-- requires Phase 2 (NaN/Inf model, Flocq-style).

-- ----------------------------------------------------------------
-- §3.0. Rat micro-lemmas (Mathlib-free).
-- ----------------------------------------------------------------

/-- Local Rat lemma: `x - y ≤ z ↔ x ≤ y + z`.

    Lean 4 core ships `Rat.add_le_iff_le_sub : a + b ≤ c ↔ a ≤ c - b`
    but not the symmetric subtraction form. This 5-line helper
    derives it from `Rat.sub_add_cancel`, `Rat.add_le_add_right`,
    and `Rat.add_le_add_left`. -/
private theorem rat_sub_le_iff_add (x y z : Rat) :
    x - y ≤ z ↔ x ≤ y + z := by
  constructor
  · intro h
    have h1 : (x - y) + y ≤ z + y := (Rat.add_le_add_right).mpr h
    rw [Rat.sub_add_cancel, Rat.add_comm] at h1
    exact h1
  · intro h
    have h1 : (-y) + x ≤ (-y) + (y + z) := (Rat.add_le_add_left).mpr h
    rw [show (-y) + (y + z) = z from by
          rw [← Rat.add_assoc, Rat.neg_add_cancel, Rat.zero_add]] at h1
    rw [Rat.sub_eq_add_neg, Rat.add_comm]
    exact h1

/-- Local Rat lemma: `0 ≤ rat_abs q` for any `q : Rat`.

    The spec's `rat_abs` is encoded via `if 0 ≤ q then q else 0 - q`
    rather than `Rat.abs`, so the standard nonnegativity follows
    from a case split + `Rat.le_iff_sub_nonneg`. -/
private theorem rat_abs_nonneg (q : Rat) : 0 ≤ Sounio.IEEE754.rat_abs q := by
  unfold Sounio.IEEE754.rat_abs
  by_cases h : 0 ≤ q
  · simp [h]
  · simp [h]
    have h1 : q < 0 := Rat.not_le.mp h
    have h3 : q ≤ 0 := Rat.le_of_lt h1
    exact (Rat.le_iff_sub_nonneg q 0).mp h3

-- ----------------------------------------------------------------
-- §3.1. Spec-derived mul (conditional).
-- ----------------------------------------------------------------

/-- Phase 1.5 derivation: the bounded-mul law follows from the
    IEEE-754 spec under finite-normal hypotheses + a per-call
    cookbook (the third typeclass-shape axiom in §1 is the
    unconditional weakening that backs the typeclass instance
    over arbitrary Float).

    The cookbook is **tight** — exactly the inequality the proof
    chain produces from chaining `mul_rne_bound` (twice) with
    `add_rne_bound` for `(b·c) + eps_inf`:

        u·|a.toRat · c.toRat|
        + u·|b.toRat · c.toRat|
        + u·|(b·c).toRat + eps_inf.toRat|
        ≤ eps_inf.toRat

    Note the **third** term references the **rounded** float-mul
    `(b·c).toRat` (not the exact Rat product `b.toRat · c.toRat`),
    matching the operand of `add_rne_bound`. Math-review (Grok 4.1,
    2026-05-02) caught and corrected this in the pre-impl thesis
    (CAUGHT_BUG): v1 used the exact product, leaving a `u·|β·γ|`
    symbolic gap; v2 (here) uses the rounded operand and the chain
    closes with no slack.

    For clinical PK regimes (operands `[10⁻², 10⁴]`), all three
    terms are `≈ 10⁻⁸`, so the cookbook is satisfied by
    `eps_inf ≈ 3.3 × 10⁻⁸` — eight orders of magnitude below the
    trough/MIC clinical threshold.

    The Rat-level monotonicity hypotheses (`hab_rat`, `hc_nn_rat`)
    avoid needing a 6th spec axiom `Float.toRat 0.0 = 0`. Users
    lift Float-level `a ≤ b` and `0.0 ≤ c` via
    `Float.toRat_le_iff_finite` plus the trivial bridge.

    Math-review record:
      [OK] Mul proof structure (14 steps)
        Each step derives symbolically from prior + spec axioms
        (`mul_rne_bound`/.1/.2, `add_rne_bound`/.1) + Rat core
        (`mul_le_mul_of_nonneg_right`, `add_le_add_right`,
        `le_trans`); v2 uses Bc (rounded) matching `add_rne_bound`
        operand. — Grok 4.1, 2026-05-02 -/
theorem Float.mul_le_mul_of_nonneg_right_bounded_from_spec
    {a b c eps_inf : Float}
    (ha     : Sounio.IEEE754.Float.IsFiniteNormal a)
    (hb     : Sounio.IEEE754.Float.IsFiniteNormal b)
    (hc     : Sounio.IEEE754.Float.IsFiniteNormal c)
    (heps   : Sounio.IEEE754.Float.IsFiniteNormal eps_inf)
    (hac    : Sounio.IEEE754.Float.IsFiniteNormal (a * c))
    (hbc    : Sounio.IEEE754.Float.IsFiniteNormal (b * c))
    (hbceps : Sounio.IEEE754.Float.IsFiniteNormal ((b * c) + eps_inf))
    (hab_rat   : Sounio.IEEE754.Float.toRat a ≤ Sounio.IEEE754.Float.toRat b)
    (hc_nn_rat : (0 : Rat) ≤ Sounio.IEEE754.Float.toRat c)
    (cookbook  :
       Sounio.IEEE754.unit_roundoff
         * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                    * Sounio.IEEE754.Float.toRat c)
       + Sounio.IEEE754.unit_roundoff
         * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                    * Sounio.IEEE754.Float.toRat c)
       + Sounio.IEEE754.unit_roundoff
         * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat (b * c)
                                    + Sounio.IEEE754.Float.toRat eps_inf)
       ≤ Sounio.IEEE754.Float.toRat eps_inf) :
    a * c ≤ (b * c) + eps_inf := by
  -- Step 1: α·γ ≤ β·γ
  have step1 :
      Sounio.IEEE754.Float.toRat a * Sounio.IEEE754.Float.toRat c
        ≤ Sounio.IEEE754.Float.toRat b * Sounio.IEEE754.Float.toRat c :=
    Rat.mul_le_mul_of_nonneg_right hab_rat hc_nn_rat
  -- Steps 2-3: (a*c).toRat ≤ α·γ + u·|α·γ|
  have hmul_ac := Sounio.IEEE754.Float.mul_rne_bound ha hc hac
  have step3 :
      Sounio.IEEE754.Float.toRat (a * c)
        ≤ Sounio.IEEE754.Float.toRat a * Sounio.IEEE754.Float.toRat c
            + Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                         * Sounio.IEEE754.Float.toRat c) :=
    (rat_sub_le_iff_add _ _ _).mp hmul_ac.1
  -- Step 4: (a*c).toRat ≤ β·γ + u·|α·γ|
  have step4 :
      Sounio.IEEE754.Float.toRat (a * c)
        ≤ Sounio.IEEE754.Float.toRat b * Sounio.IEEE754.Float.toRat c
            + Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                         * Sounio.IEEE754.Float.toRat c) :=
    Rat.le_trans step3 ((Rat.add_le_add_right).mpr step1)
  -- Steps 5-6: β·γ ≤ (b*c).toRat + u·|β·γ|
  have hmul_bc := Sounio.IEEE754.Float.mul_rne_bound hb hc hbc
  have step6 :
      Sounio.IEEE754.Float.toRat b * Sounio.IEEE754.Float.toRat c
        ≤ Sounio.IEEE754.Float.toRat (b * c)
            + Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                         * Sounio.IEEE754.Float.toRat c) :=
    (rat_sub_le_iff_add _ _ _).mp hmul_bc.2
  -- Step 7: (a*c).toRat ≤ (b*c).toRat + u·|β·γ| + u·|α·γ|
  have step7 :
      Sounio.IEEE754.Float.toRat (a * c)
        ≤ Sounio.IEEE754.Float.toRat (b * c)
            + Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                         * Sounio.IEEE754.Float.toRat c)
            + Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                         * Sounio.IEEE754.Float.toRat c) :=
    Rat.le_trans step4 ((Rat.add_le_add_right).mpr step6)
  -- Steps 8-9: Bc + ε - u·|Bc + ε| ≤ Bceps  (Bc = (b·c).toRat)
  have hadd := Sounio.IEEE754.Float.add_rne_bound hbc heps hbceps
  have step9a :
      Sounio.IEEE754.Float.toRat (b * c) + Sounio.IEEE754.Float.toRat eps_inf
        ≤ Sounio.IEEE754.Float.toRat ((b * c) + eps_inf)
            + Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat (b * c)
                                         + Sounio.IEEE754.Float.toRat eps_inf) :=
    (rat_sub_le_iff_add _ _ _).mp hadd.2
  have step9 :
      Sounio.IEEE754.Float.toRat (b * c) + Sounio.IEEE754.Float.toRat eps_inf
        - Sounio.IEEE754.unit_roundoff
          * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat (b * c)
                                     + Sounio.IEEE754.Float.toRat eps_inf)
        ≤ Sounio.IEEE754.Float.toRat ((b * c) + eps_inf) := by
    apply (rat_sub_le_iff_add _ _ _).mpr
    have hcomm :
        Sounio.IEEE754.unit_roundoff
            * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat (b * c)
                                       + Sounio.IEEE754.Float.toRat eps_inf)
          + Sounio.IEEE754.Float.toRat ((b * c) + eps_inf)
        = Sounio.IEEE754.Float.toRat ((b * c) + eps_inf)
            + Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat (b * c)
                                         + Sounio.IEEE754.Float.toRat eps_inf) :=
      Rat.add_comm _ _
    rw [hcomm]
    exact step9a
  -- Cookbook rearranged: u·|α·γ| + u·|β·γ| ≤ ε - u·|Bc + ε|
  have h_cb_sub :
      Sounio.IEEE754.unit_roundoff
        * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                   * Sounio.IEEE754.Float.toRat c)
      + Sounio.IEEE754.unit_roundoff
        * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                   * Sounio.IEEE754.Float.toRat c)
      ≤ Sounio.IEEE754.Float.toRat eps_inf
          - Sounio.IEEE754.unit_roundoff
            * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat (b * c)
                                       + Sounio.IEEE754.Float.toRat eps_inf) :=
    (Rat.add_le_iff_le_sub).mp cookbook
  -- Combine: Bc + (u·|α·γ| + u·|β·γ|) ≤ Bc + (ε - u·|Bc+ε|) = (Bc+ε) - u·|Bc+ε|
  have h12_lhs :
      Sounio.IEEE754.Float.toRat (b * c)
        + (Sounio.IEEE754.unit_roundoff
            * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                       * Sounio.IEEE754.Float.toRat c)
           + Sounio.IEEE754.unit_roundoff
             * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                        * Sounio.IEEE754.Float.toRat c))
        ≤ Sounio.IEEE754.Float.toRat (b * c)
            + (Sounio.IEEE754.Float.toRat eps_inf
               - Sounio.IEEE754.unit_roundoff
                 * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat (b * c)
                                            + Sounio.IEEE754.Float.toRat eps_inf)) :=
    (Rat.add_le_add_left).mpr h_cb_sub
  have step12 :
      Sounio.IEEE754.Float.toRat (b * c)
        + Sounio.IEEE754.unit_roundoff
          * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                     * Sounio.IEEE754.Float.toRat c)
        + Sounio.IEEE754.unit_roundoff
          * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                     * Sounio.IEEE754.Float.toRat c)
        ≤ Sounio.IEEE754.Float.toRat ((b * c) + eps_inf) := by
    have h_lhs_assoc :
        Sounio.IEEE754.Float.toRat (b * c)
          + (Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                         * Sounio.IEEE754.Float.toRat c)
             + Sounio.IEEE754.unit_roundoff
               * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                          * Sounio.IEEE754.Float.toRat c))
        = Sounio.IEEE754.Float.toRat (b * c)
          + Sounio.IEEE754.unit_roundoff
            * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                       * Sounio.IEEE754.Float.toRat c)
          + Sounio.IEEE754.unit_roundoff
            * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                       * Sounio.IEEE754.Float.toRat c) :=
      (Rat.add_assoc _ _ _).symm
    have h_rhs_assoc :
        Sounio.IEEE754.Float.toRat (b * c)
          + (Sounio.IEEE754.Float.toRat eps_inf
             - Sounio.IEEE754.unit_roundoff
               * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat (b * c)
                                          + Sounio.IEEE754.Float.toRat eps_inf))
        = Sounio.IEEE754.Float.toRat (b * c) + Sounio.IEEE754.Float.toRat eps_inf
            - Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat (b * c)
                                         + Sounio.IEEE754.Float.toRat eps_inf) := by
      rw [Rat.sub_eq_add_neg, Rat.sub_eq_add_neg, ← Rat.add_assoc]
    rw [h_lhs_assoc, h_rhs_assoc] at h12_lhs
    exact Rat.le_trans h12_lhs step9
  -- Step 13: align step7 with step12 (commute u·|α·γ| ↔ u·|β·γ|)
  have step7' :
      Sounio.IEEE754.Float.toRat (a * c)
        ≤ Sounio.IEEE754.Float.toRat (b * c)
            + Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                         * Sounio.IEEE754.Float.toRat c)
            + Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                         * Sounio.IEEE754.Float.toRat c) := by
    have heq :
        Sounio.IEEE754.Float.toRat (b * c)
          + Sounio.IEEE754.unit_roundoff
            * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                       * Sounio.IEEE754.Float.toRat c)
          + Sounio.IEEE754.unit_roundoff
            * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                       * Sounio.IEEE754.Float.toRat c)
        = Sounio.IEEE754.Float.toRat (b * c)
          + Sounio.IEEE754.unit_roundoff
            * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                       * Sounio.IEEE754.Float.toRat c)
          + Sounio.IEEE754.unit_roundoff
            * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                       * Sounio.IEEE754.Float.toRat c) := by
      rw [Rat.add_assoc,
          Rat.add_comm (Sounio.IEEE754.unit_roundoff
                          * Sounio.IEEE754.rat_abs
                              (Sounio.IEEE754.Float.toRat b
                                * Sounio.IEEE754.Float.toRat c)),
          ← Rat.add_assoc]
    rw [heq] at step7
    exact step7
  -- Step 14: lift via toRat_le_iff_finite
  exact (Sounio.IEEE754.Float.toRat_le_iff_finite hac hbceps).mpr
          (Rat.le_trans step7' step12)

-- ----------------------------------------------------------------
-- §3.2. Spec-derived add (conditional).
-- ----------------------------------------------------------------

/-- Phase 1.5 derivation: bounded-add law from the IEEE-754 spec
    under finite-normal hypotheses. Symmetric to mul, but no
    `0 ≤ c` premise is needed (Rat addition is unconditionally
    monotone on the right).

    Cookbook (tight, v2 corrected per Grok review 2026-05-02):

        u·|a.toRat + c.toRat|
        + u·|b.toRat + c.toRat|
        + u·|(b+c).toRat + eps_inf.toRat|
        ≤ eps_inf.toRat

    Same `(b+c).toRat` (rounded) third-term shape as mul. -/
theorem Float.add_le_add_right_bounded_from_spec
    {a b c eps_inf : Float}
    (ha     : Sounio.IEEE754.Float.IsFiniteNormal a)
    (hb     : Sounio.IEEE754.Float.IsFiniteNormal b)
    (hc     : Sounio.IEEE754.Float.IsFiniteNormal c)
    (heps   : Sounio.IEEE754.Float.IsFiniteNormal eps_inf)
    (hac    : Sounio.IEEE754.Float.IsFiniteNormal (a + c))
    (hbc    : Sounio.IEEE754.Float.IsFiniteNormal (b + c))
    (hbceps : Sounio.IEEE754.Float.IsFiniteNormal ((b + c) + eps_inf))
    (hab_rat   : Sounio.IEEE754.Float.toRat a ≤ Sounio.IEEE754.Float.toRat b)
    (cookbook  :
       Sounio.IEEE754.unit_roundoff
         * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                    + Sounio.IEEE754.Float.toRat c)
       + Sounio.IEEE754.unit_roundoff
         * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                    + Sounio.IEEE754.Float.toRat c)
       + Sounio.IEEE754.unit_roundoff
         * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat (b + c)
                                    + Sounio.IEEE754.Float.toRat eps_inf)
       ≤ Sounio.IEEE754.Float.toRat eps_inf) :
    a + c ≤ (b + c) + eps_inf := by
  -- Step 1: α + γ ≤ β + γ
  have step1 :
      Sounio.IEEE754.Float.toRat a + Sounio.IEEE754.Float.toRat c
        ≤ Sounio.IEEE754.Float.toRat b + Sounio.IEEE754.Float.toRat c :=
    (Rat.add_le_add_right).mpr hab_rat
  -- Steps 2-3: (a+c).toRat ≤ (α + γ) + u·|α + γ|
  have hadd_ac := Sounio.IEEE754.Float.add_rne_bound ha hc hac
  have step3 :
      Sounio.IEEE754.Float.toRat (a + c)
        ≤ Sounio.IEEE754.Float.toRat a + Sounio.IEEE754.Float.toRat c
            + Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                         + Sounio.IEEE754.Float.toRat c) :=
    (rat_sub_le_iff_add _ _ _).mp hadd_ac.1
  -- Step 4: (a+c).toRat ≤ (β + γ) + u·|α + γ|
  have step4 :
      Sounio.IEEE754.Float.toRat (a + c)
        ≤ Sounio.IEEE754.Float.toRat b + Sounio.IEEE754.Float.toRat c
            + Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                         + Sounio.IEEE754.Float.toRat c) :=
    Rat.le_trans step3 ((Rat.add_le_add_right).mpr step1)
  -- Steps 5-6: β + γ ≤ (b+c).toRat + u·|β + γ|
  have hadd_bc := Sounio.IEEE754.Float.add_rne_bound hb hc hbc
  have step6 :
      Sounio.IEEE754.Float.toRat b + Sounio.IEEE754.Float.toRat c
        ≤ Sounio.IEEE754.Float.toRat (b + c)
            + Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                         + Sounio.IEEE754.Float.toRat c) :=
    (rat_sub_le_iff_add _ _ _).mp hadd_bc.2
  -- Step 7: (a+c).toRat ≤ (b+c).toRat + u·|β + γ| + u·|α + γ|
  have step7 :
      Sounio.IEEE754.Float.toRat (a + c)
        ≤ Sounio.IEEE754.Float.toRat (b + c)
            + Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                         + Sounio.IEEE754.Float.toRat c)
            + Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                         + Sounio.IEEE754.Float.toRat c) :=
    Rat.le_trans step4 ((Rat.add_le_add_right).mpr step6)
  -- Steps 8-9: (b+c).toRat + ε - u·|(b+c).toRat + ε| ≤ Bceps
  have hadd_bceps := Sounio.IEEE754.Float.add_rne_bound hbc heps hbceps
  have step9a :
      Sounio.IEEE754.Float.toRat (b + c) + Sounio.IEEE754.Float.toRat eps_inf
        ≤ Sounio.IEEE754.Float.toRat ((b + c) + eps_inf)
            + Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat (b + c)
                                         + Sounio.IEEE754.Float.toRat eps_inf) :=
    (rat_sub_le_iff_add _ _ _).mp hadd_bceps.2
  have step9 :
      Sounio.IEEE754.Float.toRat (b + c) + Sounio.IEEE754.Float.toRat eps_inf
        - Sounio.IEEE754.unit_roundoff
          * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat (b + c)
                                     + Sounio.IEEE754.Float.toRat eps_inf)
        ≤ Sounio.IEEE754.Float.toRat ((b + c) + eps_inf) := by
    apply (rat_sub_le_iff_add _ _ _).mpr
    have hcomm :
        Sounio.IEEE754.unit_roundoff
            * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat (b + c)
                                       + Sounio.IEEE754.Float.toRat eps_inf)
          + Sounio.IEEE754.Float.toRat ((b + c) + eps_inf)
        = Sounio.IEEE754.Float.toRat ((b + c) + eps_inf)
            + Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat (b + c)
                                         + Sounio.IEEE754.Float.toRat eps_inf) :=
      Rat.add_comm _ _
    rw [hcomm]
    exact step9a
  -- Cookbook rearranged
  have h_cb_sub :
      Sounio.IEEE754.unit_roundoff
        * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                   + Sounio.IEEE754.Float.toRat c)
      + Sounio.IEEE754.unit_roundoff
        * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                   + Sounio.IEEE754.Float.toRat c)
      ≤ Sounio.IEEE754.Float.toRat eps_inf
          - Sounio.IEEE754.unit_roundoff
            * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat (b + c)
                                       + Sounio.IEEE754.Float.toRat eps_inf) :=
    (Rat.add_le_iff_le_sub).mp cookbook
  have h12_lhs :
      Sounio.IEEE754.Float.toRat (b + c)
        + (Sounio.IEEE754.unit_roundoff
            * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                       + Sounio.IEEE754.Float.toRat c)
           + Sounio.IEEE754.unit_roundoff
             * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                        + Sounio.IEEE754.Float.toRat c))
        ≤ Sounio.IEEE754.Float.toRat (b + c)
            + (Sounio.IEEE754.Float.toRat eps_inf
               - Sounio.IEEE754.unit_roundoff
                 * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat (b + c)
                                            + Sounio.IEEE754.Float.toRat eps_inf)) :=
    (Rat.add_le_add_left).mpr h_cb_sub
  have step12 :
      Sounio.IEEE754.Float.toRat (b + c)
        + Sounio.IEEE754.unit_roundoff
          * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                     + Sounio.IEEE754.Float.toRat c)
        + Sounio.IEEE754.unit_roundoff
          * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                     + Sounio.IEEE754.Float.toRat c)
        ≤ Sounio.IEEE754.Float.toRat ((b + c) + eps_inf) := by
    have h_lhs_assoc :
        Sounio.IEEE754.Float.toRat (b + c)
          + (Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                         + Sounio.IEEE754.Float.toRat c)
             + Sounio.IEEE754.unit_roundoff
               * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                          + Sounio.IEEE754.Float.toRat c))
        = Sounio.IEEE754.Float.toRat (b + c)
          + Sounio.IEEE754.unit_roundoff
            * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                       + Sounio.IEEE754.Float.toRat c)
          + Sounio.IEEE754.unit_roundoff
            * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                       + Sounio.IEEE754.Float.toRat c) :=
      (Rat.add_assoc _ _ _).symm
    have h_rhs_assoc :
        Sounio.IEEE754.Float.toRat (b + c)
          + (Sounio.IEEE754.Float.toRat eps_inf
             - Sounio.IEEE754.unit_roundoff
               * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat (b + c)
                                          + Sounio.IEEE754.Float.toRat eps_inf))
        = Sounio.IEEE754.Float.toRat (b + c) + Sounio.IEEE754.Float.toRat eps_inf
            - Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat (b + c)
                                         + Sounio.IEEE754.Float.toRat eps_inf) := by
      rw [Rat.sub_eq_add_neg, Rat.sub_eq_add_neg, ← Rat.add_assoc]
    rw [h_lhs_assoc, h_rhs_assoc] at h12_lhs
    exact Rat.le_trans h12_lhs step9
  -- Align step7 with step12 (commute u·|α + γ| ↔ u·|β + γ|)
  have step7' :
      Sounio.IEEE754.Float.toRat (a + c)
        ≤ Sounio.IEEE754.Float.toRat (b + c)
            + Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                         + Sounio.IEEE754.Float.toRat c)
            + Sounio.IEEE754.unit_roundoff
              * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                         + Sounio.IEEE754.Float.toRat c) := by
    have heq :
        Sounio.IEEE754.Float.toRat (b + c)
          + Sounio.IEEE754.unit_roundoff
            * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                       + Sounio.IEEE754.Float.toRat c)
          + Sounio.IEEE754.unit_roundoff
            * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                       + Sounio.IEEE754.Float.toRat c)
        = Sounio.IEEE754.Float.toRat (b + c)
          + Sounio.IEEE754.unit_roundoff
            * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat a
                                       + Sounio.IEEE754.Float.toRat c)
          + Sounio.IEEE754.unit_roundoff
            * Sounio.IEEE754.rat_abs (Sounio.IEEE754.Float.toRat b
                                       + Sounio.IEEE754.Float.toRat c) := by
      rw [Rat.add_assoc,
          Rat.add_comm (Sounio.IEEE754.unit_roundoff
                          * Sounio.IEEE754.rat_abs
                              (Sounio.IEEE754.Float.toRat b
                                + Sounio.IEEE754.Float.toRat c)),
          ← Rat.add_assoc]
    rw [heq] at step7
    exact step7
  exact (Sounio.IEEE754.Float.toRat_le_iff_finite hac hbceps).mpr
          (Rat.le_trans step7' step12)

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
-- §6. Roadmap (honest endpoint after Phase 1.5).
-- ================================================================
--
-- **Phase 1.5 status (this commit, 2026-05-02)**: COMPLETE.
--   The dual representation is closed. All four typeclass
--   methods now have **two** representations:
--     - Unconditional Phase 0 axioms in §1 (back the
--       BoundedOrderedCarrier instance over arbitrary Float).
--     - Conditional Phase 1.5 theorems in §2 + §3 (derived from
--       the IEEE-754 spec, with finite-normal hypotheses + tight
--       cookbook).
--   Net axioms in `formal/lean4/Sounio*.lean`:
--     - 5 IEEE-754 spec axioms (`SounioIEEE754Spec.lean`)
--     - 4 typeclass-shape axioms (this file, §1) — RETAINED
--   The spec-derived theorems use ONLY the 5 IEEE-754 axioms +
--   Lean core (verified via `#print axioms`).
--
-- **Why the 4 typeclass-shape axioms are RETAINED (honest scope
--  correction to the original Phase 1 plan)**:
--   The original §6 in Phase 1 promised "delete the 4 typeclass-
--   shape axioms in §1" at the end of Phase 1.5. That was
--   optimistic. The spec-derived theorems require finite-normal
--   witnesses for every Float subexpression (`a`, `b`, `c`,
--   `eps_inf`, `a*c`, `b*c`, `(b*c)+eps_inf`). The typeclass
--   instance, however, must hold over ARBITRARY Float — including
--   NaN, ±Inf, and subnormal operands not modelled by the 5
--   IEEE-754 axioms. Eliminating the typeclass-shape axioms
--   requires modelling NaN/Inf semantics (e.g., NaN ≤ NaN is
--   `True` in `Float.lt`'s decidable encoding but the spec axioms
--   say nothing about it).
--   Phase 1.5 endpoint: dual representation complete (4
--   unconditional axioms + 4 conditional theorems). Axiom
--   elimination → Phase 2 (in-tree NaN/Inf model, ~5000 LOC).
--
-- **Phase 2** (long-term, ~5000 LOC):
--   - In-tree IEEE-754 binary64 model formalisation (rounding
--     modes, ulp, guard bits, NaN/Inf totalisation, c.f. Coq's
--     Flocq).
--   - The 5 spec axioms in `SounioIEEE754Spec.lean` become
--     theorems.
--   - The 4 typeclass-shape axioms in §1 here become theorems
--     too (lifted from spec-derived theorems via NaN/Inf model
--     + total-order semantics for non-finite operands).
--   - This file's spec-derived theorems remain unchanged.
--   - Net: 0 axioms total in the formal/lean4/ tree; full
--     IEEE-754-2008 binary64 compliance proven from first
--     principles.

end Sounio.FloatInstance
