-- formal/lean4/SounioRealCauchyPartial.lean
import SounioOrderedCarrier
import SounioRealCauchy

/-!
# Sounio.RealCauchy.Partial — partial Stage 3a-Cauchy progress

This file makes incremental progress on closing the
`OrderedCarrierObligation_RealCauchy` from
`SounioRealCauchy.lean`. It proves the **easy half** of the
obligation (the `mul_le_mul_of_nonneg_right` law over the
pointwise eventual order, given Cauchy witnesses for the
products) and provides supporting infrastructure (`max_prefix`,
`TwoSidedBound`).

The **hard half** (`MulPreservesCauchy`: the proof that
pointwise products of Cauchy sequences are themselves Cauchy)
is left to a separate `SounioRealCauchyMul.lean` milestone.
That proof is ~150 LOC of ε-N analysis with a uniform
boundedness lemma; without Mathlib's ordered-field automation
it requires careful Rat case analysis we don't ship in this
commit.

## Honest scope statement

This file ships:
  - `max_prefix : (Nat → Rat) → Nat → Rat` recursive max over
    a finite prefix (reserved for the deferred boundedness
    milestone).
  - `TwoSidedBound : Rat → Rat → Prop` (avoids `Rat.abs`).
  - `mul_le_mul_of_nonneg_right_pointwise` (proven): the
    `OrderedCarrier`-shape law over `≤_p` given Cauchy
    witnesses for the products.
  - `orderedCarrierObligation_RealCauchy_holds_given_mulPres`
    (proven): `OrderedCarrierObligation_RealCauchy` follows
    from `MulPreservesCauchy`.
  - `le_p_implies_le_eps` (proven): the canonical
    `≤_p → ≤_ε` lift.

This file does NOT ship:
  - `isCauchy_bounded` (the uniform boundedness lemma).
  - `mul_seq_isCauchy` (the Cauchy-mul preservation).
  - The full `instance : OrderedCarrier SounioRealCauchy`.

The deferred pieces are documented in
`Sounio.RealCauchy.SounioRealCauchy.OrderedCarrierObligation_RealCauchy`
(in `SounioRealCauchy.lean`) and will be closed by a future
`SounioRealCauchyMul.lean` milestone.

## Math-review record

- Pre-impl thesis (`/tmp/cauchy_complete_thesis.md`): Grok 4.1
  8/8 OK. The proof sketches were approved; the full ε-N
  analysis was deferred for the reasons above.
- Post-impl: see commit message.

## Status

`lake build SounioRealCauchyComplete` builds clean. No `axiom`,
no `sorry`, no Mathlib import.
-/

namespace Sounio.RealCauchy.Partial

open Sounio.RealCauchy
open Sounio.RealCauchy.SounioRealCauchy

-- ================================================================
-- §1. Supporting infrastructure (not the deferred lemmas).
-- ================================================================

/-- The maximum of `f 0, f 1, ..., f N` over `Rat`. Reserved
    for the deferred boundedness lemma (a future
    `SounioRealCauchyMul.lean` milestone).

    Defined here for forward-compatibility — the deferred
    milestone will use it. Not used by any theorem in this
    file. -/
def max_prefix (f : Nat → Rat) : Nat → Rat
  | 0 => f 0
  | n + 1 => if f (n + 1) ≤ max_prefix f n
             then max_prefix f n
             else f (n + 1)

/-- Two-sided bound for a `Rat` value: `−K ≤ x ≤ K`. Reserved
    for the deferred boundedness lemma. We avoid `Rat.abs` to
    keep the file independent of the absolute-value lemma chain. -/
def TwoSidedBound (x K : Rat) : Prop :=
  x ≤ K ∧ (0 - K) ≤ x

-- ================================================================
-- §2. The easy half: mul_le_mul_of_nonneg_right_pointwise.
-- ================================================================

/-- **Pointwise `mul_le_mul_of_nonneg_right`** over the
    pointwise eventual order on `SounioRealCauchy`.

    Given:
      - `a ≤_p b` (∃ N₁, ∀ n ≥ N₁, a.seq n ≤ b.seq n)
      - `zero ≤_p c` (∃ N₂, ∀ n ≥ N₂, 0 ≤ c.seq n)
      - Cauchy witnesses for the products `a · c`, `b · c`
        (supplied as hypotheses; provided by the deferred
        `MulPreservesCauchy` lemma)
    Then `a · c ≤_p b · c`.

    The proof is direct via `Rat.mul_le_mul_of_nonneg_right`
    at each `n ≥ max(N₁, N₂)`. This is the **easy half** of
    the OrderedCarrier instance; the hard half (proving the
    Cauchy witnesses themselves, i.e., `MulPreservesCauchy`)
    is deferred to a separate milestone. -/
theorem mul_le_mul_of_nonneg_right_pointwise
    (a b c : SounioRealCauchy)
    (hab : a ≤ b) (hc : zero ≤ c)
    (h_mul_a_c : IsCauchy (mul_seq a c))
    (h_mul_b_c : IsCauchy (mul_seq b c))
    : (⟨mul_seq a c, h_mul_a_c⟩ : SounioRealCauchy)
        ≤ ⟨mul_seq b c, h_mul_b_c⟩ := by
  obtain ⟨N1, h1⟩ := hab
  obtain ⟨N2, h2⟩ := hc
  refine ⟨max N1 N2, ?_⟩
  intro n hn
  have hn1 : N1 ≤ n :=
    Nat.le_trans (Nat.le_max_left N1 N2) hn
  have hn2 : N2 ≤ n :=
    Nat.le_trans (Nat.le_max_right N1 N2) hn
  have h_a_le_b : a.seq n ≤ b.seq n := h1 n hn1
  -- hc : zero ≤ c, where zero.seq n = 0 (constant from ofRat 0)
  -- So h2 n hn2 : zero.seq n ≤ c.seq n, i.e., 0 ≤ c.seq n
  have h_0_le_c : (0 : Rat) ≤ c.seq n := by
    have h_eq : zero.seq n = 0 := by
      show (Sounio.RealCauchy.SounioRealCauchy.ofRat 0).seq n = 0
      rfl
    have := h2 n hn2
    rw [h_eq] at this
    exact this
  -- Goal: (mul_seq a c).seq n ≤ (mul_seq b c).seq n
  -- mul_seq a c = fun n => a.seq n * c.seq n
  show a.seq n * c.seq n ≤ b.seq n * c.seq n
  exact Rat.mul_le_mul_of_nonneg_right h_a_le_b h_0_le_c

-- ================================================================
-- §3. OrderedCarrier modulo MulPreservesCauchy.
-- ================================================================

/-- The full `OrderedCarrierObligation_RealCauchy` is
    discharged given `MulPreservesCauchy`.

    This combines both conjuncts:
      - first: `MulPreservesCauchy` is exactly the hypothesis;
      - second: the `mul_le_mul_of_nonneg_right`-shape
        obligation is discharged by §2. -/
theorem orderedCarrierObligation_RealCauchy_holds_given_mulPres
    (h_mul_pres : MulPreservesCauchy) :
    OrderedCarrierObligation_RealCauchy := by
  refine ⟨h_mul_pres, ?_⟩
  intro a b c hab hc h_a_c h_b_c
  exact mul_le_mul_of_nonneg_right_pointwise a b c hab hc h_a_c h_b_c

-- ================================================================
-- §4. ≤_p → ≤_ε lift (optional documentation).
-- ================================================================

/-- The pointwise eventual order `≤_p` implies the
    canonical ε-quantified order `≤_ε` on Cauchy sequences.

    Intuition: if `∃ N, ∀ n ≥ N, a.seq n ≤ b.seq n`, then for
    any `ε > 0`, the same `N` witnesses `a.seq n ≤ b.seq n + ε`
    (since `b.seq n ≤ b.seq n + ε` for `0 ≤ ε`).

    This documents the relationship between Sounio's working
    pointwise order and the canonical real-line order. The
    structural Sounio theorems use only `≤_p` properties
    (`le_trans`, `mul_le_mul_of_nonneg_right`), so all
    theorems lift to `≤_ε` via this map. -/
theorem le_p_implies_le_eps (a b : SounioRealCauchy) (h : a ≤ b) :
    ∀ ε : Rat, 0 < ε →
      ∃ N : Nat, ∀ n : Nat, N ≤ n → a.seq n ≤ b.seq n + ε := by
  intro ε hε
  obtain ⟨N, hN⟩ := h
  refine ⟨N, ?_⟩
  intro n hn
  have h_a_le_b : a.seq n ≤ b.seq n := hN n hn
  -- b.seq n ≤ b.seq n + ε since 0 ≤ ε
  have h_b_le_b_eps : b.seq n ≤ b.seq n + ε := by
    have h_step : b.seq n + 0 ≤ b.seq n + ε :=
      (Rat.add_le_add_left
        (a := 0) (b := ε) (c := b.seq n)).mpr (Rat.le_of_lt hε)
    have h_eq : b.seq n + 0 = b.seq n := Rat.add_zero (b.seq n)
    rw [h_eq] at h_step
    exact h_step
  exact Rat.le_trans h_a_le_b h_b_le_b_eps

end Sounio.RealCauchy.Partial
