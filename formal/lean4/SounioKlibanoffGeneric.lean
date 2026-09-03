-- formal/lean4/SounioKlibanoffGeneric.lean
import SounioOrderedCarrier
import SounioFrechetGeneric

/-!
# Sounio.KlibanoffGeneric — Stage 2 lift of Klibanoff boundary theorems

This file lifts the structural boundary theorems from
`SounioKlibanoff.lean` (Stage 0, `Nat`-shadow with discrete
encoding) to the Stage 2 typeclass abstraction
`Sounio.OrderedCarrier α`.

## What lifts cleanly

The Klibanoff CARA-CE has *two* operational regimes:
  - **Boundary** cases (α = 0, α → ∞): closed-form linear /
    min functionals over the credal-set support points.
  - **Smooth** case (0 < α < ∞): involves `exp` and `log`,
    requires analytic structure beyond `OrderedCarrier`.

The boundary cases lift cleanly to `OrderedCarrier`. The smooth
case needs a Stage 3a-Cauchy or Mathlib `Real` instance (with
analysis), which is deferred per the Float-Real lift roadmap.

## Theorems mechanised

  - `kl_walley_ce_at_lambda_one_generic` — at λ = 1 with
    `s_lo ≤ μ_0`, the Walley-CE (α → ∞ limit) equals s_lo.
  - `kl_walley_ce_at_lambda_zero_generic` — at λ = 0 with
    `μ_0 ≤ s_hi`, the Walley-CE equals μ_0 (the math-review
    correction from the original thesis).
  - `kl_compose_inc_dec_generic` — Klibanoff composition with
    Fréchet for monotone-(↑x, ↓y) f.

## Min-typeclass requirement

The `min` operation isn't part of `OrderedCarrier`. We expose
the user's local `min` as a parameter with `min_le_left` and
`min_le_right` hypotheses (the carrier's own min lemmas).

For `Nat` / `Rat` / `SounioReal`, the canonical `min` is
available; the user supplies the relevant lemmas at instance
time. This pattern keeps the `OrderedCarrier` typeclass minimal.

## Status

All theorems below build cleanly under
`lake build SounioKlibanoffGeneric`. No `axiom`, no `sorry`, no
Mathlib import.
-/

namespace Sounio.KlibanoffGeneric

open Sounio
open Sounio.OrderedCarrier

-- ================================================================
-- §1. Walley-CE at λ = 1.
-- ================================================================

/-- Generic version of `kl_walley_ce_at_lambda_one`: at λ = 1
    with `s_lo ≤ μ_0`, the Walley-CE (taken as `min(μ_0, s_lo)`
    by the user-supplied min) equals s_lo.

    The proof reduces to the user's `min_eq_right_of_le` lemma
    (which is exactly `Nat.min_eq_right` / `Rat.min_eq_right`
    in the canonical instances). -/
theorem kl_walley_ce_at_lambda_one_generic
    {α : Type} [OrderedCarrier α]
    (min : α → α → α)
    (min_eq_right_of_le : ∀ (a b : α), a ≤ b → min b a = a)
    (μ_0 s_lo : α) (h : s_lo ≤ μ_0)
    : min μ_0 s_lo = s_lo :=
  min_eq_right_of_le s_lo μ_0 h

-- ================================================================
-- §2. Walley-CE at λ = 0.
-- ================================================================

/-- Generic version of `kl_walley_ce_at_lambda_zero`: at λ = 0
    with `μ_0 ≤ s_hi`, the Walley-CE (taken as `min(μ_0, s_hi)`)
    equals μ_0.

    This corresponds to the math-review correction from the
    original thesis (2026-05-01): when contamination has zero
    weight on s_lo, the minimum is over `{μ_0, s_hi}`, hence
    `μ_0` (since `μ_0 ≤ s_hi`). -/
theorem kl_walley_ce_at_lambda_zero_generic
    {α : Type} [OrderedCarrier α]
    (min : α → α → α)
    (min_eq_left_of_le : ∀ (a b : α), a ≤ b → min a b = a)
    (μ_0 s_hi : α) (h : μ_0 ≤ s_hi)
    : min μ_0 s_hi = μ_0 :=
  min_eq_left_of_le μ_0 s_hi h

-- ================================================================
-- §3. Klibanoff/Fréchet composition (generic).
-- ================================================================

/-- Composition obligation: for monotone-(↑x, ↓y) `f` and the
    second-order beliefs over two credal sets `(C_X, C_Y)`,
    Klibanoff CE_α at α = 0 (precise) reduces to a 9-term
    linear sum over the support corners. The minimum / maximum
    of the 9 corners is given by `f s_lo_x s_hi_y` /
    `f s_hi_x s_lo_y`.

    This is the carrier-generic statement of the Stage 0
    `kl_walley_compose_inc_dec_obligation`. -/
def kl_compose_inc_dec_obligation_generic
    (α : Type) [OrderedCarrier α] : Prop :=
  ∀ (f : α → α → α),
    (∀ x x' y, x ≤ x' → f x y ≤ f x' y) →
    (∀ x y y', y ≤ y' → f x y' ≤ f x y) →
    ∀ (μ_x s_lo_x s_hi_x μ_y s_lo_y s_hi_y : α),
    s_lo_x ≤ μ_x → μ_x ≤ s_hi_x →
    s_lo_y ≤ μ_y → μ_y ≤ s_hi_y →
    f s_lo_x s_hi_y ≤ f μ_x μ_y ∧ f μ_x μ_y ≤ f s_hi_x s_lo_y

/-- The generic Klibanoff/Fréchet composition obligation is
    discharged by `Sounio.FrechetGeneric.frechet_enclosure_inc_dec`. -/
theorem kl_compose_inc_dec_holds_generic
    (α : Type) [OrderedCarrier α] :
    kl_compose_inc_dec_obligation_generic α := by
  unfold kl_compose_inc_dec_obligation_generic
  intros f mono_x mono_y μ_x s_lo_x s_hi_x μ_y s_lo_y s_hi_y
         hxa hxb hyc hyd
  exact Sounio.FrechetGeneric.frechet_enclosure_inc_dec
          f mono_x mono_y
          s_lo_x s_hi_x s_lo_y s_hi_y μ_x μ_y
          hxa hxb hyc hyd

-- ================================================================
-- §4. Stage 0 specialisation corollaries.
-- ================================================================

/-- Stage 0 (Nat) specialisation of `kl_walley_ce_at_lambda_one`,
    using `Nat.min_eq_right`. -/
theorem kl_walley_ce_at_lambda_one_nat
    (μ_0 s_lo : Nat) (h : s_lo ≤ μ_0) :
    Nat.min μ_0 s_lo = s_lo :=
  kl_walley_ce_at_lambda_one_generic
    Nat.min (fun _ _ hab => Nat.min_eq_right hab) μ_0 s_lo h

/-- Stage 0 (Nat) specialisation of `kl_walley_ce_at_lambda_zero`,
    using `Nat.min_eq_left`. -/
theorem kl_walley_ce_at_lambda_zero_nat
    (μ_0 s_hi : Nat) (h : μ_0 ≤ s_hi) :
    Nat.min μ_0 s_hi = μ_0 :=
  kl_walley_ce_at_lambda_zero_generic
    Nat.min (fun _ _ hab => Nat.min_eq_left hab) μ_0 s_hi h

end Sounio.KlibanoffGeneric
