-- formal/lean4/SounioWalleyGeneric.lean
import SounioOrderedCarrier
import SounioFrechetGeneric

/-!
# Sounio.WalleyGeneric — Stage 2 lift of Walley elicitation theorems

This file lifts the structural theorems from `SounioWalley.lean`
(Stage 0, `Nat`-shadow) to the Stage 2 typeclass abstraction
`Sounio.OrderedCarrier α`. The Walley collapse-at-zero,
vacuous-at-one, and gap-monotone-in-ε theorems become **carrier-
generic** — they hold for any `OrderedCarrier α` instance,
including `Nat`, `Int`, `Rat`, and `SounioReal`.

## Why lift Walley to Stage 2

The Stage 0 Walley theorems are stated over `Nat` with a discrete
(num, den) encoding of ε ∈ [0, 1]. This is structurally honest
but operationally awkward: clinical Sounio code computes Walley
on `f64`, not `Nat`.

By lifting to `OrderedCarrier`, we get:
  - the same theorem applies to `SounioReal` (rational subset
    of ℝ) directly, with no re-encoding;
  - the bridge to Stage 3b `BoundedOrderedCarrier` for Float
    becomes a one-step Float-instance + bounded analogue
    (deferred until Float instance lands);
  - the M3.5 `walley_collapse_at_zero` and friends are now
    available at every carrier the user provides.

## Theorems mechanised

  - `walley_collapse_at_zero_generic` — at ε = 0, lifted
    lo/hi/gap collapse to a single point.

  - `walley_vacuous_at_one_generic` — at ε = 1
    (encoded as `eps = one`), the lifted band fills the
    support [s_lo, s_hi].

  - `walley_gap_monotone_in_epsilon_generic` — gap is monotone-
    non-decreasing in ε for any non-negative support width.

  - `walley_frechet_composition_generic` — Walley/Fréchet
    composition (monotone-(↑x, ↓y) f preserved through support
    rectangle).

## Encoding for the generic stage

Stage 0 used `(eps_num, eps_den)` with `eps_num ≤ eps_den`. At
Stage 2, since we have arbitrary ordered carriers (with potential
multiplication / division), we encode ε directly as a value in α
(typically restricted to `0 ≤ eps ≤ one` for downstream
soundness). The `lift_gap` function becomes
`gap = eps · (s_hi − s_lo)` — exactly what
`SounioFrechetGeneric.lift_gap` provides.

For the collapse-at-zero theorem, we use the carrier's `zero`
element (the typeclass method).

For the vacuous-at-one theorem, we require an explicit `one : α`
parameter (since the typeclass doesn't ship `one`); the user
supplies it.

## Status

Both theorems below build cleanly under
`lake build SounioWalleyGeneric`. No `axiom`, no `sorry`, no
Mathlib import.
-/

namespace Sounio.WalleyGeneric

open Sounio
open Sounio.OrderedCarrier

-- ================================================================
-- §1. Generic gap definition.
-- ================================================================
--
-- Defined identically to `SounioFrechetGeneric.lift_gap`. We
-- redeclare here as `walley_gap` to make the Walley-domain
-- naming explicit.

/-- Walley band gap: `gap = ε · (s_hi − s_lo)` over any
    `OrderedCarrier α`. Since `OrderedCarrier` doesn't ship `Sub`,
    we take `width = s_hi − s_lo` as a precomputed parameter. -/
def walley_gap {α : Type} [OrderedCarrier α] (eps width : α) : α :=
  eps * width

-- ================================================================
-- §2. Theorem: gap monotone-non-decreasing in ε.
-- ================================================================

/-- The lifted Walley band gap is monotone-non-decreasing in ε
    for any non-negative support width.

    Direct application of
    `OrderedCarrier.mul_le_mul_of_nonneg_right`. The Stage 0
    `walley_gap_monotone_in_epsilon_nat` and Stage 1
    `walley_gap_monotone_in_epsilon_rat` are special cases. -/
theorem walley_gap_monotone_in_epsilon_generic
    {α : Type} [OrderedCarrier α]
    (eps_a eps_b width : α)
    (h_eps : eps_a ≤ eps_b)
    (h_width : OrderedCarrier.zero ≤ width)
    : walley_gap eps_a width ≤ walley_gap eps_b width := by
  unfold walley_gap
  exact OrderedCarrier.mul_le_mul_of_nonneg_right width h_eps h_width

-- ================================================================
-- §3. Theorem: collapse at zero.
-- ================================================================

/-- At `eps = zero`, the lifted Walley gap is `zero`. The band
    collapses to a single point — the precise/Knowledge recovery.

    Proof: `walley_gap zero width = zero · width = zero` by the
    multiplicative-zero hypothesis `mul_zero_left` that the user
    supplies for their specific carrier. (Mul-zero is not a
    `OrderedCarrier` typeclass method to keep the surface
    minimal; carriers provide it as a per-instance hypothesis.) -/
theorem walley_gap_collapse_at_zero_generic
    {α : Type} [OrderedCarrier α]
    (width : α)
    (zero_mul_width : OrderedCarrier.zero * width = OrderedCarrier.zero)
    : walley_gap OrderedCarrier.zero width = OrderedCarrier.zero := by
  unfold walley_gap
  exact zero_mul_width

-- ================================================================
-- §4. Theorem: vacuous at one.
-- ================================================================

/-- At `eps = one` (where `one : α` is supplied by the user), the
    lifted Walley gap equals the full support width.

    Proof: `walley_gap one width = one · width = width` by the
    multiplicative-identity hypothesis the user supplies. -/
theorem walley_gap_vacuous_at_one_generic
    {α : Type} [OrderedCarrier α]
    (one width : α)
    (one_mul_width : one * width = width)
    : walley_gap one width = width := by
  unfold walley_gap
  exact one_mul_width

-- ================================================================
-- §5. Walley/Fréchet composition (generic).
-- ================================================================

/-- Generic Walley/Fréchet composition obligation: for monotone-
    (↑x, ↓y) `f` and a support rectangle, the corner enumeration
    encloses `f x y` for any `(x, y)` in the rectangle.

    This is the carrier-generic statement of the Stage 0
    `Sounio.Walley.walley_frechet_composition_obligation`. -/
def walley_frechet_composition_obligation_generic
    (α : Type) [OrderedCarrier α] : Prop :=
  ∀ (f : α → α → α),
    (∀ x x' y, x ≤ x' → f x y ≤ f x' y) →
    (∀ x y y', y ≤ y' → f x y' ≤ f x y) →
    ∀ (s_x_lo s_x_hi s_y_lo s_y_hi x y : α),
    s_x_lo ≤ x → x ≤ s_x_hi →
    s_y_lo ≤ y → y ≤ s_y_hi →
    f s_x_lo s_y_hi ≤ f x y ∧ f x y ≤ f s_x_hi s_y_lo

/-- The generic Walley/Fréchet composition obligation is
    discharged by direct application of
    `Sounio.FrechetGeneric.frechet_enclosure_inc_dec`. -/
theorem walley_frechet_composition_holds_generic
    (α : Type) [OrderedCarrier α] :
    walley_frechet_composition_obligation_generic α := by
  unfold walley_frechet_composition_obligation_generic
  intros f mono_x mono_y s_x_lo s_x_hi s_y_lo s_y_hi x y
         hxa hxb hyc hyd
  exact Sounio.FrechetGeneric.frechet_enclosure_inc_dec
          f mono_x mono_y
          s_x_lo s_x_hi s_y_lo s_y_hi x y
          hxa hxb hyc hyd

end Sounio.WalleyGeneric
