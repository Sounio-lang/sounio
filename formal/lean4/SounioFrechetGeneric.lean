-- formal/lean4/SounioFrechetGeneric.lean
import SounioOrderedCarrier

/-!
# Sounio.FrechetGeneric — generic Fréchet enclosure (Stage 2)

This file is the Stage 2 milestone of the Float-Real lift roadmap
(`docs/research/lean_float_real_roadmap.md`).

It proves the structural Fréchet enclosure theorems **once** over
the abstract `Sounio.OrderedCarrier` typeclass, and recovers the
Stage 0 (`Nat`) and Stage 1 (`Rat`) versions as direct
specialisations.

## Why this milestone matters

Stage 0 (`SounioFrechet.lean`) and Stage 1 (`SounioFrechetRat.lean`)
proved the same structural content twice — once for `Nat` and once
for `Rat` — using carrier-specific lemma names. This file
**unifies** the two by stating the theorem generically over any
`OrderedCarrier α`. The carrier-specific theorems are now
corollaries.

The payoff is:

  - Adding a new carrier (e.g., `Int`, `Real`, `Float`) requires
    *only* providing a typeclass instance — the structural theorem
    transfers automatically.

  - The proof itself is shorter and clearer: it uses only
    `OrderedCarrier.le_trans` (no carrier-specific lemma name).

  - Any future epistemic operator (e.g., a new credal-set lift, a
    non-CARA Klibanoff variant) that depends on the same
    structural shape can be proved generically as well.

## Status

All four theorems below are **fully proven** in core Lean 4 over
the `OrderedCarrier` typeclass. No `axiom`, no `sorry`, no
Mathlib import. `lake build SounioFrechetGeneric` builds clean.

## Coverage

  - 3 Fréchet variants (inc-dec, inc-inc, dec-dec)
  - 1 Walley gap-monotonicity theorem (uses
    `mul_le_mul_of_nonneg_right`)
  - 2 corollary specialisations: `_nat` and `_rat`
-/

namespace Sounio.FrechetGeneric

open Sounio
open Sounio.OrderedCarrier

-- ================================================================
-- §1. Generic Fréchet enclosure theorems.
-- ================================================================

/-- **Generic Fréchet outer enclosure** for monotone-(↑x, ↓y) `f`
    on an interval rectangle `[a, b] × [c, d]` over any
    `OrderedCarrier α`.

    Given the monotonicity hypotheses and `(x, y) ∈ rectangle`,
    the value `f x y` is sandwiched between the two diagonally-
    opposite corner evaluations:

        f a d ≤ f x y ≤ f b c.

    The proof uses **only** `OrderedCarrier.le_trans`. No
    carrier-specific lemma is invoked. -/
theorem frechet_enclosure_inc_dec
    {α : Type} [OrderedCarrier α]
    (f : α → α → α)
    (mono_x_inc : ∀ x x' y, x ≤ x' → f x y ≤ f x' y)
    (mono_y_dec : ∀ x y y', y ≤ y' → f x y' ≤ f x y)
    (a b c d x y : α)
    (hxa : a ≤ x) (hxb : x ≤ b)
    (hyc : c ≤ y) (hyd : y ≤ d)
    : f a d ≤ f x y ∧ f x y ≤ f b c := by
  refine ⟨?_, ?_⟩
  · exact OrderedCarrier.le_trans
            (mono_y_dec a y d hyd) (mono_x_inc a x y hxa)
  · exact OrderedCarrier.le_trans
            (mono_x_inc x b y hxb) (mono_y_dec b c y hyc)

/-- Generic Fréchet enclosure for monotone-(↑x, ↑y) `f`. -/
theorem frechet_enclosure_inc_inc
    {α : Type} [OrderedCarrier α]
    (f : α → α → α)
    (mono_x_inc : ∀ x x' y, x ≤ x' → f x y ≤ f x' y)
    (mono_y_inc : ∀ x y y', y ≤ y' → f x y ≤ f x y')
    (a b c d x y : α)
    (hxa : a ≤ x) (hxb : x ≤ b)
    (hyc : c ≤ y) (hyd : y ≤ d)
    : f a c ≤ f x y ∧ f x y ≤ f b d := by
  refine ⟨?_, ?_⟩
  · exact OrderedCarrier.le_trans
            (mono_y_inc a c y hyc) (mono_x_inc a x y hxa)
  · exact OrderedCarrier.le_trans
            (mono_x_inc x b y hxb) (mono_y_inc b y d hyd)

/-- Generic Fréchet enclosure for monotone-(↓x, ↓y) `f`. -/
theorem frechet_enclosure_dec_dec
    {α : Type} [OrderedCarrier α]
    (f : α → α → α)
    (mono_x_dec : ∀ x x' y, x ≤ x' → f x' y ≤ f x y)
    (mono_y_dec : ∀ x y y', y ≤ y' → f x y' ≤ f x y)
    (a b c d x y : α)
    (hxa : a ≤ x) (hxb : x ≤ b)
    (hyc : c ≤ y) (hyd : y ≤ d)
    : f b d ≤ f x y ∧ f x y ≤ f a c := by
  refine ⟨?_, ?_⟩
  · exact OrderedCarrier.le_trans
            (mono_y_dec b y d hyd) (mono_x_dec x b y hxb)
  · exact OrderedCarrier.le_trans
            (mono_x_dec a x y hxa) (mono_y_dec a c y hyc)

-- ================================================================
-- §2. Generic Walley gap monotonicity.
-- ================================================================
--
-- The lifted band gap `gap = ε · width` is monotone-non-decreasing
-- in ε, given a non-negative width. This requires
-- `OrderedCarrier.mul_le_mul_of_nonneg_right`.

/-- Generic gap function: the Walley/Kantorovich-style band gap is
    `ε · width`. -/
def lift_gap {α : Type} [OrderedCarrier α] (eps width : α) : α :=
  eps * width

/-- The lifted gap is monotone-non-decreasing in ε for any
    non-negative width. -/
theorem lift_gap_monotone_in_epsilon
    {α : Type} [OrderedCarrier α]
    (eps_a eps_b width : α)
    (h_eps : eps_a ≤ eps_b)
    (h_width : OrderedCarrier.zero ≤ width)
    : lift_gap eps_a width ≤ lift_gap eps_b width := by
  unfold lift_gap
  exact OrderedCarrier.mul_le_mul_of_nonneg_right width h_eps h_width

-- ================================================================
-- §3. Stage 0 / Stage 1 specialisation corollaries.
-- ================================================================
--
-- The Stage 0 (Nat) and Stage 1 (Rat) versions of the inc-dec
-- Fréchet theorem are direct specialisations of the generic
-- theorem under the canonical `OrderedCarrier` instances. These
-- corollaries *demonstrate* the subsumption claim from the
-- Stage 2 thesis — they are not the source of any soundness
-- (the source is the generic theorem above).

/-- Stage 0 (Nat) specialisation of `frechet_enclosure_inc_dec`.
    By typeclass resolution this picks up the canonical Nat
    instance from `SounioOrderedCarrier.lean`. -/
theorem frechet_enclosure_inc_dec_nat
    (f : Nat → Nat → Nat)
    (mono_x_inc : ∀ x x' y, x ≤ x' → f x y ≤ f x' y)
    (mono_y_dec : ∀ x y y', y ≤ y' → f x y' ≤ f x y)
    (a b c d x y : Nat)
    (hxa : a ≤ x) (hxb : x ≤ b)
    (hyc : c ≤ y) (hyd : y ≤ d)
    : f a d ≤ f x y ∧ f x y ≤ f b c :=
  frechet_enclosure_inc_dec f mono_x_inc mono_y_dec
    a b c d x y hxa hxb hyc hyd

/-- Stage 1 (Rat) specialisation of `frechet_enclosure_inc_dec`. -/
theorem frechet_enclosure_inc_dec_rat
    (f : Rat → Rat → Rat)
    (mono_x_inc : ∀ x x' y, x ≤ x' → f x y ≤ f x' y)
    (mono_y_dec : ∀ x y y', y ≤ y' → f x y' ≤ f x y)
    (a b c d x y : Rat)
    (hxa : a ≤ x) (hxb : x ≤ b)
    (hyc : c ≤ y) (hyd : y ≤ d)
    : f a d ≤ f x y ∧ f x y ≤ f b c :=
  frechet_enclosure_inc_dec f mono_x_inc mono_y_dec
    a b c d x y hxa hxb hyc hyd

-- ================================================================
-- §4. Vancomycin Cmin substrate-generality (Stage 2 perspective).
-- ================================================================

/-- The Vancomycin Cmin Fréchet enclosure obligation in its most
    abstract form: any `OrderedCarrier α` with monotone-(↑x, ↓y)
    `cmin` admits the corner-enumeration band as a sound outer
    enclosure.

    This is the Stage 2 statement of the
    `SounioFrechet.vancomycin_cmin_frechet_enclosure_obligation` /
    `SounioFrechetRat.vancomycin_cmin_frechet_enclosure_rat`
    theorems. It demonstrates that the Sounio Fréchet machinery is
    **carrier-agnostic**: any future Float instance (Stage 3)
    automatically inherits this enclosure (modulo the Float
    bounded-soundness inflation). -/
def vancomycin_cmin_frechet_enclosure_obligation_generic
    (α : Type) [OrderedCarrier α] : Prop :=
  ∀ (cmin : α → α → α),
    (∀ vc vc' cl, vc ≤ vc' → cmin vc cl ≤ cmin vc' cl) →
    (∀ vc cl cl', cl ≤ cl' → cmin vc cl' ≤ cmin vc cl) →
    ∀ (vc_lo vc_hi cl_lo cl_hi vc cl : α),
    vc_lo ≤ vc → vc ≤ vc_hi →
    cl_lo ≤ cl → cl ≤ cl_hi →
    cmin vc_lo cl_hi ≤ cmin vc cl ∧ cmin vc cl ≤ cmin vc_hi cl_lo

/-- The generic Vancomycin Cmin Fréchet enclosure obligation is
    discharged by the abstract Fréchet theorem. -/
theorem vancomycin_cmin_frechet_enclosure_generic
    (α : Type) [OrderedCarrier α] :
    vancomycin_cmin_frechet_enclosure_obligation_generic α := by
  unfold vancomycin_cmin_frechet_enclosure_obligation_generic
  intros cmin mono_vc mono_cl
         vc_lo vc_hi cl_lo cl_hi vc cl
         h_vc_lo h_vc_hi h_cl_lo h_cl_hi
  exact frechet_enclosure_inc_dec
          cmin mono_vc mono_cl
          vc_lo vc_hi cl_lo cl_hi vc cl
          h_vc_lo h_vc_hi h_cl_lo h_cl_hi

end Sounio.FrechetGeneric
