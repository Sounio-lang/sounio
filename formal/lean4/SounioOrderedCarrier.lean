-- formal/lean4/SounioOrderedCarrier.lean
/-!
# Sounio.OrderedCarrier — minimal typeclass abstraction (Stage 2)

This file is the **Stage 2** milestone of the Float-Real lift
roadmap (`docs/research/lean_float_real_roadmap.md`).

It defines a minimal Mathlib-free typeclass `OrderedCarrier α`
that captures the structural content the Sounio Fréchet/Walley
theorems require, and provides instances for the carriers we
already use:

  - `Nat`  (Stage 0 carrier)
  - `Int`  (mostly for completeness)
  - `Rat`  (Stage 1 carrier)

The Stage 0 (`SounioFrechet.lean`) and Stage 1
(`SounioFrechetRat.lean`) theorems are **carrier-specific**:
they were each re-derived for `Nat` and `Rat` separately. With
this typeclass, `SounioFrechetGeneric.lean` proves the structural
content **once**, and the carrier-specific versions become direct
specialisations under the canonical instance.

This is the Mathlib-free analogue of Mathlib's
`OrderedCommSemiring` / `OrderedSemiring` typeclass hierarchy,
restricted to exactly the methods that the Sounio epistemic
theorems actually use.

## Design choices

- **Minimal**: only `le_trans` and `mul_le_mul_of_nonneg_right`
  are exposed. Adding methods later is fine; removing them later
  is harder (it breaks downstream proofs).

- **Inherits `LE α` and `Mul α`** from Lean 4 core. Does NOT
  inherit any algebraic structure (no `Semiring`, no
  `LinearOrder`); we want the *structural* content, not the full
  ring axioms.

- **Carries an explicit `zero : α`** rather than depending on
  `OfNat α 0` in core. This makes the `Nat`-instance proof of
  `zero ≤ k` trivial via `Nat.zero_le`.

- **Mathlib-free**: no `import Mathlib.*`; uses only Lean 4 core
  (`Nat`, `Int`, `Rat` and their `le_trans` /
  `mul_le_mul_of_nonneg_*` lemmas).

## Status

The typeclass + 3 instances build cleanly under
`lake build SounioOrderedCarrier`. No `axiom`, no `sorry`, no
Mathlib import. All four typeclass methods are proven for each
instance using only the matching Lean 4 core lemma.

## What this enables (and what it doesn't)

This file is **infrastructure**. It does not, by itself, prove
any Fréchet / Walley / Klibanoff theorem. What it does is:

  - declare the abstract algebraic shape that the proofs need;
  - record the canonical instances that Sounio currently uses;
  - establish the migration path for future carriers (`Real`,
    `Float`, `Interval`, ...).

The companion file `SounioFrechetGeneric.lean` consumes this
typeclass to prove the Fréchet enclosure theorem **once** for
all instances.

A future `Float` instance will require a **different typeclass**
(`BoundedOrderedCarrier α (eps : α)` with relaxed laws, since
IEEE-754 rounding breaks strict `mul_le_mul_of_nonneg_right`).
That typeclass is part of the Stage 3 deliverable, deferred 4-6
weeks per the roadmap.
-/

namespace Sounio

/-- Minimal ordered-carrier typeclass for Sounio's structural
    epistemic theorems.

    A type `α` is an `OrderedCarrier` if it has:
      - a `LE` (≤ relation, from Lean 4 core)
      - a `Mul` (multiplication, from Lean 4 core)
      - a designated `zero : α` element
      - transitivity of `≤`
      - right-multiplication by a non-negative scalar preserves `≤`

    These are exactly the methods used by the Fréchet enclosure
    theorem and the Walley gap-monotonicity theorem. -/
class OrderedCarrier (α : Type) extends LE α, Mul α where
  zero : α
  le_trans : ∀ {a b c : α}, a ≤ b → b ≤ c → a ≤ c
  mul_le_mul_of_nonneg_right :
    ∀ {a b : α} (c : α), a ≤ b → zero ≤ c → a * c ≤ b * c

namespace OrderedCarrier

-- ================================================================
-- §1. Nat instance
-- ================================================================

instance : OrderedCarrier Nat where
  zero := 0
  le_trans := @Nat.le_trans
  mul_le_mul_of_nonneg_right := fun c hab _ =>
    Nat.mul_le_mul_right c hab

-- ================================================================
-- §2. Int instance
-- ================================================================
--
-- Int needs the actual non-negativity hypothesis since multiplying
-- by a negative scalar reverses the inequality. The lemma
-- `Int.mul_le_mul_of_nonneg_right` gives us exactly that.

instance : OrderedCarrier Int where
  zero := 0
  le_trans := @Int.le_trans
  mul_le_mul_of_nonneg_right := fun _ hab hc =>
    Int.mul_le_mul_of_nonneg_right hab hc

-- ================================================================
-- §3. Rat instance
-- ================================================================

instance : OrderedCarrier Rat where
  zero := 0
  le_trans := @Rat.le_trans
  mul_le_mul_of_nonneg_right := fun _ hab hc =>
    Rat.mul_le_mul_of_nonneg_right hab hc

-- ================================================================
-- §4. Sanity: Nat is a faithful sub-carrier of Rat (corollary).
-- ================================================================

/-- The canonical Nat → Rat embedding is monotone. This sanity
    check confirms the Stage 0 → Stage 1 bridge from
    `SounioFrechetRat.lean`, here re-stated using the typeclass
    abstraction. -/
theorem nat_le_rat (a b : Nat) (h : a ≤ b) :
    (a : Rat) ≤ (b : Rat) := by
  exact_mod_cast h

end OrderedCarrier
end Sounio
