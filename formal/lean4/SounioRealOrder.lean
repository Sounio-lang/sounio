-- formal/lean4/SounioRealOrder.lean
import SounioOrderedCarrier

/-!
# Sounio.RealOrder — Mathlib-free Stage 3a Real instance (Route A)

This file is the Stage 3a milestone of the Float-Real lift roadmap
(`docs/research/lean_float_real_roadmap.md`).

## Honest scope statement

A *complete* in-tree Real construction (Cauchy sequences modulo
ε-equivalence with full operations and ordered-field axioms
proven) is a 500-1000 LOC undertaking that we deliberately defer.
What this file provides is a **`SounioReal` whose denotation is
the rational subset of ℝ**, with an `OrderedCarrier` instance
inherited from `Rat`.

The rationale is structural: every theorem in the Sounio
epistemic stack that holds on `ℚ` extends to `ℝ` by density (ℚ
is dense in ℝ in the sup-norm; the Fréchet enclosure / Walley
gap monotonicity / Klibanoff sandwich are all monotone-continuous
operations whose validity transfers across uniform limits).

So we get:

  - **Now**: `SounioReal` covers ℝ ∩ ℚ (the rational reals); the
    typeclass instance is bit-perfect.
  - **Future** (deferred Stage 3a-Cauchy milestone, ~500-1000 LOC):
    a separate `SounioRealCauchy` type with Cauchy-quotient
    semantics, plus a bridge `SounioReal → SounioRealCauchy`
    (canonical embedding) and `densify : SounioRealCauchy → ℝ`
    (the limit operation, requires either Mathlib `Real` or
    in-tree Real-as-Cauchy).

This is the same pattern Mathlib follows: `Rat ↪ Real` with `Real`
the Cauchy quotient. The honest difference is that we ship the
embedding NOW (~30 LOC, structural milestone) and defer the
quotient until Sounio's epistemic stack actually requires
irrationals (which it currently does not — all PK regimes
operate on rational dose / weight / clearance / time).

## Mathematical soundness

For any operation `f : SounioReal → SounioReal → SounioReal` that
is a pure delegation to a `Rat`-level operation, the soundness
property "`f` preserves the structural ordering content" lifts
mechanically from `Rat` to `SounioReal`. The Fréchet enclosure,
the Walley `lift_gap`, and the Klibanoff CARA-CE boundary cases
all fall in this class.

For operations that require **continuous extension to
irrationals** (e.g., `exp`, `log`, infinite series limits), the
rational `SounioReal` is insufficient. These remain on the
Stage 3a-Cauchy roadmap.

## Status

The instance below builds cleanly under `lake build SounioRealOrder`
with no `axiom`, no `sorry`, no Mathlib import. The Stage 0/1/2
theorems all instantiate on `SounioReal` automatically.

Math-review record:
- Pre-impl thesis: Grok 4.1 9/10 OK + 1 TIGHTENABLE (density-
  bridge formalization — requires Real type, deferred to
  Cauchy milestone).
-/

namespace Sounio

/-- `SounioReal` is a thin wrapper over `Rat` representing the
    rational subset of ℝ. Its denotation is `{q ∈ ℚ : q ∈ ℝ}` =
    ℚ. The Cauchy completion to cover irrationals is deferred to
    a future Stage 3a-Cauchy milestone (separate type
    `SounioRealCauchy`). -/
structure SounioReal where
  /-- The exact rational value. -/
  approx : Rat

namespace SounioReal

/-- The canonical zero. -/
def zero : SounioReal := ⟨0⟩

/-- The canonical one (useful for sanity tests, not strictly
    required by `OrderedCarrier`). -/
def one : SounioReal := ⟨1⟩

instance : LE SounioReal where
  le a b := a.approx ≤ b.approx

instance : Mul SounioReal where
  mul a b := ⟨a.approx * b.approx⟩

/-- The canonical embedding `Rat → SounioReal`. -/
def ofRat (q : Rat) : SounioReal := ⟨q⟩

/-- The canonical projection `SounioReal → Rat` (inverse of
    `ofRat` since `SounioReal` is a thin wrapper). -/
def toRat (r : SounioReal) : Rat := r.approx

-- ================================================================
-- §1. The OrderedCarrier instance.
-- ================================================================

instance : Sounio.OrderedCarrier SounioReal where
  zero := zero
  le_trans := fun {a b c} hab hbc =>
    Rat.le_trans (a := a.approx) (b := b.approx) (c := c.approx) hab hbc
  mul_le_mul_of_nonneg_right := fun {a b} c hab hc =>
    Rat.mul_le_mul_of_nonneg_right (a := a.approx) (b := b.approx)
      (c := c.approx) hab hc

-- ================================================================
-- §2. Sanity: the canonical embedding is monotone.
-- ================================================================

/-- The Rat → SounioReal embedding preserves the order. -/
theorem ofRat_le_ofRat (a b : Rat) (h : a ≤ b) :
    ofRat a ≤ ofRat b := h

/-- The SounioReal → Rat projection preserves the order. -/
theorem toRat_le_toRat (a b : SounioReal) (h : a ≤ b) :
    toRat a ≤ toRat b := h

-- ================================================================
-- §3. Stage 3a substrate-generality witnesses.
-- ================================================================
--
-- The Stage 0/1/2 theorems all instantiate on SounioReal because
-- it is an OrderedCarrier instance. These witnesses spell out
-- some of the more important specialisations.

/-- Vancomycin Cmin Fréchet enclosure on SounioReal. -/
theorem vancomycin_cmin_frechet_enclosure_real
    (cmin : SounioReal → SounioReal → SounioReal)
    (mono_vc : ∀ vc vc' cl, vc ≤ vc' → cmin vc cl ≤ cmin vc' cl)
    (mono_cl : ∀ vc cl cl', cl ≤ cl' → cmin vc cl' ≤ cmin vc cl)
    (vc_lo vc_hi cl_lo cl_hi vc cl : SounioReal)
    (h_vc_lo : vc_lo ≤ vc) (h_vc_hi : vc ≤ vc_hi)
    (h_cl_lo : cl_lo ≤ cl) (h_cl_hi : cl ≤ cl_hi)
    : cmin vc_lo cl_hi ≤ cmin vc cl
      ∧ cmin vc cl ≤ cmin vc_hi cl_lo := by
  refine ⟨?_, ?_⟩
  · exact Sounio.OrderedCarrier.le_trans
            (mono_cl vc_lo cl cl_hi h_cl_hi)
            (mono_vc vc_lo vc cl h_vc_lo)
  · exact Sounio.OrderedCarrier.le_trans
            (mono_vc vc vc_hi cl h_vc_hi)
            (mono_cl vc_hi cl_lo cl h_cl_lo)

/-- Walley gap monotone-in-ε on SounioReal. -/
theorem walley_gap_monotone_real
    (eps_a eps_b width : SounioReal)
    (h_eps : eps_a ≤ eps_b)
    (h_width : Sounio.OrderedCarrier.zero ≤ width)
    : eps_a * width ≤ eps_b * width :=
  Sounio.OrderedCarrier.mul_le_mul_of_nonneg_right
    (a := eps_a) (b := eps_b) width h_eps h_width

end SounioReal
end Sounio
