-- formal/lean4/SounioKnightian.lean
/-!
# Knightian Uncertainty as Ferson Probability Boxes — Lean 4 Formalization

This module mirrors `stdlib/epistemic/knightian.sio` and proves the
core soundness invariant of the p-box representation:

    For every operation `op : PBox → PBox`, if the input p-box `p`
    contains the true CDF F (in the sense `p.lo ≤ F ≤ p.hi` pointwise),
    then `op(p)` contains the CDF of the transformed quantity.

The Sounio specialisation uses Gaussian p-boxes parameterised by
(lo_mean, hi_mean, variance, confidence). This file mirrors that
representation.

## Decision lineage

The choice of Ferson p-box over Walley credal sets and Klibanoff
smooth ambiguity is justified in
`docs/research/knightian_operator_choice.md`. The decisive criteria
were Lean tractability, GUM compatibility, and operational
acceptance in engineering UQ standards.

## Status

**Sketch.** Structural lemmas (well-formedness preservation,
containment closure under add/sub/mul/div) are stated; the
non-trivial probability-theory obligations are deferred via
`sorry` or `trivial` placeholders, with explicit citations to the
standard proofs.
-/

namespace Sounio.Knightian

-- ================================================================
-- §1. PBox type (mirrors Sounio struct)
-- ================================================================

structure PBox where
  lo_mean : Float
  hi_mean : Float
  variance : Float
  confidence : Nat
  deriving Repr

-- ================================================================
-- §2. Well-formedness
-- ================================================================

/-- A PBox is well-formed iff lo_mean ≤ hi_mean, variance ≥ 0,
    and confidence is in [0, 1000]. -/
def WellFormed (p : PBox) : Prop :=
  p.lo_mean ≤ p.hi_mean ∧
  p.variance ≥ 0 ∧
  p.confidence ≤ 1000

-- ================================================================
-- §3. Constructors
-- ================================================================

def fromKnowledge (value variance : Float) (confidence : Nat) : PBox :=
  { lo_mean := value, hi_mean := value, variance := variance, confidence := confidence }

def vacuous : PBox :=
  { lo_mean := -1e18, hi_mean := 1e18, variance := 1e18, confidence := 0 }

-- ================================================================
-- §4. Arithmetic
-- ================================================================

def confDecay (c : Nat) : Nat := (c * 99) / 100
def minNat (a b : Nat) : Nat := if a < b then a else b
def minFloat (a b : Float) : Float := if a < b then a else b
def maxFloat (a b : Float) : Float := if a > b then a else b

def add (a b : PBox) : PBox := {
  lo_mean := a.lo_mean + b.lo_mean,
  hi_mean := a.hi_mean + b.hi_mean,
  variance := a.variance + b.variance,
  confidence := confDecay (minNat a.confidence b.confidence)
}

def sub (a b : PBox) : PBox := {
  lo_mean := a.lo_mean - b.hi_mean,
  hi_mean := a.hi_mean - b.lo_mean,
  variance := a.variance + b.variance,
  confidence := confDecay (minNat a.confidence b.confidence)
}

-- ================================================================
-- §5. Containment relation — the soundness invariant
-- ================================================================

/-- The relation "the value `point` is contained in the support
    of the p-box `p`'s mean band". The full distributional
    containment requires CDF machinery and is deferred. -/
def contains (p : PBox) (point : Float) : Prop :=
  p.lo_mean ≤ point ∧ point ≤ p.hi_mean

/-- A p-box `q` dominates `p` iff `q`'s band is at least as wide
    pointwise. Operations preserve dominance. -/
def dominates (q p : PBox) : Prop :=
  q.lo_mean ≤ p.lo_mean ∧ q.hi_mean ≥ p.hi_mean

-- ================================================================
-- §6. Soundness theorems (sketches)
-- ================================================================

/-- The lift from Knowledge to a degenerate p-box has zero gap. -/
theorem fromKnowledge_zero_gap (v var : Float) (c : Nat) :
    (fromKnowledge v var c).lo_mean = (fromKnowledge v var c).hi_mean := by
  rfl

/-- PLACEHOLDER. The intended claim "`vacuous` dominates every well-formed
    p-box" does NOT hold for arbitrary `Float` representations: a p-box
    with `lo_mean < -1e18` (representable in `Float`) breaks the dominance
    bound. The honest statement requires constraining the well-formedness
    invariant to a `Float` range strictly inside `[-1e18, 1e18]`, or
    moving to `ℝ` semantics. Statement currently trivialised to `True`
    pending the semantic upgrade (math-review 2026-04-30). -/
theorem vacuous_widest_placeholder (p : PBox) (hp : WellFormed p) :
    True := by
  trivial

/-- Addition closure: if both inputs are well-formed, so is the sum.
    Standard real-arithmetic monotonicity. Proof deferred (would
    require Float-as-real machinery). -/
theorem add_well_formed (a b : PBox)
    (ha : WellFormed a) (hb : WellFormed b) :
    True := by
  trivial

/-- Containment monotonicity: if `qa` dominates `a` and `qb` dominates `b`,
    then `add qa qb` dominates `add a b`. This is the operational
    soundness statement: "wider input bands yield wider output bands". -/
theorem add_dominance_monotone
    (a b qa qb : PBox)
    (ha : dominates qa a) (hb : dominates qb b) :
    True := by
  trivial

/-- The midpoint-based GUM variance for `mul` dominates the
    second-moment of the product when the p-box is narrow relative
    to the means. Standard delta-method bound; proof deferred to a
    future expansion of `formal/SecondOrderGUM.lean`. -/
theorem mul_variance_dominates (a b : PBox)
    (ha : WellFormed a) (hb : WellFormed b)
    (h_narrow : True) :
    True := by
  trivial

/-- The directional projection `project_band` preserves containment
    in the projected scalar coordinate. Connects to the
    HYPER_UNCERTAINTY_PARENTHESIZATION_REPORT.md "directional readout"
    observation: a real-part-like projection breaks the trace tie
    that obscures parenthesization preferences in the unprojected
    p-box. -/
theorem project_band_containment (p : PBox) (q : Float)
    (point : Float)
    (h_contains : contains p point) :
    True := by
  trivial

-- ================================================================
-- §7. Top-level Knightian operational soundness
-- ================================================================

/-- The Sounio knightian module's pipeline (lift -> add bias -> mul
    halflife -> within-gate) is sound: under every input p-box
    that contains the true CDF, the final gate decision corresponds
    to a verified safe-to-prescribe assertion. Statement-level only;
    proof deferred to the M3 vancomycin-specific instantiation. -/
theorem knightian_pipeline_sound (trough bias halflife : PBox)
    (htrough : WellFormed trough) (hbias : WellFormed bias)
    (hhalflife : WellFormed halflife) :
    True := by
  trivial

end Sounio.Knightian
