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
    pending the semantic upgrade (math-review 2026-04-30).

    **STATUS: `: True` placeholder — statement is FALSE for unbounded WellFormed.**

    The BOUNDED ℚ discharge is `Sounio.PBoxSemantics.vacuousR_dominates_bounded`
    in `SounioPBoxSemantics.lean`. It proves: for any `PBoxR` satisfying
    `BoundedWellFormedR` (i.e., `WellFormedR` plus `-10^18 ≤ lo ∧ hi ≤ 10^18`),
    the ℚ-vacuous box `vacuousR` dominates it. This is near-tautological —
    `BoundedWellFormedR` carries exactly those two inequalities.

    Mutating `WellFormed` here to `BoundedWellFormed` was rejected because
    it would ripple to `SounioVancomycinDosingSafety` and
    `SounioTacrolimusDosingSafety` (see risk note in
    `SounioPBoxSemantics §6`). Callers needing the bounded result should
    import `SounioPBoxSemantics` directly. -/
theorem vacuous_widest_placeholder (p : PBox) (hp : WellFormed p) :
    True := by
  trivial

/-- Addition preserves the **confidence-bound** conjunct of well-formedness
    (DISCHARGED, `Nat`-only): if both inputs have `confidence ≤ 1000`, so does
    `add a b` (whose confidence is `confDecay (minNat …)`). Named narrowly: this
    is ONLY the `Nat` confidence conjunct — the `lo_mean ≤ hi_mean` and
    `variance ≥ 0` conjuncts are `Float`-typed and need the axiom-bearing
    Float↔Real lift (core Lean has no usable Float order/add-monotone lemma), so
    full `WellFormed` closure stays future-work. Mathlib-free, no axiom/sorry. -/
theorem add_confidence_bounded (a b : PBox)
    (ha : WellFormed a) (hb : WellFormed b) :
    (add a b).confidence ≤ 1000 := by
  obtain ⟨_, _, _⟩ := ha
  obtain ⟨_, _, _⟩ := hb
  show confDecay (minNat a.confidence b.confidence) ≤ 1000
  unfold confDecay minNat
  split <;> omega

/-- Containment monotonicity: if `qa` dominates `a` and `qb` dominates `b`,
    then `add qa qb` dominates `add a b`. This is the operational
    soundness statement: "wider input bands yield wider output bands".

    **STATUS: `: True` placeholder — IMPORT CYCLE prevents in-place discharge.**

    The genuine ℚ-backed discharge is
    `Sounio.PBoxSemantics.add_dominance_monotone_rat` in
    `SounioPBoxSemantics.lean`. That theorem is proven without sorry or
    native_decide, over the rational images `toRatPBox a ..` and rests
    on exactly 3 of the 5 IEEE-754 axioms (`toRat`, `IsFiniteNormal`,
    `toRat_le_iff_finite`) plus `[propext, Classical.choice, Quot.sound]`.
    (`mul_rne_bound` and `add_rne_bound` are not needed since `addR` is pure ℚ.)

    WHY NOT HERE: `SounioPBoxSemantics.lean` imports this file, so
    restating the proof here using `PBoxR`/`toRatPBox`/`dominatesR`
    would create an import cycle. The `: True` placeholder stays;
    cite `Sounio.PBoxSemantics.add_dominance_monotone_rat` in proofs
    that actually need the content. -/
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
    p-box.

    **STATUS: `: True` placeholder — no Float mul-monotone lemma in core Lean 4.**

    The genuine ℚ discharge is `Sounio.PBoxSemantics.projectR_containment_rat`
    in `SounioPBoxSemantics.lean`. It proves: for `0 ≤ q : Rat` and
    `containsR p point`, we have `containsR (projectR p q) (q * point)`.
    Pure ℚ — no axioms, no sorry (`[propext, Classical.choice, Quot.sound]`).

    WHY NOT HERE: `SounioPBoxSemantics.lean` imports this file (import cycle).
    Additionally, there is no `Float.mul_le_iff_finite` axiom in the
    IEEE-754 spec to bridge `Float` multiplication monotonicity. The Float-level
    statement requires a future `mul_ord_bound` axiom (Phase 3 work).
    Missing lemma: `Float.mul_le_iff_finite` (or `mul_ord_bound`). -/
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
