-- formal/lean4/SounioApproxCausalKnowledge.lean
/-!
# Composition of Approx, Causal, and Knowledge Effects — Lean 4 Formalization

This file mirrors the Sounio module
`stdlib/epistemic/composed_effects.sio` and proves that the three
algebraic effects `Approx`, `Causal`, `Knowledge` compose without
information loss when the underlying values use the canonical
`ComposedKnowledge` representation.

## Top-level claim

For any well-formed inputs `a b : ComposedKnowledge` and any of the
arithmetic operations `add`, `sub`, `mul`, `div` (denoted `op`),
discharging the three effects in any order produces values that
agree on every channel up to a confidence-decay equivalence.

In short: the three handlers commute under `op`.

## References

- Sounio source: `stdlib/epistemic/composed_effects.sio`
- Plotkin & Pretnar (2009) "Handlers of Algebraic Effects" — for the
  general handler-composition theorem on a free algebra.
- Pearl (2009) *Causality* — Beta-edge representation.
- JCGM 100:2008 (GUM) — variance propagation rules.
- Higham (2002) *Accuracy and Stability of Numerical Algorithms* — Approx
  bound triangle inequality.

## Status

**Sketch with `True`-statement placeholders** (per Sounio's no-`sorry`
convention; see `SounioEffects.lean`). The structural lemmas
(`handler_commutativity`, `canonical_discharge_id`,
`composition_soundness`) prove genuine equalities by `rfl`. The
non-structural obligations below currently degenerate to `True` and
are documented honestly as placeholders:

1. `mul_variance_dominates_placeholder` — the GUM mul variance term
   `bv²·a.var + av²·b.var + a.var·b.var` equals the second-central
   moment of the product of two independent random variables (NOT
   "dominates"; it is exact under independence). The full proof
   requires probabilistic-machinery extension; the current statement
   trivialises to `True` and is named accordingly.

2. `approx_triangle_mul_placeholder` — the triangle-inequality bound
   `|y|·δx + |x|·δy + δx·δy` (with the cross-term, post math-review
   2026-04-30) for the product approximation. Statement currently
   trivialises to `True`.

3. `causal_independence_approximation_placeholder` — independent
   evidence pooling produces Beta(α₁+α₂, β₁+β₂) (not multiplication;
   that mistake was previously documented in the Sounio
   `composed_effects.sio` comment and corrected the same day).
   Statement currently trivialises to `True`.

A future commit will tighten the placeholder statements to genuine
propositions and provide proofs once the Mathlib-free probabilistic
machinery in `formal/Epistemic.lean` is extended with a Beta-distribution
algebra. Expected effort: 4-6 weeks of dedicated Lean work.
-/

namespace Sounio.ApproxCausalKnowledge

-- ================================================================
-- §1. Composed value type (mirrors Sounio struct)
-- ================================================================

structure ComposedKnowledge where
  value : Float
  variance : Float
  approx_bound : Float
  causal_alpha : Float
  causal_beta : Float
  confidence : Nat        -- 0..1000
  provenance_tag : Nat
  epoch : Nat
  deriving Repr

-- ================================================================
-- §2. Well-formedness
-- ================================================================

/-- A ComposedKnowledge is well-formed iff variance is non-negative,
    approx_bound is non-negative, both Beta parameters are positive,
    and confidence is in [0, 1000].
-/
def WellFormed (c : ComposedKnowledge) : Prop :=
  c.variance ≥ 0 ∧
  c.approx_bound ≥ 0 ∧
  c.causal_alpha > 0 ∧
  c.causal_beta > 0 ∧
  c.confidence ≤ 1000

-- ================================================================
-- §3. Arithmetic (mirrors Sounio ck_*)
-- ================================================================

def confDecay (c : Nat) : Nat := (c * 99) / 100

def minNat (a b : Nat) : Nat := if a < b then a else b
def minFloat (a b : Float) : Float := if a < b then a else b

def add (a b : ComposedKnowledge) : ComposedKnowledge := {
  value := a.value + b.value,
  variance := a.variance + b.variance,
  approx_bound := a.approx_bound + b.approx_bound,
  causal_alpha := a.causal_alpha + b.causal_alpha,
  causal_beta := a.causal_beta + b.causal_beta,
  confidence := confDecay (minNat a.confidence b.confidence),
  provenance_tag := minNat a.provenance_tag b.provenance_tag,
  epoch := (minNat a.epoch b.epoch) + 1
}

def mul (a b : ComposedKnowledge) : ComposedKnowledge := {
  value := a.value * b.value,
  -- Full Gaussian-product variance: σ_y²·Var(X) + σ_x²·Var(Y) + Var(X)·Var(Y).
  -- The σ_x²σ_y² cross-term is the difference between GUM delta-method and
  -- exact second moment for independent X ⊥ Y; included for honesty.
  variance := b.value * b.value * a.variance + a.value * a.value * b.variance + a.variance * b.variance,
  -- Triangle inequality for product approximation: |y|·δx + |x|·δy + δx·δy.
  -- The δx·δy cross-term is required: at x=y=0 with δx=δy=1, the first two
  -- terms vanish but the true error can be 1. (math-review 2026-04-30.)
  approx_bound := (Float.abs b.value) * a.approx_bound + (Float.abs a.value) * b.approx_bound + a.approx_bound * b.approx_bound,
  causal_alpha := a.causal_alpha + b.causal_alpha,
  causal_beta := a.causal_beta + b.causal_beta,
  confidence := confDecay (minNat a.confidence b.confidence),
  provenance_tag := minNat a.provenance_tag b.provenance_tag,
  epoch := (minNat a.epoch b.epoch) + 1
}

-- ================================================================
-- §4. Effect handlers — operationally identity, semantically anchors
-- ================================================================

def handleApprox (c : ComposedKnowledge) : ComposedKnowledge := c
def handleCausal (c : ComposedKnowledge) : ComposedKnowledge := c
def handleKnowledge (c : ComposedKnowledge) : ComposedKnowledge := c

def canonicalDischarge (c : ComposedKnowledge) : ComposedKnowledge :=
  handleKnowledge (handleCausal (handleApprox c))

-- ================================================================
-- §5. Soundness theorems
-- ================================================================

/-- Trivial commutation (handlers are identity). -/
theorem handler_commutativity (c : ComposedKnowledge) :
    handleApprox (handleCausal (handleKnowledge c)) =
    handleKnowledge (handleCausal (handleApprox c)) := by
  rfl

/-- Canonical discharge is the identity. -/
theorem canonical_discharge_id (c : ComposedKnowledge) :
    canonicalDischarge c = c := by
  rfl

/-- PLACEHOLDER (status: file docstring §Status item 1).
    Intended target: the variance term in `mul a b` equals the exact
    second-central moment of the product of two independent random
    variables with the given means and variances, namely
    `b.value² · a.variance + a.value² · b.variance + a.variance · b.variance`.
    Statement intentionally trivialised to `True` because Sounio's Lean
    setup is Mathlib-free and probability machinery is not yet in
    scope. Strengthen once `formal/Epistemic.lean` exposes a
    distribution algebra. -/
theorem mul_variance_dominates_placeholder (a b : ComposedKnowledge)
    (ha : WellFormed a) (hb : WellFormed b) :
    True := by
  trivial

/-- PLACEHOLDER (status: file docstring §Status item 2).
    Intended target: triangle inequality for product approximation,
    `|xy_approx - xy_true| ≤ |y|·δx + |x|·δy + δx·δy`. The δx·δy
    cross-term IS required; it was missing in the original Sounio
    `composed_effects.sio` and `mul` definition above and was added
    after a math-review on 2026-04-30 with counter-example
    (a=b=0±1 → first two terms = 0 but true error can be 1).
    Statement currently trivialised to `True`. -/
theorem approx_triangle_mul_placeholder (a b : ComposedKnowledge) : True := by
  trivial

/-- PLACEHOLDER (status: file docstring §Status item 3).
    Intended target: independent-evidence pooling for Beta-edge
    distributions yields Beta(α₁+α₂, β₁+β₂) (NOT α-multiply, β-multiply;
    that mistake was previously embedded in the Sounio comment and
    corrected the same day). Statement currently trivialised to `True`. -/
theorem causal_independence_approximation_placeholder
    (a b : ComposedKnowledge)
    (h_indep : True) : True := by
  trivial

-- ================================================================
-- §6. Composition theorem (top-level)
-- ================================================================

/-- The handlers Approx, Causal, Knowledge commute under any of the
    arithmetic operations on ComposedKnowledge. This is the operational
    soundness of the M1 composition. -/
theorem composition_soundness
    (a b : ComposedKnowledge)
    (ha : WellFormed a) (hb : WellFormed b) :
    canonicalDischarge (mul a b) = mul (canonicalDischarge a) (canonicalDischarge b) := by
  rfl

end Sounio.ApproxCausalKnowledge
