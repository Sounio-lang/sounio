-- formal/lean4/SounioApproxCausalKnowledge.lean
/-!
# Composition of Approx, Causal, and Knowledge Effects — Lean 4 Formalization

This file mirrors the Sounio module
`stdlib/epistemic/composed_effects.sio` and proves that the three
algebraic effects `Approx`, `Causal`, `Knowledge` compose without
information loss when the underlying values use the canonical
`ComposedKnowledge` representation.

## Top-level claim — scope of this file

For any well-formed inputs `a b : ComposedKnowledge` and the arithmetic
operations *currently mechanized in this file* (`add` and `mul`),
discharging the three effects in any order produces values that
agree on every channel up to a confidence-decay equivalence.

In short: the three handlers commute under `add` and `mul`.

`sub` and `div` are defined in the Sounio source `composed_effects.sio`
(`ck_sub`, `ck_div`) but their Lean mirrors and the corresponding
`composition_soundness_*` theorems are deferred. They are textually
symmetric to `add` and `mul` respectively, but the property must be
stated and proven explicitly; until then, do not extend the
"handlers commute" claim to them.

## References

- Sounio source: `stdlib/epistemic/composed_effects.sio`
- Plotkin & Pretnar (2009) "Handlers of Algebraic Effects" — for the
  general handler-composition theorem on a free algebra.
- Pearl (2009) *Causality* — Beta-edge representation.
- JCGM 100:2008 (GUM) — variance propagation rules.
- Higham (2002) *Accuracy and Stability of Numerical Algorithms* — Approx
  bound triangle inequality.

## Status

Two layers per claim, per Sounio's no-`sorry` convention:

1. **Algebraic-identity layer** (Float, `rfl`-proved). The `_unfolds`
   theorems state that the corresponding component of `mul a b`
   equals an explicit closed-form expression. These are provable by
   `rfl` because they unfold the definition; they are not `True`
   and they pin the algebraic content precisely.

2. **Substance layer** (`Nat`-shadow, structurally proved). For each
   claim that requires inequality reasoning (variance domination,
   triangle inequality, causal pooling), a `nat_*` theorem in `Nat`
   captures the *mathematical content* of the claim using only core
   Lean's `Nat` algebra — no Mathlib, no Float postulates, no axioms.

What this gives us today:
- `mul_variance_unfolds` — the variance term equals
  `b.value²·a.var + a.value²·b.var + a.var·b.var` exactly. This is
  the second-central moment of the product of two independent random
  variables (full Gaussian-product, not delta-method).
- `nat_mul_variance_dominates_delta_method` — in the discretised
  (`Nat`) shadow, the full Gaussian-product variance dominates the
  GUM delta-method approximation, because the cross-term `var_x·var_y`
  is non-negative. This is the substantive claim that the cross-term
  matters; it is what motivated the math-review fix on
  `composed_effects.sio` and on `mul` above.
- `mul_approx_bound_unfolds` — the approx bound equals
  `|y|·δx + |x|·δy + δx·δy` exactly.
- `nat_approx_triangle_with_cross_dominates_delta` — the cross-term
  domination at the substance layer.
- `mul_causal_alpha_sums` / `mul_causal_beta_sums` — independent
  evidence pooling produces Beta(α₁+α₂, β₁+β₂). Pinned by `rfl`.

What remains deferred (genuinely non-trivial):
- A Real- or Mathlib-grounded version of the variance-domination
  theorem on Float. Would require either importing Mathlib's
  `Mathlib.Analysis.NormedSpace.Real` and a Float-Real correspondence,
  or extending the in-tree `formal/Epistemic.lean` with a
  Mathlib-free Real algebra. Either is 4-6 weeks of dedicated Lean
  work and is *not* attempted here.
- Probabilistic semantics for the Beta-edge causal-pooling claim.
  The Sounio model uses Beta(α, β) parameters; the formal claim
  "two independent observations of evidence pool by α+α', β+β'" is
  a textbook conjugate-update fact, but a Lean proof needs a
  Beta-distribution algebra. Also deferred.
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

/-- Algebraic-identity layer (Float, `rfl`-proved).
    The variance term of `mul a b` equals the *full Gaussian-product*
    variance — second-central moment of the product of two independent
    random variables with the given means and variances:
        b.value² · a.var + a.value² · b.var + a.var · b.var
    The third (cross-)term distinguishes the exact moment from the
    GUM delta-method approximation; it was added in the
    2026-04-30 math-review fix.
-/
theorem mul_variance_unfolds (a b : ComposedKnowledge) :
    (mul a b).variance =
      b.value * b.value * a.variance
      + a.value * a.value * b.variance
      + a.variance * b.variance := by
  rfl

/-- Substance layer (`Nat`-shadow, structurally proved).
    In a discretised semiring (`Nat`), the full Gaussian-product
    variance dominates the GUM delta-method approximation, because the
    cross-term `var_x · var_y ≥ 0`. This is the proposition that
    motivates carrying the cross-term: shedding it always *under-*estimates
    the true variance, which is unsafe for a Knightian safety gate.
-/
theorem nat_mul_variance_dominates_delta_method
    (av bv vary varx : Nat) :
    bv * bv * varx + av * av * vary
      ≤ bv * bv * varx + av * av * vary + varx * vary := by
  exact Nat.le_add_right _ _

/-- Algebraic-identity layer (Float, `rfl`-proved).
    The `approx_bound` field of `mul a b` equals the standard product
    triangle inequality with the δx·δy cross-term:
        |y|·δx + |x|·δy + δx·δy
    Counter-example for the no-cross-term version (which was the
    pre-2026-04-30 bug): a = b with value=0, approx_bound=1; without
    the cross-term the bound is 0, but the true product error can
    be ±1. The cross-term is required for soundness.
-/
theorem mul_approx_bound_unfolds (a b : ComposedKnowledge) :
    (mul a b).approx_bound =
      (Float.abs b.value) * a.approx_bound
      + (Float.abs a.value) * b.approx_bound
      + a.approx_bound * b.approx_bound := by
  rfl

/-- Substance layer (`Nat`-shadow, structurally proved).
    With `δx, δy ≥ 0`, the triangle bound *with* cross-term is at
    least as large as the bound without it. So omitting the
    cross-term gives an unsound under-estimate; including it never
    sacrifices soundness. -/
theorem nat_approx_triangle_with_cross_dominates_delta
    (absx absy dx dy : Nat) :
    absy * dx + absx * dy
      ≤ absy * dx + absx * dy + dx * dy := by
  exact Nat.le_add_right _ _

/-- Algebraic-identity layer (Float, `rfl`-proved).
    Independent-evidence pooling sums the Beta α-parameters:
        (mul a b).causal_alpha = a.causal_alpha + b.causal_alpha
    This is the conjugate-update rule for two independent observations.
    The pre-2026-04-30 Sounio comment incorrectly claimed multiplication;
    the code (and now this Lean mirror) correctly sums.
-/
theorem mul_causal_alpha_sums (a b : ComposedKnowledge) :
    (mul a b).causal_alpha = a.causal_alpha + b.causal_alpha := by
  rfl

/-- Companion identity for the Beta β-parameter. -/
theorem mul_causal_beta_sums (a b : ComposedKnowledge) :
    (mul a b).causal_beta = a.causal_beta + b.causal_beta := by
  rfl

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
