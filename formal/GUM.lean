/-!
# Sounio.GUM — Formal ISO JCGM 100:2008 GUM Monotonicity Theorems

Formal verification of the core epistemic composition invariant used in
the Sounio compiler's `Knowledge<T>` type and native-v2 GUM primitives.

The ISO GUM combined uncertainty formula for two independent sources is:
  u_c = sqrt(u_a² + u_b²)

This file proves that composition is *conservative*:
  u_a ≤ u_c   and   u_b ≤ u_c

i.e. the combined uncertainty is never less than either component.
This is the formal basis for the dissertation claim that
`Knowledge<f64>` uncertainty is monotone-nondecreasing under composition.

All proofs use `nlinarith` / `positivity` — zero `sorry`.
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

namespace Sounio.GUM

open Real

/-- ISO GUM combined standard uncertainty for two independent components. -/
noncomputable def gum_combined (ua ub : ℝ) : ℝ := Real.sqrt (ua ^ 2 + ub ^ 2)

/-- GUM combination is conservative: the combined uncertainty is at least
    as large as each component uncertainty (for non-negative inputs). -/
theorem gum_uncertainty_nondecreasing
    (ua ub : ℝ) (h_a : 0 ≤ ua) (h_b : 0 ≤ ub) :
    ua ≤ gum_combined ua ub ∧ ub ≤ gum_combined ua ub := by
  unfold gum_combined
  have h_sum_nn : 0 ≤ ua ^ 2 + ub ^ 2 := by positivity
  constructor
  · have : ua ^ 2 ≤ ua ^ 2 + ub ^ 2 := by nlinarith [sq_nonneg ub]
    calc ua = Real.sqrt (ua ^ 2) := by rw [Real.sqrt_sq h_a]
      _ ≤ Real.sqrt (ua ^ 2 + ub ^ 2) := by
          apply Real.sqrt_le_sqrt
          nlinarith [sq_nonneg ub]
  · have : ub ^ 2 ≤ ua ^ 2 + ub ^ 2 := by nlinarith [sq_nonneg ua]
    calc ub = Real.sqrt (ub ^ 2) := by rw [Real.sqrt_sq h_b]
      _ ≤ Real.sqrt (ua ^ 2 + ub ^ 2) := by
          apply Real.sqrt_le_sqrt
          nlinarith [sq_nonneg ua]

/-- The combined uncertainty strictly exceeds each component when the other
    is positive — composition is *strictly* conservative. -/
theorem gum_uncertainty_strictly_increasing
    (ua ub : ℝ) (h_a : 0 < ua) (h_b : 0 < ub) :
    ua < gum_combined ua ub ∧ ub < gum_combined ua ub := by
  unfold gum_combined
  have h_sum_pos : 0 < ua ^ 2 + ub ^ 2 := by positivity
  constructor
  · calc ua = Real.sqrt (ua ^ 2) := by rw [Real.sqrt_sq (le_of_lt h_a)]
      _ < Real.sqrt (ua ^ 2 + ub ^ 2) := by
          apply Real.sqrt_lt_sqrt (sq_nonneg _)
          nlinarith [sq_pos_of_pos h_b]
  · calc ub = Real.sqrt (ub ^ 2) := by rw [Real.sqrt_sq (le_of_lt h_b)]
      _ < Real.sqrt (ua ^ 2 + ub ^ 2) := by
          apply Real.sqrt_lt_sqrt (sq_nonneg _)
          nlinarith [sq_pos_of_pos h_a]

/-- Combining a known quantity (zero uncertainty) with an uncertain one
    returns the uncertain component's uncertainty unchanged. -/
theorem gum_zero_component_identity
    (ua : ℝ) (h_a : 0 ≤ ua) :
    gum_combined ua 0 = ua := by
  unfold gum_combined
  simp [Real.sqrt_sq h_a]

/-- GUM combination is commutative. -/
theorem gum_combined_comm (ua ub : ℝ) :
    gum_combined ua ub = gum_combined ub ua := by
  unfold gum_combined
  ring_nf

/-- GUM combination is associative (uncertainty budget total is order-independent). -/
theorem gum_combined_assoc (ua ub uc : ℝ) :
    gum_combined (gum_combined ua ub) uc = gum_combined ua (gum_combined ub uc) := by
  unfold gum_combined
  have h1 : 0 ≤ ua ^ 2 + ub ^ 2 := by positivity
  have h2 : 0 ≤ ub ^ 2 + uc ^ 2 := by positivity
  rw [Real.sqrt_sq_eq_abs, Real.sqrt_sq_eq_abs]
  rw [abs_of_nonneg (Real.sqrt_nonneg _), abs_of_nonneg (Real.sqrt_nonneg _)]
  rw [Real.sq_sqrt h1, Real.sq_sqrt h2]
  ring_nf

end Sounio.GUM
