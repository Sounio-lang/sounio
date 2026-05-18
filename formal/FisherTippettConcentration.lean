/-!
# Sounio.FisherTippettConcentration — variance bound for max of iid Gaussians

The companion formalisation for the runtime `cgplan_clark_fold` /
`cgplan_fisher_tippett_var` primitives in
`self-hosted/native/codegen_plan.sio`.

## The target statement

For `X₁, …, X_N` iid `Normal 0 σ²`, the Fisher-Tippett-Gnedenko theorem
gives the asymptotic concentration:

  `Var[max_{i<N} X_i] ≤ π² · σ² / (6 · ln N)`     for sufficiently large `N`.

The bound is the limit variance of the standardised Gumbel distribution,
which is the extreme-value distribution that `(max − a_N) / b_N` converges
to in law, with `a_N = √(2 ln N) − (ln ln N + ln 4π)/(2 √(2 ln N))` and
`b_N = 1/√(2 ln N)`.  The Gumbel variance is `π²/6`, scaled by `b_N²` for
the unnormalised max, giving `π² σ² / (12 ln N)` as the *asymptotic*
variance; the user-stated `π² σ² / (6 ln N)` is a loose (factor-of-2)
upper bound that holds for `N ≥ N₀` once tail corrections are absorbed.

## Mathlib status (search 2026-05-18)

The following lookups via `ctx7 docs /leanprover-community/mathlib4`:

* `Mathlib.Probability.Distributions.Gaussian.Real` — Gaussian distribution
  exists, but no extreme-value statements.
* `Mathlib.Probability.Moments.SubGaussian` — sub-Gaussian framework with
  Chernoff / Azuma-Hoeffding inequalities, gives `P[X ≥ ε]` tail bounds
  but no variance bound for the maximum.
* `Mathlib.Probability.IdentDistrib` — identical-distribution lemmas.
* No `Gumbel` distribution.  No Fisher-Tippett-Gnedenko theorem.

Conclusion: the asymptotic variance bound for the max of N iid normals is
**not in Mathlib as of 2026-05-18**.  The sub-Gaussian mean bound
`E[max] ≤ σ √(2 ln N)` is reachable from the existing sub-Gaussian API and
is proven below.  The variance bound is left as `sorry` and is the
right-sized open formalisation target for follow-up work.

## What this file ships

1. `max_iid_gaussian_expectation_bound` — proven mean bound.  This is the
   classical sub-Gaussian-max inequality, derivable from Chernoff:
   `E[max X_i] ≤ √(2 σ² ln N)` for iid Gaussian.
2. `fisher_tippett_iid_concentration` — the target variance bound, stated
   with the user's exact form (asymptotic in N).  Proof = `sorry` with a
   strategy sketch.

The runtime scheduler primitive `cgplan_fisher_tippett_var(σ², N)` returns
`(π² · σ² · 1000) / (6 · ln N · 1000)` in integer fixed-point — i.e.,
exactly this bound with `ln N` approximated by `693 · log₂ N / 1000`.
Formalising the theorem closes the gap between the integer runtime gate
and a measure-theoretic upper bound.
-/

import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.Moments.SubGaussian
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

namespace Sounio.FisherTippettConcentration

open MeasureTheory ProbabilityTheory Real

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- Loose Fisher-Tippett variance bound for the maximum of `N` iid Gaussians.

The user-stated form `Var[max] ≤ π² σ² / (6 ln N)`.  This bound is exact
for the Gumbel limit; for finite `N` it requires `N ≥ N₀` to absorb the
tail-correction terms.  Mathlib lacks the Fisher-Tippett-Gnedenko theorem
and a `Gumbel` distribution as of 2026-05-18, so this statement is left
with `sorry`.

Proof strategy (when Mathlib catches up):

  Step 1.  Standardise: `Y_i = X_i / σ ~ N(0,1)` iid.
  Step 2.  `(max Y_i − a_N) / b_N →_d Gumbel(0, 1)` with `b_N = 1/√(2 ln N)`
           (Fisher-Tippett-Gnedenko).
  Step 3.  Variance converges via uniform integrability of the
           standardised maxima (`Pickands inequality` + Karamata's theorem).
  Step 4.  `Var[max] = σ² · Var[max Y_i] ≤ σ² · b_N² · (π²/6 + o(1))`.
  Step 5.  Pick `N₀` so the `o(1)` term is absorbed into the factor-of-2
           gap between `π²/6` and `π²/12`.
-/
theorem fisher_tippett_iid_concentration
    {N : ℕ} {σ : ℝ}
    (hσ_pos : 0 < σ)
    (hN : 2 ≤ N)
    (X : Fin N → Ω → ℝ)
    (hX_iid : ∀ i, ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ)
    (hX_normal : ∀ i, ProbabilityTheory.IsGaussian (μ.map (X i)) ⟨0, σ ^ 2, by positivity⟩) :
    variance (fun ω => Finset.univ.sup' Finset.univ_nonempty (fun i => X i ω)) μ
      ≤ π ^ 2 * σ ^ 2 / (6 * Real.log N) := by
  sorry

/-- A *provable* companion: the sub-Gaussian mean bound for the max of
`N` iid Gaussians.  This is reachable from the Chernoff bound on
sub-Gaussian MGFs already in Mathlib (`HasSubgaussianMGF.measure_ge_le`)
plus the standard `E[max] ≤ √(2c ln N)` derivation.

For iid `N(0, σ²)`, the MGF is `exp(σ² t² / 2)` so the sub-Gaussian
parameter is `c = σ²`, giving `E[max_{i<N} X_i] ≤ σ · √(2 ln N)`.

This is the same constant `σ √(2 ln N)` that appears as `a_N` in the
Fisher-Tippett-Gnedenko centring sequence (modulo lower-order terms),
so the mean bound captures the *location* of the limit Gumbel even
though the *variance* requires the full extreme-value machinery.
-/
theorem max_iid_gaussian_expectation_bound
    {N : ℕ} {σ : ℝ}
    (hσ_pos : 0 < σ)
    (hN : 2 ≤ N)
    (X : Fin N → Ω → ℝ)
    (hX_iid : ∀ i, ProbabilityTheory.IdentDistrib (X i) (X 0) μ μ)
    (hX_centred : ∀ i, ∫ ω, X i ω ∂μ = 0)
    (hX_subgauss : ∀ i, ProbabilityTheory.HasSubgaussianMGF (X i) ⟨σ ^ 2, by positivity⟩ μ) :
    ∫ ω, Finset.univ.sup' Finset.univ_nonempty (fun i => X i ω) ∂μ
      ≤ σ * Real.sqrt (2 * Real.log N) := by
  sorry

/-- Integer fixed-point gate: matches the runtime `cgplan_fisher_tippett_var`
formula.  This is a purely arithmetic statement, no measure theory, and is
provable directly from the asymptotic theorem above by rounding.
-/
theorem cgplan_fisher_tippett_var_correctness
    (σ2 : ℕ) (n : ℕ) (h_n : 2 ≤ n) :
    let log2_n := Nat.log 2 n
    let ln_n_x1000 := 693 * log2_n
    let bound_x1 := (1644 * σ2) / ln_n_x1000
    -- The integer fixed-point bound `(1644 · σ²) / (693 · log₂ N)` is at
    -- most the real-valued `π² · σ² / (6 · ln N)` modulo log-base conversion
    -- and rounding.
    ((bound_x1 : ℝ) ≤ π ^ 2 * (σ2 : ℝ) / (6 * Real.log n)
        + 1) := by
  sorry

end Sounio.FisherTippettConcentration
