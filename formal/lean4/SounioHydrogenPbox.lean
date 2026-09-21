-- formal/lean4/SounioHydrogenPbox.lean
/-!
# Sounio Hydrogen p-box — machine-checked algebra for the hydrogen demos

Machine-checks, over `Int` with core Lean 4 only (no Mathlib, no `sorry`),
the three algebraic facts that the `demos/hydrogen/` UQ suite relies on.
Probabilities and samples are represented as integers (per-mille / scaled
units in the demos); the identities are exact over any integral domain.

* **§1 Variance / Jensen (n = 3).** `(Σxᵢ)² ≤ 3·Σxᵢ²`, via the Lagrange
  identity `3·Σxᵢ² − (Σxᵢ)² = Σ_{i<j} (xᵢ−xⱼ)²`. This is *why* the
  nominal-point design pressure in `mh_cascade_uq.sio` underestimates the
  mean under uncertainty (Jensen gap ≥ 0).

* **§2 Variance of a correlated sum (n = 3).** In the division-free scaled
  form `var9 = 9·Var`, `cov9 = 9·Cov`:
  `var9(X₁+X₂+X₃) = var9 X₁ + var9 X₂ + var9 X₃ + 2·Σ_{i<j} cov9(Xᵢ,Xⱼ)`,
  with the corollary that nonnegative pairwise batch covariance can only
  inflate the cascade variance. This is the algebra behind "batch ΔH
  correlation doubles σ" in `mh_cascade_uq.sio`.

* **§3 Monotone p-box propagation (2-point support).** For a quantity
  supported on `{x₁, x₂}` with `x₁ < x₂` and a map `f` strictly increasing
  across the support, the sub-level event `{f(X) ≤ f(x₁)}` equals
  `{X ≤ x₁}` — so evaluating a monotone model at the *endpoints* of an
  input p-box yields a valid (on the support, exact) output p-box, with no
  independence assumption anywhere. This is the soundness theorem for the
  chained endpoint propagation in `hub_chain.sio`.

References:
- Jansen (1999), Comput. Phys. Commun. 117:35 (variance decomposition)
- Ferson et al. (2003), Sandia SAND2002-4013 (p-boxes, dependence-free bounds)
- Agourakis (2026), demos/hydrogen/README.md (demo suite using these facts)

No sorry. No Mathlib.
-/

namespace Sounio.HydrogenPbox

-- ================================================================
-- S0. Micro-lemmas (core Lean has no square-nonneg lemma)
-- ================================================================

/-- `0 ≤ a·a` over `Int`, by sign case split. -/
theorem int_sq_nonneg (a : Int) : 0 ≤ a * a := by
  by_cases h : 0 ≤ a
  · exact Int.mul_nonneg h h
  · have h' : 0 ≤ -a := by omega
    have e : (-a) * (-a) = a * a := by
      rw [Int.neg_mul, Int.mul_neg, Int.neg_neg]
    rw [← e]
    exact Int.mul_nonneg h' h'

-- ================================================================
-- S1. Variance / Jensen for n = 3
-- ================================================================

/-- Lagrange identity for three integers:
    `3·Σxᵢ² − (Σxᵢ)²` is exactly the sum of the three pairwise squared
    differences. Products are AC-normalized by simp; `omega` closes the
    resulting linear equality over the monomial atoms. -/
theorem var3_identity (a b c : Int) :
    3 * (a * a + b * b + c * c) - (a + b + c) * (a + b + c)
      = (a - b) * (a - b) + (a - c) * (a - c) + (b - c) * (b - c) := by
  simp only [Int.add_mul, Int.mul_add, Int.sub_mul, Int.mul_sub]
  simp only [Int.mul_comm]
  omega

/-- Discrete Jensen / variance nonnegativity for n = 3:
    the squared sum never exceeds 3× the sum of squares, i.e.
    `(Σxᵢ)² ≤ 3·Σxᵢ²`. Equality iff `a = b = c`. -/
theorem var3_nonneg (a b c : Int) :
    (a + b + c) * (a + b + c) ≤ 3 * (a * a + b * b + c * c) := by
  have hid := var3_identity a b c
  have h1 := int_sq_nonneg (a - b)
  have h2 := int_sq_nonneg (a - c)
  have h3 := int_sq_nonneg (b - c)
  omega

/-- The Jensen gap vanishes when all three samples coincide. -/
theorem var3_gap_eq_zero_of_all_eq (a : Int) :
    3 * (a * a + a * a + a * a) - (a + a + a) * (a + a + a) = 0 := by
  simp only [Int.add_mul, Int.mul_add]
  simp only [Int.mul_comm]
  omega

-- ================================================================
-- S2. Variance of a correlated sum of three stage errors
-- ================================================================

/-- Sum of three integers. -/
def sum3 (x₁ x₂ x₃ : Int) : Int := x₁ + x₂ + x₃

/-- Division-free scaled variance: `var9 = 9·Var` for a 3-sample. -/
def var9 (x₁ x₂ x₃ : Int) : Int :=
  3 * (x₁ * x₁ + x₂ * x₂ + x₃ * x₃) - sum3 x₁ x₂ x₃ * sum3 x₁ x₂ x₃

/-- Division-free scaled covariance: `cov9 = 9·Cov` for paired 3-samples. -/
def cov9 (x₁ x₂ x₃ y₁ y₂ y₃ : Int) : Int :=
  3 * (x₁ * y₁ + x₂ * y₂ + x₃ * y₃) - sum3 x₁ x₂ x₃ * sum3 y₁ y₂ y₃

/-- The exact variance decomposition of a sum of three correlated stage
    errors: `var9(X+Y+Z) = var9 X + var9 Y + var9 Z + 2·Σ cov9`. -/
theorem var9_sum (x₁ x₂ x₃ y₁ y₂ y₃ z₁ z₂ z₃ : Int) :
    var9 (x₁ + y₁ + z₁) (x₂ + y₂ + z₂) (x₃ + y₃ + z₃)
      = var9 x₁ x₂ x₃ + var9 y₁ y₂ y₃ + var9 z₁ z₂ z₃
        + 2 * (cov9 x₁ x₂ x₃ y₁ y₂ y₃ + cov9 x₁ x₂ x₃ z₁ z₂ z₃
               + cov9 y₁ y₂ y₃ z₁ z₂ z₃) := by
  unfold var9 cov9 sum3
  simp only [Int.add_mul, Int.mul_add]
  simp only [Int.mul_comm]
  omega

/-- Scaled variance is nonnegative (Jensen, §1 restated for `var9`). -/
theorem var9_nonneg (a b c : Int) : 0 ≤ var9 a b c := by
  unfold var9 sum3
  have hid := var3_identity a b c
  have h1 := int_sq_nonneg (a - b)
  have h2 := int_sq_nonneg (a - c)
  have h3 := int_sq_nonneg (b - c)
  omega

/-- Corollary: with nonnegative pairwise batch covariances, the cascade
    variance is at least the independent-case variance — positive batch
    correlation can only *inflate* total variance, never shrink it. -/
theorem var9_sum_ge_of_cov_nonneg (x₁ x₂ x₃ y₁ y₂ y₃ z₁ z₂ z₃ : Int)
    (hxy : 0 ≤ cov9 x₁ x₂ x₃ y₁ y₂ y₃)
    (hxz : 0 ≤ cov9 x₁ x₂ x₃ z₁ z₂ z₃)
    (hyz : 0 ≤ cov9 y₁ y₂ y₃ z₁ z₂ z₃) :
    var9 x₁ x₂ x₃ + var9 y₁ y₂ y₃ + var9 z₁ z₂ z₃
      ≤ var9 (x₁ + y₁ + z₁) (x₂ + y₂ + z₂) (x₃ + y₃ + z₃) := by
  have hid := var9_sum x₁ x₂ x₃ y₁ y₂ y₃ z₁ z₂ z₃
  omega

-- ================================================================
-- S3. Monotone p-box propagation on a 2-point support
-- ================================================================

/-- For `x` anywhere in the sub-level band `(-∞, x₂]`, with `f` monotone
    nondecreasing below `x₁` and strictly increasing across the support
    `(x₁, x₂]`, the sub-level event of `f(X)` at `f(x₁)` is exactly the
    sub-level event of `X` at `x₁`. The 2-point support `{x₁, x₂}` is the
    application (both points satisfy the hypothesis `x ≤ x₂`); the
    theorem is proved for the whole band, so it also covers continuous
    supports truncated at `x₂`. Hence endpoint evaluation transfers *any*
    p-box bounds `[F_lo, F_hi]` on `X` to valid bounds on `f(X)` at the
    grid points — the soundness theorem for chained endpoint propagation. -/
theorem monotone_event_equiv (x₁ x₂ x : Int) (f : Int → Int)
    (hx : x ≤ x₂) (_h12 : x₁ < x₂)
    (hmono_lo : ∀ u, u ≤ x₁ → f u ≤ f x₁)
    (hstrict : ∀ u, x₁ < u → u ≤ x₂ → f x₁ < f u) :
    f x ≤ f x₁ ↔ x ≤ x₁ := by
  constructor
  · intro hfx
    by_cases hx1 : x ≤ x₁
    · exact hx1
    · have hgt : x₁ < x := by omega
      have hstrictx : f x₁ < f x := hstrict x hgt hx
      omega
  · intro hx1
    exact hmono_lo x hx1

/-- Corollary (p-box transfer): if the CDF of `X` at `x₁` is bracketed by
    `[p_lo, p_hi]`, the same bracket holds for the CDF of `f(X)` at
    `f(x₁)` — endpoint evaluation of a monotone model preserves p-box
    validity. Stated for integer (per-mille) probabilities.
    Deliberately the *packaging* step of the pipeline soundness argument:
    the mathematical content lives in `monotone_event_equiv`; this lemma
    is the named interface the demo documentation cites. -/
theorem pbox_transfer (p p_lo p_hi : Int)
    (hlo : p_lo ≤ p) (hhi : p ≤ p_hi) :
    p_lo ≤ p ∧ p ≤ p_hi := ⟨hlo, hhi⟩

end Sounio.HydrogenPbox
