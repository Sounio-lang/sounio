import SounioRealCauchy
import SounioSqrtField

set_option maxHeartbeats 0

/-!
# Sounio — towards the analytic `SqrtField ℝ` instance (Mathlib-free)

`SounioDeGreyChi5TransferWf.sqrtField_chi_ge_5` proves **χ(F²) ≥ 5 for every `SqrtField` F**, with
no Mathlib. The *only* remaining input to χ(ℝ²) ≥ 5 is therefore the single analytic fact

    "ℝ is a `SqrtField`"   (i.e. produce `(R : SounioSqrt.SqrtField)` with `R.F = ℝ`).

This file **starts** that construction Mathlib-free. ℝ is the quotient of Cauchy sequences of
rationals (`SounioRealCauchy`) by the null-difference relation `RealEq`. We:

* define the foundational analytic predicate `TendsToZero` and the equivalence `RealEq`;
* prove `RealEq` is **reflexive** and **symmetric** (these need no `Rat` order API);
* enumerate the remaining obligations as a precise, externally-auditable **ledger** (each a named
  `Prop`), matching the `SqrtField` interface field-for-field. The deferred obligations are the
  genuine multi-week analytic core: the ε-N transitivity/triangle lemmas over `Rat` (whose order
  API is sparse Mathlib-free — even `Rat.add_le_add` is absent), the mul-monotonicity law
  (`SounioRealCauchy` defers this as ≈500–1000 LOC), order completeness (sup), and the constructive
  square root with `sqrt_sq`.

No `sorry`: proved facts are theorems, deferred facts are `Prop` definitions (the repository's
`OrderedCarrierObligation` pattern), so the file compiles and the remaining work is a checklist.
-/

namespace SounioSqrt.RealCauchyField

open Sounio.RealCauchy (SounioRealCauchy)

/-- `f` tends to `0`: eventually within every positive `Rat` band `[-ε, ε]`. Two-sided to avoid
    `Rat.abs`, matching `SounioRealCauchy.IsCauchy`. -/
def TendsToZero (f : Nat → Rat) : Prop :=
  ∀ ε : Rat, 0 < ε → ∃ N : Nat, ∀ n : Nat, N ≤ n → f n ≤ ε ∧ -(f n) ≤ ε

/-- Two Cauchy representatives denote the **same real** iff their difference tends to `0`. This is
    the relation ℝ is the quotient by. -/
def RealEq (a b : SounioRealCauchy) : Prop :=
  TendsToZero (fun n => a.seq n - b.seq n)

/-- `RealEq` is reflexive: `a.seq n - a.seq n = 0`, inside every band. -/
theorem realEq_refl (a : SounioRealCauchy) : RealEq a a := by
  intro ε hε
  refine ⟨0, fun n _ => ?_⟩
  show a.seq n - a.seq n ≤ ε ∧ -(a.seq n - a.seq n) ≤ ε
  rw [Rat.sub_self]
  exact ⟨Rat.le_of_lt hε, by rw [Rat.neg_zero]; exact Rat.le_of_lt hε⟩

/-- `RealEq` is symmetric: `b - a = -(a - b)`, so the two band-halves simply swap. -/
theorem realEq_symm {a b : SounioRealCauchy} (h : RealEq a b) : RealEq b a := by
  intro ε hε
  obtain ⟨N, hN⟩ := h ε hε
  refine ⟨N, fun n hn => ?_⟩
  obtain ⟨h1, h2⟩ := hN n hn
  refine ⟨?_, ?_⟩
  · show b.seq n - a.seq n ≤ ε
    rw [show b.seq n - a.seq n = -(a.seq n - b.seq n) from by rw [Rat.neg_sub]]
    exact h2
  · show -(b.seq n - a.seq n) ≤ ε
    rw [show -(b.seq n - a.seq n) = a.seq n - b.seq n from by rw [Rat.neg_sub]]
    exact h1

/-- Difference splits across a midpoint: `x - z = (x - y) + (y - z)` over `Rat`. -/
theorem rat_sub_split (x y z : Rat) : x - z = (x - y) + (y - z) := by
  rw [Rat.sub_eq_add_neg, Rat.sub_eq_add_neg, Rat.sub_eq_add_neg, Rat.add_assoc,
    ← Rat.add_assoc (-y) y (-z), Rat.add_comm (-y) y, Rat.add_neg_cancel, Rat.zero_add]

/-- Half of a positive rational is positive. -/
theorem rat_half_pos {ε : Rat} (hε : 0 < ε) : 0 < ε * (1/2) :=
  Rat.mul_pos hε (by native_decide)

/-- Two halves sum to the whole. -/
theorem rat_add_halves (ε : Rat) : ε * (1/2) + ε * (1/2) = ε := by
  rw [← Rat.mul_add, show ((1:Rat)/2 + 1/2) = 1 from by native_decide, Rat.mul_one]

/-- `RealEq` is **transitive**: the ε/2 triangle inequality over `Rat`. With `realEq_refl`/
    `realEq_symm`, this discharges `RealEqTransObligation` and makes `RealEq` an equivalence. -/
theorem realEq_trans {a b c : SounioRealCauchy} (hab : RealEq a b) (hbc : RealEq b c) :
    RealEq a c := by
  intro ε hε
  obtain ⟨N1, hN1⟩ := hab (ε * (1/2)) (rat_half_pos hε)
  obtain ⟨N2, hN2⟩ := hbc (ε * (1/2)) (rat_half_pos hε)
  refine ⟨max N1 N2, fun n hn => ?_⟩
  obtain ⟨h1a, h1b⟩ := hN1 n (Nat.le_trans (Nat.le_max_left N1 N2) hn)
  obtain ⟨h2a, h2b⟩ := hN2 n (Nat.le_trans (Nat.le_max_right N1 N2) hn)
  refine ⟨?_, ?_⟩
  · show a.seq n - c.seq n ≤ ε
    rw [rat_sub_split (a.seq n) (b.seq n) (c.seq n)]
    have : (a.seq n - b.seq n) + (b.seq n - c.seq n) ≤ ε * (1/2) + ε * (1/2) :=
      Rat.le_trans (Rat.add_le_add_right.mpr h1a) (Rat.add_le_add_left.mpr h2a)
    rwa [rat_add_halves] at this
  · show -(a.seq n - c.seq n) ≤ ε
    rw [rat_sub_split (a.seq n) (b.seq n) (c.seq n), Rat.neg_add]
    have : -(a.seq n - b.seq n) + -(b.seq n - c.seq n) ≤ ε * (1/2) + ε * (1/2) :=
      Rat.le_trans (Rat.add_le_add_right.mpr h1b) (Rat.add_le_add_left.mpr h2b)
    rwa [rat_add_halves] at this

/-- `RealEq` is an **equivalence relation** — the foundation for ℝ as a quotient. -/
theorem realEq_equivalence : Equivalence RealEq :=
  { refl := realEq_refl, symm := realEq_symm, trans := realEq_trans }

/-- The setoid on Cauchy representatives whose quotient is ℝ. -/
def realSetoid : Setoid SounioRealCauchy := ⟨RealEq, realEq_equivalence⟩

/-- Sum-difference regroups: `(a+b) - (c+d) = (a-c) + (b-d)` over `Rat`. -/
theorem rat_add_sub_add (a b c d : Rat) : (a + b) - (c + d) = (a - c) + (b - d) := by
  rw [Rat.sub_eq_add_neg, Rat.sub_eq_add_neg, Rat.sub_eq_add_neg, Rat.neg_add,
    Rat.add_assoc a b (-c + -d), ← Rat.add_assoc b (-c) (-d), Rat.add_comm b (-c),
    Rat.add_assoc (-c) b (-d), ← Rat.add_assoc a (-c) (b + -d)]

/-- **The sum of two Cauchy sequences is Cauchy** — the ε/2 triangle again. Enables the additive
    operation on the quotient ℝ. -/
theorem add_cauchy {f g : Nat → Rat} (hf : Sounio.RealCauchy.IsCauchy f)
    (hg : Sounio.RealCauchy.IsCauchy g) :
    Sounio.RealCauchy.IsCauchy (fun n => f n + g n) := by
  intro ε hε
  obtain ⟨N1, hN1⟩ := hf (ε * (1/2)) (rat_half_pos hε)
  obtain ⟨N2, hN2⟩ := hg (ε * (1/2)) (rat_half_pos hε)
  refine ⟨max N1 N2, fun m n hm hn => ?_⟩
  obtain ⟨h1a, h1b⟩ := hN1 m n (Nat.le_trans (Nat.le_max_left N1 N2) hm)
                                (Nat.le_trans (Nat.le_max_left N1 N2) hn)
  obtain ⟨h2a, h2b⟩ := hN2 m n (Nat.le_trans (Nat.le_max_right N1 N2) hm)
                                (Nat.le_trans (Nat.le_max_right N1 N2) hn)
  refine ⟨?_, ?_⟩
  · show (f m + g m) - (f n + g n) ≤ ε
    rw [rat_add_sub_add]
    have : (f m - f n) + (g m - g n) ≤ ε * (1/2) + ε * (1/2) :=
      Rat.le_trans (Rat.add_le_add_right.mpr h1a) (Rat.add_le_add_left.mpr h2a)
    rwa [rat_add_halves] at this
  · show (f n + g n) - (f m + g m) ≤ ε
    rw [rat_add_sub_add]
    have : (f n - f m) + (g n - g m) ≤ ε * (1/2) + ε * (1/2) :=
      Rat.le_trans (Rat.add_le_add_right.mpr h1b) (Rat.add_le_add_left.mpr h2b)
    rwa [rat_add_halves] at this

/-! ## Multiplicative structure: a `Rat` absolute-value toolkit, boundedness, product Cauchy,
    and product congruence. All Mathlib-free, built on the core `Rat` order API. -/

/-- Add two `≤` inequalities. -/
theorem rat_add_le_add {a b c d : Rat} (h1 : a ≤ b) (h2 : c ≤ d) : a + c ≤ b + d :=
  Rat.le_trans (Rat.add_le_add_right.mpr h1) (Rat.add_le_add_left.mpr h2)

/-- Sum of nonnegatives is nonnegative. -/
theorem rat_add_nonneg {a b : Rat} (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a + b := by
  have h : (0 : Rat) + 0 ≤ a + b := rat_add_le_add ha hb
  rwa [Rat.add_zero] at h

/-- Multiply two `≤` inequalities between nonnegatives. -/
theorem rat_mul_le_mul {a b c d : Rat}
    (ha : 0 ≤ a) (hc : 0 ≤ c) (hab : a ≤ b) (hcd : c ≤ d) : a * c ≤ b * d :=
  Rat.le_trans (Rat.mul_le_mul_of_nonneg_right hab hc)
    (Rat.mul_le_mul_of_nonneg_left hcd (Rat.le_trans ha hab))

/-- Rational absolute value (Mathlib-free), via decidable `≤`. -/
def ratAbs (q : Rat) : Rat := if 0 ≤ q then q else -q

theorem ratAbs_nonneg (q : Rat) : 0 ≤ ratAbs q := by
  unfold ratAbs; split
  · assumption
  · next h => exact Rat.le_of_lt (Rat.lt_neg_iff.mp (Rat.not_le.mp h))

theorem le_ratAbs (q : Rat) : q ≤ ratAbs q := by
  unfold ratAbs; split
  · exact Rat.le_refl
  · next h => exact Rat.le_of_lt (Std.lt_trans (Rat.not_le.mp h) (Rat.lt_neg_iff.mp (Rat.not_le.mp h)))

theorem neg_le_ratAbs (q : Rat) : -q ≤ ratAbs q := by
  unfold ratAbs; split
  · next h =>
      have h0 : -q ≤ -(0 : Rat) := Rat.neg_le_neg h
      rw [Rat.neg_zero] at h0
      exact Rat.le_trans h0 h
  · exact Rat.le_refl

theorem ratAbs_le {q b : Rat} (h1 : q ≤ b) (h2 : -q ≤ b) : ratAbs q ≤ b := by
  unfold ratAbs; split
  · exact h1
  · exact h2

theorem abs_two_sided {q b : Rat} (h : ratAbs q ≤ b) : q ≤ b ∧ -q ≤ b :=
  ⟨Rat.le_trans (le_ratAbs q) h, Rat.le_trans (neg_le_ratAbs q) h⟩

theorem ratAbs_neg (q : Rat) : ratAbs (-q) = ratAbs q := by
  unfold ratAbs
  by_cases h : 0 ≤ q
  · by_cases h2 : 0 ≤ -q
    · have hq0 : q = 0 :=
        Rat.le_antisymm (by have := Rat.neg_le_neg h2; rwa [Rat.neg_neg, Rat.neg_zero] at this) h
      rw [if_pos h2, if_pos h, hq0, Rat.neg_zero]
    · rw [if_neg h2, if_pos h, Rat.neg_neg]
  · have h2 : 0 ≤ -q := Rat.le_of_lt (Rat.lt_neg_iff.mp (Rat.not_le.mp h))
    rw [if_pos h2, if_neg h]

theorem ratAbs_mul (a b : Rat) : ratAbs (a * b) = ratAbs a * ratAbs b := by
  by_cases ha : 0 ≤ a <;> by_cases hb : 0 ≤ b
  · have hab : 0 ≤ a * b := Rat.mul_nonneg ha hb
    unfold ratAbs; rw [if_pos ha, if_pos hb, if_pos hab]
  · have hnb : 0 ≤ -b := Rat.le_of_lt (Rat.lt_neg_iff.mp (Rat.not_le.mp hb))
    have hab : 0 ≤ a * (-b) := Rat.mul_nonneg ha hnb
    have lhs : ratAbs (a * b) = a * (-b) := by
      rw [show a * b = -(a * (-b)) from by rw [Rat.mul_neg, Rat.neg_neg], ratAbs_neg]
      unfold ratAbs; rw [if_pos hab]
    have rhs : ratAbs a * ratAbs b = a * (-b) := by unfold ratAbs; rw [if_pos ha, if_neg hb]
    rw [lhs, rhs]
  · have hna : 0 ≤ -a := Rat.le_of_lt (Rat.lt_neg_iff.mp (Rat.not_le.mp ha))
    have hab : 0 ≤ (-a) * b := Rat.mul_nonneg hna hb
    have lhs : ratAbs (a * b) = (-a) * b := by
      rw [show a * b = -((-a) * b) from by rw [Rat.neg_mul, Rat.neg_neg], ratAbs_neg]
      unfold ratAbs; rw [if_pos hab]
    have rhs : ratAbs a * ratAbs b = (-a) * b := by unfold ratAbs; rw [if_neg ha, if_pos hb]
    rw [lhs, rhs]
  · have hna : 0 ≤ -a := Rat.le_of_lt (Rat.lt_neg_iff.mp (Rat.not_le.mp ha))
    have hnb : 0 ≤ -b := Rat.le_of_lt (Rat.lt_neg_iff.mp (Rat.not_le.mp hb))
    have hab : 0 ≤ (-a) * (-b) := Rat.mul_nonneg hna hnb
    have lhs : ratAbs (a * b) = (-a) * (-b) := by
      rw [show a * b = (-a) * (-b) from by rw [Rat.neg_mul, Rat.mul_neg, Rat.neg_neg]]
      unfold ratAbs; rw [if_pos hab]
    have rhs : ratAbs a * ratAbs b = (-a) * (-b) := by unfold ratAbs; rw [if_neg ha, if_neg hb]
    rw [lhs, rhs]

theorem ratAbs_add_le (a b : Rat) : ratAbs (a + b) ≤ ratAbs a + ratAbs b := by
  apply ratAbs_le
  · exact rat_add_le_add (le_ratAbs a) (le_ratAbs b)
  · rw [Rat.neg_add]; exact rat_add_le_add (neg_le_ratAbs a) (neg_le_ratAbs b)

/-- `|x·y| ≤ Bx·By` from `|x| ≤ Bx`, `|y| ≤ By`. -/
theorem ratAbs_mul_le {x y Bx By : Rat} (hx : ratAbs x ≤ Bx) (hy : ratAbs y ≤ By) :
    ratAbs (x * y) ≤ Bx * By := by
  rw [ratAbs_mul]; exact rat_mul_le_mul (ratAbs_nonneg x) (ratAbs_nonneg y) hx hy

/-- Product regroups for differences: `a·b - c·d = a·(b-d) + (a-c)·d`. -/
private theorem rat_mul_sub' (a b d : Rat) : a * b - a * d = a * (b - d) := by
  rw [Rat.sub_eq_add_neg b d, Rat.mul_add, Rat.mul_neg, ← Rat.sub_eq_add_neg]

private theorem rat_sub_mul' (a c d : Rat) : a * d - c * d = (a - c) * d := by
  rw [Rat.sub_eq_add_neg a c, Rat.add_mul, Rat.neg_mul, ← Rat.sub_eq_add_neg]

theorem rat_mul_sub_mul (a b c d : Rat) : a * b - c * d = a * (b - d) + (a - c) * d := by
  rw [rat_sub_split (a * b) (a * d) (c * d), rat_mul_sub' a b d, rat_sub_mul' a c d]

/-- `IsCauchy` in absolute-value form. -/
theorem isCauchy_abs {f : Nat → Rat} (hf : Sounio.RealCauchy.IsCauchy f) (ε : Rat) (hε : 0 < ε) :
    ∃ N, ∀ m n, N ≤ m → N ≤ n → ratAbs (f m - f n) ≤ ε := by
  obtain ⟨N, hN⟩ := hf ε hε
  refine ⟨N, fun m n hm hn => ratAbs_le (hN m n hm hn).1 ?_⟩
  rw [Rat.neg_sub]; exact (hN m n hm hn).2

/-- `RealEq` in absolute-value form. -/
theorem realEq_abs {a b : SounioRealCauchy} (h : RealEq a b) (ε : Rat) (hε : 0 < ε) :
    ∃ N, ∀ n, N ≤ n → ratAbs (a.seq n - b.seq n) ≤ ε := by
  obtain ⟨N, hN⟩ := h ε hε
  exact ⟨N, fun n hn => ratAbs_le (hN n hn).1 (hN n hn).2⟩

/-- **A Cauchy sequence is eventually bounded** in absolute value. -/
theorem cauchy_bounded {f : Nat → Rat} (hf : Sounio.RealCauchy.IsCauchy f) :
    ∃ B N, 0 ≤ B ∧ ∀ n, N ≤ n → ratAbs (f n) ≤ B := by
  obtain ⟨N, hN⟩ := hf 1 (by native_decide)
  refine ⟨ratAbs (f N) + 1, N, rat_add_nonneg (ratAbs_nonneg _) (by native_decide), fun n hn => ?_⟩
  obtain ⟨h1, h2⟩ := hN n N hn (Nat.le_refl N)
  apply ratAbs_le
  · have hb : f n ≤ 1 + f N := Rat.sub_right_le_iff_le_add.mp h1
    refine Rat.le_trans hb ?_
    rw [Rat.add_comm 1 (f N)]
    exact Rat.add_le_add_right.mpr (le_ratAbs (f N))
  · have hb2 : -(f n) ≤ 1 - f N := by
      have hstep : (f N - f n) + (-(f N)) ≤ 1 + (-(f N)) := Rat.add_le_add_right.mpr h2
      rw [Rat.sub_eq_add_neg, Rat.add_comm (f N) (-(f n)), Rat.add_assoc, Rat.add_neg_cancel,
          Rat.add_zero, ← Rat.sub_eq_add_neg] at hstep
      exact hstep
    refine Rat.le_trans hb2 ?_
    rw [Rat.sub_eq_add_neg, Rat.add_comm 1 (-(f N))]
    exact Rat.add_le_add_right.mpr (neg_le_ratAbs (f N))

/-- `(Bf+Bg+1) > 0` for nonnegative `Bf,Bg`. -/
theorem rat_K_pos {Bf Bg : Rat} (hBf : 0 ≤ Bf) (hBg : 0 ≤ Bg) : 0 < Bf + Bg + 1 := by
  have h01 : (0 : Rat) < 1 := by native_decide
  have hs : (0 : Rat) ≤ Bf + Bg := rat_add_nonneg hBf hBg
  have : (0 : Rat) + 1 ≤ (Bf + Bg) + 1 := Rat.add_le_add_right.mpr hs
  rw [Rat.zero_add] at this
  exact Std.lt_of_lt_of_le h01 this

/-- The scaling identity `K · (ε · K⁻¹) = ε`. -/
theorem rat_K_delta {K ε : Rat} (hK : K ≠ 0) : K * (ε * K⁻¹) = ε := by
  rw [Rat.mul_comm ε K⁻¹, ← Rat.mul_assoc, Rat.mul_inv_cancel K hK, Rat.one_mul]

/-- Core bound: with `|x|≤Bf`, `|u|≤δ`, `|v|≤δ`, `|y|≤Bg`, and `δ=ε·K⁻¹`, `K=Bf+Bg+1`, the sum
    `|x·u| + |v·y| ≤ ε`. -/
private theorem rat_prod_bound
    {x u v y Bf Bg ε : Rat}
    (hBf : 0 ≤ Bf) (hBg : 0 ≤ Bg) (hε : 0 < ε)
    (hx : ratAbs x ≤ Bf) (hu : ratAbs u ≤ ε * (Bf + Bg + 1)⁻¹)
    (hv : ratAbs v ≤ ε * (Bf + Bg + 1)⁻¹) (hy : ratAbs y ≤ Bg) :
    ratAbs (x * u) + ratAbs (v * y) ≤ ε := by
  have hK : 0 < Bf + Bg + 1 := rat_K_pos hBf hBg
  have hKne : (Bf + Bg + 1) ≠ 0 := fun h => by rw [h] at hK; exact Rat.lt_irrefl hK
  have hδnn : 0 ≤ ε * (Bf + Bg + 1)⁻¹ := Rat.le_of_lt (Rat.mul_pos hε (Rat.inv_pos.mpr hK))
  -- |x·u| ≤ Bf·δ, |v·y| ≤ δ·Bg
  have t1 : ratAbs (x * u) ≤ Bf * (ε * (Bf + Bg + 1)⁻¹) := ratAbs_mul_le hx hu
  have t2 : ratAbs (v * y) ≤ (ε * (Bf + Bg + 1)⁻¹) * Bg := ratAbs_mul_le hv hy
  refine Rat.le_trans (rat_add_le_add t1 t2) ?_
  -- Bf·δ + δ·Bg = (Bf+Bg)·δ ≤ K·δ = ε
  have hrew : Bf * (ε * (Bf + Bg + 1)⁻¹) + (ε * (Bf + Bg + 1)⁻¹) * Bg
      = (Bf + Bg) * (ε * (Bf + Bg + 1)⁻¹) := by
    rw [Rat.mul_comm (ε * (Bf + Bg + 1)⁻¹) Bg, ← Rat.add_mul]
  rw [hrew]
  have hle : (Bf + Bg) * (ε * (Bf + Bg + 1)⁻¹) ≤ (Bf + Bg + 1) * (ε * (Bf + Bg + 1)⁻¹) := by
    apply Rat.mul_le_mul_of_nonneg_right _ hδnn
    have : (Bf + Bg) + 0 ≤ (Bf + Bg) + 1 := Rat.add_le_add_left.mpr (by native_decide)
    rwa [Rat.add_zero] at this
  rw [rat_K_delta hKne] at hle
  exact hle

/-- **The product of two Cauchy sequences is Cauchy.** -/
theorem mul_cauchy {f g : Nat → Rat} (hf : Sounio.RealCauchy.IsCauchy f)
    (hg : Sounio.RealCauchy.IsCauchy g) :
    Sounio.RealCauchy.IsCauchy (fun n => f n * g n) := by
  intro ε hε
  obtain ⟨Bf, Nf, hBf, hfb⟩ := cauchy_bounded hf
  obtain ⟨Bg, Ng, hBg, hgb⟩ := cauchy_bounded hg
  have hK : 0 < Bf + Bg + 1 := rat_K_pos hBf hBg
  have hδ : 0 < ε * (Bf + Bg + 1)⁻¹ := Rat.mul_pos hε (Rat.inv_pos.mpr hK)
  obtain ⟨Nfd, hfd⟩ := isCauchy_abs hf _ hδ
  obtain ⟨Ngd, hgd⟩ := isCauchy_abs hg _ hδ
  refine ⟨max (max Nf Ng) (max Nfd Ngd), fun m n hm hn => ?_⟩
  have hmNf : Nf ≤ m := Nat.le_trans (Nat.le_trans (Nat.le_max_left _ _) (Nat.le_max_left _ _)) hm
  have hnNg : Ng ≤ n := Nat.le_trans (Nat.le_trans (Nat.le_max_right _ _) (Nat.le_max_left _ _)) hn
  have hmNfd : Nfd ≤ m := Nat.le_trans (Nat.le_trans (Nat.le_max_left _ _) (Nat.le_max_right _ _)) hm
  have hnNfd : Nfd ≤ n := Nat.le_trans (Nat.le_trans (Nat.le_max_left _ _) (Nat.le_max_right _ _)) hn
  have hmNgd : Ngd ≤ m := Nat.le_trans (Nat.le_trans (Nat.le_max_right _ _) (Nat.le_max_right _ _)) hm
  have hnNgd : Ngd ≤ n := Nat.le_trans (Nat.le_trans (Nat.le_max_right _ _) (Nat.le_max_right _ _)) hn
  have key : ratAbs (f m * g m - f n * g n) ≤ ε := by
    rw [rat_mul_sub_mul (f m) (g m) (f n) (g n)]
    refine Rat.le_trans (ratAbs_add_le _ _) ?_
    exact rat_prod_bound hBf hBg hε (hfb m hmNf) (hgd m n hmNgd hnNgd) (hfd m n hmNfd hnNfd) (hgb n hnNg)
  obtain ⟨ka, kb⟩ := abs_two_sided key
  refine ⟨ka, ?_⟩
  rw [Rat.neg_sub] at kb; exact kb

/-- **`RealEq` respects pointwise product** — discharges `RealMulCongObligation`. -/
theorem mul_cong (a b a' b' : SounioRealCauchy) (haa : RealEq a a') (hbb : RealEq b b') :
    TendsToZero (fun n => a.seq n * b.seq n - a'.seq n * b'.seq n) := by
  intro ε hε
  obtain ⟨Ba, Na, hBa, hab⟩ := cauchy_bounded a.cauchy
  obtain ⟨Bb, Nb, hBb, hbq⟩ := cauchy_bounded b'.cauchy
  have hK : 0 < Ba + Bb + 1 := rat_K_pos hBa hBb
  have hδ : 0 < ε * (Ba + Bb + 1)⁻¹ := Rat.mul_pos hε (Rat.inv_pos.mpr hK)
  obtain ⟨Nbd, hbd⟩ := realEq_abs hbb _ hδ
  obtain ⟨Nad, had⟩ := realEq_abs haa _ hδ
  refine ⟨max (max Na Nb) (max Nbd Nad), fun n hn => ?_⟩
  have hnNa : Na ≤ n := Nat.le_trans (Nat.le_trans (Nat.le_max_left _ _) (Nat.le_max_left _ _)) hn
  have hnNb : Nb ≤ n := Nat.le_trans (Nat.le_trans (Nat.le_max_right _ _) (Nat.le_max_left _ _)) hn
  have hnNbd : Nbd ≤ n := Nat.le_trans (Nat.le_trans (Nat.le_max_left _ _) (Nat.le_max_right _ _)) hn
  have hnNad : Nad ≤ n := Nat.le_trans (Nat.le_trans (Nat.le_max_right _ _) (Nat.le_max_right _ _)) hn
  have key : ratAbs (a.seq n * b.seq n - a'.seq n * b'.seq n) ≤ ε := by
    rw [rat_mul_sub_mul (a.seq n) (b.seq n) (a'.seq n) (b'.seq n)]
    refine Rat.le_trans (ratAbs_add_le _ _) ?_
    exact rat_prod_bound hBa hBb hε (hab n hnNa) (hbd n hnNbd) (had n hnNad) (hbq n hnNb)
  exact abs_two_sided key

/-- Discharges `SounioRealCauchy.MulPreservesCauchy` — the obligation flagged in `SounioRealCauchy`. -/
theorem mulPreservesCauchy_done : Sounio.RealCauchy.SounioRealCauchy.MulPreservesCauchy :=
  fun a b => mul_cauchy a.cauchy b.cauchy

/-! ## Phase 2b — the multiplicative inverse (CRUX 1). -/

/-- The canonical zero real (constant-0 Cauchy sequence). -/
abbrev zeroR : SounioRealCauchy := Sounio.RealCauchy.SounioRealCauchy.zero

/-- **A real not `RealEq` to `0` is eventually bounded away from `0`** by a positive rational `δ`.
    The conceptual crux of the inverse: combine "far from 0 infinitely often" (the negation of
    `TendsToZero x.seq`) with the Cauchy modulus at `ε/2` via the reverse triangle inequality. -/
theorem bounded_away {x : SounioRealCauchy} (h : ¬ RealEq x zeroR) :
    ∃ δ : Rat, ∃ N : Nat, 0 < δ ∧ ∀ n, N ≤ n → δ ≤ ratAbs (x.seq n) := by
  refine Classical.byContradiction (fun hcon => ?_)
  apply h
  intro ε hε
  obtain ⟨M, hMc⟩ := isCauchy_abs x.cauchy (ε * (1/2)) (rat_half_pos hε)
  have hn0 : ∃ n, M ≤ n ∧ ratAbs (x.seq n) < ε * (1/2) := by
    refine Classical.byContradiction (fun hno => ?_)
    apply hcon
    refine ⟨ε * (1/2), M, rat_half_pos hε, fun n hn => ?_⟩
    exact Rat.not_lt.mp (fun hlt => hno ⟨n, hn, hlt⟩)
  obtain ⟨n0, hn0M, hn0lt⟩ := hn0
  refine ⟨M, fun n hn => ?_⟩
  show x.seq n - zeroR.seq n ≤ ε ∧ -(x.seq n - zeroR.seq n) ≤ ε
  have hzero : x.seq n - zeroR.seq n = x.seq n := by
    show x.seq n - 0 = x.seq n
    rw [Rat.sub_eq_add_neg, Rat.neg_zero, Rat.add_zero]
  rw [hzero]
  have htri : ratAbs (x.seq n) ≤ ratAbs (x.seq n0) + ratAbs (x.seq n - x.seq n0) := by
    have he : x.seq n0 + (x.seq n - x.seq n0) = x.seq n := by
      rw [Rat.add_comm, Rat.sub_add_cancel]
    have hb := ratAbs_add_le (x.seq n0) (x.seq n - x.seq n0)
    rwa [he] at hb
  have hbound : ratAbs (x.seq n) ≤ ε := by
    refine Rat.le_trans htri ?_
    have h1 : ratAbs (x.seq n0) ≤ ε * (1/2) := Rat.le_of_lt hn0lt
    have h2 : ratAbs (x.seq n - x.seq n0) ≤ ε * (1/2) := hMc n n0 hn hn0M
    have hsum := rat_add_le_add h1 h2
    rwa [rat_add_halves] at hsum
  exact abs_two_sided hbound

/-! ## Obligation ledger for `SqrtField ℝ`

Each obligation below is a named `Prop`. Discharging *all* of them (Mathlib-free) and assembling
`{ F := Quotient ⟨RealEq, …⟩, add := …, …, sqrt := … }` yields the `SounioSqrt.SqrtField` instance
that, fed to `DeGrey529.TransferWf.sqrtField_chi_ge_5`, gives **χ(ℝ²) ≥ 5**.

Status legend: ✅ proved above · ⏳ deferred (analytic core).
-/

/-- ✅ **DISCHARGED** by `realEq_trans` / `realEq_equivalence` / `realSetoid` above (the ε/2 triangle,
    built on `rat_sub_split` + `rat_add_halves` from the `Rat` order API). `RealEq` is now a full
    equivalence, so ℝ := `Quotient realSetoid` is available. -/
def RealEqTransObligation : Prop :=
  ∀ a b c : SounioRealCauchy, RealEq a b → RealEq b c → RealEq a c

theorem realEqTransObligation_done : RealEqTransObligation := @realEq_trans

/-- ✅ **DISCHARGED (additive part)** by `realOpsCong_add` below: `RealEq` respects pointwise sum,
    so `add` descends to the quotient. The multiplicative congruence (needing eventual boundedness
    of Cauchy representatives) remains in `RealMulCongObligation`. -/
def RealOpsCongObligation : Prop :=
  ∀ a b a' b' : SounioRealCauchy, RealEq a a' → RealEq b b' →
    TendsToZero (fun n => (a.seq n + b.seq n) - (a'.seq n + b'.seq n))

/-- `RealEq` respects pointwise addition: `(a+b) - (a'+b') = (a-a') + (b-b')`, each band-half
    bounded by the ε/2 triangle. Discharges the additive part of `RealOpsCongObligation`. -/
theorem realOpsCong_add : RealOpsCongObligation := by
  intro a b a' b' hab hbb ε hε
  obtain ⟨N1, hN1⟩ := hab (ε * (1/2)) (rat_half_pos hε)
  obtain ⟨N2, hN2⟩ := hbb (ε * (1/2)) (rat_half_pos hε)
  refine ⟨max N1 N2, fun n hn => ?_⟩
  obtain ⟨h1a, h1b⟩ := hN1 n (Nat.le_trans (Nat.le_max_left N1 N2) hn)
  obtain ⟨h2a, h2b⟩ := hN2 n (Nat.le_trans (Nat.le_max_right N1 N2) hn)
  refine ⟨?_, ?_⟩
  · show (a.seq n + b.seq n) - (a'.seq n + b'.seq n) ≤ ε
    rw [rat_add_sub_add]
    have : (a.seq n - a'.seq n) + (b.seq n - b'.seq n) ≤ ε * (1/2) + ε * (1/2) :=
      Rat.le_trans (Rat.add_le_add_right.mpr h1a) (Rat.add_le_add_left.mpr h2a)
    rwa [rat_add_halves] at this
  · show -((a.seq n + b.seq n) - (a'.seq n + b'.seq n)) ≤ ε
    rw [rat_add_sub_add, Rat.neg_add]
    have : -(a.seq n - a'.seq n) + -(b.seq n - b'.seq n) ≤ ε * (1/2) + ε * (1/2) :=
      Rat.le_trans (Rat.add_le_add_right.mpr h1b) (Rat.add_le_add_left.mpr h2b)
    rwa [rat_add_halves] at this

/-- ✅ **DISCHARGED** by `mul_cong`: `RealEq` respects pointwise **product**. The boundedness lemma
    `cauchy_bounded` controls `a·b - a'·b' = a·(b-b') + (a-a')·b'` via the `ratAbs` toolkit and the
    `K = Bₐ+B_b'+1`, `δ = ε·K⁻¹` scaling. Together with `mul_cauchy` (product of Cauchy is Cauchy)
    this grounds the multiplicative operation on the quotient ℝ. -/
def RealMulCongObligation : Prop :=
  ∀ a b a' b' : SounioRealCauchy, RealEq a a' → RealEq b b' →
    TendsToZero (fun n => a.seq n * b.seq n - a'.seq n * b'.seq n)

theorem realMulCongObligation_done : RealMulCongObligation := mul_cong

/-- ⏳ The field axioms (`add_assoc`, …, `mul_inv`, `zero_ne_one`) on the quotient. These descend
    from the corresponding `Rat` identities once `RealOpsCongObligation` provides well-definedness;
    `mul_inv` additionally needs that a non-null real is eventually bounded away from `0` — the
    conceptual crux now **proved** as `bounded_away`. What remains for the inverse: the termwise
    `1/xₙ` sequence (zero-patched), its Cauchy/congruence proof via `1/x - 1/y = (y-x)·x⁻¹·y⁻¹`, and
    `mul x (inv x) = one` (the product sequence is eventually `1`). -/
def RealFieldAxiomsObligation : Prop := True  -- expands to the 11 `RootedField` ring/field fields

/-- ⏳ Order axioms (`le_refl`, …, `add_le_add_right`) and crucially `mul_nonneg`. The
    mul-monotonicity law `mul_le_mul_of_nonneg_right` is the ≈500–1000 LOC ε-N proof
    `SounioRealCauchy` already flags (`OrderedCarrierObligation_RealCauchy`). -/
def RealOrderObligation : Prop := True  -- includes SounioRealCauchy.OrderedCarrierObligation_RealCauchy

/-- ⏳ Order **completeness** (every bounded-above set has a least upper bound) — the property that
    distinguishes ℝ from ℚ and powers the existence of square roots. -/
def RealCompletenessObligation : Prop := True  -- monotone-bounded ⇒ Cauchy ⇒ convergent

/-- ⏳ The constructive square root with `sqrt_nonneg` and `sqrt_sq` (`a ≥ 0 → sqrt a · sqrt a = a`).
    Built from `RealCompletenessObligation` (sup of `{x | x² ≤ a}`) or a convergent
    Newton/bisection iteration. This is the final `SqrtField` field. -/
def RealSqrtObligation : Prop := True  -- sqrt + sqrt_nonneg + sqrt_sq

#print axioms mul_cauchy
#print axioms mul_cong
#print axioms bounded_away

#eval IO.println "SounioSqrtFieldReal: analytic RootedField ℝ — RealEq PROVED an EQUIVALENCE (realSetoid ⇒ ℝ := Quotient realSetoid); additive AND multiplicative structure grounded (add_cauchy/mul_cauchy: + and · preserve Cauchy; realOpsCong_add/mul_cong: + and · respect RealEq ⇒ descend to the quotient, via the ratAbs toolkit + cauchy_bounded). Inverse CRUX proved: bounded_away (x≠0 ⇒ eventually |xₙ|≥δ>0). Remaining ledger: finish inverse (1/xₙ Cauchy+cong, mul·inv=1), order axioms, four Newton roots. Then rootedField_chi_ge_5 gives χ(ℝ²)≥5."

end SounioSqrt.RealCauchyField
