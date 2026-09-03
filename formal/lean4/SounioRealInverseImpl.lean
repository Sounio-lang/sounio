import SounioSqrtFieldReal

set_option maxHeartbeats 0

/-!
# Sounio — sequence-level lemmas for the multiplicative inverse of Cauchy sequences (Mathlib-free)

This file supplies the termwise reciprocal `invSeq` (zero-patched) for a Cauchy sequence of
rationals, together with the four downstream facts the quotient-ℝ field-inverse needs:

* `inv_cauchy`     — the reciprocal of a bounded-away-from-`0` Cauchy sequence is Cauchy;
* `mul_inv_tendsto`— `fₙ · invSeq f n − 1 → 0` (so `x · x⁻¹ = 1` on the quotient);
* `inv_cong`       — the reciprocal respects `RealEq` (well-definedness on the quotient).

All proofs are core-Lean only: no Mathlib, no `sorry`, no `axiom`, no `ring`/`linarith`/`omega`.
The analytic engine is the reciprocal-difference identity `a⁻¹ − b⁻¹ = (b − a)·(a⁻¹·b⁻¹)` plus the
`ratAbs` toolkit and the eventual lower bound `δ ≤ |fₙ|` from `bounded_away`.
-/

namespace SounioSqrt.RealCauchyField

open Sounio.RealCauchy (SounioRealCauchy)

/-! ## Scalar `Rat` reciprocal algebra (sign + identities) -/

/-- Multiplicative inverse is unique: `a·b = 1` with `a ≠ 0` forces `b = a⁻¹`. -/
private theorem inv_unique (a b : Rat) (ha : a ≠ 0) (h : a * b = 1) : b = a⁻¹ := by
  have hinv : a⁻¹ * a = 1 := by rw [Rat.mul_comm]; exact Rat.mul_inv_cancel a ha
  calc b = 1 * b := by rw [Rat.one_mul]
    _ = (a⁻¹ * a) * b := by rw [hinv]
    _ = a⁻¹ * (a * b) := by rw [Rat.mul_assoc]
    _ = a⁻¹ * 1 := by rw [h]
    _ = a⁻¹ := by rw [Rat.mul_one]

/-- Negation preserves non-zeroness. -/
private theorem neg_ne_zero_of_ne {x : Rat} (hx : x ≠ 0) : -x ≠ 0 := by
  intro h
  apply hx
  have hxx : -(-x) = -0 := by rw [h]
  rwa [Rat.neg_neg, Rat.neg_zero] at hxx

/-- `(-x)⁻¹ = -(x⁻¹)` (Mathlib-free; `Rat.inv_neg` does not exist here). -/
private theorem neg_inv (x : Rat) : (-x)⁻¹ = -(x⁻¹) := by
  by_cases hx : x = 0
  · subst hx; rw [Rat.neg_zero, Rat.inv_zero, Rat.neg_zero]
  · have hxne : -x ≠ 0 := neg_ne_zero_of_ne hx
    have hkey : (-x) * (-(x⁻¹)) = 1 := by
      rw [Rat.neg_mul, Rat.mul_neg, Rat.neg_neg, Rat.mul_inv_cancel x hx]
    exact (inv_unique (-x) (-(x⁻¹)) hxne hkey).symm

/-- `0 ≤ x` keeps the reciprocal nonnegative. -/
private theorem inv_nonneg {x : Rat} (h : 0 ≤ x) : 0 ≤ x⁻¹ := by
  by_cases hx : 0 < x
  · exact Rat.le_of_lt (Rat.inv_pos.mpr hx)
  · have hxle : x ≤ 0 := Rat.not_lt.mp hx
    have hx0 : x = 0 := Rat.le_antisymm hxle h
    rw [hx0, Rat.inv_zero]; exact Rat.le_refl

/-- A negative rational has a negative reciprocal. -/
private theorem inv_neg_of_neg {x : Rat} (h : x < 0) : x⁻¹ < 0 := by
  have hpos : 0 < -x := Rat.lt_neg_iff.mp h
  have hinvpos : 0 < (-x)⁻¹ := Rat.inv_pos.mpr hpos
  rw [neg_inv] at hinvpos
  have hlt := Rat.lt_neg_iff.mp hinvpos
  rwa [Rat.neg_zero] at hlt

/-- `|x⁻¹| = |x|⁻¹`. -/
theorem ratAbs_inv (x : Rat) : ratAbs (x⁻¹) = (ratAbs x)⁻¹ := by
  by_cases h : 0 ≤ x
  · have hnn : 0 ≤ x⁻¹ := inv_nonneg h
    unfold ratAbs
    rw [if_pos h, if_pos hnn]
  · have hlt : x < 0 := Rat.not_le.mp h
    have hinvlt : x⁻¹ < 0 := inv_neg_of_neg hlt
    have hninv : ¬ 0 ≤ x⁻¹ := fun hle => Rat.lt_irrefl (Std.lt_of_lt_of_le hinvlt hle)
    unfold ratAbs
    rw [if_neg h, if_neg hninv, neg_inv]

/-- The reciprocal-difference identity: `a⁻¹ − b⁻¹ = (b − a)·(a⁻¹·b⁻¹)`. -/
private theorem inv_sub_inv (a b : Rat) (ha : a ≠ 0) (hb : b ≠ 0) :
    a⁻¹ - b⁻¹ = (b - a) * (a⁻¹ * b⁻¹) := by
  have sub_mul : ∀ x y z : Rat, (x - y) * z = x * z - y * z := by
    intro x y z
    rw [Rat.sub_eq_add_neg x y, Rat.add_mul, Rat.neg_mul, ← Rat.sub_eq_add_neg]
  have hb1 : b * (a⁻¹ * b⁻¹) = a⁻¹ := by
    rw [Rat.mul_comm a⁻¹ b⁻¹, ← Rat.mul_assoc, Rat.mul_inv_cancel b hb, Rat.one_mul]
  have ha1 : a * (a⁻¹ * b⁻¹) = b⁻¹ := by
    rw [← Rat.mul_assoc, Rat.mul_inv_cancel a ha, Rat.one_mul]
  rw [sub_mul b a (a⁻¹ * b⁻¹), hb1, ha1]

/-- Reciprocal is antitone on the positives: `0 < a ≤ b ⇒ b⁻¹ ≤ a⁻¹`. -/
private theorem inv_antitone {a b : Rat} (ha : 0 < a) (hab : a ≤ b) : b⁻¹ ≤ a⁻¹ := by
  have hb : 0 < b := Std.lt_of_lt_of_le ha hab
  have hane : a ≠ 0 := fun he => Rat.lt_irrefl (he ▸ ha)
  have hbne : b ≠ 0 := fun he => Rat.lt_irrefl (he ▸ hb)
  have hbainn : (0 : Rat) ≤ b - a := by
    have hstep : a + (-a) ≤ b + (-a) := Rat.add_le_add_right.mpr hab
    rwa [Rat.add_neg_cancel, ← Rat.sub_eq_add_neg] at hstep
  have hnn : (0 : Rat) ≤ a⁻¹ - b⁻¹ := by
    rw [inv_sub_inv a b hane hbne]
    exact Rat.mul_nonneg hbainn
      (Rat.le_of_lt (Rat.mul_pos (Rat.inv_pos.mpr ha) (Rat.inv_pos.mpr hb)))
  have h2 : (0 : Rat) + b⁻¹ ≤ (a⁻¹ - b⁻¹) + b⁻¹ := Rat.add_le_add_right.mpr hnn
  rwa [Rat.zero_add, Rat.sub_add_cancel] at h2

/-! ## The zero-patched reciprocal sequence -/

/-- Termwise reciprocal, patched to `1` where the sequence vanishes. -/
def invSeq (f : Nat → Rat) : Nat → Rat := fun n => if f n = 0 then 1 else (f n)⁻¹

/-- A term bounded below in absolute value by a positive `δ` is non-zero. -/
private theorem ne_zero_of_bound {q δ : Rat} (hδ : 0 < δ) (hb : δ ≤ ratAbs q) : q ≠ 0 := by
  intro h
  have hr0 : ratAbs q = 0 := by
    rw [h]
    show (if (0:Rat) ≤ 0 then (0:Rat) else -0) = 0
    rw [if_pos Rat.le_refl]
  rw [hr0] at hb
  exact Rat.lt_irrefl (Std.lt_of_lt_of_le hδ hb)

/-! ## The three downstream facts -/

/-- **The reciprocal of a Cauchy sequence bounded away from `0` is Cauchy.** -/
theorem inv_cauchy {f : Nat → Rat} (hf : Sounio.RealCauchy.IsCauchy f)
    (δ : Rat) (Nba : Nat) (hδ : 0 < δ)
    (hba : ∀ n, Nba ≤ n → δ ≤ ratAbs (f n)) :
    Sounio.RealCauchy.IsCauchy (invSeq f) := by
  intro ε hε
  have hδne : δ ≠ 0 := fun he => Rat.lt_irrefl (he ▸ hδ)
  have hδδ : 0 < δ * δ := Rat.mul_pos hδ hδ
  have hεδδ : 0 < ε * (δ * δ) := Rat.mul_pos hε hδδ
  obtain ⟨M, hM⟩ := isCauchy_abs hf (ε * (δ * δ)) hεδδ
  refine ⟨max M Nba, fun m n hm hn => ?_⟩
  have hmM : M ≤ m := Nat.le_trans (Nat.le_max_left M Nba) hm
  have hnM : M ≤ n := Nat.le_trans (Nat.le_max_left M Nba) hn
  have hmNba : Nba ≤ m := Nat.le_trans (Nat.le_max_right M Nba) hm
  have hnNba : Nba ≤ n := Nat.le_trans (Nat.le_max_right M Nba) hn
  have hfm : f m ≠ 0 := ne_zero_of_bound hδ (hba m hmNba)
  have hfn : f n ≠ 0 := ne_zero_of_bound hδ (hba n hnNba)
  have him : invSeq f m = (f m)⁻¹ := by
    show (if f m = 0 then (1:Rat) else (f m)⁻¹) = (f m)⁻¹
    rw [if_neg hfm]
  have hin : invSeq f n = (f n)⁻¹ := by
    show (if f n = 0 then (1:Rat) else (f n)⁻¹) = (f n)⁻¹
    rw [if_neg hfn]
  have hbm : ratAbs ((f m)⁻¹) ≤ δ⁻¹ := by
    rw [ratAbs_inv]; exact inv_antitone hδ (hba m hmNba)
  have hbn : ratAbs ((f n)⁻¹) ≤ δ⁻¹ := by
    rw [ratAbs_inv]; exact inv_antitone hδ (hba n hnNba)
  have hscale : (ε * (δ * δ)) * (δ⁻¹ * δ⁻¹) = ε := by
    have hc : δ * δ⁻¹ = 1 := Rat.mul_inv_cancel δ hδne
    have hdd : (δ * δ) * (δ⁻¹ * δ⁻¹) = 1 := by
      rw [Rat.mul_assoc, ← Rat.mul_assoc δ δ⁻¹ δ⁻¹, hc, Rat.one_mul, hc]
    rw [Rat.mul_assoc, hdd, Rat.mul_one]
  have key : ratAbs (invSeq f m - invSeq f n) ≤ ε := by
    rw [him, hin, inv_sub_inv (f m) (f n) hfm hfn]
    have hx : ratAbs (f n - f m) ≤ ε * (δ * δ) := by
      rw [show f n - f m = -(f m - f n) from by rw [Rat.neg_sub], ratAbs_neg]
      exact hM m n hmM hnM
    have hy : ratAbs ((f m)⁻¹ * (f n)⁻¹) ≤ δ⁻¹ * δ⁻¹ := ratAbs_mul_le hbm hbn
    have hbound := ratAbs_mul_le hx hy
    rw [hscale] at hbound
    exact hbound
  obtain ⟨ka, kb⟩ := abs_two_sided key
  refine ⟨ka, ?_⟩
  rw [Rat.neg_sub] at kb
  exact kb

/-- **`fₙ · invSeq f n − 1 → 0`** — the product sequence is eventually exactly `1`. -/
theorem mul_inv_tendsto {f : Nat → Rat} (hf : Sounio.RealCauchy.IsCauchy f)
    (δ : Rat) (Nba : Nat) (hδ : 0 < δ)
    (hba : ∀ n, Nba ≤ n → δ ≤ ratAbs (f n)) :
    TendsToZero (fun n => f n * invSeq f n - 1) := by
  intro ε hε
  refine ⟨Nba, fun n hn => ?_⟩
  have hfn : f n ≠ 0 := ne_zero_of_bound hδ (hba n hn)
  have hi : invSeq f n = (f n)⁻¹ := by
    show (if f n = 0 then (1:Rat) else (f n)⁻¹) = (f n)⁻¹
    rw [if_neg hfn]
  have hval : f n * invSeq f n - 1 = 0 := by
    rw [hi, Rat.mul_inv_cancel (f n) hfn, Rat.sub_self]
  show f n * invSeq f n - 1 ≤ ε ∧ -(f n * invSeq f n - 1) ≤ ε
  rw [hval, Rat.neg_zero]
  exact ⟨Rat.le_of_lt hε, Rat.le_of_lt hε⟩

/-- **The reciprocal respects `RealEq`** — well-definedness of inverse on the quotient ℝ. -/
theorem inv_cong {f g : Nat → Rat} (hf : Sounio.RealCauchy.IsCauchy f)
    (hg : Sounio.RealCauchy.IsCauchy g) (δ : Rat) (Nf Ng : Nat) (hδ : 0 < δ)
    (hbaf : ∀ n, Nf ≤ n → δ ≤ ratAbs (f n))
    (hbag : ∀ n, Ng ≤ n → δ ≤ ratAbs (g n))
    (hfg : TendsToZero (fun n => f n - g n)) :
    TendsToZero (fun n => invSeq f n - invSeq g n) := by
  intro ε hε
  have hδne : δ ≠ 0 := fun he => Rat.lt_irrefl (he ▸ hδ)
  have hδδ : 0 < δ * δ := Rat.mul_pos hδ hδ
  have hεδδ : 0 < ε * (δ * δ) := Rat.mul_pos hε hδδ
  obtain ⟨Nd, hNd⟩ := hfg (ε * (δ * δ)) hεδδ
  refine ⟨max (max Nf Ng) Nd, fun n hn => ?_⟩
  have hnNf : Nf ≤ n :=
    Nat.le_trans (Nat.le_trans (Nat.le_max_left Nf Ng) (Nat.le_max_left _ _)) hn
  have hnNg : Ng ≤ n :=
    Nat.le_trans (Nat.le_trans (Nat.le_max_right Nf Ng) (Nat.le_max_left _ _)) hn
  have hnNd : Nd ≤ n := Nat.le_trans (Nat.le_max_right _ _) hn
  have hfn : f n ≠ 0 := ne_zero_of_bound hδ (hbaf n hnNf)
  have hgn : g n ≠ 0 := ne_zero_of_bound hδ (hbag n hnNg)
  have hif : invSeq f n = (f n)⁻¹ := by
    show (if f n = 0 then (1:Rat) else (f n)⁻¹) = (f n)⁻¹
    rw [if_neg hfn]
  have hig : invSeq g n = (g n)⁻¹ := by
    show (if g n = 0 then (1:Rat) else (g n)⁻¹) = (g n)⁻¹
    rw [if_neg hgn]
  have hbf : ratAbs ((f n)⁻¹) ≤ δ⁻¹ := by
    rw [ratAbs_inv]; exact inv_antitone hδ (hbaf n hnNf)
  have hbg : ratAbs ((g n)⁻¹) ≤ δ⁻¹ := by
    rw [ratAbs_inv]; exact inv_antitone hδ (hbag n hnNg)
  have habsfg : ratAbs (f n - g n) ≤ ε * (δ * δ) := by
    obtain ⟨h1, h2⟩ := hNd n hnNd
    exact ratAbs_le h1 h2
  have hscale : (ε * (δ * δ)) * (δ⁻¹ * δ⁻¹) = ε := by
    have hc : δ * δ⁻¹ = 1 := Rat.mul_inv_cancel δ hδne
    have hdd : (δ * δ) * (δ⁻¹ * δ⁻¹) = 1 := by
      rw [Rat.mul_assoc, ← Rat.mul_assoc δ δ⁻¹ δ⁻¹, hc, Rat.one_mul, hc]
    rw [Rat.mul_assoc, hdd, Rat.mul_one]
  have key : ratAbs (invSeq f n - invSeq g n) ≤ ε := by
    rw [hif, hig, inv_sub_inv (f n) (g n) hfn hgn]
    have hx : ratAbs (g n - f n) ≤ ε * (δ * δ) := by
      rw [show g n - f n = -(f n - g n) from by rw [Rat.neg_sub], ratAbs_neg]
      exact habsfg
    have hy : ratAbs ((f n)⁻¹ * (g n)⁻¹) ≤ δ⁻¹ * δ⁻¹ := ratAbs_mul_le hbf hbg
    have hbound := ratAbs_mul_le hx hy
    rw [hscale] at hbound
    exact hbound
  show invSeq f n - invSeq g n ≤ ε ∧ -(invSeq f n - invSeq g n) ≤ ε
  exact abs_two_sided key

#print axioms inv_cauchy
#print axioms mul_inv_tendsto
#print axioms inv_cong
#print axioms ratAbs_inv

end SounioSqrt.RealCauchyField
