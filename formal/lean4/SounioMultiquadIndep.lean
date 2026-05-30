import SounioSqrtFieldReal
import SounioNewtonSqrtImpl

set_option maxHeartbeats 400000

/-!
# Sounio — irrationality core for the multiquadratic linear-independence programme

Mathlib-free (core Lean 4 only). This file proves the arithmetic heart needed to show that the
square roots of the seven squarefree radicands `{3, 5, 11, 15, 33, 55, 165}` are irrational, and
bridges that fact to the Cauchy-real model (`SounioRealCauchy` / `RealEq`) built in
`SounioSqrtFieldReal` and the Newton square-root iteration of `SounioNewtonSqrtImpl`.

Main results (all `sorry`/`axiom`/Mathlib-free):

* `not_sq_radicand` — none of the seven radicands is a perfect square.
* `no_rat_sqrt` — **the irrationality core**: if `m` is a non-square natural, no rational squares
  to `m`. Proof is the coprimality route (no `p`-adic valuation, no unique factorisation): from
  `q*q = m` derive the integer identity `q.num² = m·q.den²`, take `natAbs`, and use that a divisor
  coprime to the dividend is `1`.
* `ofRat_inj` — the rational embedding into the Cauchy reals is injective modulo `RealEq`.
* `sqrt_radicand_irrational` — the Newton class of `√m` is not `RealEq` to any rational class.
-/

namespace SounioSqrt.RealCauchyField

open Sounio.RealCauchy (SounioRealCauchy)

/-! ## 1. The seven radicands are not perfect squares. -/

/-- None of the seven squarefree radicands `{3,5,11,15,33,55,165}` is a perfect square.

    For a fixed `m < 169`, `k*k = m` forces `k < 13` (else `k*k ≥ 13*13 = 169 > m`), and the
    finite check `∀ k < 13, k*k ≠ m` is decidable. -/
theorem not_sq_radicand (m : Nat)
    (hm : m = 3 ∨ m = 5 ∨ m = 11 ∨ m = 15 ∨ m = 33 ∨ m = 55 ∨ m = 165) :
    ∀ k : Nat, k * k ≠ m := by
  have hbound : m < 169 := by rcases hm with h|h|h|h|h|h|h <;> subst h <;> decide
  have hcheck : ∀ k : Nat, k < 13 → k * k ≠ m := by
    rcases hm with h|h|h|h|h|h|h <;> subst h <;> decide
  intro k hk
  rcases Nat.lt_or_ge k 13 with hk13 | hk13
  · exact hcheck k hk13 hk
  · have hmono : 13 * 13 ≤ k * k := Nat.mul_le_mul hk13 hk13
    rw [hk] at hmono
    omega

/-! ## 2. The irrationality core. -/

/-- **The irrationality core.** If `m : Nat` is not a perfect square, then no rational squares to
    `m`.

    Coprimality route: write `q = q.num /. q.den`. From `q*q = m` and
    `divInt n₁ d₁ * divInt n₂ d₂ = divInt (n₁n₂)(d₁d₂)` plus `divInt_eq_divInt_iff` (cross
    multiplication, denominators `≠ 0`) we get the **integer identity**
    `q.num · q.num = m · (q.den · q.den)`. Taking `Int.natAbs` (via `Int.natAbs_mul` and
    `Int.natAbs_natCast`) gives the **Nat identity** `A·A = m·(b·b)` with `A := q.num.natAbs`,
    `b := q.den`. Then `b·b ∣ A·A`, while `q.reduced` (`A` coprime `b`) lifts by
    `Nat.Coprime.pow` to `gcd (A·A)(b·b) = 1`; `Nat.dvd_gcd` forces `b·b ∣ 1`, i.e. `b·b = 1`.
    Hence `A·A = m`, contradicting non-squareness. -/
theorem no_rat_sqrt (m : Nat) (hns : ∀ k : Nat, k * k ≠ m) :
    ¬ ∃ q : Rat, q * q = (m : Rat) := by
  intro hex
  obtain ⟨q, hq⟩ := hex
  -- denominators are nonzero over ℤ
  have hden : (↑q.den : Int) ≠ 0 := Int.natCast_ne_zero.mpr q.den_nz
  have hbb : (↑q.den : Int) * (↑q.den : Int) ≠ 0 := Int.mul_ne_zero hden hden
  -- q = q.num /. q.den
  have hqd : Rat.divInt q.num (↑q.den) = q := Rat.num_divInt_den q
  -- (m : ℚ) = m /. 1
  have hmd : Rat.divInt (↑m : Int) (↑(1 : Nat)) = (m : Rat) := by
    have h := Rat.num_divInt_den (m : Rat)
    rw [Rat.num_natCast, Rat.den_natCast] at h
    exact h
  -- assemble the rational identity (a·a)/(b·b) = m/1
  have hq2 : Rat.divInt (q.num * q.num) ((↑q.den : Int) * (↑q.den : Int))
      = Rat.divInt (↑m : Int) (↑(1 : Nat)) := by
    rw [← Rat.divInt_mul_divInt, hqd, hq, hmd]
  -- cross-multiply: this is the INTEGER identity
  have h1ne : (↑(1 : Nat) : Int) ≠ 0 := by native_decide
  have hkey0 := (Rat.divInt_eq_divInt_iff hbb h1ne).mp hq2
  have hone : (↑(1 : Nat) : Int) = 1 := by native_decide
  rw [hone, Int.mul_one] at hkey0
  -- hkey0 : q.num * q.num = ↑m * (↑q.den * ↑q.den)
  -- take natAbs to land in ℕ
  have hAA : q.num.natAbs * q.num.natAbs = m * (q.den * q.den) := by
    have h := congrArg Int.natAbs hkey0
    simp only [Int.natAbs_mul, Int.natAbs_natCast] at h
    exact h
  -- coprimality: gcd (A·A) (b·b) = 1
  have hcop : Nat.gcd (q.num.natAbs * q.num.natAbs) (q.den * q.den) = 1 := by
    have h := q.reduced.pow 2 2
    rw [Nat.pow_two, Nat.pow_two] at h
    exact h
  -- b·b divides A·A
  have hdvd : (q.den * q.den) ∣ (q.num.natAbs * q.num.natAbs) :=
    ⟨m, by rw [hAA, Nat.mul_comm m (q.den * q.den)]⟩
  -- divisor coprime to dividend is 1
  have hg : (q.den * q.den) ∣ Nat.gcd (q.num.natAbs * q.num.natAbs) (q.den * q.den) :=
    Nat.dvd_gcd hdvd (Nat.dvd_refl _)
  rw [hcop] at hg
  have hb1 : q.den * q.den = 1 := Nat.eq_one_of_dvd_one hg
  -- so A·A = m, contradiction
  rw [hb1, Nat.mul_one] at hAA
  exact hns q.num.natAbs hAA

/-! ## 3. Rational embedding is injective modulo `RealEq`. -/

/-- If `c ≤ ε` for every positive `ε`, then `c ≤ 0`. (The "no positive infinitesimal" fact.)

    By contradiction: if `0 < c`, instantiate at `ε = c/2`. Then `c ≤ c/2`, and multiplying by `2`
    gives `c + c ≤ c`, i.e. `c ≤ 0`, contradicting `0 < c`. -/
theorem nonpos_of_forall_pos_le {c : Rat} (h : ∀ ε : Rat, 0 < ε → c ≤ ε) : c ≤ 0 := by
  refine Classical.byContradiction (fun hcon => ?_)
  have hc : 0 < c := Rat.not_le.mp hcon
  have hhalf : 0 < c * (1 / 2) := rat_half_pos hc
  have hle := h (c * (1 / 2)) hhalf
  have h2 : c * 2 ≤ c := by
    have hm := Rat.mul_le_mul_of_nonneg_right hle (by native_decide : (0 : Rat) ≤ 2)
    rwa [Rat.mul_assoc, show ((1 / 2 : Rat) * 2) = 1 from by native_decide, Rat.mul_one] at hm
  have h3 : c + c ≤ c := by
    rwa [show c * 2 = c + c from by
      rw [show (2 : Rat) = 1 + 1 from by native_decide, Rat.mul_add, Rat.mul_one]] at h2
  have h4 := (Rat.add_le_add_right (a := c + c) (b := c) (c := -c)).mpr h3
  rw [Rat.add_assoc, Rat.add_neg_cancel, Rat.add_zero] at h4
  exact hcon h4

/-- A constant sequence tends to `0` iff the constant is `0`: the converse direction. -/
theorem eq_zero_of_tendsToZero_const {c : Rat} (h : TendsToZero (fun _ => c)) : c = 0 := by
  have hc : ∀ ε : Rat, 0 < ε → c ≤ ε := by
    intro ε hε; obtain ⟨N, hN⟩ := h ε hε; exact (hN N (Nat.le_refl N)).1
  have hnc : ∀ ε : Rat, 0 < ε → -c ≤ ε := by
    intro ε hε; obtain ⟨N, hN⟩ := h ε hε; exact (hN N (Nat.le_refl N)).2
  have h1 : c ≤ 0 := nonpos_of_forall_pos_le hc
  have h2 : -c ≤ 0 := nonpos_of_forall_pos_le hnc
  have h3 : 0 ≤ c := by
    have hh := Rat.neg_le_neg h2
    rwa [Rat.neg_zero, Rat.neg_neg] at hh
  exact Rat.le_antisymm h1 h3

/-- **The rational embedding is injective modulo `RealEq`.** If the constant reals `ofRat x` and
    `ofRat y` denote the same real, then `x = y`. (A nonzero constant difference does not tend to
    `0`.) -/
theorem ofRat_inj {x y : Rat}
    (h : RealEq (SounioRealCauchy.ofRat x) (SounioRealCauchy.ofRat y)) : x = y := by
  have h' : TendsToZero (fun _ => x - y) := h
  have hxy : x - y = 0 := eq_zero_of_tendsToZero_const h'
  have hh := eq_add_of_sub_eq hxy
  rwa [Rat.add_zero] at hh

/-! ## 4. Bridge: the Newton class of `√m` is irrational. -/

/-- Each radicand satisfies `1 ≤ m`, the hypothesis of the Newton convergence lemmas. -/
theorem one_le_radicand (m : Nat)
    (hm : m = 3 ∨ m = 5 ∨ m = 11 ∨ m = 15 ∨ m = 33 ∨ m = 55 ∨ m = 165) :
    (1 : Rat) ≤ (m : Rat) := by
  rcases hm with h|h|h|h|h|h|h <;> subst h <;> native_decide

/-- The sum of two null sequences is null (the ε/2 triangle). -/
theorem tendsToZero_add {f g : Nat → Rat} (hf : TendsToZero f) (hg : TendsToZero g) :
    TendsToZero (fun n => f n + g n) := by
  intro ε hε
  obtain ⟨N1, hN1⟩ := hf (ε * (1 / 2)) (rat_half_pos hε)
  obtain ⟨N2, hN2⟩ := hg (ε * (1 / 2)) (rat_half_pos hε)
  refine ⟨max N1 N2, fun n hn => ?_⟩
  obtain ⟨h1a, h1b⟩ := hN1 n (Nat.le_trans (Nat.le_max_left N1 N2) hn)
  obtain ⟨h2a, h2b⟩ := hN2 n (Nat.le_trans (Nat.le_max_right N1 N2) hn)
  refine ⟨?_, ?_⟩
  · show f n + g n ≤ ε
    have hsum := rat_add_le_add h1a h2a
    rwa [rat_add_halves] at hsum
  · show -(f n + g n) ≤ ε
    rw [Rat.neg_add]
    have hsum := rat_add_le_add h1b h2b
    rwa [rat_add_halves] at hsum

/-- **The square root of a radicand is irrational.** For `m ∈ {3,5,11,15,33,55,165}`, the Newton
    class `√m` is not `RealEq` to any rational constant class.

    Proof: if `RealEq (ofRat q) √m`, then by `mul_cong` the squares are `RealEq`, i.e.
    `q² - (√m)ₙ² → 0`; `newton_sq_tendsto` gives `(√m)ₙ² - m → 0`; summing (`tendsToZero_add`) and
    telescoping (`rat_sub_split`) yields the constant `q² - m → 0`, so `q² = m`
    (`eq_zero_of_tendsToZero_const`), contradicting `no_rat_sqrt`. -/
theorem sqrt_radicand_irrational (m : Nat)
    (hm : m = 3 ∨ m = 5 ∨ m = 11 ∨ m = 15 ∨ m = 33 ∨ m = 55 ∨ m = 165) :
    ¬ ∃ q : Rat, RealEq (SounioRealCauchy.ofRat q)
        ⟨newton (m : Rat), newton_cauchy (m : Rat) (one_le_radicand m hm)⟩ := by
  intro hex
  obtain ⟨q, hqeq⟩ := hex
  -- square the congruence: q² - (√m)ₙ² → 0
  have hA : TendsToZero (fun n => q * q - newton (m : Rat) n * newton (m : Rat) n) :=
    mul_cong (SounioRealCauchy.ofRat q) (SounioRealCauchy.ofRat q)
      ⟨newton (m : Rat), newton_cauchy (m : Rat) (one_le_radicand m hm)⟩
      ⟨newton (m : Rat), newton_cauchy (m : Rat) (one_le_radicand m hm)⟩
      hqeq hqeq
  -- (√m)ₙ² - m → 0
  have hB : TendsToZero (fun n => newton (m : Rat) n * newton (m : Rat) n - (m : Rat)) :=
    newton_sq_tendsto (m : Rat) (one_le_radicand m hm)
  -- sum telescopes to the constant q² - m
  have hAB := tendsToZero_add hA hB
  have hfun : (fun n => (q * q - newton (m : Rat) n * newton (m : Rat) n)
      + (newton (m : Rat) n * newton (m : Rat) n - (m : Rat)))
      = (fun _ : Nat => q * q - (m : Rat)) := by
    funext n
    exact (rat_sub_split (q * q) (newton (m : Rat) n * newton (m : Rat) n) (m : Rat)).symm
  rw [hfun] at hAB
  have hzero : q * q - (m : Rat) = 0 := eq_zero_of_tendsToZero_const hAB
  have hqm : q * q = (m : Rat) := by
    have hh := eq_add_of_sub_eq hzero
    rwa [Rat.add_zero] at hh
  exact no_rat_sqrt m (not_sq_radicand m hm) ⟨q, hqm⟩

#print axioms not_sq_radicand
#print axioms no_rat_sqrt
#print axioms ofRat_inj
#print axioms sqrt_radicand_irrational

end SounioSqrt.RealCauchyField
