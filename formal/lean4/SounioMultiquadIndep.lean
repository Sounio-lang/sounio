import SounioSqrtFieldReal
import SounioNewtonSqrtImpl
import SounioRootedFieldReal

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

/-! ## 5. Linear independence of `{1, √3}`: base case `√5 ∉ ℚ(√3)`.

    We work in the Mathlib-free Cauchy-quotient reals `Real := Quotient realSetoid` assembled in
    `SounioRootedFieldReal`, with the rational embedding `qR q := mkR (ofRat q)` and the first
    Newton root `rootR 0 = √3` (since `primeNatLit 0 = 3`).

    The deliverable is `sqrt5_not_in_Q_sqrt3`: no `a + b√3` (a,b ∈ ℚ) squares to `5`. Its heart is
    `indep_1_R3`, the ℚ-linear independence of `{1, √3}`. -/

/-- The rational embedding into the quotient reals: class of the constant Cauchy sequence. -/
def qR (q : Rat) : Real := mkR (SounioRealCauchy.ofRat q)

/-- `qR` is additive (constant sequences add pointwise). -/
theorem qR_add (x y : Rat) : addR (qR x) (qR y) = qR (x + y) := by
  show mkR (addC (SounioRealCauchy.ofRat x) (SounioRealCauchy.ofRat y))
      = mkR (SounioRealCauchy.ofRat (x + y))
  exact mk_eq_of_seq_eq (fun _ => rfl)

/-- `qR` is multiplicative (constant sequences multiply pointwise). -/
theorem qR_mul (x y : Rat) : mulR (qR x) (qR y) = qR (x * y) := by
  show mkR (mulC (SounioRealCauchy.ofRat x) (SounioRealCauchy.ofRat y))
      = mkR (SounioRealCauchy.ofRat (x * y))
  exact mk_eq_of_seq_eq (fun _ => rfl)

/-- `qR 0` is the additive identity (`zero` is literally `ofRat 0`). -/
theorem qR_zero : qR 0 = zeroR' := rfl

/-- `qR 1` is the multiplicative identity (`one` is literally `ofRat 1`). -/
theorem qR_one : qR 1 = oneR := rfl

/-- Negation commutes with the embedding (`negC` negates the constant sequence pointwise). -/
theorem qR_neg (p : Rat) : negR (qR p) = qR (-p) := by
  show mkR (negC (SounioRealCauchy.ofRat p)) = mkR (SounioRealCauchy.ofRat (-p))
  exact mk_eq_of_seq_eq (fun _ => rfl)

/-- The embedding is injective: from `qR x = qR y` recover `x = y` via `Quotient.exact` and
    `ofRat_inj`. -/
theorem qR_inj {x y : Rat} (h : qR x = qR y) : x = y := by
  apply ofRat_inj
  exact Quotient.exact h

/-- Zero annihilates on the left in the quotient ring. -/
theorem zeroR'_mul (x : Real) : mulR zeroR' x = zeroR' := by
  refine Quotient.inductionOn x (fun a => ?_)
  exact mk_eq_of_seq_eq (fun n => Rat.zero_mul (a.seq n))

/-- Abelian regrouping `(w + x) + (x + z) = (w + z) + (x + x)`, from `addR_assoc`/`addR_comm`. -/
theorem four_regroup (w x z : Real) :
    addR (addR w x) (addR x z) = addR (addR w z) (addR x x) := by
  rw [addR_assoc w x (addR x z), ← addR_assoc x x z, addR_comm (addR x x) z,
      ← addR_assoc w z (addR x x)]

/-- `Rat` cross-term identity: `a·b + a·b = 2·a·b`. -/
theorem rat_cross (a b : Rat) : a * b + a * b = 2 * a * b := by
  rw [show (2 : Rat) = 1 + 1 from by native_decide, Rat.add_mul, Rat.one_mul, Rat.add_mul]

/-- `Rat` diagonal identity: `b·(b·3) = 3·(b·b)`. -/
theorem rat_diag (b : Rat) : b * (b * 3) = 3 * (b * b) := by
  rw [Rat.mul_comm b 3, ← Rat.mul_assoc b 3 b, Rat.mul_comm b 3, Rat.mul_assoc 3 b b]

/-- `Rat` identity for the `a = 0` branch: `(3·b)·(3·b) = 3·(3·(b·b))`. -/
theorem rat_three (b : Rat) : (3 * b) * (3 * b) = 3 * (3 * (b * b)) := by
  rw [Rat.mul_assoc 3 b (3 * b), ← Rat.mul_assoc b 3 b, Rat.mul_comm b 3, Rat.mul_assoc 3 b b]

/-- `Rat` has no zero divisors: multiply by the inverse of the nonzero factor. -/
theorem rat_mul_eq_zero {a b : Rat} (h : a * b = 0) : a = 0 ∨ b = 0 := by
  rcases Classical.em (a = 0) with ha | ha
  · exact Or.inl ha
  · refine Or.inr ?_
    have h2 : a⁻¹ * (a * b) = a⁻¹ * 0 := by rw [h]
    rw [← Rat.mul_assoc, Rat.mul_comm a⁻¹ a, Rat.mul_inv_cancel a ha, Rat.one_mul,
        Rat.mul_zero] at h2
    exact h2

/-- `(√3)² = 3` in the quotient reals: `rootR_sq 0` collapsed through `rfOfNat_real` to `qR 3`. -/
theorem R3_sq : mulR (rootR 0) (rootR 0) = qR 3 := by
  rw [rootR_sq 0, rfOfNat_real (primeNatLit 0)]
  have H : rfOfNat Rat.add 0 1 (primeNatLit 0) = (3 : Rat) := by native_decide
  show mkR (SounioRealCauchy.ofRat (rfOfNat Rat.add 0 1 (primeNatLit 0)))
      = mkR (SounioRealCauchy.ofRat (3 : Rat))
  rw [H]

/-- `√3` is irrational: it is not the embedding of any rational. (Squaring would give a rational
    square root of `3`, impossible by `no_rat_sqrt 3`.) -/
theorem R3_irrational : ¬ ∃ q : Rat, rootR 0 = qR q := by
  intro hex
  obtain ⟨q, hq⟩ := hex
  have hsq : mulR (rootR 0) (rootR 0) = qR (q * q) := by rw [hq, qR_mul]
  rw [R3_sq] at hsq
  have h3 : (3 : Rat) = q * q := qR_inj hsq
  apply no_rat_sqrt 3 (not_sq_radicand 3 (Or.inl rfl))
  refine ⟨q, ?_⟩
  rw [← h3]
  native_decide

/-- **The square expansion in `ℚ(√3)`.** `(a + b√3)² = (a² + 3b²) + (2ab)√3`. Proved by
    `right`/`left` distributivity, collapsing the four cross terms with `mulR_assoc`/`mulR_comm`,
    `R3_sq` (for `√3·√3 = 3`), and the embedding homomorphism lemmas `qR_mul`/`qR_add`, then a
    single abelian regrouping (`four_regroup`) and the `Rat` coefficient identities. -/
theorem E_sq (a b : Rat) :
    mulR (addR (qR a) (mulR (qR b) (rootR 0))) (addR (qR a) (mulR (qR b) (rootR 0)))
      = addR (qR (a * a + 3 * (b * b))) (mulR (qR (2 * a * b)) (rootR 0)) := by
  have e1 : mulR (qR a) (qR a) = qR (a * a) := qR_mul a a
  have e2 : mulR (qR a) (mulR (qR b) (rootR 0)) = mulR (qR (a * b)) (rootR 0) := by
    rw [← mulR_assoc, qR_mul]
  have e3 : mulR (mulR (qR b) (rootR 0)) (qR a) = mulR (qR (a * b)) (rootR 0) := by
    rw [mulR_comm (mulR (qR b) (rootR 0)) (qR a), ← mulR_assoc, qR_mul]
  have e4 : mulR (mulR (qR b) (rootR 0)) (mulR (qR b) (rootR 0)) = qR (3 * (b * b)) := by
    rw [mulR_assoc, ← mulR_assoc (rootR 0) (qR b) (rootR 0), mulR_comm (rootR 0) (qR b),
        mulR_assoc (qR b) (rootR 0) (rootR 0), R3_sq, qR_mul, qR_mul, rat_diag]
  rw [rightDistribR, leftDistribR, leftDistribR, e1, e2, e3, e4, four_regroup, qR_add,
      ← rightDistribR, qR_add, rat_cross]

/-- **ℚ-linear independence of `{1, √3}`.** If `p + q√3 = r` with `p,q,r ∈ ℚ`, then `q = 0` and
    `p = r`. The `q = 0` half is the crux: were `q ≠ 0`, isolating and dividing by `q` would
    exhibit `√3` as a rational, contradicting `R3_irrational`. -/
theorem indep_1_R3 (p q r : Rat)
    (h : addR (qR p) (mulR (qR q) (rootR 0)) = qR r) : p = r ∧ q = 0 := by
  have hq0 : q = 0 := by
    refine Classical.byContradiction (fun hqne => ?_)
    -- isolate the √3 term: q√3 = (-p + r)
    have hM : mulR (qR q) (rootR 0) = qR (-p + r) := by
      have hstep := congrArg (addR (negR (qR p))) h
      rw [← addR_assoc, addR_comm (negR (qR p)) (qR p), addR_neg,
          addR_comm zeroR' (mulR (qR q) (rootR 0)), addR_zero, qR_neg, qR_add] at hstep
      exact hstep
    -- divide by q on the left: √3 = q⁻¹·(-p + r), a rational — contradiction
    have hqq : q⁻¹ * q = 1 := by rw [Rat.mul_comm]; exact Rat.mul_inv_cancel q hqne
    have hmul := congrArg (mulR (qR q⁻¹)) hM
    rw [← mulR_assoc, qR_mul, hqq, qR_one, mulR_comm oneR (rootR 0), mulR_one, qR_mul] at hmul
    exact R3_irrational ⟨q⁻¹ * (-p + r), hmul⟩
  have hpr : p = r := by
    rw [hq0, qR_zero, zeroR'_mul, addR_zero] at h
    exact qR_inj h
  exact ⟨hpr, hq0⟩

/-- **`√5 ∉ ℚ(√3)` — base case of the multiquadratic linear-independence tower.** No element
    `a + b√3` of `ℚ(√3)` squares to `5`.

    Expand the square with `E_sq` to `(a² + 3b²) + (2ab)√3 = 5`; linear independence
    (`indep_1_R3`) forces `a² + 3b² = 5` and `2ab = 0`. As `ℚ` has no zero divisors and `2 ≠ 0`,
    either `a = 0` (whence `(3b)² = 15`) or `b = 0` (whence `a² = 5`); both contradict the
    irrationality core `no_rat_sqrt` on the non-squares `15` and `5`. -/
theorem sqrt5_not_in_Q_sqrt3 :
    ¬ ∃ a b : Rat,
      mulR (addR (qR a) (mulR (qR b) (rootR 0)))
           (addR (qR a) (mulR (qR b) (rootR 0))) = qR 5 := by
  intro hex
  obtain ⟨a, b, h⟩ := hex
  rw [E_sq] at h
  obtain ⟨hdiag, hcross⟩ := indep_1_R3 (a * a + 3 * (b * b)) (2 * a * b) 5 h
  rcases rat_mul_eq_zero hcross with h2a | hb0
  · -- 2·a = 0 ⇒ a = 0 (since 2 ≠ 0)
    rcases rat_mul_eq_zero h2a with h2 | ha0
    · exact absurd h2 (by native_decide : (2 : Rat) ≠ 0)
    · -- a = 0 ⇒ 3·(b·b) = 5 ⇒ (3b)² = 15, contradicting no_rat_sqrt 15
      rw [ha0, Rat.zero_mul, Rat.zero_add] at hdiag
      have hb15 : (3 * b) * (3 * b) = (15 : Rat) := by
        rw [rat_three b, hdiag]; native_decide
      exact no_rat_sqrt 15 (not_sq_radicand 15 (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
        ⟨3 * b, by rw [hb15]; native_decide⟩
  · -- b = 0 ⇒ a·a = 5, contradicting no_rat_sqrt 5
    rw [hb0, Rat.mul_zero, Rat.mul_zero, Rat.add_zero] at hdiag
    exact no_rat_sqrt 5 (not_sq_radicand 5 (Or.inr (Or.inl rfl)))
      ⟨a, by rw [hdiag]; native_decide⟩

#print axioms sqrt5_not_in_Q_sqrt3

end SounioSqrt.RealCauchyField
