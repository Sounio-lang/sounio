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

/-! ## 6. LEVA 1 — foundation + mechanical pieces for `√11 ∉ ℚ(√3,√5)`.

    This section adds the `√5` analogues of the `√3` helpers, the base-shape lemma
    `sqrt3_not_in_Q_sqrt5` (target `3` over `ℚ(√5)`), the four-generator product table and the
    16-term squared-norm expansion `E_sq4` over the basis `{1, √3, √5, √15}`, and the pure-rational
    zero-divisor lemma `Qsqrt5_no_zero_div` for `ℚ(√5)`. LEVA 2 consumes these by name. -/

/-- Abelian regrouping `(w + x) + (y + z) = (w + y) + (x + z)` on the quotient reals. -/
theorem addR_swap4 (w x y z : Real) :
    addR (addR w x) (addR y z) = addR (addR w y) (addR x z) := by
  rw [addR_assoc w x (addR y z), ← addR_assoc x y z, addR_comm x y, addR_assoc y x z,
      ← addR_assoc w y (addR x z)]

/-- Abelian regrouping `(w + x) + (y + z) = (w + z) + (x + y)` on the quotient reals. -/
theorem addR_regroup_ad (w x y z : Real) :
    addR (addR w x) (addR y z) = addR (addR w z) (addR x y) := by
  rw [addR_assoc w x (addR y z), ← addR_assoc x y z, addR_comm (addR x y) z,
      ← addR_assoc w z (addR x y)]

/-- `Rat`: from `x + y = 0` recover `x = -y`. -/
theorem rat_eq_neg_of_add_eq_zero {x y : Rat} (h : x + y = 0) : x = -y := by
  have h2 : (x + y) + (-y) = (0:Rat) + (-y) := by rw [h]
  rw [Rat.add_assoc, Rat.add_neg_cancel, Rat.add_zero, Rat.zero_add] at h2
  exact h2

/-- `Rat` 2×2-determinant expansion `(a² − 5c²)·x = a(ax) − 5(c(cx))`. -/
theorem rat_det_mul (a c x : Rat) : (a*a - 5*(c*c)) * x = a*(a*x) - 5*(c*(c*x)) := by
  rw [Rat.sub_eq_add_neg, Rat.add_mul, Rat.neg_mul, Rat.mul_assoc a a x,
      Rat.mul_assoc 5 (c*c) x, Rat.mul_assoc c c x, ← Rat.sub_eq_add_neg]

/-- `Rat` diagonal identity (radicand `5`): `c·(c·5) = 5·(c·c)`. -/
theorem rat_diag5 (c : Rat) : c * (c * 5) = 5 * (c * c) := by
  rw [Rat.mul_comm c 5, ← Rat.mul_assoc c 5 c, Rat.mul_comm c 5, Rat.mul_assoc 5 c c]

/-- `Rat` identity for the `a = 0` branch over `ℚ(√5)`: `(5·c)·(5·c) = 5·(5·(c·c))`. -/
theorem rat_five (c : Rat) : (5 * c) * (5 * c) = 5 * (5 * (c * c)) := by
  rw [Rat.mul_assoc 5 c (5 * c), ← Rat.mul_assoc c 5 c, Rat.mul_comm c 5, Rat.mul_assoc 5 c c]

/-- `Rat` cross-diagonal identity: `b·(d·3) = 3·(b·d)`. -/
theorem rat_diag_bd (b d : Rat) : b * (d * 3) = 3 * (b * d) := by
  rw [Rat.mul_comm d 3, ← Rat.mul_assoc b 3 d, Rat.mul_comm b 3, Rat.mul_assoc 3 b d]

/-- `Rat` reassociation `a·(5·(c·d)) = 5·(c·(a·d))`. -/
theorem rat_5acd (a c d : Rat) : a * (5 * (c * d)) = 5 * (c * (a * d)) := by
  rw [← Rat.mul_assoc a 5 (c*d), Rat.mul_comm a 5, Rat.mul_assoc 5 a (c*d),
      ← Rat.mul_assoc a c d, Rat.mul_comm a c, Rat.mul_assoc c a d]

/-- `Rat` reassociation `5·(c·(c·d)) = c·(5·(c·d))`. -/
theorem rat_5ccd (c d : Rat) : 5 * (c * (c * d)) = c * (5 * (c * d)) := by
  rw [← Rat.mul_assoc 5 c (c*d), Rat.mul_comm 5 c, Rat.mul_assoc c 5 (c*d)]

/-- `Rat` reassociation `a·(c·b) = c·(a·b)`. -/
theorem rat_acb (a b c : Rat) : a * (c * b) = c * (a * b) := by
  rw [← Rat.mul_assoc a c b, Rat.mul_comm a c, Rat.mul_assoc c a b]

/-! ### 6.1. The `√5` helpers (analogues of `R3_sq`/`R3_irrational`/`indep_1_R3`). -/

/-- `(√5)² = 5` in the quotient reals: `rootR_sq 1` collapsed through `rfOfNat_real` to `qR 5`
    (`primeNatLit 1 = 5`). -/
theorem R5_sq : mulR (rootR 1) (rootR 1) = qR 5 := by
  rw [rootR_sq 1, rfOfNat_real (primeNatLit 1)]
  have H : rfOfNat Rat.add 0 1 (primeNatLit 1) = (5 : Rat) := by native_decide
  show mkR (SounioRealCauchy.ofRat (rfOfNat Rat.add 0 1 (primeNatLit 1)))
      = mkR (SounioRealCauchy.ofRat (5 : Rat))
  rw [H]

/-- `√5` is irrational: squaring would give a rational square root of `5`. -/
theorem R5_irrational : ¬ ∃ q : Rat, rootR 1 = qR q := by
  intro hex
  obtain ⟨q, hq⟩ := hex
  have hsq : mulR (rootR 1) (rootR 1) = qR (q * q) := by rw [hq, qR_mul]
  rw [R5_sq] at hsq
  have h5 : (5 : Rat) = q * q := qR_inj hsq
  apply no_rat_sqrt 5 (not_sq_radicand 5 (Or.inr (Or.inl rfl)))
  refine ⟨q, ?_⟩
  rw [← h5]
  native_decide

/-- **ℚ-linear independence of `{1, √5}`.** If `p + q√5 = r`, then `p = r` and `q = 0`. -/
theorem indep_1_R5 (p q r : Rat)
    (h : addR (qR p) (mulR (qR q) (rootR 1)) = qR r) : p = r ∧ q = 0 := by
  have hq0 : q = 0 := by
    refine Classical.byContradiction (fun hqne => ?_)
    have hM : mulR (qR q) (rootR 1) = qR (-p + r) := by
      have hstep := congrArg (addR (negR (qR p))) h
      rw [← addR_assoc, addR_comm (negR (qR p)) (qR p), addR_neg,
          addR_comm zeroR' (mulR (qR q) (rootR 1)), addR_zero, qR_neg, qR_add] at hstep
      exact hstep
    have hqq : q⁻¹ * q = 1 := by rw [Rat.mul_comm]; exact Rat.mul_inv_cancel q hqne
    have hmul := congrArg (mulR (qR q⁻¹)) hM
    rw [← mulR_assoc, qR_mul, hqq, qR_one, mulR_comm oneR (rootR 1), mulR_one, qR_mul] at hmul
    exact R5_irrational ⟨q⁻¹ * (-p + r), hmul⟩
  have hpr : p = r := by
    rw [hq0, qR_zero, zeroR'_mul, addR_zero] at h
    exact qR_inj h
  exact ⟨hpr, hq0⟩

/-! ### 6.2. Base-shape lemma: `√3 ∉ ℚ(√5)`. -/

/-- **The square expansion in `ℚ(√5)`.** `(a + c√5)² = (a² + 5c²) + (2ac)√5`. -/
theorem E5_sq (a c : Rat) :
    mulR (addR (qR a) (mulR (qR c) (rootR 1))) (addR (qR a) (mulR (qR c) (rootR 1)))
      = addR (qR (a * a + 5 * (c * c))) (mulR (qR (2 * a * c)) (rootR 1)) := by
  have e1 : mulR (qR a) (qR a) = qR (a * a) := qR_mul a a
  have e2 : mulR (qR a) (mulR (qR c) (rootR 1)) = mulR (qR (a * c)) (rootR 1) := by
    rw [← mulR_assoc, qR_mul]
  have e3 : mulR (mulR (qR c) (rootR 1)) (qR a) = mulR (qR (a * c)) (rootR 1) := by
    rw [mulR_comm (mulR (qR c) (rootR 1)) (qR a), ← mulR_assoc, qR_mul]
  have e4 : mulR (mulR (qR c) (rootR 1)) (mulR (qR c) (rootR 1)) = qR (5 * (c * c)) := by
    rw [mulR_assoc, ← mulR_assoc (rootR 1) (qR c) (rootR 1), mulR_comm (rootR 1) (qR c),
        mulR_assoc (qR c) (rootR 1) (rootR 1), R5_sq, qR_mul, qR_mul, rat_diag5]
  rw [rightDistribR, leftDistribR, leftDistribR, e1, e2, e3, e4, four_regroup, qR_add,
      ← rightDistribR, qR_add, rat_cross]

/-- **`√3 ∉ ℚ(√5)` — base-shape over the conjugate quadratic field.** No element `a + c√5` of
    `ℚ(√5)` squares to `3`. (Linear independence `indep_1_R5` plus the irrationality of `√3` and
    `√15`.) -/
theorem sqrt3_not_in_Q_sqrt5 :
    ¬ ∃ a c : Rat,
      mulR (addR (qR a) (mulR (qR c) (rootR 1)))
           (addR (qR a) (mulR (qR c) (rootR 1))) = qR 3 := by
  intro hex
  obtain ⟨a, c, h⟩ := hex
  rw [E5_sq] at h
  obtain ⟨hdiag, hcross⟩ := indep_1_R5 (a * a + 5 * (c * c)) (2 * a * c) 3 h
  rcases rat_mul_eq_zero hcross with h2a | hc0
  · rcases rat_mul_eq_zero h2a with h2 | ha0
    · exact absurd h2 (by native_decide : (2 : Rat) ≠ 0)
    · rw [ha0, Rat.zero_mul, Rat.zero_add] at hdiag
      have hc15 : (5 * c) * (5 * c) = (15 : Rat) := by
        rw [rat_five c, hdiag]; native_decide
      exact no_rat_sqrt 15 (not_sq_radicand 15 (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
        ⟨5 * c, by rw [hc15]; native_decide⟩
  · rw [hc0, Rat.mul_zero, Rat.mul_zero, Rat.add_zero] at hdiag
    exact no_rat_sqrt 3 (not_sq_radicand 3 (Or.inl rfl))
      ⟨a, by rw [hdiag]; native_decide⟩

/-! ### 6.3. The fourth generator `√15`, the product table, and the squared-norm expansion. -/

/-- `√15 = √3·√5` in the quotient reals. -/
noncomputable def R15 : Real := mulR (rootR 0) (rootR 1)

/-- Definitional unfolding of `R15`. -/
theorem R15_def : R15 = mulR (rootR 0) (rootR 1) := rfl

/-- A general element of `ℚ(√3,√5)` in the basis `{1, √3, √5, √15}`. -/
noncomputable def alpha4 (a b c d : Rat) : Real :=
  addR (addR (addR (qR a) (mulR (qR b) (rootR 0))) (mulR (qR c) (rootR 1))) (mulR (qR d) R15)

/-- Product table: `√15·√15 = 15` (via `mulR_assoc`/`mulR_comm` + `R3_sq` + `R5_sq`). -/
theorem R15_sq : mulR R15 R15 = qR 15 := by
  show mulR (mulR (rootR 0) (rootR 1)) (mulR (rootR 0) (rootR 1)) = qR 15
  rw [mulR_assoc (rootR 0) (rootR 1) (mulR (rootR 0) (rootR 1)),
      ← mulR_assoc (rootR 1) (rootR 0) (rootR 1),
      mulR_comm (rootR 1) (rootR 0),
      mulR_assoc (rootR 0) (rootR 1) (rootR 1),
      ← mulR_assoc (rootR 0) (rootR 0) (mulR (rootR 1) (rootR 1)),
      R3_sq, R5_sq, qR_mul]
  have h : (3 : Rat) * 5 = 15 := by native_decide
  rw [h]

/-- Product table: `√3·√15 = 3·√5`. -/
theorem R3_R15 : mulR (rootR 0) R15 = mulR (qR 3) (rootR 1) := by
  show mulR (rootR 0) (mulR (rootR 0) (rootR 1)) = mulR (qR 3) (rootR 1)
  rw [← mulR_assoc (rootR 0) (rootR 0) (rootR 1), R3_sq]

/-- Product table: `√5·√15 = 5·√3`. -/
theorem R5_R15 : mulR (rootR 1) R15 = mulR (qR 5) (rootR 0) := by
  show mulR (rootR 1) (mulR (rootR 0) (rootR 1)) = mulR (qR 5) (rootR 0)
  rw [mulR_comm (rootR 0) (rootR 1), ← mulR_assoc (rootR 1) (rootR 1) (rootR 0), R5_sq]

/-- **General `(u + v√5)²` formula** with `u, v` arbitrary reals:
    `(u + v√5)² = (u² + 5v²) + (2uv)√5`. The reduction that turns the 4-generator square into
    two `ℚ(√3)` squares and a `ℚ(√3)` product. -/
theorem gen_sq5 (u v : Real) :
    mulR (addR u (mulR v (rootR 1))) (addR u (mulR v (rootR 1)))
      = addR (addR (mulR u u) (mulR (qR 5) (mulR v v)))
             (mulR (addR (mulR u v) (mulR u v)) (rootR 1)) := by
  have e2 : mulR u (mulR v (rootR 1)) = mulR (mulR u v) (rootR 1) := by rw [← mulR_assoc]
  have e3 : mulR (mulR v (rootR 1)) u = mulR (mulR u v) (rootR 1) := by
    rw [mulR_comm (mulR v (rootR 1)) u, ← mulR_assoc]
  have e4 : mulR (mulR v (rootR 1)) (mulR v (rootR 1)) = mulR (qR 5) (mulR v v) := by
    rw [mulR_assoc v (rootR 1) (mulR v (rootR 1)), ← mulR_assoc (rootR 1) v (rootR 1),
        mulR_comm (rootR 1) v, mulR_assoc v (rootR 1) (rootR 1), R5_sq,
        mulR_comm v (qR 5), ← mulR_assoc v (qR 5) v, mulR_comm v (qR 5),
        mulR_assoc (qR 5) v v]
  rw [rightDistribR, leftDistribR, leftDistribR, e2, e3, e4, four_regroup, ← rightDistribR]

/-- **The product in `ℚ(√3)`.** `(a + b√3)(c + d√3) = (ac + 3bd) + (ad + bc)√3`. -/
theorem E_mul (a b c d : Rat) :
    mulR (addR (qR a) (mulR (qR b) (rootR 0))) (addR (qR c) (mulR (qR d) (rootR 0)))
      = addR (qR (a*c + 3*(b*d))) (mulR (qR (a*d + b*c)) (rootR 0)) := by
  have e1 : mulR (qR a) (qR c) = qR (a * c) := qR_mul a c
  have e2 : mulR (qR a) (mulR (qR d) (rootR 0)) = mulR (qR (a * d)) (rootR 0) := by
    rw [← mulR_assoc, qR_mul]
  have e3 : mulR (mulR (qR b) (rootR 0)) (qR c) = mulR (qR (b * c)) (rootR 0) := by
    rw [mulR_assoc (qR b) (rootR 0) (qR c), mulR_comm (rootR 0) (qR c), ← mulR_assoc, qR_mul]
  have e4 : mulR (mulR (qR b) (rootR 0)) (mulR (qR d) (rootR 0)) = qR (3 * (b * d)) := by
    rw [mulR_assoc, ← mulR_assoc (rootR 0) (qR d) (rootR 0), mulR_comm (rootR 0) (qR d),
        mulR_assoc (qR d) (rootR 0) (rootR 0), R3_sq, qR_mul, qR_mul, rat_diag_bd]
  rw [rightDistribR, leftDistribR, leftDistribR, e1, e2, e3, e4, addR_regroup_ad, qR_add,
      ← rightDistribR (qR (a*d)) (qR (b*c)) (rootR 0), qR_add]

/-- **The 16-term squared-norm expansion over the basis `{1, √3, √5, √15}`.**
    `(a + b√3 + c√5 + d√15)² = (a²+3b²+5c²+15d²) + (2ab+10cd)√3 + (2ac+6bd)√5 + (2ad+2bc)√15`.

    Proof route (no `ring`): rewrite `α = u + v√5` with `u = a+b√3`, `v = c+d√3` (`hu`); square via
    `gen_sq5`; expand `u²`, `v²`, `uv` with `E_sq`/`E_mul`; collect each basis bucket
    (`addR_swap4`, `rightDistribR`, `qR_add`) and close the rational coefficient arithmetic with the
    explicit `Rat` lemma chains `hTC`/`hT3c`/`hT5c`/`hT15c`. -/
theorem E_sq4 (a b c d : Rat) :
    mulR (alpha4 a b c d) (alpha4 a b c d)
      = addR (addR (addR
          (qR (a*a + 3*(b*b) + 5*(c*c) + 15*(d*d)))
          (mulR (qR (2*a*b + 10*(c*d))) (rootR 0)))
          (mulR (qR (2*a*c + 6*(b*d))) (rootR 1)))
          (mulR (qR (2*a*d + 2*(b*c))) R15) := by
  rw [R15_def]
  have hu : alpha4 a b c d
      = addR (addR (qR a) (mulR (qR b) (rootR 0)))
             (mulR (addR (qR c) (mulR (qR d) (rootR 0))) (rootR 1)) := by
    show addR (addR (addR (qR a) (mulR (qR b) (rootR 0))) (mulR (qR c) (rootR 1)))
              (mulR (qR d) R15)
        = addR (addR (qR a) (mulR (qR b) (rootR 0)))
               (mulR (addR (qR c) (mulR (qR d) (rootR 0))) (rootR 1))
    rw [R15_def, rightDistribR (qR c) (mulR (qR d) (rootR 0)) (rootR 1),
        mulR_assoc (qR d) (rootR 0) (rootR 1), ← addR_assoc]
  have hTC : (a*a + 3*(b*b)) + 5*(c*c + 3*(d*d))
      = a*a + 3*(b*b) + 5*(c*c) + 15*(d*d) := by
    rw [Rat.mul_add, ← Rat.mul_assoc 5 3 (d*d),
        show (5:Rat)*3 = 15 from by native_decide, ← Rat.add_assoc]
  have hT3c : 2*a*b + 5*(2*c*d) = 2*a*b + 10*(c*d) := by
    rw [← Rat.mul_assoc 5 (2*c) d, ← Rat.mul_assoc 5 2 c,
        show (5:Rat)*2 = 10 from by native_decide, Rat.mul_assoc 10 c d]
  have h3bd : 3*(b*d) + 3*(b*d) = 6*(b*d) := by
    rw [← Rat.add_mul, show (3:Rat)+3 = 6 from by native_decide]
  have h2bc : b*c + b*c = 2*(b*c) := by
    rw [show (2:Rat) = 1+1 from by native_decide, Rat.add_mul, Rat.one_mul]
  have hT5c : (a*c+3*(b*d)) + (a*c+3*(b*d)) = 2*a*c + 6*(b*d) := by
    rw [add_swap4, rat_cross a c, h3bd]
  have hT15c : (a*d+b*c) + (a*d+b*c) = 2*a*d + 2*(b*c) := by
    rw [add_swap4, rat_cross a d, h2bc]
  have hA : mulR (qR 5) (addR (qR (c*c + 3*(d*d))) (mulR (qR (2*c*d)) (rootR 0)))
      = addR (qR (5*(c*c + 3*(d*d)))) (mulR (qR (5*(2*c*d))) (rootR 0)) := by
    rw [leftDistribR, qR_mul, ← mulR_assoc, qR_mul]
  have hPartA :
      addR (addR (qR (a*a + 3*(b*b))) (mulR (qR (2*a*b)) (rootR 0)))
           (mulR (qR 5) (addR (qR (c*c + 3*(d*d))) (mulR (qR (2*c*d)) (rootR 0))))
        = addR (qR (a*a + 3*(b*b) + 5*(c*c) + 15*(d*d)))
               (mulR (qR (2*a*b + 10*(c*d))) (rootR 0)) := by
    rw [hA, addR_swap4, qR_add, ← rightDistribR (qR (2*a*b)) (qR (5*(2*c*d))) (rootR 0),
        qR_add, hTC, hT3c]
  have hPartB :
      mulR (addR (addR (qR (a*c+3*(b*d))) (mulR (qR (a*d+b*c)) (rootR 0)))
                 (addR (qR (a*c+3*(b*d))) (mulR (qR (a*d+b*c)) (rootR 0)))) (rootR 1)
        = addR (mulR (qR (2*a*c + 6*(b*d))) (rootR 1))
               (mulR (qR (2*a*d + 2*(b*c))) (mulR (rootR 0) (rootR 1))) := by
    rw [addR_swap4, qR_add, ← rightDistribR (qR (a*d+b*c)) (qR (a*d+b*c)) (rootR 0), qR_add,
        hT5c, hT15c, rightDistribR, mulR_assoc (qR (2*a*d+2*(b*c))) (rootR 0) (rootR 1)]
  rw [hu, gen_sq5, E_sq a b, E_sq c d, E_mul a b c d, hPartA, hPartB, ← addR_assoc]

/-! ### 6.4. Pure-rational zero-divisor lemma for `ℚ(√5)`. -/

/-- **`ℚ(√5)` has no zero divisors (rational coordinate form).** If `(a + c√5)(b + d√5) = 0`,
    encoded by the rational identities `a·b + 5(c·d) = 0` and `a·d + c·b = 0`, then either the left
    factor vanishes (`a = c = 0`) or the right factor vanishes (`b = d = 0`).

    Proof (pure `Rat`, only the irrationality core `no_rat_sqrt 5`): case on `(a,c) = (0,0)`.
    Otherwise the determinant `det := a² − 5c²` is nonzero (else `(a/c)² = 5` or `a = 0` with
    `c = 0`), and the linear eliminations `det·b = 0`, `det·d = 0` (substituting the two hypotheses)
    force `b = d = 0`. -/
theorem Qsqrt5_no_zero_div (a b c d : Rat)
    (h1 : a*b + 5*(c*d) = 0) (h2 : a*d + c*b = 0) :
    (a = 0 ∧ c = 0) ∨ (b = 0 ∧ d = 0) := by
  rcases Classical.em (a = 0 ∧ c = 0) with hac | hac
  · exact Or.inl hac
  · refine Or.inr ?_
    have hdet : a*a - 5*(c*c) ≠ 0 := by
      intro hd0
      have haa : a*a = 5*(c*c) := by
        have h := eq_add_of_sub_eq hd0
        rwa [Rat.add_zero] at h
      rcases Classical.em (c = 0) with hc0 | hc0
      · rw [hc0, Rat.mul_zero, Rat.mul_zero] at haa
        have ha0 : a = 0 := by
          rcases rat_mul_eq_zero haa with h | h <;> exact h
        exact hac ⟨ha0, hc0⟩
      · have hcc : c * c⁻¹ = 1 := Rat.mul_inv_cancel c hc0
        have key : (a * c⁻¹) * (a * c⁻¹) = 5 := by
          have rearr : (a * c⁻¹) * (a * c⁻¹) = (a*a) * (c⁻¹ * c⁻¹) := by
            rw [Rat.mul_assoc a c⁻¹ (a * c⁻¹), ← Rat.mul_assoc c⁻¹ a c⁻¹, Rat.mul_comm c⁻¹ a,
                Rat.mul_assoc a c⁻¹ c⁻¹, ← Rat.mul_assoc a a (c⁻¹ * c⁻¹)]
          rw [rearr, haa, Rat.mul_assoc 5 (c*c) (c⁻¹ * c⁻¹)]
          have inner : (c*c)*(c⁻¹ * c⁻¹) = 1 := by
            rw [Rat.mul_assoc c c (c⁻¹ * c⁻¹), ← Rat.mul_assoc c c⁻¹ c⁻¹, hcc, Rat.one_mul, hcc]
          rw [inner, Rat.mul_one]
        exact no_rat_sqrt 5 (not_sq_radicand 5 (Or.inr (Or.inl rfl)))
          ⟨a * c⁻¹, by rw [key]; native_decide⟩
    have hab : a*b = -(5*(c*d)) := rat_eq_neg_of_add_eq_zero h1
    have h2c : c*b + a*d = 0 := by rw [Rat.add_comm]; exact h2
    have hcb : c*b = -(a*d) := rat_eq_neg_of_add_eq_zero h2c
    have had : a*d = -(c*b) := rat_eq_neg_of_add_eq_zero h2
    have h1c : 5*(c*d) + a*b = 0 := by rw [Rat.add_comm]; exact h1
    have h5cd : 5*(c*d) = -(a*b) := rat_eq_neg_of_add_eq_zero h1c
    have hdetb : (a*a - 5*(c*c)) * b = 0 := by
      rw [rat_det_mul a c b, hab, hcb, Rat.mul_neg, Rat.mul_neg, Rat.mul_neg, rat_5acd,
          Rat.sub_eq_add_neg, Rat.neg_neg, Rat.neg_add_cancel]
    have hdetd : (a*a - 5*(c*c)) * d = 0 := by
      rw [rat_det_mul a c d, had, rat_5ccd, h5cd, Rat.mul_neg, Rat.mul_neg, rat_acb,
          Rat.sub_eq_add_neg, Rat.neg_neg, Rat.neg_add_cancel]
    have hb : b = 0 := by
      rcases rat_mul_eq_zero hdetb with hh | hh
      · exact absurd hh hdet
      · exact hh
    have hd : d = 0 := by
      rcases rat_mul_eq_zero hdetd with hh | hh
      · exact absurd hh hdet
      · exact hh
    exact ⟨hb, hd⟩

#print axioms R5_sq
#print axioms indep_1_R5
#print axioms sqrt3_not_in_Q_sqrt5
#print axioms E_sq4
#print axioms Qsqrt5_no_zero_div

/-! ## 7. LEVA 2 — 4-dimensional ℚ-independence of `{1,√3,√5,√15}` and `√11 ∉ ℚ(√3,√5)`.

    This section closes the multiquadratic tower. It factors the four-generator independence
    through the quadratic subfield `ℚ(√5)`: the heart is `indep4`, which rewrites a general element
    `p + q√3 + s√5 + t√15` as `P + Q·√3` with `P, Q ∈ ℚ(√5)`, inverts the `ℚ(√5)`-coefficient `Q`
    (via `Q5_inv`) and lands a `ℚ(√5)`-representation of `√3`, contradicting `not_R3_in_Q5`. The
    final theorem `sqrt11_not_in_Q_sqrt3_sqrt5` consumes `indep4` + `Qsqrt5_no_zero_div` + the
    irrationality core on the radicands `{11,33,55,165}`. -/

/-- `Rat`: `0 - x = -x` (no `Rat.zero_sub` in core; via `sub_eq_add_neg` + `zero_add`). -/
theorem rat_zero_sub (x : Rat) : (0 : Rat) - x = -x := by
  rw [Rat.sub_eq_add_neg, Rat.zero_add]

/-- `Rat` diagonal identity (radicand `5`, product form): `b·(d·5) = 5·(b·d)`. -/
theorem rat_diag_bd5 (b d : Rat) : b * (d * 5) = 5 * (b * d) := by
  rw [Rat.mul_comm d 5, ← Rat.mul_assoc b 5 d, Rat.mul_comm b 5, Rat.mul_assoc 5 b d]

/-- `Rat`: the `√5` off-diagonal of `(q + t√5)·(its conjugate-inverse numerator)` vanishes, for any
    common denominator factor `e`: `q·(0 − t·e) + t·(q·e) = 0`. -/
theorem rat_cross_inv (q t e : Rat) : q * (0 - t * e) + t * (q * e) = 0 := by
  rw [rat_zero_sub (t * e), Rat.mul_neg, rat_acb t e q, Rat.neg_add_cancel]

/-- `Rat` identity for the `b = 0` branch over `ℚ(√15)`: `(15·d)·(15·d) = 15·(15·(d·d))`. -/
theorem rat_fifteen (d : Rat) : (15 * d) * (15 * d) = 15 * (15 * (d * d)) := by
  rw [Rat.mul_assoc 15 d (15 * d), ← Rat.mul_assoc d 15 d, Rat.mul_comm d 15, Rat.mul_assoc 15 d d]

/-- `Rat`: from `2·x = 0` recover `x = 0` (since `2 ≠ 0`). -/
theorem rat_of_two_mul_zero {x : Rat} (h : 2 * x = 0) : x = 0 := by
  rcases rat_mul_eq_zero h with h2 | hx
  · exact absurd h2 (by native_decide : (2 : Rat) ≠ 0)
  · exact hx

/-- Negation distributes over `addR` on the quotient reals (pointwise `Rat.neg_add`). -/
theorem negR_addR (x y : Real) : negR (addR x y) = addR (negR x) (negR y) := by
  refine Quotient.inductionOn₂ x y (fun a b => ?_)
  show mkR (negC (addC a b)) = mkR (addC (negC a) (negC b))
  exact mk_eq_of_seq_eq (fun n => @Rat.neg_add (a.seq n) (b.seq n))

/-- Negation pulls out of the left factor of `mulR` (pointwise `Rat.neg_mul`). -/
theorem negR_mulR_left (x y : Real) : negR (mulR x y) = mulR (negR x) y := by
  refine Quotient.inductionOn₂ x y (fun a b => ?_)
  show mkR (negC (mulC a b)) = mkR (mulC (negC a) b)
  exact mk_eq_of_seq_eq (fun n => (Rat.neg_mul (a.seq n) (b.seq n)).symm)

/-- Negation of a `ℚ(√5)`-element stays a `ℚ(√5)`-element:
    `−(p + s√5) = (0−p) + (0−s)√5`. -/
theorem negR_Q5 (p s : Rat) :
    negR (addR (qR p) (mulR (qR s) (rootR 1)))
      = addR (qR (0 - p)) (mulR (qR (0 - s)) (rootR 1)) := by
  rw [negR_addR, qR_neg p, negR_mulR_left, qR_neg s, rat_zero_sub p, rat_zero_sub s]

/-- Abelian regrouping `((a₁+a₂)+a₃)+a₄ = (a₁+a₃)+(a₂+a₄)` on the quotient reals. -/
theorem addR_regroup4 (a1 a2 a3 a4 : Real) :
    addR (addR (addR a1 a2) a3) a4 = addR (addR a1 a3) (addR a2 a4) := by
  rw [addR_assoc (addR a1 a2) a3 a4, addR_swap4 a1 a2 a3 a4]

/-! ### 7.1. `ℚ(√5)` product and the `√3 ∉ ℚ(√5)` reformulation. -/

/-- **The product in `ℚ(√5)`.** `(x₀ + x₁√5)(y₀ + y₁√5) = (x₀y₀ + 5x₁y₁) + (x₀y₁ + x₁y₀)√5`.
    The `√5` analogue of the in-file `E_mul` (`rootR 0 → rootR 1`, `R3_sq → R5_sq`,
    `3·(b·d) → 5·(x₁·y₁)`). -/
theorem E_mul5 (x0 x1 y0 y1 : Rat) :
    mulR (addR (qR x0) (mulR (qR x1) (rootR 1))) (addR (qR y0) (mulR (qR y1) (rootR 1)))
      = addR (qR (x0*y0 + 5*(x1*y1))) (mulR (qR (x0*y1 + x1*y0)) (rootR 1)) := by
  have e1 : mulR (qR x0) (qR y0) = qR (x0 * y0) := qR_mul x0 y0
  have e2 : mulR (qR x0) (mulR (qR y1) (rootR 1)) = mulR (qR (x0 * y1)) (rootR 1) := by
    rw [← mulR_assoc, qR_mul]
  have e3 : mulR (mulR (qR x1) (rootR 1)) (qR y0) = mulR (qR (x1 * y0)) (rootR 1) := by
    rw [mulR_assoc (qR x1) (rootR 1) (qR y0), mulR_comm (rootR 1) (qR y0), ← mulR_assoc, qR_mul]
  have e4 : mulR (mulR (qR x1) (rootR 1)) (mulR (qR y1) (rootR 1)) = qR (5 * (x1 * y1)) := by
    rw [mulR_assoc, ← mulR_assoc (rootR 1) (qR y1) (rootR 1), mulR_comm (rootR 1) (qR y1),
        mulR_assoc (qR y1) (rootR 1) (rootR 1), R5_sq, qR_mul, qR_mul, rat_diag_bd5]
  rw [rightDistribR, leftDistribR, leftDistribR, e1, e2, e3, e4, addR_regroup_ad, qR_add,
      ← rightDistribR (qR (x0*y1)) (qR (x1*y0)) (rootR 1), qR_add]

/-- **`√3 ∉ ℚ(√5)` (membership form).** There is no `ℚ(√5)`-representation `x + y√5` of `√3`.
    Squaring such a representation would square to `3` inside `ℚ(√5)`, contradicting
    `sqrt3_not_in_Q_sqrt5`. -/
theorem not_R3_in_Q5 : ¬ ∃ x y : Rat, rootR 0 = addR (qR x) (mulR (qR y) (rootR 1)) := by
  intro hex
  obtain ⟨x, y, hxy⟩ := hex
  have hsq : mulR (addR (qR x) (mulR (qR y) (rootR 1)))
                  (addR (qR x) (mulR (qR y) (rootR 1))) = qR 3 := by
    rw [← hxy, R3_sq]
  exact sqrt3_not_in_Q_sqrt5 ⟨x, y, hsq⟩

/-! ### 7.2. The `ℚ(√5)` inverse for nonzero elements. -/

/-- The `ℚ(√5)`-norm `q² − 5t²` is nonzero whenever `(q,t) ≠ (0,0)`. (If it vanished then
    `(q/t)² = 5` for `t ≠ 0`, impossible by `no_rat_sqrt 5`; and `t = 0` forces `q = 0`.) -/
theorem N5_ne_zero (q t : Rat) (hqt : ¬ (q = 0 ∧ t = 0)) : q*q - 5*(t*t) ≠ 0 := by
  intro hd0
  have haa : q*q = 5*(t*t) := by
    have h := eq_add_of_sub_eq hd0
    rwa [Rat.add_zero] at h
  rcases Classical.em (t = 0) with ht0 | ht0
  · rw [ht0, Rat.mul_zero, Rat.mul_zero] at haa
    have hq0 : q = 0 := by
      rcases rat_mul_eq_zero haa with h | h <;> exact h
    exact hqt ⟨hq0, ht0⟩
  · have htt : t * t⁻¹ = 1 := Rat.mul_inv_cancel t ht0
    have key : (q * t⁻¹) * (q * t⁻¹) = 5 := by
      have rearr : (q * t⁻¹) * (q * t⁻¹) = (q*q) * (t⁻¹ * t⁻¹) := by
        rw [Rat.mul_assoc q t⁻¹ (q * t⁻¹), ← Rat.mul_assoc t⁻¹ q t⁻¹, Rat.mul_comm t⁻¹ q,
            Rat.mul_assoc q t⁻¹ t⁻¹, ← Rat.mul_assoc q q (t⁻¹ * t⁻¹)]
      rw [rearr, haa, Rat.mul_assoc 5 (t*t) (t⁻¹ * t⁻¹)]
      have inner : (t*t)*(t⁻¹ * t⁻¹) = 1 := by
        rw [Rat.mul_assoc t t (t⁻¹ * t⁻¹), ← Rat.mul_assoc t t⁻¹ t⁻¹, htt, Rat.one_mul, htt]
      rw [inner, Rat.mul_one]
    exact no_rat_sqrt 5 (not_sq_radicand 5 (Or.inr (Or.inl rfl)))
      ⟨q * t⁻¹, by rw [key]; native_decide⟩

/-- **The `ℚ(√5)` inverse.** For `(q,t) ≠ (0,0)`, the element `q + t√5` has explicit inverse
    `(q/N) + (0 − t/N)√5` with `N = q² − 5t²`: their product is `1`. The off-diagonal coefficient
    `q(0−t/N) + t(q/N)` is `0` (`rat_cross_inv`); the diagonal `q(q/N) + 5t(0−t/N)` is `N/N = 1`. -/
theorem Q5_inv (q t : Rat) (hqt : ¬ (q = 0 ∧ t = 0)) :
    mulR (addR (qR q) (mulR (qR t) (rootR 1)))
         (addR (qR (q / (q*q - 5*(t*t)))) (mulR (qR (0 - t / (q*q - 5*(t*t)))) (rootR 1)))
      = qR 1 := by
  have hN : q*q - 5*(t*t) ≠ 0 := N5_ne_zero q t hqt
  have hS : q * (0 - t / (q*q - 5*(t*t))) + t * (q / (q*q - 5*(t*t))) = 0 := by
    rw [Rat.div_def t (q*q - 5*(t*t)), Rat.div_def q (q*q - 5*(t*t))]
    exact rat_cross_inv q t (q*q - 5*(t*t))⁻¹
  have hC : q * (q / (q*q - 5*(t*t))) + 5 * (t * (0 - t / (q*q - 5*(t*t)))) = 1 := by
    rw [Rat.div_def q (q*q - 5*(t*t)), Rat.div_def t (q*q - 5*(t*t)),
        ← Rat.mul_assoc q q (q*q - 5*(t*t))⁻¹,
        rat_zero_sub (t * (q*q - 5*(t*t))⁻¹),
        Rat.mul_neg t (t * (q*q - 5*(t*t))⁻¹),
        ← Rat.mul_assoc t t (q*q - 5*(t*t))⁻¹,
        Rat.mul_neg 5 ((t*t) * (q*q - 5*(t*t))⁻¹),
        ← Rat.mul_assoc 5 (t*t) (q*q - 5*(t*t))⁻¹,
        ← Rat.neg_mul (5*(t*t)) (q*q - 5*(t*t))⁻¹,
        ← Rat.add_mul (q*q) (-(5*(t*t))) (q*q - 5*(t*t))⁻¹,
        ← Rat.sub_eq_add_neg (q*q) (5*(t*t))]
    exact Rat.mul_inv_cancel (q*q - 5*(t*t)) hN
  rw [E_mul5 q t (q / (q*q - 5*(t*t))) (0 - t / (q*q - 5*(t*t))), hC, hS, qR_zero,
      zeroR'_mul, addR_zero]

/-! ### 7.3. The 4-dimensional independence `indep4`. -/

/-- **ℚ-linear independence of `{1, √3, √5, √15}`.** If `p + q√3 + s√5 + t√15 = r` with all
    coordinates rational, then `p = r` and `q = s = t = 0`.

    Proof factors through `ℚ(√5)`. Regroup the LHS as `P + Q·√3` where `P = p + s√5` and
    `Q = q + t√5` are `ℚ(√5)`-elements, using `√t·√15 = (t√5)·√3` (`R15 = √3·√5`,
    `mulR_assoc`/`mulR_comm`) and `← rightDistribR`. If `Q ≠ 0` (as a `ℚ(√5)`-element), invert it
    by `Q5_inv`: from `Q·√3 = r + (−P)` we get `√3 = Q⁻¹·(r − P)`, a `ℚ(√5)`-element (by `E_mul5`),
    contradicting `not_R3_in_Q5`. Hence `q = t = 0`, and `indep_1_R5` finishes `p = r`, `s = 0`. -/
theorem indep4 (p q s t r : Rat)
    (h : addR (addR (addR (qR p) (mulR (qR q) (rootR 0))) (mulR (qR s) (rootR 1)))
              (mulR (qR t) R15) = qR r) :
    p = r ∧ q = 0 ∧ s = 0 ∧ t = 0 := by
  have hA4 : mulR (qR t) R15 = mulR (mulR (qR t) (rootR 1)) (rootR 0) := by
    rw [R15_def, mulR_comm (rootR 0) (rootR 1), ← mulR_assoc (qR t) (rootR 1) (rootR 0)]
  have hq0t0 : q = 0 ∧ t = 0 := by
    refine Classical.byContradiction (fun hqt => ?_)
    have hPQ : addR (addR (qR p) (mulR (qR s) (rootR 1)))
                    (mulR (addR (qR q) (mulR (qR t) (rootR 1))) (rootR 0)) = qR r := by
      rw [rightDistribR (qR q) (mulR (qR t) (rootR 1)) (rootR 0), ← hA4,
          ← addR_regroup4 (qR p) (mulR (qR q) (rootR 0)) (mulR (qR s) (rootR 1)) (mulR (qR t) R15)]
      exact h
    have hiso : mulR (addR (qR q) (mulR (qR t) (rootR 1))) (rootR 0)
        = addR (qR r) (negR (addR (qR p) (mulR (qR s) (rootR 1)))) := by
      have hstep := congrArg (addR (negR (addR (qR p) (mulR (qR s) (rootR 1))))) hPQ
      rw [← addR_assoc, addR_comm (negR (addR (qR p) (mulR (qR s) (rootR 1))))
            (addR (qR p) (mulR (qR s) (rootR 1))), addR_neg,
          addR_comm zeroR' (mulR (addR (qR q) (mulR (qR t) (rootR 1))) (rootR 0)), addR_zero,
          addR_comm (negR (addR (qR p) (mulR (qR s) (rootR 1)))) (qR r)] at hstep
      exact hstep
    have hQinv : mulR (addR (qR q) (mulR (qR t) (rootR 1)))
                      (addR (qR (q / (q*q - 5*(t*t))))
                            (mulR (qR (0 - t / (q*q - 5*(t*t)))) (rootR 1)))
        = qR 1 := Q5_inv q t hqt
    have hmul := congrArg
      (mulR (addR (qR (q / (q*q - 5*(t*t)))) (mulR (qR (0 - t / (q*q - 5*(t*t)))) (rootR 1))))
      hiso
    rw [← mulR_assoc (addR (qR (q / (q*q - 5*(t*t))))
            (mulR (qR (0 - t / (q*q - 5*(t*t)))) (rootR 1)))
          (addR (qR q) (mulR (qR t) (rootR 1))) (rootR 0),
        mulR_comm (addR (qR (q / (q*q - 5*(t*t))))
            (mulR (qR (0 - t / (q*q - 5*(t*t)))) (rootR 1)))
          (addR (qR q) (mulR (qR t) (rootR 1))),
        hQinv, qR_one, mulR_comm oneR (rootR 0), mulR_one] at hmul
    rw [negR_Q5 p s, ← addR_assoc (qR r) (qR (0 - p)) (mulR (qR (0 - s)) (rootR 1)),
        qR_add r (0 - p),
        E_mul5 (q / (q*q - 5*(t*t))) (0 - t / (q*q - 5*(t*t))) (r + (0 - p)) (0 - s)] at hmul
    exact not_R3_in_Q5 ⟨_, _, hmul⟩
  obtain ⟨hq0, ht0⟩ := hq0t0
  rw [hq0, ht0, qR_zero, zeroR'_mul (rootR 0), zeroR'_mul R15,
      addR_zero (qR p), addR_zero (addR (qR p) (mulR (qR s) (rootR 1)))] at h
  obtain ⟨hpr, hs0⟩ := indep_1_R5 p s r h
  exact ⟨hpr, hq0, hs0, ht0⟩

/-! ### 7.4. The final theorem: `√11 ∉ ℚ(√3,√5)`. -/

/-- **`√11 ∉ ℚ(√3,√5)` — the multiquadratic capstone.** No element
    `a + b√3 + c√5 + d√15` of `ℚ(√3,√5)` squares to `11`.

    Expand the square (`E_sq4`); 4-dimensional independence (`indep4`) forces the four basis
    coordinates to match `(11,0,0,0)`. Halving the `√3` and `√15` coordinates gives the two
    `ℚ(√5)` zero-divisor equations `ab+5cd = 0`, `ad+cb = 0`; `Qsqrt5_no_zero_div` splits into
    `(a=c=0)` or `(b=d=0)`. In each branch the constant equation plus the residual `√5`/`√15`
    coordinate force a rational square root of one of `{11,33,55,165}`, all impossible by the
    irrationality core. -/
theorem sqrt11_not_in_Q_sqrt3_sqrt5 :
    ¬ ∃ a b c d : Rat, mulR (alpha4 a b c d) (alpha4 a b c d) = qR 11 := by
  intro hex
  obtain ⟨a, b, c, d, h⟩ := hex
  rw [E_sq4] at h
  obtain ⟨hC1, hC3, hC5, hC15⟩ := indep4
    (a*a + 3*(b*b) + 5*(c*c) + 15*(d*d)) (2*a*b + 10*(c*d))
    (2*a*c + 6*(b*d)) (2*a*d + 2*(b*c)) 11 h
  have hab : a*b + 5*(c*d) = 0 := by
    apply rat_of_two_mul_zero
    rw [Rat.mul_add, ← Rat.mul_assoc 2 a b, ← Rat.mul_assoc 2 5 (c*d),
        show (2:Rat)*5 = 10 from by native_decide]
    exact hC3
  have hadbc : a*d + b*c = 0 := by
    apply rat_of_two_mul_zero
    rw [Rat.mul_add, ← Rat.mul_assoc 2 a d]
    exact hC15
  have had : a*d + c*b = 0 := by rw [Rat.mul_comm c b]; exact hadbc
  rcases Qsqrt5_no_zero_div a b c d hab had with ⟨ha0, hc0⟩ | ⟨hb0, hd0⟩
  · -- a = 0, c = 0
    rw [ha0, Rat.mul_zero, Rat.zero_mul, Rat.zero_add] at hC5
    have hbd : b * d = 0 := by
      rcases rat_mul_eq_zero hC5 with h6 | hbd'
      · exact absurd h6 (by native_decide : (6 : Rat) ≠ 0)
      · exact hbd'
    rw [ha0, hc0, Rat.mul_zero, Rat.mul_zero, Rat.zero_add, Rat.add_zero] at hC1
    -- hC1 : 3*(b*b) + 15*(d*d) = 11
    rcases rat_mul_eq_zero hbd with hb0 | hd0
    · -- b = 0 ⇒ 15*(d*d) = 11 ⇒ (15d)² = 165
      rw [hb0, Rat.mul_zero, Rat.mul_zero, Rat.zero_add] at hC1
      have h165 : (15 * d) * (15 * d) = (165 : Rat) := by rw [rat_fifteen d, hC1]; native_decide
      exact no_rat_sqrt 165
        (not_sq_radicand 165 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr rfl)))))))
        ⟨15 * d, by rw [h165]; native_decide⟩
    · -- d = 0 ⇒ 3*(b*b) = 11 ⇒ (3b)² = 33
      rw [hd0, Rat.mul_zero, Rat.mul_zero, Rat.add_zero] at hC1
      have h33 : (3 * b) * (3 * b) = (33 : Rat) := by rw [rat_three b, hC1]; native_decide
      exact no_rat_sqrt 33
        (not_sq_radicand 33 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))
        ⟨3 * b, by rw [h33]; native_decide⟩
  · -- b = 0, d = 0
    rw [hb0, Rat.zero_mul, Rat.mul_zero, Rat.add_zero] at hC5
    have hac : a * c = 0 := by
      apply rat_of_two_mul_zero
      rw [← Rat.mul_assoc 2 a c]
      exact hC5
    rw [hb0, hd0, Rat.mul_zero, Rat.mul_zero, Rat.mul_zero,
        Rat.add_zero, Rat.add_zero] at hC1
    -- hC1 : a*a + 5*(c*c) = 11
    rcases rat_mul_eq_zero hac with ha0 | hc0
    · -- a = 0 ⇒ 5*(c*c) = 11 ⇒ (5c)² = 55
      rw [ha0, Rat.mul_zero, Rat.zero_add] at hC1
      have h55 : (5 * c) * (5 * c) = (55 : Rat) := by rw [rat_five c, hC1]; native_decide
      exact no_rat_sqrt 55
        (not_sq_radicand 55 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))))
        ⟨5 * c, by rw [h55]; native_decide⟩
    · -- c = 0 ⇒ a*a = 11
      rw [hc0, Rat.mul_zero, Rat.mul_zero, Rat.add_zero] at hC1
      exact no_rat_sqrt 11
        (not_sq_radicand 11 (Or.inr (Or.inr (Or.inl rfl))))
        ⟨a, by rw [hC1]; native_decide⟩

#print axioms sqrt11_not_in_Q_sqrt3_sqrt5

end SounioSqrt.RealCauchyField
