set_option maxHeartbeats 0

/-!
# SounioSqrtField — abstract ordered field with square roots (Mathlib-free)

Target interface for embedding ℚ(√3,√5,√7,√11) into a real-like field (e.g. ℝ via
`Real.sqrt`). Supports the de Grey χ(ℝ²)≥5 programme: `SounioDeGreyChi5Transfer` proves
χ(F²)≥5 for any ring receiving QF via a homomorphism; this file supplies the *target*
algebraic interface a real field should satisfy.
-/

namespace SounioSqrt

/-- An abstract ordered field equipped with a non-negative square root. -/
structure SqrtField where
  F : Type
  add : F → F → F
  mul : F → F → F
  neg : F → F
  zero : F
  one : F
  inv : F → F
  le : F → F → Prop
  sqrt : F → F
  add_assoc : ∀ a b c, add (add a b) c = add a (add b c)
  add_comm : ∀ a b, add a b = add b a
  mul_assoc : ∀ a b c, mul (mul a b) c = mul a (mul b c)
  mul_comm : ∀ a b, mul a b = mul b a
  left_distrib : ∀ a b c, mul a (add b c) = add (mul a b) (mul a c)
  right_distrib : ∀ a b c, mul (add a b) c = add (mul a c) (mul b c)
  add_zero : ∀ a, add a zero = a
  mul_one : ∀ a, mul a one = a
  add_neg : ∀ a, add a (neg a) = zero
  mul_inv : ∀ a, a ≠ zero → mul a (inv a) = one
  zero_ne_one : zero ≠ one
  le_refl : ∀ a, le a a
  le_trans : ∀ {a b c}, le a b → le b c → le a c
  le_antisymm : ∀ {a b}, le a b → le b a → a = b
  le_total : ∀ a b, le a b ∨ le b a
  add_le_add_right : ∀ {a b c}, le a b → le (add a c) (add b c)
  mul_nonneg : ∀ {a b}, le zero a → le zero b → le zero (mul a b)
  sqrt_nonneg : ∀ a, le zero (sqrt a)
  sqrt_sq : ∀ a, le zero a → mul (sqrt a) (sqrt a) = a

namespace SqrtField

variable {R : SqrtField}

def sub (a b : R.F) : R.F := R.add a (R.neg b)

private theorem sf_add_zero_left (a : R.F) : R.add R.zero a = a := by
  rw [R.add_comm, R.add_zero]

private theorem sf_add_left_cancel (a b c : R.F) (h : R.add a b = R.add a c) : b = c := by
  have hb : R.add (R.neg a) (R.add a b) = b := by
    calc
      R.add (R.neg a) (R.add a b) = R.add (R.add (R.neg a) a) b := (R.add_assoc _ _ _).symm
      _ = R.add R.zero b := by rw [R.add_comm (R.neg a) a, R.add_neg, sf_add_zero_left]
      _ = b := sf_add_zero_left b
  have hc : R.add (R.neg a) (R.add a c) = c := by
    calc
      R.add (R.neg a) (R.add a c) = R.add (R.add (R.neg a) a) c := (R.add_assoc _ _ _).symm
      _ = R.add R.zero c := by rw [R.add_comm (R.neg a) a, R.add_neg, sf_add_zero_left]
      _ = c := sf_add_zero_left c
  rw [h] at hb
  exact hb.symm.trans hc

private theorem sf_add_right_cancel (a b c : R.F) (h : R.add b a = R.add c a) : b = c := by
  rw [R.add_comm b, R.add_comm c] at h
  exact sf_add_left_cancel a b c h

private theorem sf_mul_one_left (a : R.F) : R.mul R.one a = a := by
  rw [R.mul_comm, R.mul_one]

private theorem sf_mul_inv_left (a : R.F) (ha : a ≠ R.zero) : R.mul (R.inv a) a = R.one := by
  rw [R.mul_comm, R.mul_inv a ha]

private theorem sf_mul_zero_right (a : R.F) : R.mul a R.zero = R.zero := by
  let x := R.mul a R.zero
  have hdouble : x = R.add x x := by
    calc
      x = R.mul a R.zero := rfl
      _ = R.mul a (R.add R.zero R.zero) := by rw [sf_add_zero_left]
      _ = R.add (R.mul a R.zero) (R.mul a R.zero) := R.left_distrib _ _ _
      _ = R.add x x := rfl
  have hzero : R.zero = x := by
    calc
      R.zero = R.add x (R.neg x) := (R.add_neg x).symm
      _ = R.add (R.add x x) (R.neg x) := by rw [← hdouble]
      _ = R.add x (R.add x (R.neg x)) := by rw [R.add_assoc]
      _ = R.add x R.zero := by rw [R.add_neg]
      _ = x := R.add_zero x
  exact hzero.symm

private theorem sf_mul_zero_left (a : R.F) : R.mul R.zero a = R.zero := by
  rw [R.mul_comm, sf_mul_zero_right]

private theorem sf_mul_eq_zero (a b : R.F) (h : R.mul a b = R.zero) :
    a = R.zero ∨ b = R.zero := by
  by_cases ha : a = R.zero
  · exact Or.inl ha
  · apply Or.inr
    calc
      b = R.mul R.one b := (sf_mul_one_left b).symm
      _ = R.mul (R.mul (R.inv a) a) b := by rw [sf_mul_inv_left a ha]
      _ = R.mul (R.inv a) (R.mul a b) := by rw [R.mul_assoc]
      _ = R.mul (R.inv a) R.zero := by rw [h]
      _ = R.zero := sf_mul_zero_right _

private theorem sf_neg_mul (a b : R.F) : R.mul (R.neg a) b = R.neg (R.mul a b) := by
  have h0 : R.add (R.mul a b) (R.mul (R.neg a) b) = R.zero := by
    calc
      R.add (R.mul a b) (R.mul (R.neg a) b) =
          R.mul (R.add a (R.neg a)) b := by rw [← R.right_distrib]
      _ = R.mul R.zero b := by rw [R.add_neg]
      _ = R.zero := sf_mul_zero_left _
  have h1 : R.add (R.mul a b) (R.neg (R.mul a b)) = R.zero := R.add_neg _
  exact sf_add_left_cancel (R.mul a b) (R.mul (R.neg a) b) (R.neg (R.mul a b)) (h0.trans h1.symm)

private theorem sf_neg_neg (a : R.F) : R.neg (R.neg a) = a := by
  have h0 : R.add (R.neg (R.neg a)) (R.neg a) = R.zero := by
    rw [R.add_comm, R.add_neg]
  have h1 : R.add a (R.neg a) = R.zero := R.add_neg a
  exact sf_add_right_cancel (R.neg a) (R.neg (R.neg a)) a (h0.trans h1.symm)

private theorem sf_neg_one_mul (a : R.F) : R.mul (R.neg R.one) a = R.neg a := by
  rw [sf_neg_mul R.one a, sf_mul_one_left]

private theorem sf_neg_one_mul_neg_one : R.mul (R.neg R.one) (R.neg R.one) = R.one := by
  have h0 : R.add (R.mul (R.neg R.one) (R.neg R.one)) (R.neg R.one) = R.zero := by
    calc
      R.add (R.mul (R.neg R.one) (R.neg R.one)) (R.neg R.one) =
          R.add (R.mul (R.neg R.one) (R.neg R.one)) (R.mul (R.neg R.one) R.one) := by rw [R.mul_one]
      _ = R.mul (R.neg R.one) (R.add (R.neg R.one) R.one) := by rw [← R.left_distrib]
      _ = R.mul (R.neg R.one) R.zero := by
            rw [show R.add (R.neg R.one) R.one = R.zero from by rw [R.add_comm, R.add_neg]]
      _ = R.zero := sf_mul_zero_right _
  have h1 : R.add (R.neg R.one) R.one = R.zero := by rw [R.add_comm, R.add_neg]
  have hcancel :
      R.add (R.neg R.one) (R.mul (R.neg R.one) (R.neg R.one)) =
        R.add (R.neg R.one) R.one := by
    rw [R.add_comm, h0, h1]
  exact sf_add_left_cancel (R.neg R.one) (R.mul (R.neg R.one) (R.neg R.one)) R.one hcancel

private theorem sf_sub_self (a : R.F) : R.sub a a = R.zero := by
  unfold sub
  exact R.add_neg a

private theorem sf_mul_mul_mul_comm (a b c d : R.F) :
    R.mul (R.mul a b) (R.mul c d) = R.mul (R.mul a c) (R.mul b d) := by
  have hbc : R.mul b (R.mul c d) = R.mul c (R.mul b d) := by
    calc
      R.mul b (R.mul c d) = R.mul (R.mul b c) d := (R.mul_assoc _ _ _).symm
      _ = R.mul (R.mul c b) d := by rw [R.mul_comm b c]
      _ = R.mul c (R.mul b d) := R.mul_assoc _ _ _
  calc
    R.mul (R.mul a b) (R.mul c d) = R.mul a (R.mul b (R.mul c d)) := by rw [R.mul_assoc]
    _ = R.mul a (R.mul c (R.mul b d)) := by rw [hbc]
    _ = R.mul (R.mul a c) (R.mul b d) := by rw [← R.mul_assoc, ← R.mul_assoc]

private theorem sf_sub_mul_add (x y : R.F) :
    R.mul (R.sub x y) (R.add x y) = R.sub (R.mul x x) (R.mul y y) := by
  unfold sub
  have h1 :
      R.mul (R.neg y) (R.add x y) =
        R.add (R.mul (R.neg y) x) (R.mul (R.neg y) y) := R.left_distrib _ _ _
  have h2 : R.mul (R.neg y) y = R.neg (R.mul y y) := sf_neg_mul y y
  have h3 : R.mul (R.neg y) x = R.neg (R.mul y x) := sf_neg_mul y x
  calc
    R.mul (R.add x (R.neg y)) (R.add x y) =
        R.add (R.mul x (R.add x y)) (R.mul (R.neg y) (R.add x y)) := R.right_distrib _ _ _
    _ = R.add (R.add (R.mul x x) (R.mul x y)) (R.add (R.neg (R.mul y x)) (R.neg (R.mul y y))) := by
          rw [R.left_distrib, h1, h3, h2]
    _ = R.add (R.mul x x) (R.neg (R.mul y y)) := by
          have hinner :
              R.add (R.mul x y) (R.add (R.neg (R.mul y x)) (R.neg (R.mul y y))) =
                R.neg (R.mul y y) := by
            calc
              _ = R.add (R.add (R.mul x y) (R.neg (R.mul y x))) (R.neg (R.mul y y)) := by
                    rw [R.add_assoc]
              _ = R.add R.zero (R.neg (R.mul y y)) := by rw [R.mul_comm x y, R.add_neg]
              _ = R.neg (R.mul y y) := sf_add_zero_left _
          rw [R.add_assoc, hinner]

private theorem sf_add_nonneg {a b : R.F} (ha : R.le R.zero a) (hb : R.le R.zero b) :
    R.le R.zero (R.add a b) := by
  have h1 := R.add_le_add_right (a := R.zero) (b := a) (c := b) ha
  rw [sf_add_zero_left] at h1
  exact R.le_trans hb h1

private theorem sf_nonneg_add_eq_zero {x y : R.F}
    (hx : R.le R.zero x) (hy : R.le R.zero y) (h : R.add x y = R.zero) :
    x = R.zero ∧ y = R.zero := by
  have hy0 : R.le y R.zero := by
    have := R.add_le_add_right (a := R.zero) (b := x) (c := y) hx
    rwa [sf_add_zero_left, h] at this
  have hx0 : R.le x R.zero := by
    have := R.add_le_add_right (a := R.zero) (b := y) (c := x) hy
    rwa [sf_add_zero_left, R.add_comm, h] at this
  exact ⟨R.le_antisymm hx0 hx, R.le_antisymm hy0 hy⟩

private theorem sf_le_zero_one : R.le R.zero R.one := by
  rcases R.le_total R.zero R.one with h | h
  · exact h
  · exfalso
    have hneg : R.le R.zero (R.neg R.one) := by
      have := R.add_le_add_right (a := R.one) (b := R.zero) (c := R.neg R.one) h
      simpa [R.add_neg, sf_add_zero_left] using this
    have hsq : R.le R.zero (R.mul (R.neg R.one) (R.neg R.one)) := R.mul_nonneg hneg hneg
    rw [sf_neg_one_mul_neg_one] at hsq
    exact R.zero_ne_one (R.le_antisymm hsq h)

/-! ## Square-root uniqueness and multiplication law -/

/-- Non-negative square roots of equal squares agree. -/
theorem nonneg_sqrt_unique {x y : R.F}
    (hx : R.le R.zero x) (hy : R.le R.zero y)
    (h : R.mul x x = R.mul y y) : x = y := by
  have hdiff : R.mul (R.sub x y) (R.add x y) = R.zero := by
    rw [sf_sub_mul_add, h, sf_sub_self]
  rcases sf_mul_eq_zero (R.sub x y) (R.add x y) hdiff with hxy | hsum
  · unfold sub at hxy
    have : R.add x (R.neg y) = R.add y (R.neg y) := by
      rw [hxy, (R.add_neg y).symm]
    exact sf_add_right_cancel (R.neg y) x y this
  · rcases sf_nonneg_add_eq_zero hx hy hsum with ⟨hx0, hy0⟩
    rw [hx0, hy0]

/-- `√a · √b = √(a·b)` for non-negative `a`, `b`. -/
theorem mul_sqrt {a b : R.F} (ha : R.le R.zero a) (hb : R.le R.zero b) :
    R.mul (R.sqrt a) (R.sqrt b) = R.sqrt (R.mul a b) := by
  have hleft : R.le R.zero (R.mul (R.sqrt a) (R.sqrt b)) :=
    R.mul_nonneg (R.sqrt_nonneg a) (R.sqrt_nonneg b)
  have hright : R.le R.zero (R.sqrt (R.mul a b)) := R.sqrt_nonneg _
  have hsq :
      R.mul (R.mul (R.sqrt a) (R.sqrt b)) (R.mul (R.sqrt a) (R.sqrt b)) =
        R.mul (R.sqrt (R.mul a b)) (R.sqrt (R.mul a b)) := by
    calc
      R.mul (R.mul (R.sqrt a) (R.sqrt b)) (R.mul (R.sqrt a) (R.sqrt b)) =
          R.mul (R.mul (R.sqrt a) (R.sqrt a)) (R.mul (R.sqrt b) (R.sqrt b)) := by
            rw [sf_mul_mul_mul_comm (R.sqrt a) (R.sqrt b) (R.sqrt a) (R.sqrt b)]
      _ = R.mul a b := by rw [R.sqrt_sq a ha, R.sqrt_sq b hb]
      _ = R.mul (R.sqrt (R.mul a b)) (R.sqrt (R.mul a b)) :=
          (R.sqrt_sq (R.mul a b) (R.mul_nonneg ha hb)).symm
  exact nonneg_sqrt_unique hleft hright hsq

/-! ## Natural-number cast and nonnegativity -/

def ofNat (n : Nat) : R.F :=
  match n with
  | 0 => R.zero
  | n + 1 => R.add (ofNat n) R.one

private theorem ofNat_succ (n : Nat) : ofNat (n + 1) = R.add (ofNat n) R.one := rfl

theorem ofNat_nonneg (n : Nat) : R.le R.zero (ofNat n) := by
  induction n with
  | zero => exact R.le_refl _
  | succ n ih =>
    rw [ofNat_succ]
    exact sf_add_nonneg ih sf_le_zero_one

/-! ## The four primes and their square roots -/

def primeNat : Fin 4 → Nat
  | ⟨0, _⟩ => 3
  | ⟨1, _⟩ => 5
  | ⟨2, _⟩ => 7
  | ⟨3, _⟩ => 11

def p (j : Fin 4) : R.F := ofNat (primeNat j)

def s (j : Fin 4) : R.F := R.sqrt (p j)

theorem s_sq (j : Fin 4) : R.mul (s j) (s j) = p j :=
  R.sqrt_sq _ (ofNat_nonneg (primeNat j))

/-! ## Radical monomial `r m` over the four low bits -/

def radicalBit (m : Nat) (k : Fin 4) : R.F :=
  if Nat.testBit m k.val then s ⟨k.val, k.isLt⟩ else R.one

def r (m : Nat) : R.F :=
  R.mul (radicalBit m ⟨0, by decide⟩)
    (R.mul (radicalBit m ⟨1, by decide⟩)
      (R.mul (radicalBit m ⟨2, by decide⟩) (radicalBit m ⟨3, by decide⟩)))

theorem r_zero : r 0 = R.one := by
  simp [r, radicalBit, Nat.zero_testBit, R.mul_one]

/-- Integer coefficient `∏ primes selected by low 4 bits of `m` (matches `MultiquadRing.bcoeff`). -/
def bcoeff (m : Nat) : Nat :=
  (if Nat.testBit m 0 then 3 else 1)
  * (if Nat.testBit m 1 then 5 else 1)
  * (if Nat.testBit m 2 then 7 else 1)
  * (if Nat.testBit m 3 then 11 else 1)

def ofNatProd (m : Nat) : R.F := ofNat (bcoeff m)

/-! ## Generator law (multiquadratic √a·√b = √(ab) image) — STAGED -/

/-- The abstract image of `MultiquadRing.basis_mul_law` on radical generators.
    Discharging this for a concrete `ℝ` instance is exactly the multiplicative core of
    the eventual `QF ↪ ℝ` homomorphism. -/
def GeneratorLawObligation : Prop :=
  ∀ i j : Nat, i < 16 → j < 16 →
    R.mul (r i) (r j) = R.mul (ofNatProd (Nat.land i j)) (r (Nat.xor i j))

end SqrtField

end SounioSqrt

#print axioms SounioSqrt.SqrtField.nonneg_sqrt_unique
#print axioms SounioSqrt.SqrtField.mul_sqrt
#print axioms SounioSqrt.SqrtField.ofNat_nonneg
#print axioms SounioSqrt.SqrtField.s_sq
#print axioms SounioSqrt.SqrtField.r_zero

#eval IO.println "SounioSqrtField: SqrtField structure; PROVED nonneg_sqrt_unique, mul_sqrt, ofNat_nonneg, s_sq, r_zero; STAGED GeneratorLawObligation."
