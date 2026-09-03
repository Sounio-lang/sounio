set_option maxHeartbeats 0

/-!
# SounioSqrtField — abstract ordered field with square roots (Mathlib-free)

Target interface for embedding ℚ(√3,√5,√7,√11) into a real-like field (e.g. ℝ via
`Real.sqrt`). Supports the de Grey χ(ℝ²)≥5 programme: `SounioDeGreyChi5Transfer` proves
χ(F²)≥5 for any ring receiving QF via a homomorphism; this file supplies the *target*
algebraic interface a real field should satisfy.
-/

namespace SounioSqrt

/-- Standalone nat literal for the four primes (3,5,7,11), usable inside the structure axioms
    before the `ofNat` cast is available. -/
def primeNatLit : Fin 4 → Nat
  | ⟨0, _⟩ => 3
  | ⟨1, _⟩ => 5
  | ⟨2, _⟩ => 7
  | ⟨3, _⟩ => 11

/-- Operation-parametric ℕ→F cast, used to state `root_sq` self-containedly inside the structure
    (it is definitionally `RootedField.ofNat` once the field operations are fixed). -/
def rfOfNat {F : Type} (add : F → F → F) (zero one : F) : Nat → F
  | 0 => zero
  | n + 1 => add (rfOfNat add zero one n) one

/-- An abstract ordered field equipped with square roots of the four primes `3,5,7,11`.

    This is exactly the interface the de Grey χ≥5 transfer needs: it never requires a *total*
    square root, only the four generators `root j` with `root j ≥ 0` and `root j · root j = pⱼ`.
    A classic ordered field with a *total* `sqrt` yields a `RootedField` via
    `SqrtField.toRootedField`, so nothing is lost. -/
structure RootedField where
  F : Type
  add : F → F → F
  mul : F → F → F
  neg : F → F
  zero : F
  one : F
  inv : F → F
  le : F → F → Prop
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
  root : Fin 4 → F
  root_nonneg : ∀ j, le zero (root j)
  root_sq : ∀ j, mul (root j) (root j) = rfOfNat add zero one (primeNatLit j)

namespace RootedField

variable {R : RootedField}

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

/-! ## Natural-number cast and nonnegativity -/

def ofNat (n : Nat) : R.F :=
  match n with
  | 0 => R.zero
  | n + 1 => R.add (ofNat n) R.one

private theorem ofNat_succ (n : Nat) : ofNat (n + 1) = R.add (ofNat n) R.one := rfl

/-- `ofNat` is the operation-parametric cast `rfOfNat` specialised to `R`'s operations. Bridges the
    structure-level `root_sq` (stated via `rfOfNat`) to the `ofNat`-based prime values. -/
theorem ofNat_eq_rfOfNat (n : Nat) : (ofNat n : R.F) = rfOfNat R.add R.zero R.one n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [ofNat_succ]; show R.add (ofNat n) R.one = R.add (rfOfNat R.add R.zero R.one n) R.one; rw [ih]

theorem ofNat_nonneg (n : Nat) : R.le R.zero (ofNat n) := by
  induction n with
  | zero => exact R.le_refl _
  | succ n ih =>
    rw [ofNat_succ]
    exact sf_add_nonneg ih sf_le_zero_one

/-! ## The four primes and their square roots -/

def primeNat : Fin 4 → Nat := primeNatLit

def p (j : Fin 4) : R.F := ofNat (primeNat j)

/-- The square-root generator of the `j`-th prime, supplied by the `RootedField` structure. -/
def s (j : Fin 4) : R.F := R.root j

theorem s_sq (j : Fin 4) : R.mul (s j) (s j) = p j := by
  show R.mul (R.root j) (R.root j) = ofNat (primeNat j)
  rw [R.root_sq j, ofNat_eq_rfOfNat]
  rfl

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

/-! ## Cast homomorphism ℕ → F (reusable: needed by the generator law and later the QF hom) -/

theorem ofNat_one : (ofNat 1 : R.F) = R.one := sf_add_zero_left R.one

theorem ofNat_add (m n : Nat) : (ofNat (m + n) : R.F) = R.add (ofNat m) (ofNat n) := by
  induction n with
  | zero => exact (R.add_zero (ofNat m)).symm
  | succ n ih =>
    show R.add (ofNat (m + n)) R.one = R.add (ofNat m) (ofNat (n + 1))
    rw [ih]
    exact R.add_assoc (ofNat m) (ofNat n) R.one

theorem ofNat_mul (m n : Nat) : (ofNat (m * n) : R.F) = R.mul (ofNat m) (ofNat n) := by
  induction n with
  | zero =>
    show (ofNat (m * 0) : R.F) = R.mul (ofNat m) (ofNat 0)
    rw [Nat.mul_zero]
    exact (sf_mul_zero_right (ofNat m)).symm
  | succ n ih =>
    show (ofNat (m * (n + 1)) : R.F) = R.mul (ofNat m) (ofNat (n + 1))
    rw [Nat.mul_succ, ofNat_add, ih,
        show (ofNat (n + 1) : R.F) = R.add (ofNat n) R.one from rfl,
        R.left_distrib, R.mul_one]

/-! ## Eight-factor regroup (3× the existing `sf_mul_mul_mul_comm`) -/

/-- Interleave two nested 4-products: pairs up the matching factors. -/
private theorem mul8 (a0 a1 a2 a3 b0 b1 b2 b3 : R.F) :
    R.mul (R.mul a0 (R.mul a1 (R.mul a2 a3))) (R.mul b0 (R.mul b1 (R.mul b2 b3)))
      = R.mul (R.mul a0 b0) (R.mul (R.mul a1 b1) (R.mul (R.mul a2 b2) (R.mul a3 b3))) := by
  rw [sf_mul_mul_mul_comm a0 (R.mul a1 (R.mul a2 a3)) b0 (R.mul b1 (R.mul b2 b3)),
      sf_mul_mul_mul_comm a1 (R.mul a2 a3) b1 (R.mul b2 b3),
      sf_mul_mul_mul_comm a2 a3 b2 b3]

/-! ## Per-bit radical multiplication -/

/-- `√pₖ^[iₖ] · √pₖ^[jₖ] = (pₖ if both bits set else 1) · √pₖ^[(i⊕j)ₖ]` — the four-case core. -/
theorem radicalBit_mul (i j : Nat) (k : Fin 4) :
    R.mul (radicalBit i k) (radicalBit j k)
      = R.mul (ofNat (if Nat.testBit i k.val && Nat.testBit j k.val then primeNat k else 1))
              (radicalBit (Nat.xor i j) k) := by
  unfold radicalBit
  rw [show Nat.xor i j = i ^^^ j from rfl, Nat.testBit_xor]
  cases Nat.testBit i k.val <;> cases Nat.testBit j k.val <;>
    simp [ofNat_one, p, s_sq, R.mul_one, sf_mul_one_left]

/-! ## Generator law (multiquadratic √a·√b = √(ab) image) — PROVED -/

/-- The abstract image of `MultiquadRing.basis_mul_law` on radical generators.
    Discharging this for a concrete `ℝ` instance is exactly the multiplicative core of
    the eventual `QF ↪ ℝ` homomorphism. -/
def GeneratorLawObligation : Prop :=
  ∀ i j : Nat, i < 16 → j < 16 →
    R.mul (r i) (r j) = R.mul (ofNatProd (Nat.land i j)) (r (Nat.xor i j))

/-- **Generator law (PROVED).** `rᵢ · rⱼ = ofNatProd(i ∧ j) · r_{i⊕j}` for all `i,j < 16`,
    by finite four-bit radical factorisation — no Mathlib, no `Classical`, no new axioms. -/
theorem generator_law (i j : Nat) :
    R.mul (r i) (r j) = R.mul (ofNatProd (Nat.land i j)) (r (Nat.xor i j)) := by
  unfold r
  rw [mul8]
  simp only [radicalBit_mul]
  rw [← mul8]
  congr 1
  -- coefficient product = ofNatProd (i ∧ j)
  rw [← ofNat_mul, ← ofNat_mul, ← ofNat_mul]
  show (ofNat _ : R.F) = ofNat (bcoeff (Nat.land i j))
  congr 1
  rw [show Nat.land i j = i &&& j from rfl]
  simp [bcoeff, Nat.testBit_and, primeNat, primeNatLit, Nat.mul_assoc]

/-- `GeneratorLawObligation` is now a theorem (the `i,j < 16` hypotheses are unused: the law
    holds for all naturals, but we keep the bounded statement for the obligation interface). -/
theorem generatorLaw_solved : (GeneratorLawObligation (R := R)) :=
  fun i j _ _ => generator_law i j

/-! ## Characteristic zero: positive casts are invertible (denominator toolkit for QF↪ℝ)

An ordered field has characteristic zero, so every nonzero integer denominator maps to a
nonzero — hence invertible — element of `F`. These lemmas are the field-of-fractions toolkit
that the eventual `φ : QF → F`, `φ(c,d) = (Σ cᵢ rᵢ) · inv (ofInt d)`, needs to invert
denominators. (See the structural note in the roadmap: a total `QFTransfer` instance into a
field is impossible — `hadd` forces denominators, `hmul` forbids `den = 0` — so the ℝ
instance must be guarded by `den ≠ 0`, which is exactly what `ofInt_ne_zero` enables.) -/

private theorem sf_neg_zero : R.neg R.zero = R.zero := by
  have h := R.add_neg R.zero
  rwa [sf_add_zero_left] at h

/-- Characteristic zero: a successor natural cast never collapses to `0`. -/
theorem ofNat_ne_zero (n : Nat) : ofNat (n + 1) ≠ R.zero := by
  intro h
  have hcancel : R.add (ofNat n) R.one = R.add (ofNat n) (R.neg (ofNat n)) :=
    h.trans (R.add_neg (ofNat n)).symm
  have hone : R.one = R.neg (ofNat n) :=
    sf_add_left_cancel (ofNat n) R.one (R.neg (ofNat n)) hcancel
  have hneg_le : R.le (R.neg (ofNat n)) R.zero := by
    have hle :=
      R.add_le_add_right (a := R.zero) (b := ofNat n) (c := R.neg (ofNat n)) (ofNat_nonneg n)
    rwa [sf_add_zero_left, R.add_neg] at hle
  rw [← hone] at hneg_le
  exact R.zero_ne_one (R.le_antisymm sf_le_zero_one hneg_le)

theorem sf_inv_ne_zero {a : R.F} (ha : a ≠ R.zero) : R.inv a ≠ R.zero := by
  intro h
  have hk : R.mul a (R.inv a) = R.one := R.mul_inv a ha
  rw [h, sf_mul_zero_right] at hk
  exact R.zero_ne_one hk

theorem sf_inv_one : R.inv R.one = R.one := by
  have h := R.mul_inv R.one (fun he => R.zero_ne_one he.symm)
  rwa [sf_mul_one_left] at h

/-- Inversion is multiplicative on nonzero elements (needed to split `inv (ofInt (d₁·d₂))`). -/
theorem sf_inv_mul_inv {a b : R.F} (ha : a ≠ R.zero) (hb : b ≠ R.zero) :
    R.inv (R.mul a b) = R.mul (R.inv a) (R.inv b) := by
  have hab : R.mul a b ≠ R.zero := by
    intro h
    apply hb
    calc
      b = R.mul R.one b := (sf_mul_one_left b).symm
      _ = R.mul (R.mul (R.inv a) a) b := by rw [sf_mul_inv_left a ha]
      _ = R.mul (R.inv a) (R.mul a b) := R.mul_assoc _ _ _
      _ = R.mul (R.inv a) R.zero := by rw [h]
      _ = R.zero := sf_mul_zero_right _
  have key : R.mul (R.mul (R.inv a) (R.inv b)) (R.mul a b) = R.one := by
    rw [sf_mul_mul_mul_comm, sf_mul_inv_left a ha, sf_mul_inv_left b hb, R.mul_one]
  calc
    R.inv (R.mul a b)
        = R.mul R.one (R.inv (R.mul a b)) := (sf_mul_one_left _).symm
    _ = R.mul (R.mul (R.mul (R.inv a) (R.inv b)) (R.mul a b)) (R.inv (R.mul a b)) := by rw [key]
    _ = R.mul (R.mul (R.inv a) (R.inv b)) (R.mul (R.mul a b) (R.inv (R.mul a b))) := by
          rw [R.mul_assoc]
    _ = R.mul (R.mul (R.inv a) (R.inv b)) R.one := by rw [R.mul_inv (R.mul a b) hab]
    _ = R.mul (R.inv a) (R.inv b) := R.mul_one _

/-! ## Integer cast `ℤ → F` (for the signed coefficients/denominators of QF) -/

def ofInt : Int → R.F
  | Int.ofNat n => ofNat n
  | Int.negSucc n => R.neg (ofNat (n + 1))

theorem ofInt_ofNat (n : Nat) : (ofInt (Int.ofNat n) : R.F) = ofNat n := rfl

theorem ofInt_one : (ofInt 1 : R.F) = R.one := ofNat_one

/-- Characteristic zero, integer form: a nonzero integer never casts to `0` —
    the exact side-condition that makes a QF denominator invertible in `F`. -/
theorem ofInt_ne_zero {d : Int} (hd : d ≠ 0) : ofInt d ≠ R.zero := by
  match d with
  | Int.ofNat 0 => exact absurd rfl hd
  | Int.ofNat (n + 1) => exact ofNat_ne_zero n
  | Int.negSucc n =>
    intro h
    apply ofNat_ne_zero n
    have h' : R.neg (ofNat (n + 1)) = R.zero := h
    have h2 := congrArg R.neg h'
    rwa [sf_neg_neg, sf_neg_zero] at h2

/-! ## `ofInt` is a ring homomorphism `ℤ → F` -/

private theorem sf_neg_add (a b : R.F) : R.neg (R.add a b) = R.add (R.neg a) (R.neg b) := by
  rw [← sf_neg_one_mul (R.add a b), R.left_distrib, sf_neg_one_mul, sf_neg_one_mul]

private theorem sf_mul_neg (a b : R.F) : R.mul a (R.neg b) = R.neg (R.mul a b) := by
  rw [R.mul_comm a (R.neg b), sf_neg_mul, R.mul_comm b a]

theorem ofInt_neg (a : Int) : ofInt (-a) = R.neg (ofInt a) := by
  match a with
  | Int.ofNat 0 => exact sf_neg_zero.symm
  | Int.ofNat (n + 1) =>
    rw [show (-(Int.ofNat (n + 1))) = Int.negSucc n from rfl]; rfl
  | Int.negSucc n =>
    rw [show (-(Int.negSucc n)) = Int.ofNat (n + 1) from rfl]
    exact (sf_neg_neg (ofNat (n + 1))).symm

private theorem ofInt_add_one (a : Int) : ofInt (a + 1) = R.add (ofInt a) R.one := by
  match a with
  | Int.ofNat n =>
    rw [show (Int.ofNat n + 1) = Int.ofNat (n + 1) from rfl]; rfl
  | Int.negSucc 0 =>
    rw [show (Int.negSucc 0 + 1) = Int.ofNat 0 from by decide]
    show R.zero = R.add (R.neg (ofNat 1)) R.one
    rw [ofNat_one, R.add_comm, R.add_neg]
  | Int.negSucc (n + 1) =>
    rw [show (Int.negSucc (n + 1) + 1) = Int.negSucc n from by omega]
    show R.neg (ofNat (n + 1)) = R.add (R.neg (ofNat (n + 2))) R.one
    rw [show ofNat (n + 2) = R.add (ofNat (n + 1)) R.one from rfl, sf_neg_add, R.add_assoc,
        R.add_comm (R.neg R.one) R.one, R.add_neg, R.add_zero]

private theorem ofInt_sub_one (a : Int) : ofInt (a - 1) = R.add (ofInt a) (R.neg R.one) := by
  match a with
  | Int.ofNat 0 =>
    rw [show (Int.ofNat 0 - 1) = Int.negSucc 0 from by decide]
    show R.neg (ofNat 1) = R.add (ofInt (Int.ofNat 0)) (R.neg R.one)
    rw [ofNat_one, show ofInt (Int.ofNat 0) = R.zero from rfl, sf_add_zero_left]
  | Int.ofNat (n + 1) =>
    rw [show (Int.ofNat (n + 1) - 1) = Int.ofNat n from by
          rw [show Int.ofNat (n + 1) = Int.ofNat n + 1 from rfl]; omega]
    show ofNat n = R.add (ofInt (Int.ofNat (n + 1))) (R.neg R.one)
    rw [show ofInt (Int.ofNat (n + 1)) = R.add (ofNat n) R.one from rfl, R.add_assoc, R.add_neg,
        R.add_zero]
  | Int.negSucc n =>
    rw [show (Int.negSucc n - 1) = Int.negSucc (n + 1) from by omega]
    show R.neg (ofNat (n + 2)) = R.add (R.neg (ofNat (n + 1))) (R.neg R.one)
    rw [show ofNat (n + 2) = R.add (ofNat (n + 1)) R.one from rfl, sf_neg_add]

private theorem ofInt_add_ofNat (a : Int) (n : Nat) :
    ofInt (a + Int.ofNat n) = R.add (ofInt a) (ofNat n) := by
  induction n with
  | zero =>
    rw [show (a + Int.ofNat 0) = a from by rw [show Int.ofNat 0 = (0 : Int) from rfl]; omega]
    exact (R.add_zero (ofInt a)).symm
  | succ n ih =>
    rw [show (a + Int.ofNat (n + 1)) = (a + Int.ofNat n) + 1 from by
          rw [show Int.ofNat (n + 1) = Int.ofNat n + 1 from rfl]; omega,
        ofInt_add_one, ih, R.add_assoc]; rfl

private theorem ofInt_add_negSucc (a : Int) (n : Nat) :
    ofInt (a + Int.negSucc n) = R.add (ofInt a) (R.neg (ofNat (n + 1))) := by
  induction n with
  | zero =>
    rw [show (a + Int.negSucc 0) = (a - 1) from by omega, ofInt_sub_one, ofNat_one]
  | succ n ih =>
    rw [show (a + Int.negSucc (n + 1)) = (a + Int.negSucc n) - 1 from by omega, ofInt_sub_one, ih,
        R.add_assoc, show ofNat (n + 2) = R.add (ofNat (n + 1)) R.one from rfl, sf_neg_add]

theorem ofInt_add (a b : Int) : ofInt (a + b) = R.add (ofInt a) (ofInt b) := by
  match b with
  | Int.ofNat n => exact ofInt_add_ofNat a n
  | Int.negSucc n => exact ofInt_add_negSucc a n

private theorem ofInt_mul_ofNat (a : Int) (n : Nat) :
    ofInt (a * Int.ofNat n) = R.mul (ofInt a) (ofNat n) := by
  induction n with
  | zero =>
    rw [show (a * Int.ofNat 0) = 0 from by rw [show Int.ofNat 0 = (0 : Int) from rfl, Int.mul_zero]]
    exact (sf_mul_zero_right (ofInt a)).symm
  | succ n ih =>
    rw [show (a * Int.ofNat (n + 1)) = a * Int.ofNat n + a from by
          rw [show Int.ofNat (n + 1) = Int.ofNat n + 1 from rfl, Int.mul_add, Int.mul_one],
        ofInt_add, ih, show ofNat (n + 1) = R.add (ofNat n) R.one from rfl, R.left_distrib,
        R.mul_one]

theorem ofInt_mul (a b : Int) : ofInt (a * b) = R.mul (ofInt a) (ofInt b) := by
  match b with
  | Int.ofNat n => exact ofInt_mul_ofNat a n
  | Int.negSucc n =>
    rw [show (a * Int.negSucc n) = -(a * Int.ofNat (n + 1)) from by
          rw [show Int.negSucc n = -(Int.ofNat (n + 1)) from rfl, Int.mul_neg],
        ofInt_neg, ofInt_mul_ofNat]
    show R.neg (R.mul (ofInt a) (ofNat (n + 1))) = R.mul (ofInt a) (R.neg (ofNat (n + 1)))
    rw [sf_mul_neg]

end RootedField

/-! ## Classic interface: an ordered field with a *total* square root.

A `SqrtField` bundles a `RootedField` with a total `sqrt` (and its nonnegativity / square law).
Every `SqrtField` yields a `RootedField` by projection (`toRootedField`), so the χ(F²)≥5 result
proved for `RootedField` specialises to any total-`sqrt` ordered field. -/
structure SqrtField where
  toRootedField : RootedField
  sqrt : toRootedField.F → toRootedField.F
  sqrt_nonneg : ∀ a, toRootedField.le toRootedField.zero (sqrt a)
  sqrt_sq : ∀ a, toRootedField.le toRootedField.zero a →
    toRootedField.mul (sqrt a) (sqrt a) = a

end SounioSqrt

#print axioms SounioSqrt.RootedField.nonneg_sqrt_unique
#print axioms SounioSqrt.RootedField.ofNat_nonneg
#print axioms SounioSqrt.RootedField.s_sq
#print axioms SounioSqrt.RootedField.r_zero
#print axioms SounioSqrt.RootedField.ofNat_mul
#print axioms SounioSqrt.RootedField.radicalBit_mul
#print axioms SounioSqrt.RootedField.generator_law
#print axioms SounioSqrt.RootedField.generatorLaw_solved
#print axioms SounioSqrt.RootedField.ofNat_ne_zero
#print axioms SounioSqrt.RootedField.sf_inv_mul_inv
#print axioms SounioSqrt.RootedField.ofInt_ne_zero
#print axioms SounioSqrt.RootedField.ofInt_add
#print axioms SounioSqrt.RootedField.ofInt_mul

#eval IO.println "SounioSqrtField: RootedField structure (ordered field + four prime roots, NO total sqrt); PROVED nonneg_sqrt_unique, ofNat_nonneg, s_sq, r_zero, ofNat_mul, radicalBit_mul, generator_law; char-0 denominator toolkit ofNat_ne_zero/sf_inv_mul_inv/ofInt_ne_zero; ℤ→F ring hom ofInt_add/ofInt_mul/ofInt_neg. SqrtField (total sqrt) bundles a RootedField via toRootedField."
