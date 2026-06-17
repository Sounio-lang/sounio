import SounioDeGreyChi5Transfer3511

set_option maxHeartbeats 0

/-!
# Sounio — de Grey G529 over a three-root `{3,5,11}` target interface

`SounioDeGreyChi5Transfer3511` proved the current G529 obstruction for any target receiving the
checked `{3,5,11}` QF fragment. This file gives that target a named three-root interface:
`RootedField3511`.

The scope is deliberately exact. A `RootedField3511` supplies commutative-field-like operations,
roots for `3,5,11`, proves the 3-bit radical generator law, and carries a QF evaluator `phi` whose
homomorphism/unit laws hold on `qf3511Wf` inputs. The theorem below therefore no longer quantifies
over the repository's four-root `RootedField` interface. It is still not the final 8-mask evaluator
construction from roots alone inside this file: `SounioDeGreyChi5Eval3511` is the adjacent rung
that derives the `phi3511` laws from the 3-bit evaluator and packages that derived evaluator as a
`QF3511TransferWf` target.
-/

namespace DeGrey529.Rooted3511

open UnitDistanceChromatic

/-- Operation-parametric natural cast for the three-root interface. -/
def ofNatWith {F : Type} (add : F → F → F) (zero one : F) : Nat → F
  | 0 => zero
  | n + 1 => add (ofNatWith add zero one n) one

/-- A three-root target carrying exactly the evaluator laws needed by the checked current G529
`{3,5,11}` fragment.

The commutative-field-like laws below are exactly what the local three-root radical algebra needs
to prove the 3-bit generator law. The guarded `phi` laws remain explicit fields on this interface
for compatibility with the existing transfer theorem; the adjacent evaluator file derives an
independent `phi3511` instance from this algebra. -/
structure RootedField3511 where
  F : Type
  add : F → F → F
  mul : F → F → F
  neg : F → F
  zero : F
  one : F
  inv : F → F
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
  root3 : F
  root5 : F
  root11 : F
  root3_sq : mul root3 root3 = ofNatWith add zero one 3
  root5_sq : mul root5 root5 = ofNatWith add zero one 5
  root11_sq : mul root11 root11 = ofNatWith add zero one 11
  phi : DeGrey529.QF → F
  phi_qadd :
    ∀ a b, DeGrey529.Transfer3511.qf3511Wf a → DeGrey529.Transfer3511.qf3511Wf b →
      phi (DeGrey529.qadd a b) = add (phi a) (phi b)
  phi_qmul :
    ∀ a b, DeGrey529.Transfer3511.qf3511Wf a → DeGrey529.Transfer3511.qf3511Wf b →
      phi (DeGrey529.qmul a b) = mul (phi a) (phi b)
  phi_qsub :
    ∀ a b, DeGrey529.Transfer3511.qf3511Wf a → DeGrey529.Transfer3511.qf3511Wf b →
      phi (DeGrey529.qsub a b) = add (phi a) (neg (phi b))
  phi_unit :
    ∀ d, DeGrey529.Transfer3511.qf3511Wf d → DeGrey529.isOne d = true → phi d = one

namespace RootedField3511

variable (R : RootedField3511)

/-! ## Three-root algebra toolkit -/

def ofNat : Nat → R.F
  | 0 => R.zero
  | n + 1 => R.add (ofNat n) R.one

private theorem ofNat_succ (n : Nat) : ofNat R (n + 1) = R.add (ofNat R n) R.one := rfl

theorem ofNat_eq_ofNatWith (n : Nat) :
    ofNat R n = ofNatWith R.add R.zero R.one n := by
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [ofNat_succ]
      show R.add (ofNat R n) R.one = R.add (ofNatWith R.add R.zero R.one n) R.one
      rw [ih]

private theorem add_zero_left (a : R.F) : R.add R.zero a = a := by
  rw [R.add_comm, R.add_zero]

private theorem add_left_cancel (a b c : R.F) (h : R.add a b = R.add a c) : b = c := by
  have h2 : R.add (R.neg a) (R.add a b) = R.add (R.neg a) (R.add a c) := by rw [h]
  rwa [← R.add_assoc, ← R.add_assoc, R.add_comm (R.neg a) a, R.add_neg, add_zero_left,
    add_zero_left] at h2

private theorem add_right_cancel (a b c : R.F) (h : R.add b a = R.add c a) : b = c := by
  rw [R.add_comm b, R.add_comm c] at h
  exact add_left_cancel R a b c h

private theorem add4comm (a b c d : R.F) :
    R.add (R.add a b) (R.add c d) = R.add (R.add a c) (R.add b d) := by
  have hbc : R.add b (R.add c d) = R.add c (R.add b d) := by
    rw [← R.add_assoc, R.add_comm b c, R.add_assoc]
  rw [R.add_assoc, hbc, ← R.add_assoc]

private theorem mul_zero_right (a : R.F) : R.mul a R.zero = R.zero := by
  have hd : R.add (R.mul a R.zero) (R.mul a R.zero) = R.add (R.mul a R.zero) R.zero := by
    rw [← R.left_distrib, add_zero_left, R.add_zero]
  exact add_left_cancel R (R.mul a R.zero) (R.mul a R.zero) R.zero hd

private theorem mul_zero_left (a : R.F) : R.mul R.zero a = R.zero := by
  rw [R.mul_comm, mul_zero_right]

private theorem mul_one_left (a : R.F) : R.mul R.one a = a := by
  rw [R.mul_comm, R.mul_one]

private theorem neg_zero : R.neg R.zero = R.zero := by
  have h := R.add_neg R.zero
  rwa [add_zero_left] at h

private theorem neg_neg (a : R.F) : R.neg (R.neg a) = a := by
  have h0 : R.add (R.neg (R.neg a)) (R.neg a) = R.zero := by
    rw [R.add_comm, R.add_neg]
  have h1 : R.add a (R.neg a) = R.zero := R.add_neg a
  exact add_right_cancel R (R.neg a) (R.neg (R.neg a)) a (h0.trans h1.symm)

private theorem neg_add (a b : R.F) : R.neg (R.add a b) = R.add (R.neg a) (R.neg b) := by
  apply add_left_cancel R (R.add a b)
  rw [R.add_neg, add4comm, R.add_neg, R.add_neg, R.add_zero]

private theorem neg_mul (a b : R.F) : R.mul (R.neg a) b = R.neg (R.mul a b) := by
  apply add_left_cancel R (R.mul a b)
  rw [← R.right_distrib, R.add_neg, mul_zero_left, R.add_neg]

private theorem mul_neg (a b : R.F) : R.mul a (R.neg b) = R.neg (R.mul a b) := by
  rw [R.mul_comm a (R.neg b), neg_mul, R.mul_comm b a]

private theorem mul_eq_zero (a b : R.F) (h : R.mul a b = R.zero) :
    a = R.zero ∨ b = R.zero := by
  by_cases ha : a = R.zero
  · exact Or.inl ha
  · apply Or.inr
    calc
      b = R.mul R.one b := (mul_one_left R b).symm
      _ = R.mul (R.mul (R.inv a) a) b := by
            rw [R.mul_comm (R.inv a) a, R.mul_inv a ha]
      _ = R.mul (R.inv a) (R.mul a b) := R.mul_assoc _ _ _
      _ = R.mul (R.inv a) R.zero := by rw [h]
      _ = R.zero := mul_zero_right R _

private theorem mul4comm (a b c d : R.F) :
    R.mul (R.mul a b) (R.mul c d) = R.mul (R.mul a c) (R.mul b d) := by
  have hbc : R.mul b (R.mul c d) = R.mul c (R.mul b d) := by
    rw [← R.mul_assoc, R.mul_comm b c, R.mul_assoc]
  rw [R.mul_assoc, hbc, ← R.mul_assoc]

theorem ofNat_one : ofNat R 1 = R.one := add_zero_left R R.one

theorem ofNat_add (m n : Nat) : ofNat R (m + n) = R.add (ofNat R m) (ofNat R n) := by
  induction n with
  | zero => exact (R.add_zero (ofNat R m)).symm
  | succ n ih =>
    show R.add (ofNat R (m + n)) R.one = R.add (ofNat R m) (ofNat R (n + 1))
    rw [ih]
    exact R.add_assoc (ofNat R m) (ofNat R n) R.one

theorem ofNat_mul (m n : Nat) : ofNat R (m * n) = R.mul (ofNat R m) (ofNat R n) := by
  induction n with
  | zero =>
    show ofNat R (m * 0) = R.mul (ofNat R m) (ofNat R 0)
    rw [Nat.mul_zero]
    exact (mul_zero_right R (ofNat R m)).symm
  | succ n ih =>
    show ofNat R (m * (n + 1)) = R.mul (ofNat R m) (ofNat R (n + 1))
    rw [Nat.mul_succ, ofNat_add, ih,
        show ofNat R (n + 1) = R.add (ofNat R n) R.one from rfl,
        R.left_distrib, R.mul_one]

/-! ## Integer casts and inverse algebra -/

def ofInt : Int → R.F
  | Int.ofNat n => ofNat R n
  | Int.negSucc n => R.neg (ofNat R (n + 1))

theorem ofInt_ofNat (n : Nat) : ofInt R (Int.ofNat n) = ofNat R n := rfl

theorem ofInt_one : ofInt R 1 = R.one := ofNat_one R

theorem ofInt_neg (a : Int) : ofInt R (-a) = R.neg (ofInt R a) := by
  match a with
  | Int.ofNat 0 => exact (neg_zero R).symm
  | Int.ofNat (n + 1) =>
    rw [show (-(Int.ofNat (n + 1))) = Int.negSucc n from rfl]; rfl
  | Int.negSucc n =>
    rw [show (-(Int.negSucc n)) = Int.ofNat (n + 1) from rfl]
    exact (neg_neg R (ofNat R (n + 1))).symm

private theorem ofInt_add_one (a : Int) : ofInt R (a + 1) = R.add (ofInt R a) R.one := by
  match a with
  | Int.ofNat n =>
    rw [show (Int.ofNat n + 1) = Int.ofNat (n + 1) from rfl]; rfl
  | Int.negSucc 0 =>
    rw [show (Int.negSucc 0 + 1) = Int.ofNat 0 from by decide]
    show R.zero = R.add (R.neg (ofNat R 1)) R.one
    rw [ofNat_one, R.add_comm, R.add_neg]
  | Int.negSucc (n + 1) =>
    rw [show (Int.negSucc (n + 1) + 1) = Int.negSucc n from by omega]
    show R.neg (ofNat R (n + 1)) = R.add (R.neg (ofNat R (n + 2))) R.one
    rw [show ofNat R (n + 2) = R.add (ofNat R (n + 1)) R.one from rfl,
      neg_add, R.add_assoc, R.add_comm (R.neg R.one) R.one, R.add_neg, R.add_zero]

private theorem ofInt_sub_one (a : Int) :
    ofInt R (a - 1) = R.add (ofInt R a) (R.neg R.one) := by
  match a with
  | Int.ofNat 0 =>
    rw [show (Int.ofNat 0 - 1) = Int.negSucc 0 from by decide]
    show R.neg (ofNat R 1) = R.add (ofInt R (Int.ofNat 0)) (R.neg R.one)
    rw [ofNat_one, show ofInt R (Int.ofNat 0) = R.zero from rfl, add_zero_left]
  | Int.ofNat (n + 1) =>
    rw [show (Int.ofNat (n + 1) - 1) = Int.ofNat n from by
          rw [show Int.ofNat (n + 1) = Int.ofNat n + 1 from rfl]; omega]
    show ofNat R n = R.add (ofInt R (Int.ofNat (n + 1))) (R.neg R.one)
    rw [show ofInt R (Int.ofNat (n + 1)) = R.add (ofNat R n) R.one from rfl,
      R.add_assoc, R.add_neg, R.add_zero]
  | Int.negSucc n =>
    rw [show (Int.negSucc n - 1) = Int.negSucc (n + 1) from by omega]
    show R.neg (ofNat R (n + 2)) = R.add (R.neg (ofNat R (n + 1))) (R.neg R.one)
    rw [show ofNat R (n + 2) = R.add (ofNat R (n + 1)) R.one from rfl, neg_add]

private theorem ofInt_add_ofNat (a : Int) (n : Nat) :
    ofInt R (a + Int.ofNat n) = R.add (ofInt R a) (ofNat R n) := by
  induction n with
  | zero =>
    rw [show (a + Int.ofNat 0) = a from by rw [show Int.ofNat 0 = (0 : Int) from rfl]; omega]
    exact (R.add_zero (ofInt R a)).symm
  | succ n ih =>
    rw [show (a + Int.ofNat (n + 1)) = (a + Int.ofNat n) + 1 from by
          rw [show Int.ofNat (n + 1) = Int.ofNat n + 1 from rfl]; omega,
        ofInt_add_one, ih, R.add_assoc]; rfl

private theorem ofInt_add_negSucc (a : Int) (n : Nat) :
    ofInt R (a + Int.negSucc n) = R.add (ofInt R a) (R.neg (ofNat R (n + 1))) := by
  induction n with
  | zero =>
    rw [show (a + Int.negSucc 0) = (a - 1) from by omega, ofInt_sub_one, ofNat_one]
  | succ n ih =>
    rw [show (a + Int.negSucc (n + 1)) = (a + Int.negSucc n) - 1 from by omega,
      ofInt_sub_one, ih, R.add_assoc,
      show ofNat R (n + 2) = R.add (ofNat R (n + 1)) R.one from rfl, neg_add]

theorem ofInt_add (a b : Int) : ofInt R (a + b) = R.add (ofInt R a) (ofInt R b) := by
  match b with
  | Int.ofNat n => exact ofInt_add_ofNat R a n
  | Int.negSucc n => exact ofInt_add_negSucc R a n

private theorem ofInt_mul_ofNat (a : Int) (n : Nat) :
    ofInt R (a * Int.ofNat n) = R.mul (ofInt R a) (ofNat R n) := by
  induction n with
  | zero =>
    rw [show (a * Int.ofNat 0) = 0 from by
      rw [show Int.ofNat 0 = (0 : Int) from rfl, Int.mul_zero]]
    exact (mul_zero_right R (ofInt R a)).symm
  | succ n ih =>
    rw [show (a * Int.ofNat (n + 1)) = a * Int.ofNat n + a from by
          rw [show Int.ofNat (n + 1) = Int.ofNat n + 1 from rfl, Int.mul_add, Int.mul_one],
        ofInt_add, ih, show ofNat R (n + 1) = R.add (ofNat R n) R.one from rfl,
        R.left_distrib, R.mul_one]

theorem ofInt_mul (a b : Int) : ofInt R (a * b) = R.mul (ofInt R a) (ofInt R b) := by
  match b with
  | Int.ofNat n => exact ofInt_mul_ofNat R a n
  | Int.negSucc n =>
    rw [show (a * Int.negSucc n) = -(a * Int.ofNat (n + 1)) from by
          rw [show Int.negSucc n = -(Int.ofNat (n + 1)) from rfl, Int.mul_neg],
        ofInt_neg, ofInt_mul_ofNat]
    show R.neg (R.mul (ofInt R a) (ofNat R (n + 1))) =
      R.mul (ofInt R a) (R.neg (ofNat R (n + 1)))
    rw [mul_neg]

theorem ofInt_sub (a b : Int) :
    ofInt R (a - b) = R.add (ofInt R a) (R.neg (ofInt R b)) := by
  rw [show a - b = a + (-b) from by omega, ofInt_add, ofInt_neg]

theorem inv_mul_inv {a b : R.F} (ha : a ≠ R.zero) (hb : b ≠ R.zero) :
    R.inv (R.mul a b) = R.mul (R.inv a) (R.inv b) := by
  have hab : R.mul a b ≠ R.zero := by
    intro h
    rcases mul_eq_zero R a b h with ha' | hb'
    · exact ha ha'
    · exact hb hb'
  have key : R.mul (R.mul (R.inv a) (R.inv b)) (R.mul a b) = R.one := by
    rw [mul4comm, R.mul_comm (R.inv a) a, R.mul_inv a ha,
      R.mul_comm (R.inv b) b, R.mul_inv b hb, R.mul_one]
  calc
    R.inv (R.mul a b)
        = R.mul R.one (R.inv (R.mul a b)) := (mul_one_left R _).symm
    _ = R.mul (R.mul (R.mul (R.inv a) (R.inv b)) (R.mul a b))
          (R.inv (R.mul a b)) := by rw [key]
    _ = R.mul (R.mul (R.inv a) (R.inv b))
          (R.mul (R.mul a b) (R.inv (R.mul a b))) := by rw [R.mul_assoc]
    _ = R.mul (R.mul (R.inv a) (R.inv b)) R.one := by rw [R.mul_inv (R.mul a b) hab]
    _ = R.mul (R.inv a) (R.inv b) := R.mul_one _

def primeNat : Fin 3 → Nat
  | ⟨0, _⟩ => 3
  | ⟨1, _⟩ => 5
  | ⟨2, _⟩ => 11

def s : Fin 3 → R.F
  | ⟨0, _⟩ => R.root3
  | ⟨1, _⟩ => R.root5
  | ⟨2, _⟩ => R.root11

theorem s_sq (j : Fin 3) : R.mul (s R j) (s R j) = ofNat R (primeNat j) := by
  match j with
  | ⟨0, _⟩ =>
      show R.mul R.root3 R.root3 = ofNat R 3
      rw [R.root3_sq, ofNat_eq_ofNatWith]
  | ⟨1, _⟩ =>
      show R.mul R.root5 R.root5 = ofNat R 5
      rw [R.root5_sq, ofNat_eq_ofNatWith]
  | ⟨2, _⟩ =>
      show R.mul R.root11 R.root11 = ofNat R 11
      rw [R.root11_sq, ofNat_eq_ofNatWith]

def radicalBit (m : Nat) (k : Fin 3) : R.F :=
  if Nat.testBit m k.val then s R k else R.one

def r3511 (m : Nat) : R.F :=
  R.mul (radicalBit R m ⟨0, by decide⟩)
    (R.mul (radicalBit R m ⟨1, by decide⟩) (radicalBit R m ⟨2, by decide⟩))

theorem r3511_zero : r3511 R 0 = R.one := by
  simp [r3511, radicalBit, Nat.zero_testBit, R.mul_one]

def bcoeff3511 (m : Nat) : Nat :=
  (if Nat.testBit m 0 then 3 else 1)
  * (if Nat.testBit m 1 then 5 else 1)
  * (if Nat.testBit m 2 then 11 else 1)

def ofNatProd3511 (m : Nat) : R.F := ofNat R (bcoeff3511 m)

private theorem mul6 (a0 a1 a2 b0 b1 b2 : R.F) :
    R.mul (R.mul a0 (R.mul a1 a2)) (R.mul b0 (R.mul b1 b2))
      = R.mul (R.mul a0 b0) (R.mul (R.mul a1 b1) (R.mul a2 b2)) := by
  rw [mul4comm R a0 (R.mul a1 a2) b0 (R.mul b1 b2),
      mul4comm R a1 a2 b1 b2]

theorem radicalBit_mul (i j : Nat) (k : Fin 3) :
    R.mul (radicalBit R i k) (radicalBit R j k)
      = R.mul (ofNat R (if Nat.testBit i k.val && Nat.testBit j k.val then primeNat k else 1))
          (radicalBit R (Nat.xor i j) k) := by
  unfold radicalBit
  rw [show Nat.xor i j = i ^^^ j from rfl, Nat.testBit_xor]
  cases Nat.testBit i k.val <;> cases Nat.testBit j k.val <;>
    simp [ofNat_one, s_sq, R.mul_one, mul_one_left]

/-- Three-root generator law: the product of two 3-bit radical monomials is the selected rational
coefficient times the XOR radical monomial. This is the 8-mask analogue of the older four-root
`SounioSqrt.RootedField.generator_law`. -/
theorem generator_law3511 (i j : Nat) :
    R.mul (r3511 R i) (r3511 R j) =
      R.mul (ofNatProd3511 R (Nat.land i j)) (r3511 R (Nat.xor i j)) := by
  unfold r3511
  rw [mul6]
  simp only [radicalBit_mul]
  rw [← mul6]
  congr 1
  rw [← ofNat_mul, ← ofNat_mul]
  show ofNat R _ = ofNat R (bcoeff3511 (Nat.land i j))
  congr 1
  rw [show Nat.land i j = i &&& j from rfl]
  simp [bcoeff3511, Nat.testBit_and, primeNat, Nat.mul_assoc]

/-- The three-root interface as the generic `QF3511TransferWf` target. -/
def toQF3511TransferWf : DeGrey529.Transfer3511.QF3511TransferWf where
  F := R.F
  add := R.add
  mul := R.mul
  sub := fun a b => R.add a (R.neg b)
  phi := R.phi
  isUnitVal := fun x => x = R.one
  hadd := R.phi_qadd
  hmul := R.phi_qmul
  hsub := R.phi_qsub
  hunit := R.phi_unit
  hcoord_edge_endpoints3511 := by
    simpa [DeGrey529.Transfer3511.primeSupport] using
      DeGrey529.Param.current_g529_full_prime_subplane_contains_all_edge_endpoints
  hedge_terms3511 := by
    simpa [DeGrey529.Transfer3511.primeSupport] using
      DeGrey529.Param.current_g529_edge_distances_supported_in_3511
  hcurrent_lrat_support := by
    simpa [DeGrey529.Transfer3511.primeSupport] using
      DeGrey529.Param.current_g529_full_prime_subplane_carries_current_lrat_obstruction_support

/-- Unit relation induced by the three-root `{3,5,11}` evaluator. -/
def unit (p q : R.F × R.F) : Prop :=
  (R.toQF3511TransferWf).unit p q

/-- Image of the current exact G529 embedding under the three-root evaluator. -/
def embF (v : Nat) : R.F × R.F :=
  (R.toQF3511TransferWf).embF v

end RootedField3511

/-- **Three-root transfer theorem.** Any `RootedField3511` target satisfying the guarded evaluator
laws for the checked current `{3,5,11}` QF fragment carries the reflected G529 no-4-colouring
obstruction. -/
theorem rootedField3511_chi_ge_5_current_embedding (R : RootedField3511) :
    ¬ Nonempty (PlaneColouring (R.F × R.F) R.unit 4) :=
  DeGrey529.Transfer3511.qf3511Transfer_chi_ge_5_current_embedding R.toQF3511TransferWf

/-- Compatibility adapter from the older four-root `RootedField` interface. This witnesses that the
new three-root interface is at least as usable as the previous transfer surface, but the theorem
above itself quantifies over `RootedField3511`. -/
def ofRootedField (R : SounioSqrt.RootedField) : RootedField3511 where
  F := R.F
  add := R.add
  mul := R.mul
  neg := R.neg
  zero := R.zero
  one := R.one
  inv := R.inv
  add_assoc := R.add_assoc
  add_comm := R.add_comm
  mul_assoc := R.mul_assoc
  mul_comm := R.mul_comm
  left_distrib := R.left_distrib
  right_distrib := R.right_distrib
  add_zero := R.add_zero
  mul_one := R.mul_one
  add_neg := R.add_neg
  mul_inv := R.mul_inv
  zero_ne_one := R.zero_ne_one
  root3 := R.root ⟨0, by decide⟩
  root5 := R.root ⟨1, by decide⟩
  root11 := R.root ⟨3, by decide⟩
  root3_sq := by
    simpa [ofNatWith, SounioSqrt.primeNatLit] using R.root_sq ⟨0, by decide⟩
  root5_sq := by
    simpa [ofNatWith, SounioSqrt.primeNatLit] using R.root_sq ⟨1, by decide⟩
  root11_sq := by
    simpa [ofNatWith, SounioSqrt.primeNatLit] using R.root_sq ⟨3, by decide⟩
  phi := @SounioMultiquadHom.phi R
  phi_qadd := fun a b ha hb => SounioMultiquadHom.phi_qadd a b ha.1 hb.1
  phi_qmul := fun a b ha hb => SounioMultiquadHom.phi_qmul a b ha.1 hb.1
  phi_qsub := fun a b ha hb => SounioMultiquadHom.phi_qsub a b ha.1 hb.1
  phi_unit := fun d hd hone => by
    exact (DeGrey529.TransferWf.rootedTransfer R).hunit d hd.1 hone

/-- The old four-root theorem recovered through the new three-root interface. -/
theorem rootedField_via_3511_chi_ge_5_current_embedding (R : SounioSqrt.RootedField) :
    ¬ Nonempty (PlaneColouring
      ((ofRootedField R).F × (ofRootedField R).F) (ofRootedField R).unit 4) :=
  rootedField3511_chi_ge_5_current_embedding (ofRootedField R)

/-- Review-facing package for the three-root transfer theorem and its compatibility surface. -/
structure RootedField3511CurrentEmbeddingCertificate where
  three_bit_generator_law :
    ∀ R : RootedField3511, ∀ i j : Nat,
      R.mul (R.r3511 i) (R.r3511 j) =
        R.mul (R.ofNatProd3511 (Nat.land i j)) (R.r3511 (Nat.xor i j))
  three_root_transfer :
    ∀ R : RootedField3511,
      ¬ Nonempty (PlaneColouring (R.F × R.F) R.unit 4)
  compatibility_from_four_root :
    ∀ R : SounioSqrt.RootedField,
      ¬ Nonempty (PlaneColouring
        ((ofRootedField R).F × (ofRootedField R).F) (ofRootedField R).unit 4)
  endpoint_support :
    DeGrey529.Param.edgeEndpointsInPrimeSubplane
      DeGrey529.Transfer3511.primeSupport = true
  edge_distance_support :
    DeGrey529.edges.toList.all (fun e =>
      DeGrey529.Support.edgeDistanceTermsSupportedByPrimes e
        DeGrey529.Transfer3511.primeSupport) = true
  full_current_lrat_support :
    DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane
      DeGrey529.Transfer3511.primeSupport

/-- Single object exposing the three-root current-embedding transfer boundary. -/
def rootedField3511CurrentEmbeddingCertificate :
    RootedField3511CurrentEmbeddingCertificate where
  three_bit_generator_law := RootedField3511.generator_law3511
  three_root_transfer := rootedField3511_chi_ge_5_current_embedding
  compatibility_from_four_root := rootedField_via_3511_chi_ge_5_current_embedding
  endpoint_support := by
    simpa [DeGrey529.Transfer3511.primeSupport] using
      DeGrey529.Param.current_g529_full_prime_subplane_contains_all_edge_endpoints
  edge_distance_support := by
    simpa [DeGrey529.Transfer3511.primeSupport] using
      DeGrey529.Param.current_g529_edge_distances_supported_in_3511
  full_current_lrat_support := by
    simpa [DeGrey529.Transfer3511.primeSupport] using
      DeGrey529.Param.current_g529_full_prime_subplane_carries_current_lrat_obstruction_support

#check @RootedField3511
#check @RootedField3511.r3511
#check @RootedField3511.generator_law3511
#check @RootedField3511.toQF3511TransferWf
#check @rootedField3511_chi_ge_5_current_embedding
#check @ofRootedField
#check @rootedField_via_3511_chi_ge_5_current_embedding
#check @RootedField3511CurrentEmbeddingCertificate
#check @rootedField3511CurrentEmbeddingCertificate

#print axioms rootedField3511_chi_ge_5_current_embedding
#print axioms RootedField3511.generator_law3511
#print axioms rootedField_via_3511_chi_ge_5_current_embedding
#print axioms rootedField3511CurrentEmbeddingCertificate

#eval IO.println "SounioDeGreyChi5Rooted3511: current G529 obstruction transfers over a named three-root {3,5,11} compatibility interface; SounioDeGreyChi5Eval3511 derives the canonical phi3511/QF3511TransferWf path from the 8-mask evaluator."

end DeGrey529.Rooted3511
