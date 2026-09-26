import SounioCDCocycle

/-!
A universal bridge for actual two-term basis-pair products in the principal
Cayley-Dickson convention. Coefficients are functions Nat -> Int, so zero
means zero at every basis coordinate, not merely a pair of asserted tests.
The multiplication below is the bilinear expansion of
(ac - conjugate(d)b, da + b conjugate(c)) on four basis monomials.
The all-level basis sign is the repository's certified cdSigma.

This file does not yet prove the full graph-isomorphism classification.
-/
namespace Sounio.ZDAlgebraBridge
open SounioCDCocycle

def Sign (x : Int) : Prop := x = 1 ∨ x = -1

theorem neg_sign {x : Int} (h : Sign x) : Sign (-x) := by
  rcases h with rfl | rfl <;> simp [Sign]

theorem sigma_sign : ∀ n a b, Sign (cdSigma a b n) := by
  intro n
  induction n with
  | zero => intro a b; exact Or.inr rfl
  | succ n ih =>
    intro a b
    cases n with
    | zero =>
      simp only [cdSigma]
      split
      · exact Or.inl rfl
      · exact Or.inr rfl
    | succ n =>
      simp only [cdSigma]
      split
      · exact Or.inl rfl
      · split
        · exact ih _ _
        · split
          · exact ih _ _
          · split
            · split
              · exact ih _ _
              · exact neg_sign (ih _ _)
            · split
              · exact neg_sign (ih _ _)
              · exact ih _ _

theorem sigma_antisym (n a b : Nat) (hn : 1 ≤ n)
    (ha : a < 2^n) (hb : b < 2^n)
    (ha0 : a ≠ 0) (hb0 : b ≠ 0) (hab : a ≠ b) :
    cdSigma a b n = -cdSigma b a n := by
  rw [← sgn_eq_cdSigma n a b hn ha hb,
      ← sgn_eq_cdSigma n b a hn hb ha]
  apply antisym _ _ (by simp only [bitsOf_length])
    (isZ_bitsOf_false n a ha ha0) (isZ_bitsOf_false n b hb hb0)
  intro he
  have hz : isZ (xorL (bitsOf n a) (bitsOf n b)) = true :=
    (xorL_isZ_iff _ _ (by simp only [bitsOf_length])).mpr he
  rw [xorL_bitsOf n a b ha hb] at hz
  have hx := (isZ_bitsOf n (a ^^^ b) (Nat.xor_lt_two_pow ha hb)).mp hz
  have hh := congrArg (fun x : Nat => x ^^^ b) hx
  exact hab (by simpa [Nat.xor_assoc] using hh)

theorem sigma_diag (n a : Nat) (hn : 1 ≤ n)
    (ha : a < 2^n) (ha0 : a ≠ 0) : cdSigma a a n = -1 := by
  rw [← sgn_eq_cdSigma n a a hn ha ha]
  exact diag _ (isZ_bitsOf_false n a ha ha0)

def kappa (a : Nat) : Int := if a = 0 then 1 else -1
abbrev Coeff := Nat → Int
def mono (c : Int) (a : Nat) : Coeff := fun k => if k = a then c else 0
def zeroCoeff : Coeff := fun _ => 0

structure Term where
  coeff : Int
  index : Nat

def termMul (n : Nat) (a b : Term) : Term :=
  ⟨a.coeff * b.coeff * cdSigma a.index b.index n, a.index ^^^ b.index⟩
def termConj (a : Term) : Term := ⟨a.coeff * kappa a.index, a.index⟩
def termValue (a : Term) : Coeff := mono a.coeff a.index

structure BasisPair where
  lower : Term
  upper : Term

def pairProduct (n : Nat) (x y : BasisPair) : Coeff × Coeff :=
  (fun k => termValue (termMul n x.lower y.lower) k -
             termValue (termMul n (termConj y.upper) x.upper) k,
   fun k => termValue (termMul n y.upper x.lower) k +
             termValue (termMul n x.upper (termConj y.lower)) k)

def native (W a : Nat) (s : Int) : BasisPair :=
  ⟨⟨1,a⟩,⟨s,a ^^^ W⟩⟩

def nativeProduct (n W a b : Nat) (s t : Int) : Coeff × Coeff :=
  pairProduct n (native W a s) (native W b t)

def T (n W a b : Nat) : Int :=
  cdSigma a b n * cdSigma (a ^^^ W) (b ^^^ W) n
def Q (n W a b : Nat) : Int :=
  T n W a b * cdSigma a (b ^^^ W) n * cdSigma (a ^^^ W) b n

theorem mono_zero_iff (c : Int) (a : Nat) :
    mono c a = zeroCoeff ↔ c = 0 := by
  constructor
  · intro h
    have h' := congrFun h a
    simpa [mono, zeroCoeff] using h'
  · intro h; subst h
    funext k
    simp [mono, zeroCoeff]

theorem scalar_adjacency_criterion (A B C D s t : Int)
    (hA : Sign A) (hB : Sign B) (hC : Sign C)
    (hD : Sign D) (hs : Sign s) (ht : Sign t) :
    (A - s*t*B = 0 ∧ -t*C-s*D = 0) ↔
      (A*B*C*D = -1 ∧ s*t = A*B) := by
  rcases hA with rfl | rfl <;>
  rcases hB with rfl | rfl <;>
  rcases hC with rfl | rfl <;>
  rcases hD with rfl | rfl <;>
  rcases hs with rfl | rfl <;>
  rcases ht with rfl | rfl <;> decide

#print axioms sigma_sign
#print axioms sigma_antisym
#print axioms scalar_adjacency_criterion

theorem xor_zero_iff (a b : Nat) : a ^^^ b = 0 ↔ a = b := by
  constructor
  · intro h
    have hh := congrArg (fun x : Nat => x ^^^ b) h
    simpa [Nat.xor_assoc] using hh
  · intro h; subst b; simp

theorem xor_left_comm (a b c : Nat) :
    a ^^^ (b ^^^ c) = b ^^^ (a ^^^ c) := by
  rw [← Nat.xor_assoc, Nat.xor_comm a b, Nat.xor_assoc]

/-- Equality of coefficient functions obtained by bilinear basis multiplication. -/
theorem nativeProduct_normal_form (n W a b : Nat) (s t : Int)
    (hn : 1 ≤ n) (hW : W < 2^n)
    (ha : a < 2^n) (hb : b < 2^n)
    (ha0 : a ≠ 0) (hb0 : b ≠ 0)
    (haW : a ≠ W) (hbW : b ≠ W)
    (hab : a ≠ b) (habW : a ≠ b ^^^ W) :
    nativeProduct n W a b s t =
      (mono (cdSigma a b n - s*t*cdSigma (a ^^^ W) (b ^^^ W) n) (a ^^^ b),
       mono (-t*cdSigma a (b ^^^ W) n - s*cdSigma (a ^^^ W) b n)
         (a ^^^ b ^^^ W)) := by
  have hap : a ^^^ W < 2^n := Nat.xor_lt_two_pow ha hW
  have hbp : b ^^^ W < 2^n := Nat.xor_lt_two_pow hb hW
  have hap0 : a ^^^ W ≠ 0 := fun h => haW ((xor_zero_iff a W).mp h)
  have hbp0 : b ^^^ W ≠ 0 := fun h => hbW ((xor_zero_iff b W).mp h)
  have happ : a ^^^ W ≠ b ^^^ W := by
    intro h
    have hh := congrArg (fun x : Nat => x ^^^ W) h
    exact hab (by simpa [Nat.xor_assoc] using hh)
  have hBA := sigma_antisym n (b ^^^ W) (a ^^^ W) hn hbp hap hbp0 hap0
    (Ne.symm happ)
  have hCA := sigma_antisym n (b ^^^ W) a hn hbp ha hbp0 ha0 (Ne.symm habW)
  have hx1 : (b ^^^ W) ^^^ (a ^^^ W) = a ^^^ b := by
    simp only [Nat.xor_assoc]
    rw [xor_left_comm W a W]
    simp [Nat.xor_comm]
  have hx2 : (b ^^^ W) ^^^ a = a ^^^ b ^^^ W := by
    simp [Nat.xor_comm, xor_left_comm]
  have hx3 : (a ^^^ W) ^^^ b = a ^^^ b ^^^ W := by
    simp [Nat.xor_comm, xor_left_comm]
  apply Prod.ext
  · funext k
    simp only [nativeProduct, pairProduct, native, termMul, termConj,
      termValue, mono, kappa, hBA, hx1,
      Int.one_mul, Int.mul_one]
    by_cases hk : k = a ^^^ b <;>
      simp [hk, hbp0, Int.sub_eq_add_neg, Int.mul_comm, Int.mul_assoc, Int.mul_neg, Int.neg_mul]
  · funext k
    simp only [nativeProduct, pairProduct, native, termMul, termConj,
      termValue, mono, kappa, hCA, hx2, hx3,
      Int.one_mul, Int.mul_one]
    by_cases hk : k = a ^^^ b ^^^ W <;>
      simp [hk, hb0, Int.sub_eq_add_neg, Int.mul_neg, Int.neg_mul]

/-- Universal criterion for actual zero coefficient functions; n is the parent level. -/
theorem nativeProduct_zero_iff (n W a b : Nat) (s t : Int)
    (hn : 1 ≤ n) (hW : W < 2^n)
    (ha : a < 2^n) (hb : b < 2^n)
    (ha0 : a ≠ 0) (hb0 : b ≠ 0)
    (haW : a ≠ W) (hbW : b ≠ W)
    (hab : a ≠ b) (habW : a ≠ b ^^^ W)
    (hs : Sign s) (ht : Sign t) :
    nativeProduct n W a b s t = (zeroCoeff,zeroCoeff) ↔
      Q n W a b = -1 ∧ s*t = T n W a b := by
  rw [nativeProduct_normal_form n W a b s t hn hW ha hb ha0 hb0 haW hbW hab habW]
  rw [Prod.mk.injEq, mono_zero_iff, mono_zero_iff]
  exact scalar_adjacency_criterion _ _ _ _ _ _
    (sigma_sign _ _ _) (sigma_sign _ _ _) (sigma_sign _ _ _)
    (sigma_sign _ _ _) hs ht

#print axioms nativeProduct_normal_form
#print axioms nativeProduct_zero_iff

theorem xor_ne_reverse (a b W : Nat) (h : a ≠ b ^^^ W) :
    b ≠ a ^^^ W := by
  intro e
  apply h
  rw [e]
  simp [Nat.xor_assoc]

/-- Adjacency uses both ordered products, as in the native orthogonality graph. -/
theorem native_both_products_zero_iff (n W a b : Nat) (s t : Int)
    (hn : 1 ≤ n) (hW : W < 2^n)
    (ha : a < 2^n) (hb : b < 2^n)
    (ha0 : a ≠ 0) (hb0 : b ≠ 0)
    (haW : a ≠ W) (hbW : b ≠ W)
    (hab : a ≠ b) (habW : a ≠ b ^^^ W)
    (hs : Sign s) (ht : Sign t) :
    (nativeProduct n W a b s t = (zeroCoeff,zeroCoeff) ∧
     nativeProduct n W b a t s = (zeroCoeff,zeroCoeff)) ↔
      Q n W a b = -1 ∧ s*t = T n W a b := by
  have hap := Nat.xor_lt_two_pow ha hW
  have hbp := Nat.xor_lt_two_pow hb hW
  have hap0 : a ^^^ W ≠ 0 := fun h => haW ((xor_zero_iff a W).mp h)
  have hbp0 : b ^^^ W ≠ 0 := fun h => hbW ((xor_zero_iff b W).mp h)
  have happ : a ^^^ W ≠ b ^^^ W := by
    intro h
    have hh := congrArg (fun x : Nat => x ^^^ W) h
    exact hab (by simpa [Nat.xor_assoc] using hh)
  have hrev := xor_ne_reverse a b W habW
  have hBA := sigma_antisym n b a hn hb ha hb0 ha0 (Ne.symm hab)
  have hBB := sigma_antisym n (b ^^^ W) (a ^^^ W) hn hbp hap hbp0 hap0
    (Ne.symm happ)
  have hBC := sigma_antisym n b (a ^^^ W) hn hb hap hb0 hap0 hrev
  have hBD := sigma_antisym n (b ^^^ W) a hn hbp ha hbp0 ha0 (Ne.symm habW)
  have hT : T n W b a = T n W a b := by
    simp [T, hBA, hBB, Int.neg_mul, Int.mul_neg]
  have hQ : Q n W b a = Q n W a b := by
    simp [Q, hT, hBC, hBD, Int.mul_comm, Int.mul_left_comm, Int.neg_mul, Int.mul_neg]
  rw [nativeProduct_zero_iff n W a b s t hn hW ha hb ha0 hb0 haW hbW hab habW hs ht,
      nativeProduct_zero_iff n W b a t s hn hW hb ha hb0 ha0 hbW haW
        (Ne.symm hab) hrev ht hs, hQ, hT, Int.mul_comm t s]
  constructor
  · exact And.left
  · intro h; exact ⟨h,h⟩

/-- Opposite signs at a fixed lower index never annihilate. -/
theorem native_opposite_nonzero (n W a : Nat) (s : Int)
    (ha0 : a ≠ 0) (hs : Sign s) :
    nativeProduct n W a a s (-s) ≠ (zeroCoeff,zeroCoeff) := by
  intro hz
  have h := congrArg (fun p : Coeff × Coeff => p.2 ((a ^^^ W) ^^^ a)) hz
  simp only [nativeProduct, pairProduct, native, termMul, termConj,
    termValue, mono, zeroCoeff, kappa] at h
  rcases hs with rfl | rfl <;>
  rcases sigma_sign n (a ^^^ W) a with hh | hh <;>
    simp [hh, ha0] at h

/-- A squared native vector has a nonzero scalar coordinate. -/
theorem native_self_nonzero (n W a : Nat) (s : Int)
    (hn : 1 ≤ n) (hW : W < 2^n) (ha : a < 2^n)
    (ha0 : a ≠ 0) (haW : a ≠ W) (hs : Sign s) :
    nativeProduct n W a a s s ≠ (zeroCoeff,zeroCoeff) := by
  have hap := Nat.xor_lt_two_pow ha hW
  have hap0 : a ^^^ W ≠ 0 := fun h => haW ((xor_zero_iff a W).mp h)
  have hd := sigma_diag n a hn ha ha0
  have hdp := sigma_diag n (a ^^^ W) hn hap hap0
  intro hz
  have h := congrArg (fun p : Coeff × Coeff => p.1 0) hz
  simp only [nativeProduct, pairProduct, native, termMul, termConj,
    termValue, mono, zeroCoeff, kappa] at h
  rcases hs with rfl | rfl <;> simp [hap0, hd, hdp] at h

/-- Swapping the two indices within a nonzero W-coset never annihilates. -/
theorem native_swapped_nonzero (n W a : Nat) (s t : Int)
    (hn : 1 ≤ n) (hW : W < 2^n) (ha : a < 2^n)
    (ha0 : a ≠ 0) (haW : a ≠ W) (hs : Sign s) (ht : Sign t) :
    nativeProduct n W a (a ^^^ W) s t ≠ (zeroCoeff,zeroCoeff) := by
  have hap := Nat.xor_lt_two_pow ha hW
  have hap0 : a ^^^ W ≠ 0 := fun h => haW ((xor_zero_iff a W).mp h)
  have hd := sigma_diag n a hn ha ha0
  have hdp := sigma_diag n (a ^^^ W) hn hap hap0
  intro hz
  have hlo := congrArg (fun p : Coeff × Coeff => p.1 W) hz
  have hhi := congrArg (fun p : Coeff × Coeff => p.2 0) hz
  simp only [nativeProduct, pairProduct, native, termMul, termConj,
    termValue, mono, zeroCoeff, kappa] at hlo hhi
  have hx : a ^^^ (a ^^^ W) = W := by rw [← Nat.xor_assoc]; simp
  rcases hs with rfl | rfl <;> rcases ht with rfl | rfl <;>
  rcases sigma_sign n a (a ^^^ W) with hh | hh <;>
    simp [Nat.xor_assoc, hx, ha0, hap0, hd, hdp, hh] at hlo hhi

#print axioms native_both_products_zero_iff
#print axioms native_opposite_nonzero
#print axioms native_self_nonzero
#print axioms native_swapped_nonzero

/-- Same lower index, with any two projective signs. -/
theorem native_same_lower_nonzero (n W a : Nat) (s t : Int)
    (hn : 1 ≤ n) (hW : W < 2^n) (ha : a < 2^n)
    (ha0 : a ≠ 0) (haW : a ≠ W) (hs : Sign s) (ht : Sign t) :
    nativeProduct n W a a s t ≠ (zeroCoeff,zeroCoeff) := by
  rcases hs with rfl | rfl <;> rcases ht with rfl | rfl
  · exact native_self_nonzero n W a 1 hn hW ha ha0 haW (Or.inl rfl)
  · exact native_opposite_nonzero n W a 1 ha0 (Or.inl rfl)
  · exact native_opposite_nonzero n W a (-1) ha0 (Or.inr rfl)
  · exact native_self_nonzero n W a (-1) hn hW ha ha0 haW (Or.inr rfl)

/-- Complete criterion, including repeated and swapped indices.
The actual algebra has level n+1; n indexes its parent basis sign table. -/
theorem native_adjacency_iff (n W a b : Nat) (s t : Int)
    (hn : 1 ≤ n) (hW : W < 2^n)
    (ha : a < 2^n) (hb : b < 2^n)
    (ha0 : a ≠ 0) (hb0 : b ≠ 0)
    (haW : a ≠ W) (hbW : b ≠ W)
    (hs : Sign s) (ht : Sign t) :
    (nativeProduct n W a b s t = (zeroCoeff,zeroCoeff) ∧
     nativeProduct n W b a t s = (zeroCoeff,zeroCoeff)) ↔
      a ≠ b ∧ a ≠ b ^^^ W ∧ Q n W a b = -1 ∧ s*t = T n W a b := by
  constructor
  · intro h
    have hab : a ≠ b := by
      intro e
      subst b
      exact native_same_lower_nonzero n W a s t hn hW ha ha0 haW hs ht h.1
    have habW : a ≠ b ^^^ W := by
      intro e
      have eb : b = a ^^^ W := by rw [e]; simp [Nat.xor_assoc]
      rw [eb] at h
      exact native_swapped_nonzero n W a s t hn hW ha ha0 haW hs ht h.1
    exact ⟨hab,habW,
      (native_both_products_zero_iff n W a b s t hn hW ha hb ha0 hb0
        haW hbW hab habW hs ht).mp h⟩
  · rintro ⟨hab,habW,h⟩
    exact (native_both_products_zero_iff n W a b s t hn hW ha hb ha0 hb0
      haW hbW hab habW hs ht).mpr h

#print axioms native_same_lower_nonzero
#print axioms native_adjacency_iff

end Sounio.ZDAlgebraBridge
