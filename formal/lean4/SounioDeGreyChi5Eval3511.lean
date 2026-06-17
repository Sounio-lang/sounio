import SounioDeGreyChi5Rooted3511

set_option maxHeartbeats 0

/-!
# Sounio — derived `{3,5,11}` evaluator laws for `RootedField3511`

`SounioDeGreyChi5Rooted3511` exposes the named three-root target and proves the 3-bit generator
law, but its public transfer interface still carries the guarded `phi_*` laws as fields. This file
derives the four guarded laws from an explicit 8-mask evaluator.

The multiplicative law has one extra bridge: the de Grey coordinates are still 16-mask tuples, so
the `{3,5,11}` fragment is compressed through `radIdx3511` before using the 8-mask convolution.
-/

namespace DeGrey529.Rooted3511

open DeGrey529
open DeGrey529.Transfer3511

namespace RootedField3511

variable (R : RootedField3511)

/-! ## Local ring toolkit -/

private theorem add_zero_left (a : R.F) : R.add R.zero a = a := by
  rw [R.add_comm, R.add_zero]

private theorem add_left_cancel (a b c : R.F) (h : R.add a b = R.add a c) : b = c := by
  have h2 : R.add (R.neg a) (R.add a b) = R.add (R.neg a) (R.add a c) := by rw [h]
  rwa [← R.add_assoc, ← R.add_assoc, R.add_comm (R.neg a) a, R.add_neg, add_zero_left,
    add_zero_left] at h2

private theorem mul_zero_right (a : R.F) : R.mul a R.zero = R.zero := by
  have hd : R.add (R.mul a R.zero) (R.mul a R.zero) = R.add (R.mul a R.zero) R.zero := by
    rw [← R.left_distrib, add_zero_left, R.add_zero]
  exact add_left_cancel R (R.mul a R.zero) (R.mul a R.zero) R.zero hd

private theorem mul_zero_left (a : R.F) : R.mul R.zero a = R.zero := by
  rw [R.mul_comm, mul_zero_right]

private theorem neg_zero : R.neg R.zero = R.zero := by
  have h := R.add_neg R.zero
  rwa [add_zero_left] at h

private theorem neg_add (a b : R.F) : R.neg (R.add a b) = R.add (R.neg a) (R.neg b) := by
  apply add_left_cancel R (R.add a b)
  have hcomm : R.add (R.add a b) (R.add (R.neg a) (R.neg b)) =
      R.add (R.add a (R.neg a)) (R.add b (R.neg b)) := by
    have hbc : R.add b (R.add (R.neg a) (R.neg b)) =
        R.add (R.neg a) (R.add b (R.neg b)) := by
      rw [← R.add_assoc, R.add_comm b (R.neg a), R.add_assoc]
    rw [R.add_assoc, hbc, ← R.add_assoc]
  rw [R.add_neg, hcomm, R.add_neg, R.add_neg, R.add_zero]

private theorem neg_mul (a b : R.F) : R.mul (R.neg a) b = R.neg (R.mul a b) := by
  apply add_left_cancel R (R.mul a b)
  rw [← R.right_distrib, R.add_neg, mul_zero_left, R.add_neg]

private theorem mul_neg (a b : R.F) : R.mul a (R.neg b) = R.neg (R.mul a b) := by
  rw [R.mul_comm a (R.neg b), neg_mul, R.mul_comm b a]

private theorem mul4comm (a b c d : R.F) :
    R.mul (R.mul a b) (R.mul c d) = R.mul (R.mul a c) (R.mul b d) := by
  have hbc : R.mul b (R.mul c d) = R.mul c (R.mul b d) := by
    rw [← R.mul_assoc, R.mul_comm b c, R.mul_assoc]
  rw [R.mul_assoc, hbc, ← R.mul_assoc]

/-! ## Finite sums -/

def fsum {ι : Type} (l : List ι) (f : ι → R.F) : R.F :=
  l.foldr (fun i acc => R.add (f i) acc) R.zero

theorem fsum_cons {ι} (a : ι) (l : List ι) (f : ι → R.F) :
    fsum R (a :: l) f = R.add (f a) (fsum R l f) := rfl

private theorem add4comm (a b c d : R.F) :
    R.add (R.add a b) (R.add c d) = R.add (R.add a c) (R.add b d) := by
  have hbc : R.add b (R.add c d) = R.add c (R.add b d) := by
    rw [← R.add_assoc, R.add_comm b c, R.add_assoc]
  rw [R.add_assoc, hbc, ← R.add_assoc]

theorem fsum_add {ι} (l : List ι) (f g : ι → R.F) :
    fsum R l (fun i => R.add (f i) (g i)) = R.add (fsum R l f) (fsum R l g) := by
  induction l with
  | nil => exact (R.add_zero R.zero).symm
  | cons a l ih =>
    rw [fsum_cons, fsum_cons, fsum_cons, ih, add4comm]

theorem fsum_zero {ι} (l : List ι) : fsum R l (fun _ => R.zero) = R.zero := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    rw [fsum_cons, ih]
    exact R.add_zero R.zero

theorem fsum_congr {ι} (l : List ι) (f g : ι → R.F) (h : ∀ i ∈ l, f i = g i) :
    fsum R l f = fsum R l g := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    rw [fsum_cons, fsum_cons, h a (by simp),
      ih (fun i hi => h i (List.mem_cons_of_mem a hi))]

theorem fsum_neg {ι} (l : List ι) (f : ι → R.F) :
    fsum R l (fun i => R.neg (f i)) = R.neg (fsum R l f) := by
  induction l with
  | nil => exact (neg_zero R).symm
  | cons a l ih =>
    rw [fsum_cons, ih, fsum_cons, neg_add]

theorem mul_fsum_right {ι} (c : R.F) (l : List ι) (f : ι → R.F) :
    R.mul (fsum R l f) c = fsum R l (fun i => R.mul (f i) c) := by
  induction l with
  | nil =>
    show R.mul R.zero c = R.zero
    rw [mul_zero_left]
  | cons a l ih =>
    rw [fsum_cons, fsum_cons, R.right_distrib, ih]

theorem mul_fsum_left {ι} (c : R.F) (l : List ι) (f : ι → R.F) :
    R.mul c (fsum R l f) = fsum R l (fun i => R.mul c (f i)) := by
  induction l with
  | nil =>
    show R.mul c R.zero = R.zero
    rw [mul_zero_right]
  | cons a l ih =>
    rw [fsum_cons, fsum_cons, R.left_distrib, ih]

theorem fsum_mul_fsum {ι κ} (l : List ι) (m : List κ) (f : ι → R.F) (g : κ → R.F) :
    R.mul (fsum R l f) (fsum R m g) =
      fsum R l (fun i => fsum R m (fun j => R.mul (f i) (g j))) := by
  rw [mul_fsum_right]
  apply fsum_congr
  intro i _
  rw [mul_fsum_left]

theorem fsum_map {ι κ} (l : List ι) (φ : ι → κ) (f : κ → R.F) :
    fsum R (l.map φ) f = fsum R l (fun i => f (φ i)) := by
  induction l with
  | nil => rfl
  | cons a l ih => rw [List.map_cons, fsum_cons, fsum_cons, ih]

theorem fsum_perm {ι} {l l' : List ι} (f : ι → R.F) (h : l.Perm l') :
    fsum R l f = fsum R l' f := by
  induction h with
  | nil => rfl
  | cons a _ ih => rw [fsum_cons, fsum_cons, ih]
  | swap x y l =>
    rw [fsum_cons, fsum_cons, fsum_cons, fsum_cons, ← R.add_assoc, ← R.add_assoc,
      R.add_comm (f y) (f x)]
  | trans _ _ ih1 ih2 => rw [ih1, ih2]

theorem fsum_comm {ι κ} (l : List ι) (m : List κ) (h : ι → κ → R.F) :
    fsum R l (fun i => fsum R m (fun j => h i j)) =
      fsum R m (fun j => fsum R l (fun i => h i j)) := by
  induction l with
  | nil =>
    rw [show fsum R ([] : List ι) (fun i => fsum R m (fun j => h i j)) = R.zero from rfl,
      show (fun j => fsum R ([] : List ι) (fun i => h i j)) = (fun _ : κ => R.zero) from rfl,
      fsum_zero]
  | cons a l ih =>
    rw [fsum_cons]
    simp only [fsum_cons]
    rw [ih, ← fsum_add]

theorem perm_range_xor8 (idx : Nat) (hidx : idx < 8) :
    List.Perm ((List.range 8).map (fun i => i ^^^ idx)) (List.range 8) := by
  match idx with
  | 0 => native_decide
  | 1 => native_decide
  | 2 => native_decide
  | 3 => native_decide
  | 4 => native_decide
  | 5 => native_decide
  | 6 => native_decide
  | 7 => native_decide
  | idx + 8 => omega

theorem fsum_xor8 (k : Nat) (hk : k < 8) (G : Nat → R.F) :
    fsum R (List.range 8) G = fsum R (List.range 8) (fun i => G (i ^^^ k)) := by
  rw [show fsum R (List.range 8) (fun i => G (i ^^^ k)) =
        fsum R ((List.range 8).map (fun i => i ^^^ k)) G from (fsum_map R _ _ _).symm]
  exact fsum_perm R G (perm_range_xor8 k hk).symm

theorem ofInt_fsum {ι} (l : List ι) (g : ι → Int) :
    ofInt R (l.foldr (fun i acc => g i + acc) 0) = fsum R l (fun i => ofInt R (g i)) := by
  induction l with
  | nil => rfl
  | cons a l ih => rw [List.foldr_cons, ofInt_add, fsum_cons, ih]

theorem foldl_add_int {ι} (l : List ι) (g : ι → Int) (a : Int) :
    l.foldl (fun acc i => acc + g i) a = a + l.foldr (fun i acc => g i + acc) 0 := by
  induction l generalizing a with
  | nil => simp
  | cons x l ih =>
    simp only [List.foldl_cons, List.foldr_cons]
    rw [ih (a + g x)]
    omega

private theorem range8_eq :
    List.range 8 = [0, 1, 2, 3, 4, 5, 6, 7] := by native_decide

private theorem range16_eq :
    List.range 16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] := by
  native_decide

/-! ## The 8-mask `{3,5,11}` evaluator and 16-mask support bridge -/

def radIdx3511 (k : Nat) : Nat :=
  if k < 4 then k else if k < 8 then k + 4 else 0

theorem radIdx3511_lt16 (k : Nat) : radIdx3511 k < 16 := by
  unfold radIdx3511
  by_cases h4 : k < 4
  · simp [h4]
    omega
  · simp [h4]
    by_cases h8 : k < 8
    · simp [h8]
      omega
    · simp [h8]

theorem xor_lt8 {i j : Nat} (hi : i < 8) (hj : j < 8) : Nat.xor i j < 8 := by
  match i with
  | 0 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 1 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 2 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 3 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 4 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 5 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 6 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 7 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | i + 8 => omega

theorem radIdx3511_xor {i j : Nat} (hi : i < 8) (hj : j < 8) :
    radIdx3511 (Nat.xor i j) = Nat.xor (radIdx3511 i) (radIdx3511 j) := by
  match i with
  | 0 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 1 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 2 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 3 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 4 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 5 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 6 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 7 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | i + 8 => omega

theorem radIdx3511_land {i j : Nat} (hi : i < 8) (hj : j < 8) :
    radIdx3511 (Nat.land i j) = Nat.land (radIdx3511 i) (radIdx3511 j) := by
  match i with
  | 0 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 1 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 2 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 3 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 4 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 5 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 6 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | 7 =>
    match j with
    | 0 => native_decide | 1 => native_decide | 2 => native_decide | 3 => native_decide
    | 4 => native_decide | 5 => native_decide | 6 => native_decide | 7 => native_decide
    | j + 8 => omega
  | i + 8 => omega

theorem qf3511Wf_coeff_zero_of_unsupported (x : DeGrey529.QF) (hx : qf3511Wf x)
    {i : Nat} (hi : i < 16)
    (hunsup : DeGrey529.Support.idxSupportedByPrimes i primeSupport = false) :
    DeGrey529.gi x.1 i = 0 := by
  have hall :
      ∀ i, i ∈ List.range 16 →
        (DeGrey529.Support.coeffAt x i == 0 ||
          DeGrey529.Support.idxSupportedByPrimes i primeSupport) = true :=
    List.all_eq_true.mp (by
      simpa [DeGrey529.Support.qfSupportedByPrimes] using hx.2)
  have hb := hall i (List.mem_range.mpr hi)
  rw [hunsup, Bool.or_false] at hb
  simpa [DeGrey529.Support.coeffAt] using hb

theorem qf3511Wf_sqrt7_coeffs_zero (x : DeGrey529.QF) (hx : qf3511Wf x) :
    DeGrey529.gi x.1 4 = 0 ∧ DeGrey529.gi x.1 5 = 0 ∧ DeGrey529.gi x.1 6 = 0 ∧
    DeGrey529.gi x.1 7 = 0 ∧ DeGrey529.gi x.1 12 = 0 ∧ DeGrey529.gi x.1 13 = 0 ∧
    DeGrey529.gi x.1 14 = 0 ∧ DeGrey529.gi x.1 15 = 0 := by
  exact ⟨
    qf3511Wf_coeff_zero_of_unsupported x hx (by decide) (by native_decide),
    qf3511Wf_coeff_zero_of_unsupported x hx (by decide) (by native_decide),
    qf3511Wf_coeff_zero_of_unsupported x hx (by decide) (by native_decide),
    qf3511Wf_coeff_zero_of_unsupported x hx (by decide) (by native_decide),
    qf3511Wf_coeff_zero_of_unsupported x hx (by decide) (by native_decide),
    qf3511Wf_coeff_zero_of_unsupported x hx (by decide) (by native_decide),
    qf3511Wf_coeff_zero_of_unsupported x hx (by decide) (by native_decide),
    qf3511Wf_coeff_zero_of_unsupported x hx (by decide) (by native_decide)⟩

def evalNum8 (l : List Int) : R.F :=
  fsum R (List.range 8) (fun k => R.mul (ofInt R (DeGrey529.gi l k)) (r3511 R k))

def qmulTerm3511 (x y : List Int) (i idx : Nat) : Int :=
  DeGrey529.gi x i * DeGrey529.gi y (Nat.xor i idx) *
    Int.ofNat (bcoeff3511 (Nat.land i (Nat.xor i idx)))

def qmulCoeff3511 (x y : List Int) (idx : Nat) : Int :=
  (List.range 8).foldl (fun acc i => acc + qmulTerm3511 x y i idx) 0

def qmulNum3511 (x y : List Int) : List Int :=
  (List.range 8).map (fun idx => qmulCoeff3511 x y idx)

def compressNum3511 (l : List Int) : List Int :=
  (List.range 8).map (fun k => DeGrey529.gi l (radIdx3511 k))

private theorem xor_cancel2 (i idx : Nat) : Nat.xor i (i ^^^ idx) = idx := by
  show i ^^^ (i ^^^ idx) = idx
  rw [← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]

private theorem W_eq3511 (x y : List Int) (i idx : Nat) :
    R.mul (R.mul (ofInt R (DeGrey529.gi x i)) (ofInt R (DeGrey529.gi y (i ^^^ idx))))
          (ofNatProd3511 R (Nat.land i (i ^^^ idx)))
      = ofInt R (qmulTerm3511 x y i idx) := by
  unfold qmulTerm3511 ofNatProd3511
  rw [← ofInt_mul]
  show R.mul (ofInt R (DeGrey529.gi x i * DeGrey529.gi y (i ^^^ idx)))
      (ofInt R (Int.ofNat (bcoeff3511 (Nat.land i (i ^^^ idx))))) =
    ofInt R (DeGrey529.gi x i * DeGrey529.gi y (i ^^^ idx) *
      Int.ofNat (bcoeff3511 (Nat.land i (i ^^^ idx))))
  rw [← ofInt_mul]

private theorem ofInt_qmulCoeff3511 (x y : List Int) (idx : Nat) :
    ofInt R (qmulCoeff3511 x y idx) =
      fsum R (List.range 8) (fun i => ofInt R (qmulTerm3511 x y i idx)) := by
  unfold qmulCoeff3511
  rw [foldl_add_int, Int.zero_add, ofInt_fsum]

private theorem qmulNum3511_gi (x y : List Int) (idx : Nat) (hidx : idx < 8) :
    DeGrey529.gi (qmulNum3511 x y) idx = qmulCoeff3511 x y idx := by
  unfold DeGrey529.gi qmulNum3511
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by simp [hidx]), Option.getD_some]
  simp [List.getElem_map, List.getElem_range]

private theorem compressNum3511_gi (l : List Int) (idx : Nat) (hidx : idx < 8) :
    DeGrey529.gi (compressNum3511 l) idx = DeGrey529.gi l (radIdx3511 idx) := by
  unfold DeGrey529.gi compressNum3511
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by simp [hidx]), Option.getD_some]
  simp [DeGrey529.gi, List.getElem_map, List.getElem_range]

theorem evalNum8_qmulNum3511 (x y : List Int) :
    evalNum8 R (qmulNum3511 x y) = R.mul (evalNum8 R x) (evalNum8 R y) := by
  symm
  unfold evalNum8
  rw [fsum_mul_fsum]
  calc
    fsum R (List.range 8) (fun i => fsum R (List.range 8)
        (fun j => R.mul (R.mul (ofInt R (DeGrey529.gi x i)) (r3511 R i))
          (R.mul (ofInt R (DeGrey529.gi y j)) (r3511 R j))))
      = fsum R (List.range 8) (fun i => fsum R (List.range 8)
          (fun j => R.mul
            (R.mul (R.mul (ofInt R (DeGrey529.gi x i)) (ofInt R (DeGrey529.gi y j)))
              (ofNatProd3511 R (Nat.land i j))) (r3511 R (Nat.xor i j)))) := by
        apply fsum_congr
        intro i _
        apply fsum_congr
        intro j _
        rw [mul4comm, generator_law3511, ← R.mul_assoc]
    _ = fsum R (List.range 8) (fun i => fsum R (List.range 8)
          (fun idx => R.mul
            (R.mul (R.mul (ofInt R (DeGrey529.gi x i)) (ofInt R (DeGrey529.gi y (i ^^^ idx))))
              (ofNatProd3511 R (Nat.land i (i ^^^ idx)))) (r3511 R idx))) := by
        apply fsum_congr
        intro i hi
        rw [fsum_xor8 R i (List.mem_range.mp hi)]
        apply fsum_congr
        intro idx _
        rw [Nat.xor_comm idx i, xor_cancel2]
    _ = fsum R (List.range 8) (fun idx => fsum R (List.range 8)
          (fun i => R.mul
            (R.mul (R.mul (ofInt R (DeGrey529.gi x i)) (ofInt R (DeGrey529.gi y (i ^^^ idx))))
              (ofNatProd3511 R (Nat.land i (i ^^^ idx)))) (r3511 R idx))) :=
        fsum_comm R (List.range 8) (List.range 8) _
    _ = fsum R (List.range 8) (fun idx =>
          R.mul (fsum R (List.range 8)
            (fun i => R.mul
              (R.mul (ofInt R (DeGrey529.gi x i)) (ofInt R (DeGrey529.gi y (i ^^^ idx))))
              (ofNatProd3511 R (Nat.land i (i ^^^ idx))))) (r3511 R idx)) := by
        apply fsum_congr
        intro idx _
        rw [mul_fsum_right]
    _ = fsum R (List.range 8) (fun idx =>
          R.mul (fsum R (List.range 8) (fun i => ofInt R (qmulTerm3511 x y i idx)))
            (r3511 R idx)) := by
        apply fsum_congr
        intro idx _
        congr 1
        apply fsum_congr
        intro i _
        exact W_eq3511 R x y i idx
    _ = fsum R (List.range 8) (fun idx =>
          R.mul (ofInt R (qmulCoeff3511 x y idx)) (r3511 R idx)) := by
        apply fsum_congr
        intro idx _
        rw [ofInt_qmulCoeff3511]
    _ = fsum R (List.range 8) (fun idx =>
          R.mul (ofInt R (DeGrey529.gi (qmulNum3511 x y) idx)) (r3511 R idx)) := by
        apply fsum_congr
        intro idx hidx
        rw [qmulNum3511_gi x y idx (List.mem_range.mp hidx)]

def evalNum3511 (l : List Int) : R.F :=
  fsum R (List.range 8) (fun k => R.mul (ofInt R (DeGrey529.gi l (radIdx3511 k))) (r3511 R k))

theorem evalNum3511_eq_evalNum8_compress (l : List Int) :
    evalNum3511 R l = evalNum8 R (compressNum3511 l) := by
  unfold evalNum3511 evalNum8
  apply fsum_congr
  intro k hk
  rw [compressNum3511_gi l k (List.mem_range.mp hk)]

def phi3511 (x : DeGrey529.QF) : R.F :=
  R.mul (evalNum3511 R x.1) (R.inv (ofInt R x.2))

/-- Characteristic-zero denominator guard for the derived fraction evaluator. The older ordered
four-root interface derives this from order; the bare three-root interface keeps it explicit. -/
def IntCastNonzero : Prop :=
  ∀ d : Int, d ≠ 0 → ofInt R d ≠ R.zero

private theorem gi_qadd (x y : DeGrey529.QF) (i : Nat) (hi : i < 16) :
    DeGrey529.gi (DeGrey529.qadd x y).1 i =
      DeGrey529.gi x.1 i * y.2 + DeGrey529.gi y.1 i * x.2 := by
  unfold DeGrey529.gi DeGrey529.qadd
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by simp [List.length_map, hi])]
  simp [DeGrey529.gi, List.getElem_map, List.getElem_range]

private theorem gi_qsub (x y : DeGrey529.QF) (i : Nat) (hi : i < 16) :
    DeGrey529.gi (DeGrey529.qsub x y).1 i =
      DeGrey529.gi x.1 i * y.2 - DeGrey529.gi y.1 i * x.2 := by
  unfold DeGrey529.gi DeGrey529.qsub
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by simp [List.length_map, hi])]
  simp [DeGrey529.gi, List.getElem_map, List.getElem_range]

private theorem gi_qmul (x y : DeGrey529.QF) (idx : Nat) (hidx : idx < 16) :
    DeGrey529.gi (DeGrey529.qmul x y).1 idx =
      (List.range 16).foldl (fun acc i =>
        acc + DeGrey529.gi x.1 i * DeGrey529.gi y.1 (Nat.xor i idx) *
          DeGrey529.bcoeff (Nat.land i (Nat.xor i idx))) 0 := by
  unfold DeGrey529.gi DeGrey529.qmul
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by simp [List.length_map, hidx]), Option.getD_some]
  simp [DeGrey529.gi, List.getElem_map, List.getElem_range]

private theorem qmulCoeff3511_bridge (x y : DeGrey529.QF)
    (hx : qf3511Wf x) (hy : qf3511Wf y) (idx : Nat) (hidx : idx < 8) :
    DeGrey529.gi (DeGrey529.qmul x y).1 (radIdx3511 idx) =
      qmulCoeff3511 (compressNum3511 x.1) (compressNum3511 y.1) idx := by
  have hxz := qf3511Wf_sqrt7_coeffs_zero x hx
  have hyz := qf3511Wf_sqrt7_coeffs_zero y hy
  have hx4 : DeGrey529.gi x.1 4 = 0 := hxz.1
  have hx5 : DeGrey529.gi x.1 5 = 0 := hxz.2.1
  have hx6 : DeGrey529.gi x.1 6 = 0 := hxz.2.2.1
  have hx7 : DeGrey529.gi x.1 7 = 0 := hxz.2.2.2.1
  have hx12 : DeGrey529.gi x.1 12 = 0 := hxz.2.2.2.2.1
  have hx13 : DeGrey529.gi x.1 13 = 0 := hxz.2.2.2.2.2.1
  have hx14 : DeGrey529.gi x.1 14 = 0 := hxz.2.2.2.2.2.2.1
  have hx15 : DeGrey529.gi x.1 15 = 0 := hxz.2.2.2.2.2.2.2
  have hy4 : DeGrey529.gi y.1 4 = 0 := hyz.1
  have hy5 : DeGrey529.gi y.1 5 = 0 := hyz.2.1
  have hy6 : DeGrey529.gi y.1 6 = 0 := hyz.2.2.1
  have hy7 : DeGrey529.gi y.1 7 = 0 := hyz.2.2.2.1
  have hy12 : DeGrey529.gi y.1 12 = 0 := hyz.2.2.2.2.1
  have hy13 : DeGrey529.gi y.1 13 = 0 := hyz.2.2.2.2.2.1
  have hy14 : DeGrey529.gi y.1 14 = 0 := hyz.2.2.2.2.2.2.1
  have hy15 : DeGrey529.gi y.1 15 = 0 := hyz.2.2.2.2.2.2.2
  match idx with
  | 0 =>
      rw [gi_qmul x y (radIdx3511 0) (radIdx3511_lt16 0)]
      simp [range16_eq, range8_eq, qmulCoeff3511, qmulTerm3511, compressNum3511_gi, radIdx3511,
        DeGrey529.bcoeff, bcoeff3511, Nat.testBit, hx4, hx5, hx6, hx7, hx12, hx13, hx14, hx15,
        hy4, hy5, hy6, hy7, hy12, hy13, hy14, hy15]
  | 1 =>
      rw [gi_qmul x y (radIdx3511 1) (radIdx3511_lt16 1)]
      simp [range16_eq, range8_eq, qmulCoeff3511, qmulTerm3511, compressNum3511_gi, radIdx3511,
        DeGrey529.bcoeff, bcoeff3511, Nat.testBit, hx4, hx5, hx6, hx7, hx12, hx13, hx14, hx15,
        hy4, hy5, hy6, hy7, hy12, hy13, hy14, hy15]
  | 2 =>
      rw [gi_qmul x y (radIdx3511 2) (radIdx3511_lt16 2)]
      simp [range16_eq, range8_eq, qmulCoeff3511, qmulTerm3511, compressNum3511_gi, radIdx3511,
        DeGrey529.bcoeff, bcoeff3511, Nat.testBit, hx4, hx5, hx6, hx7, hx12, hx13, hx14, hx15,
        hy4, hy5, hy6, hy7, hy12, hy13, hy14, hy15]
  | 3 =>
      rw [gi_qmul x y (radIdx3511 3) (radIdx3511_lt16 3)]
      simp [range16_eq, range8_eq, qmulCoeff3511, qmulTerm3511, compressNum3511_gi, radIdx3511,
        DeGrey529.bcoeff, bcoeff3511, Nat.testBit, hx4, hx5, hx6, hx7, hx12, hx13, hx14, hx15,
        hy4, hy5, hy6, hy7, hy12, hy13, hy14, hy15]
  | 4 =>
      rw [gi_qmul x y (radIdx3511 4) (radIdx3511_lt16 4)]
      simp [range16_eq, range8_eq, qmulCoeff3511, qmulTerm3511, compressNum3511_gi, radIdx3511,
        DeGrey529.bcoeff, bcoeff3511, Nat.testBit, hx4, hx5, hx6, hx7, hx12, hx13, hx14, hx15,
        hy4, hy5, hy6, hy7, hy12, hy13, hy14, hy15]
  | 5 =>
      rw [gi_qmul x y (radIdx3511 5) (radIdx3511_lt16 5)]
      simp [range16_eq, range8_eq, qmulCoeff3511, qmulTerm3511, compressNum3511_gi, radIdx3511,
        DeGrey529.bcoeff, bcoeff3511, Nat.testBit, hx4, hx5, hx6, hx7, hx12, hx13, hx14, hx15,
        hy4, hy5, hy6, hy7, hy12, hy13, hy14, hy15]
  | 6 =>
      rw [gi_qmul x y (radIdx3511 6) (radIdx3511_lt16 6)]
      simp [range16_eq, range8_eq, qmulCoeff3511, qmulTerm3511, compressNum3511_gi, radIdx3511,
        DeGrey529.bcoeff, bcoeff3511, Nat.testBit, hx4, hx5, hx6, hx7, hx12, hx13, hx14, hx15,
        hy4, hy5, hy6, hy7, hy12, hy13, hy14, hy15]
  | 7 =>
      rw [gi_qmul x y (radIdx3511 7) (radIdx3511_lt16 7)]
      simp [range16_eq, range8_eq, qmulCoeff3511, qmulTerm3511, compressNum3511_gi, radIdx3511,
        DeGrey529.bcoeff, bcoeff3511, Nat.testBit, hx4, hx5, hx6, hx7, hx12, hx13, hx14, hx15,
        hy4, hy5, hy6, hy7, hy12, hy13, hy14, hy15]
  | idx + 8 => omega

theorem evalNum3511_mul_right (l : List Int) (c : R.F) :
    R.mul (evalNum3511 R l) c =
      fsum R (List.range 8)
        (fun k => R.mul (R.mul (ofInt R (DeGrey529.gi l (radIdx3511 k))) c) (r3511 R k)) := by
  unfold evalNum3511
  rw [mul_fsum_right]
  apply fsum_congr
  intro k _
  rw [R.mul_assoc, R.mul_comm (r3511 R k) c, ← R.mul_assoc]

theorem evalNum3511_qmul_num_bridge (x y : DeGrey529.QF)
    (hx : qf3511Wf x) (hy : qf3511Wf y) :
    evalNum3511 R (DeGrey529.qmul x y).1 =
      evalNum8 R (qmulNum3511 (compressNum3511 x.1) (compressNum3511 y.1)) := by
  rw [evalNum3511_eq_evalNum8_compress]
  unfold evalNum8
  apply fsum_congr
  intro k hk
  have hk8 : k < 8 := List.mem_range.mp hk
  rw [compressNum3511_gi (DeGrey529.qmul x y).1 k hk8,
    qmulNum3511_gi (compressNum3511 x.1) (compressNum3511 y.1) k hk8,
    qmulCoeff3511_bridge x y hx hy k hk8]

theorem evalNum3511_qmul (x y : DeGrey529.QF)
    (hx : qf3511Wf x) (hy : qf3511Wf y) :
    evalNum3511 R (DeGrey529.qmul x y).1 =
      R.mul (evalNum3511 R x.1) (evalNum3511 R y.1) := by
  rw [evalNum3511_qmul_num_bridge R x y hx hy, evalNum8_qmulNum3511,
    ← evalNum3511_eq_evalNum8_compress R x.1,
    ← evalNum3511_eq_evalNum8_compress R y.1]

theorem evalNum3511_qadd (x y : DeGrey529.QF) :
    evalNum3511 R (DeGrey529.qadd x y).1 =
      R.add (R.mul (evalNum3511 R x.1) (ofInt R y.2))
        (R.mul (evalNum3511 R y.1) (ofInt R x.2)) := by
  rw [evalNum3511_mul_right, evalNum3511_mul_right, ← fsum_add]
  unfold evalNum3511
  apply fsum_congr
  intro k hk
  have hidx : radIdx3511 k < 16 := radIdx3511_lt16 k
  rw [gi_qadd x y (radIdx3511 k) hidx, ofInt_add, ofInt_mul, ofInt_mul,
    R.right_distrib]

theorem evalNum3511_qsub (x y : DeGrey529.QF) :
    evalNum3511 R (DeGrey529.qsub x y).1 =
      R.add (R.mul (evalNum3511 R x.1) (ofInt R y.2))
        (R.neg (R.mul (evalNum3511 R y.1) (ofInt R x.2))) := by
  rw [evalNum3511_mul_right, evalNum3511_mul_right, ← fsum_neg, ← fsum_add]
  unfold evalNum3511
  apply fsum_congr
  intro k hk
  have hidx : radIdx3511 k < 16 := radIdx3511_lt16 k
  rw [gi_qsub x y (radIdx3511 k) hidx, ofInt_sub, ofInt_mul, ofInt_mul,
    R.right_distrib, neg_mul]

theorem frac_add (a1 a2 d1 d2 : R.F) (h1 : d1 ≠ R.zero) (h2 : d2 ≠ R.zero) :
    R.mul (R.add (R.mul a1 d2) (R.mul a2 d1)) (R.inv (R.mul d1 d2))
      = R.add (R.mul a1 (R.inv d1)) (R.mul a2 (R.inv d2)) := by
  rw [inv_mul_inv R h1 h2, R.right_distrib]
  congr 1
  · rw [mul4comm, R.mul_inv d2 h2, R.mul_one]
  · rw [R.mul_comm (R.inv d1) (R.inv d2), mul4comm, R.mul_inv d1 h1, R.mul_one]

theorem frac_sub (a1 a2 d1 d2 : R.F) (h1 : d1 ≠ R.zero) (h2 : d2 ≠ R.zero) :
    R.mul (R.add (R.mul a1 d2) (R.neg (R.mul a2 d1))) (R.inv (R.mul d1 d2))
      = R.add (R.mul a1 (R.inv d1)) (R.neg (R.mul a2 (R.inv d2))) := by
  rw [inv_mul_inv R h1 h2, R.right_distrib]
  congr 1
  · rw [mul4comm, R.mul_inv d2 h2, R.mul_one]
  · rw [neg_mul, R.mul_comm (R.inv d1) (R.inv d2), mul4comm, R.mul_inv d1 h1, R.mul_one]

theorem frac_mul (a1 a2 d1 d2 : R.F) (h1 : d1 ≠ R.zero) (h2 : d2 ≠ R.zero) :
    R.mul (R.mul a1 a2) (R.inv (R.mul d1 d2)) =
      R.mul (R.mul a1 (R.inv d1)) (R.mul a2 (R.inv d2)) := by
  rw [inv_mul_inv R h1 h2, mul4comm]

theorem phi3511_qmul (hden : IntCastNonzero R) (x y : DeGrey529.QF)
    (hx : qf3511Wf x) (hy : qf3511Wf y) :
    phi3511 R (DeGrey529.qmul x y) = R.mul (phi3511 R x) (phi3511 R y) := by
  unfold phi3511
  rw [show (DeGrey529.qmul x y).2 = x.2 * y.2 from rfl, ofInt_mul,
    evalNum3511_qmul R x y hx hy]
  exact frac_mul R (evalNum3511 R x.1) (evalNum3511 R y.1)
    (ofInt R x.2) (ofInt R y.2) (hden x.2 hx.1) (hden y.2 hy.1)

theorem phi3511_qadd (hden : IntCastNonzero R) (x y : DeGrey529.QF)
    (hx : qf3511Wf x) (hy : qf3511Wf y) :
    phi3511 R (DeGrey529.qadd x y) = R.add (phi3511 R x) (phi3511 R y) := by
  unfold phi3511
  rw [show (DeGrey529.qadd x y).2 = x.2 * y.2 from rfl, ofInt_mul, evalNum3511_qadd]
  exact frac_add R (evalNum3511 R x.1) (evalNum3511 R y.1)
    (ofInt R x.2) (ofInt R y.2) (hden x.2 hx.1) (hden y.2 hy.1)

theorem phi3511_qsub (hden : IntCastNonzero R) (x y : DeGrey529.QF)
    (hx : qf3511Wf x) (hy : qf3511Wf y) :
    phi3511 R (DeGrey529.qsub x y) =
      R.add (phi3511 R x) (R.neg (phi3511 R y)) := by
  unfold phi3511
  rw [show (DeGrey529.qsub x y).2 = x.2 * y.2 from rfl, ofInt_mul, evalNum3511_qsub]
  exact frac_sub R (evalNum3511 R x.1) (evalNum3511 R y.1)
    (ofInt R x.2) (ofInt R y.2) (hden x.2 hx.1) (hden y.2 hy.1)

private theorem isOne_coeff0 (d : DeGrey529.QF) (hone : DeGrey529.isOne d = true) :
    DeGrey529.gi d.1 0 = d.2 := by
  have hAll :
      ∀ i, i ∈ List.range 16 →
        (if i = 0 then DeGrey529.gi d.1 0 == d.2 else DeGrey529.gi d.1 i == 0) = true := by
    exact List.all_eq_true.mp (by simpa [DeGrey529.isOne] using hone)
  have h0 := hAll 0 (by simp)
  simpa using h0

private theorem isOne_coeff_zero (d : DeGrey529.QF) (hone : DeGrey529.isOne d = true)
    {i : Nat} (hi : i < 16) (hiz : i ≠ 0) : DeGrey529.gi d.1 i = 0 := by
  have hAll :
      ∀ i, i ∈ List.range 16 →
        (if i = 0 then DeGrey529.gi d.1 0 == d.2 else DeGrey529.gi d.1 i == 0) = true := by
    exact List.all_eq_true.mp (by simpa [DeGrey529.isOne] using hone)
  have hb := hAll i (List.mem_range.mpr hi)
  simpa [hiz] using hb

theorem evalNum3511_isOne (d : DeGrey529.QF) (hone : DeGrey529.isOne d = true) :
    evalNum3511 R d.1 = ofInt R d.2 := by
  have h0 : DeGrey529.gi d.1 0 = d.2 := isOne_coeff0 d hone
  have h1 : DeGrey529.gi d.1 1 = 0 := isOne_coeff_zero d hone (by decide) (by decide)
  have h2 : DeGrey529.gi d.1 2 = 0 := isOne_coeff_zero d hone (by decide) (by decide)
  have h3 : DeGrey529.gi d.1 3 = 0 := isOne_coeff_zero d hone (by decide) (by decide)
  have h8 : DeGrey529.gi d.1 8 = 0 := isOne_coeff_zero d hone (by decide) (by decide)
  have h9 : DeGrey529.gi d.1 9 = 0 := isOne_coeff_zero d hone (by decide) (by decide)
  have h10 : DeGrey529.gi d.1 10 = 0 := isOne_coeff_zero d hone (by decide) (by decide)
  have h11 : DeGrey529.gi d.1 11 = 0 := isOne_coeff_zero d hone (by decide) (by decide)
  rw [show evalNum3511 R d.1 =
      R.add (R.mul (ofInt R (DeGrey529.gi d.1 0)) (r3511 R 0))
      (R.add (R.mul (ofInt R (DeGrey529.gi d.1 1)) (r3511 R 1))
      (R.add (R.mul (ofInt R (DeGrey529.gi d.1 2)) (r3511 R 2))
      (R.add (R.mul (ofInt R (DeGrey529.gi d.1 3)) (r3511 R 3))
      (R.add (R.mul (ofInt R (DeGrey529.gi d.1 8)) (r3511 R 4))
      (R.add (R.mul (ofInt R (DeGrey529.gi d.1 9)) (r3511 R 5))
      (R.add (R.mul (ofInt R (DeGrey529.gi d.1 10)) (r3511 R 6))
      (R.add (R.mul (ofInt R (DeGrey529.gi d.1 11)) (r3511 R 7)) R.zero))))))) from rfl]
  rw [h0, h1, h2, h3, h8, h9, h10, h11, r3511_zero]
  simp only [show ofInt R (0 : Int) = R.zero from rfl, mul_zero_left, R.add_zero, R.mul_one]

theorem phi3511_unit (hden : IntCastNonzero R) (d : DeGrey529.QF)
    (hd : qf3511Wf d) (hone : DeGrey529.isOne d = true) :
    phi3511 R d = R.one := by
  unfold phi3511
  rw [evalNum3511_isOne R d hone]
  exact R.mul_inv (ofInt R d.2) (hden d.2 hd.1)

/-- Review-facing package for the derived `phi3511` laws. -/
structure Phi3511AddSubUnitCertificate where
  int_cast_nonzero : IntCastNonzero R
  compressed_qmul_core :
    ∀ x y, evalNum8 R (qmulNum3511 x y) = R.mul (evalNum8 R x) (evalNum8 R y)
  qmul_law :
    ∀ x y, qf3511Wf x → qf3511Wf y →
      phi3511 R (DeGrey529.qmul x y) = R.mul (phi3511 R x) (phi3511 R y)
  qadd_law :
    ∀ x y, qf3511Wf x → qf3511Wf y →
      phi3511 R (DeGrey529.qadd x y) = R.add (phi3511 R x) (phi3511 R y)
  qsub_law :
    ∀ x y, qf3511Wf x → qf3511Wf y →
      phi3511 R (DeGrey529.qsub x y) =
        R.add (phi3511 R x) (R.neg (phi3511 R y))
  unit_law :
    ∀ d, qf3511Wf d → DeGrey529.isOne d = true → phi3511 R d = R.one

def phi3511AddSubUnitCertificate (hden : IntCastNonzero R) :
    Phi3511AddSubUnitCertificate R where
  int_cast_nonzero := hden
  compressed_qmul_core := evalNum8_qmulNum3511 R
  qmul_law := phi3511_qmul R hden
  qadd_law := phi3511_qadd R hden
  qsub_law := phi3511_qsub R hden
  unit_law := phi3511_unit R hden

def toDerivedQF3511TransferWf (hden : IntCastNonzero R) : QF3511TransferWf where
  F := R.F
  add := R.add
  mul := R.mul
  sub := fun a b => R.add a (R.neg b)
  phi := phi3511 R
  isUnitVal := fun x => x = R.one
  hadd := phi3511_qadd R hden
  hmul := phi3511_qmul R hden
  hsub := phi3511_qsub R hden
  hunit := phi3511_unit R hden
  hcoord_edge_endpoints3511 := by
    simpa [DeGrey529.Transfer3511.primeSupport] using
      DeGrey529.Param.current_g529_full_prime_subplane_contains_all_edge_endpoints
  hedge_terms3511 := by
    simpa [DeGrey529.Transfer3511.primeSupport] using
      DeGrey529.Param.current_g529_edge_distances_supported_in_3511
  hcurrent_lrat_support := by
    simpa [DeGrey529.Transfer3511.primeSupport] using
      DeGrey529.Param.current_g529_full_prime_subplane_carries_current_lrat_obstruction_support

theorem derived_phi3511_chi_ge_5_current_embedding (hden : IntCastNonzero R) :
    ¬ Nonempty (UnitDistanceChromatic.PlaneColouring
      (R.F × R.F) (toDerivedQF3511TransferWf R hden).unit 4) :=
  DeGrey529.Transfer3511.qf3511Transfer_chi_ge_5_current_embedding
    (toDerivedQF3511TransferWf R hden)

/-- Single citation object for the evaluator-derived `{3,5,11}` transfer path. -/
structure Phi3511DerivedTransferCertificate (hden : IntCastNonzero R) where
  law_certificate : Phi3511AddSubUnitCertificate R
  transfer_target : QF3511TransferWf
  transfer_target_is_derived : transfer_target = toDerivedQF3511TransferWf R hden
  transfer_no_four_colouring :
    ¬ Nonempty (UnitDistanceChromatic.PlaneColouring
      (R.F × R.F) (toDerivedQF3511TransferWf R hden).unit 4)

def phi3511DerivedTransferCertificate (hden : IntCastNonzero R) :
    Phi3511DerivedTransferCertificate R hden where
  law_certificate := phi3511AddSubUnitCertificate R hden
  transfer_target := toDerivedQF3511TransferWf R hden
  transfer_target_is_derived := rfl
  transfer_no_four_colouring := derived_phi3511_chi_ge_5_current_embedding R hden

#check @RootedField3511.evalNum3511
#check @RootedField3511.evalNum8
#check @RootedField3511.compressNum3511
#check @RootedField3511.qmulNum3511
#check @RootedField3511.radIdx3511_xor
#check @RootedField3511.radIdx3511_land
#check @RootedField3511.qf3511Wf_coeff_zero_of_unsupported
#check @RootedField3511.qf3511Wf_sqrt7_coeffs_zero
#check @RootedField3511.evalNum8_qmulNum3511
#check @RootedField3511.evalNum3511_eq_evalNum8_compress
#check @RootedField3511.evalNum3511_qmul_num_bridge
#check @RootedField3511.evalNum3511_qmul
#check @RootedField3511.phi3511
#check @RootedField3511.IntCastNonzero
#check @RootedField3511.evalNum3511_qadd
#check @RootedField3511.evalNum3511_qsub
#check @RootedField3511.phi3511_qmul
#check @RootedField3511.phi3511_qadd
#check @RootedField3511.phi3511_qsub
#check @RootedField3511.phi3511_unit
#check @RootedField3511.Phi3511AddSubUnitCertificate
#check @RootedField3511.phi3511AddSubUnitCertificate
#check @RootedField3511.toDerivedQF3511TransferWf
#check @RootedField3511.derived_phi3511_chi_ge_5_current_embedding
#check @RootedField3511.Phi3511DerivedTransferCertificate
#check @RootedField3511.phi3511DerivedTransferCertificate

#print axioms RootedField3511.evalNum3511_qadd
#print axioms RootedField3511.evalNum3511_qsub
#print axioms RootedField3511.qf3511Wf_coeff_zero_of_unsupported
#print axioms RootedField3511.qf3511Wf_sqrt7_coeffs_zero
#print axioms RootedField3511.evalNum8_qmulNum3511
#print axioms RootedField3511.evalNum3511_qmul_num_bridge
#print axioms RootedField3511.evalNum3511_qmul
#print axioms RootedField3511.phi3511_qmul
#print axioms RootedField3511.phi3511_qadd
#print axioms RootedField3511.phi3511_qsub
#print axioms RootedField3511.phi3511_unit
#print axioms RootedField3511.toDerivedQF3511TransferWf
#print axioms RootedField3511.derived_phi3511_chi_ge_5_current_embedding
#print axioms RootedField3511.phi3511DerivedTransferCertificate

#eval IO.println "SounioDeGreyChi5Eval3511: explicit 8-mask evaluator proves the compressed qmul core, bridges de Grey's 16-mask qmul through {3,5,11} support, derives phi3511 qmul/qadd/qsub/unit laws under IntCastNonzero, and packages the derived evaluator as a QF3511TransferWf target with a single transfer certificate."

end RootedField3511

end DeGrey529.Rooted3511
