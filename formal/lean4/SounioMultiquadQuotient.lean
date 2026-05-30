import SounioMultiquadRing

set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false

namespace MultiquadQuotient

open MultiquadRing List

/-- Positive-denominator, length-16 representatives (de Grey pipeline output). -/
def QFp : Type := { x : QF // x.2 ≠ 0 ∧ x.1.length = 16 }

namespace QFp

def mk (x : QF) (hden : x.2 ≠ 0) (hlen : x.1.length = 16) : QFp := ⟨x, hden, hlen⟩

def val (p : QFp) : QF := p.1

@[simp] theorem val_mk (x : QF) (hden : x.2 ≠ 0) (hlen : x.1.length = 16) :
    val ⟨x, hden, hlen⟩ = x := rfl

theorem den_ne_zero (p : QFp) : p.1.2 ≠ 0 := p.2.1

theorem len16 (p : QFp) : p.1.1.length = 16 := p.2.2

def zero : QFp := mk qzero (by decide) (by decide)

def one : QFp := mk qfone (by decide) (by decide)

@[simp] theorem val_zero : zero.val = qzero := rfl

@[simp] theorem val_one : one.val = qfone := rfl

end QFp

open QFp

/-- Value equivalence: same rational multiquadratic field element. -/
def QFeq (x y : QF) : Prop :=
  ∀ i, i < 16 → gi x.1 i * y.2 = gi y.1 i * x.2

namespace QFeq

theorem refl (x : QF) : QFeq x x := fun _ _ => rfl

theorem symm {x y : QF} (h : QFeq x y) : QFeq y x := fun i hi => (h i hi).symm

theorem trans {x y z : QF} (hxy : QFeq x y) (hyz : QFeq y z)
    (hy : y.2 ≠ 0) : QFeq x z := fun i hi => by
  have hx := hxy i hi
  have hz := hyz i hi
  have hmul : (gi x.1 i * z.2) * y.2 = (gi z.1 i * x.2) * y.2 := by
    calc (gi x.1 i * z.2) * y.2
        = gi x.1 i * (z.2 * y.2) := by rw [← Int.mul_assoc]
      _ = gi x.1 i * (y.2 * z.2) := by rw [Int.mul_comm z.2]
      _ = (gi x.1 i * y.2) * z.2 := by rw [Int.mul_assoc]
      _ = (gi y.1 i * x.2) * z.2 := by rw [hx]
      _ = gi y.1 i * (x.2 * z.2) := by rw [← Int.mul_assoc]
      _ = gi y.1 i * (z.2 * x.2) := by rw [Int.mul_comm x.2]
      _ = (gi y.1 i * z.2) * x.2 := by rw [Int.mul_assoc]
      _ = (gi z.1 i * y.2) * x.2 := by rw [hz]
      _ = gi z.1 i * (y.2 * x.2) := by rw [← Int.mul_assoc]
      _ = gi z.1 i * (x.2 * y.2) := by rw [Int.mul_comm y.2]
      _ = (gi z.1 i * x.2) * y.2 := by rw [Int.mul_assoc]
  exact Int.eq_of_mul_eq_mul_right hy hmul

theorem qfp_refl (x : QFp) : QFeq x.1 x.1 := refl x.1

theorem qfp_symm {x y : QFp} (h : QFeq x.1 y.1) : QFeq y.1 x.1 := symm h

theorem qfp_trans {x y z : QFp} (hxy : QFeq x.1 y.1) (hyz : QFeq y.1 z.1) :
    QFeq x.1 z.1 :=
  trans hxy hyz y.2.1

end QFeq

instance qfpSetoid : Setoid QFp where
  r a b := QFeq a.1 b.1
  iseqv := {
    refl := QFeq.qfp_refl
    symm := fun {_ _} h => QFeq.qfp_symm h
    trans := fun {_ _ _} h1 h2 => QFeq.qfp_trans h1 h2
  }

abbrev QQuotient := Quotient qfpSetoid

/-! ## `qsub` list facts (before `gi` bridge) -/

theorem qsub_list_len (x y : QF) : (qsub x y).1.length = 16 := by
  simp [qsub, List.length_map, List.length_range]

theorem qsub_getElem (x y : QF) (i : Nat) (hi : i < 16) :
    (qsub x y).1[i]'(by rw [qsub_list_len]; exact hi) =
      gi x.1 i * y.2 - gi y.1 i * x.2 := by
  simp [qsub, gi, List.getElem_map, List.getElem_range (by simp; exact hi)]

/-! ## Bridge `gi` with proven `getElem` coefficients -/

theorem qadd_gi (x y : QF) (i : Nat) (hi : i < 16) :
    gi (qadd x y).1 i = gi x.1 i * y.2 + gi y.1 i * x.2 := by
  have hlen : i < (qadd x y).1.length := by rw [qadd_list_len]; exact hi
  rw [gi_eq_getElem (qadd x y).1 i hlen, qadd_getElem x y i hi]

theorem qsub_gi (x y : QF) (i : Nat) (hi : i < 16) :
    gi (qsub x y).1 i = gi x.1 i * y.2 - gi y.1 i * x.2 := by
  have hlen : i < (qsub x y).1.length := by rw [qsub_list_len]; exact hi
  rw [gi_eq_getElem (qsub x y).1 i hlen, qsub_getElem x y i hi]

theorem qmul_gi (x y : QF) (idx : Nat) (hidx : idx < 16) :
    gi (qmul x y).1 idx = qmulCoeff x y idx := by
  have hlen : idx < (qmul x y).1.length := by rw [qmul_list_len]; exact hidx
  rw [gi_eq_getElem (qmul x y).1 idx hlen, qmul_getElem x y idx hidx]

theorem xor_lt_16 {i idx : Nat} (hi : i < 16) (hidx : idx < 16) : i ^^^ idx < 16 := by
  match idx with
  | 0 => simpa [Nat.zero_xor] using hi
  | 1 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by grind)
  | 2 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by grind)
  | 3 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by grind)
  | 4 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by grind)
  | 5 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by grind)
  | 6 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by grind)
  | 7 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by grind)
  | 8 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by grind)
  | 9 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by grind)
  | 10 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by grind)
  | 11 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by grind)
  | 12 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by grind)
  | 13 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by grind)
  | 14 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by grind)
  | 15 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by grind)
  | idx + 16 => exact absurd hidx (by grind)

/-! ## List fold helpers for convolution linearity -/

theorem foldl_add_init (l : List Nat) (g : Nat → Int) (init : Int) :
    l.foldl (fun acc i => acc + g i) init = init + l.foldl (fun acc i => acc + g i) 0 := by
  induction l generalizing init with
  | nil => simp
  | cons i l ih =>
    simp only [List.foldl, show (0 : Int) + g i = g i from by simp]
    calc List.foldl (fun acc i => acc + g i) init (i :: l)
        = List.foldl (fun acc i => acc + g i) (init + g i) l := rfl
      _ = init + g i + List.foldl (fun acc i => acc + g i) 0 l := ih (init + g i)
      _ = init + List.foldl (fun acc i => acc + g i) (g i) l := by
          rw [show init + g i + _ = init + (g i + _) from by rw [Int.add_assoc]]
          congr 1
          exact (ih (g i)).symm

theorem foldl_add_split (l : List Nat) (f g : Nat → Int) :
    l.foldl (fun acc i => acc + (f i + g i)) (0 : Int) =
      l.foldl (fun acc i => acc + f i) 0 + l.foldl (fun acc i => acc + g i) 0 := by
  induction l with
  | nil => simp
  | cons i l ih =>
    simp only [List.foldl, show (0 : Int) + (f i + g i) = f i + g i from by simp,
               show (0 : Int) + f i = f i from by simp, show (0 : Int) + g i = g i from by simp]
    rw [foldl_add_init l (fun j => f j + g j) (f i + g i),
        foldl_add_init l f (f i), foldl_add_init l g (g i), ih]
    omega

theorem foldl_smul_split (l : List Nat) (c : Int) (f : Nat → Int) :
    l.foldl (fun acc i => acc + c * f i) (0 : Int) =
      c * l.foldl (fun acc i => acc + f i) 0 := by
  induction l with
  | nil => simp
  | cons i l ih =>
    simp only [List.foldl, show (0 : Int) + c * f i = c * f i from by simp,
               show (0 : Int) + f i = f i from by simp]
    rw [foldl_add_init l (fun j => c * f j) (c * f i), ih, ← Int.mul_add, foldl_add_init l f (f i)]

/-! ## QF operations preserve `QFp` -/

def qaddQfp (x y : QFp) : QFp :=
  QFp.mk (qadd x.1 y.1)
    (Int.mul_ne_zero x.2.1 y.2.1)
    (by rw [qadd_list_len])

def qmulQfp (x y : QFp) : QFp :=
  QFp.mk (qmul x.1 y.1)
    (Int.mul_ne_zero x.2.1 y.2.1)
    (by rw [qmul_list_len])

def qsubQfp (x y : QFp) : QFp :=
  QFp.mk (qsub x.1 y.1)
    (Int.mul_ne_zero x.2.1 y.2.1)
    (by rw [qsub_list_len])

def qnegQfp (x : QFp) : QFp := qsubQfp QFp.zero x

/-! ## Congruence lemmas -/

theorem foldl_add_mul_comm (l : List Nat) (c : Int) (f : Nat → Int) :
    l.foldl (fun acc i => acc + f i * c) (0 : Int) =
      l.foldl (fun acc i => acc + c * f i) 0 := by
  apply foldl_add_pointwise l (fun i => f i * c) (fun i => c * f i) 0
  intro i _
  exact Int.mul_comm (f i) c

theorem qadd_QFeq_congr (x x' y y' : QF) (hx : QFeq x x') (hy : QFeq y y') :
    QFeq (qadd x y) (qadd x' y') := by
  intro i hi
  rw [qadd_gi x y i hi, qadd_gi x' y' i hi]
  have t1 : gi x.1 i * y.2 * (x'.2 * y'.2) = gi x'.1 i * y'.2 * (x.2 * y.2) := by
    calc gi x.1 i * y.2 * (x'.2 * y'.2)
        = (gi x.1 i * x'.2) * y.2 * y'.2 := by grind
      _ = (gi x'.1 i * x.2) * y.2 * y'.2 := by rw [hx i hi]
      _ = gi x'.1 i * (x.2 * y.2 * y'.2) := by grind
      _ = gi x'.1 i * y'.2 * (x.2 * y.2) := by grind
  have t2 : gi y.1 i * x.2 * (x'.2 * y'.2) = gi y'.1 i * x'.2 * (x.2 * y.2) := by
    grind [hy i hi]
  calc (gi x.1 i * y.2 + gi y.1 i * x.2) * (x'.2 * y'.2)
      = gi x.1 i * y.2 * (x'.2 * y'.2) + gi y.1 i * x.2 * (x'.2 * y'.2) := by rw [Int.add_mul]
    _ = gi x'.1 i * y'.2 * (x.2 * y.2) + gi y'.1 i * x'.2 * (x.2 * y.2) := by rw [t1, t2]
    _ = (gi x'.1 i * y'.2 + gi y'.1 i * x'.2) * (x.2 * y.2) := by rw [← Int.add_mul]

theorem qsub_QFeq_congr (x x' y y' : QF) (hx : QFeq x x') (hy : QFeq y y') :
    QFeq (qsub x y) (qsub x' y') := by
  intro i hi
  rw [qsub_gi x y i hi, qsub_gi x' y' i hi]
  have t1 : gi x.1 i * y.2 * (x'.2 * y'.2) = gi x'.1 i * y'.2 * (x.2 * y.2) := by
    calc gi x.1 i * y.2 * (x'.2 * y'.2)
        = (gi x.1 i * x'.2) * y.2 * y'.2 := by grind
      _ = (gi x'.1 i * x.2) * y.2 * y'.2 := by rw [hx i hi]
      _ = gi x'.1 i * (x.2 * y.2 * y'.2) := by grind
      _ = gi x'.1 i * y'.2 * (x.2 * y.2) := by grind
  have t2 : gi y.1 i * x.2 * (x'.2 * y'.2) = gi y'.1 i * x'.2 * (x.2 * y.2) := by
    grind [hy i hi]
  calc (gi x.1 i * y.2 - gi y.1 i * x.2) * (x'.2 * y'.2)
      = gi x.1 i * y.2 * (x'.2 * y'.2) - gi y.1 i * x.2 * (x'.2 * y'.2) := by rw [Int.sub_mul]
    _ = gi x'.1 i * y'.2 * (x.2 * y.2) - gi y'.1 i * x'.2 * (x.2 * y.2) := by rw [t1, t2]
    _ = (gi x'.1 i * y'.2 - gi y'.1 i * x'.2) * (x.2 * y.2) := by rw [← Int.sub_mul]

theorem qmulTerm_left_scale (x x' y : QF) (h : QFeq x x') (i idx : Nat) (hi : i < 16) :
    qmulTerm x y i idx * x'.2 = qmulTerm x' y i idx * x.2 := by
  unfold qmulTerm
  have hx' := h i hi
  calc gi x.1 i * gi y.1 (i ^^^ idx) * bcoeff (i &&& (i ^^^ idx)) * x'.2
      = (gi x.1 i * x'.2) * (gi y.1 (i ^^^ idx) * bcoeff (i &&& (i ^^^ idx))) := by
          simp [Int.mul_assoc, Int.mul_comm, Int.mul_left_comm]
    _ = (gi x'.1 i * x.2) * (gi y.1 (i ^^^ idx) * bcoeff (i &&& (i ^^^ idx))) := by rw [hx']
    _ = gi x'.1 i * gi y.1 (i ^^^ idx) * bcoeff (i &&& (i ^^^ idx)) * x.2 := by
          simp [Int.mul_assoc, Int.mul_comm, Int.mul_left_comm]

theorem qmulTerm_right_scale (x y y' : QF) (h : QFeq y y') (i idx : Nat) (hi : i < 16)
    (hidx : idx < 16) :
    qmulTerm x y i idx * y'.2 = qmulTerm x y' i idx * y.2 := by
  unfold qmulTerm
  have hj := xor_lt_16 hi hidx
  have hy' := h (i ^^^ idx) hj
  calc gi x.1 i * gi y.1 (i ^^^ idx) * bcoeff (i &&& (i ^^^ idx)) * y'.2
      = gi x.1 i * (gi y.1 (i ^^^ idx) * y'.2) * bcoeff (i &&& (i ^^^ idx)) := by
          simp [Int.mul_assoc, Int.mul_comm, Int.mul_left_comm]
    _ = gi x.1 i * (gi y'.1 (i ^^^ idx) * y.2) * bcoeff (i &&& (i ^^^ idx)) := by rw [hy']
    _ = gi x.1 i * gi y'.1 (i ^^^ idx) * bcoeff (i &&& (i ^^^ idx)) * y.2 := by
          simp [Int.mul_assoc, Int.mul_comm, Int.mul_left_comm]

theorem qmulCoeff_left_congr (x x' y : QF) (h : QFeq x x') (idx : Nat) :
    qmulCoeff x y idx * x'.2 = qmulCoeff x' y idx * x.2 := by
  unfold qmulCoeff
  calc (List.range 16).foldl (fun acc i => acc + qmulTerm x y i idx) 0 * x'.2
      = x'.2 * (List.range 16).foldl (fun acc i => acc + qmulTerm x y i idx) 0 := by
          rw [Int.mul_comm]
      _ = (List.range 16).foldl (fun acc i => acc + x'.2 * qmulTerm x y i idx) 0 := by
          exact (foldl_smul_split (List.range 16) x'.2 (fun i => qmulTerm x y i idx)).symm
      _ = (List.range 16).foldl (fun acc i => acc + qmulTerm x' y i idx * x.2) 0 := by
          apply foldl_add_pointwise (List.range 16)
            (fun i => x'.2 * qmulTerm x y i idx)
            (fun i => qmulTerm x' y i idx * x.2) 0
          intro i mem
          have hi : i < 16 := by simpa using mem
          rw [Int.mul_comm (x'.2), qmulTerm_left_scale x x' y h i idx hi]
      _ = (List.range 16).foldl (fun acc i => acc + x.2 * qmulTerm x' y i idx) 0 := by
          exact foldl_add_mul_comm (List.range 16) x.2 (fun i => qmulTerm x' y i idx)
      _ = x.2 * (List.range 16).foldl (fun acc i => acc + qmulTerm x' y i idx) 0 := by
          rw [(foldl_smul_split (List.range 16) x.2 (fun i => qmulTerm x' y i idx)).symm]
      _ = (List.range 16).foldl (fun acc i => acc + qmulTerm x' y i idx) 0 * x.2 := by
          rw [Int.mul_comm]

theorem qmulCoeff_right_congr (x y y' : QF) (h : QFeq y y') (idx : Nat) (hidx : idx < 16) :
    qmulCoeff x y idx * y'.2 = qmulCoeff x y' idx * y.2 := by
  unfold qmulCoeff
  calc (List.range 16).foldl (fun acc i => acc + qmulTerm x y i idx) 0 * y'.2
      = y'.2 * (List.range 16).foldl (fun acc i => acc + qmulTerm x y i idx) 0 := by
          rw [Int.mul_comm]
      _ = (List.range 16).foldl (fun acc i => acc + y'.2 * qmulTerm x y i idx) 0 := by
          exact (foldl_smul_split (List.range 16) y'.2 (fun i => qmulTerm x y i idx)).symm
      _ = (List.range 16).foldl (fun acc i => acc + qmulTerm x y' i idx * y.2) 0 := by
          apply foldl_add_pointwise (List.range 16)
            (fun i => y'.2 * qmulTerm x y i idx)
            (fun i => qmulTerm x y' i idx * y.2) 0
          intro i mem
          have hi : i < 16 := by simpa using mem
          rw [Int.mul_comm (y'.2), qmulTerm_right_scale x y y' h i idx hi hidx]
      _ = (List.range 16).foldl (fun acc i => acc + y.2 * qmulTerm x y' i idx) 0 := by
          exact foldl_add_mul_comm (List.range 16) y.2 (fun i => qmulTerm x y' i idx)
      _ = y.2 * (List.range 16).foldl (fun acc i => acc + qmulTerm x y' i idx) 0 := by
          rw [(foldl_smul_split (List.range 16) y.2 (fun i => qmulTerm x y' i idx)).symm]
      _ = (List.range 16).foldl (fun acc i => acc + qmulTerm x y' i idx) 0 * y.2 := by
          rw [Int.mul_comm]

theorem qmul_QFeq_congr (x x' y y' : QF) (hx : QFeq x x') (hy : QFeq y y') :
    QFeq (qmul x y) (qmul x' y') := by
  intro idx hidx
  rw [qmul_gi x y idx hidx, qmul_gi x' y' idx hidx]
  have h1 := qmulCoeff_left_congr x x' y hx idx
  have h2 := qmulCoeff_right_congr x' y y' hy idx hidx
  calc qmulCoeff x y idx * (x'.2 * y'.2)
      = (qmulCoeff x y idx * x'.2) * y'.2 := by rw [Int.mul_assoc]
    _ = (qmulCoeff x' y idx * x.2) * y'.2 := by rw [h1]
    _ = qmulCoeff x' y idx * (x.2 * y'.2) := by rw [Int.mul_assoc]
    _ = (qmulCoeff x' y idx * y'.2) * x.2 := by grind
    _ = (qmulCoeff x' y' idx * y.2) * x.2 := by rw [h2]
    _ = qmulCoeff x' y' idx * (x.2 * y.2) := by grind

/-! ## Additive inverse up to `QFeq` -/

theorem qsub_qzero_left_coeff (x : QF) (i : Nat) (hi : i < 16) :
    gi (qsub qzero x).1 i = 0 - gi x.1 i := by
  rw [qsub_gi qzero x i hi, qzero_coeff i hi]
  simp [qzero, Int.zero_mul, Int.sub_zero, Int.mul_one]

theorem qadd_zero_left_QFeq (x : QF) (_hx : qfLen16 x) : QFeq (qadd qzero x) x := by
  intro i hi
  rw [qadd_gi qzero x i hi, Int.add_mul, qzero_coeff i hi]
  simp [qadd, qzero, Int.zero_mul, Int.zero_add, Int.mul_one]

theorem qmul_one_left_QFeq (x : QF) (hx : qfLen16 x) : QFeq (qmul qfone x) x := by
  intro i hi
  rw [qmul_gi qfone x i hi, qmulCoeff_one_left x i hi,
      gi_eq_getElem x.1 i (by rw [hx]; exact hi)]
  simp [qmul, qfone, Int.mul_one]

theorem qadd_neg_QFeq (x : QF) (_hx : qfLen16 x) : QFeq (qadd x (qsub qzero x)) qzero := by
  intro i hi
  rw [qadd_gi x (qsub qzero x) i hi, qsub_qzero_left_coeff x i hi, qzero_coeff i hi]
  simp only [qadd, qsub, qzero, Int.mul_sub, Int.zero_mul, Int.mul_zero]
  rw [Int.zero_sub, Int.neg_mul]
  grind

/-! ## Distributivity up to `QFeq` -/

theorem qmulTerm_add_right (x y z : QF) (i idx : Nat) (hi : i < 16) (hidx : idx < 16) :
    qmulTerm x (qadd y z) i idx =
      z.2 * qmulTerm x y i idx + y.2 * qmulTerm x z i idx := by
  unfold qmulTerm
  have hj := xor_lt_16 hi hidx
  simp only [Nat.xor_eq]
  rw [qadd_gi y z (i ^^^ idx) hj]
  simp [Int.mul_add, Int.mul_assoc, Int.mul_comm, Int.mul_left_comm]

theorem qmulCoeff_add_right (x y z : QF) (idx : Nat) (hidx : idx < 16) :
    qmulCoeff x (qadd y z) idx =
      z.2 * qmulCoeff x y idx + y.2 * qmulCoeff x z idx := by
  unfold qmulCoeff
  rw [foldl_add_pointwise (List.range 16) _ _ 0
      fun i mem => qmulTerm_add_right x y z i idx (by simpa using mem) hidx]
  rw [foldl_add_split (List.range 16)
      (fun i => z.2 * qmulTerm x y i idx) (fun i => y.2 * qmulTerm x z i idx)]
  rw [foldl_smul_split (List.range 16) z.2 (fun i => qmulTerm x y i idx),
      foldl_smul_split (List.range 16) y.2 (fun i => qmulTerm x z i idx)]

theorem qmulTerm_add_left (x y z : QF) (i idx : Nat) (hi : i < 16) :
    qmulTerm (qadd x y) z i idx =
      y.2 * qmulTerm x z i idx + x.2 * qmulTerm y z i idx := by
  unfold qmulTerm
  rw [qadd_gi x y i hi]
  simp [Int.mul_add, Int.mul_assoc, Int.mul_comm, Int.mul_left_comm, Int.add_mul]

theorem qmulCoeff_add_left (x y z : QF) (idx : Nat) (hidx : idx < 16) :
    qmulCoeff (qadd x y) z idx =
      y.2 * qmulCoeff x z idx + x.2 * qmulCoeff y z idx := by
  unfold qmulCoeff
  rw [foldl_add_pointwise (List.range 16) _ _ 0
      fun i mem => qmulTerm_add_left x y z i idx (by simpa using mem)]
  rw [foldl_add_split (List.range 16)
      (fun i => y.2 * qmulTerm x z i idx) (fun i => x.2 * qmulTerm y z i idx)]
  rw [foldl_smul_split (List.range 16) y.2 (fun i => qmulTerm x z i idx),
      foldl_smul_split (List.range 16) x.2 (fun i => qmulTerm y z i idx)]

theorem qmul_left_distrib_QFeq (x y z : QF) :
    QFeq (qmul x (qadd y z)) (qadd (qmul x y) (qmul x z)) := by
  intro idx hidx
  rw [qmul_gi x (qadd y z) idx hidx,
      qadd_gi (qmul x y) (qmul x z) idx hidx,
      qmul_gi x y idx hidx, qmul_gi x z idx hidx,
      qmulCoeff_add_right x y z idx hidx]
  simp only [qadd, qmul]
  grind

theorem qmul_right_distrib_QFeq (x y z : QF) :
    QFeq (qmul (qadd x y) z) (qadd (qmul x z) (qmul y z)) := by
  intro idx hidx
  rw [qmul_gi (qadd x y) z idx hidx,
      qadd_gi (qmul x z) (qmul y z) idx hidx,
      qmul_gi x z idx hidx, qmul_gi y z idx hidx,
      qmulCoeff_add_left x y z idx hidx]
  simp only [qadd, qmul]
  grind

/-! ## Quotient operations -/

theorem qaddQfp_congr (a b c d : QFp) (h1 : a ≈ c) (h2 : b ≈ d) :
    qaddQfp a b ≈ qaddQfp c d :=
  qadd_QFeq_congr a.1 c.1 b.1 d.1 h1 h2

theorem qmulQfp_congr (a b c d : QFp) (h1 : a ≈ c) (h2 : b ≈ d) :
    qmulQfp a b ≈ qmulQfp c d :=
  qmul_QFeq_congr a.1 c.1 b.1 d.1 h1 h2

theorem qsubQfp_congr (a b c d : QFp) (h1 : a ≈ c) (h2 : b ≈ d) :
    qsubQfp a b ≈ qsubQfp c d :=
  qsub_QFeq_congr a.1 c.1 b.1 d.1 h1 h2

def qaddQ : QQuotient → QQuotient → QQuotient := fun x y =>
  Quotient.liftOn₂ x y
    (fun a b => Quotient.mk _ (qaddQfp a b))
    fun a b c d h1 h2 => Quotient.sound (qaddQfp_congr a b c d h1 h2)

def qmulQ : QQuotient → QQuotient → QQuotient := fun x y =>
  Quotient.liftOn₂ x y
    (fun a b => Quotient.mk _ (qmulQfp a b))
    fun a b c d h1 h2 => Quotient.sound (qmulQfp_congr a b c d h1 h2)

def qsubQ : QQuotient → QQuotient → QQuotient := fun x y =>
  Quotient.liftOn₂ x y
    (fun a b => Quotient.mk _ (qsubQfp a b))
    fun a b c d h1 h2 => Quotient.sound (qsubQfp_congr a b c d h1 h2)

def qnegQ : QQuotient → QQuotient := fun x =>
  Quotient.liftOn x
    (fun a => Quotient.mk _ (qnegQfp a))
    fun a c h => Quotient.sound (qsub_QFeq_congr qzero qzero a.1 c.1 (QFeq.refl qzero) h)

def qzeroQ : QQuotient := Quotient.mk _ QFp.zero

def qoneQ : QQuotient := Quotient.mk _ QFp.one

/-! ## CommRing-style bundle (associativity of `qmul` still open on raw `QF`) -/

/-- Multiplication associativity on raw `QF` (triple-sum XOR reindex; still open). -/
def QmulAssocObligation : Prop :=
  ∀ x y z : QF, qmul (qmul x y) z = qmul x (qmul y z)

structure QCommRingBundle where
  carrier : Type := QQuotient
  add : QQuotient → QQuotient → QQuotient := qaddQ
  mul : QQuotient → QQuotient → QQuotient := qmulQ
  neg : QQuotient → QQuotient := qnegQ
  sub : QQuotient → QQuotient → QQuotient := qsubQ
  zero : QQuotient := qzeroQ
  one : QQuotient := qoneQ
  add_comm : ∀ a b : QQuotient, qaddQ a b = qaddQ b a
  add_assoc : ∀ a b c : QQuotient, qaddQ (qaddQ a b) c = qaddQ a (qaddQ b c)
  add_zero : ∀ a : QQuotient, qaddQ qzeroQ a = a
  neg_add : ∀ a : QQuotient, qaddQ a (qnegQ a) = qzeroQ
  mul_comm : ∀ a b : QQuotient, qmulQ a b = qmulQ b a
  mul_one : ∀ a : QQuotient, qmulQ qoneQ a = a
  left_distrib : ∀ a b c : QQuotient, qmulQ a (qaddQ b c) = qaddQ (qmulQ a b) (qmulQ a c)
  right_distrib : ∀ a b c : QQuotient, qmulQ (qaddQ a b) c = qaddQ (qmulQ a c) (qmulQ b c)

def qCommRing : QCommRingBundle where
  add_comm := fun a b =>
    Quotient.ind₂ (motive := fun x y => qaddQ x y = qaddQ y x)
      (fun x y => Quotient.sound (fun i hi => by
        dsimp [qaddQfp, QFp.mk]
        rw [qadd_gi x.1 y.1 i hi, qadd_gi y.1 x.1 i hi, Int.add_comm]
        simp [qadd, Int.mul_comm]))
      a b
  add_assoc := fun a b c =>
    Quotient.inductionOn₃ a b c
      (motive := fun x y z => qaddQ (qaddQ x y) z = qaddQ x (qaddQ y z))
      fun x y z => Quotient.sound (fun i hi => by
        dsimp [qaddQfp, QFp.mk]
        rw [qadd_gi (qadd x.1 y.1) z.1 i hi, qadd_gi x.1 (qadd y.1 z.1) i hi,
            qadd_gi x.1 y.1 i hi, qadd_gi y.1 z.1 i hi]
        simp [qadd]
        grind)
  add_zero := fun a =>
    Quotient.ind (motive := fun x => qaddQ qzeroQ x = x) (fun x =>
      Quotient.sound (qadd_zero_left_QFeq x.1 x.2.2)) a
  neg_add := fun a =>
    Quotient.ind (motive := fun x => qaddQ x (qnegQ x) = qzeroQ) (fun x =>
      Quotient.sound (qadd_neg_QFeq x.1 x.2.2)) a
  mul_comm := fun a b =>
    Quotient.ind₂ (motive := fun x y => qmulQ x y = qmulQ y x)
      (fun x y => Quotient.sound (fun i hi => by
        dsimp [qmulQfp, QFp.mk]
        rw [qmul_gi x.1 y.1 i hi, qmul_gi y.1 x.1 i hi, qmulCoeff_comm x.1 y.1 i hi]
        simp [qmul, Int.mul_comm])) a b
  mul_one := fun a =>
    Quotient.ind (motive := fun x => qmulQ qoneQ x = x) (fun x =>
      Quotient.sound (qmul_one_left_QFeq x.1 x.2.2)) a
  left_distrib := fun a b c =>
    Quotient.inductionOn₃ a b c
      (motive := fun x y z => qmulQ x (qaddQ y z) = qaddQ (qmulQ x y) (qmulQ x z))
      fun x y z => Quotient.sound (qmul_left_distrib_QFeq x.1 y.1 z.1)
  right_distrib := fun a b c =>
    Quotient.inductionOn₃ a b c
      (motive := fun x y z => qmulQ (qaddQ x y) z = qaddQ (qmulQ x z) (qmulQ y z))
      fun x y z => Quotient.sound (qmul_right_distrib_QFeq x.1 y.1 z.1)

#print axioms QFeq.trans
#print axioms qadd_QFeq_congr
#print axioms qmul_QFeq_congr
#print axioms qadd_neg_QFeq
#print axioms qmul_left_distrib_QFeq
#print axioms qmul_right_distrib_QFeq
#print axioms qCommRing

#eval IO.println "SounioMultiquadQuotient: QFp + QFeq Setoid + quotient ops; PROVED congruence, neg, distrib; QmulAssocObligation staged."

end MultiquadQuotient
