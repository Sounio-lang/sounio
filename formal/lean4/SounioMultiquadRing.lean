set_option maxHeartbeats 0

/-!
  SounioMultiquadRing.lean

  Mathlib-free, ℝ-free groundwork: prove whichever commutative-ring laws of the
  QF kernel (from SounioDeGreyUnitDistance.lean) are tractable in core Lean.

  The QF definitions below are copied verbatim from
  `formal/lean4/SounioDeGreyUnitDistance.lean` lines 24–44 (no import — that module
  triggers a ~3-minute native_decide build).
-/

namespace MultiquadRing

open List

/-- Copied from SounioDeGreyUnitDistance.lean line 24. -/
abbrev QF := List Int × Int

/-- Copied from SounioDeGreyUnitDistance.lean lines 27–31. -/
def bcoeff (m : Nat) : Int :=
  (if m % 2 = 1 then 3 else 1)
  * (if (m / 2) % 2 = 1 then 5 else 1)
  * (if (m / 4) % 2 = 1 then 7 else 1)
  * (if (m / 8) % 2 = 1 then 11 else 1)

/-- Copied from SounioDeGreyUnitDistance.lean line 33. -/
def gi (l : List Int) (i : Nat) : Int := l.getD i 0

/-- Copied from SounioDeGreyUnitDistance.lean lines 35–36. -/
def qsub (x y : QF) : QF :=
  ((List.range 16).map (fun i => gi x.1 i * y.2 - gi y.1 i * x.2), x.2 * y.2)

/-- Copied from SounioDeGreyUnitDistance.lean lines 38–39. -/
def qadd (x y : QF) : QF :=
  ((List.range 16).map (fun i => gi x.1 i * y.2 + gi y.1 i * x.2), x.2 * y.2)

/-- Copied from SounioDeGreyUnitDistance.lean lines 41–45. -/
def qmul (x y : QF) : QF :=
  ((List.range 16).map (fun idx =>
      (List.range 16).foldl (fun acc i =>
          acc + gi x.1 i * gi y.1 (Nat.xor i idx) * bcoeff (Nat.land i (Nat.xor i idx))) 0),
   x.2 * y.2)

/-- Canonical additive zero: all basis coefficients 0, denominator 1. -/
def qzero : QF := (List.replicate 16 0, 1)

/-- De Grey coordinates carry all 16 basis coefficients; `qadd`/`qmul` output length 16. -/
def qfLen16 (x : QF) : Prop := x.1.length = 16

/-- One summand inside the `qmul` fold (indexed by `i`, output mask `idx`). -/
def qmulTerm (x y : QF) (i idx : Nat) : Int :=
  gi x.1 i * gi y.1 (Nat.xor i idx) * bcoeff (Nat.land i (Nat.xor i idx))

/-- Coefficient at basis mask `idx` before pairing with the denominator. -/
def qmulCoeff (x y : QF) (idx : Nat) : Int :=
  (List.range 16).foldl (fun acc i => acc + qmulTerm x y i idx) 0

theorem qadd_list_len (x y : QF) : (qadd x y).1.length = 16 := by
  simp [qadd, List.length_map, List.length_range]

theorem qmul_list_len (x y : QF) : (qmul x y).1.length = 16 := by
  simp [qmul, List.length_map, List.length_range]

theorem qadd_getElem (x y : QF) (i : Nat) (hi : i < 16) :
    (qadd x y).1[i]'(by rw [qadd_list_len]; exact hi) = gi x.1 i * y.2 + gi y.1 i * x.2 := by
  simp [qadd, gi, List.getElem_map, List.getElem_range (by simp; exact hi)]

theorem gi_eq_getElem (l : List Int) (i : Nat) (hi : i < l.length) : gi l i = l[i]'hi := by
  unfold gi
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi, Option.getD_some]

theorem qmul_getElem (x y : QF) (idx : Nat) (hidx : idx < 16) :
    (qmul x y).1[idx]'(by rw [qmul_list_len]; exact hidx) = qmulCoeff x y idx := by
  simp [qmul, qmulCoeff, qmulTerm, gi, List.getElem_map, List.getElem_range (by simp; exact hidx)]

theorem qzero_coeff (i : Nat) (hi : i < 16) : gi qzero.1 i = 0 := by
  unfold qzero gi
  rw [List.getD_eq_getElem?_getD, List.getElem?_replicate, if_pos hi, Option.getD_some]

theorem qmulTerm_symm (x y : QF) (i idx : Nat) :
    qmulTerm x y i idx = qmulTerm y x (Nat.xor i idx) idx := by
  unfold qmulTerm
  have hk : (i ^^^ idx) ^^^ idx = i := by rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
  show gi x.1 i * gi y.1 (i ^^^ idx) * bcoeff (i &&& (i ^^^ idx)) =
       gi y.1 (i ^^^ idx) * gi x.1 ((i ^^^ idx) ^^^ idx) * bcoeff ((i ^^^ idx) &&& ((i ^^^ idx) ^^^ idx))
  rw [hk, Nat.and_comm]
  simp [Int.mul_comm]

theorem perm_range_xor (idx : Nat) (hidx : idx < 16) :
    (List.range 16).map (fun i => i ^^^ idx) ~ List.range 16 := by
  match idx with
  | 0 => native_decide
  | 1 => native_decide
  | 2 => native_decide
  | 3 => native_decide
  | 4 => native_decide
  | 5 => native_decide
  | 6 => native_decide
  | 7 => native_decide
  | 8 => native_decide
  | 9 => native_decide
  | 10 => native_decide
  | 11 => native_decide
  | 12 => native_decide
  | 13 => native_decide
  | 14 => native_decide
  | 15 => native_decide
  | idx + 16 => omega

theorem foldl_add_pointwise {α} (l : List α) (f g : α → Int) (init : Int)
    (h : ∀ a, a ∈ l → f a = g a) :
    l.foldl (fun acc a => acc + f a) init = l.foldl (fun acc a => acc + g a) init := by
  induction l generalizing init with
  | nil => rfl
  | cons a l ih =>
    simp only [List.foldl]
    rw [h a (by simp)]
    exact ih _ fun b hb => h b (by simp [hb])

theorem qmulCoeff_comm (x y : QF) (idx : Nat) (hidx : idx < 16) :
    qmulCoeff x y idx = qmulCoeff y x idx := by
  unfold qmulCoeff
  have hmap := List.foldl_map (l := List.range 16) (f := fun i => i ^^^ idx)
    (g := fun acc k => acc + qmulTerm y x k idx) (init := (0 : Int))
  calc
    (List.range 16).foldl (fun acc i => acc + qmulTerm x y i idx) 0
        = (List.range 16).foldl (fun acc i => acc + qmulTerm y x (i ^^^ idx) idx) 0 :=
          foldl_add_pointwise _ _ _ _ fun i _ => qmulTerm_symm x y i idx
    _ = ((List.range 16).map (fun i => i ^^^ idx)).foldl (fun acc k => acc + qmulTerm y x k idx) 0 :=
          hmap.symm
    _ = (List.range 16).foldl (fun acc k => acc + qmulTerm y x k idx) 0 :=
          Perm.foldl_eq' (perm_range_xor idx hidx)
            (fun a ha b hb z =>
              by simp [Int.add_assoc, Int.add_comm (a := qmulTerm y x b idx)]) 0

theorem qadd_comm (x y : QF) : qadd x y = qadd y x := by
  apply Prod.ext
  · apply List.ext_getElem?
    intro i
    by_cases hi : i < 16
    · have hlen : i < (qadd x y).1.length := by rw [qadd_list_len]; exact hi
      have hlen' : i < (qadd y x).1.length := by rw [qadd_list_len]; exact hi
      rw [List.getElem?_eq_getElem hlen, List.getElem?_eq_getElem hlen']
      rw [qadd_getElem x y i hi, qadd_getElem y x i hi, Int.add_comm]
    · have hlen : (qadd x y).1.length ≤ i := by rw [qadd_list_len]; exact Nat.le_of_not_lt hi
      rw [List.getElem?_eq_none (by simpa using hlen), List.getElem?_eq_none (by rw [qadd_list_len]; simpa using hlen)]
  · exact Int.mul_comm x.2 y.2

theorem qadd_zero_left (x : QF) (hx : qfLen16 x) : qadd qzero x = x := by
  apply Prod.ext
  · apply List.ext_getElem?
    intro i
    by_cases hi : i < 16
    · have hlen : i < (qadd qzero x).1.length := by rw [qadd_list_len]; exact hi
      have hlen' : i < x.1.length := by rw [hx]; exact hi
      rw [List.getElem?_eq_getElem hlen, List.getElem?_eq_getElem hlen']
      rw [qadd_getElem qzero x i hi, qzero_coeff i hi, Int.zero_mul, Int.zero_add, qzero]
      simp only [Option.some.injEq, Int.mul_one]
      exact gi_eq_getElem x.1 i hlen'
    · have hlen : (qadd qzero x).1.length ≤ i := by rw [qadd_list_len]; exact Nat.le_of_not_lt hi
      rw [List.getElem?_eq_none (by simpa using hlen), List.getElem?_eq_none (by rw [hx]; simpa using hlen)]
  · simp [qadd, qzero]

theorem qadd_zero_right (x : QF) (hx : qfLen16 x) : qadd x qzero = x := by
  rw [qadd_comm, qadd_zero_left x hx]

theorem qmul_comm (x y : QF) : qmul x y = qmul y x := by
  apply Prod.ext
  · apply List.ext_getElem?
    intro d
    by_cases hd : d < 16
    · have hlen : d < (qmul x y).1.length := by rw [qmul_list_len]; exact hd
      have hlen' : d < (qmul y x).1.length := by rw [qmul_list_len]; exact hd
      rw [List.getElem?_eq_getElem hlen, List.getElem?_eq_getElem hlen']
      rw [qmul_getElem x y d hd, qmul_getElem y x d hd, qmulCoeff_comm x y d hd]
    · have hlen : (qmul x y).1.length ≤ d := by rw [qmul_list_len]; exact Nat.le_of_not_lt hd
      rw [List.getElem?_eq_none (by simpa using hlen), List.getElem?_eq_none (by rw [qmul_list_len]; simpa using hlen)]
  · exact Int.mul_comm x.2 y.2

/-- Open obligation: associativity of `qmul` (heavy sum reindexing; no `ring` / BigOperators). -/
def QmulAssocObligation : Prop :=
  ∀ x y z : QF, qmul (qmul x y) z = qmul x (qmul y z)

/-- Open obligation: left distributivity of `qmul` over `qadd`. -/
def QmulLeftDistribObligation : Prop :=
  ∀ x y z : QF, qmul x (qadd y z) = qadd (qmul x y) (qmul x z)

/-- Open obligation: right distributivity of `qmul` over `qadd`. -/
def QmulRightDistribObligation : Prop :=
  ∀ x y z : QF, qmul (qadd x y) z = qadd (qmul x z) (qmul y z)

/-- Open obligation: additive inverse to `qzero` on length-16 representatives. -/
def QaddNegObligation : Prop :=
  ∀ x : QF, qfLen16 x → qadd x (qsub qzero x) = qzero

/-- Open obligation: existence of a multiplicative unit for `qmul`. -/
def QmulOneObligation : Prop :=
  ∃ one : QF, (∀ x : QF, qmul one x = x) ∧ (∀ x : QF, qmul x one = x)

#print axioms qadd_comm
#print axioms qadd_zero_left
#print axioms qadd_zero_right
#print axioms qmul_comm

#eval IO.println "SounioMultiquadRing: PROVED qadd_comm, qadd_zero_left, qadd_zero_right, qmul_comm; OPEN QmulAssocObligation, Qmul{Left,Right}DistribObligation, QaddNegObligation, QmulOneObligation."

end MultiquadRing
