set_option maxHeartbeats 0
set_option linter.unusedSimpArgs false

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

@[simp]
theorem qadd_list_len (x y : QF) : (qadd x y).1.length = 16 := by
  simp [qadd, List.length_map, List.length_range]

@[simp]
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

/-! ## Multiquadratic generator law — QF *is* ℚ(√3,√5,√7,√11) at the basis level

The 16 basis elements `basis m` are the squarefree radical monomials
`√(∏ primes selected by m)` over {3,5,7,11}. The defining algebraic fact of the
multiquadratic ring is that they multiply by *adding masks* (XOR) while the shared
primes (AND) come out as the rational coefficient `bcoeff (i &&& j)`:

    √a · √b  =  (∏ shared primes) · √(symmetric difference).

We certify this **exactly** (no floats, no Mathlib) by exhaustion over all 16×16
basis pairs. This is precisely the relation a real embedding `basis m ↦ ∏√pⱼ ∈ ℝ`
must respect; it fixes the multiplication table on the *generators*. (This alone is
NOT a full ring/field certification — `qmul` associativity, distributivity and
inverses remain open obligations below — but it is the generator-level law the
eventual `QF ↪ ℝ` map is built on.) -/

/-- The `m`-th multiquadratic basis radical, denominator 1. -/
def basis (m : Nat) : QF := ((List.range 16).map (fun i => if i = m then (1 : Int) else 0), 1)

/-- Multiply all numerator coefficients of a QF value by an integer scalar. -/
def scaleC (c : Int) (x : QF) : QF := (x.1.map (· * c), x.2)

/-- **Multiquadratic generator law (exact).** For every pair of basis radicals,
`√i · √j = bcoeff(i ∧ j) · √(i ⊕ j)`. Verified for all 256 basis pairs. -/
theorem basis_mul_law :
    ∀ i ∈ List.range 16, ∀ j ∈ List.range 16,
      qmul (basis i) (basis j) = scaleC (bcoeff (Nat.land i j)) (basis (Nat.xor i j)) := by
  native_decide

/-- `√3 · √3 = 3`  (basis 1, shared prime 3, mask 1⊕1 = 0 ⇒ rational). -/
theorem sqrt3_sq  : qmul (basis 1) (basis 1) = scaleC 3 (basis 0) := by native_decide
/-- `√5 · √5 = 5`. -/
theorem sqrt5_sq  : qmul (basis 2) (basis 2) = scaleC 5 (basis 0) := by native_decide
/-- `√7 · √7 = 7`. -/
theorem sqrt7_sq  : qmul (basis 4) (basis 4) = scaleC 7 (basis 0) := by native_decide
/-- `√11 · √11 = 11`. -/
theorem sqrt11_sq : qmul (basis 8) (basis 8) = scaleC 11 (basis 0) := by native_decide
/-- `√3 · √5 = √15`  (no shared prime ⇒ mask 1⊕2 = 3, coefficient 1). -/
theorem sqrt3_sqrt5  : qmul (basis 1) (basis 2) = basis 3 := by native_decide
/-- `√3 · √15 = 3·√5`  (shared prime 3: mask 1∧3 = 1 ⇒ bcoeff 3; 1⊕3 = 2 = √5). -/
theorem sqrt3_sqrt15 : qmul (basis 1) (basis 3) = scaleC 3 (basis 2) := by native_decide
/-- `√15 · √35 = 5·√21`  (shared prime 5: 3∧6 = 2 ⇒ bcoeff 5; 3⊕6 = 5 = √21). -/
theorem sqrt15_sqrt35 : qmul (basis 3) (basis 6) = scaleC 5 (basis 5) := by native_decide

/-! ## More certified ring laws (no Mathlib): additive associativity + multiplicative unit -/

theorem qadd_assoc (x y z : QF) : qadd (qadd x y) z = qadd x (qadd y z) := by
  apply Prod.ext
  · apply List.ext_getElem?
    intro i
    by_cases hi : i < 16
    · have h1 : i < (qadd (qadd x y) z).1.length := by rw [qadd_list_len]; exact hi
      have h2 : i < (qadd x (qadd y z)).1.length := by rw [qadd_list_len]; exact hi
      have hxy : i < (qadd x y).1.length := by rw [qadd_list_len]; exact hi
      have hyz : i < (qadd y z).1.length := by rw [qadd_list_len]; exact hi
      rw [List.getElem?_eq_getElem h1, List.getElem?_eq_getElem h2,
          qadd_getElem (qadd x y) z i hi, qadd_getElem x (qadd y z) i hi,
          gi_eq_getElem (qadd x y).1 i hxy, gi_eq_getElem (qadd y z).1 i hyz,
          qadd_getElem x y i hi, qadd_getElem y z i hi]
      simp only [Option.some.injEq]
      show (gi x.1 i * y.2 + gi y.1 i * x.2) * z.2 + gi z.1 i * (x.2 * y.2)
         = gi x.1 i * (y.2 * z.2) + (gi y.1 i * z.2 + gi z.1 i * y.2) * x.2
      simp only [Int.add_mul, Int.mul_add, Int.mul_assoc, Int.mul_comm, Int.mul_left_comm,
                 Int.add_assoc, Int.add_left_comm]
    · have hlen : (qadd (qadd x y) z).1.length ≤ i := by rw [qadd_list_len]; exact Nat.le_of_not_lt hi
      rw [List.getElem?_eq_none (by simpa using hlen),
          List.getElem?_eq_none (by rw [qadd_list_len]; simpa using hlen)]
  · show (x.2 * y.2) * z.2 = x.2 * (y.2 * z.2)
    exact Int.mul_assoc x.2 y.2 z.2

/-- Canonical multiplicative unit: rational `1`, denominator `1`. -/
def qfone : QF := ((List.range 16).map (fun i => if i = 0 then (1 : Int) else 0), 1)

theorem qfone_coeff (i : Nat) (hi : i < 16) : gi qfone.1 i = if i = 0 then 1 else 0 := by
  unfold qfone gi
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by simp [hi]),
      List.getElem_map, List.getElem_range, Option.getD_some]

theorem qmulCoeff_one_left (x : QF) (idx : Nat) (hidx : idx < 16) :
    qmulCoeff qfone x idx = gi x.1 idx := by
  unfold qmulCoeff qmulTerm
  -- only the `i = 0` summand is nonzero: `gi qfone.1 i = 0` for `i ≠ 0`
  have key : ∀ i ∈ List.range 16,
      gi qfone.1 i * gi x.1 (Nat.xor i idx) * bcoeff (Nat.land i (Nat.xor i idx))
        = (if i = 0 then gi x.1 idx else 0) := by
    intro i hi
    have hi16 : i < 16 := by simpa using hi
    rw [qfone_coeff i hi16]
    by_cases h0 : i = 0
    · subst h0
      simp [Nat.zero_xor, Nat.zero_and, bcoeff]
    · simp [h0]
  rw [foldl_add_pointwise (List.range 16) _ (fun i => if i = 0 then gi x.1 idx else 0) 0 key]
  rw [show List.range 16 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] from by decide]
  simp

theorem qmul_one_left (x : QF) (hx : qfLen16 x) : qmul qfone x = x := by
  apply Prod.ext
  · apply List.ext_getElem?
    intro i
    by_cases hi : i < 16
    · have hlen : i < (qmul qfone x).1.length := by rw [qmul_list_len]; exact hi
      have hlen' : i < x.1.length := by rw [hx]; exact hi
      rw [List.getElem?_eq_getElem hlen, List.getElem?_eq_getElem hlen']
      rw [qmul_getElem qfone x i hi, qmulCoeff_one_left x i hi, gi_eq_getElem x.1 i hlen']
    · have hlen : (qmul qfone x).1.length ≤ i := by rw [qmul_list_len]; exact Nat.le_of_not_lt hi
      rw [List.getElem?_eq_none (by simpa using hlen),
          List.getElem?_eq_none (by rw [hx]; simpa using hlen)]
  · show qfone.2 * x.2 = x.2
    simp [qfone]

theorem qmul_one_right (x : QF) (hx : qfLen16 x) : qmul x qfone = x := by
  rw [qmul_comm, qmul_one_left x hx]

/-- Discharges `QmulOneObligation` on the length-16 representatives that carry the
    de Grey coordinates (the only ones `qadd`/`qmul` ever produce). -/
theorem qmul_one_obligation :
    ∀ x : QF, qfLen16 x → qmul qfone x = x ∧ qmul x qfone = x :=
  fun x hx => ⟨qmul_one_left x hx, qmul_one_right x hx⟩

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

/-- DISCHARGED (on length-16 representatives) by `qmul_one_obligation` above:
    `qfone` is a two-sided multiplicative unit for every `qfLen16` value. The unrestricted
    `∀ x` form is false only for malformed (non-length-16) representatives, which `qadd`/`qmul`
    never produce. -/
def QmulOneObligation : Prop :=
  ∃ one : QF, (∀ x : QF, qfLen16 x → qmul one x = x) ∧ (∀ x : QF, qfLen16 x → qmul x one = x)

/-- `QmulOneObligation` is now a theorem (witness `qfone`). -/
theorem qmulOne_solved : QmulOneObligation :=
  ⟨qfone, fun x hx => qmul_one_left x hx, fun x hx => qmul_one_right x hx⟩

#print axioms qadd_comm
#print axioms qadd_assoc
#print axioms qadd_zero_left
#print axioms qadd_zero_right
#print axioms qmul_comm
#print axioms qmul_one_left
#print axioms qmulOne_solved
#print axioms basis_mul_law
#print axioms sqrt3_sq

#eval IO.println "SounioMultiquadRing: PROVED qadd_{comm,assoc,zero_left,zero_right}, qmul_{comm,one_left,one_right} (QmulOneObligation discharged on length-16 reps), basis_mul_law (multiquadratic generator law, all 256 basis pairs) + √pᵢ²=pᵢ & cross-products; OPEN QmulAssocObligation, Qmul{Left,Right}DistribObligation, QaddNegObligation."

end MultiquadRing
