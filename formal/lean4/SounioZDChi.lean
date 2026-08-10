/-
  SounioZDChi — the commutation sign of the CD cocycle, in closed form.

  `chi(x,y) = sigma(x,y) * sigma(y,x)` is `+1` exactly when one argument is zero or the two
  coincide, and `-1` otherwise. All four ingredients were already proven in
  `SounioZDFiberAntisym.lean` -- `cdSig0`, `cdSig0'`, `cdSq`, `antisym` -- and are carried here
  verbatim from the committed version; this file just puts them together.

  WHY IT MATTERS. The symmetry `Qgen L a b = Qgen L b a` (measured unconditionally, contract
  K19) is exactly the statement that a product of four such signs is `+1`; with `chi_char` that
  becomes a finite parity argument. That symmetry collapses the gap root `a = H` into the
  already-proven `b = H`.

  WHY IT IS A SEPARATE FILE. `SounioZDFiberAntisym.lean` is being edited concurrently by another
  agent working the same two gap roots. An earlier version of `chi_char` was appended there and
  was overwritten by their next write before it could be committed. This file touches nothing of
  theirs.

  NOT CLAIMED: the symmetry itself. Its parity case analysis needs the dependencies between the
  six degeneracy conditions (`a = 0` and `a = L` are mutually exclusive when `L != 0`, and so on)
  -- XOR facts that neither `simp` nor `omega` discharges. `chi_char` is the INPUT to that
  argument, not the argument.

  Mathlib-free, no `sorry`, no `native_decide`; check with `#print axioms`.
  (`tauto` does not exist without Mathlib -- an explicit `rintro` is the replacement.)
-/

namespace SounioZDChi

def cdSigma (a b : Nat) : Nat → Int
  | 0 => -1
  | 1 => if a == 0 || b == 0 then 1 else -1
  | (n+2) =>
      if a == 0 || b == 0 then 1
      else
        let half := 2 ^ (n+1)
        if !(a ≥ half) && !(b ≥ half) then cdSigma (a%half) (b%half) (n+1)
        else if !(a ≥ half) && (b ≥ half) then cdSigma (b%half) (a%half) (n+1)
        else if (a ≥ half) && !(b ≥ half) then (if b%half == 0 then cdSigma (a%half) 0 (n+1) else - cdSigma (a%half) (b%half) (n+1))
        else (if b%half == 0 then - cdSigma 0 (a%half) (n+1) else cdSigma (b%half) (a%half) (n+1))

theorem cdSig0 (b m : Nat) : cdSigma 0 b (m+1) = 1 := by cases m <;> simp [cdSigma]

theorem cdSig0' (a m : Nat) : cdSigma a 0 (m+1) = 1 := by cases m <;> simp [cdSigma]

/-- R, both-lower branch (unconditional). -/

theorem R_ll (u v n : Nat) (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) :
    cdSigma u v (n+2) = cdSigma u v (n+1) := by
  by_cases hu0 : u = 0
  · subst hu0; simp [cdSig0]
  by_cases hv0 : v = 0
  · subst hv0; simp [cdSig0']
  have e1 : (u == 0) = false := by simp [hu0]
  have e2 : (v == 0) = false := by simp [hv0]
  have h1 : (decide (u ≥ 2^(n+1))) = false := by simp only [decide_eq_false_iff_not]; omega
  have h2 : (decide (v ≥ 2^(n+1))) = false := by simp only [decide_eq_false_iff_not]; omega
  generalize hR : cdSigma u v (n+1) = R
  unfold cdSigma
  simp only [e1, e2, Bool.or_self, h1, h2, Bool.not_false, Bool.and_self,
             Nat.mod_eq_of_lt hu, Nat.mod_eq_of_lt hv, hR, if_true]
  exact if_neg (by decide)

/-- R, swap branch (u lower, v upper; unconditional). -/

theorem R_lu (u v n : Nat) (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) :
    cdSigma u (v + 2^(n+1)) (n+2) = cdSigma v u (n+1) := by
  have hpos : 0 < 2^(n+1) := Nat.two_pow_pos (n+1)
  by_cases hu0 : u = 0
  · subst hu0; simp [cdSig0, cdSig0']
  have e1 : (u == 0) = false := by simp [hu0]
  have eb : ((v + 2^(n+1)) == 0) = false := by have : v + 2^(n+1) ≠ 0 := by omega
                                               simp [this]
  have h1 : (decide (u ≥ 2^(n+1))) = false := by simp only [decide_eq_false_iff_not]; omega
  have h2 : (decide (v + 2^(n+1) ≥ 2^(n+1))) = true := by simp only [decide_eq_true_eq]; omega
  have hmod : (v + 2^(n+1)) % 2^(n+1) = v := by rw [Nat.add_mod_right]; exact Nat.mod_eq_of_lt hv
  generalize hR : cdSigma v u (n+1) = R
  unfold cdSigma
  simp only [e1, eb, Bool.or_self, h1, h2, Bool.not_false, Bool.not_true, Bool.and_false,
             Bool.and_true, Bool.true_and, hmod, Nat.mod_eq_of_lt hu, hR, if_true, if_false]
  exact if_neg (by decide)

/-- R, upper×lower branch. -/

theorem R_ul (u v n : Nat) (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) :
    cdSigma (u + 2^(n+1)) v (n+2) = (if v = 0 then 1 else - cdSigma u v (n+1)) := by
  have hpos : 0 < 2^(n+1) := Nat.two_pow_pos (n+1)
  have ea : ((u + 2^(n+1)) == 0) = false := by have : u + 2^(n+1) ≠ 0 := by omega
                                               simp [this]
  have h1 : (decide (u + 2^(n+1) ≥ 2^(n+1))) = true := by simp only [decide_eq_true_eq]; omega
  have h2 : (decide (v ≥ 2^(n+1))) = false := by simp only [decide_eq_false_iff_not]; omega
  have hma : (u + 2^(n+1)) % 2^(n+1) = u := by rw [Nat.add_mod_right]; exact Nat.mod_eq_of_lt hu
  by_cases hv0 : v = 0
  · subst hv0
    unfold cdSigma
    simp [ea, h1, h2, hma]
  · have e2 : (v == 0) = false := by simp [hv0]
    generalize hR : cdSigma u v (n+1) = R
    unfold cdSigma
    simp [ea, e2, h1, h2, hma, Nat.mod_eq_of_lt hv, hR, hv0]

/-- R, both-upper branch. -/

theorem R_uu (u v n : Nat) (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) :
    cdSigma (u + 2^(n+1)) (v + 2^(n+1)) (n+2) = (if v = 0 then -1 else cdSigma v u (n+1)) := by
  have hpos : 0 < 2^(n+1) := Nat.two_pow_pos (n+1)
  have ea : ((u + 2^(n+1)) == 0) = false := by have : u + 2^(n+1) ≠ 0 := by omega
                                               simp [this]
  have eb : ((v + 2^(n+1)) == 0) = false := by have : v + 2^(n+1) ≠ 0 := by omega
                                               simp [this]
  have h1 : (decide (u + 2^(n+1) ≥ 2^(n+1))) = true := by simp only [decide_eq_true_eq]; omega
  have h2 : (decide (v + 2^(n+1) ≥ 2^(n+1))) = true := by simp only [decide_eq_true_eq]; omega
  have hmb : (v + 2^(n+1)) % 2^(n+1) = v := by rw [Nat.add_mod_right]; exact Nat.mod_eq_of_lt hv
  have hma : (u + 2^(n+1)) % 2^(n+1) = u := by rw [Nat.add_mod_right]; exact Nat.mod_eq_of_lt hu
  by_cases hv0 : v = 0
  · subst hv0
    unfold cdSigma
    simp [ea, eb, h1, h2, hmb, cdSig0]
  · generalize hR : cdSigma v u (n+1) = R
    unfold cdSigma
    simp [ea, eb, h1, h2, hma, hmb, hR, hv0]

/-- Adding the seam bit to a sub-seam value is the same as XORing it. -/

theorem antisym : ∀ (m a b : Nat), a < 2^m → b < 2^m → a ≠ 0 → b ≠ 0 → a ≠ b →
    cdSigma a b m = - cdSigma b a m
  | 0, a, _, ha, _, ha0, _, _ => by
      have : (2:Nat)^0 = 1 := rfl
      omega
  | 1, a, b, ha, hb, ha0, hb0, hab => by
      have : (2:Nat)^1 = 2 := rfl
      omega
  | (n+2), a, b, ha, hb, ha0, hb0, hab => by
    have h2H : 2^(n+2) = 2^(n+1) + 2^(n+1) := by rw [Nat.pow_succ]; omega
    by_cases haU : a ≥ 2^(n+1) <;> by_cases hbU : b ≥ 2^(n+1)
    · obtain ⟨ua, hae, hual⟩ : ∃ ua, a = ua + 2^(n+1) ∧ ua < 2^(n+1) :=
        ⟨a - 2^(n+1), by omega, by omega⟩
      obtain ⟨ub, hbe, hubl⟩ : ∃ ub, b = ub + 2^(n+1) ∧ ub < 2^(n+1) :=
        ⟨b - 2^(n+1), by omega, by omega⟩
      have huab : ua ≠ ub := by omega
      rw [hae, hbe, R_uu ua ub n hual hubl, R_uu ub ua n hubl hual]
      by_cases hub0 : ub = 0
      · by_cases hua0 : ua = 0
        · exfalso; omega
        · simp [hub0, hua0, cdSig0, cdSig0']
      · by_cases hua0 : ua = 0
        · simp [hub0, hua0, cdSig0, cdSig0']
        · simp only [if_neg hub0, if_neg hua0]
          rw [antisym (n+1) ua ub hual hubl hua0 hub0 huab]; omega
    · obtain ⟨ua, hae, hual⟩ : ∃ ua, a = ua + 2^(n+1) ∧ ua < 2^(n+1) :=
        ⟨a - 2^(n+1), by omega, by omega⟩
      have hbl : b < 2^(n+1) := by omega
      rw [hae, R_ul ua b n hual hbl, R_lu b ua n hbl hual, if_neg hb0]
    · obtain ⟨ub, hbe, hubl⟩ : ∃ ub, b = ub + 2^(n+1) ∧ ub < 2^(n+1) :=
        ⟨b - 2^(n+1), by omega, by omega⟩
      have hal : a < 2^(n+1) := by omega
      rw [hbe, R_lu a ub n hal hubl, R_ul ub a n hubl hal, if_neg ha0]; omega
    · have hal : a < 2^(n+1) := by omega
      have hbl : b < 2^(n+1) := by omega
      rw [R_ll a b n hal hbl, R_ll b a n hbl hal]
      exact antisym (n+1) a b hal hbl ha0 hb0 hab

theorem cdSigma_pm : ∀ (m a b : Nat), cdSigma a b m = 1 ∨ cdSigma a b m = -1 := by
  intro m
  induction m with
  | zero => exact fun a b => Or.inr rfl
  | succ k ih =>
    match k, ih with
    | 0, _ => intro a b; unfold cdSigma; by_cases h : a == 0 || b == 0 <;> simp [h]
    | (n+1), ih =>
      intro a b
      unfold cdSigma
      by_cases h : a == 0 || b == 0
      · simp [h]
      · by_cases ha : 2^(n+1) ≤ a <;> by_cases hb : 2^(n+1) ≤ b <;>
          simp only [h, ha, hb, ge_iff_le, Bool.false_eq_true, if_false, decide_true, decide_false,
            decide_not, Bool.not_true, Bool.not_false, Bool.and_self, Bool.and_true, Bool.true_and,
            Bool.and_false, Bool.false_and, if_true]
        · by_cases hb0 : b % 2^(n+1) == 0 <;> simp only [hb0, if_true, if_false]
          · rcases ih 0 (a % 2^(n+1)) with hh | hh <;> simp [hh]
          · exact ih _ _
        · by_cases hb0 : b % 2^(n+1) == 0 <;> simp only [hb0, if_true, if_false]
          · exact ih _ _
          · rcases ih (a % 2^(n+1)) (b % 2^(n+1)) with hh | hh <;> simp [hh]
        · exact ih _ _
        · exact ih _ _

/-- `P1` on the diagonal is `+1`. -/

theorem cdSq (a b m : Nat) : cdSigma a b m * cdSigma a b m = 1 := by
  rcases cdSigma_pm m a b with h | h <;> rw [h] <;> decide

/-- Degenerate branch, first-argument form: `sigma(u^L, L) * sigma(u, L) = -1`. -/

def Qgen (L a b m : Nat) : Int :=
  cdSigma a b m * cdSigma (a ^^^ L) (b ^^^ L) m * cdSigma a (b ^^^ L) m * cdSigma (a ^^^ L) b m

/-- `Q` depends only on the two COSETS `{a, a^L}`, `{b, b^L}` -- the four factors are permuted.
    Used to halve the case analysis in any induction on `Qgen`. -/

theorem chi_char (m x y : Nat) (hx : x < 2^m) (hy : y < 2^m) :
    cdSigma x y m * cdSigma y x m = (if x = 0 ∨ y = 0 ∨ x = y then (1:Int) else -1) := by
  cases m with
  | zero =>
      have : x = 0 := by have : (2:Nat)^0 = 1 := rfl; omega
      simp [this, cdSigma]
  | succ m' =>
      by_cases hx0 : x = 0
      · subst hx0; rw [cdSig0, cdSig0', if_pos (Or.inl rfl)]; decide
      by_cases hy0 : y = 0
      · subst hy0; rw [cdSig0, cdSig0', if_pos (Or.inr (Or.inl rfl))]; decide
      by_cases hxy : x = y
      · subst hxy; rw [if_pos (Or.inr (Or.inr rfl))]; exact cdSq x x (m'+1)
      · have hno : ¬ (x = 0 ∨ y = 0 ∨ x = y) := by
          rintro (h | h | h)
          · exact hx0 h
          · exact hy0 h
          · exact hxy h
        rw [if_neg hno, antisym (m'+1) x y hx hy hx0 hy0 hxy]
        rcases cdSigma_pm (m'+1) y x with h | h <;> rw [h] <;> decide

theorem Qgen_pm (L a b m : Nat) : Qgen L a b m = 1 ∨ Qgen L a b m = -1 := by
  unfold Qgen
  rcases cdSigma_pm m a b with h1 | h1 <;>
    rcases cdSigma_pm m (a ^^^ L) (b ^^^ L) with h2 | h2 <;>
    rcases cdSigma_pm m a (b ^^^ L) with h3 | h3 <;>
    rcases cdSigma_pm m (a ^^^ L) b with h4 | h4 <;>
    rw [h1, h2, h3, h4] <;> decide

/-! ## The symmetry -/

theorem xor_eq_zero_iff (x y : Nat) : x ^^^ y = 0 ↔ x = y := by
  constructor
  · intro h
    have h2 : (x ^^^ y) ^^^ y = 0 ^^^ y := by rw [h]
    rwa [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero, Nat.zero_xor] at h2
  · intro h; rw [h, Nat.xor_self]

theorem xor_right_cancel (x y L : Nat) : x ^^^ L = y ^^^ L ↔ x = y := by
  constructor
  · intro h
    have h2 : (x ^^^ L) ^^^ L = (y ^^^ L) ^^^ L := by rw [h]
    rwa [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero, Nat.xor_assoc, Nat.xor_self,
         Nat.xor_zero] at h2
  · intro h; rw [h]

/-- **`Qgen` is symmetric in its two arguments, unconditionally.**

    The ratio of the two orders is `chi(a,b)·chi(a⊕L,b⊕L)·chi(a,b⊕L)·chi(a⊕L,b)`, and by
    `chi_char` each factor is `+1` unless its arguments are distinct and nonzero. In EVERY
    configuration two of the four conditions coincide, so the product is a square. -/
theorem Qgen_symm (L a b m : Nat) (hL : L < 2^m) (ha : a < 2^m) (hb : b < 2^m) :
    Qgen L a b m = Qgen L b a m := by
  have haL : a ^^^ L < 2^m := Nat.xor_lt_two_pow ha hL
  have hbL : b ^^^ L < 2^m := Nat.xor_lt_two_pow hb hL
  have key : Qgen L a b m * Qgen L b a m
      = (cdSigma a b m * cdSigma b a m)
        * (cdSigma (a ^^^ L) (b ^^^ L) m * cdSigma (b ^^^ L) (a ^^^ L) m)
        * (cdSigma a (b ^^^ L) m * cdSigma (b ^^^ L) a m)
        * (cdSigma (a ^^^ L) b m * cdSigma b (a ^^^ L) m) := by
    unfold Qgen; ac_rfl
  rw [chi_char m a b ha hb, chi_char m (a ^^^ L) (b ^^^ L) haL hbL,
      chi_char m a (b ^^^ L) ha hbL, chi_char m (a ^^^ L) b haL hb] at key
  have hprod : Qgen L a b m * Qgen L b a m = 1 := by
    rw [key]
    by_cases hA : a = 0
    · -- P1, P3 hold by their first disjunct; P2 and P4 both reduce to L=0 ∨ b=0 ∨ b=L
      subst hA
      by_cases h : L = 0 ∨ b = 0 ∨ b = L
      · have p2 : (0:Nat) ^^^ L = 0 ∨ b ^^^ L = 0 ∨ (0:Nat) ^^^ L = b ^^^ L := by
          rcases h with h | h | h
          · exact Or.inl (by rw [Nat.zero_xor, h])
          · exact Or.inr (Or.inr (by rw [h]))
          · exact Or.inr (Or.inl ((xor_eq_zero_iff b L).mpr h))
        have p4 : (0:Nat) ^^^ L = 0 ∨ b = 0 ∨ (0:Nat) ^^^ L = b := by
          rcases h with h | h | h
          · exact Or.inl (by rw [Nat.zero_xor, h])
          · exact Or.inr (Or.inl h)
          · exact Or.inr (Or.inr (by rw [Nat.zero_xor, h]))
        rw [if_pos (Or.inl rfl), if_pos p2, if_pos (Or.inl rfl), if_pos p4]; decide
      · have p2 : ¬ ((0:Nat) ^^^ L = 0 ∨ b ^^^ L = 0 ∨ (0:Nat) ^^^ L = b ^^^ L) := by
          rintro (g | g | g)
          · exact h (Or.inl (by rwa [Nat.zero_xor] at g))
          · exact h (Or.inr (Or.inr ((xor_eq_zero_iff b L).mp g)))
          · exact h (Or.inr (Or.inl ((xor_right_cancel 0 b L).mp g).symm))
        have p4 : ¬ ((0:Nat) ^^^ L = 0 ∨ b = 0 ∨ (0:Nat) ^^^ L = b) := by
          rintro (g | g | g)
          · exact h (Or.inl (by rwa [Nat.zero_xor] at g))
          · exact h (Or.inr (Or.inl g))
          · exact h (Or.inr (Or.inr (by rwa [Nat.zero_xor, eq_comm] at g)))
        rw [if_pos (Or.inl rfl), if_neg p2, if_pos (Or.inl rfl), if_neg p4]; decide
    · by_cases hB : b = 0
      · subst hB
        by_cases h : L = 0 ∨ a = 0 ∨ a = L
        · have p2 : a ^^^ L = 0 ∨ (0:Nat) ^^^ L = 0 ∨ a ^^^ L = (0:Nat) ^^^ L := by
            rcases h with h | h | h
            · exact Or.inr (Or.inl (by rw [Nat.zero_xor, h]))
            · exact Or.inr (Or.inr (by rw [h]))
            · exact Or.inl ((xor_eq_zero_iff a L).mpr h)
          have p3 : a = 0 ∨ (0:Nat) ^^^ L = 0 ∨ a = (0:Nat) ^^^ L := by
            rcases h with h | h | h
            · exact Or.inr (Or.inl (by rw [Nat.zero_xor, h]))
            · exact Or.inl h
            · exact Or.inr (Or.inr (by rw [Nat.zero_xor, h]))
          rw [if_pos (Or.inr (Or.inl rfl)), if_pos p2, if_pos p3,
              if_pos (Or.inr (Or.inl rfl))]; decide
        · have p2 : ¬ (a ^^^ L = 0 ∨ (0:Nat) ^^^ L = 0 ∨ a ^^^ L = (0:Nat) ^^^ L) := by
            rintro (g | g | g)
            · exact h (Or.inr (Or.inr ((xor_eq_zero_iff a L).mp g)))
            · exact h (Or.inl (by rwa [Nat.zero_xor] at g))
            · exact h (Or.inr (Or.inl ((xor_right_cancel a 0 L).mp g)))
          have p3 : ¬ (a = 0 ∨ (0:Nat) ^^^ L = 0 ∨ a = (0:Nat) ^^^ L) := by
            rintro (g | g | g)
            · exact h (Or.inr (Or.inl g))
            · exact h (Or.inl (by rwa [Nat.zero_xor] at g))
            · exact h (Or.inr (Or.inr (by rwa [Nat.zero_xor] at g)))
          rw [if_pos (Or.inr (Or.inl rfl)), if_neg p2, if_neg p3,
              if_pos (Or.inr (Or.inl rfl))]; decide
      · by_cases hA' : a ^^^ L = 0
        · -- a = L.  P2, P4 hold by their first disjunct; P1 and P3 both reduce to a = b
          by_cases h : a = b
          · have p3 : a = 0 ∨ b ^^^ L = 0 ∨ a = b ^^^ L :=
              Or.inr (Or.inl (by rw [← h]; exact hA'))
            rw [if_pos (Or.inr (Or.inr h)), if_pos (Or.inl hA'), if_pos p3,
                if_pos (Or.inl hA')]; decide
          · have p1 : ¬ (a = 0 ∨ b = 0 ∨ a = b) := by
              rintro (g | g | g)
              · exact hA g
              · exact hB g
              · exact h g
            have p3 : ¬ (a = 0 ∨ b ^^^ L = 0 ∨ a = b ^^^ L) := by
              rintro (g | g | g)
              · exact hA g
              · exact h (((xor_eq_zero_iff b L).mp g).trans ((xor_eq_zero_iff a L).mp hA').symm).symm
              · exact hB (by
                  have haL2 : a = L := (xor_eq_zero_iff a L).mp hA'
                  rw [haL2] at g
                  have h2 : b ^^^ L = 0 ^^^ L := by rw [Nat.zero_xor]; exact g.symm
                  exact (xor_right_cancel b 0 L).mp h2)
            rw [if_neg p1, if_pos (Or.inl hA'), if_neg p3, if_pos (Or.inl hA')]; decide
        · by_cases hB' : b ^^^ L = 0
          · -- b = L.  P2, P3 hold; P1 and P4 both reduce to a = b
            by_cases h : a = b
            · have p4 : a ^^^ L = 0 ∨ b = 0 ∨ a ^^^ L = b := Or.inl (by rw [h]; exact hB')
              rw [if_pos (Or.inr (Or.inr h)), if_pos (Or.inr (Or.inl hB')),
                  if_pos (Or.inr (Or.inl hB')), if_pos p4]; decide
            · have p1 : ¬ (a = 0 ∨ b = 0 ∨ a = b) := by
                rintro (g | g | g)
                · exact hA g
                · exact hB g
                · exact h g
              have p4 : ¬ (a ^^^ L = 0 ∨ b = 0 ∨ a ^^^ L = b) := by
                rintro (g | g | g)
                · exact hA' g
                · exact hB g
                · exact hA (by
                    have hbL' : b = L := (xor_eq_zero_iff b L).mp hB'
                    rw [hbL'] at g
                    have h2 : a ^^^ L = 0 ^^^ L := by rw [Nat.zero_xor]; exact g
                    exact (xor_right_cancel a 0 L).mp h2)
              rw [if_neg p1, if_pos (Or.inr (Or.inl hB')), if_pos (Or.inr (Or.inl hB')),
                  if_neg p4]; decide
          · by_cases hE : a = b
            · -- P1, P2 hold; P3 and P4 both reduce to L = 0
              have p2 : a ^^^ L = 0 ∨ b ^^^ L = 0 ∨ a ^^^ L = b ^^^ L :=
                Or.inr (Or.inr (by rw [hE]))
              by_cases hL0 : L = 0
              · subst hL0
                rw [if_pos (Or.inr (Or.inr hE)), if_pos p2,
                    if_pos (Or.inr (Or.inr (by rw [Nat.xor_zero]; exact hE))),
                    if_pos (Or.inr (Or.inr (by rw [Nat.xor_zero]; exact hE)))]; decide
              · have p3 : ¬ (a = 0 ∨ b ^^^ L = 0 ∨ a = b ^^^ L) := by
                  rintro (g | g | g)
                  · exact hA g
                  · exact hB' g
                  · exact hL0 (by
                      rw [← hE] at g
                      have h2 : a ^^^ a = (a ^^^ L) ^^^ a := by rw [← g]
                      rw [Nat.xor_self, Nat.xor_comm a L, Nat.xor_assoc, Nat.xor_self,
                          Nat.xor_zero] at h2
                      exact h2.symm)
                have p4 : ¬ (a ^^^ L = 0 ∨ b = 0 ∨ a ^^^ L = b) := by
                  rintro (g | g | g)
                  · exact hA' g
                  · exact hB g
                  · exact hL0 (by
                      rw [← hE] at g
                      have h2 : (a ^^^ L) ^^^ a = a ^^^ a := by rw [g]
                      rw [Nat.xor_self, Nat.xor_comm a L, Nat.xor_assoc, Nat.xor_self,
                          Nat.xor_zero] at h2
                      exact h2)
                rw [if_pos (Or.inr (Or.inr hE)), if_pos p2, if_neg p3, if_neg p4]; decide
            · -- none of the five; P1 and P2 are both false, P3 and P4 agree
              have p1 : ¬ (a = 0 ∨ b = 0 ∨ a = b) := by
                rintro (g | g | g)
                · exact hA g
                · exact hB g
                · exact hE g
              have p2 : ¬ (a ^^^ L = 0 ∨ b ^^^ L = 0 ∨ a ^^^ L = b ^^^ L) := by
                rintro (g | g | g)
                · exact hA' g
                · exact hB' g
                · exact hE ((xor_right_cancel a b L).mp g)
              by_cases hF : a ^^^ L = b
              · have p3 : a = 0 ∨ b ^^^ L = 0 ∨ a = b ^^^ L :=
                  Or.inr (Or.inr (by rw [← hF, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]))
                rw [if_neg p1, if_neg p2, if_pos p3, if_pos (Or.inr (Or.inr hF))]; decide
              · have p3 : ¬ (a = 0 ∨ b ^^^ L = 0 ∨ a = b ^^^ L) := by
                  rintro (g | g | g)
                  · exact hA g
                  · exact hB' g
                  · exact hF (by rw [g, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero])
                have p4 : ¬ (a ^^^ L = 0 ∨ b = 0 ∨ a ^^^ L = b) := by
                  rintro (g | g | g)
                  · exact hA' g
                  · exact hB g
                  · exact hF g
                rw [if_neg p1, if_neg p2, if_neg p3, if_neg p4]; decide
  rcases Qgen_pm L a b m with h1 | h1 <;> rcases Qgen_pm L b a m with h2 | h2 <;>
    rw [h1, h2] at hprod ⊢ <;> revert hprod <;> decide

/-! ## The bit-swap `tau`, and the properties the assembly needs

`tau j` swaps bit 0 with bit `j`. It is used throughout `(*)`, but NO lemma about it exists
anywhere in the tree -- this is the prerequisite layer the assembly is missing. -/

/-- `tau j` swaps bit `0` and bit `j`: it xors by `1 ||| 2^j` exactly when the two bits differ. -/
def tau (j x : Nat) : Nat := if (x &&& 1) == ((x >>> j) &&& 1) then x else x ^^^ (1 ||| (1 <<< j))

theorem tau_zero (j : Nat) : tau j 0 = 0 := by
  unfold tau; simp

/-- `tau` never leaves the level: it only permutes bits below `m`. -/
theorem tau_lt (j m x : Nat) (hj : j < m) (hx : x < 2^m) : tau j x < 2^m := by
  have h1 : (1:Nat) < 2^m := by
    have h := Nat.one_lt_two_pow_iff (n := m); omega
  have h2 : (1:Nat) <<< j < 2^m := by
    rw [Nat.shiftLeft_eq, Nat.one_mul]
    exact Nat.pow_lt_pow_right (by omega) hj
  unfold tau
  split
  · exact hx
  · exact Nat.xor_lt_two_pow hx (Nat.or_lt_two_pow h1 h2)


/-- Bridge: the numeric bit test used in `tau`'s definition is `Nat.testBit`. -/
theorem and_one_testBit (x i : Nat) : (x >>> i) &&& 1 = if x.testBit i then 1 else 0 := by
  rw [Nat.and_one_is_mod, Nat.testBit]
  split <;> simp_all <;> omega

/-- `tau` in `testBit` form: xor by the mask exactly when bits `0` and `j` differ. -/
theorem tau_spec (j x : Nat) :
    tau j x = if x.testBit 0 = x.testBit j then x else x ^^^ (1 ||| (1 <<< j)) := by
  unfold tau
  have h0 : x &&& 1 = if x.testBit 0 then 1 else 0 := by
    have := and_one_testBit x 0; rwa [Nat.shiftRight_zero] at this
  have hj : (x >>> j) &&& 1 = if x.testBit j then 1 else 0 := and_one_testBit x j
  rw [h0, hj]
  cases x.testBit 0 <;> cases x.testBit j <;> simp

/-- The mask has exactly bits `0` and `j` set. -/
theorem mask_testBit_zero (j : Nat) : ((1:Nat) ||| (1 <<< j)).testBit 0 = true := by
  rw [Nat.testBit_or]; simp

theorem mask_testBit_j (j : Nat) : ((1:Nat) ||| (1 <<< j)).testBit j = true := by
  rw [Nat.testBit_or, Nat.shiftLeft_eq, Nat.one_mul, Nat.testBit_two_pow]
  simp

/-- At `j = 0` the swap is the identity. -/
theorem tau_id_zero (x : Nat) : tau 0 x = x := by
  rw [tau_spec]; simp

/-- **`tau` is an involution.** -/
theorem tau_involutive (j x : Nat) : tau j (tau j x) = x := by
  by_cases hj : j = 0
  · subst hj; rw [tau_id_zero, tau_id_zero]
  · rw [tau_spec j x]
    split
    · rename_i h; rw [tau_spec j x]; simp [h]
    · rename_i h
      have b0 : (x ^^^ ((1:Nat) ||| (1 <<< j))).testBit 0 = !x.testBit 0 := by
        rw [Nat.testBit_xor, mask_testBit_zero]; cases x.testBit 0 <;> simp
      have bj : (x ^^^ ((1:Nat) ||| (1 <<< j))).testBit j = !x.testBit j := by
        rw [Nat.testBit_xor, mask_testBit_j]; cases x.testBit j <;> simp
      rw [tau_spec j (x ^^^ ((1:Nat) ||| (1 <<< j))), b0, bj]
      have hne : ¬ ((!x.testBit 0) = (!x.testBit j)) := by
        intro hh; exact h (by cases hb : x.testBit 0 <;> cases hc : x.testBit j <;>
          simp_all)
      rw [if_neg hne, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

/-- `(u ⊕ w) ⊕ (v ⊕ w) = u ⊕ v` — the mask cancels when both sides carry it. -/
theorem xor_pair_cancel (u v w : Nat) : (u ^^^ w) ^^^ (v ^^^ w) = u ^^^ v := by
  rw [Nat.xor_assoc, ← Nat.xor_assoc w v w, Nat.xor_comm w v, Nat.xor_assoc, Nat.xor_self,
      Nat.xor_zero]

theorem xor_pair_cancel' (u v w : Nat) : (u ^^^ w) ^^^ (w ^^^ v) = u ^^^ v := by
  rw [Nat.xor_assoc, ← Nat.xor_assoc w w v, Nat.xor_self, Nat.zero_xor]

/-- **`tau` is F2-linear.** This is the property the assembly uses in every branch, to move
    `tau` through `a ⊕ Y`. -/
theorem tau_xor (j x y : Nat) : tau j (x ^^^ y) = tau j x ^^^ tau j y := by
  have hb0 : (x ^^^ y).testBit 0 = (x.testBit 0 ^^ y.testBit 0) := Nat.testBit_xor x y 0
  have hbj : (x ^^^ y).testBit j = (x.testBit j ^^ y.testBit j) := Nat.testBit_xor x y j
  rw [tau_spec j (x ^^^ y), tau_spec j x, tau_spec j y, hb0, hbj]
  cases hx0 : x.testBit 0 <;> cases hxj : x.testBit j <;>
    cases hy0 : y.testBit 0 <;> cases hyj : y.testBit j <;>
    simp only [Bool.xor_false, Bool.xor_true, Bool.not_true, Bool.not_false,
               if_true, if_false, if_pos rfl,
               if_neg (by decide : ¬ (true = false)),
               if_neg (by decide : ¬ (false = true)), reduceIte] <;>
    first
      | rfl
      | exact Nat.xor_assoc x y _
      | (rw [Nat.xor_assoc, Nat.xor_comm y _, ← Nat.xor_assoc]; done)
      | exact (xor_pair_cancel x y _).symm
      | exact (xor_pair_cancel' x y ((1:Nat) ||| 1 <<< j)).symm
      | exact (xor_pair_cancel x y ((1:Nat) ||| 1 <<< j)).symm
      | exact (xor_pair_cancel' x y _).symm
      | exact xor_pair_cancel' x y _
      | exact xor_pair_cancel x y _
      | ac_rfl
      | (simp [xor_pair_cancel]; done)


theorem seam_add_xor (v n : Nat) (hv : v < 2^(n+1)) : v + 2^(n+1) = v ^^^ 2^(n+1) := by
  have h := Nat.two_pow_add_eq_or_of_lt hv 1
  rw [Nat.mul_one] at h
  rw [Nat.add_comm, h]
  apply Nat.eq_of_testBit_eq; intro k
  rw [Nat.testBit_or, Nat.testBit_xor, Nat.testBit_two_pow]
  by_cases hk : n+1 = k
  · subst hk; rw [Nat.testBit_lt_two_pow hv]; simp
  · simp [hk]

/-- The seam bit is fixed by `tau`, because both of its bits `0` and `j` are clear. -/
theorem tau_seam_fixed (j m : Nat) (hj : j < m + 1) : tau j (2^(m+1)) = 2^(m+1) := by
  rw [tau_spec]
  have h0 : ((2:Nat)^(m+1)).testBit 0 = false := by
    rw [Nat.testBit_two_pow]; simp
  have hjb : ((2:Nat)^(m+1)).testBit j = false := by
    rw [Nat.testBit_two_pow]; simp; omega
  rw [h0, hjb, if_pos rfl]

/-- **`tau` preserves the half.** Immediate from linearity plus `tau_seam_fixed`: this is what
    puts both sides of `(*)` in the SAME quadrant, so they reduce by the same lemma. -/
theorem tau_seam (j m u : Nat) (hj : j < m + 1) (hu : u < 2^(m+1)) :
    tau j (u + 2^(m+1)) = tau j u + 2^(m+1) := by
  rw [seam_add_xor u m hu, tau_xor, tau_seam_fixed j m hj,
      ← seam_add_xor (tau j u) m (tau_lt j (m+1) u hj hu)]

end SounioZDChi
