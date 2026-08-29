/-
  SounioSeamFlip — the ∀n ONE-STEP RECURSION (R) for the Cayley-Dickson sign cocycle, the keystone under
  the seam-flip law (and hence under the whole exact-algebra 168 lane: lift, orbit theorem, the ZD
  annihilation=associator bridge, the core-law twin recursion — all bottom out on it).

  R is the algebraic content of the cdSigma four-branch case split: prepending the seam bit H = 2^{n+1}
  to the arguments descends one Cayley-Dickson level, with the four branches (u,v < H, u ≠ 0)
        cdSigma (u)   (v)     (n+2) = cdSigma u v (n+1)                    (both lower)
        cdSigma (u)   (v+H)   (n+2) = cdSigma v u (n+1)                    (swap)
        cdSigma (u+H) (v)     (n+2) = if v=0 then 1  else - cdSigma u v (n+1)
        cdSigma (u+H) (v+H)   (n+2) = if v=0 then -1 else   cdSigma v u (n+1)
  PROVED here for ALL n by definitional unfolding — Mathlib-free, no sorry, no native_decide.  Axioms:
  R_ll is kernel-clean [propext, Quot.sound]; R_lu/R_ul/R_uu are [propext, Classical.choice, Quot.sound]
  (Classical.choice enters via `simp`, a standard logical axiom — NOT a computational trust anchor like
  native_decide).  Previously this recursion was only verified EXHAUSTIVELY AT FIXED n (the ingredient (R)
  of scripts/research/cd_tower_seam_unique_argmax_proof.py, checked over all (u,v,p,q) per level); this
  file certifies it structurally ∀n — upgrading the base of the keystone from "verified" to "proven".

  This file also proves, ∀n, the second ingredient of the seam-flip law:
    `antisym` : cdSigma a b m = - cdSigma b a m for distinct nonzero a,b (< 2^m), by structural
    induction on the level using the four R branches (Mathlib-free, no sorry, no native_decide; axioms
    [propext, Classical.choice, Quot.sound]).  (This is the Nat-`cdSigma` analogue of the bit-list
    `antisym` in SounioCDCocycle.lean, reproven here self-contained since that file does not compile in
    the current resource-constrained env.)

  THE SEAM-FLIP LAW (F) — NOW PROVEN ∀n HERE (generic locus), all three seam directions, as the
  R + antisym assembly.  With psiSign i j k m := cdSigma i j m * cdSigma (i⊕j) k m * cdSigma j k m *
  cdSigma i (j⊕k) m  (= (-1)^{Ψ(i,j,k)}), we prove, ∀n, on the generic locus (the args and their XORs
  nonzero + the antisym distinctness conditions):
    seamflip_mid : psiSign u (v+H) w (n+2) = psiSign u v w (n+1)        (middle slot: net correction +1)
    seamflip_lo  : psiSign (u+H) v w (n+2) = - psiSign u v w (n+1)      (low slot:    correction (-1)^χ(v,w))
    seamflip_hi  : psiSign u v (w+H) (n+2) = - psiSign u v w (n+1)      (high slot:   correction (-1)^χ(u,v))
  Each is the exact assembly the Python keystone describes: reduce the four cdSigma factors by R (R_ll/
  R_lu/R_ul + the xor_seam helpers for the i⊕j, j⊕k cross terms), unify the argument-swaps via antisym,
  and the sign flips cancel in the predicted pattern.  All kernel-checked, Mathlib-free, no sorry/
  native_decide (axioms [propext, Classical.choice, Quot.sound]).

  DEGENERATE LOCUS — NOW CLOSED (∀n, all u,v,w, no genericity).  Using cdSq (cdSigma=±1) + swap_uniform
  (antisym extended to the whole locus, no distinctness) + chi_eq (chi = zero-pattern product) + Rul_pair
  /Rul_factored (the R_ul if-y=0 degeneracy folded into a ±1) + chi_triple, the three single-bit flips
  hold for ALL u,v,w with the exact chi-correction:
    seamflip_mid_full : psiSign u (v+H) w (n+2) = psiSign u v w (n+1) · chi u v · chi u (v⊕w)
    seamflip_lo_full  : psiSign (u+H) v w (n+2) = psiSign u v w (n+1) · chi v w
    seamflip_hi_full  : psiSign u v (w+H) (n+2) = psiSign u v w (n+1) · chi u v
  (chi a b = cdSigma a b · cdSigma b a = (-1)^{χ(a,b)}).  All kernel-checked, no sorry/native_decide,
  axioms [propext, Classical.choice, Quot.sound].  So the DEGENERATE LOCUS is fully closed for every
  single-bit flip.

  MULTI-BIT (p,q,r) — NOW ALL PROVEN ∀n (all u,v,w).  All eight seam configurations are formalised:
    seamflip_none (0,0,0), seamflip_lo/mid/hi_full (single bit), and
    seamflip_110 / seamflip_101 / seamflip_011 / seamflip_111 (multiple bits),
  each  psiSign (u+pH)(v+qH)(w+rH) (n+2) = psiSign u v w (n+1) · (correction),  correction =
    (if p then chi v w) · (if q then chi u v · chi u (v⊕w)) · (if r then chi u v).
  The multi-bit combos use R_uu_factored (both-upper factor) + xor_seam_cancel ((x+H)⊕(y+H)=x⊕y) + the
  chi-product identities chi_swap / chi_swap2 / chi_sq (all proven via chi_eq + sgn0_sq + ac_rfl).  All
  kernel-checked, no sorry/native_decide, axioms [propext, Classical.choice, Quot.sound].

  ==> THE FULL SEAM-FLIP LAW (F) IS NOW PROVEN ∀n IN LEAN: every seam configuration, the whole locus
  (generic + degenerate), with exact chi-corrections.  So the session's entire ∀n stack (lift / orbit
  theorem / annihilation=associator bridge / core-law twin recursion), all of which bottom out on the
  seam-flip law, is now completely formally ∀n-grounded in Lean — kernel-checked, Mathlib-free, no sorry.
-/
namespace SounioSeamFlip

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

/-- cdSigma with a zero first argument is +1 at every real level (m ≥ 1). -/
theorem cdSig0 (b m : Nat) : cdSigma 0 b (m+1) = 1 := by cases m <;> simp [cdSigma]
/-- cdSigma with a zero second argument is +1 at every real level (m ≥ 1). -/
theorem cdSig0' (a m : Nat) : cdSigma a 0 (m+1) = 1 := by cases m <;> simp [cdSigma]

/-- R, both-lower branch (unconditional: holds for all u,v < H). -/
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

/-- ANTISYMMETRY of the sign cocycle: cdSigma a b m = - cdSigma b a m for distinct nonzero a,b (< 2^m).
    Proved ∀n by structural induction on the level `m`, using the four R branches. -/
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
    · -- both upper: a = ua+H, b = ub+H
      obtain ⟨ua, hae, hual⟩ : ∃ ua, a = ua + 2^(n+1) ∧ ua < 2^(n+1) :=
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
    · -- a upper, b lower
      obtain ⟨ua, hae, hual⟩ : ∃ ua, a = ua + 2^(n+1) ∧ ua < 2^(n+1) :=
        ⟨a - 2^(n+1), by omega, by omega⟩
      have hbl : b < 2^(n+1) := by omega
      rw [hae, R_ul ua b n hual hbl, R_lu b ua n hbl hual, if_neg hb0]
    · -- a lower, b upper
      obtain ⟨ub, hbe, hubl⟩ : ∃ ub, b = ub + 2^(n+1) ∧ ub < 2^(n+1) :=
        ⟨b - 2^(n+1), by omega, by omega⟩
      have hal : a < 2^(n+1) := by omega
      rw [hbe, R_lu a ub n hal hubl, R_ul ub a n hubl hal, if_neg ha0]; omega
    · -- both lower
      have hal : a < 2^(n+1) := by omega
      have hbl : b < 2^(n+1) := by omega
      rw [R_ll a b n hal hbl, R_ll b a n hbl hal]
      exact antisym (n+1) a b hal hbl ha0 hb0 hab

-- ===== cdSigma is ±1, and a UNIFORM swap law (handles the degenerate locus without distinctness) =====

/-- Every cdSigma value is +1 or -1. -/
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

/-- cdSigma squared is 1. -/
theorem cdSq (a b m : Nat) : cdSigma a b m * cdSigma a b m = 1 := by
  rcases cdSigma_pm m a b with h | h <;> rw [h] <;> decide

/-- Uniform swap (no distinctness / nonzero needed): cdSigma b a = (cdSigma a b · cdSigma b a) · cdSigma a b.
    The prefactor is chi a b = (-1)^{χ(a,b)}; this is antisym extended to the degenerate locus. -/
theorem swap_uniform (a b m : Nat) :
    cdSigma b a m = (cdSigma a b m * cdSigma b a m) * cdSigma a b m := by
  have := cdSq a b m
  rw [Int.mul_comm (cdSigma a b m) (cdSigma b a m), Int.mul_assoc, this, Int.mul_one]

/-- chi a b = (-1)^{χ(a,b)} = cdSigma a b · cdSigma b a — the seam-flip correction factor. -/
def chi (a b m : Nat) : Int := cdSigma a b m * cdSigma b a m

/-- The two R_ul factors of a seam-flip multiply cleanly for ALL w (the if-w=0 degeneracy is absorbed):
    cdSigma (x+H) w (n+2) · cdSigma (y+H) w (n+2) = cdSigma x w (n+1) · cdSigma y w (n+1). -/
theorem Rul_pair (x y w n : Nat) (hx : x < 2^(n+1)) (hy : y < 2^(n+1)) (hw : w < 2^(n+1)) :
    cdSigma (x + 2^(n+1)) w (n+2) * cdSigma (y + 2^(n+1)) w (n+2)
    = cdSigma x w (n+1) * cdSigma y w (n+1) := by
  rw [R_ul x w n hx hw, R_ul y w n hy hw]
  by_cases hw0 : w = 0
  · subst hw0; simp [cdSig0']
  · rw [if_neg hw0, if_neg hw0]; simp [Int.neg_mul, Int.mul_neg, Int.neg_neg]

/-- A single upper×lower factor, uniformly (the if-y=0 degeneracy folded into a ±1 sign). -/
theorem Rul_factored (x y n : Nat) (hx : x < 2^(n+1)) (hy : y < 2^(n+1)) :
    cdSigma (x + 2^(n+1)) y (n+2) = cdSigma x y (n+1) * (if y = 0 then 1 else -1) := by
  rw [R_ul x y n hx hy]
  by_cases hy0 : y = 0
  · subst hy0; simp [cdSig0']
  · rw [if_neg hy0, if_neg hy0, Int.mul_neg_one]

/-- chi as the zero-pattern product: chi v w = sgn0(v)·sgn0(w)·sgn0(v⊕w).  This is antisymmetry extended
    to the whole locus (= (-1)^{χ(v,w)} with χ = n0(v)⊕n0(w)⊕n0(v⊕w)). -/
theorem chi_eq (v w n : Nat) (hv : v < 2^(n+1)) (hw : w < 2^(n+1)) :
    chi v w (n+1)
    = (if v = 0 then 1 else -1) * (if w = 0 then 1 else -1) * (if v ^^^ w = 0 then 1 else -1) := by
  unfold chi
  by_cases hv0 : v = 0
  · subst hv0; by_cases hw0 : w = 0 <;> simp [hw0, cdSig0, cdSig0', Nat.zero_xor]
  by_cases hw0 : w = 0
  · subst hw0; simp [hv0, cdSig0, cdSig0', Nat.xor_zero]
  by_cases hvw : v = w
  · subst hvw; simp [hv0, Nat.xor_self, cdSq]
  · have hvw0 : v ^^^ w ≠ 0 := by
      intro h
      apply hvw
      have h2 : (v ^^^ w) ^^^ w = 0 ^^^ w := by rw [h]
      rwa [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero, Nat.zero_xor] at h2
    rw [if_neg hv0, if_neg hw0, if_neg hvw0,
        antisym (n+1) w v hw hv hw0 hv0 (Ne.symm hvw), Int.mul_neg, cdSq]
    decide

/-- Swap law with chi folded (a restatement of swap_uniform). -/
theorem swap_chi (a b m : Nat) : cdSigma b a m = chi a b m * cdSigma a b m := swap_uniform a b m

/-- sgn0 squared is 1. -/
theorem sgn0_sq (x : Nat) : (if x = 0 then (1:Int) else -1) * (if x = 0 then 1 else -1) = 1 := by
  by_cases hx : x = 0 <;> simp [hx]

/-- chi triple identity (the high-slot correction collapse): the three swap-corrections multiply to
    the single chi u v (all the doubled zero-pattern factors square away). -/
theorem chi_triple (u v w n : Nat) (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) (hw : w < 2^(n+1)) :
    chi (u ^^^ v) w (n+1) * chi v w (n+1) * chi u (v ^^^ w) (n+1) = chi u v (n+1) := by
  rw [chi_eq (u ^^^ v) w n (Nat.xor_lt_two_pow hu hv) hw, chi_eq v w n hv hw,
      chi_eq u (v ^^^ w) n hu (Nat.xor_lt_two_pow hv hw), chi_eq u v n hu hv, Nat.xor_assoc,
      show ((if u ^^^ v = 0 then (1:Int) else -1) * (if w = 0 then 1 else -1)
             * (if u ^^^ (v ^^^ w) = 0 then 1 else -1))
           * ((if v = 0 then 1 else -1) * (if w = 0 then 1 else -1) * (if v ^^^ w = 0 then 1 else -1))
           * ((if u = 0 then 1 else -1) * (if v ^^^ w = 0 then 1 else -1)
             * (if u ^^^ (v ^^^ w) = 0 then 1 else -1))
         = ((if w = 0 then 1 else -1) * (if w = 0 then 1 else -1))
           * ((if v ^^^ w = 0 then 1 else -1) * (if v ^^^ w = 0 then 1 else -1))
           * ((if u ^^^ (v ^^^ w) = 0 then 1 else -1) * (if u ^^^ (v ^^^ w) = 0 then 1 else -1))
           * ((if u = 0 then 1 else -1) * (if v = 0 then 1 else -1) * (if u ^^^ v = 0 then 1 else -1))
         from by ac_rfl,
      sgn0_sq w, sgn0_sq (v ^^^ w), sgn0_sq (u ^^^ (v ^^^ w)), Int.one_mul, Int.one_mul, Int.one_mul]

-- ===== XOR-seam helpers (a lower index XORed with a seam-shifted one stays seam-shifted) =====

/-- Adding the seam bit to a sub-seam value is the same as XORing it. -/
theorem seam_add_xor (v n : Nat) (hv : v < 2^(n+1)) : v + 2^(n+1) = v ^^^ 2^(n+1) := by
  have h := Nat.two_pow_add_eq_or_of_lt hv 1
  rw [Nat.mul_one] at h
  rw [Nat.add_comm, h]
  apply Nat.eq_of_testBit_eq; intro k
  rw [Nat.testBit_or, Nat.testBit_xor, Nat.testBit_two_pow]
  by_cases hk : n+1 = k
  · subst hk; rw [Nat.testBit_lt_two_pow hv]; simp
  · simp [hk]

theorem xor_seam (u v n : Nat) (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) :
    u ^^^ (v + 2^(n+1)) = (u ^^^ v) + 2^(n+1) := by
  rw [seam_add_xor v n hv, seam_add_xor (u ^^^ v) n (Nat.xor_lt_two_pow hu hv), ← Nat.xor_assoc]

theorem xor_seam2 (v w n : Nat) (hv : v < 2^(n+1)) (hw : w < 2^(n+1)) :
    (v + 2^(n+1)) ^^^ w = (v ^^^ w) + 2^(n+1) := by
  rw [Nat.xor_comm, xor_seam w v n hw hv, Nat.xor_comm w v]

-- ===== the SEAM-FLIP LAW for the associator (middle-slot bit), generic locus =====

/-- (−1)^{Ψ(i,j,k)} as the product of the associator's four cdSigma factors. -/
def psiSign (i j k m : Nat) : Int :=
  cdSigma i j m * cdSigma (i ^^^ j) k m * cdSigma j k m * cdSigma i (j ^^^ k) m

/-- SEAM-FLIP LAW, middle slot (q=1), on the generic locus: putting the seam bit on the MIDDLE argument
    of the associator descends one level unchanged.  This is the R + antisym assembly: the two R_ul sign
    flips and the two antisym swaps cancel in fours.  (On the degenerate locus the χ-corrections appear;
    here u,v,w and v⊕w are nonzero, u∉{v, v⊕w} — exactly where χ(u,v)=χ(u,v⊕w)=1 so the net correction
    vanishes.) -/
theorem seamflip_mid (u v w n : Nat)
    (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) (hw : w < 2^(n+1))
    (hu0 : u ≠ 0) (hv0 : v ≠ 0) (hw0 : w ≠ 0)
    (huv : u ≠ v) (hvw0 : v ^^^ w ≠ 0) (huvw : u ≠ v ^^^ w) :
    psiSign u (v + 2^(n+1)) w (n+2) = psiSign u v w (n+1) := by
  unfold psiSign
  rw [R_lu u v n hu hv,
      xor_seam u v n hu hv, R_ul (u ^^^ v) w n (Nat.xor_lt_two_pow hu hv) hw, if_neg hw0,
      R_ul v w n hv hw, if_neg hw0,
      xor_seam2 v w n hv hw, R_lu u (v ^^^ w) n hu (Nat.xor_lt_two_pow hv hw),
      antisym (n+1) v u hv hu hv0 hu0 (Ne.symm huv),
      antisym (n+1) (v ^^^ w) u (Nat.xor_lt_two_pow hv hw) hu hvw0 hu0 (Ne.symm huvw)]
  simp [Int.neg_mul, Int.mul_neg, Int.neg_neg]

/-- chi squared is 1. -/
theorem chi_sq (a b m : Nat) : chi a b m * chi a b m = 1 := by
  unfold chi
  have h1 := cdSq a b m; have h2 := cdSq b a m
  rw [show (cdSigma a b m * cdSigma b a m) * (cdSigma a b m * cdSigma b a m)
        = (cdSigma a b m * cdSigma a b m) * (cdSigma b a m * cdSigma b a m) from by ac_rfl,
      h1, h2, Int.one_mul]

/-- chi swap identity (for the multi-bit combos): chi u v · chi (u⊕v) w = chi v w · chi u (v⊕w). -/
theorem chi_swap (u v w n : Nat) (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) (hw : w < 2^(n+1)) :
    chi u v (n+1) * chi (u ^^^ v) w (n+1) = chi v w (n+1) * chi u (v ^^^ w) (n+1) := by
  rw [chi_eq u v n hu hv, chi_eq (u ^^^ v) w n (Nat.xor_lt_two_pow hu hv) hw,
      chi_eq v w n hv hw, chi_eq u (v ^^^ w) n hu (Nat.xor_lt_two_pow hv hw), Nat.xor_assoc,
      show ((if u = 0 then (1:Int) else -1) * (if v = 0 then 1 else -1) * (if u ^^^ v = 0 then 1 else -1))
           * ((if u ^^^ v = 0 then 1 else -1) * (if w = 0 then 1 else -1)
              * (if u ^^^ (v ^^^ w) = 0 then 1 else -1))
         = ((if u ^^^ v = 0 then 1 else -1) * (if u ^^^ v = 0 then 1 else -1))
           * ((if u = 0 then 1 else -1) * (if v = 0 then 1 else -1) * (if w = 0 then 1 else -1)
              * (if u ^^^ (v ^^^ w) = 0 then 1 else -1)) from by ac_rfl,
      sgn0_sq (u ^^^ v), Int.one_mul,
      show ((if v = 0 then (1:Int) else -1) * (if w = 0 then 1 else -1) * (if v ^^^ w = 0 then 1 else -1))
           * ((if u = 0 then 1 else -1) * (if v ^^^ w = 0 then 1 else -1)
              * (if u ^^^ (v ^^^ w) = 0 then 1 else -1))
         = ((if v ^^^ w = 0 then 1 else -1) * (if v ^^^ w = 0 then 1 else -1))
           * ((if u = 0 then 1 else -1) * (if v = 0 then 1 else -1) * (if w = 0 then 1 else -1)
              * (if u ^^^ (v ^^^ w) = 0 then 1 else -1)) from by ac_rfl,
      sgn0_sq (v ^^^ w), Int.one_mul]

/-- chi swap identity 2: chi (u⊕v) w · chi u (v⊕w) = chi v w · chi u v. -/
theorem chi_swap2 (u v w n : Nat) (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) (hw : w < 2^(n+1)) :
    chi (u ^^^ v) w (n+1) * chi u (v ^^^ w) (n+1) = chi v w (n+1) * chi u v (n+1) := by
  rw [chi_eq (u ^^^ v) w n (Nat.xor_lt_two_pow hu hv) hw,
      chi_eq u (v ^^^ w) n hu (Nat.xor_lt_two_pow hv hw),
      chi_eq v w n hv hw, chi_eq u v n hu hv, Nat.xor_assoc,
      show ((if u ^^^ v = 0 then (1:Int) else -1) * (if w = 0 then 1 else -1)
              * (if u ^^^ (v ^^^ w) = 0 then 1 else -1))
           * ((if u = 0 then 1 else -1) * (if v ^^^ w = 0 then 1 else -1)
              * (if u ^^^ (v ^^^ w) = 0 then 1 else -1))
         = ((if u ^^^ (v ^^^ w) = 0 then 1 else -1) * (if u ^^^ (v ^^^ w) = 0 then 1 else -1))
           * ((if u = 0 then 1 else -1) * (if u ^^^ v = 0 then 1 else -1) * (if w = 0 then 1 else -1)
              * (if v ^^^ w = 0 then 1 else -1)) from by ac_rfl,
      sgn0_sq (u ^^^ (v ^^^ w)), Int.one_mul,
      show ((if v = 0 then (1:Int) else -1) * (if w = 0 then 1 else -1) * (if v ^^^ w = 0 then 1 else -1))
           * ((if u = 0 then 1 else -1) * (if v = 0 then 1 else -1) * (if u ^^^ v = 0 then 1 else -1))
         = ((if v = 0 then 1 else -1) * (if v = 0 then 1 else -1))
           * ((if u = 0 then 1 else -1) * (if u ^^^ v = 0 then 1 else -1) * (if w = 0 then 1 else -1)
              * (if v ^^^ w = 0 then 1 else -1)) from by ac_rfl,
      sgn0_sq v, Int.one_mul]

/-- Two seam bits cancel under XOR: (x+H) ⊕ (y+H) = x ⊕ y. -/
theorem xor_seam_cancel (x y n : Nat) (hx : x < 2^(n+1)) (hy : y < 2^(n+1)) :
    (x + 2^(n+1)) ^^^ (y + 2^(n+1)) = x ^^^ y := by
  rw [seam_add_xor x n hx, seam_add_xor y n hy, Nat.xor_assoc,
      Nat.xor_comm (2^(n+1)) (y ^^^ 2^(n+1)), Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

/-- Both-upper factor, uniformly: cdSigma (x+H)(y+H) = cdSigma y x · (if y=0 then -1 else 1). -/
theorem R_uu_factored (x y n : Nat) (hx : x < 2^(n+1)) (hy : y < 2^(n+1)) :
    cdSigma (x + 2^(n+1)) (y + 2^(n+1)) (n+2) = cdSigma y x (n+1) * (if y = 0 then -1 else 1) := by
  rw [R_uu x y n hx hy]
  by_cases hy0 : y = 0
  · subst hy0; simp [cdSig0]
  · rw [if_neg hy0, if_neg hy0, Int.mul_one]

/-- SEAM-FLIP, no seam (level descent): psiSign u v w (n+2) = psiSign u v w (n+1). -/
theorem seamflip_none (u v w n : Nat) (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) (hw : w < 2^(n+1)) :
    psiSign u v w (n+2) = psiSign u v w (n+1) := by
  unfold psiSign
  rw [R_ll u v n hu hv, R_ll (u ^^^ v) w n (Nat.xor_lt_two_pow hu hv) hw, R_ll v w n hv hw,
      R_ll u (v ^^^ w) n hu (Nat.xor_lt_two_pow hv hw)]

/-- SEAM-FLIP LAW, MULTI-BIT (1,1,0): psiSign (u+H)(v+H) w. -/
theorem seamflip_110 (u v w n : Nat) (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) (hw : w < 2^(n+1)) :
    psiSign (u + 2^(n+1)) (v + 2^(n+1)) w (n+2)
    = psiSign u v w (n+1) * chi v w (n+1) * (chi u v (n+1) * chi u (v ^^^ w) (n+1)) := by
  unfold psiSign
  rw [R_uu_factored u v n hu hv, xor_seam_cancel u v n hu hv,
      R_ll (u ^^^ v) w n (Nat.xor_lt_two_pow hu hv) hw,
      Rul_factored v w n hv hw, xor_seam2 v w n hv hw,
      R_uu_factored u (v ^^^ w) n hu (Nat.xor_lt_two_pow hv hw),
      swap_chi u v (n+1), swap_chi u (v ^^^ w) (n+1)]
  have hj : (if v = 0 then (-1:Int) else 1) * (if w = 0 then 1 else -1) * (if v ^^^ w = 0 then -1 else 1)
          = chi v w (n+1) := by
    rw [chi_eq v w n hv hw]
    by_cases hv0 : v = 0 <;> by_cases hw0 : w = 0 <;> by_cases hvw0 : v ^^^ w = 0 <;> simp_all
  rw [show (chi u v (n+1) * cdSigma u v (n+1) * (if v = 0 then (-1:Int) else 1))
             * cdSigma (u ^^^ v) w (n+1)
             * (cdSigma v w (n+1) * (if w = 0 then (1:Int) else -1))
             * (chi u (v ^^^ w) (n+1) * cdSigma u (v ^^^ w) (n+1) * (if v ^^^ w = 0 then (-1:Int) else 1))
           = (chi u v (n+1) * chi u (v ^^^ w) (n+1))
             * ((if v = 0 then (-1:Int) else 1) * (if w = 0 then 1 else -1) * (if v ^^^ w = 0 then -1 else 1))
             * (cdSigma u v (n+1) * cdSigma (u ^^^ v) w (n+1) * cdSigma v w (n+1) * cdSigma u (v ^^^ w) (n+1))
         from by ac_rfl, hj]
  ac_rfl

/-- SEAM-FLIP LAW, MULTI-BIT (1,1,1): all three seam bits. -/
theorem seamflip_111 (u v w n : Nat) (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) (hw : w < 2^(n+1)) :
    psiSign (u + 2^(n+1)) (v + 2^(n+1)) (w + 2^(n+1)) (n+2)
    = psiSign u v w (n+1) * chi v w (n+1) * (chi u v (n+1) * chi u (v ^^^ w) (n+1)) * chi u v (n+1) := by
  unfold psiSign
  rw [R_uu_factored u v n hu hv, xor_seam_cancel u v n hu hv,
      R_lu (u ^^^ v) w n (Nat.xor_lt_two_pow hu hv) hw,
      R_uu_factored v w n hv hw, xor_seam_cancel v w n hv hw,
      Rul_factored u (v ^^^ w) n hu (Nat.xor_lt_two_pow hv hw),
      swap_chi u v (n+1), swap_chi (u ^^^ v) w (n+1), swap_chi v w (n+1)]
  have hj : (if v = 0 then (-1:Int) else 1) * (if w = 0 then -1 else 1) * (if v ^^^ w = 0 then 1 else -1)
          = chi v w (n+1) := by
    rw [chi_eq v w n hv hw]
    by_cases hv0 : v = 0 <;> by_cases hw0 : w = 0 <;> by_cases hvw0 : v ^^^ w = 0 <;> simp_all
  have hchi : chi u v (n+1) * chi (u ^^^ v) w (n+1) * (chi v w (n+1) * chi v w (n+1))
            = chi v w (n+1) * chi u (v ^^^ w) (n+1) * (chi u v (n+1) * chi u v (n+1)) := by
    rw [chi_sq v w, chi_sq u v, Int.mul_one, Int.mul_one]; exact chi_swap u v w n hu hv hw
  rw [show (chi u v (n+1) * cdSigma u v (n+1) * (if v = 0 then (-1:Int) else 1))
             * (chi (u ^^^ v) w (n+1) * cdSigma (u ^^^ v) w (n+1))
             * (chi v w (n+1) * cdSigma v w (n+1) * (if w = 0 then -1 else 1))
             * (cdSigma u (v ^^^ w) (n+1) * (if v ^^^ w = 0 then 1 else -1))
           = (chi u v (n+1) * chi (u ^^^ v) w (n+1)
               * (chi v w (n+1) * ((if v = 0 then (-1:Int) else 1) * (if w = 0 then -1 else 1)
                    * (if v ^^^ w = 0 then 1 else -1))))
             * (cdSigma u v (n+1) * cdSigma (u ^^^ v) w (n+1) * cdSigma v w (n+1) * cdSigma u (v ^^^ w) (n+1))
         from by ac_rfl, hj, hchi]
  ac_rfl

/-- SEAM-FLIP LAW, MULTI-BIT (1,0,1): psiSign (u+H) v (w+H). -/
theorem seamflip_101 (u v w n : Nat) (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) (hw : w < 2^(n+1)) :
    psiSign (u + 2^(n+1)) v (w + 2^(n+1)) (n+2)
    = psiSign u v w (n+1) * chi v w (n+1) * chi u v (n+1) := by
  unfold psiSign
  rw [Rul_factored u v n hu hv,
      xor_seam2 u v n hu hv, R_uu_factored (u ^^^ v) w n (Nat.xor_lt_two_pow hu hv) hw,
      R_lu v w n hv hw,
      xor_seam v w n hv hw, R_uu_factored u (v ^^^ w) n hu (Nat.xor_lt_two_pow hv hw),
      swap_chi (u ^^^ v) w (n+1), swap_chi v w (n+1), swap_chi u (v ^^^ w) (n+1)]
  have hj : (if v = 0 then (1:Int) else -1) * (if w = 0 then -1 else 1) * (if v ^^^ w = 0 then -1 else 1)
          = chi v w (n+1) := by
    rw [chi_eq v w n hv hw]
    by_cases hv0 : v = 0 <;> by_cases hw0 : w = 0 <;> by_cases hvw0 : v ^^^ w = 0 <;> simp_all
  have hchi : chi (u ^^^ v) w (n+1) * chi u (v ^^^ w) (n+1) * (chi v w (n+1) * chi v w (n+1))
            = chi v w (n+1) * chi u v (n+1) := by
    rw [chi_sq v w, Int.mul_one]; exact chi_swap2 u v w n hu hv hw
  rw [show (cdSigma u v (n+1) * (if v = 0 then (1:Int) else -1))
             * (chi (u ^^^ v) w (n+1) * cdSigma (u ^^^ v) w (n+1) * (if w = 0 then -1 else 1))
             * (chi v w (n+1) * cdSigma v w (n+1))
             * (chi u (v ^^^ w) (n+1) * cdSigma u (v ^^^ w) (n+1) * (if v ^^^ w = 0 then -1 else 1))
           = (chi (u ^^^ v) w (n+1) * chi u (v ^^^ w) (n+1)
               * (chi v w (n+1) * ((if v = 0 then (1:Int) else -1) * (if w = 0 then -1 else 1)
                    * (if v ^^^ w = 0 then -1 else 1))))
             * (cdSigma u v (n+1) * cdSigma (u ^^^ v) w (n+1) * cdSigma v w (n+1) * cdSigma u (v ^^^ w) (n+1))
         from by ac_rfl, hj, hchi]
  ac_rfl

/-- SEAM-FLIP LAW, MULTI-BIT (0,1,1): psiSign u (v+H) (w+H). -/
theorem seamflip_011 (u v w n : Nat) (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) (hw : w < 2^(n+1)) :
    psiSign u (v + 2^(n+1)) (w + 2^(n+1)) (n+2)
    = psiSign u v w (n+1) * (chi u v (n+1) * chi u (v ^^^ w) (n+1)) * chi u v (n+1) := by
  unfold psiSign
  rw [R_lu u v n hu hv,
      xor_seam u v n hu hv, R_uu_factored (u ^^^ v) w n (Nat.xor_lt_two_pow hu hv) hw,
      R_uu_factored v w n hv hw, xor_seam_cancel v w n hv hw,
      R_ll u (v ^^^ w) n hu (Nat.xor_lt_two_pow hv hw),
      swap_chi u v (n+1), swap_chi (u ^^^ v) w (n+1), swap_chi v w (n+1)]
  have hj : (if w = 0 then (-1:Int) else 1) * (if w = 0 then -1 else 1) = 1 := by
    by_cases hw0 : w = 0 <;> simp [hw0]
  have hchi : chi u v (n+1) * chi (u ^^^ v) w (n+1) * chi v w (n+1)
            = chi u v (n+1) * chi u (v ^^^ w) (n+1) * chi u v (n+1) := by
    rw [chi_swap u v w n hu hv hw,
        show chi v w (n+1) * chi u (v ^^^ w) (n+1) * chi v w (n+1)
           = (chi v w (n+1) * chi v w (n+1)) * chi u (v ^^^ w) (n+1) from by ac_rfl,
        chi_sq v w, Int.one_mul,
        show chi u v (n+1) * chi u (v ^^^ w) (n+1) * chi u v (n+1)
           = (chi u v (n+1) * chi u v (n+1)) * chi u (v ^^^ w) (n+1) from by ac_rfl,
        chi_sq u v, Int.one_mul]
  rw [show (chi u v (n+1) * cdSigma u v (n+1))
             * (chi (u ^^^ v) w (n+1) * cdSigma (u ^^^ v) w (n+1) * (if w = 0 then (-1:Int) else 1))
             * (chi v w (n+1) * cdSigma v w (n+1) * (if w = 0 then -1 else 1))
             * cdSigma u (v ^^^ w) (n+1)
           = (chi u v (n+1) * chi (u ^^^ v) w (n+1) * chi v w (n+1))
             * (cdSigma u v (n+1) * cdSigma (u ^^^ v) w (n+1) * cdSigma v w (n+1) * cdSigma u (v ^^^ w) (n+1))
             * ((if w = 0 then (-1:Int) else 1) * (if w = 0 then -1 else 1))
         from by ac_rfl, hj, Int.mul_one, hchi]
  ac_rfl

/-- SEAM-FLIP LAW, middle slot, FULL (all u,v,w — degenerate locus included via chi).
    psiSign u (v+H) w (n+2) = psiSign u v w (n+1) · chi u v · chi u (v⊕w). -/
theorem seamflip_mid_full (u v w n : Nat)
    (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) (hw : w < 2^(n+1)) :
    psiSign u (v + 2^(n+1)) w (n+2)
      = psiSign u v w (n+1) * chi u v (n+1) * chi u (v ^^^ w) (n+1) := by
  unfold psiSign chi
  rw [R_lu u v n hu hv, xor_seam u v n hu hv, xor_seam2 v w n hv hw,
      R_lu u (v ^^^ w) n hu (Nat.xor_lt_two_pow hv hw),
      show cdSigma v u (n+1) * cdSigma ((u ^^^ v) + 2^(n+1)) w (n+2)
             * cdSigma (v + 2^(n+1)) w (n+2) * cdSigma (v ^^^ w) u (n+1)
           = cdSigma v u (n+1) * cdSigma (v ^^^ w) u (n+1)
             * (cdSigma ((u ^^^ v) + 2^(n+1)) w (n+2) * cdSigma (v + 2^(n+1)) w (n+2)) from by ac_rfl,
      Rul_pair (u ^^^ v) v w n (Nat.xor_lt_two_pow hu hv) hv hw]
  have hA := cdSq u v (n+1)
  have hD := cdSq u (v ^^^ w) (n+1)
  rw [show cdSigma v u (n+1) * cdSigma (v ^^^ w) u (n+1)
          * (cdSigma (u ^^^ v) w (n+1) * cdSigma v w (n+1))
        = (cdSigma u v (n+1) * cdSigma u v (n+1))
          * (cdSigma u (v ^^^ w) (n+1) * cdSigma u (v ^^^ w) (n+1))
          * (cdSigma v u (n+1) * cdSigma (v ^^^ w) u (n+1)
             * (cdSigma (u ^^^ v) w (n+1) * cdSigma v w (n+1)))
        from by rw [hA, hD, Int.one_mul, Int.one_mul]]
  ac_rfl

/-- SEAM-FLIP LAW, low slot (p=1), generic locus: the seam bit on the FIRST argument flips the sign
    (correction (-1)^{χ(v,w)} = -1 generically).  Pure R (three R_ul + one R_ll) — no antisym needed. -/
theorem seamflip_lo (u v w n : Nat)
    (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) (hw : w < 2^(n+1))
    (hv0 : v ≠ 0) (hw0 : w ≠ 0) (hvw0 : v ^^^ w ≠ 0) :
    psiSign (u + 2^(n+1)) v w (n+2) = - psiSign u v w (n+1) := by
  unfold psiSign
  rw [R_ul u v n hu hv, if_neg hv0,
      xor_seam2 u v n hu hv, R_ul (u ^^^ v) w n (Nat.xor_lt_two_pow hu hv) hw, if_neg hw0,
      R_ll v w n hv hw,
      R_ul u (v ^^^ w) n hu (Nat.xor_lt_two_pow hv hw), if_neg hvw0]
  simp [Int.neg_mul, Int.mul_neg, Int.neg_neg]

/-- SEAM-FLIP LAW, low slot (p=1), FULL (all u,v,w): psiSign (u+H) v w (n+2) = psiSign u v w (n+1) · chi v w. -/
theorem seamflip_lo_full (u v w n : Nat)
    (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) (hw : w < 2^(n+1)) :
    psiSign (u + 2^(n+1)) v w (n+2) = psiSign u v w (n+1) * chi v w (n+1) := by
  unfold psiSign
  rw [Rul_factored u v n hu hv, xor_seam2 u v n hu hv,
      Rul_factored (u ^^^ v) w n (Nat.xor_lt_two_pow hu hv) hw,
      R_ll v w n hv hw, Rul_factored u (v ^^^ w) n hu (Nat.xor_lt_two_pow hv hw),
      chi_eq v w n hv hw]
  ac_rfl

/-- SEAM-FLIP LAW, high slot (r=1), generic locus: the seam bit on the THIRD argument flips the sign
    (correction (-1)^{χ(u,v)} = -1 generically).  R (one R_ll + three R_lu) + three antisym swaps. -/
theorem seamflip_hi (u v w n : Nat)
    (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) (hw : w < 2^(n+1))
    (hu0 : u ≠ 0) (hv0 : v ≠ 0) (hw0 : w ≠ 0)
    (huv0 : u ^^^ v ≠ 0) (hw_uv : w ≠ u ^^^ v) (hwv : w ≠ v)
    (hvw0 : v ^^^ w ≠ 0) (hvwu : v ^^^ w ≠ u) :
    psiSign u v (w + 2^(n+1)) (n+2) = - psiSign u v w (n+1) := by
  unfold psiSign
  rw [R_ll u v n hu hv,
      R_lu (u ^^^ v) w n (Nat.xor_lt_two_pow hu hv) hw,
      R_lu v w n hv hw,
      xor_seam v w n hv hw, R_lu u (v ^^^ w) n hu (Nat.xor_lt_two_pow hv hw),
      antisym (n+1) w (u ^^^ v) hw (Nat.xor_lt_two_pow hu hv) hw0 huv0 hw_uv,
      antisym (n+1) w v hw hv hw0 hv0 hwv,
      antisym (n+1) (v ^^^ w) u (Nat.xor_lt_two_pow hv hw) hu hvw0 hu0 hvwu]
  simp [Int.neg_mul, Int.mul_neg, Int.neg_neg]

/-- SEAM-FLIP LAW, high slot (r=1), FULL (all u,v,w): psiSign u v (w+H) (n+2) = psiSign u v w (n+1) · chi u v. -/
theorem seamflip_hi_full (u v w n : Nat)
    (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) (hw : w < 2^(n+1)) :
    psiSign u v (w + 2^(n+1)) (n+2) = psiSign u v w (n+1) * chi u v (n+1) := by
  unfold psiSign
  rw [R_ll u v n hu hv,
      R_lu (u ^^^ v) w n (Nat.xor_lt_two_pow hu hv) hw,
      R_lu v w n hv hw,
      xor_seam v w n hv hw, R_lu u (v ^^^ w) n hu (Nat.xor_lt_two_pow hv hw),
      swap_chi (u ^^^ v) w (n+1), swap_chi v w (n+1), swap_chi u (v ^^^ w) (n+1),
      show cdSigma u v (n+1) * (chi (u ^^^ v) w (n+1) * cdSigma (u ^^^ v) w (n+1))
             * (chi v w (n+1) * cdSigma v w (n+1)) * (chi u (v ^^^ w) (n+1) * cdSigma u (v ^^^ w) (n+1))
           = (chi (u ^^^ v) w (n+1) * chi v w (n+1) * chi u (v ^^^ w) (n+1))
             * (cdSigma u v (n+1) * cdSigma (u ^^^ v) w (n+1) * cdSigma v w (n+1)
                * cdSigma u (v ^^^ w) (n+1))
         from by ac_rfl,
      chi_triple u v w n hu hv hw]
  ac_rfl

end SounioSeamFlip
