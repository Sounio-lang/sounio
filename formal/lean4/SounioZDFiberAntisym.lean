/-
  SounioZDFiberAntisym — the ALGEBRAIC CORE of the ZD-fiber antisymmetry lemma, proven ∀n.

  Context. `scripts/research/cd_tower_zd_fiber_antisymmetry_lemma_contract.py` (rung of
  2026-07-31) found the explicit low-rank factorisation that the prior ∀n rung listed as OPEN.
  It rests on one identity, clause `A3`: the involution `l ↦ l ⊕ L_lo` on a fiber SWAPS the two
  resonance products `P1`, `P3` and NEGATES them. That identity is pure rewriting through the
  four branches of the Cayley-Dickson sign recursion — no induction — and this file proves it
  for ALL n, kernel-checked, Mathlib-free, no `sorry`, no `native_decide`.

  What is proven here (`P1`, `P3` as in the contract, `H = 2^(n+1)`, `hi x = (x ⊕ L_lo) + H`):

    core_P1     P1 (l ⊕ L_lo) y = - P3 l y        (needs y ≠ 0, y ⊕ L_lo ≠ 0)
    core_P3     P3 (l ⊕ L_lo) y = - P1 l y        (needs y ≠ 0, y ⊕ L_lo ≠ 0)
    P3_symm     P3 l y = P3 y l                    (needs l ≠ 0, y ≠ 0)

  `core_P1`/`core_P3` are the contract's `A3`; `P3_symm` is half of its `A2_VACUITY`. The two
  side conditions are exactly the contract's excluded column: `y ≠ L_lo` is `y ⊕ L_lo ≠ 0`.

  What is NOT proven here — stated so the spec cannot overclaim:
    * `A1` itself (the antisymmetry of `A_σ`) additionally needs the mask to be invariant, which
      needs P1-symmetry, which reduces to `antisym` (cdSigma a b = - cdSigma b a for distinct
      nonzero a,b). `antisym` is proven ∀n in `SounioSeamFlip.lean` on the UNMERGED branch
      `lean/cd-seamflip-forall-n` — verified to compile clean in this environment
      ([propext, Classical.choice, Quot.sound], no sorryAx) but NOT in this tree.
    * `A4`'s level-(n−1) sub-lemma `τ(l,L_lo) = −τ(l ⊕ L_lo, L_lo)` (the isolated vertex).
    * everything numerical: rank, spectra, the factorisation `A_σ = Jᵀ M J`.

  Provenance. `cdSigma`, `cdSig0/'`, the four branch reductions `R_ll/R_lu/R_ul/R_uu`, and the
  seam XOR bridge `seam_add_xor`/`xor_seam` are reproduced from `SounioSeamFlip.lean` on
  `lean/cd-seamflip-forall-n` (they are definitional unfoldings of `cdSigma`'s case split). They
  are carried here verbatim so this file is self-contained and so those lemmas exist IN THE TREE
  for the first time — the branch they live on has never been merged.

  Axioms: `[propext, Quot.sound]` or `[propext, Classical.choice, Quot.sound]` throughout
  (Classical.choice enters via `simp`; a standard logical axiom, NOT a computational trust
  anchor like `native_decide`). Check with `#print axioms` — `grep error` is not sufficient,
  Lean silently recovers unknown identifiers with `sorryAx`.
-/

namespace SounioZDFiberAntisym

/-! ## Carried prerequisites (from SounioSeamFlip.lean, branch lean/cd-seamflip-forall-n) -/

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

/-! ## The fiber: hi-partner, and the two resonance products -/

/-- The hi-partner of a lo-label `x` in the fiber `L = L_lo + H`, `H = 2^(n+1)`. -/
def hi (x Llo n : Nat) : Nat := (x ^^^ Llo) + 2^(n+1)

/-- The additive form used here IS the XOR form `x ⊕ L` of the contract and the spec. -/
theorem hi_eq_xor (x Llo n : Nat) (hx : x < 2^(n+1)) (hL : Llo < 2^(n+1)) :
    x ^^^ (Llo + 2^(n+1)) = hi x Llo n :=
  xor_seam x Llo n hx hL

/-- `P1 l y = σ(l,y)·σ(h l, h y)`. -/
def P1 (l y Llo n : Nat) : Int :=
  cdSigma l y (n+2) * cdSigma (hi l Llo n) (hi y Llo n) (n+2)

/-- `P3 l y = σ(l, h y)·σ(h l, y)`. -/
def P3 (l y Llo n : Nat) : Int :=
  cdSigma l (hi y Llo n) (n+2) * cdSigma (hi l Llo n) y (n+2)

/-! ## Reduction of each product to level n+1 -/

private theorem xorlt {u v n : Nat} (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) :
    u ^^^ v < 2^(n+1) := Nat.xor_lt_two_pow hu hv

private theorem xor_cancel (l Llo : Nat) : (l ^^^ Llo) ^^^ Llo = l := by
  rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

/-- `P1` descends one level: `P1 l y = σ(l,y)·σ(y⊕L_lo, l⊕L_lo)`, when `y ⊕ L_lo ≠ 0`. -/
theorem P1_red (l y Llo n : Nat) (hl : l < 2^(n+1)) (hy : y < 2^(n+1)) (hL : Llo < 2^(n+1))
    (hyL : y ^^^ Llo ≠ 0) :
    P1 l y Llo n = cdSigma l y (n+1) * cdSigma (y ^^^ Llo) (l ^^^ Llo) (n+1) := by
  unfold P1 hi
  rw [R_ll l y n hl hy, R_uu (l ^^^ Llo) (y ^^^ Llo) n (xorlt hl hL) (xorlt hy hL),
      if_neg hyL]

/-- `P3` descends one level: `P3 l y = −σ(y⊕L_lo, l)·σ(l⊕L_lo, y)`, when `y ≠ 0`. -/
theorem P3_red (l y Llo n : Nat) (hl : l < 2^(n+1)) (hy : y < 2^(n+1)) (hL : Llo < 2^(n+1))
    (hy0 : y ≠ 0) :
    P3 l y Llo n = - (cdSigma (y ^^^ Llo) l (n+1) * cdSigma (l ^^^ Llo) y (n+1)) := by
  unfold P3 hi
  rw [R_lu l (y ^^^ Llo) n hl (xorlt hy hL),
      R_ul (l ^^^ Llo) y n (xorlt hl hL) hy, if_neg hy0]
  exact (Int.mul_neg _ _)

/-! ## The core: the involution swaps P1 and P3, and negates them -/

/-- **A3, first half.** `P1 (l ⊕ L_lo) y = − P3 l y`. -/
theorem core_P1 (l y Llo n : Nat) (hl : l < 2^(n+1)) (hy : y < 2^(n+1)) (hL : Llo < 2^(n+1))
    (hy0 : y ≠ 0) (hyL : y ^^^ Llo ≠ 0) :
    P1 (l ^^^ Llo) y Llo n = - P3 l y Llo n := by
  rw [P1_red (l ^^^ Llo) y Llo n (xorlt hl hL) hy hL hyL,
      P3_red l y Llo n hl hy hL hy0, xor_cancel l Llo, Int.neg_neg]
  exact Int.mul_comm _ _

/-- **A3, second half.** `P3 (l ⊕ L_lo) y = − P1 l y`. -/
theorem core_P3 (l y Llo n : Nat) (hl : l < 2^(n+1)) (hy : y < 2^(n+1)) (hL : Llo < 2^(n+1))
    (hy0 : y ≠ 0) (hyL : y ^^^ Llo ≠ 0) :
    P3 (l ^^^ Llo) y Llo n = - P1 l y Llo n := by
  rw [P3_red (l ^^^ Llo) y Llo n (xorlt hl hL) hy hL hy0,
      P1_red l y Llo n hl hy hL hyL, xor_cancel l Llo]
  exact congrArg _ (Int.mul_comm _ _)

/-- **Half of A2_VACUITY.** `P3` is symmetric, unconditionally on the fiber label. -/
theorem P3_symm (l y Llo n : Nat) (hl : l < 2^(n+1)) (hy : y < 2^(n+1)) (hL : Llo < 2^(n+1))
    (hl0 : l ≠ 0) (hy0 : y ≠ 0) :
    P3 l y Llo n = P3 y l Llo n := by
  rw [P3_red l y Llo n hl hy hL hy0, P3_red y l Llo n hy hl hL hl0]
  exact congrArg _ (Int.mul_comm _ _)

end SounioZDFiberAntisym
