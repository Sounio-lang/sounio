/-
  SounioZDFiberAntisym — the ALGEBRAIC CORE of the ZD-fiber antisymmetry lemma, proven ∀n.

  Context. `scripts/research/cd_tower_zd_fiber_antisymmetry_lemma_contract.py` (rung of
  2026-07-31) found the explicit low-rank factorisation that the prior ∀n rung listed as OPEN.
  It rests on one identity, clause `A3`: the involution `l ↦ l ⊕ L_lo` on a fiber SWAPS the two
  resonance products `P1`, `P3` and NEGATES them. That identity is pure rewriting through the
  four branches of the Cayley-Dickson sign recursion — no induction — and this file proves it
  for ALL n, kernel-checked, Mathlib-free, no `sorry`, no `native_decide`.

  **THE LEMMA ITSELF IS NOW PROVEN**, not just its core. With `SounioSeamFlip.lean` merged into
  the tree, `antisym` is available and the last gap closes. Contents (`P1`, `P3`, `resB`, `Asig`
  as in the contract's builder, `H = 2^(n+1)`, `hi x = (x ⊕ L_lo) + H`):

    core_P1     P1 (l ⊕ L_lo) y = - P3 l y        contract clause A3
    core_P3     P3 (l ⊕ L_lo) y = - P1 l y        contract clause A3
    P3_symm     P3 l y = P3 y l                    A2_VACUITY, first half
    P1_symm     P1 l y = P1 y l                    A2_VACUITY, second half (uses `antisym`)
    resB_inv    resB (l ⊕ L_lo) y = resB l y       A2_MASK
    A1          Asig (l ⊕ L_lo) y = - Asig l y     ** THE LEMMA, contract clause A1 **
    Asig_diag   Asig l l = 0                       the builder's fill_diagonal is REDUNDANT

  Hypotheses throughout: `l, y, L_lo < H`, all of `l, y, l ⊕ L_lo, y ⊕ L_lo` nonzero. The last
  two are exactly the contract's excluded row/column `l = L_lo` / `y = L_lo` — §3's isolated
  vertex — so the Lean statement's side conditions ARE the paper proof's, not a weakening.

  `Asig_diag` matters for honesty: the `Asig` defined here does NOT zero the diagonal, while the
  contract's builder calls `np.fill_diagonal(A, 0)`. The theorem shows resonance already FAILS
  on the diagonal (`P1 = 1`, `P3 = -1`), so that call is a no-op and the two matrices coincide.
  Without it, `A1` here would be about a slightly different object.

  What is still NOT proven here:
    * `A4`'s level-(n−1) sub-lemma `τ(l,L_lo) = −τ(l ⊕ L_lo, L_lo)` (why the isolated vertex sits
      at `l = L_lo` rather than merely being excluded by hypothesis).
    * everything numerical: rank, spectra, the factorisation `A_σ = Jᵀ M J`, the ∀n completeness.

  Provenance. `cdSigma`, `cdSig0/'`, the four branch reductions `R_ll/R_lu/R_ul/R_uu`, the seam
  XOR bridge `seam_add_xor`/`xor_seam`, `antisym` and `cdSigma_pm` are reproduced verbatim from
  `SounioSeamFlip.lean`, which lived only on `lean/cd-seamflip-forall-n` and was merged into this
  tree alongside this file — the lane had been citing them as "proven ∀n" while they were absent
  from the tree (an R20-class dangling citation). This file is self-contained so it compiles
  without `import`, which is unreliable in this environment.

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

/-! ## Tier 2: antisymmetry of the cocycle, P1-symmetry, and `A1` itself -/

/-- ANTISYMMETRY of the sign cocycle, ∀n. Carried from `SounioSeamFlip.lean` (now in-tree). -/
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

private theorem xor_ne {l y Llo : Nat} (h : l ≠ y) : l ^^^ Llo ≠ y ^^^ Llo := by
  intro he
  exact h (by rw [← xor_cancel l Llo, he, xor_cancel y Llo])

/-- **The other half of A2_VACUITY.** `P1` is symmetric too. This is where `antisym` enters. -/
theorem P1_symm (l y Llo n : Nat) (hl : l < 2^(n+1)) (hy : y < 2^(n+1)) (hL : Llo < 2^(n+1))
    (hl0 : l ≠ 0) (hy0 : y ≠ 0) (hlL : l ^^^ Llo ≠ 0) (hyL : y ^^^ Llo ≠ 0) :
    P1 l y Llo n = P1 y l Llo n := by
  by_cases hly : l = y
  · rw [hly]
  · rw [P1_red l y Llo n hl hy hL hyL, P1_red y l Llo n hy hl hL hlL,
        antisym (n+1) l y hl hy hl0 hy0 hly,
        antisym (n+1) (y ^^^ Llo) (l ^^^ Llo) (xorlt hy hL) (xorlt hl hL) hyL hlL
          (xor_ne (fun h => hly h.symm))]
    exact (Int.neg_mul_neg _ _)

/-- The resonance predicate, exactly as in the contract's builder. -/
def resB (l y Llo n : Nat) : Bool :=
  (P1 l y Llo n == P1 y l Llo n) && (P3 l y Llo n == P3 y l Llo n) &&
  (P1 l y Llo n == P3 l y Llo n)

/-- The signed resonance matrix. NOTE: no diagonal zeroing — see `Asig_diag`. -/
def Asig (l y Llo n : Nat) : Int := if resB l y Llo n then - P1 l y Llo n else 0

/-- The mask is invariant under the involution. -/
theorem resB_inv (l y Llo n : Nat) (hl : l < 2^(n+1)) (hy : y < 2^(n+1)) (hL : Llo < 2^(n+1))
    (hl0 : l ≠ 0) (hy0 : y ≠ 0) (hlL : l ^^^ Llo ≠ 0) (hyL : y ^^^ Llo ≠ 0) :
    resB (l ^^^ Llo) y Llo n = resB l y Llo n := by
  have hml : l ^^^ Llo < 2^(n+1) := xorlt hl hL
  have hmm : (l ^^^ Llo) ^^^ Llo ≠ 0 := by rw [xor_cancel]; exact hl0
  -- the two symmetry clauses are TRUE on both sides (A2_VACUITY), so they cannot distinguish
  have t1 : (P1 (l ^^^ Llo) y Llo n == P1 y (l ^^^ Llo) Llo n) = true := by
    rw [P1_symm (l ^^^ Llo) y Llo n hml hy hL hlL hy0 hmm hyL]; simp
  have t2 : (P3 (l ^^^ Llo) y Llo n == P3 y (l ^^^ Llo) Llo n) = true := by
    rw [P3_symm (l ^^^ Llo) y Llo n hml hy hL hlL hy0]; simp
  have t3 : (P1 l y Llo n == P1 y l Llo n) = true := by
    rw [P1_symm l y Llo n hl hy hL hl0 hy0 hlL hyL]; simp
  have t4 : (P3 l y Llo n == P3 y l Llo n) = true := by
    rw [P3_symm l y Llo n hl hy hL hl0 hy0]; simp
  have c1 : P1 (l ^^^ Llo) y Llo n = - P3 l y Llo n := core_P1 l y Llo n hl hy hL hy0 hyL
  have c2 : P3 (l ^^^ Llo) y Llo n = - P1 l y Llo n := core_P3 l y Llo n hl hy hL hy0 hyL
  unfold resB
  rw [t1, t2, t3, t4]
  by_cases h3 : P1 l y Llo n = P3 l y Llo n
  · have hm3 : P1 (l ^^^ Llo) y Llo n = P3 (l ^^^ Llo) y Llo n := by rw [c1, c2, h3]
    simp [hm3, h3]
  · have hm3 : P1 (l ^^^ Llo) y Llo n ≠ P3 (l ^^^ Llo) y Llo n := by
      rw [c1, c2]; intro hh; exact h3 (by omega)
    have f1 : (P1 (l ^^^ Llo) y Llo n == P3 (l ^^^ Llo) y Llo n) = false :=
      beq_eq_false_iff_ne.mpr hm3
    have f2 : (P1 l y Llo n == P3 l y Llo n) = false := beq_eq_false_iff_ne.mpr h3
    simp [f1, f2]

/-- **A1 — the fiber antisymmetry lemma itself, ∀n.** -/
theorem A1 (l y Llo n : Nat) (hl : l < 2^(n+1)) (hy : y < 2^(n+1)) (hL : Llo < 2^(n+1))
    (hl0 : l ≠ 0) (hy0 : y ≠ 0) (hlL : l ^^^ Llo ≠ 0) (hyL : y ^^^ Llo ≠ 0) :
    Asig (l ^^^ Llo) y Llo n = - Asig l y Llo n := by
  unfold Asig
  rw [resB_inv l y Llo n hl hy hL hl0 hy0 hlL hyL]
  by_cases h : resB l y Llo n
  · have h3 : P1 l y Llo n = P3 l y Llo n := by
      unfold resB at h
      simp only [Bool.and_eq_true, beq_iff_eq] at h
      exact h.2
    rw [if_pos h, if_pos h, core_P1 l y Llo n hl hy hL hy0 hyL, ← h3, Int.neg_neg]
  · rw [if_neg h, if_neg h, Int.neg_zero]

/-! ## Tier 3: the diagonal is already zero — the builder's `fill_diagonal` is a no-op -/

/-- Every basis unit squares to `-1`: `σ(x,x) = -1` for `x ≠ 0`, ∀n. -/
theorem sigma_self : ∀ (m x : Nat), x < 2^m → x ≠ 0 → cdSigma x x m = -1
  | 0, x, hx, hx0 => by
      have : (2:Nat)^0 = 1 := rfl
      omega
  | 1, x, _, hx0 => by simp [cdSigma, hx0]
  | (m+2), x, hx, hx0 => by
    have h2H : 2^(m+2) = 2^(m+1) + 2^(m+1) := by rw [Nat.pow_succ]; omega
    by_cases hU : x ≥ 2^(m+1)
    · obtain ⟨u, hxe, hul⟩ : ∃ u, x = u + 2^(m+1) ∧ u < 2^(m+1) :=
        ⟨x - 2^(m+1), by omega, by omega⟩
      rw [hxe, R_uu u u m hul hul]
      by_cases hu0 : u = 0
      · simp [hu0]
      · rw [if_neg hu0]; exact sigma_self (m+1) u hul hu0
    · have hxl : x < 2^(m+1) := by omega
      rw [R_ll x x m hxl hxl]
      exact sigma_self (m+1) x hxl hx0

/-- `cdSigma` takes values in `{1, -1}`. Carried from `SounioSeamFlip.lean`. -/
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
theorem P1_diag (l Llo n : Nat) (hl : l < 2^(n+1)) (hL : Llo < 2^(n+1))
    (hl0 : l ≠ 0) (hlL : l ^^^ Llo ≠ 0) : P1 l l Llo n = 1 := by
  rw [P1_red l l Llo n hl hl hL hlL, sigma_self (n+1) l hl hl0,
      sigma_self (n+1) (l ^^^ Llo) (xorlt hl hL) hlL]
  decide

/-- `P3` on the diagonal is `-1`. -/
theorem P3_diag (l Llo n : Nat) (hl : l < 2^(n+1)) (hL : Llo < 2^(n+1))
    (hl0 : l ≠ 0) : P3 l l Llo n = -1 := by
  rw [P3_red l l Llo n hl hl hL hl0]
  rcases cdSigma_pm (n+1) (l ^^^ Llo) l with h | h <;> rw [h] <;> decide

/-- **The builder's `np.fill_diagonal(A, 0)` is redundant**: resonance already FAILS on the
    diagonal (`P1 = 1 ≠ -1 = P3`), so `Asig l l = 0` without zeroing it by hand. This is what
    makes the `Asig` defined here the same matrix the contract measures. -/
theorem Asig_diag (l Llo n : Nat) (hl : l < 2^(n+1)) (hL : Llo < 2^(n+1))
    (hl0 : l ≠ 0) (hlL : l ^^^ Llo ≠ 0) : Asig l l Llo n = 0 := by
  have hP1 : P1 l l Llo n = 1 := P1_diag l Llo n hl hL hl0 hlL
  have hP3 : P3 l l Llo n = -1 := P3_diag l Llo n hl hL hl0
  have hb : (P1 l l Llo n == P3 l l Llo n) = false := by rw [hP1, hP3]; decide
  unfold Asig resB
  simp [hb]

end SounioZDFiberAntisym
