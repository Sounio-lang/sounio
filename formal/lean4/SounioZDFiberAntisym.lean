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
    A4_sub      σ(a,b) = - σ(a ⊕ b, b)             contract clause A4, IN FULL GENERALITY
    Asig_isolated / Asig_isolated_row              the column and the row at l = L_lo are ZERO
    Asig_symm   Asig l y = Asig y l

  Hypotheses throughout: `l, y, L_lo < H`, all of `l, y, l ⊕ L_lo, y ⊕ L_lo` nonzero. The last
  two are exactly the contract's excluded row/column `l = L_lo` / `y = L_lo` — the isolated
  vertex — so the Lean statement's side conditions ARE the paper proof's, not a weakening. And
  `Asig_isolated`/`Asig_isolated_row` show those two lines really are zero, so nothing is
  quietly assumed away by the hypotheses: the excluded row and column are DERIVED to vanish.

  `A4_sub` turned out to be more general than the fiber statement that motivated it: it holds for
  ALL nonzero `a, b` with `a ≠ b`, not only for `b = L_lo`. Two of its four induction branches
  need the swapped-argument form `σ(a,b) = -σ(a, a ⊕ b)`, which is derived inside the induction
  from the hypothesis plus `antisym`.

  `Asig_diag` matters for honesty: the `Asig` defined here does NOT zero the diagonal, while the
  contract's builder calls `np.fill_diagonal(A, 0)`. The theorem shows resonance already FAILS
  on the diagonal (`P1 = 1`, `P3 = -1`), so that call is a no-op and the two matrices coincide.
  Without it, `A1` here would be about a slightly different object.

  `Asig_symm` cannot be used to get `Asig_isolated_row` from `Asig_isolated`: it requires
  `l ⊕ L_lo ≠ 0`, which is exactly what `l = L_lo` violates. The row is proved directly.

  What is still NOT proven here: everything numerical — rank, spectra, the factorisation
  `A_σ = Jᵀ M J`, and the ∀n spectral completeness (`#spectra = 3·2^{n-5}`), which stays OPEN.

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

/-! ## Tier 4: `A4` — why the isolated vertex sits at exactly `l = L_lo` -/

theorem xor_zero_eq (x y : Nat) (h : x ^^^ y = 0) : x = y := by
  have h2 : (x ^^^ y) ^^^ y = 0 ^^^ y := by rw [h]
  rwa [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero, Nat.zero_xor] at h2

theorem seam_xor_left (u v n : Nat) (hu : u < 2^(n+1)) (hv : v < 2^(n+1)) :
    (u + 2^(n+1)) ^^^ v = (u ^^^ v) + 2^(n+1) := by
  rw [Nat.xor_comm, xor_seam v u n hv hu, Nat.xor_comm v u]

theorem xor_seam_cancel (x y n : Nat) (hx : x < 2^(n+1)) (hy : y < 2^(n+1)) :
    (x + 2^(n+1)) ^^^ (y + 2^(n+1)) = x ^^^ y := by
  rw [seam_add_xor x n hx, seam_add_xor y n hy, Nat.xor_assoc,
      Nat.xor_comm (2^(n+1)) (y ^^^ 2^(n+1)), Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

/-- **The `A4` sub-lemma, ∀n and in full generality** — not only on a fiber:
    `σ(a,b) = − σ(a ⊕ b, b)` for all nonzero `a, b` with `a ≠ b`. Proved by induction on the
    level through the four branch reductions; the swapped-argument form needed by two of the
    branches is derived inside from the induction hypothesis and `antisym`. -/
theorem A4_sub : ∀ (m a b : Nat), a < 2^m → b < 2^m → a ≠ 0 → b ≠ 0 → a ^^^ b ≠ 0 →
    cdSigma a b m = - cdSigma (a ^^^ b) b m
  | 0, a, _, ha, _, ha0, _, _ => by
      have : (2:Nat)^0 = 1 := rfl
      omega
  | 1, a, b, ha, hb, ha0, hb0, hab => by
      have h2 : (2:Nat)^1 = 2 := rfl
      have ha1 : a = 1 := by omega
      have hb1 : b = 1 := by omega
      subst ha1; subst hb1; simp at hab
  | (m+2), a, b, ha, hb, ha0, hb0, hab => by
    have h2H : 2^(m+2) = 2^(m+1) + 2^(m+1) := by rw [Nat.pow_succ]; omega
    -- the swapped form at level m+1, from the induction hypothesis plus `antisym`
    have second : ∀ x y, x < 2^(m+1) → y < 2^(m+1) → x ≠ 0 → y ≠ 0 → x ^^^ y ≠ 0 →
        cdSigma x y (m+1) = - cdSigma x (x ^^^ y) (m+1) := by
      intro x y hx hy hx0 hy0 hxy
      have hcl : x ^^^ y < 2^(m+1) := xorlt hx hy
      have hcx : (x ^^^ y) ^^^ x = y := by
        rw [Nat.xor_comm x y, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
      have i1 : cdSigma (x ^^^ y) x (m+1) = - cdSigma y x (m+1) := by
        have := A4_sub (m+1) (x ^^^ y) x hcl hx hxy hx0 (by rw [hcx]; exact hy0)
        rw [hcx] at this; exact this
      have hne : x ≠ x ^^^ y := by
        intro h
        exact hy0 (by rw [← hcx, ← h, Nat.xor_self])
      have i2 : cdSigma x (x ^^^ y) (m+1) = - cdSigma (x ^^^ y) x (m+1) :=
        antisym (m+1) x (x ^^^ y) hx hcl hx0 hxy hne
      have hxney : x ≠ y := by intro h; exact hxy (by rw [h, Nat.xor_self])
      have i3 : cdSigma x y (m+1) = - cdSigma y x (m+1) :=
        antisym (m+1) x y hx hy hx0 hy0 hxney
      rw [i2, i1, i3]; omega
    by_cases haU : a ≥ 2^(m+1) <;> by_cases hbU : b ≥ 2^(m+1)
    · -- both upper: a = u+H, b = v+H, a ⊕ b = u ⊕ v (lower)
      obtain ⟨u, hae, hul⟩ : ∃ u, a = u + 2^(m+1) ∧ u < 2^(m+1) :=
        ⟨a - 2^(m+1), by omega, by omega⟩
      obtain ⟨v, hbe, hvl⟩ : ∃ v, b = v + 2^(m+1) ∧ v < 2^(m+1) :=
        ⟨b - 2^(m+1), by omega, by omega⟩
      have huv : u ^^^ v ≠ 0 := by rw [hae, hbe, xor_seam_cancel u v m hul hvl] at hab; exact hab
      rw [hae, hbe, xor_seam_cancel u v m hul hvl, R_uu u v m hul hvl,
          R_lu (u ^^^ v) v m (xorlt hul hvl) hvl]
      by_cases hv0 : v = 0
      · subst hv0; rw [if_pos rfl, Nat.xor_zero, cdSig0]
      · rw [if_neg hv0]
        by_cases hu0 : u = 0
        · subst hu0
          rw [Nat.zero_xor, cdSig0', sigma_self (m+1) v hvl hv0]
          decide
        · have hvu : v ^^^ u ≠ 0 := by
            intro h; exact huv (by rw [Nat.xor_comm]; exact h)
          have := second v u hvl hul hv0 hu0 hvu
          rw [this, Nat.xor_comm v u]
    · -- a upper, b lower
      obtain ⟨u, hae, hul⟩ : ∃ u, a = u + 2^(m+1) ∧ u < 2^(m+1) :=
        ⟨a - 2^(m+1), by omega, by omega⟩
      have hbl : b < 2^(m+1) := by omega
      rw [hae, seam_xor_left u b m hul hbl, R_ul u b m hul hbl,
          R_ul (u ^^^ b) b m (xorlt hul hbl) hbl, if_neg hb0, if_neg hb0]
      by_cases hu0 : u = 0
      · subst hu0
        rw [Nat.zero_xor, cdSig0, sigma_self (m+1) b hbl hb0]
        decide
      · by_cases hub : u ^^^ b = 0
        · have hueq : u = b := xor_zero_eq u b hub
          subst hueq
          rw [hub, cdSig0, sigma_self (m+1) u hul hu0]
        · rw [A4_sub (m+1) u b hul hbl hu0 hb0 hub]
    · -- a lower, b upper
      obtain ⟨v, hbe, hvl⟩ : ∃ v, b = v + 2^(m+1) ∧ v < 2^(m+1) :=
        ⟨b - 2^(m+1), by omega, by omega⟩
      have hal : a < 2^(m+1) := by omega
      rw [hbe, xor_seam a v m hal hvl, R_lu a v m hal hvl,
          R_uu (a ^^^ v) v m (xorlt hal hvl) hvl]
      by_cases hv0 : v = 0
      · subst hv0; rw [if_pos rfl, cdSig0]; decide
      · rw [if_neg hv0]
        by_cases hav : a ^^^ v = 0
        · have haeq : a = v := xor_zero_eq a v hav
          subst haeq
          rw [hav, cdSig0', sigma_self (m+1) a hal ha0]
        · have hva : v ^^^ a ≠ 0 := by
            intro h; exact hav (by rw [Nat.xor_comm]; exact h)
          have := second v a hvl hal hv0 ha0 hva
          rw [this, Nat.xor_comm v a]
    · -- both lower
      have hal : a < 2^(m+1) := by omega
      have hbl : b < 2^(m+1) := by omega
      rw [R_ll a b m hal hbl, R_ll (a ^^^ b) b m (xorlt hal hbl) hbl]
      exact A4_sub (m+1) a b hal hbl ha0 hb0 hab

/-- **A4 itself.** On the excluded column `y = L_lo` the resonance predicate fails identically,
    so the row and column at `l = L_lo` are ZERO — the isolated vertex is at exactly `L_lo`,
    and that is the source of the `−1` in `rank = 2^{n-2} − 1`. -/
theorem Asig_isolated (l Llo n : Nat) (hl : l < 2^(n+1)) (hL : Llo < 2^(n+1))
    (hl0 : l ≠ 0) (hL0 : Llo ≠ 0) (hlL : l ^^^ Llo ≠ 0) :
    Asig l Llo Llo n = 0 := by
  have hml : l ^^^ Llo < 2^(n+1) := xorlt hl hL
  -- P1 and P3 on the excluded column, via the m' = 0 branches
  have hP1 : P1 l Llo Llo n = - cdSigma l Llo (n+1) := by
    unfold P1 hi
    rw [Nat.xor_self, R_ll l Llo n hl hL, R_uu (l ^^^ Llo) 0 n hml (Nat.two_pow_pos (n+1)),
        if_pos rfl]
    exact (Int.mul_neg_one _)
  have hP3 : P3 l Llo Llo n = - cdSigma (l ^^^ Llo) Llo (n+1) := by
    unfold P3 hi
    rw [Nat.xor_self, R_lu l 0 n hl (Nat.two_pow_pos (n+1)), cdSig0,
        R_ul (l ^^^ Llo) Llo n hml hL, if_neg hL0]
    exact (Int.one_mul _)
  have hne : (P1 l Llo Llo n == P3 l Llo Llo n) = false := by
    rw [hP1, hP3, A4_sub (n+1) l Llo hl hL hl0 hL0 hlL]
    rcases cdSigma_pm (n+1) (l ^^^ Llo) Llo with h | h <;> rw [h] <;> decide
  unfold Asig resB
  simp [hne]

/-- `A_σ` is symmetric (both products are, and so is the mask). -/
theorem Asig_symm (l y Llo n : Nat) (hl : l < 2^(n+1)) (hy : y < 2^(n+1)) (hL : Llo < 2^(n+1))
    (hl0 : l ≠ 0) (hy0 : y ≠ 0) (hlL : l ^^^ Llo ≠ 0) (hyL : y ^^^ Llo ≠ 0) :
    Asig l y Llo n = Asig y l Llo n := by
  have e1 : P1 l y Llo n = P1 y l Llo n := P1_symm l y Llo n hl hy hL hl0 hy0 hlL hyL
  have e2 : P3 l y Llo n = P3 y l Llo n := P3_symm l y Llo n hl hy hL hl0 hy0
  have hres : resB l y Llo n = resB y l Llo n := by
    unfold resB; rw [e1, e2]
  unfold Asig
  rw [hres, e1]

/-- **A4, the other direction.** The ROW at `l = L_lo` is zero too. `Asig_symm` cannot be used
    here — it requires `l ⊕ L_lo ≠ 0`, which is exactly what `l = L_lo` violates — so this is
    proved directly, through the `u = 0` branches. -/
theorem Asig_isolated_row (y Llo n : Nat) (hy : y < 2^(n+1)) (hL : Llo < 2^(n+1))
    (hy0 : y ≠ 0) (hL0 : Llo ≠ 0) (hyL : y ^^^ Llo ≠ 0) :
    Asig Llo y Llo n = 0 := by
  have hml : y ^^^ Llo < 2^(n+1) := xorlt hy hL
  have hP1 : P1 Llo y Llo n = cdSigma Llo y (n+1) := by
    unfold P1 hi
    rw [Nat.xor_self, R_ll Llo y n hL hy,
        R_uu 0 (y ^^^ Llo) n (Nat.two_pow_pos (n+1)) hml, if_neg hyL, cdSig0']
    exact (Int.mul_one _)
  have hP3 : P3 Llo y Llo n = - cdSigma (y ^^^ Llo) Llo (n+1) := by
    unfold P3 hi
    rw [Nat.xor_self, R_lu Llo (y ^^^ Llo) n hL hml,
        R_ul 0 y n (Nat.two_pow_pos (n+1)) hy, if_neg hy0, cdSig0]
    exact (Int.mul_neg_one _)
  have hA : cdSigma y Llo (n+1) = - cdSigma (y ^^^ Llo) Llo (n+1) :=
    A4_sub (n+1) y Llo hy hL hy0 hL0 hyL
  have hLy : Llo ≠ y := by intro h; exact hyL (by rw [← h, Nat.xor_self])
  have hanti : cdSigma Llo y (n+1) = - cdSigma y Llo (n+1) :=
    antisym (n+1) Llo y hL hy hL0 hy0 hLy
  have hb : (P1 Llo y Llo n == P3 Llo y Llo n) = false := by
    rw [hP1, hP3, ← hA, hanti]
    rcases cdSigma_pm (n+1) y Llo with h | h <;> rw [h] <;> decide
  unfold Asig resB
  simp [hb]

/-! ## Tier 5: the second-argument form, and the object `(*)` is about -/

/-- The second form of `A4_sub`: shifting the SECOND argument. Derived from `A4_sub` + `antisym`;
    the derivation genuinely needs both arguments nonzero and distinct, which is why the L1 rung
    measured the corresponding statement rather than assuming it -- its degenerate locus is
    nonempty (contract clause K6). Two of the four branches of the `(*)` induction need this. -/
theorem A4_sub' (m a b : Nat) (ha : a < 2^m) (hb : b < 2^m) (ha0 : a ≠ 0) (hb0 : b ≠ 0)
    (hab : a ^^^ b ≠ 0) : cdSigma a b m = - cdSigma a (a ^^^ b) m := by
  have hcl : a ^^^ b < 2^m := Nat.xor_lt_two_pow ha hb
  have hac : (a ^^^ b) ^^^ a = b := by
    rw [Nat.xor_comm a b, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
  have hne : a ≠ b := by intro h; exact hab (by rw [h, Nat.xor_self])
  have hnec : a ≠ a ^^^ b := by
    intro h; exact hb0 (by rw [← hac, ← h, Nat.xor_self])
  have hba : b ^^^ a ≠ 0 := by rw [Nat.xor_comm]; exact hab
  have i1 : cdSigma b a m = - cdSigma (a ^^^ b) a m := by
    have h := A4_sub m b a hb ha hb0 ha0 hba
    rwa [Nat.xor_comm b a] at h
  have i2 : cdSigma a b m = - cdSigma b a m := antisym m a b ha hb ha0 hb0 hne
  have i3 : cdSigma a (a ^^^ b) m = - cdSigma (a ^^^ b) a m :=
    antisym m a (a ^^^ b) ha hcl ha0 hab hnec
  rw [i2, i1, i3]

/-- The coset-square product `Q L a b = sigma(a,b) sigma(a^L,b^L) sigma(a,b^L) sigma(a^L,b)`.
    Resonance is `Q = 1`, and `(*)` is the statement that `Q` is `tau`-equivariant. -/
def Qgen (L a b m : Nat) : Int :=
  cdSigma a b m * cdSigma (a ^^^ L) (b ^^^ L) m * cdSigma a (b ^^^ L) m * cdSigma (a ^^^ L) b m

/-- `Q` depends only on the two COSETS `{a, a^L}`, `{b, b^L}` -- the four factors are permuted.
    Used to halve the case analysis in any induction on `Qgen`. -/
theorem Qgen_coset_left (L a b m : Nat) : Qgen L a b m = Qgen L (a ^^^ L) b m := by
  unfold Qgen
  rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
  ac_rfl

theorem Qgen_coset_right (L a b m : Nat) : Qgen L a b m = Qgen L a (b ^^^ L) m := by
  unfold Qgen
  rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
  ac_rfl

/-- `cdSigma` squared is 1. -/
theorem cdSq (a b m : Nat) : cdSigma a b m * cdSigma a b m = 1 := by
  rcases cdSigma_pm m a b with h | h <;> rw [h] <;> decide

/-- Degenerate branch, first-argument form: `sigma(u^L, L) * sigma(u, L) = -1`. -/
theorem deg_left (m L u : Nat) (hL : L < 2^m) (hu : u < 2^m) (hL0 : L ≠ 0) :
    cdSigma (u ^^^ L) L m * cdSigma u L m = -1 := by
  have hm : ∃ m', m = m' + 1 := by
    cases m with
    | zero => exact absurd (by omega : L = 0) hL0
    | succ k => exact ⟨k, rfl⟩
  obtain ⟨m', rfl⟩ := hm
  by_cases hu0 : u = 0
  · subst hu0
    rw [Nat.zero_xor, cdSig0, sigma_self (m'+1) L hL hL0]
    decide
  · by_cases huL : u ^^^ L = 0
    · have : u = L := xor_zero_eq u L huL
      subst this
      rw [huL, cdSig0, sigma_self (m'+1) u hu hu0]
      decide
    · rw [A4_sub (m'+1) u L hu hL hu0 hL0 huL, Int.mul_neg, cdSq]

/-- Degenerate branch, second-argument form: `sigma(L, u^L) * sigma(L, u) = -1`. -/
theorem deg_right (m L u : Nat) (hL : L < 2^m) (hu : u < 2^m) (hL0 : L ≠ 0) :
    cdSigma L (u ^^^ L) m * cdSigma L u m = -1 := by
  have hm : ∃ m', m = m' + 1 := by
    cases m with
    | zero => exact absurd (by omega : L = 0) hL0
    | succ k => exact ⟨k, rfl⟩
  obtain ⟨m', rfl⟩ := hm
  by_cases hu0 : u = 0
  · subst hu0
    rw [Nat.zero_xor, cdSig0', sigma_self (m'+1) L hL hL0]
    decide
  · by_cases huL : u ^^^ L = 0
    · have he : u = L := xor_zero_eq u L huL
      subst he
      rw [huL, cdSig0', sigma_self (m'+1) u hu hu0]
      decide
    · have hLu : L ^^^ u ≠ 0 := by
        intro h; exact huL (by rw [Nat.xor_comm]; exact h)
      have h := A4_sub' (m'+1) L u hL hu hL0 hu0 hLu
      rw [Nat.xor_comm L u] at h
      rw [h, Int.mul_neg, cdSq]

/-- `Q` at the level's TOP bit is `-1`, for lower-half arguments. This is the case where `tau`
    would move the top bit -- exactly what the four branch reductions hold fixed. -/
theorem Qgen_top (m a b : Nat) (ha : a < 2^(m+1)) (hb : b < 2^(m+1)) :
    Qgen (2^(m+1)) a b (m+2) = -1 := by
  have hxa : a ^^^ 2^(m+1) = a + 2^(m+1) := (seam_add_xor a m ha).symm
  have hxb : b ^^^ 2^(m+1) = b + 2^(m+1) := (seam_add_xor b m hb).symm
  unfold Qgen
  rw [hxa, hxb, R_ll a b m ha hb, R_uu a b m ha hb, R_lu a b m ha hb, R_ul a b m ha hb]
  by_cases hb0 : b = 0
  · subst hb0
    rw [if_pos rfl, if_pos rfl, cdSig0' a m, cdSig0 a m]
    decide
  · rw [if_neg hb0, if_neg hb0]
    have hstep : cdSigma a b (m+1) * cdSigma b a (m+1) * cdSigma b a (m+1)
                   * -cdSigma a b (m+1)
               = -((cdSigma a b (m+1) * cdSigma a b (m+1))
                   * (cdSigma b a (m+1) * cdSigma b a (m+1))) := by
      rw [Int.mul_neg]
      have : cdSigma a b (m+1) * cdSigma b a (m+1) * cdSigma b a (m+1) * cdSigma a b (m+1)
           = cdSigma a b (m+1) * cdSigma a b (m+1)
               * (cdSigma b a (m+1) * cdSigma b a (m+1)) := by ac_rfl
      rw [this]
    rw [hstep, cdSq, cdSq]
    decide

/-- **THE BASE CASE OF (*), PROVEN forall n.** `Q` at a single-bit label is identically `-1`.

    When `Y` is a lone power of two equal to the level's top bit, `j = lsb(Y)` IS that top bit,
    `tau Y = 1` is again a single bit, and both sides of `(*)` are this constant -- so the case
    where `tau` moves the top bit, which the four branch reductions hold fixed and which
    `A4_sub` never faced, is discharged.

    `(*)` itself is still NOT proven: its inductive step is the remaining work. -/
theorem Qgen_pow2 : ∀ (m k a b : Nat), k < m → a < 2^m → b < 2^m → Qgen (2^k) a b m = -1
  | 0, _, _, _, hk, _, _ => by omega
  | 1, k, a, b, hk, ha, hb => by
      have hk0 : k = 0 := by omega
      subst hk0
      have h2 : (2:Nat)^1 = 2 := rfl
      have hA : a = 0 ∨ a = 1 := by omega
      have hB : b = 0 ∨ b = 1 := by omega
      rcases hA with rfl | rfl <;> rcases hB with rfl | rfl <;> decide
  | (m+2), k, a, b, hk, ha, hb => by
    have hHpos : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
    have h2H : 2^(m+2) = 2^(m+1) + 2^(m+1) := by rw [Nat.pow_succ]; omega
    by_cases hkT : k = m+1
    · -- Y is the level's TOP bit: fold both arguments into the lower half, then Qgen_top
      subst hkT
      by_cases haU : a ≥ 2^(m+1)
      · obtain ⟨u, hae, hul⟩ : ∃ u, a = u + 2^(m+1) ∧ u < 2^(m+1) :=
          ⟨a - 2^(m+1), by omega, by omega⟩
        have hfold : a ^^^ 2^(m+1) = u := by
          rw [hae, seam_add_xor u m hul, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
        rw [Qgen_coset_left, hfold]
        by_cases hbU : b ≥ 2^(m+1)
        · obtain ⟨v, hbe, hvl⟩ : ∃ v, b = v + 2^(m+1) ∧ v < 2^(m+1) :=
            ⟨b - 2^(m+1), by omega, by omega⟩
          have hfb : b ^^^ 2^(m+1) = v := by
            rw [hbe, seam_add_xor v m hvl, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
          rw [Qgen_coset_right, hfb]
          exact Qgen_top m u v hul hvl
        · exact Qgen_top m u b hul (by omega)
      · by_cases hbU : b ≥ 2^(m+1)
        · obtain ⟨v, hbe, hvl⟩ : ∃ v, b = v + 2^(m+1) ∧ v < 2^(m+1) :=
            ⟨b - 2^(m+1), by omega, by omega⟩
          have hfb : b ^^^ 2^(m+1) = v := by
            rw [hbe, seam_add_xor v m hvl, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
          rw [Qgen_coset_right, hfb]
          exact Qgen_top m a v (by omega) hvl
        · exact Qgen_top m a b (by omega) (by omega)
    · -- Y strictly below the top bit: four quadrants, each dropping a level
      have hkm : k < m+1 := by omega
      have hLle : (2:Nat)^k ≤ 2^m := Nat.pow_le_pow_right (by omega) (by omega)
      have hMpos : (0:Nat) < 2^m := Nat.two_pow_pos m
      have hLlt : (2:Nat)^k < 2^(m+1) := by
        have : (2:Nat)^(m+1) = 2^m * 2 := by rw [Nat.pow_succ]
        omega
      by_cases haU : a ≥ 2^(m+1) <;> by_cases hbU : b ≥ 2^(m+1)
      · -- both upper
        obtain ⟨u, hae, hul⟩ : ∃ u, a = u + 2^(m+1) ∧ u < 2^(m+1) :=
          ⟨a - 2^(m+1), by omega, by omega⟩
        obtain ⟨v, hbe, hvl⟩ : ∃ v, b = v + 2^(m+1) ∧ v < 2^(m+1) :=
          ⟨b - 2^(m+1), by omega, by omega⟩
        have hxa : (u + 2^(m+1)) ^^^ 2^k = (u ^^^ 2^k) + 2^(m+1) :=
          seam_xor_left u (2^k) m hul hLlt
        have hxb : (v + 2^(m+1)) ^^^ 2^k = (v ^^^ 2^k) + 2^(m+1) :=
          seam_xor_left v (2^k) m hvl hLlt
        unfold Qgen
        rw [hae, hbe, hxa, hxb,
            R_uu u v m hul hvl, R_uu (u ^^^ 2^k) (v ^^^ 2^k) m (xorlt hul hLlt) (xorlt hvl hLlt),
            R_uu u (v ^^^ 2^k) m hul (xorlt hvl hLlt),
            R_uu (u ^^^ 2^k) v m (xorlt hul hLlt) hvl]
        by_cases hv0 : v = 0
        · subst hv0
          have hL0 : (2:Nat)^k ≠ 0 := by have := Nat.two_pow_pos k; omega
          simp only [Nat.zero_xor, if_true, if_neg hL0]
          have hd := deg_right (m+1) (2^k) u hLlt hul hL0
          rcases cdSigma_pm (m+1) (2^k) (u ^^^ 2^k) with h1 | h1 <;>
            rcases cdSigma_pm (m+1) (2^k) u with h2 | h2 <;>
            rw [h1, h2] at hd ⊢ <;> revert hd <;> decide
        · by_cases hvL : v ^^^ 2^k = 0
          · have hev : v = 2^k := xor_zero_eq v (2^k) hvL
            subst hev
            have hL0 : (2:Nat)^k ≠ 0 := by have := Nat.two_pow_pos k; omega
            rw [if_neg hv0, hvL, if_pos rfl, if_pos rfl, if_neg hv0]
            have hd := deg_right (m+1) (2^k) u hLlt hul hL0
            rcases cdSigma_pm (m+1) (2^k) (u ^^^ 2^k) with h1 | h1 <;>
              rcases cdSigma_pm (m+1) (2^k) u with h2 | h2 <;>
              rw [h1, h2] at hd ⊢ <;> revert hd <;> decide
          · rw [if_neg hv0, if_neg hvL, if_neg hvL, if_neg hv0]
            have h := Qgen_pow2 (m+1) k v u hkm hvl hul
            unfold Qgen at h
            rcases cdSigma_pm (m+1) v u with h1 | h1 <;>
              rcases cdSigma_pm (m+1) (v ^^^ 2^k) (u ^^^ 2^k) with h2 | h2 <;>
              rcases cdSigma_pm (m+1) v (u ^^^ 2^k) with h3 | h3 <;>
              rcases cdSigma_pm (m+1) (v ^^^ 2^k) u with h4 | h4 <;>
              rw [h1, h2, h3, h4] at h ⊢ <;> revert h <;> decide
      · -- a upper, b lower
        obtain ⟨u, hae, hul⟩ : ∃ u, a = u + 2^(m+1) ∧ u < 2^(m+1) :=
          ⟨a - 2^(m+1), by omega, by omega⟩
        have hbl : b < 2^(m+1) := by omega
        have hxa : (u + 2^(m+1)) ^^^ 2^k = (u ^^^ 2^k) + 2^(m+1) :=
          seam_xor_left u (2^k) m hul hLlt
        unfold Qgen
        rw [hae, hxa,
            R_ul u b m hul hbl, R_ul (u ^^^ 2^k) (b ^^^ 2^k) m (xorlt hul hLlt) (xorlt hbl hLlt),
            R_ul u (b ^^^ 2^k) m hul (xorlt hbl hLlt), R_ul (u ^^^ 2^k) b m (xorlt hul hLlt) hbl]
        have hL0 : (2:Nat)^k ≠ 0 := by have := Nat.two_pow_pos k; omega
        by_cases hb0 : b = 0
        · subst hb0
          simp only [Nat.zero_xor, if_true, if_neg hL0]
          have hd := deg_left (m+1) (2^k) u hLlt hul hL0
          rcases cdSigma_pm (m+1) (u ^^^ 2^k) (2^k) with h1 | h1 <;>
            rcases cdSigma_pm (m+1) u (2^k) with h2 | h2 <;>
            rw [h1, h2] at hd ⊢ <;> revert hd <;> decide
        · by_cases hbL : b ^^^ 2^k = 0
          · have heb : b = 2^k := xor_zero_eq b (2^k) hbL
            subst heb
            rw [if_neg hb0, hbL, if_pos rfl, if_pos rfl, if_neg hb0]
            have hd := deg_left (m+1) (2^k) u hLlt hul hL0
            rcases cdSigma_pm (m+1) (u ^^^ 2^k) (2^k) with h1 | h1 <;>
              rcases cdSigma_pm (m+1) u (2^k) with h2 | h2 <;>
              rw [h1, h2] at hd ⊢ <;> revert hd <;> decide
          · rw [if_neg hb0, if_neg hbL, if_neg hbL, if_neg hb0]
            have h := Qgen_pow2 (m+1) k u b hkm hul hbl
            unfold Qgen at h
            rcases cdSigma_pm (m+1) u b with h1 | h1 <;>
              rcases cdSigma_pm (m+1) (u ^^^ 2^k) (b ^^^ 2^k) with h2 | h2 <;>
              rcases cdSigma_pm (m+1) u (b ^^^ 2^k) with h3 | h3 <;>
              rcases cdSigma_pm (m+1) (u ^^^ 2^k) b with h4 | h4 <;>
              rw [h1, h2, h3, h4] at h ⊢ <;> revert h <;> decide
      · -- a lower, b upper
        obtain ⟨v, hbe, hvl⟩ : ∃ v, b = v + 2^(m+1) ∧ v < 2^(m+1) :=
          ⟨b - 2^(m+1), by omega, by omega⟩
        have hal : a < 2^(m+1) := by omega
        have hxb : (v + 2^(m+1)) ^^^ 2^k = (v ^^^ 2^k) + 2^(m+1) :=
          seam_xor_left v (2^k) m hvl hLlt
        unfold Qgen
        rw [hbe, hxb,
            R_lu a v m hal hvl, R_lu (a ^^^ 2^k) (v ^^^ 2^k) m (xorlt hal hLlt) (xorlt hvl hLlt),
            R_lu a (v ^^^ 2^k) m hal (xorlt hvl hLlt), R_lu (a ^^^ 2^k) v m (xorlt hal hLlt) hvl]
        have h := Qgen_pow2 (m+1) k v a hkm hvl hal
        unfold Qgen at h
        rcases cdSigma_pm (m+1) v a with h1 | h1 <;>
          rcases cdSigma_pm (m+1) (v ^^^ 2^k) (a ^^^ 2^k) with h2 | h2 <;>
          rcases cdSigma_pm (m+1) v (a ^^^ 2^k) with h3 | h3 <;>
          rcases cdSigma_pm (m+1) (v ^^^ 2^k) a with h4 | h4 <;>
          rw [h1, h2, h3, h4] at h ⊢ <;> revert h <;> decide
      · -- both lower
        have hal : a < 2^(m+1) := by omega
        have hbl : b < 2^(m+1) := by omega
        unfold Qgen
        rw [R_ll a b m hal hbl, R_ll (a ^^^ 2^k) (b ^^^ 2^k) m (xorlt hal hLlt) (xorlt hbl hLlt),
            R_ll a (b ^^^ 2^k) m hal (xorlt hbl hLlt), R_ll (a ^^^ 2^k) b m (xorlt hal hLlt) hbl]
        have h := Qgen_pow2 (m+1) k a b hkm hal hbl
        unfold Qgen at h
        exact h

/-! ## Tier 6: the SECOND product the inductive step of (*) lands on -/

/-- `Q'`, the product the `Y = W + H` half of the `(*)` inductive step reduces to. It differs
    from `Qgen` in two factors, whose arguments are swapped:
    `Q' W a b = sigma(a,b) sigma(b^W,a^W) sigma(b^W,a) sigma(a^W,b)`.
    Measured: `Q'` is `tau`-equivariant in its own right, and `Q = Q'` EXCEPT on the degenerate
    locus -- so the inductive step of `(*)` is a MUTUAL induction on the pair, not a single one. -/
def Qgen' (W a b m : Nat) : Int :=
  cdSigma a b m * cdSigma (b ^^^ W) (a ^^^ W) m * cdSigma (b ^^^ W) a m * cdSigma (a ^^^ W) b m

/-- **`Q` and `Q'` agree off the degenerate locus**, by two applications of `antisym` whose sign
    flips cancel. The hypotheses are exactly the degeneracies: the numerical rung measured that
    they are not removable (100% of the disagreements sit there). -/
theorem Qgen_eq_Qgen' (W a b m : Nat)
    (ha : a < 2^m) (hb : b < 2^m) (hW : W < 2^m)
    (h1 : a ^^^ W ≠ 0) (h2 : b ^^^ W ≠ 0) (h3 : (a ^^^ W) ^^^ (b ^^^ W) ≠ 0)
    (h4 : a ≠ 0) (h5 : a ^^^ (b ^^^ W) ≠ 0) :
    Qgen W a b m = Qgen' W a b m := by
  have haW : a ^^^ W < 2^m := Nat.xor_lt_two_pow ha hW
  have hbW : b ^^^ W < 2^m := Nat.xor_lt_two_pow hb hW
  have hne1 : a ^^^ W ≠ b ^^^ W := by
    intro h; exact h3 (by rw [h, Nat.xor_self])
  have hne2 : a ≠ b ^^^ W := by
    intro h; exact h5 (by rw [h, Nat.xor_self])
  have e1 : cdSigma (a ^^^ W) (b ^^^ W) m = - cdSigma (b ^^^ W) (a ^^^ W) m :=
    antisym m (a ^^^ W) (b ^^^ W) haW hbW h1 h2 hne1
  have e2 : cdSigma a (b ^^^ W) m = - cdSigma (b ^^^ W) a m :=
    antisym m a (b ^^^ W) ha hbW h4 h2 hne2
  unfold Qgen Qgen'
  rw [e1, e2]
  rcases cdSigma_pm m a b with c1 | c1 <;>
    rcases cdSigma_pm m (b ^^^ W) (a ^^^ W) with c2 | c2 <;>
    rcases cdSigma_pm m (b ^^^ W) a with c3 | c3 <;>
    rcases cdSigma_pm m (a ^^^ W) b with c4 | c4 <;>
    rw [c1, c2, c3, c4] <;> decide

/-! ## Tier 7: reduction lemmas for the mutual step (the K11 table, one lemma per case)

Six of the eight `Q`-cases hold with MINIMAL hypotheses and are proved here. The two remaining
`Q`-cases (`Y` high with `b` upper) need the FULL non-degeneracy -- measured, contract K11 -- and
the eight `Q'`-cases are not written. -/

/-- Table row `Q / Y low / ll`. -/
theorem Qred_low_ll (m W a b : Nat) (hW : W < 2^(m+1)) (ha : a < 2^(m+1)) (hb : b < 2^(m+1)) :
    Qgen W a b (m+2) = Qgen W a b (m+1) := by
  unfold Qgen
  rw [R_ll a b m ha hb, R_ll (a ^^^ W) (b ^^^ W) m (xorlt ha hW) (xorlt hb hW),
      R_ll a (b ^^^ W) m ha (xorlt hb hW), R_ll (a ^^^ W) b m (xorlt ha hW) hb]

/-- Table row `Q / Y low / lu`. -/
theorem Qred_low_lu (m W a v : Nat) (hW : W < 2^(m+1)) (ha : a < 2^(m+1)) (hv : v < 2^(m+1)) :
    Qgen W a (v + 2^(m+1)) (m+2) = Qgen W v a (m+1) := by
  have hx : (v + 2^(m+1)) ^^^ W = (v ^^^ W) + 2^(m+1) := seam_xor_left v W m hv hW
  unfold Qgen
  rw [hx, R_lu a v m ha hv, R_lu (a ^^^ W) (v ^^^ W) m (xorlt ha hW) (xorlt hv hW),
      R_lu a (v ^^^ W) m ha (xorlt hv hW), R_lu (a ^^^ W) v m (xorlt ha hW) hv]
  ac_rfl

/-- Table row `Q / Y low / ul` (needs `b ≠ 0`, `b ⊕ W ≠ 0`). -/
theorem Qred_low_ul (m W u b : Nat) (hW : W < 2^(m+1)) (hu : u < 2^(m+1)) (hb : b < 2^(m+1))
    (hb0 : b ≠ 0) (hbW : b ^^^ W ≠ 0) :
    Qgen W (u + 2^(m+1)) b (m+2) = Qgen W u b (m+1) := by
  have hx : (u + 2^(m+1)) ^^^ W = (u ^^^ W) + 2^(m+1) := seam_xor_left u W m hu hW
  unfold Qgen
  rw [hx, R_ul u b m hu hb, R_ul (u ^^^ W) (b ^^^ W) m (xorlt hu hW) (xorlt hb hW),
      R_ul u (b ^^^ W) m hu (xorlt hb hW), R_ul (u ^^^ W) b m (xorlt hu hW) hb,
      if_neg hb0, if_neg hbW, if_neg hbW, if_neg hb0]
  rcases cdSigma_pm (m+1) u b with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (u ^^^ W) (b ^^^ W) with h2 | h2 <;>
    rcases cdSigma_pm (m+1) u (b ^^^ W) with h3 | h3 <;>
    rcases cdSigma_pm (m+1) (u ^^^ W) b with h4 | h4 <;>
    rw [h1, h2, h3, h4] <;> decide

/-- Table row `Q / Y low / uu` (needs `v ≠ 0`, `v ⊕ W ≠ 0`). -/
theorem Qred_low_uu (m W u v : Nat) (hW : W < 2^(m+1)) (hu : u < 2^(m+1)) (hv : v < 2^(m+1))
    (hv0 : v ≠ 0) (hvW : v ^^^ W ≠ 0) :
    Qgen W (u + 2^(m+1)) (v + 2^(m+1)) (m+2) = Qgen W v u (m+1) := by
  have hxu : (u + 2^(m+1)) ^^^ W = (u ^^^ W) + 2^(m+1) := seam_xor_left u W m hu hW
  have hxv : (v + 2^(m+1)) ^^^ W = (v ^^^ W) + 2^(m+1) := seam_xor_left v W m hv hW
  unfold Qgen
  rw [hxu, hxv, R_uu u v m hu hv, R_uu (u ^^^ W) (v ^^^ W) m (xorlt hu hW) (xorlt hv hW),
      R_uu u (v ^^^ W) m hu (xorlt hv hW), R_uu (u ^^^ W) v m (xorlt hu hW) hv,
      if_neg hv0, if_neg hvW, if_neg hvW, if_neg hv0]
  rcases cdSigma_pm (m+1) v u with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (v ^^^ W) (u ^^^ W) with h2 | h2 <;>
    rcases cdSigma_pm (m+1) (v ^^^ W) u with h3 | h3 <;>
    rcases cdSigma_pm (m+1) v (u ^^^ W) with h4 | h4 <;>
    rw [h1, h2, h3, h4] <;> decide

/-- Table row `Q / Y high / ll` (needs `b ≠ 0`, `b ⊕ W ≠ 0`). The sign flips and the product
    becomes `Q'` -- this is the row that makes the induction MUTUAL. -/
theorem Qred_hi_ll (m W a b : Nat) (hW : W < 2^(m+1)) (ha : a < 2^(m+1)) (hb : b < 2^(m+1))
    (hb0 : b ≠ 0) (hbW : b ^^^ W ≠ 0) :
    Qgen (W + 2^(m+1)) a b (m+2) = - Qgen' W a b (m+1) := by
  have hxa : a ^^^ (W + 2^(m+1)) = (a ^^^ W) + 2^(m+1) := xor_seam a W m ha hW
  have hxb : b ^^^ (W + 2^(m+1)) = (b ^^^ W) + 2^(m+1) := xor_seam b W m hb hW
  unfold Qgen Qgen'
  rw [hxa, hxb, R_ll a b m ha hb,
      R_uu (a ^^^ W) (b ^^^ W) m (xorlt ha hW) (xorlt hb hW), if_neg hbW,
      R_lu a (b ^^^ W) m ha (xorlt hb hW),
      R_ul (a ^^^ W) b m (xorlt ha hW) hb, if_neg hb0]
  rcases cdSigma_pm (m+1) a b with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (b ^^^ W) (a ^^^ W) with h2 | h2 <;>
    rcases cdSigma_pm (m+1) (b ^^^ W) a with h3 | h3 <;>
    rcases cdSigma_pm (m+1) (a ^^^ W) b with h4 | h4 <;>
    rw [h1, h2, h3, h4] <;> decide

/-- Table row `Q / Y high / ul` (needs `b ≠ 0`, `b ⊕ W ≠ 0`). -/
theorem Qred_hi_ul (m W u b : Nat) (hW : W < 2^(m+1)) (hu : u < 2^(m+1)) (hb : b < 2^(m+1))
    (hb0 : b ≠ 0) (hbW : b ^^^ W ≠ 0) :
    Qgen (W + 2^(m+1)) (u + 2^(m+1)) b (m+2) = - Qgen' W u b (m+1) := by
  have hxa : (u + 2^(m+1)) ^^^ (W + 2^(m+1)) = u ^^^ W := xor_seam_cancel u W m hu hW
  have hxb : b ^^^ (W + 2^(m+1)) = (b ^^^ W) + 2^(m+1) := xor_seam b W m hb hW
  unfold Qgen Qgen'
  rw [hxa, hxb, R_ul u b m hu hb, if_neg hb0,
      R_lu (u ^^^ W) (b ^^^ W) m (xorlt hu hW) (xorlt hb hW),
      R_uu u (b ^^^ W) m hu (xorlt hb hW), if_neg hbW,
      R_ll (u ^^^ W) b m (xorlt hu hW) hb]
  rcases cdSigma_pm (m+1) u b with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (b ^^^ W) (u ^^^ W) with h2 | h2 <;>
    rcases cdSigma_pm (m+1) (b ^^^ W) u with h3 | h3 <;>
    rcases cdSigma_pm (m+1) (u ^^^ W) b with h4 | h4 <;>
    rw [h1, h2, h3, h4] <;> decide

/-- From `a ⊕ v ⊕ W ≠ 0`: the two distinctness side conditions the hard cases need. -/
theorem xor3_ne_left {a v W : Nat} (h : a ^^^ v ^^^ W ≠ 0) : a ≠ v ^^^ W := by
  intro he; exact h (by rw [he, Nat.xor_assoc, Nat.xor_self])

theorem xor3_ne_right {a v W : Nat} (h : a ^^^ v ^^^ W ≠ 0) : v ≠ a ^^^ W := by
  intro he
  exact h (by rw [he, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor, Nat.xor_self])

/-- Table row `Q / Y high / lu` -- one of the TWO hard cases. Here `b ⊕ Y` crosses from the
    upper half to the LOWER one, so the branch reductions that apply are different, and the
    identification with `Q'` costs two `antisym` transpositions. Their side conditions collapse
    to the single extra hypothesis `a ⊕ v ⊕ W ≠ 0`. -/
theorem Qred_hi_lu (m W a v : Nat) (hW : W < 2^(m+1)) (ha : a < 2^(m+1)) (hv : v < 2^(m+1))
    (ha0 : a ≠ 0) (hv0 : v ≠ 0) (haW : a ^^^ W ≠ 0) (hvW : v ^^^ W ≠ 0)
    (h3 : a ^^^ v ^^^ W ≠ 0) :
    Qgen (W + 2^(m+1)) a (v + 2^(m+1)) (m+2) = - Qgen' W v a (m+1) := by
  have hxa : a ^^^ (W + 2^(m+1)) = (a ^^^ W) + 2^(m+1) := xor_seam a W m ha hW
  have hxb : (v + 2^(m+1)) ^^^ (W + 2^(m+1)) = v ^^^ W := xor_seam_cancel v W m hv hW
  have e1 : cdSigma a (v ^^^ W) (m+1) = - cdSigma (v ^^^ W) a (m+1) :=
    antisym (m+1) a (v ^^^ W) ha (xorlt hv hW) ha0 hvW (xor3_ne_left h3)
  have e2 : cdSigma v (a ^^^ W) (m+1) = - cdSigma (a ^^^ W) v (m+1) :=
    antisym (m+1) v (a ^^^ W) hv (xorlt ha hW) hv0 haW (xor3_ne_right h3)
  unfold Qgen Qgen'
  rw [hxa, hxb, R_lu a v m ha hv,
      R_ul (a ^^^ W) (v ^^^ W) m (xorlt ha hW) (xorlt hv hW), if_neg hvW,
      R_ll a (v ^^^ W) m ha (xorlt hv hW),
      R_uu (a ^^^ W) v m (xorlt ha hW) hv, if_neg hv0, e1, e2]
  rcases cdSigma_pm (m+1) v a with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (a ^^^ W) (v ^^^ W) with h2 | h2 <;>
    rcases cdSigma_pm (m+1) (v ^^^ W) a with h4 | h4 <;>
    rcases cdSigma_pm (m+1) (a ^^^ W) v with h5 | h5 <;>
    rw [h1, h2, h4, h5] <;> decide

/-- Table row `Q / Y high / uu` -- the other hard case. Both `a ⊕ Y` and `b ⊕ Y` land in the
    lower half; same two transpositions, same extra hypothesis. -/
theorem Qred_hi_uu (m W u v : Nat) (hW : W < 2^(m+1)) (hu : u < 2^(m+1)) (hv : v < 2^(m+1))
    (hu0 : u ≠ 0) (hv0 : v ≠ 0) (huW : u ^^^ W ≠ 0) (hvW : v ^^^ W ≠ 0)
    (h3 : u ^^^ v ^^^ W ≠ 0) :
    Qgen (W + 2^(m+1)) (u + 2^(m+1)) (v + 2^(m+1)) (m+2) = - Qgen' W v u (m+1) := by
  have hxa : (u + 2^(m+1)) ^^^ (W + 2^(m+1)) = u ^^^ W := xor_seam_cancel u W m hu hW
  have hxb : (v + 2^(m+1)) ^^^ (W + 2^(m+1)) = v ^^^ W := xor_seam_cancel v W m hv hW
  have e1 : cdSigma u (v ^^^ W) (m+1) = - cdSigma (v ^^^ W) u (m+1) :=
    antisym (m+1) u (v ^^^ W) hu (xorlt hv hW) hu0 hvW (xor3_ne_left h3)
  have e2 : cdSigma v (u ^^^ W) (m+1) = - cdSigma (u ^^^ W) v (m+1) :=
    antisym (m+1) v (u ^^^ W) hv (xorlt hu hW) hv0 huW (xor3_ne_right h3)
  unfold Qgen Qgen'
  rw [hxa, hxb, R_uu u v m hu hv, if_neg hv0,
      R_ll (u ^^^ W) (v ^^^ W) m (xorlt hu hW) (xorlt hv hW),
      R_ul u (v ^^^ W) m hu (xorlt hv hW), if_neg hvW,
      R_lu (u ^^^ W) v m (xorlt hu hW) hv, e1, e2]
  rcases cdSigma_pm (m+1) v u with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (u ^^^ W) (v ^^^ W) with h2 | h2 <;>
    rcases cdSigma_pm (m+1) (v ^^^ W) u with h4 | h4 <;>
    rcases cdSigma_pm (m+1) (u ^^^ W) v with h5 | h5 <;>
    rw [h1, h2, h4, h5] <;> decide

/-- `a ≠ b` survives xoring both sides. -/
theorem xorW_ne {a b W : Nat} (h : a ≠ b) : a ^^^ W ≠ b ^^^ W := by
  intro he
  apply h
  have := congrArg (fun z => z ^^^ W) he
  simpa [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero] using this

/-! ### The eight `Q'` rows of the K11 table.

All eight are stated under one uniform hypothesis set -- `a, b, a ⊕ W, b ⊕ W, a ⊕ b ⊕ W` all
nonzero and `a ≠ b` -- which is sufficient for every row (contract K14). Individual rows need
less, but a uniform set keeps the eventual mutual induction's case analysis uniform too. -/

theorem Q'red_low_ll (m W a b : Nat) (hW : W < 2^(m+1)) (ha : a < 2^(m+1)) (hb : b < 2^(m+1)) :
    Qgen' W a b (m+2) = Qgen' W a b (m+1) := by
  unfold Qgen'
  rw [R_ll a b m ha hb, R_ll (b ^^^ W) (a ^^^ W) m (xorlt hb hW) (xorlt ha hW),
      R_ll (b ^^^ W) a m (xorlt hb hW) ha, R_ll (a ^^^ W) b m (xorlt ha hW) hb]

theorem Q'red_low_lu (m W a v : Nat) (hW : W < 2^(m+1)) (ha : a < 2^(m+1)) (hv : v < 2^(m+1))
    (ha0 : a ≠ 0) (haW : a ^^^ W ≠ 0) :
    Qgen' W a (v + 2^(m+1)) (m+2) = Qgen W v a (m+1) := by
  have hx : (v + 2^(m+1)) ^^^ W = (v ^^^ W) + 2^(m+1) := seam_xor_left v W m hv hW
  unfold Qgen Qgen'
  rw [hx, R_lu a v m ha hv,
      R_ul (v ^^^ W) (a ^^^ W) m (xorlt hv hW) (xorlt ha hW), if_neg haW,
      R_ul (v ^^^ W) a m (xorlt hv hW) ha, if_neg ha0,
      R_lu (a ^^^ W) v m (xorlt ha hW) hv]
  rcases cdSigma_pm (m+1) v a with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (v ^^^ W) (a ^^^ W) with h2 | h2 <;>
    rcases cdSigma_pm (m+1) v (a ^^^ W) with h3 | h3 <;>
    rcases cdSigma_pm (m+1) (v ^^^ W) a with h4 | h4 <;>
    rw [h1, h2, h3, h4] <;> decide

theorem Q'red_low_ul (m W u b : Nat) (hW : W < 2^(m+1)) (hu : u < 2^(m+1)) (hb : b < 2^(m+1))
    (hb0 : b ≠ 0) :
    Qgen' W (u + 2^(m+1)) b (m+2) = Qgen W u b (m+1) := by
  have hx : (u + 2^(m+1)) ^^^ W = (u ^^^ W) + 2^(m+1) := seam_xor_left u W m hu hW
  unfold Qgen Qgen'
  rw [hx, R_ul u b m hu hb, if_neg hb0,
      R_lu (b ^^^ W) (u ^^^ W) m (xorlt hb hW) (xorlt hu hW),
      R_lu (b ^^^ W) u m (xorlt hb hW) hu,
      R_ul (u ^^^ W) b m (xorlt hu hW) hb, if_neg hb0]
  rcases cdSigma_pm (m+1) u b with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (u ^^^ W) (b ^^^ W) with h2 | h2 <;>
    rcases cdSigma_pm (m+1) u (b ^^^ W) with h3 | h3 <;>
    rcases cdSigma_pm (m+1) (u ^^^ W) b with h4 | h4 <;>
    rw [h1, h2, h3, h4] <;> decide

theorem Q'red_low_uu (m W u v : Nat) (hW : W < 2^(m+1)) (hu : u < 2^(m+1)) (hv : v < 2^(m+1))
    (hu0 : u ≠ 0) (hv0 : v ≠ 0) (huW : u ^^^ W ≠ 0) (hvW : v ^^^ W ≠ 0)
    (h3 : u ^^^ v ^^^ W ≠ 0) :
    Qgen' W (u + 2^(m+1)) (v + 2^(m+1)) (m+2) = Qgen' W v u (m+1) := by
  have hxu : (u + 2^(m+1)) ^^^ W = (u ^^^ W) + 2^(m+1) := seam_xor_left u W m hu hW
  have hxv : (v + 2^(m+1)) ^^^ W = (v ^^^ W) + 2^(m+1) := seam_xor_left v W m hv hW
  have e1 : cdSigma u (v ^^^ W) (m+1) = - cdSigma (v ^^^ W) u (m+1) :=
    antisym (m+1) u (v ^^^ W) hu (xorlt hv hW) hu0 hvW (xor3_ne_left h3)
  have e2 : cdSigma v (u ^^^ W) (m+1) = - cdSigma (u ^^^ W) v (m+1) :=
    antisym (m+1) v (u ^^^ W) hv (xorlt hu hW) hv0 huW (xor3_ne_right h3)
  unfold Qgen'
  rw [hxu, hxv, R_uu u v m hu hv, if_neg hv0,
      R_uu (v ^^^ W) (u ^^^ W) m (xorlt hv hW) (xorlt hu hW), if_neg huW,
      R_uu (v ^^^ W) u m (xorlt hv hW) hu, if_neg hu0,
      R_uu (u ^^^ W) v m (xorlt hu hW) hv, if_neg hv0, e1, e2]
  rcases cdSigma_pm (m+1) v u with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (u ^^^ W) (v ^^^ W) with h2 | h2 <;>
    rcases cdSigma_pm (m+1) (v ^^^ W) u with h4 | h4 <;>
    rcases cdSigma_pm (m+1) (u ^^^ W) v with h5 | h5 <;>
    rw [h1, h2, h4, h5] <;> decide

theorem Q'red_hi_ll (m W a b : Nat) (hW : W < 2^(m+1)) (ha : a < 2^(m+1)) (hb : b < 2^(m+1))
    (ha0 : a ≠ 0) (hb0 : b ≠ 0) (haW : a ^^^ W ≠ 0) (hbW : b ^^^ W ≠ 0) (hab : a ≠ b) :
    Qgen' (W + 2^(m+1)) a b (m+2) = - Qgen' W a b (m+1) := by
  have hxa : a ^^^ (W + 2^(m+1)) = (a ^^^ W) + 2^(m+1) := xor_seam a W m ha hW
  have hxb : b ^^^ (W + 2^(m+1)) = (b ^^^ W) + 2^(m+1) := xor_seam b W m hb hW
  have e1 : cdSigma (a ^^^ W) (b ^^^ W) (m+1) = - cdSigma (b ^^^ W) (a ^^^ W) (m+1) :=
    antisym (m+1) (a ^^^ W) (b ^^^ W) (xorlt ha hW) (xorlt hb hW) haW hbW (xorW_ne hab)
  unfold Qgen'
  rw [hxa, hxb, R_ll a b m ha hb,
      R_uu (b ^^^ W) (a ^^^ W) m (xorlt hb hW) (xorlt ha hW), if_neg haW,
      R_ul (b ^^^ W) a m (xorlt hb hW) ha, if_neg ha0,
      R_ul (a ^^^ W) b m (xorlt ha hW) hb, if_neg hb0, e1]
  rcases cdSigma_pm (m+1) a b with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (b ^^^ W) (a ^^^ W) with h2 | h2 <;>
    rcases cdSigma_pm (m+1) (b ^^^ W) a with h3 | h3 <;>
    rcases cdSigma_pm (m+1) (a ^^^ W) b with h4 | h4 <;>
    rw [h1, h2, h3, h4] <;> decide

theorem Q'red_hi_lu (m W a v : Nat) (hW : W < 2^(m+1)) (ha : a < 2^(m+1)) (hv : v < 2^(m+1))
    (hv0 : v ≠ 0) (haW : a ^^^ W ≠ 0) (hvW : v ^^^ W ≠ 0) (hav : a ≠ v) :
    Qgen' (W + 2^(m+1)) a (v + 2^(m+1)) (m+2) = - Qgen W v a (m+1) := by
  have hxa : a ^^^ (W + 2^(m+1)) = (a ^^^ W) + 2^(m+1) := xor_seam a W m ha hW
  have hxb : (v + 2^(m+1)) ^^^ (W + 2^(m+1)) = v ^^^ W := xor_seam_cancel v W m hv hW
  have e1 : cdSigma (a ^^^ W) (v ^^^ W) (m+1) = - cdSigma (v ^^^ W) (a ^^^ W) (m+1) :=
    antisym (m+1) (a ^^^ W) (v ^^^ W) (xorlt ha hW) (xorlt hv hW) haW hvW (xorW_ne hav)
  unfold Qgen Qgen'
  rw [hxa, hxb, R_lu a v m ha hv,
      R_lu (v ^^^ W) (a ^^^ W) m (xorlt hv hW) (xorlt ha hW),
      R_ll (v ^^^ W) a m (xorlt hv hW) ha,
      R_uu (a ^^^ W) v m (xorlt ha hW) hv, if_neg hv0, e1]
  rcases cdSigma_pm (m+1) v a with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (v ^^^ W) (a ^^^ W) with h2 | h2 <;>
    rcases cdSigma_pm (m+1) (v ^^^ W) a with h3 | h3 <;>
    rcases cdSigma_pm (m+1) v (a ^^^ W) with h4 | h4 <;>
    rw [h1, h2, h3, h4] <;> decide

theorem Q'red_hi_ul (m W u b : Nat) (hW : W < 2^(m+1)) (hu : u < 2^(m+1)) (hb : b < 2^(m+1))
    (hu0 : u ≠ 0) (hb0 : b ≠ 0) (huW : u ^^^ W ≠ 0) (hbW : b ^^^ W ≠ 0) (hub : u ≠ b) :
    Qgen' (W + 2^(m+1)) (u + 2^(m+1)) b (m+2) = - Qgen W u b (m+1) := by
  have hxa : (u + 2^(m+1)) ^^^ (W + 2^(m+1)) = u ^^^ W := xor_seam_cancel u W m hu hW
  have hxb : b ^^^ (W + 2^(m+1)) = (b ^^^ W) + 2^(m+1) := xor_seam b W m hb hW
  have e1 : cdSigma (b ^^^ W) (u ^^^ W) (m+1) = - cdSigma (u ^^^ W) (b ^^^ W) (m+1) :=
    antisym (m+1) (b ^^^ W) (u ^^^ W) (xorlt hb hW) (xorlt hu hW) hbW huW
      (fun h => hub (xorW_ne (fun hh => hub hh.symm) h).elim)
  unfold Qgen Qgen'
  rw [hxa, hxb, R_ul u b m hu hb, if_neg hb0,
      R_ul (b ^^^ W) (u ^^^ W) m (xorlt hb hW) (xorlt hu hW), if_neg huW,
      R_uu (b ^^^ W) u m (xorlt hb hW) hu, if_neg hu0,
      R_ll (u ^^^ W) b m (xorlt hu hW) hb, e1]
  rcases cdSigma_pm (m+1) u b with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (u ^^^ W) (b ^^^ W) with h2 | h2 <;>
    rcases cdSigma_pm (m+1) u (b ^^^ W) with h3 | h3 <;>
    rcases cdSigma_pm (m+1) (u ^^^ W) b with h4 | h4 <;>
    rw [h1, h2, h3, h4] <;> decide

theorem Q'red_hi_uu (m W u v : Nat) (hW : W < 2^(m+1)) (hu : u < 2^(m+1)) (hv : v < 2^(m+1))
    (hu0 : u ≠ 0) (hv0 : v ≠ 0) (huW : u ^^^ W ≠ 0) (hvW : v ^^^ W ≠ 0)
    (huv : u ≠ v) (h3 : u ^^^ v ^^^ W ≠ 0) :
    Qgen' (W + 2^(m+1)) (u + 2^(m+1)) (v + 2^(m+1)) (m+2) = - Qgen' W v u (m+1) := by
  have hxa : (u + 2^(m+1)) ^^^ (W + 2^(m+1)) = u ^^^ W := xor_seam_cancel u W m hu hW
  have hxb : (v + 2^(m+1)) ^^^ (W + 2^(m+1)) = v ^^^ W := xor_seam_cancel v W m hv hW
  have e1 : cdSigma (v ^^^ W) (u ^^^ W) (m+1) = - cdSigma (u ^^^ W) (v ^^^ W) (m+1) :=
    antisym (m+1) (v ^^^ W) (u ^^^ W) (xorlt hv hW) (xorlt hu hW) hvW huW
      (xorW_ne (fun h => huv h.symm))
  have e2 : cdSigma u (v ^^^ W) (m+1) = - cdSigma (v ^^^ W) u (m+1) :=
    antisym (m+1) u (v ^^^ W) hu (xorlt hv hW) hu0 hvW (xor3_ne_left h3)
  have e3 : cdSigma v (u ^^^ W) (m+1) = - cdSigma (u ^^^ W) v (m+1) :=
    antisym (m+1) v (u ^^^ W) hv (xorlt hu hW) hv0 huW (xor3_ne_right h3)
  unfold Qgen'
  rw [hxa, hxb, R_uu u v m hu hv, if_neg hv0,
      R_ll (v ^^^ W) (u ^^^ W) m (xorlt hv hW) (xorlt hu hW),
      R_lu (v ^^^ W) u m (xorlt hv hW) hu,
      R_lu (u ^^^ W) v m (xorlt hu hW) hv, e1, e2, e3]
  rcases cdSigma_pm (m+1) v u with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (u ^^^ W) (v ^^^ W) with h2 | h2 <;>
    rcases cdSigma_pm (m+1) (v ^^^ W) u with h4 | h4 <;>
    rcases cdSigma_pm (m+1) (u ^^^ W) v with h5 | h5 <;>
    rw [h1, h2, h4, h5] <;> decide

/-! ## Tier 8: (★) for single-bit labels — equivariance from the constant base case -/

/-- Swap bits 0 and `j` of `x` (identity when they already agree). Matches the contract's `sw`. -/
def sw (x j : Nat) : Nat :=
  if (x % 2) = ((x / 2^j) % 2) then x else x ^^^ (1 ^^^ (2^j))
  -- Note: for j > 0, bits 0 and j are distinct so 1 ||| 2^j = 1 ^^^ 2^j.
  -- We use xor form directly to avoid relying on `Nat.or_eq_xor_*`.

/-- Bit 0 of a pure power `2^k` is unset for `k > 0`. -/
private theorem pow2_bit0 (k : Nat) (hk : k ≠ 0) : (2^k) % 2 = 0 := by
  cases k with
  | zero => exact absurd rfl hk
  | succ k' =>
    -- 2^(k'+1) = 2^k' * 2, and anything * 2 is 0 mod 2
    rw [Nat.pow_succ]
    omega

/-- Bit `k` of `2^k` is set. -/
private theorem pow2_bitk (k : Nat) : (2^k / 2^k) % 2 = 1 := by
  have : 2^k / 2^k = 1 := Nat.div_self (Nat.two_pow_pos k)
  rw [this]

/-- `sw (2^k) k = 1` for `k > 0`: bits 0 and k disagree, so the swap yields pure bit 0. -/
theorem sw_pow2 (k : Nat) (hk : k ≠ 0) : sw (2^k) k = 1 := by
  unfold sw
  have hb0 : (2^k % 2) = 0 := pow2_bit0 k hk
  have hbj : ((2^k / 2^k) % 2) = 1 := pow2_bitk k
  rw [hb0, hbj]
  have : ¬ ((0 : Nat) = 1) := by decide
  simp only [this, ↓reduceIte]
  -- Goal: 2^k ^^^ (1 ^^^ 2^k) = 1
  calc
    2^k ^^^ (1 ^^^ 2^k) = 2^k ^^^ (2^k ^^^ 1) := by rw [Nat.xor_comm 1 (2^k)]
    _ = (2^k ^^^ 2^k) ^^^ 1 := by rw [← Nat.xor_assoc]
    _ = 0 ^^^ 1 := by rw [Nat.xor_self]
    _ = 1 := by rw [Nat.zero_xor]

/-- `sw x 0 = x` — bit 0 swapped with itself is a no-op. -/
theorem sw_zero (x : Nat) : sw x 0 = x := by
  unfold sw
  -- (x/1)%2 = x%2, so the bits agree
  simp [Nat.div_one]

/-- `sw` never increases the bit-width when `j < m`.
    Proof: the mask `1 ^^^ 2^j` only touches bits 0 and j, both `< m`. -/
theorem sw_lt (x j m : Nat) (hx : x < 2^m) (hj : j < m) : sw x j < 2^m := by
  unfold sw
  by_cases h : (x % 2) = ((x / 2^j) % 2)
  · simpa [h]
  · simp only [h, ↓reduceIte]
    -- x ^^^ (1 ^^^ 2^j) < 2^m
    have hj2 : 2^j < 2^m := Nat.pow_lt_pow_right (by decide : (1:Nat) < 2) hj
    have h1 : (1 : Nat) < 2^m := by
      have : m ≠ 0 := by omega
      exact Nat.one_lt_two_pow this
    have hmask : (1 : Nat) ^^^ (2^j) < 2^m := Nat.xor_lt_two_pow h1 hj2
    exact Nat.xor_lt_two_pow hx hmask

/-- **(★) for single-bit labels, proven ∀n.** Both sides of the equivariance are the
    constant `-1` (`Qgen_pow2`), so (★) holds for every pure power-of-two label.
    This is the equivariant reading of the base case — not just `Q = -1`, but
    `Q_Y(a,b) = Q_{τY}(τa,τb)` when `Y = 2^k`. Multi-bit `Y` still needs the mutual step. -/
theorem star_pow2 (m k a b : Nat) (hk : k < m) (ha : a < 2^m) (hb : b < 2^m) :
    Qgen (2^k) a b m = Qgen (sw (2^k) k) (sw a k) (sw b k) m := by
  by_cases hk0 : k = 0
  · -- j = 0 ⇒ τ = id, (★) is reflexivity
    subst hk0
    simp only [sw_zero]
  · -- j > 0 ⇒ τY = 1, both products are -1 by Qgen_pow2
    have hY : sw (2^k) k = 1 := sw_pow2 k hk0
    rw [hY]
    have hL : Qgen (2^k) a b m = -1 := Qgen_pow2 m k a b hk ha hb
    have hR : Qgen (2^0) (sw a k) (sw b k) m = -1 :=
      Qgen_pow2 m 0 (sw a k) (sw b k) (by omega)
        (sw_lt a k m ha hk) (sw_lt b k m hb hk)
    -- rewrite RHS target 1 as 2^0
    have hone : (1 : Nat) = 2^0 := rfl
    rw [hone]
    rw [hL, hR]

/-! ## Tier 8: the degenerate locus -/

/-- **`Q` is identically `-1` on the whole degenerate locus, ∀n.** Every one of the six
    degeneracies collapses to `deg_left`, `deg_right` or a `sigma_self`/`antisym` pair. This is
    what closes the degenerate branches of `(*)`: both sides are the same constant, so no
    induction is needed there. (`W ≠ 0` is required -- at `W = 0` the four factors coincide and
    `Q = +1`.) -/
theorem Qgen_degen (m W a b : Nat) (hW : W < 2^m) (ha : a < 2^m) (hb : b < 2^m) (hW0 : W ≠ 0)
    (hd : a = 0 ∨ b = 0 ∨ a ^^^ W = 0 ∨ b ^^^ W = 0 ∨ a = b ∨ a ^^^ b ^^^ W = 0) :
    Qgen W a b m = -1 := by
  have hm : ∃ m', m = m' + 1 := by
    cases m with
    | zero => exact absurd (by omega : W = 0) hW0
    | succ k => exact ⟨k, rfl⟩
  obtain ⟨m', rfl⟩ := hm
  by_cases ha0 : a = 0
  · subst ha0
    unfold Qgen
    rw [Nat.zero_xor, cdSig0, cdSig0]
    have := deg_right (m'+1) W b hW hb hW0
    rcases cdSigma_pm (m'+1) W (b ^^^ W) with h1 | h1 <;>
      rcases cdSigma_pm (m'+1) W b with h2 | h2 <;>
      rw [h1, h2] at this ⊢ <;> revert this <;> decide
  by_cases hb0 : b = 0
  · subst hb0
    unfold Qgen
    rw [Nat.zero_xor, cdSig0', cdSig0']
    have := deg_left (m'+1) W a hW ha hW0
    rcases cdSigma_pm (m'+1) (a ^^^ W) W with h1 | h1 <;>
      rcases cdSigma_pm (m'+1) a W with h2 | h2 <;>
      rw [h1, h2] at this ⊢ <;> revert this <;> decide
  by_cases haW : a ^^^ W = 0
  · have hea : a = W := xor_zero_eq a W haW
    subst hea
    unfold Qgen
    rw [haW, cdSig0, cdSig0]
    have := deg_right (m'+1) a b hW hb hW0
    rcases cdSigma_pm (m'+1) a (b ^^^ a) with h1 | h1 <;>
      rcases cdSigma_pm (m'+1) a b with h2 | h2 <;>
      rw [h1, h2] at this ⊢ <;> revert this <;> decide
  by_cases hbW : b ^^^ W = 0
  · have heb : b = W := xor_zero_eq b W hbW
    subst heb
    unfold Qgen
    rw [hbW, cdSig0', cdSig0']
    have := deg_left (m'+1) b a hW ha hW0
    rcases cdSigma_pm (m'+1) (a ^^^ b) b with h1 | h1 <;>
      rcases cdSigma_pm (m'+1) a b with h2 | h2 <;>
      rw [h1, h2] at this ⊢ <;> revert this <;> decide
  -- the remaining two degeneracies, with all four of a, b, a^W, b^W nonzero
  have haWne : a ≠ a ^^^ W := by
    intro h
    apply hW0
    have hh : a ^^^ a = a ^^^ (a ^^^ W) := congrArg (fun z => a ^^^ z) h
    rw [Nat.xor_self, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor] at hh
    exact hh.symm
  rcases hd with h | h | h | h | h | h
  · exact absurd h ha0
  · exact absurd h hb0
  · exact absurd h haW
  · exact absurd h hbW
  · subst h
    unfold Qgen
    rw [sigma_self (m'+1) a ha ha0, sigma_self (m'+1) (a ^^^ W) (Nat.xor_lt_two_pow ha hW) haW,
        antisym (m'+1) a (a ^^^ W) ha (Nat.xor_lt_two_pow ha hW) ha0 haW haWne]
    rcases cdSigma_pm (m'+1) (a ^^^ W) a with h1 | h1 <;> rw [h1] <;> decide
  · have hab : a ^^^ b = W := xor_zero_eq (a ^^^ b) W h
    have heb : b = a ^^^ W := by
      have hh : a ^^^ (a ^^^ b) = a ^^^ W := congrArg (fun z => a ^^^ z) hab
      rwa [← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor] at hh
    have hcancel : (a ^^^ W) ^^^ W = a := by rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
    rw [heb]
    unfold Qgen
    rw [hcancel,
        sigma_self (m'+1) a ha ha0,
        sigma_self (m'+1) (a ^^^ W) (Nat.xor_lt_two_pow ha hW) haW,
        antisym (m'+1) a (a ^^^ W) ha (Nat.xor_lt_two_pow ha hW) ha0 haW haWne]
    rcases cdSigma_pm (m'+1) (a ^^^ W) a with h1 | h1 <;> rw [h1] <;> decide

/-- `sw 0 j = 0` for every j. -/
theorem sw_map_zero (j : Nat) : sw 0 j = 0 := by
  unfold sw; simp

/-- Degenerate (★) when BOTH sides are degenerate: both products are `-1`.
    Together with `Qgen_degen` and the fact that `τ` preserves the six degeneracies
    (linear injective bit-swap; measured K16), this closes the degenerate half of the
    mutual assembly without induction. The non-degenerate half still needs the 16-case
    table assembled under the induction hypothesis. -/
theorem star_both_degen (m Y a b j : Nat)
    (hY : Y < 2^m) (ha : a < 2^m) (hb : b < 2^m) (hY0 : Y ≠ 0)
    (hYa : sw Y j < 2^m) (ha' : sw a j < 2^m) (hb' : sw b j < 2^m)
    (hY0' : sw Y j ≠ 0)
    (hd : a = 0 ∨ b = 0 ∨ a ^^^ Y = 0 ∨ b ^^^ Y = 0 ∨ a = b ∨ a ^^^ b ^^^ Y = 0)
    (hd' : sw a j = 0 ∨ sw b j = 0 ∨ sw a j ^^^ sw Y j = 0 ∨ sw b j ^^^ sw Y j = 0
         ∨ sw a j = sw b j ∨ sw a j ^^^ sw b j ^^^ sw Y j = 0) :
    Qgen Y a b m = Qgen (sw Y j) (sw a j) (sw b j) m := by
  have hL : Qgen Y a b m = -1 := Qgen_degen m Y a b hY ha hb hY0 hd
  have hR : Qgen (sw Y j) (sw a j) (sw b j) m = -1 :=
    Qgen_degen m (sw Y j) (sw a j) (sw b j) hYa ha' hb' hY0' hd'
  rw [hL, hR]

/-! ## Tier 9: the GAP lemma -- reduced-degenerate tuples also give `-1`

The reduction lemmas' hypotheses are about the REDUCED arguments; `Qgen_degen` is about the
CURRENT ones, and they do not coincide. The gap is exactly the tuples whose reduced form is
degenerate without the current one being so -- equivalently (contract K17), those where one of
`a, b, a⊕Y, b⊕Y, a⊕b, a⊕b⊕Y` equals the seam bit `H` rather than `0`. They give the same
constant. Here is the central case, `b = H`. -/

theorem Qgen_H_right_low (m W a : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0) (ha : a < 2^(m+2)) :
    Qgen W a (2^(m+1)) (m+2) = -1 := by
  have hpos : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have h2H : 2^(m+2) = 2^(m+1) + 2^(m+1) := by rw [Nat.pow_succ]; omega
  have hHW : (2:Nat)^(m+1) ^^^ W = W + 2^(m+1) := by
    rw [Nat.xor_comm]; exact (seam_add_xor W m hW).symm
  by_cases haU : a ≥ 2^(m+1)
  · obtain ⟨u, hae, hul⟩ : ∃ u, a = u + 2^(m+1) ∧ u < 2^(m+1) :=
      ⟨a - 2^(m+1), by omega, by omega⟩
    have hxa : (u + 2^(m+1)) ^^^ W = (u ^^^ W) + 2^(m+1) := seam_xor_left u W m hul hW
    have e1 : cdSigma (u + 2^(m+1)) (2^(m+1)) (m+2) = -1 := by
      have h := R_uu u 0 m hul hpos; rw [Nat.zero_add] at h; rw [h, if_pos rfl]
    have e4 : cdSigma ((u ^^^ W) + 2^(m+1)) (2^(m+1)) (m+2) = -1 := by
      have h := R_uu (u ^^^ W) 0 m (xorlt hul hW) hpos; rw [Nat.zero_add] at h
      rw [h, if_pos rfl]
    have e2 : cdSigma ((u ^^^ W) + 2^(m+1)) (W + 2^(m+1)) (m+2) = cdSigma W (u ^^^ W) (m+1) := by
      rw [R_uu (u ^^^ W) W m (xorlt hul hW) hW, if_neg hW0]
    have e3 : cdSigma (u + 2^(m+1)) (W + 2^(m+1)) (m+2) = cdSigma W u (m+1) := by
      rw [R_uu u W m hul hW, if_neg hW0]
    unfold Qgen
    rw [hae, hxa, hHW, e1, e2, e3, e4]
    have hd := deg_right (m+1) W u hW hul hW0
    rcases cdSigma_pm (m+1) W (u ^^^ W) with h1 | h1 <;>
      rcases cdSigma_pm (m+1) W u with h2 | h2 <;>
      rw [h1, h2] at hd ⊢ <;> revert hd <;> decide
  · have hal : a < 2^(m+1) := by omega
    have e1 : cdSigma a (2^(m+1)) (m+2) = 1 := by
      have h := R_lu a 0 m hal hpos; rw [Nat.zero_add] at h; rw [h, cdSig0]
    have e4 : cdSigma (a ^^^ W) (2^(m+1)) (m+2) = 1 := by
      have h := R_lu (a ^^^ W) 0 m (xorlt hal hW) hpos; rw [Nat.zero_add] at h
      rw [h, cdSig0]
    have e2 : cdSigma (a ^^^ W) (W + 2^(m+1)) (m+2) = cdSigma W (a ^^^ W) (m+1) :=
      R_lu (a ^^^ W) W m (xorlt hal hW) hW
    have e3 : cdSigma a (W + 2^(m+1)) (m+2) = cdSigma W a (m+1) := R_lu a W m hal hW
    unfold Qgen
    rw [hHW, e1, e2, e3, e4]
    have hd := deg_right (m+1) W a hW hal hW0
    rcases cdSigma_pm (m+1) W (a ^^^ W) with h1 | h1 <;>
      rcases cdSigma_pm (m+1) W a with h2 | h2 <;>
      rw [h1, h2] at hd ⊢ <;> revert hd <;> decide

theorem Qgen_H_right_hi (m W a : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0) (ha : a < 2^(m+2)) :
    Qgen (W + 2^(m+1)) a (2^(m+1)) (m+2) = -1 := by
  have hpos : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have h2H : 2^(m+2) = 2^(m+1) + 2^(m+1) := by rw [Nat.pow_succ]; omega
  have hHY : (2:Nat)^(m+1) ^^^ (W + 2^(m+1)) = W := by
    rw [seam_add_xor W m hW, Nat.xor_comm W (2^(m+1)), ← Nat.xor_assoc, Nat.xor_self,
        Nat.zero_xor]
  by_cases haU : a ≥ 2^(m+1)
  · obtain ⟨u, hae, hul⟩ : ∃ u, a = u + 2^(m+1) ∧ u < 2^(m+1) :=
      ⟨a - 2^(m+1), by omega, by omega⟩
    have hxa : (u + 2^(m+1)) ^^^ (W + 2^(m+1)) = u ^^^ W := xor_seam_cancel u W m hul hW
    have e1 : cdSigma (u + 2^(m+1)) (2^(m+1)) (m+2) = -1 := by
      have h := R_uu u 0 m hul hpos; rw [Nat.zero_add] at h; rw [h, if_pos rfl]
    have e2 : cdSigma (u ^^^ W) W (m+2) = cdSigma (u ^^^ W) W (m+1) :=
      R_ll (u ^^^ W) W m (xorlt hul hW) hW
    have e3 : cdSigma (u + 2^(m+1)) W (m+2) = - cdSigma u W (m+1) := by
      rw [R_ul u W m hul hW, if_neg hW0]
    have e4 : cdSigma (u ^^^ W) (2^(m+1)) (m+2) = 1 := by
      have h := R_lu (u ^^^ W) 0 m (xorlt hul hW) hpos; rw [Nat.zero_add] at h
      rw [h, cdSig0]
    unfold Qgen
    rw [hae, hxa, hHY, e1, e2, e3, e4]
    have hd := deg_left (m+1) W u hW hul hW0
    rcases cdSigma_pm (m+1) (u ^^^ W) W with h1 | h1 <;>
      rcases cdSigma_pm (m+1) u W with h2 | h2 <;>
      rw [h1, h2] at hd ⊢ <;> revert hd <;> decide
  · have hal : a < 2^(m+1) := by omega
    have hxa : a ^^^ (W + 2^(m+1)) = (a ^^^ W) + 2^(m+1) := xor_seam a W m hal hW
    have e1 : cdSigma a (2^(m+1)) (m+2) = 1 := by
      have h := R_lu a 0 m hal hpos; rw [Nat.zero_add] at h; rw [h, cdSig0]
    have e2 : cdSigma ((a ^^^ W) + 2^(m+1)) W (m+2) = - cdSigma (a ^^^ W) W (m+1) := by
      rw [R_ul (a ^^^ W) W m (xorlt hal hW) hW, if_neg hW0]
    have e3 : cdSigma a W (m+2) = cdSigma a W (m+1) := R_ll a W m hal hW
    have e4 : cdSigma ((a ^^^ W) + 2^(m+1)) (2^(m+1)) (m+2) = -1 := by
      have h := R_uu (a ^^^ W) 0 m (xorlt hal hW) hpos; rw [Nat.zero_add] at h
      rw [h, if_pos rfl]
    unfold Qgen
    rw [hxa, hHY, e1, e2, e3, e4]
    have hd := deg_left (m+1) W a hW hal hW0
    rcases cdSigma_pm (m+1) (a ^^^ W) W with h1 | h1 <;>
      rcases cdSigma_pm (m+1) a W with h2 | h2 <;>
      rw [h1, h2] at hd ⊢ <;> revert hd <;> decide

/-- Gap corollary, `Y` below the seam: `b ⊕ Y = H` reduces to `b = H` by `Qgen_coset_right`. -/
theorem Qgen_H_right_low' (m W a b : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (ha : a < 2^(m+2)) (hbe : b ^^^ W = 2^(m+1)) : Qgen W a b (m+2) = -1 := by
  rw [Qgen_coset_right, hbe]
  exact Qgen_H_right_low m W a hW hW0 ha

/-- Gap corollary, `Y` above the seam. -/
theorem Qgen_H_right_hi' (m W a b : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (ha : a < 2^(m+2)) (hbe : b ^^^ (W + 2^(m+1)) = 2^(m+1)) :
    Qgen (W + 2^(m+1)) a b (m+2) = -1 := by
  rw [Qgen_coset_right, hbe]
  exact Qgen_H_right_hi m W a hW hW0 ha


/-! ## Tier 10: gap roots `a = H` and `a ⊕ b = H`

The six `= H` conditions have three roots (K19). Two were already closed:
`b = H` and `b ⊕ Y = H`. The remaining roots are `a = H` (proved here by the dual
case analysis of `Qgen_H_right_*`) and `a ⊕ b = H` (proved by folding to the reduced
self-pair via `Qred_low_lu` / `Qred_low_ul`). Coset then doubles each. -/

private theorem h2pow_succ_add (m : Nat) : 2^(m+2) = 2^(m+1) + 2^(m+1) := by
  rw [Nat.pow_succ]; omega

/-- Gap root `a = H`, `Y` below the seam. Dual of `Qgen_H_right_low`. -/
theorem Qgen_H_left_low (m W b : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0) (hb : b < 2^(m+2)) :
    Qgen W (2^(m+1)) b (m+2) = -1 := by
  have h2H := h2pow_succ_add m
  have hWm : W < 2^(m+2) := by omega
  have hH : (2:Nat)^(m+1) < 2^(m+2) := by omega
  have hpos : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have hHW : (2:Nat)^(m+1) ^^^ W = W + 2^(m+1) := by
    rw [Nat.xor_comm]; exact (seam_add_xor W m hW).symm
  -- Edge cases that are already current-degenerate.
  by_cases hb0 : b = 0
  · subst hb0
    exact Qgen_degen (m+2) W (2^(m+1)) 0 hWm hH (by omega) hW0 (Or.inr (Or.inl rfl))
  by_cases hbH : b = 2^(m+1)
  · subst hbH
    exact Qgen_degen (m+2) W (2^(m+1)) (2^(m+1)) hWm hH hH hW0
      (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
  by_cases hbW : b = W
  · rw [hbW]
    exact Qgen_degen (m+2) W (2^(m+1)) W hWm hH (by omega) hW0
      (Or.inr (Or.inr (Or.inr (Or.inl (Nat.xor_self W)))))
  by_cases hbWH : b = W + 2^(m+1)
  · have hd : (2:Nat)^(m+1) ^^^ (W + 2^(m+1)) ^^^ W = 0 := by
      have h1 : (2:Nat)^(m+1) ^^^ (W + 2^(m+1)) = W := by
        rw [seam_add_xor W m hW, Nat.xor_comm W (2^(m+1)), ← Nat.xor_assoc, Nat.xor_self,
            Nat.zero_xor]
      rw [h1, Nat.xor_self]
    rw [hbWH]
    exact Qgen_degen (m+2) W (2^(m+1)) (W + 2^(m+1)) hWm hH (by omega) hW0
      (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr hd)))))
  -- Residual non-degenerate gap: split on b's half.
  by_cases hbU : b ≥ 2^(m+1)
  · obtain ⟨v, hbe, hvl⟩ : ∃ v, b = v + 2^(m+1) ∧ v < 2^(m+1) :=
      ⟨b - 2^(m+1), by omega, by omega⟩
    have hv0 : v ≠ 0 := by intro h; apply hbH; rw [hbe, h]; simp
    have hvW : v ^^^ W ≠ 0 := by
      intro h; have he := xor_zero_eq v W h; apply hbWH; rw [hbe, he]
    have hxb : (v + 2^(m+1)) ^^^ W = (v ^^^ W) + 2^(m+1) := seam_xor_left v W m hvl hW
    have e1 : cdSigma (2^(m+1)) (v + 2^(m+1)) (m+2) = 1 := by
      have h := R_uu 0 v m hpos hvl
      rw [Nat.zero_add] at h; rw [h, if_neg hv0, cdSig0']
    have e4 : cdSigma (W + 2^(m+1)) (v + 2^(m+1)) (m+2) = cdSigma v W (m+1) := by
      have h := R_uu W v m hW hvl; rw [h, if_neg hv0]
    have e2 : cdSigma (W + 2^(m+1)) ((v ^^^ W) + 2^(m+1)) (m+2) =
        cdSigma (v ^^^ W) W (m+1) := by
      have h := R_uu W (v ^^^ W) m hW (xorlt hvl hW); rw [h, if_neg hvW]
    have e3 : cdSigma (2^(m+1)) ((v ^^^ W) + 2^(m+1)) (m+2) = 1 := by
      have h := R_uu 0 (v ^^^ W) m hpos (xorlt hvl hW)
      rw [Nat.zero_add] at h; rw [h, if_neg hvW, cdSig0']
    unfold Qgen
    rw [hbe, hxb, hHW, e1, e2, e3, e4]
    have hd := deg_left (m+1) W v hW hvl hW0
    rcases cdSigma_pm (m+1) (v ^^^ W) W with h1 | h1 <;>
      rcases cdSigma_pm (m+1) v W with h2 | h2 <;>
      rw [h1, h2] at hd ⊢ <;> revert hd <;> decide
  · have hbl : b < 2^(m+1) := by omega
    have hbW' : b ^^^ W ≠ 0 := by intro h; exact hbW (xor_zero_eq b W h)
    have e1 : cdSigma (2^(m+1)) b (m+2) = -1 := by
      have h := R_ul 0 b m hpos hbl
      rw [Nat.zero_add] at h; rw [h, if_neg hb0, cdSig0]
    have e4 : cdSigma (W + 2^(m+1)) b (m+2) = - cdSigma W b (m+1) := by
      have h := R_ul W b m hW hbl; rw [h, if_neg hb0]
    have e2 : cdSigma (W + 2^(m+1)) (b ^^^ W) (m+2) = - cdSigma W (b ^^^ W) (m+1) := by
      have h := R_ul W (b ^^^ W) m hW (xorlt hbl hW); rw [h, if_neg hbW']
    have e3 : cdSigma (2^(m+1)) (b ^^^ W) (m+2) = -1 := by
      have h := R_ul 0 (b ^^^ W) m hpos (xorlt hbl hW)
      rw [Nat.zero_add] at h; rw [h, if_neg hbW', cdSig0]
    unfold Qgen
    rw [hHW, e1, e2, e3, e4]
    have hd := deg_right (m+1) W b hW hbl hW0
    rcases cdSigma_pm (m+1) W (b ^^^ W) with h1 | h1 <;>
      rcases cdSigma_pm (m+1) W b with h2 | h2 <;>
      rw [h1, h2] at hd ⊢ <;> revert hd <;> decide

/-- Gap root `a = H`, `Y` above the seam. Dual of `Qgen_H_right_hi`. -/
theorem Qgen_H_left_hi (m W b : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0) (hb : b < 2^(m+2)) :
    Qgen (W + 2^(m+1)) (2^(m+1)) b (m+2) = -1 := by
  have h2H := h2pow_succ_add m
  have hYm : W + 2^(m+1) < 2^(m+2) := by omega
  have hH : (2:Nat)^(m+1) < 2^(m+2) := by omega
  have hpos : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have hY0 : W + 2^(m+1) ≠ 0 := by omega
  have hHY : (2:Nat)^(m+1) ^^^ (W + 2^(m+1)) = W := by
    rw [seam_add_xor W m hW, Nat.xor_comm W (2^(m+1)), ← Nat.xor_assoc, Nat.xor_self,
        Nat.zero_xor]
  by_cases hb0 : b = 0
  · subst hb0
    exact Qgen_degen (m+2) (W + 2^(m+1)) (2^(m+1)) 0 hYm hH (by omega) hY0
      (Or.inr (Or.inl rfl))
  by_cases hbH : b = 2^(m+1)
  · subst hbH
    exact Qgen_degen (m+2) (W + 2^(m+1)) (2^(m+1)) (2^(m+1)) hYm hH hH hY0
      (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
  by_cases hbW : b = W
  · have hd : (2:Nat)^(m+1) ^^^ W ^^^ (W + 2^(m+1)) = 0 := by
      -- H ⊕ W ⊕ Y = (H ⊕ Y) ⊕ W = W ⊕ W = 0
      calc (2:Nat)^(m+1) ^^^ W ^^^ (W + 2^(m+1))
          = (2:Nat)^(m+1) ^^^ (W + 2^(m+1)) ^^^ W := by
              rw [Nat.xor_assoc, Nat.xor_comm W (W + 2^(m+1)), ← Nat.xor_assoc]
        _ = W ^^^ W := by rw [hHY]
        _ = 0 := Nat.xor_self W
    rw [hbW]
    exact Qgen_degen (m+2) (W + 2^(m+1)) (2^(m+1)) W hYm hH (by omega) hY0
      (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr hd)))))
  by_cases hbWH : b = W + 2^(m+1)
  · rw [hbWH]
    exact Qgen_degen (m+2) (W + 2^(m+1)) (2^(m+1)) (W + 2^(m+1)) hYm hH hYm hY0
      (Or.inr (Or.inr (Or.inr (Or.inl (Nat.xor_self _)))))
  by_cases hbU : b ≥ 2^(m+1)
  · obtain ⟨v, hbe, hvl⟩ : ∃ v, b = v + 2^(m+1) ∧ v < 2^(m+1) :=
      ⟨b - 2^(m+1), by omega, by omega⟩
    have hv0 : v ≠ 0 := by intro h; apply hbH; rw [hbe, h]; simp
    have hvW : v ^^^ W ≠ 0 := by
      intro h; have he := xor_zero_eq v W h; apply hbWH; rw [hbe, he]
    have hne : v ≠ W := by intro he; exact hvW (by rw [he, Nat.xor_self])
    have hxb : (v + 2^(m+1)) ^^^ (W + 2^(m+1)) = v ^^^ W := xor_seam_cancel v W m hvl hW
    have e1 : cdSigma (2^(m+1)) (v + 2^(m+1)) (m+2) = 1 := by
      have h := R_uu 0 v m hpos hvl
      rw [Nat.zero_add] at h; rw [h, if_neg hv0, cdSig0']
    have e2 : cdSigma W (v ^^^ W) (m+2) = cdSigma W (v ^^^ W) (m+1) :=
      R_ll W (v ^^^ W) m hW (xorlt hvl hW)
    have e3 : cdSigma (2^(m+1)) (v ^^^ W) (m+2) = -1 := by
      have h := R_ul 0 (v ^^^ W) m hpos (xorlt hvl hW)
      rw [Nat.zero_add] at h; rw [h, if_neg hvW, cdSig0]
    have e4 : cdSigma W (v + 2^(m+1)) (m+2) = cdSigma v W (m+1) :=
      R_lu W v m hW hvl
    have eA : cdSigma v W (m+1) = - cdSigma W v (m+1) :=
      antisym (m+1) v W hvl hW hv0 hW0 hne
    unfold Qgen
    rw [hbe, hxb, hHY, e1, e2, e3, e4, eA]
    have hd := deg_right (m+1) W v hW hvl hW0
    rcases cdSigma_pm (m+1) W (v ^^^ W) with h1 | h1 <;>
      rcases cdSigma_pm (m+1) W v with h2 | h2 <;>
      rw [h1, h2] at hd ⊢ <;> revert hd <;> decide
  · have hbl : b < 2^(m+1) := by omega
    have hbW' : b ^^^ W ≠ 0 := by intro h; exact hbW (xor_zero_eq b W h)
    have hne : b ^^^ W ≠ W := by
      intro he
      have hb0' : b = 0 := by
        have hh : (b ^^^ W) ^^^ W = W ^^^ W := congrArg (fun z => z ^^^ W) he
        -- b ^^^ W ^^^ W = 0
        have : b = 0 := by
          simp only [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero] at hh
          exact hh
        exact this
      exact hb0 hb0'
    have hxb : b ^^^ (W + 2^(m+1)) = (b ^^^ W) + 2^(m+1) := xor_seam b W m hbl hW
    have e1 : cdSigma (2^(m+1)) b (m+2) = -1 := by
      have h := R_ul 0 b m hpos hbl
      rw [Nat.zero_add] at h; rw [h, if_neg hb0, cdSig0]
    have e2 : cdSigma W ((b ^^^ W) + 2^(m+1)) (m+2) = cdSigma (b ^^^ W) W (m+1) :=
      R_lu W (b ^^^ W) m hW (xorlt hbl hW)
    have e3 : cdSigma (2^(m+1)) ((b ^^^ W) + 2^(m+1)) (m+2) = 1 := by
      have h := R_uu 0 (b ^^^ W) m hpos (xorlt hbl hW)
      rw [Nat.zero_add] at h; rw [h, if_neg hbW', cdSig0']
    have e4 : cdSigma W b (m+2) = cdSigma W b (m+1) := R_ll W b m hW hbl
    have eA : cdSigma (b ^^^ W) W (m+1) = - cdSigma W (b ^^^ W) (m+1) :=
      antisym (m+1) (b ^^^ W) W (xorlt hbl hW) hW hbW' hW0 hne
    unfold Qgen
    rw [hxb, hHY, e1, e2, e3, e4, eA]
    have hd := deg_right (m+1) W b hW hbl hW0
    rcases cdSigma_pm (m+1) W (b ^^^ W) with h1 | h1 <;>
      rcases cdSigma_pm (m+1) W b with h2 | h2 <;>
      rw [h1, h2] at hd ⊢ <;> revert hd <;> decide

theorem Qgen_H_left_low' (m W a b : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (hb : b < 2^(m+2)) (hae : a ^^^ W = 2^(m+1)) : Qgen W a b (m+2) = -1 := by
  rw [Qgen_coset_left, hae]
  exact Qgen_H_left_low m W b hW hW0 hb

theorem Qgen_H_left_hi' (m W a b : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (hb : b < 2^(m+2)) (hae : a ^^^ (W + 2^(m+1)) = 2^(m+1)) :
    Qgen (W + 2^(m+1)) a b (m+2) = -1 := by
  rw [Qgen_coset_left, hae]
  exact Qgen_H_left_hi m W b hW hW0 hb

/-! ### Gap root `a ⊕ b = H` -/

theorem Qgen_H_diff_low (m W a : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0) (ha : a < 2^(m+1)) :
    Qgen W a (a + 2^(m+1)) (m+2) = -1 := by
  have hred : Qgen W a (a + 2^(m+1)) (m+2) = Qgen W a a (m+1) :=
    Qred_low_lu m W a a hW ha ha
  rw [hred]
  exact Qgen_degen (m+1) W a a hW ha ha hW0 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))

theorem Qgen_H_diff_low_hi (m W u : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0) (hu : u < 2^(m+1)) :
    Qgen W (u + 2^(m+1)) u (m+2) = -1 := by
  have h2H := h2pow_succ_add m
  have hWm : W < 2^(m+2) := by omega
  have huH : u + 2^(m+1) < 2^(m+2) := by omega
  have hu' : u < 2^(m+2) := by omega
  by_cases hu0 : u = 0
  · subst hu0
    rw [Nat.zero_add]
    exact Qgen_degen (m+2) W (2^(m+1)) 0 hWm (by omega) (by omega) hW0
      (Or.inr (Or.inl rfl))
  · by_cases huW : u ^^^ W = 0
    · have heb : u = W := xor_zero_eq u W huW
      rw [heb]
      exact Qgen_degen (m+2) W (W + 2^(m+1)) W hWm (by omega) (by omega) hW0
        (Or.inr (Or.inr (Or.inr (Or.inl (Nat.xor_self W)))))
    · have hred : Qgen W (u + 2^(m+1)) u (m+2) = Qgen W u u (m+1) :=
        Qred_low_ul m W u u hW hu hu hu0 huW
      rw [hred]
      exact Qgen_degen (m+1) W u u hW hu hu hW0
        (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))

theorem Qgen_H_diff_low_any (m W a : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0) (ha : a < 2^(m+2)) :
    Qgen W a (a ^^^ 2^(m+1)) (m+2) = -1 := by
  have h2H := h2pow_succ_add m
  by_cases haU : a ≥ 2^(m+1)
  · have hul : a - 2^(m+1) < 2^(m+1) := by omega
    have hae : a = (a - 2^(m+1)) + 2^(m+1) := by omega
    have hxor : ((a - 2^(m+1)) + 2^(m+1)) ^^^ 2^(m+1) = a - 2^(m+1) := by
      have hse : (a - 2^(m+1)) + 2^(m+1) = (a - 2^(m+1)) ^^^ 2^(m+1) :=
        seam_add_xor (a - 2^(m+1)) m hul
      calc ((a - 2^(m+1)) + 2^(m+1)) ^^^ 2^(m+1)
          = ((a - 2^(m+1)) ^^^ 2^(m+1)) ^^^ 2^(m+1) := by rw [hse]
        _ = a - 2^(m+1) := by rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
    rw [hae, hxor]
    exact Qgen_H_diff_low_hi m W (a - 2^(m+1)) hW hW0 hul
  · have hal : a < 2^(m+1) := by omega
    have hxor : a ^^^ 2^(m+1) = a + 2^(m+1) := (seam_add_xor a m hal).symm
    rw [hxor]
    exact Qgen_H_diff_low m W a hW hW0 hal

theorem Qgen_H_diff_low_coset (m W a b : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (ha : a < 2^(m+2)) (hbe : b ^^^ W = a ^^^ 2^(m+1)) :
    Qgen W a b (m+2) = -1 := by
  rw [Qgen_coset_right, hbe]
  exact Qgen_H_diff_low_any m W a hW hW0 ha

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

/-! ## (*) on the degenerate locus -- the first branch of the assembly, closed -/

/-- `tau` is injective (it is an involution). -/
theorem tau_inj (j x y : Nat) (h : tau j x = tau j y) : x = y := by
  have := congrArg (tau j) h
  rwa [tau_involutive, tau_involutive] at this

/-- **(*) holds on the degenerate locus, forall n.** Both sides are the constant `-1`:
    `tau` carries each of the six degeneracies to itself, by `tau_zero`, `tau_xor` and
    injectivity, and `Qgen_degen` then pins both sides. No induction is needed here -- this is
    the branch of the assembly that closes outright. -/
theorem star_degen (m j Y a b : Nat) (hj : j < m) (hY : Y < 2^m) (ha : a < 2^m) (hb : b < 2^m)
    (hY0 : Y ≠ 0)
    (hd : a = 0 ∨ b = 0 ∨ a ^^^ Y = 0 ∨ b ^^^ Y = 0 ∨ a = b ∨ a ^^^ b ^^^ Y = 0) :
    Qgen Y a b m = Qgen (tau j Y) (tau j a) (tau j b) m := by
  have htY0 : tau j Y ≠ 0 := by
    intro h
    exact hY0 (tau_inj j Y 0 (by rw [h, tau_zero]))
  have hd' : tau j a = 0 ∨ tau j b = 0 ∨ tau j a ^^^ tau j Y = 0 ∨ tau j b ^^^ tau j Y = 0
           ∨ tau j a = tau j b ∨ tau j a ^^^ tau j b ^^^ tau j Y = 0 := by
    rcases hd with h | h | h | h | h | h
    · exact Or.inl (by rw [h, tau_zero])
    · exact Or.inr (Or.inl (by rw [h, tau_zero]))
    · exact Or.inr (Or.inr (Or.inl (by rw [← tau_xor, h, tau_zero])))
    · exact Or.inr (Or.inr (Or.inr (Or.inl (by rw [← tau_xor, h, tau_zero]))))
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (by rw [h])))))
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr
        (by rw [← tau_xor, ← tau_xor, h, tau_zero])))))
  rw [Qgen_degen m Y a b hY ha hb hY0 hd,
      Qgen_degen m (tau j Y) (tau j a) (tau j b)
        (tau_lt j m Y hj hY) (tau_lt j m a hj ha) (tau_lt j m b hj hb) htY0 hd']

/-- **(*) on the gap, `b = H` case, forall n.** Both sides are `-1`: `tau` fixes the seam bit
    (`tau_seam_fixed`) so the right-hand side is at the same `b`, and it preserves `Y`'s half
    (`tau_seam`) so the same root applies to both. -/
theorem star_gap_bH (m j W a : Nat) (hj : j < m+1) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (ha : a < 2^(m+2)) :
    Qgen W a (2^(m+1)) (m+2) = Qgen (tau j W) (tau j a) (tau j (2^(m+1))) (m+2) := by
  have hjm : j < m + 2 := by omega
  have htW : tau j W < 2^(m+1) := tau_lt j (m+1) W hj hW
  have htW0 : tau j W ≠ 0 := fun h => hW0 (tau_inj j W 0 (by rw [h, tau_zero]))
  have hta : tau j a < 2^(m+2) := tau_lt j (m+2) a hjm ha
  rw [tau_seam_fixed j m hj,
      Qgen_H_right_low m W a hW hW0 ha,
      Qgen_H_right_low m (tau j W) (tau j a) htW htW0 hta]

/-- The same, with `Y` above the seam. -/
theorem star_gap_bH_hi (m j W a : Nat) (hj : j < m+1) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (ha : a < 2^(m+2)) :
    Qgen (W + 2^(m+1)) a (2^(m+1)) (m+2)
      = Qgen (tau j (W + 2^(m+1))) (tau j a) (tau j (2^(m+1))) (m+2) := by
  have hjm : j < m + 2 := by omega
  have htW : tau j W < 2^(m+1) := tau_lt j (m+1) W hj hW
  have htW0 : tau j W ≠ 0 := fun h => hW0 (tau_inj j W 0 (by rw [h, tau_zero]))
  have hta : tau j a < 2^(m+2) := tau_lt j (m+2) a hjm ha
  rw [tau_seam j m W hj hW, tau_seam_fixed j m hj,
      Qgen_H_right_hi m W a hW hW0 ha,
      Qgen_H_right_hi m (tau j W) (tau j a) htW htW0 hta]

/-! ## (*) implies (*') — the mutual induction collapses to a single one -/

/-- The commutation sign. -/
def chi (x y m : Nat) : Int := cdSigma x y m * cdSigma y x m

/-- Closed form for `chi`: `+1` exactly when an argument vanishes or the two coincide.
    (All four ingredients were already proven: `cdSig0`, `cdSig0'`, `cdSq`, `antisym`.) -/
theorem chi_char (m x y : Nat) (hx : x < 2^m) (hy : y < 2^m) :
    chi x y m = (if x = 0 ∨ y = 0 ∨ x = y then 1 else -1) := by
  unfold chi
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

/-- **`Q'` factors through `Q`**: the two differ by exactly two commutation signs. Pure
    algebra -- the two transposed factors contribute their `chi`, the rest squares away. -/
theorem Qgen'_eq_chi (W a b m : Nat) :
    Qgen' W a b m = Qgen W a b m * chi (a ^^^ W) (b ^^^ W) m * chi a (b ^^^ W) m := by
  unfold Qgen Qgen' chi
  have h1 := cdSq (a ^^^ W) (b ^^^ W) m
  have h2 := cdSq a (b ^^^ W) m
  calc cdSigma a b m * cdSigma (b ^^^ W) (a ^^^ W) m * cdSigma (b ^^^ W) a m
          * cdSigma (a ^^^ W) b m
      = (cdSigma a b m * cdSigma (b ^^^ W) (a ^^^ W) m * cdSigma (b ^^^ W) a m
          * cdSigma (a ^^^ W) b m)
        * ((cdSigma (a ^^^ W) (b ^^^ W) m * cdSigma (a ^^^ W) (b ^^^ W) m)
           * (cdSigma a (b ^^^ W) m * cdSigma a (b ^^^ W) m)) := by
        rw [h1, h2]; simp
    _ = cdSigma a b m * cdSigma (a ^^^ W) (b ^^^ W) m * cdSigma a (b ^^^ W) m
          * cdSigma (a ^^^ W) b m
        * (cdSigma (a ^^^ W) (b ^^^ W) m * cdSigma (b ^^^ W) (a ^^^ W) m)
        * (cdSigma a (b ^^^ W) m * cdSigma (b ^^^ W) a m) := by ac_rfl

/-- `chi` is `tau`-invariant: by `chi_char` it depends only on `x = 0`, `y = 0`, `x = y`, and
    `tau` preserves all three. -/
theorem chi_tau (m j x y : Nat) (hj : j < m) (hx : x < 2^m) (hy : y < 2^m) :
    chi (tau j x) (tau j y) m = chi x y m := by
  rw [chi_char m (tau j x) (tau j y) (tau_lt j m x hj hx) (tau_lt j m y hj hy),
      chi_char m x y hx hy]
  have e : (tau j x = 0 ∨ tau j y = 0 ∨ tau j x = tau j y) ↔ (x = 0 ∨ y = 0 ∨ x = y) := by
    constructor
    · rintro (h | h | h)
      · exact Or.inl (tau_inj j x 0 (by rw [h, tau_zero]))
      · exact Or.inr (Or.inl (tau_inj j y 0 (by rw [h, tau_zero])))
      · exact Or.inr (Or.inr (tau_inj j x y h))
    · rintro (h | h | h)
      · exact Or.inl (by rw [h, tau_zero])
      · exact Or.inr (Or.inl (by rw [h, tau_zero]))
      · exact Or.inr (Or.inr (by rw [h]))
  by_cases h : x = 0 ∨ y = 0 ∨ x = y
  · rw [if_pos h, if_pos (e.mpr h)]
  · rw [if_neg h, if_neg (fun g => h (e.mp g))]

/-- **(*) implies (*').** So the "mutual" induction is not mutual: one induction on `Qgen`
    carries `Qgen'` with it. -/
theorem star'_of_star (m j W a b : Nat) (hj : j < m) (hW : W < 2^m) (ha : a < 2^m) (hb : b < 2^m)
    (hstar : Qgen W a b m = Qgen (tau j W) (tau j a) (tau j b) m) :
    Qgen' W a b m = Qgen' (tau j W) (tau j a) (tau j b) m := by
  have hxa : tau j a ^^^ tau j W = tau j (a ^^^ W) := (tau_xor j a W).symm
  have hxb : tau j b ^^^ tau j W = tau j (b ^^^ W) := (tau_xor j b W).symm
  rw [Qgen'_eq_chi, Qgen'_eq_chi, hstar, hxa, hxb,
      chi_tau m j (a ^^^ W) (b ^^^ W) hj (Nat.xor_lt_two_pow ha hW) (Nat.xor_lt_two_pow hb hW),
      chi_tau m j a (b ^^^ W) hj ha (Nat.xor_lt_two_pow hb hW)]

/-! ## `sw` and `tau` are the same function

Two agents introduced the bit-swap independently on this file, under different names,
different argument order and different (but equal) definitions. Without this bridge the two
halves of the assembly do not compose: the gap roots and `star_pow2` are stated in `sw`, the
degenerate and gap branches and the whole `tau` property layer in `tau`. -/

/-- The two definitions agree. At `j = 0` both are the identity (the `else` branch, where the
    masks `1 ^^^ 2^j` and `1 ||| 2^j` actually differ, is unreachable there). -/
theorem testBit_one_eq (k : Nat) : Nat.testBit 1 k = decide (0 = k) := by
  rw [show (1:Nat) = 2^0 from rfl, Nat.testBit_two_pow]

theorem sw_eq_tau (x j : Nat) : sw x j = tau j x := by
  by_cases hj : j = 0
  · subst hj
    rw [tau_id_zero]
    unfold sw
    simp
  · have hmask : (1:Nat) ^^^ 2^j = 1 ||| (1 <<< j) := by
      rw [Nat.shiftLeft_eq, Nat.one_mul]
      refine Nat.eq_of_testBit_eq fun k => ?_
      rw [Nat.testBit_xor, Nat.testBit_or, Nat.testBit_two_pow]
      rw [testBit_one_eq]
      by_cases hk : k = 0
      · subst hk; simp; omega
      · by_cases hk2 : j = k <;> simp_all <;> omega
    have hc0 : x % 2 = x &&& 1 := (Nat.and_one_is_mod x).symm
    have hcj : (x / 2^j) % 2 = (x >>> j) &&& 1 := by
      rw [Nat.and_one_is_mod, Nat.shiftRight_eq_div_pow]
    unfold sw tau
    rw [hmask, hc0, hcj]
    by_cases h : (x &&& 1) = ((x >>> j) &&& 1) <;> simp [h]

/-! ## The generic branch: the sixteen reductions under one induction hypothesis

Each case applies its reduction lemma on BOTH sides -- `tau_seam` guarantees the two land in the
same quadrant, so the same lemma applies -- and then closes on the induction hypothesis. The
`Y`-high rows land on `Q'`, which `star'_of_star` supplies from the same hypothesis: this is
where the collapse of the mutual induction pays off. -/

/-- Generic, `Y` below the seam, both arguments lower. No side conditions. -/
theorem star_gen_low_ll (m j W a b : Nat) (hj : j < m+1) (hW : W < 2^(m+1))
    (ha : a < 2^(m+1)) (hb : b < 2^(m+1))
    (IH : Qgen W a b (m+1) = Qgen (tau j W) (tau j a) (tau j b) (m+1)) :
    Qgen W a b (m+2) = Qgen (tau j W) (tau j a) (tau j b) (m+2) := by
  rw [Qred_low_ll m W a b hW ha hb,
      Qred_low_ll m (tau j W) (tau j a) (tau j b)
        (tau_lt j (m+1) W hj hW) (tau_lt j (m+1) a hj ha) (tau_lt j (m+1) b hj hb)]
  exact IH

/-- Generic, `Y` below the seam, `b` upper. No side conditions; note the induction hypothesis is
    needed at the SWAPPED pair `(v, a)`, which is what the reduction delivers. -/
theorem star_gen_low_lu (m j W a v : Nat) (hj : j < m+1) (hW : W < 2^(m+1))
    (ha : a < 2^(m+1)) (hv : v < 2^(m+1))
    (IH : Qgen W v a (m+1) = Qgen (tau j W) (tau j v) (tau j a) (m+1)) :
    Qgen W a (v + 2^(m+1)) (m+2)
      = Qgen (tau j W) (tau j a) (tau j (v + 2^(m+1))) (m+2) := by
  rw [tau_seam j m v hj hv,
      Qred_low_lu m W a v hW ha hv,
      Qred_low_lu m (tau j W) (tau j a) (tau j v)
        (tau_lt j (m+1) W hj hW) (tau_lt j (m+1) a hj ha) (tau_lt j (m+1) v hj hv)]
  exact IH

/-- Generic, `Y` below the seam, `a` upper. Needs `b ≠ 0`, `b ⊕ W ≠ 0`; `tau` preserves both. -/
theorem star_gen_low_ul (m j W u b : Nat) (hj : j < m+1) (hW : W < 2^(m+1))
    (hu : u < 2^(m+1)) (hb : b < 2^(m+1)) (hb0 : b ≠ 0) (hbW : b ^^^ W ≠ 0)
    (IH : Qgen W u b (m+1) = Qgen (tau j W) (tau j u) (tau j b) (m+1)) :
    Qgen W (u + 2^(m+1)) b (m+2)
      = Qgen (tau j W) (tau j (u + 2^(m+1))) (tau j b) (m+2) := by
  have htb0 : tau j b ≠ 0 := fun h => hb0 (tau_inj j b 0 (by rw [h, tau_zero]))
  have htbW : tau j b ^^^ tau j W ≠ 0 := by
    rw [← tau_xor]; exact fun h => hbW (tau_inj j (b ^^^ W) 0 (by rw [h, tau_zero]))
  rw [tau_seam j m u hj hu,
      Qred_low_ul m W u b hW hu hb hb0 hbW,
      Qred_low_ul m (tau j W) (tau j u) (tau j b)
        (tau_lt j (m+1) W hj hW) (tau_lt j (m+1) u hj hu) (tau_lt j (m+1) b hj hb) htb0 htbW]
  exact IH

/-- Generic, `Y` below the seam, both arguments upper. -/
theorem star_gen_low_uu (m j W u v : Nat) (hj : j < m+1) (hW : W < 2^(m+1))
    (hu : u < 2^(m+1)) (hv : v < 2^(m+1)) (hv0 : v ≠ 0) (hvW : v ^^^ W ≠ 0)
    (IH : Qgen W v u (m+1) = Qgen (tau j W) (tau j v) (tau j u) (m+1)) :
    Qgen W (u + 2^(m+1)) (v + 2^(m+1)) (m+2)
      = Qgen (tau j W) (tau j (u + 2^(m+1))) (tau j (v + 2^(m+1))) (m+2) := by
  have htv0 : tau j v ≠ 0 := fun h => hv0 (tau_inj j v 0 (by rw [h, tau_zero]))
  have htvW : tau j v ^^^ tau j W ≠ 0 := by
    rw [← tau_xor]; exact fun h => hvW (tau_inj j (v ^^^ W) 0 (by rw [h, tau_zero]))
  rw [tau_seam j m u hj hu, tau_seam j m v hj hv,
      Qred_low_uu m W u v hW hu hv hv0 hvW,
      Qred_low_uu m (tau j W) (tau j u) (tau j v)
        (tau_lt j (m+1) W hj hW) (tau_lt j (m+1) u hj hu) (tau_lt j (m+1) v hj hv) htv0 htvW]
  exact IH

/-- Generic, `Y` above the seam, both arguments lower. The reduction lands on `Q'`, and
    `star'_of_star` turns the SAME induction hypothesis into the `Q'` statement it needs -- this
    is the case that would have forced a mutual induction. -/
theorem star_gen_hi_ll (m j W a b : Nat) (hj : j < m+1) (hW : W < 2^(m+1))
    (ha : a < 2^(m+1)) (hb : b < 2^(m+1)) (hb0 : b ≠ 0) (hbW : b ^^^ W ≠ 0)
    (IH : Qgen W a b (m+1) = Qgen (tau j W) (tau j a) (tau j b) (m+1)) :
    Qgen (W + 2^(m+1)) a b (m+2)
      = Qgen (tau j (W + 2^(m+1))) (tau j a) (tau j b) (m+2) := by
  have htb0 : tau j b ≠ 0 := fun h => hb0 (tau_inj j b 0 (by rw [h, tau_zero]))
  have htbW : tau j b ^^^ tau j W ≠ 0 := by
    rw [← tau_xor]; exact fun h => hbW (tau_inj j (b ^^^ W) 0 (by rw [h, tau_zero]))
  have hQ' : Qgen' W a b (m+1) = Qgen' (tau j W) (tau j a) (tau j b) (m+1) :=
    star'_of_star (m+1) j W a b hj hW ha hb IH
  rw [tau_seam j m W hj hW,
      Qred_hi_ll m W a b hW ha hb hb0 hbW,
      Qred_hi_ll m (tau j W) (tau j a) (tau j b)
        (tau_lt j (m+1) W hj hW) (tau_lt j (m+1) a hj ha) (tau_lt j (m+1) b hj hb) htb0 htbW,
      hQ']

/-- Generic, `Y` above the seam, `a` upper. -/
theorem star_gen_hi_ul (m j W u b : Nat) (hj : j < m+1) (hW : W < 2^(m+1))
    (hu : u < 2^(m+1)) (hb : b < 2^(m+1)) (hb0 : b ≠ 0) (hbW : b ^^^ W ≠ 0)
    (IH : Qgen W u b (m+1) = Qgen (tau j W) (tau j u) (tau j b) (m+1)) :
    Qgen (W + 2^(m+1)) (u + 2^(m+1)) b (m+2)
      = Qgen (tau j (W + 2^(m+1))) (tau j (u + 2^(m+1))) (tau j b) (m+2) := by
  have htb0 : tau j b ≠ 0 := fun h => hb0 (tau_inj j b 0 (by rw [h, tau_zero]))
  have htbW : tau j b ^^^ tau j W ≠ 0 := by
    rw [← tau_xor]; exact fun h => hbW (tau_inj j (b ^^^ W) 0 (by rw [h, tau_zero]))
  have hQ' : Qgen' W u b (m+1) = Qgen' (tau j W) (tau j u) (tau j b) (m+1) :=
    star'_of_star (m+1) j W u b hj hW hu hb IH
  rw [tau_seam j m W hj hW, tau_seam j m u hj hu,
      Qred_hi_ul m W u b hW hu hb hb0 hbW,
      Qred_hi_ul m (tau j W) (tau j u) (tau j b)
        (tau_lt j (m+1) W hj hW) (tau_lt j (m+1) u hj hu) (tau_lt j (m+1) b hj hb) htb0 htbW,
      hQ']

/-- Generic, `Y` above the seam, `b` upper -- one of the two rows where `b ⊕ Y` crosses into the
    lower half, so the full non-degeneracy is needed. -/
theorem star_gen_hi_lu (m j W a v : Nat) (hj : j < m+1) (hW : W < 2^(m+1))
    (ha : a < 2^(m+1)) (hv : v < 2^(m+1))
    (ha0 : a ≠ 0) (hv0 : v ≠ 0) (haW : a ^^^ W ≠ 0) (hvW : v ^^^ W ≠ 0)
    (h3 : a ^^^ v ^^^ W ≠ 0)
    (IH : Qgen W v a (m+1) = Qgen (tau j W) (tau j v) (tau j a) (m+1)) :
    Qgen (W + 2^(m+1)) a (v + 2^(m+1)) (m+2)
      = Qgen (tau j (W + 2^(m+1))) (tau j a) (tau j (v + 2^(m+1))) (m+2) := by
  have hz : ∀ x, x ≠ 0 → tau j x ≠ 0 := fun x hx h => hx (tau_inj j x 0 (by rw [h, tau_zero]))
  have hta0 := hz a ha0
  have htv0 := hz v hv0
  have htaW : tau j a ^^^ tau j W ≠ 0 := by rw [← tau_xor]; exact hz _ haW
  have htvW : tau j v ^^^ tau j W ≠ 0 := by rw [← tau_xor]; exact hz _ hvW
  have ht3 : tau j a ^^^ tau j v ^^^ tau j W ≠ 0 := by
    rw [← tau_xor, ← tau_xor]; exact hz _ h3
  have hQ' : Qgen' W v a (m+1) = Qgen' (tau j W) (tau j v) (tau j a) (m+1) :=
    star'_of_star (m+1) j W v a hj hW hv ha IH
  rw [tau_seam j m W hj hW, tau_seam j m v hj hv,
      Qred_hi_lu m W a v hW ha hv ha0 hv0 haW hvW h3,
      Qred_hi_lu m (tau j W) (tau j a) (tau j v)
        (tau_lt j (m+1) W hj hW) (tau_lt j (m+1) a hj ha) (tau_lt j (m+1) v hj hv)
        hta0 htv0 htaW htvW ht3,
      hQ']

/-- Generic, `Y` above the seam, both arguments upper -- the other crossing row. -/
theorem star_gen_hi_uu (m j W u v : Nat) (hj : j < m+1) (hW : W < 2^(m+1))
    (hu : u < 2^(m+1)) (hv : v < 2^(m+1))
    (hu0 : u ≠ 0) (hv0 : v ≠ 0) (huW : u ^^^ W ≠ 0) (hvW : v ^^^ W ≠ 0)
    (h3 : u ^^^ v ^^^ W ≠ 0)
    (IH : Qgen W v u (m+1) = Qgen (tau j W) (tau j v) (tau j u) (m+1)) :
    Qgen (W + 2^(m+1)) (u + 2^(m+1)) (v + 2^(m+1)) (m+2)
      = Qgen (tau j (W + 2^(m+1))) (tau j (u + 2^(m+1))) (tau j (v + 2^(m+1))) (m+2) := by
  have hz : ∀ x, x ≠ 0 → tau j x ≠ 0 := fun x hx h => hx (tau_inj j x 0 (by rw [h, tau_zero]))
  have htu0 := hz u hu0
  have htv0 := hz v hv0
  have htuW : tau j u ^^^ tau j W ≠ 0 := by rw [← tau_xor]; exact hz _ huW
  have htvW : tau j v ^^^ tau j W ≠ 0 := by rw [← tau_xor]; exact hz _ hvW
  have ht3 : tau j u ^^^ tau j v ^^^ tau j W ≠ 0 := by
    rw [← tau_xor, ← tau_xor]; exact hz _ h3
  have hQ' : Qgen' W v u (m+1) = Qgen' (tau j W) (tau j v) (tau j u) (m+1) :=
    star'_of_star (m+1) j W v u hj hW hv hu IH
  rw [tau_seam j m W hj hW, tau_seam j m u hj hu, tau_seam j m v hj hv,
      Qred_hi_uu m W u v hW hu hv hu0 hv0 huW hvW h3,
      Qred_hi_uu m (tau j W) (tau j u) (tau j v)
        (tau_lt j (m+1) W hj hW) (tau_lt j (m+1) u hj hu) (tau_lt j (m+1) v hj hv)
        htu0 htv0 htuW htvW ht3,
      hQ']

/-! ## The remaining gap branches, wired

Only three of the six `= H` conditions actually arise in the induction's case split: `b = H`
(done above), `b ⊕ Y = H` and `a ⊕ Y = H`. The other three are already `m+2`-degenerate and go
to `star_degen`. `tau` preserves each condition because it fixes the seam bit. -/

theorem star_gap_bY_low (m j W a b : Nat) (hj : j < m+1) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (ha : a < 2^(m+2)) (hbe : b ^^^ W = 2^(m+1)) :
    Qgen W a b (m+2) = Qgen (tau j W) (tau j a) (tau j b) (m+2) := by
  have hjm : j < m + 2 := by omega
  have htbe : tau j b ^^^ tau j W = 2^(m+1) := by
    rw [← tau_xor, hbe, tau_seam_fixed j m hj]
  rw [Qgen_H_right_low' m W a b hW hW0 ha hbe,
      Qgen_H_right_low' m (tau j W) (tau j a) (tau j b)
        (tau_lt j (m+1) W hj hW) (fun h => hW0 (tau_inj j W 0 (by rw [h, tau_zero])))
        (tau_lt j (m+2) a hjm ha) htbe]

theorem star_gap_bY_hi (m j W a b : Nat) (hj : j < m+1) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (ha : a < 2^(m+2)) (hbe : b ^^^ (W + 2^(m+1)) = 2^(m+1)) :
    Qgen (W + 2^(m+1)) a b (m+2)
      = Qgen (tau j (W + 2^(m+1))) (tau j a) (tau j b) (m+2) := by
  have hjm : j < m + 2 := by omega
  have htbe : tau j b ^^^ (tau j W + 2^(m+1)) = 2^(m+1) := by
    rw [← tau_seam j m W hj hW, ← tau_xor, hbe, tau_seam_fixed j m hj]
  rw [tau_seam j m W hj hW,
      Qgen_H_right_hi' m W a b hW hW0 ha hbe,
      Qgen_H_right_hi' m (tau j W) (tau j a) (tau j b)
        (tau_lt j (m+1) W hj hW) (fun h => hW0 (tau_inj j W 0 (by rw [h, tau_zero])))
        (tau_lt j (m+2) a hjm ha) htbe]

theorem star_gap_aY_low (m j W a b : Nat) (hj : j < m+1) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (hb : b < 2^(m+2)) (hae : a ^^^ W = 2^(m+1)) :
    Qgen W a b (m+2) = Qgen (tau j W) (tau j a) (tau j b) (m+2) := by
  have hjm : j < m + 2 := by omega
  have htae : tau j a ^^^ tau j W = 2^(m+1) := by
    rw [← tau_xor, hae, tau_seam_fixed j m hj]
  rw [Qgen_H_left_low' m W a b hW hW0 hb hae,
      Qgen_H_left_low' m (tau j W) (tau j a) (tau j b)
        (tau_lt j (m+1) W hj hW) (fun h => hW0 (tau_inj j W 0 (by rw [h, tau_zero])))
        (tau_lt j (m+2) b hjm hb) htae]

theorem star_gap_aY_hi (m j W a b : Nat) (hj : j < m+1) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (hb : b < 2^(m+2)) (hae : a ^^^ (W + 2^(m+1)) = 2^(m+1)) :
    Qgen (W + 2^(m+1)) a b (m+2)
      = Qgen (tau j (W + 2^(m+1))) (tau j a) (tau j b) (m+2) := by
  have hjm : j < m + 2 := by omega
  have htae : tau j a ^^^ (tau j W + 2^(m+1)) = 2^(m+1) := by
    rw [← tau_seam j m W hj hW, ← tau_xor, hae, tau_seam_fixed j m hj]
  rw [tau_seam j m W hj hW,
      Qgen_H_left_hi' m W a b hW hW0 hb hae,
      Qgen_H_left_hi' m (tau j W) (tau j a) (tau j b)
        (tau_lt j (m+1) W hj hW) (fun h => hW0 (tau_inj j W 0 (by rw [h, tau_zero])))
        (tau_lt j (m+2) b hjm hb) htae]

/-! ## The recursion -/

/-- One level, `Y` below the seam. Four quadrants; `uu` is the only one with gap sub-cases. -/
theorem star_step_low (m j W a b : Nat) (hj : j < m+1) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (ha : a < 2^(m+2)) (hb : b < 2^(m+2))
    (hnd : ¬ (a = 0 ∨ b = 0 ∨ a ^^^ W = 0 ∨ b ^^^ W = 0 ∨ a = b ∨ a ^^^ b ^^^ W = 0))
    (IH : ∀ a' b', a' < 2^(m+1) → b' < 2^(m+1) →
        Qgen W a' b' (m+1) = Qgen (tau j W) (tau j a') (tau j b') (m+1)) :
    Qgen W a b (m+2) = Qgen (tau j W) (tau j a) (tau j b) (m+2) := by
  have h2H : 2^(m+2) = 2^(m+1) + 2^(m+1) := by rw [Nat.pow_succ]; omega
  have hb0 : b ≠ 0 := fun h => hnd (Or.inr (Or.inl h))
  have hbW : b ^^^ W ≠ 0 := fun h => hnd (Or.inr (Or.inr (Or.inr (Or.inl h))))
  by_cases haU : a ≥ 2^(m+1) <;> by_cases hbU : b ≥ 2^(m+1)
  · obtain ⟨u, hae, hul⟩ : ∃ u, a = u + 2^(m+1) ∧ u < 2^(m+1) :=
      ⟨a - 2^(m+1), by omega, by omega⟩
    obtain ⟨v, hbe, hvl⟩ : ∃ v, b = v + 2^(m+1) ∧ v < 2^(m+1) :=
      ⟨b - 2^(m+1), by omega, by omega⟩
    subst hae; subst hbe
    by_cases hv0 : v = 0
    · subst hv0
      simpa using star_gap_bH m j W (u + 2^(m+1)) hj hW hW0 (by omega)
    · by_cases hvW : v ^^^ W = 0
      · refine star_gap_bY_low m j W (u + 2^(m+1)) (v + 2^(m+1)) hj hW hW0 (by omega) ?_
        rw [seam_xor_left v W m hvl hW, hvW]
        omega
      · exact star_gen_low_uu m j W u v hj hW hul hvl hv0 hvW (IH v u hvl hul)
  · obtain ⟨u, hae, hul⟩ : ∃ u, a = u + 2^(m+1) ∧ u < 2^(m+1) :=
      ⟨a - 2^(m+1), by omega, by omega⟩
    subst hae
    exact star_gen_low_ul m j W u b hj hW hul (by omega) hb0 hbW (IH u b hul (by omega))
  · obtain ⟨v, hbe, hvl⟩ : ∃ v, b = v + 2^(m+1) ∧ v < 2^(m+1) :=
      ⟨b - 2^(m+1), by omega, by omega⟩
    subst hbe
    exact star_gen_low_lu m j W a v hj hW (by omega) hvl (IH v a hvl (by omega))
  · exact star_gen_low_ll m j W a b hj hW (by omega) (by omega) (IH a b (by omega) (by omega))

/-- Gap `a = H`, `Y` above the seam -- arises only in the `Y`-high quadrants. -/
theorem star_gap_aH_hi (m j W b : Nat) (hj : j < m+1) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (hb : b < 2^(m+2)) :
    Qgen (W + 2^(m+1)) (2^(m+1)) b (m+2)
      = Qgen (tau j (W + 2^(m+1))) (tau j (2^(m+1))) (tau j b) (m+2) := by
  have hjm : j < m + 2 := by omega
  rw [tau_seam j m W hj hW, tau_seam_fixed j m hj,
      Qgen_H_left_hi m W b hW hW0 hb,
      Qgen_H_left_hi m (tau j W) (tau j b)
        (tau_lt j (m+1) W hj hW) (fun h => hW0 (tau_inj j W 0 (by rw [h, tau_zero])))
        (tau_lt j (m+2) b hjm hb)]

/-! ### Gap root `a ⊕ b = W`, `Y` ABOVE the seam

The companion the `Y`-high step needs. For `Y = W + H` the third non-degeneracy
`a ⊕ b ⊕ Y = 0` is unsatisfiable when both arguments are upper, so the induction meets
tuples with `a ⊕ b = W` that no hypothesis excludes. `Q` is the constant `-1` on all of
them, and the proof is a direct four-factor evaluation: with `b = a ⊕ W`,

    a ⊕ Y = b ⊕ H,   b ⊕ Y = a ⊕ H

so the coset square is `σ(a,b)·σ(b⊕H,a⊕H)·σ(a,a⊕H)·σ(b⊕H,b)`, whose last factor is
identically `1` and whose third is `σ(a,a)`. -/

theorem Qgen_H_diff_hi_low (m W a : Nat) (hW : W < 2^(m+1)) (ha : a < 2^(m+1)) :
    Qgen (W + 2^(m+1)) a (a ^^^ W) (m+2) = -1 := by
  have hb : a ^^^ W < 2^(m+1) := Nat.xor_lt_two_pow ha hW
  have e1 : a ^^^ (W + 2^(m+1)) = (a ^^^ W) + 2^(m+1) := xor_seam a W m ha hW
  have e2 : (a ^^^ W) ^^^ (W + 2^(m+1)) = a + 2^(m+1) := by
    rw [xor_seam (a ^^^ W) W m hb hW, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
  have f4 : cdSigma ((a ^^^ W) + 2^(m+1)) (a ^^^ W) (m+2) = 1 := by
    rw [R_ul (a ^^^ W) (a ^^^ W) m hb hb]
    by_cases h : a ^^^ W = 0
    · rw [if_pos h]
    · rw [if_neg h, sigma_self (m+1) (a ^^^ W) hb h]; decide
  have f3 : cdSigma a (a + 2^(m+1)) (m+2) = (if a = 0 then 1 else -1) := by
    rw [R_lu a a m ha ha]
    by_cases h : a = 0
    · rw [if_pos h, h, cdSig0]
    · rw [if_neg h, sigma_self (m+1) a ha h]
  have f12 : cdSigma a (a ^^^ W) (m+2) * cdSigma ((a ^^^ W) + 2^(m+1)) (a + 2^(m+1)) (m+2)
      = (if a = 0 then -1 else 1) := by
    rw [R_ll a (a ^^^ W) m ha hb, R_uu (a ^^^ W) a m hb ha]
    by_cases h : a = 0
    · subst h; simp [cdSig0]
    · rw [if_neg h, if_neg h]; exact cdSq a (a ^^^ W) (m+1)
  unfold Qgen
  rw [e1, e2, f12, f3, f4]
  by_cases h : a = 0 <;> simp [h]

theorem Qgen_H_diff_hi_hi (m W u : Nat) (hW : W < 2^(m+1)) (hu : u < 2^(m+1)) :
    Qgen (W + 2^(m+1)) (u + 2^(m+1)) ((u ^^^ W) + 2^(m+1)) (m+2) = -1 := by
  have hv : u ^^^ W < 2^(m+1) := Nat.xor_lt_two_pow hu hW
  have e1 : (u + 2^(m+1)) ^^^ (W + 2^(m+1)) = u ^^^ W := xor_seam_cancel u W m hu hW
  have e2 : ((u ^^^ W) + 2^(m+1)) ^^^ (W + 2^(m+1)) = u := by
    rw [xor_seam_cancel (u ^^^ W) W m hv hW, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
  have f3 : cdSigma (u + 2^(m+1)) u (m+2) = 1 := by
    rw [R_ul u u m hu hu]
    by_cases h : u = 0
    · rw [if_pos h]
    · rw [if_neg h, sigma_self (m+1) u hu h]; decide
  have f4 : cdSigma (u ^^^ W) ((u ^^^ W) + 2^(m+1)) (m+2) = (if u ^^^ W = 0 then 1 else -1) := by
    rw [R_lu (u ^^^ W) (u ^^^ W) m hv hv]
    by_cases h : u ^^^ W = 0
    · rw [if_pos h, h, cdSig0]
    · rw [if_neg h, sigma_self (m+1) (u ^^^ W) hv h]
  have f12 : cdSigma (u + 2^(m+1)) ((u ^^^ W) + 2^(m+1)) (m+2) * cdSigma (u ^^^ W) u (m+2)
      = (if u ^^^ W = 0 then -1 else 1) := by
    rw [R_uu u (u ^^^ W) m hu hv, R_ll (u ^^^ W) u m hv hu]
    by_cases h : u ^^^ W = 0
    · rw [if_pos h, if_pos h, h, cdSig0]; decide
    · rw [if_neg h, if_neg h]; exact cdSq (u ^^^ W) u (m+1)
  unfold Qgen
  rw [e1, e2, f12, f3, f4]
  by_cases h : u ^^^ W = 0 <;> simp [h]

/-- The root, both halves. -/
theorem Qgen_H_diff_hi_any (m W a : Nat) (hW : W < 2^(m+1)) (ha : a < 2^(m+2)) :
    Qgen (W + 2^(m+1)) a (a ^^^ W) (m+2) = -1 := by
  have h2H := h2pow_succ_add m
  by_cases haU : a ≥ 2^(m+1)
  · have hul : a - 2^(m+1) < 2^(m+1) := by omega
    have hae : a = (a - 2^(m+1)) + 2^(m+1) := by omega
    have hx : ((a - 2^(m+1)) + 2^(m+1)) ^^^ W = ((a - 2^(m+1)) ^^^ W) + 2^(m+1) := by
      rw [Nat.xor_comm ((a - 2^(m+1)) + 2^(m+1)) W, xor_seam W (a - 2^(m+1)) m hW hul,
          Nat.xor_comm W (a - 2^(m+1))]
    rw [hae, hx]
    exact Qgen_H_diff_hi_hi m W (a - 2^(m+1)) hW hul
  · exact Qgen_H_diff_hi_low m W a hW (by omega)

/-- The root in the form the induction meets it: `a ⊕ b = W`. -/
theorem Qgen_H_diff_hi_coset (m W a b : Nat) (hW : W < 2^(m+1)) (ha : a < 2^(m+2))
    (hab : a ^^^ b = W) : Qgen (W + 2^(m+1)) a b (m+2) = -1 := by
  have hbe : b = a ^^^ W := by
    rw [← hab, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]
  rw [hbe]
  exact Qgen_H_diff_hi_any m W a hW ha

/-- The gap branch it feeds: `tau` preserves `a ⊕ b = W` because it is a homomorphism. -/
theorem star_gap_diff_hi (m j W a b : Nat) (hj : j < m+1) (hW : W < 2^(m+1))
    (ha : a < 2^(m+2)) (hab : a ^^^ b = W) :
    Qgen (W + 2^(m+1)) a b (m+2)
      = Qgen (tau j (W + 2^(m+1))) (tau j a) (tau j b) (m+2) := by
  have hjm : j < m + 2 := by omega
  have htab : tau j a ^^^ tau j b = tau j W := by rw [← tau_xor, hab]
  rw [tau_seam j m W hj hW,
      Qgen_H_diff_hi_coset m W a b hW ha hab,
      Qgen_H_diff_hi_coset m (tau j W) (tau j a) (tau j b)
        (tau_lt j (m+1) W hj hW) (tau_lt j (m+2) a hjm ha) htab]

/-- One level, `Y` above the seam. The `uu` quadrant is the one with the extra gap: there
    `a ⊕ b ⊕ Y` cannot vanish, so `u ⊕ v ⊕ W = 0` is unconstrained and lands on the root. -/
theorem star_step_hi (m j W a b : Nat) (hj : j < m+1) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (ha : a < 2^(m+2)) (hb : b < 2^(m+2))
    (hnd : ¬ (a = 0 ∨ b = 0 ∨ a ^^^ (W + 2^(m+1)) = 0 ∨ b ^^^ (W + 2^(m+1)) = 0 ∨ a = b ∨
              a ^^^ b ^^^ (W + 2^(m+1)) = 0))
    (IH : ∀ a' b', a' < 2^(m+1) → b' < 2^(m+1) →
        Qgen W a' b' (m+1) = Qgen (tau j W) (tau j a') (tau j b') (m+1)) :
    Qgen (W + 2^(m+1)) a b (m+2)
      = Qgen (tau j (W + 2^(m+1))) (tau j a) (tau j b) (m+2) := by
  have h2H : 2^(m+2) = 2^(m+1) + 2^(m+1) := by rw [Nat.pow_succ]; omega
  have ha0 : a ≠ 0 := fun h => hnd (Or.inl h)
  have hb0 : b ≠ 0 := fun h => hnd (Or.inr (Or.inl h))
  have haY : a ^^^ (W + 2^(m+1)) ≠ 0 := fun h => hnd (Or.inr (Or.inr (Or.inl h)))
  have hbY : b ^^^ (W + 2^(m+1)) ≠ 0 := fun h => hnd (Or.inr (Or.inr (Or.inr (Or.inl h))))
  have h3 : a ^^^ b ^^^ (W + 2^(m+1)) ≠ 0 :=
    fun h => hnd (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr h)))))
  by_cases haU : a ≥ 2^(m+1) <;> by_cases hbU : b ≥ 2^(m+1)
  · obtain ⟨u, hae, hul⟩ : ∃ u, a = u + 2^(m+1) ∧ u < 2^(m+1) :=
      ⟨a - 2^(m+1), by omega, by omega⟩
    obtain ⟨v, hbe, hvl⟩ : ∃ v, b = v + 2^(m+1) ∧ v < 2^(m+1) :=
      ⟨b - 2^(m+1), by omega, by omega⟩
    subst hae; subst hbe
    by_cases hu0 : u = 0
    · subst hu0
      simpa using star_gap_aH_hi m j W (v + 2^(m+1)) hj hW hW0 (by omega)
    by_cases hv0 : v = 0
    · subst hv0
      simpa using star_gap_bH_hi m j W (u + 2^(m+1)) hj hW hW0 (by omega)
    have huW : u ^^^ W ≠ 0 := by
      rw [← xor_seam_cancel u W m hul hW]; exact haY
    have hvW : v ^^^ W ≠ 0 := by
      rw [← xor_seam_cancel v W m hvl hW]; exact hbY
    by_cases hg : u ^^^ v ^^^ W = 0
    · refine star_gap_diff_hi m j W (u + 2^(m+1)) (v + 2^(m+1)) hj hW (by omega) ?_
      rw [xor_seam_cancel u v m hul hvl]
      exact xor_zero_eq (u ^^^ v) W hg
    · exact star_gen_hi_uu m j W u v hj hW hul hvl hu0 hv0 huW hvW hg (IH v u hvl hul)
  · obtain ⟨u, hae, hul⟩ : ∃ u, a = u + 2^(m+1) ∧ u < 2^(m+1) :=
      ⟨a - 2^(m+1), by omega, by omega⟩
    subst hae
    by_cases hbW : b ^^^ W = 0
    · refine star_gap_bY_hi m j W (u + 2^(m+1)) b hj hW hW0 (by omega) ?_
      rw [xor_seam b W m (by omega) hW, hbW]
      omega
    · exact star_gen_hi_ul m j W u b hj hW hul (by omega) hb0 hbW (IH u b hul (by omega))
  · obtain ⟨v, hbe, hvl⟩ : ∃ v, b = v + 2^(m+1) ∧ v < 2^(m+1) :=
      ⟨b - 2^(m+1), by omega, by omega⟩
    subst hbe
    by_cases haW : a ^^^ W = 0
    · refine star_gap_aY_hi m j W a (v + 2^(m+1)) hj hW hW0 (by omega) ?_
      rw [xor_seam a W m (by omega) hW, haW]
      omega
    by_cases hv0 : v = 0
    · subst hv0
      simpa using star_gap_bH_hi m j W a hj hW hW0 (by omega)
    have hvW : v ^^^ W ≠ 0 := by
      rw [← xor_seam_cancel v W m hvl hW]; exact hbY
    have hg : a ^^^ v ^^^ W ≠ 0 := by
      rw [← xor_seam_cancel (a ^^^ v) W m (Nat.xor_lt_two_pow (by omega) hvl) hW,
          ← xor_seam a v m (by omega) hvl]
      exact h3
    exact star_gen_hi_lu m j W a v hj hW (by omega) hvl ha0 hv0 haW hvW hg (IH v a hvl (by omega))
  · by_cases hbW : b ^^^ W = 0
    · refine star_gap_bY_hi m j W a b hj hW hW0 (by omega) ?_
      rw [xor_seam b W m (by omega) hW, hbW]
      omega
    · exact star_gen_hi_ll m j W a b hj hW (by omega) (by omega) hb0 hbW
        (IH a b (by omega) (by omega))

/-! ## (*) closed: the recursive theorem

The two steps plus the base assemble by structural recursion on the level. The induction
carries one arithmetic invariant beyond the obvious bounds:

    Y % 2^j = 0        (equivalently `j <= lsb Y`)

which is what makes `tau j` the RIGHT swap for `Y` at every level of the descent. It is
exactly the condition `K7` isolated: with a mismatched `j` the equivariance fails in bulk,
and the failure is visible here as the branch where the invariant cannot be re-established.

The descent strips `Y`'s top bit one level at a time. It bottoms out when only the lowest
set bit remains -- `Y = 2^(k+1)` at level `k+2` -- where `Qgen_pow2` gives `-1` on both sides,
`tau` sending the seam either to itself (`j < k+1`, `tau_seam_fixed`) or to `1` (`j = k+1`,
`sw_pow2`); a single-bit label either way. -/

theorem star_forall : ∀ (m j Y a b : Nat), Y < 2^m → Y ≠ 0 → Y % 2^j = 0 →
    a < 2^m → b < 2^m → Qgen Y a b m = Qgen (tau j Y) (tau j a) (tau j b) m
  | 0, _, Y, _, _, hY, hY0, _, _, _ => by
      have h : (2:Nat)^0 = 1 := rfl
      omega
  | 1, j, Y, a, b, hY, hY0, hmod, _, _ => by
      have h2 : (2:Nat)^1 = 2 := rfl
      have hY1 : Y = 1 := by omega
      have hj0 : j = 0 := by
        rcases Nat.eq_zero_or_pos j with h | h
        · exact h
        · exfalso
          have hle : (2:Nat)^1 ≤ 2^j := Nat.pow_le_pow_right (by omega) h
          rw [hY1, Nat.mod_eq_of_lt (by omega)] at hmod
          omega
      subst hj0
      simp only [tau_id_zero]
  | (k+2), j, Y, a, b, hY, hY0, hmod, ha, hb => by
      have h2H : 2^(k+2) = 2^(k+1) + 2^(k+1) := by rw [Nat.pow_succ]; omega
      have hjle : 2^j ≤ Y := by
        by_cases h : 2^j ≤ Y
        · exact h
        · exact absurd (by rw [Nat.mod_eq_of_lt (by omega)] at hmod; exact hmod) hY0
      have hjm : j < k+2 := by
        by_cases h : j < k+2
        · exact h
        · exfalso
          have hle : (2:Nat)^(k+2) ≤ 2^j := Nat.pow_le_pow_right (by omega) (by omega)
          omega
      by_cases hd : a = 0 ∨ b = 0 ∨ a ^^^ Y = 0 ∨ b ^^^ Y = 0 ∨ a = b ∨ a ^^^ b ^^^ Y = 0
      · exact star_degen (k+2) j Y a b hjm hY ha hb hY0 hd
      by_cases hYU : Y ≥ 2^(k+1)
      · obtain ⟨W, hYe, hWl⟩ : ∃ W, Y = W + 2^(k+1) ∧ W < 2^(k+1) :=
          ⟨Y - 2^(k+1), by omega, by omega⟩
        subst hYe
        by_cases hW0 : W = 0
        · subst hW0
          rw [Nat.zero_add]
          have hta : tau j a < 2^(k+2) := tau_lt j (k+2) a hjm ha
          have htb : tau j b < 2^(k+2) := tau_lt j (k+2) b hjm hb
          obtain ⟨i, hi, hie⟩ : ∃ i, i < k+2 ∧ tau j (2^(k+1)) = 2^i := by
            by_cases h : j < k+1
            · exact ⟨k+1, by omega, tau_seam_fixed j k h⟩
            · have hje : j = k+1 := by omega
              subst hje
              refine ⟨0, by omega, ?_⟩
              rw [← sw_eq_tau, sw_pow2 (k+1) (by omega)]
          rw [hie, Qgen_pow2 (k+2) (k+1) a b (by omega) ha hb,
              Qgen_pow2 (k+2) i (tau j a) (tau j b) hi hta htb]
        · have hdvd : (2:Nat)^j ∣ W := by
            have hd1 : (2:Nat)^j ∣ (2^(k+1) + W) := by
              rw [Nat.add_comm]; exact Nat.dvd_of_mod_eq_zero hmod
            exact (Nat.dvd_add_right (Nat.pow_dvd_pow 2 (by omega))).mp hd1
          have hjW : W % 2^j = 0 := by
            obtain ⟨c, hc⟩ := hdvd
            rw [hc, Nat.mul_mod_right]
          have hjW2 : 2^j ≤ W := by
            by_cases h : 2^j ≤ W
            · exact h
            · exact absurd (by rw [Nat.mod_eq_of_lt (by omega)] at hjW; exact hjW) hW0
          have hjk : j < k+1 := by
            by_cases h : j < k+1
            · exact h
            · exfalso
              have hle : (2:Nat)^(k+1) ≤ 2^j := Nat.pow_le_pow_right (by omega) (by omega)
              omega
          exact star_step_hi k j W a b hjk hWl hW0 ha hb hd
            (fun a' b' ha' hb' => star_forall (k+1) j W a' b' hWl hW0 hjW ha' hb')
      · have hYl : Y < 2^(k+1) := by omega
        have hjk : j < k+1 := by
          by_cases h : j < k+1
          · exact h
          · exfalso
            have hle : (2:Nat)^(k+1) ≤ 2^j := Nat.pow_le_pow_right (by omega) (by omega)
            omega
        exact star_step_low k j Y a b hjk hYl hY0 ha hb hd
          (fun a' b' ha' hb' => star_forall (k+1) j Y a' b' hYl hY0 hmod ha' hb')

/-! ## L2's reduction, PROVEN — the fiber comes off the τ-discrepancy of σ

The `l2_reduction` rung (2026-08-01) replaces L2 by a fiber-free statement one level down,
using the τ-discrepancy of the cocycle

    gdisc j x y = σ(τx, τy) · σ(x, y)

Its clause `N4` — that the fiber-level discrepancy IS the reduced product — was measured at
three levels. It does not have to be: every ingredient is already proven here, and the whole
of it is the four branch reductions plus `tau_seam`/`tau_xor`. So the reduction L2 ⟸ (♦) is a
THEOREM and only (♦) itself is measured. -/

/-- The τ-discrepancy of the cocycle. -/
def gdisc (j x y m : Nat) : Int := cdSigma (tau j x) (tau j y) m * cdSigma x y m

theorem gdisc_pm (j x y m : Nat) : gdisc j x y m = 1 ∨ gdisc j x y m = -1 := by
  unfold gdisc
  rcases cdSigma_pm m (tau j x) (tau j y) with h1 | h1 <;>
    rcases cdSigma_pm m x y with h2 | h2 <;> rw [h1, h2] <;> simp

/-- **`gdisc` is symmetric, ∀n.** This is `chi_tau` in disguise, and it is what removes the
    argument swap the raw reduction produces. -/
theorem gdisc_symm (m j x y : Nat) (hj : j < m) (hx : x < 2^m) (hy : y < 2^m) :
    gdisc j x y m = gdisc j y x m := by
  have hchi : chi (tau j x) (tau j y) m * chi x y m = 1 := by
    rw [chi_tau m j x y hj hx hy]
    unfold chi
    rcases cdSigma_pm m x y with h1 | h1 <;> rcases cdSigma_pm m y x with h2 | h2 <;>
      rw [h1, h2] <;> decide
  have hprod : gdisc j x y m * gdisc j y x m = 1 := by
    unfold gdisc
    unfold chi at hchi
    calc cdSigma (tau j x) (tau j y) m * cdSigma x y m
            * (cdSigma (tau j y) (tau j x) m * cdSigma y x m)
        = cdSigma (tau j x) (tau j y) m * cdSigma (tau j y) (tau j x) m
            * (cdSigma x y m * cdSigma y x m) := by ac_rfl
      _ = 1 := hchi
  rcases gdisc_pm j x y m with h | h <;> rcases gdisc_pm j y x m with h' | h' <;>
    rw [h, h'] <;> rw [h, h'] at hprod <;> first | rfl | (exfalso; exact absurd hprod (by decide))

/-- **`N4`, proven ∀n.** For a fiber label `L = Y + H` at level `m+2` with `a, b` below the
    seam, the level-`(m+2)` discrepancy of `P1` under `τ` collapses to a product of two
    level-`(m+1)` `gdisc`s with no fiber and no top bit. `b ⊕ Y ≠ 0` is the `R_uu` branch
    condition, and it governs BOTH sides: `τ (b ⊕ Y) = 0 ↔ b ⊕ Y = 0` by `tau_inj`. -/
theorem l2_reduction (m j Y a b : Nat) (hj : j < m+1) (hY : Y < 2^(m+1))
    (ha : a < 2^(m+1)) (hb : b < 2^(m+1)) (hbY : b ^^^ Y ≠ 0) :
    cdSigma (tau j a) (tau j b) (m+2)
        * cdSigma (tau j (a ^^^ (Y + 2^(m+1)))) (tau j (b ^^^ (Y + 2^(m+1)))) (m+2)
        * (cdSigma a b (m+2) * cdSigma (a ^^^ (Y + 2^(m+1))) (b ^^^ (Y + 2^(m+1))) (m+2))
      = gdisc j a b (m+1) * gdisc j (b ^^^ Y) (a ^^^ Y) (m+1) := by
  have haY : a ^^^ Y < 2^(m+1) := Nat.xor_lt_two_pow ha hY
  have hbY' : b ^^^ Y < 2^(m+1) := Nat.xor_lt_two_pow hb hY
  have hta : tau j a < 2^(m+1) := tau_lt j (m+1) a hj ha
  have htb : tau j b < 2^(m+1) := tau_lt j (m+1) b hj hb
  have htaY : tau j (a ^^^ Y) < 2^(m+1) := tau_lt j (m+1) (a ^^^ Y) hj haY
  have htbY : tau j (b ^^^ Y) < 2^(m+1) := tau_lt j (m+1) (b ^^^ Y) hj hbY'
  -- the seam split, on both the plain and the τ side
  have ea : a ^^^ (Y + 2^(m+1)) = (a ^^^ Y) + 2^(m+1) := xor_seam a Y m ha hY
  have eb : b ^^^ (Y + 2^(m+1)) = (b ^^^ Y) + 2^(m+1) := xor_seam b Y m hb hY
  have eta : tau j ((a ^^^ Y) + 2^(m+1)) = tau j (a ^^^ Y) + 2^(m+1) :=
    tau_seam j m (a ^^^ Y) hj haY
  have etb : tau j ((b ^^^ Y) + 2^(m+1)) = tau j (b ^^^ Y) + 2^(m+1) :=
    tau_seam j m (b ^^^ Y) hj hbY'
  -- the R_uu branch condition, on both sides
  have htbY0 : tau j (b ^^^ Y) ≠ 0 :=
    fun h => hbY (tau_inj j (b ^^^ Y) 0 (by rw [h, tau_zero]))
  rw [ea, eb, eta, etb,
      R_ll (tau j a) (tau j b) m hta htb,
      R_uu (tau j (a ^^^ Y)) (tau j (b ^^^ Y)) m htaY htbY,
      R_ll a b m ha hb,
      R_uu (a ^^^ Y) (b ^^^ Y) m haY hbY',
      if_neg htbY0, if_neg hbY]
  unfold gdisc
  ac_rfl

/-- The reduction in the form the rung states it. `R_uu` returns its arguments SWAPPED, so the
    raw reduction gives `g(b⊕Y, a⊕Y)`; `gdisc_symm` is exactly what turns that into
    `g(a⊕Y, b⊕Y)`. That is the whole job the symmetry does here. -/
theorem l2_reduction_symm (m j Y a b : Nat) (hj : j < m+1) (hY : Y < 2^(m+1))
    (ha : a < 2^(m+1)) (hb : b < 2^(m+1)) (hbY : b ^^^ Y ≠ 0) :
    cdSigma (tau j a) (tau j b) (m+2)
        * cdSigma (tau j (a ^^^ (Y + 2^(m+1)))) (tau j (b ^^^ (Y + 2^(m+1)))) (m+2)
        * (cdSigma a b (m+2) * cdSigma (a ^^^ (Y + 2^(m+1))) (b ^^^ (Y + 2^(m+1))) (m+2))
      = gdisc j a b (m+1) * gdisc j (a ^^^ Y) (b ^^^ Y) (m+1) := by
  rw [l2_reduction m j Y a b hj hY ha hb hbY,
      gdisc_symm (m+1) j (b ^^^ Y) (a ^^^ Y) hj
        (Nat.xor_lt_two_pow hb hY) (Nat.xor_lt_two_pow ha hY)]

/-! ## `N9`, proven — the τ-discrepancy DESCENDS, and so does (♦)'s conclusion

The `l2_reduction` rung measured that

    G(Y,a,b) = gdisc j a b * gdisc j (a ⊕ Y) (b ⊕ Y)

is unchanged by dropping a level and truncating every argument — unconditionally, in all eight
quadrants, with no degeneracy exceptions. That is not a coincidence and it does not need eight
cases of bookkeeping: it follows from a single fact about `gdisc` itself.

`gdisc j x y (m+2) = gdisc j (x mod H) (y mod H) (m+1)`, always.

The reason the degenerate branches never surface is that `R_ul`/`R_uu` guard on `v = 0`, and
`gdisc` pairs the plain factor with the `τ` factor — whose guard is `τ v = 0`, the SAME
condition (`tau_inj`). So the two guards fire together and their constants (`1·1` and
`(−1)·(−1)`) multiply to `1`, which is exactly what `gdisc` is at a zero argument. -/

private theorem gdisc_zero_right (m j p : Nat) : gdisc j p 0 (m+1) = 1 := by
  unfold gdisc
  rw [tau_zero, cdSig0' (tau j p) m, cdSig0' p m]
  decide

theorem gdisc_descend_ll (m j p q : Nat) (hj : j < m+1) (hp : p < 2^(m+1)) (hq : q < 2^(m+1)) :
    gdisc j p q (m+2) = gdisc j p q (m+1) := by
  unfold gdisc
  rw [R_ll (tau j p) (tau j q) m (tau_lt j (m+1) p hj hp) (tau_lt j (m+1) q hj hq),
      R_ll p q m hp hq]

theorem gdisc_descend_ul (m j p q : Nat) (hj : j < m+1) (hp : p < 2^(m+1)) (hq : q < 2^(m+1)) :
    gdisc j (p + 2^(m+1)) q (m+2) = gdisc j p q (m+1) := by
  have htp := tau_lt j (m+1) p hj hp
  have htq := tau_lt j (m+1) q hj hq
  unfold gdisc
  rw [tau_seam j m p hj hp, R_ul (tau j p) (tau j q) m htp htq, R_ul p q m hp hq]
  by_cases h : q = 0
  · rw [if_pos (show tau j q = 0 by rw [h, tau_zero]), if_pos h, h, tau_zero,
        cdSig0' (tau j p) m, cdSig0' p m]
  · have htq0 : tau j q ≠ 0 := fun e => h (tau_inj j q 0 (by rw [e, tau_zero]))
    rw [if_neg htq0, if_neg h, Int.neg_mul_neg]

theorem gdisc_descend_lu (m j p q : Nat) (hj : j < m+1) (hp : p < 2^(m+1)) (hq : q < 2^(m+1)) :
    gdisc j p (q + 2^(m+1)) (m+2) = gdisc j q p (m+1) := by
  unfold gdisc
  rw [tau_seam j m q hj hq,
      R_lu (tau j p) (tau j q) m (tau_lt j (m+1) p hj hp) (tau_lt j (m+1) q hj hq),
      R_lu p q m hp hq]

theorem gdisc_descend_uu (m j p q : Nat) (hj : j < m+1) (hp : p < 2^(m+1)) (hq : q < 2^(m+1)) :
    gdisc j (p + 2^(m+1)) (q + 2^(m+1)) (m+2) = gdisc j q p (m+1) := by
  have htp := tau_lt j (m+1) p hj hp
  have htq := tau_lt j (m+1) q hj hq
  unfold gdisc
  rw [tau_seam j m p hj hp, tau_seam j m q hj hq,
      R_uu (tau j p) (tau j q) m htp htq, R_uu p q m hp hq]
  by_cases h : q = 0
  · rw [if_pos (show tau j q = 0 by rw [h, tau_zero]), if_pos h, h, tau_zero,
        cdSig0 (tau j p) m, cdSig0 p m]
    decide
  · have htq0 : tau j q ≠ 0 := fun e => h (tau_inj j q 0 (by rw [e, tau_zero]))
    rw [if_neg htq0, if_neg h]

/-- `gdisc` descends a level, whatever half its arguments are in. The four quadrants land on
    `gdisc j p q` or on its transpose, and `gdisc_symm` makes those the same. -/
theorem gdisc_descend (m j p q : Nat) (hj : j < m+1) (hp : p < 2^(m+1)) (hq : q < 2^(m+1))
    (x y : Nat) (hx : x = p ∨ x = p + 2^(m+1)) (hy : y = q ∨ y = q + 2^(m+1)) :
    gdisc j x y (m+2) = gdisc j p q (m+1) := by
  have hsym := gdisc_symm (m+1) j q p hj hq hp
  rcases hx with hx | hx <;> rcases hy with hy | hy <;> rw [hx, hy]
  · exact gdisc_descend_ll m j p q hj hp hq
  · rw [gdisc_descend_lu m j p q hj hp hq]; exact hsym
  · exact gdisc_descend_ul m j p q hj hp hq
  · rw [gdisc_descend_uu m j p q hj hp hq]; exact hsym

private theorem seam_xor_lhs (u W n : Nat) (hu : u < 2^(n+1)) (hW : W < 2^(n+1)) :
    (u + 2^(n+1)) ^^^ W = (u ^^^ W) + 2^(n+1) := by
  rw [Nat.xor_comm, xor_seam W u n hW hu, Nat.xor_comm W u]

/-- **`N9`, proven ∀n.** (♦)'s conclusion is LEVEL-BOUNDED: the object it constrains is
    unchanged by dropping a level and truncating `Y`, `a`, `b`. Iterated, this collapses `G` at
    any level to `G` at level `j+2` — so (♦) is not a statement about an object that grows with
    the level, and its whole unbounded direction sits in the hypothesis. -/
theorem G_descend (m j W u v Y a b : Nat) (hj : j < m+1)
    (hW : W < 2^(m+1)) (hu : u < 2^(m+1)) (hv : v < 2^(m+1))
    (hY : Y = W ∨ Y = W + 2^(m+1)) (ha : a = u ∨ a = u + 2^(m+1))
    (hb : b = v ∨ b = v + 2^(m+1)) :
    gdisc j a b (m+2) * gdisc j (a ^^^ Y) (b ^^^ Y) (m+2)
      = gdisc j u v (m+1) * gdisc j (u ^^^ W) (v ^^^ W) (m+1) := by
  have huW : u ^^^ W < 2^(m+1) := Nat.xor_lt_two_pow hu hW
  have hvW : v ^^^ W < 2^(m+1) := Nat.xor_lt_two_pow hv hW
  have hxa : a ^^^ Y = u ^^^ W ∨ a ^^^ Y = (u ^^^ W) + 2^(m+1) := by
    rcases ha with ha | ha <;> rcases hY with hY | hY <;> rw [ha, hY]
    · exact Or.inl rfl
    · exact Or.inr (xor_seam u W m hu hW)
    · exact Or.inr (seam_xor_lhs u W m hu hW)
    · exact Or.inl (xor_seam_cancel u W m hu hW)
  have hxb : b ^^^ Y = v ^^^ W ∨ b ^^^ Y = (v ^^^ W) + 2^(m+1) := by
    rcases hb with hb | hb <;> rcases hY with hY | hY <;> rw [hb, hY]
    · exact Or.inl rfl
    · exact Or.inr (xor_seam v W m hv hW)
    · exact Or.inr (seam_xor_lhs v W m hv hW)
    · exact Or.inl (xor_seam_cancel v W m hv hW)
  rw [gdisc_descend m j u v hj hu hv a b ha hb,
      gdisc_descend m j (u ^^^ W) (v ^^^ W) hj huW hvW (a ^^^ Y) (b ^^^ Y) hxa hxb]

/-! ## `REACH`: the truncation is a THEOREM, so (♦) really is a finite statement

`G_descend` drops ONE level. Iterating it is what turns (♦) into "`REACH_j(Y₀) ⊆ {D = +1}`" —
a statement with no `n` in it. That iteration is proven here.

The point is not the arithmetic. It is that the *conclusion* of (♦) at any level whatsoever is
literally the value of `G` at level `k`, for every `k > j` — so the only thing (♦) can be about
is which bottom triples are reachable. `REACH` is not a convenient reformulation; after
`G_trunc` it is the whole content. -/

theorem xor_mod_two_pow (k x y : Nat) : (x ^^^ y) % 2^k = (x % 2^k) ^^^ (y % 2^k) := by
  apply Nat.eq_of_testBit_eq
  intro i
  simp only [Nat.testBit_mod_two_pow, Nat.testBit_xor]
  cases Nat.decLt i k with
  | isTrue h => simp [h]
  | isFalse h => simp [h]

private theorem split_half (m z : Nat) (hz : z < 2^(m+2)) :
    z = z % 2^(m+1) ∨ z = z % 2^(m+1) + 2^(m+1) := by
  have h2H : 2^(m+2) = 2^(m+1) + 2^(m+1) := by rw [Nat.pow_succ]; omega
  by_cases h : z < 2^(m+1)
  · exact Or.inl (Nat.mod_eq_of_lt h).symm
  · refine Or.inr ?_
    have he : z % 2^(m+1) = z - 2^(m+1) := by
      rw [Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt (by omega)]
    omega

/-- One level, in the `mod` form the iteration needs. -/
theorem gdisc_descend_mod (m j x y : Nat) (hj : j < m+1) (hx : x < 2^(m+2))
    (hy : y < 2^(m+2)) :
    gdisc j x y (m+2) = gdisc j (x % 2^(m+1)) (y % 2^(m+1)) (m+1) :=
  gdisc_descend m j (x % 2^(m+1)) (y % 2^(m+1)) hj
    (Nat.mod_lt _ (Nat.two_pow_pos (m+1))) (Nat.mod_lt _ (Nat.two_pow_pos (m+1)))
    x y (split_half m x hx) (split_half m y hy)

/-- **`gdisc` truncates to ANY level above `j`.** Induction on the number of levels dropped. -/
theorem gdisc_trunc (j kk : Nat) (hj : j < kk+1) :
    ∀ (d x y : Nat), x < 2^((kk+1)+d) → y < 2^((kk+1)+d) →
      gdisc j x y ((kk+1)+d) = gdisc j (x % 2^(kk+1)) (y % 2^(kk+1)) (kk+1) := by
  intro d
  induction d with
  | zero => intro x y hx hy; rw [Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt hy]
  | succ d ih =>
    intro x y hx hy
    have e1 : (kk+1) + (d+1) = (kk+d)+2 := by omega
    have e2 : (kk+d)+1 = (kk+1)+d := by omega
    rw [e1] at hx hy
    rw [e1, gdisc_descend_mod (kk+d) j x y (by omega) hx hy, e2,
        ih (x % 2^((kk+1)+d)) (y % 2^((kk+1)+d))
          (Nat.mod_lt _ (Nat.two_pow_pos ((kk+1)+d))) (Nat.mod_lt _ (Nat.two_pow_pos ((kk+1)+d))),
        Nat.mod_mod_of_dvd _ (Nat.pow_dvd_pow 2 (Nat.le_add_right (kk+1) d)),
        Nat.mod_mod_of_dvd _ (Nat.pow_dvd_pow 2 (Nat.le_add_right (kk+1) d))]

/-- **The truncation theorem — this is what makes `REACH` the whole content of (♦).**
    The conclusion of (♦) at ANY level is literally its value at level `k`, for every `k > j`.
    So the only thing (♦) can be about is which bottom triples are reachable. -/
theorem G_trunc (j kk d Y a b : Nat) (hj : j < kk+1) (hY : Y < 2^((kk+1)+d))
    (ha : a < 2^((kk+1)+d)) (hb : b < 2^((kk+1)+d)) :
    gdisc j a b ((kk+1)+d) * gdisc j (a ^^^ Y) (b ^^^ Y) ((kk+1)+d)
      = gdisc j (a % 2^(kk+1)) (b % 2^(kk+1)) (kk+1)
        * gdisc j ((a % 2^(kk+1)) ^^^ (Y % 2^(kk+1)))
                  ((b % 2^(kk+1)) ^^^ (Y % 2^(kk+1))) (kk+1) := by
  rw [gdisc_trunc j kk hj d a b ha hb,
      gdisc_trunc j kk hj d (a ^^^ Y) (b ^^^ Y) (Nat.xor_lt_two_pow ha hY)
        (Nat.xor_lt_two_pow hb hY),
      xor_mod_two_pow (kk+1) a Y, xor_mod_two_pow (kk+1) b Y]

/-! ## `REACH` is monotone, so its limit exists

`REACH_j(Y₀)` at level `n` is the set of bottom triples carried by some level-`n` tuple that
satisfies (♦)'s hypothesis. The rung measured that it stabilises at `n = j+4`. Half of that is
a theorem and needs no measurement at all:

`Q'red_low_ll` reduces `Qgen'` at a LOW label with BOTH arguments low **with no side conditions
whatsoever** — it is the one lemma of the sixteen that is unconditional. So a level-`n` witness
is verbatim a level-`(n+1)` witness: nothing has to be re-established, and `REACH` can only
grow. Being a monotone family of subsets of the fixed finite set `[0,2^{j+2})²`, it therefore
has a limit — which is what makes "`REACH_j(Y₀) ⊆ {D = +1}`" a well-posed statement rather than
one secretly quantified over `n`.

What is NOT proven here is the quantitative part: that the limit is *attained* at `n = j+4`.
That remains measured (`j ≤ 3`, `n ≤ 8`). -/

/-- A level-`n` witness for (♦)'s hypothesis is verbatim a level-`(n+1)` witness. -/
theorem reach_step (m W a b : Nat) (hW : W < 2^(m+1)) (ha : a < 2^(m+1)) (hb : b < 2^(m+1))
    (h : Qgen' W a b (m+1) = -1) : Qgen' W a b (m+2) = -1 := by
  rw [Q'red_low_ll m W a b hW ha hb]
  exact h

/-- `REACH j Y₀ n a₀ b₀` — the bottom triple `(Y₀,a₀,b₀)` is carried by some level-`n` tuple
    satisfying (♦)'s hypothesis. -/
def Reach (j Y0 n a0 b0 : Nat) : Prop :=
  ∃ Y a b, Y < 2^n ∧ a < 2^n ∧ b < 2^n ∧
    Y % 2^(j+2) = Y0 ∧ a % 2^(j+2) = a0 ∧ b % 2^(j+2) = b0 ∧ Qgen' Y a b n = -1

/-- **`REACH` never shrinks.** One level. -/
theorem Reach_succ (j Y0 n a0 b0 : Nat) (h : Reach j Y0 (n+1) a0 b0) :
    Reach j Y0 (n+2) a0 b0 := by
  obtain ⟨Y, a, b, hY, ha, hb, eY, ea, eb, hq⟩ := h
  have h2 : (2:Nat)^(n+1) ≤ 2^(n+2) := Nat.pow_le_pow_right (by omega) (by omega)
  exact ⟨Y, a, b, by omega, by omega, by omega, eY, ea, eb,
    reach_step n Y a b hY ha hb hq⟩

/-- **`REACH` never shrinks.** Any number of levels — so the family is monotone in `n`, sits
    inside the fixed finite square `[0,2^(j+2))²`, and its limit exists. -/
theorem Reach_mono (j Y0 a0 b0 : Nat) :
    ∀ (d n : Nat), Reach j Y0 (n+1) a0 b0 → Reach j Y0 (n+1+d) a0 b0
  | 0, _, h => h
  | (d+1), n, h => by
      have hprev := Reach_mono j Y0 a0 b0 d n h
      have e : n + 1 + (d + 1) = (n + d + 1) + 1 := by omega
      have e2 : n + 1 + d = (n + d) + 1 := by omega
      rw [e]
      rw [e2] at hprev
      exact Reach_succ j Y0 (n + d) a0 b0 hprev

/-! ## Why the stabilisation boundary is `n = j+4` and not `n = j+3`

The rung measured that `REACH` gains exactly **four** points between level `j+3` and level
`j+4`, at every `j` tested, and that those four are always the same four: the corners
`{0,Y₀} × {0,Y₀}`. This section proves the half of that which is a theorem — that level `j+3`
**cannot** carry the corner `(0,0)` — and the cause is one line of `σ`-arithmetic.

`Qgen'` on the diagonal has no freedom left. Two of its four factors are literally the same
factor, so they square away by `cdSq`, and the other two are self-pairings, which `sigma_self`
pins to `−1`:

```
Q'(W,a,a) = σ(a,a) · σ(a⊕W,a⊕W) · σ(a⊕W,a)²  =  (−1)(−1)(1)  =  +1
```

So **no diagonal tuple is ever a witness**. And at level `j+3` a bottom pair `(0,0)` forces the
diagonal: `a < 2^{j+3}` with `a mod 2^{j+2} = 0` and `a ≠ 0` leaves `a = 2^{j+2}` as the only
value, and likewise for `b`, so `a = b`. You need **two** spare bits to make `a ≠ b`, which is
exactly the `n = j+4` boundary and exactly the deficit of four. -/

/-- `x < 2^(k+1)`, `x mod 2^k = 0`, `x ≠ 0` leaves exactly one value: `x = 2^k`. -/
theorem step_forced (k x : Nat) (hx : x < 2^(k+1)) (hm : x % 2^k = 0) (hx0 : x ≠ 0) :
    x = 2^k := by
  have hp : 0 < 2^k := Nat.two_pow_pos k
  have hsplit := Nat.div_add_mod x (2^k)
  rw [hm, Nat.add_zero] at hsplit
  have hlt : x / 2^k < 2 := by
    have h2 : (2:Nat)^(k+1) = 2^k * 2 := by rw [Nat.pow_succ]
    rw [h2] at hx
    exact Nat.div_lt_of_lt_mul (by omega)
  have hq : x / 2^k = 0 ∨ x / 2^k = 1 := by
    cases hd : x / 2^k with
    | zero => exact Or.inl rfl
    | succ q =>
        cases q with
        | zero => exact Or.inr rfl
        | succ q' => rw [hd] at hlt; omega
  rcases hq with h | h <;> rw [h] at hsplit
  · rw [Nat.mul_zero] at hsplit
    exact absurd hsplit.symm hx0
  · rw [Nat.mul_one] at hsplit
    exact hsplit.symm

/-- **`Qgen'` is `+1` on the diagonal.** Two of the four factors coincide and square away
    (`cdSq`); the other two are self-pairings, which `sigma_self` pins to `−1`. -/
theorem Qgen'_diag (m W a : Nat) (hW : W < 2^m) (ha : a < 2^m) (ha0 : a ≠ 0)
    (haW : a ^^^ W ≠ 0) : Qgen' W a a m = 1 := by
  have haWlt : a ^^^ W < 2^m := Nat.xor_lt_two_pow ha hW
  have e1 : cdSigma a a m = -1 := sigma_self m a ha ha0
  have e2 : cdSigma (a ^^^ W) (a ^^^ W) m = -1 := sigma_self m (a ^^^ W) haWlt haW
  unfold Qgen'
  rw [e1, e2]
  rcases cdSigma_pm m (a ^^^ W) a with h | h <;> rw [h] <;> decide

/-- **No diagonal tuple satisfies (♦)'s hypothesis.** -/
theorem diag_not_witness (m W a : Nat) (hW : W < 2^m) (ha : a < 2^m) (ha0 : a ≠ 0)
    (haW : a ^^^ W ≠ 0) : Qgen' W a a m ≠ -1 := by
  rw [Qgen'_diag m W a hW ha ha0 haW]
  decide

/-- **`Qgen` is symmetric off the degenerate locus.** All four factors flip under `antisym`, and
    four flips cancel. The hypotheses needed are *exactly* the six non-degeneracy conditions the
    numerical rung found load-bearing — `a ≠ b` and `a ⊕ b ≠ L` are what make the second and
    third flips legal. -/
theorem Qgen_symm (m L a b : Nat) (hL : L < 2^m) (ha : a < 2^m) (hb : b < 2^m)
    (ha0 : a ≠ 0) (hb0 : b ≠ 0) (haL : a ^^^ L ≠ 0) (hbL : b ^^^ L ≠ 0)
    (hab : a ≠ b) (habL : a ^^^ b ^^^ L ≠ 0) :
    Qgen L a b m = Qgen L b a m := by
  have haLlt : a ^^^ L < 2^m := Nat.xor_lt_two_pow ha hL
  have hbLlt : b ^^^ L < 2^m := Nat.xor_lt_two_pow hb hL
  have hne2 : a ^^^ L ≠ b ^^^ L := by
    intro h
    exact hab (by
      have := congrArg (fun t => t ^^^ L) h
      simpa [Nat.xor_assoc, Nat.xor_self] using this)
  have hne3 : a ^^^ L ≠ b := (xor3_ne_right habL).symm
  have hne4 : a ≠ b ^^^ L := xor3_ne_left habL
  have e1 : cdSigma a b m = - cdSigma b a m := antisym m a b ha hb ha0 hb0 hab
  have e2 : cdSigma (a ^^^ L) (b ^^^ L) m = - cdSigma (b ^^^ L) (a ^^^ L) m :=
    antisym m (a ^^^ L) (b ^^^ L) haLlt hbLlt haL hbL hne2
  have e3 : cdSigma (a ^^^ L) b m = - cdSigma b (a ^^^ L) m :=
    antisym m (a ^^^ L) b haLlt hb haL hb0 hne3
  have e4 : cdSigma a (b ^^^ L) m = - cdSigma (b ^^^ L) a m :=
    antisym m a (b ^^^ L) ha hbLlt ha0 hbL hne4
  unfold Qgen
  rw [e1, e2, e3, e4]
  rcases cdSigma_pm m b a with h1 | h1 <;>
    rcases cdSigma_pm m (b ^^^ L) (a ^^^ L) with h2 | h2 <;>
    rcases cdSigma_pm m (b ^^^ L) a with h3 | h3 <;>
    rcases cdSigma_pm m b (a ^^^ L) with h4 | h4 <;>
    rw [h1, h2, h3, h4] <;> decide

/-- `REACH` with (♦)'s **actual** side conditions.

    `Reach` above drops `a ≠ 0`, `b ≠ 0`, `b ≠ Y`, and so it is strictly bigger than the set the
    numerical rung measures: `Q'(Y,0,0) = σ(Y,Y) = −1` for every `Y ≠ 0`, so the corner `(0,0)`
    sits in `Reach` at *every* level, trivially, and no boundary statement about it could be
    true. `ReachD` is the measured object. Monotonicity survives unchanged, because
    `reach_step` reuses the very same `(Y,a,b)`. -/
def ReachD (j Y0 n a0 b0 : Nat) : Prop :=
  ∃ Y a b, Y < 2^n ∧ a < 2^n ∧ b < 2^n ∧
    Y % 2^(j+2) = Y0 ∧ a % 2^(j+2) = a0 ∧ b % 2^(j+2) = b0 ∧
    a ≠ 0 ∧ b ≠ 0 ∧ b ≠ Y ∧ Qgen' Y a b n = -1

/-- `ReachD` never shrinks either — the guards are carried by the same tuple. -/
theorem ReachD_succ (j Y0 n a0 b0 : Nat) (h : ReachD j Y0 (n+1) a0 b0) :
    ReachD j Y0 (n+2) a0 b0 := by
  obtain ⟨Y, a, b, hY, ha, hb, eY, ea, eb, ha0, hb0, hbY, hq⟩ := h
  have h2 : (2:Nat)^(n+1) ≤ 2^(n+2) := Nat.pow_le_pow_right (by omega) (by omega)
  exact ⟨Y, a, b, by omega, by omega, by omega, eY, ea, eb, ha0, hb0, hbY,
    reach_step n Y a b hY ha hb hq⟩

theorem ReachD_mono (j Y0 a0 b0 : Nat) :
    ∀ (d n : Nat), ReachD j Y0 (n+1) a0 b0 → ReachD j Y0 (n+1+d) a0 b0
  | 0, _, h => h
  | (d+1), n, h => by
      have hprev := ReachD_mono j Y0 a0 b0 d n h
      have e : n + 1 + (d + 1) = (n + d + 1) + 1 := by omega
      have e2 : n + 1 + d = (n + d) + 1 := by omega
      rw [e]
      rw [e2] at hprev
      exact ReachD_succ j Y0 (n + d) a0 b0 hprev

/-- **The corner `(0,0)` is blocked at level `j+3`.** This is the sharpness half of the
    `n = j+4` boundary, and it needs no measurement: one spare bit forces `a = b = 2^{j+2}`,
    and the diagonal is never a witness. `Y₀ ≠ 0` is what `lsb(Y) = j` gives. -/
theorem corner_blocked_at_j3 (j Y0 : Nat) (hY0 : Y0 ≠ 0) : ¬ ReachD j Y0 (j+3) 0 0 := by
  rintro ⟨Y, a, b, hY, ha, hb, eY, ea, eb, ha0, hb0, _, hq⟩
  have hae : a = 2^(j+2) := step_forced (j+2) a ha ea ha0
  have hbe : b = 2^(j+2) := step_forced (j+2) b hb eb hb0
  have haY : a ^^^ Y ≠ 0 := by
    intro h
    have : a = Y := xor_zero_eq a Y h
    rw [this, eY] at ea
    exact hY0 ea
  rw [hbe, ← hae] at hq
  exact diag_not_witness (j+3) Y a hY ha ha0 haY hq

end SounioZDFiberAntisym










