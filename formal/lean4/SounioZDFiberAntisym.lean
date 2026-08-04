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
    satisfying (♦)'s hypothesis.

    ⚠ **DEPRECATED — use `ReachD`.** This definition drops (♦)'s side conditions `a ≠ 0`,
    `b ≠ 0`, `b ≠ Y`, so it is strictly larger than the set the numerical rung measures:
    `Q'(Y,0,0) = σ(Y,Y) = −1` for every `Y ≠ 0`, and so the corner `(0,0)` lies in `Reach` at
    *every* level. No boundary statement about `Reach` can be true. Kept only because
    `Reach_succ`/`Reach_mono` are stated for it; nothing depends on them. -/
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
    `reach_step` reuses the very same `(Y,a,b)`.

    ⚠ `ReachD` does **not** carry (♦)'s even-weight condition on `Y` — expressing it needs a
    popcount, which nothing here does. So `ReachD` is a slightly *larger* family than the
    contract's `REACH`, and the theorems below are correspondingly slightly stronger on the
    existential side: `attain_lines` builds a witness for **either** admissible label, and the
    caller takes whichever has even weight. Clause `N27` of the contract checks that the label
    `attain_nondeg` selects is the even-weight one. -/
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

/-! ## The COLLAPSE theorem — (♦)'s **hypothesis** is level-bounded too

`G_descend`/`G_trunc` bounded (♦)'s *conclusion*: nothing above bit `j+1` can affect it. The
open half was its *hypothesis*, `Qgen'(Y,a,b) = −1`, and the previous rungs recorded that as the
thing that does not survive truncation. It does — off the six degeneracy lines:

```
Qgen' Y a b n  =  dsgnN j n Y  *  Qgen (Y % 2^(j+2)) (a % 2^(j+2)) (b % 2^(j+2)) (j+2)
```

with `dsgnN` the accumulated descent sign — `−1` once per level at which the label's bit is set,
which is exactly the sign law `N11` reads off the sixteen rows. **No `n` on the right.**

Why the three pieces of state the sixteen rows carry — sign, priming, argument swap — collapse
to just the sign: off the six lines `Qgen = Qgen'` (`Qgen_eq_Qgen'`) and `Qgen` is symmetric
(`Qgen_symm`), so the priming and the swap are invisible at the bottom. That is also the reason
those six conditions, and no others, are the load-bearing ones. -/

/-- The accumulated descent sign: `−1` once per level, above `j+2`, at which the label's bit is
    set. -/
def dsgnN (j : Nat) : Nat → Nat → Int
  | 0, _ => 1
  | (n+1), Y =>
      if n+1 ≤ j+2 then 1
      else (if Y / 2^n % 2 = 1 then (-1 : Int) else 1) * dsgnN j n (Y % 2^n)

/-- The six non-degeneracy conditions, read at the bottom `j+2` bits — the `n`-free form of the
    "clean locus" the earlier rungs tested level by level. -/
def NDeg (j Y a b : Nat) : Prop :=
  a % 2^(j+2) ≠ 0 ∧ b % 2^(j+2) ≠ 0 ∧
  a % 2^(j+2) ≠ Y % 2^(j+2) ∧ b % 2^(j+2) ≠ Y % 2^(j+2) ∧
  a % 2^(j+2) ≠ b % 2^(j+2) ∧
  (a % 2^(j+2)) ^^^ (b % 2^(j+2)) ≠ Y % 2^(j+2)

theorem xor4_cancel (a b Y : Nat) : (a ^^^ Y) ^^^ (b ^^^ Y) = a ^^^ b := by
  rw [Nat.xor_assoc, ← Nat.xor_assoc Y b Y, Nat.xor_comm Y b, Nat.xor_assoc b Y Y,
      Nat.xor_self, Nat.xor_zero]

theorem NDeg_symm {j Y a b : Nat} (h : NDeg j Y a b) : NDeg j Y b a := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := h
  exact ⟨h2, h1, h4, h3, fun e => h5 e.symm, by rw [Nat.xor_comm]; exact h6⟩

theorem mod_pow_mod (k l x : Nat) (h : k ≤ l) : x % 2^l % 2^k = x % 2^k :=
  Nat.mod_mod_of_dvd x (Nat.pow_dvd_pow 2 h)

theorem add_pow_mod (k l x : Nat) (h : k ≤ l) : (x + 2^l) % 2^k = x % 2^k := by
  obtain ⟨c, hc⟩ := Nat.pow_dvd_pow 2 h
  rw [hc, Nat.add_mul_mod_self_left]

/-- `NDeg` only looks at the bottom `j+2` bits, so anything with the same residues has it. -/
theorem NDeg_congr {j Y a b Y' a' b' : Nat}
    (hY : Y' % 2^(j+2) = Y % 2^(j+2)) (ha : a' % 2^(j+2) = a % 2^(j+2))
    (hb : b' % 2^(j+2) = b % 2^(j+2)) (h : NDeg j Y a b) : NDeg j Y' a' b' := by
  unfold NDeg at h ⊢
  rw [hY, ha, hb]
  exact h

/-- The six conditions, transported from the bottom residues to the values themselves. -/
theorem NDeg_facts {j Y a b : Nat} (h : NDeg j Y a b) :
    a ≠ 0 ∧ b ≠ 0 ∧ a ^^^ Y ≠ 0 ∧ b ^^^ Y ≠ 0 ∧ a ≠ b ∧ a ^^^ b ^^^ Y ≠ 0 := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := h
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro e; rw [e] at h1; exact h1 (Nat.zero_mod _)
  · intro e; rw [e] at h2; exact h2 (Nat.zero_mod _)
  · intro e; exact h3 (by rw [xor_zero_eq a Y e])
  · intro e; exact h4 (by rw [xor_zero_eq b Y e])
  · intro e; exact h5 (by rw [e])
  · intro e
    have h7 := congrArg (fun t => t % 2^(j+2)) (xor_zero_eq (a ^^^ b) Y e)
    simp only [xor_mod_two_pow] at h7
    exact h6 h7

/-- Off the six lines, `Qgen` and `Qgen'` agree — at every level. -/
theorem QQ' (j n W x y : Nat) (hW : W < 2^n) (hx : x < 2^n) (hy : y < 2^n)
    (h : NDeg j W x y) : Qgen W x y n = Qgen' W x y n := by
  obtain ⟨f1, f2, f3, f4, f5, f6⟩ := NDeg_facts h
  refine Qgen_eq_Qgen' W x y n hx hy hW f3 f4 ?_ f1 ?_
  · rw [xor4_cancel]
    intro e; exact f5 (xor_zero_eq x y e)
  · rw [← Nat.xor_assoc]; exact f6

/-- The bottom value is symmetric in its two arguments, off the six lines. -/
theorem QB_symm (j Y a b : Nat) (h : NDeg j Y a b) :
    Qgen (Y % 2^(j+2)) (b % 2^(j+2)) (a % 2^(j+2)) (j+2)
      = Qgen (Y % 2^(j+2)) (a % 2^(j+2)) (b % 2^(j+2)) (j+2) := by
  have hmm : ∀ x : Nat, x % 2^(j+2) % 2^(j+2) = x % 2^(j+2) :=
    fun x => mod_pow_mod (j+2) (j+2) x (Nat.le_refl _)
  have hb : NDeg j (Y % 2^(j+2)) (b % 2^(j+2)) (a % 2^(j+2)) :=
    NDeg_congr (hmm Y) (hmm b) (hmm a) (NDeg_symm h)
  obtain ⟨f1, f2, f3, f4, f5, f6⟩ := NDeg_facts hb
  have hp : (0:Nat) < 2^(j+2) := Nat.two_pow_pos (j+2)
  exact Qgen_symm (j+2) (Y % 2^(j+2)) (b % 2^(j+2)) (a % 2^(j+2))
    (Nat.mod_lt _ hp) (Nat.mod_lt _ hp) (Nat.mod_lt _ hp) f1 f2 f3 f4 f5 f6

/-- `x < 2^(k+1)` splits into a low value or a low value plus the top bit. -/
theorem split_top {k x : Nat} (hx : x < 2^(k+1)) : x < 2^k ∨ ∃ y, y < 2^k ∧ x = y + 2^k := by
  by_cases h : x < 2^k
  · exact Or.inl h
  · have h2 : (2:Nat)^(k+1) = 2^k * 2 := by rw [Nat.pow_succ]
    exact Or.inr ⟨x - 2^k, by omega, by omega⟩

theorem dsgnN_low (j k Y : Nat) (hk : j+2 ≤ k+1) (hY : Y < 2^(k+1)) :
    dsgnN j (k+2) Y = dsgnN j (k+1) Y := by
  have h1 : ¬ (k+2 ≤ j+2) := by omega
  have h2 : Y / 2^(k+1) = 0 := Nat.div_eq_of_lt hY
  have h3 : Y % 2^(k+1) = Y := Nat.mod_eq_of_lt hY
  rw [dsgnN, if_neg h1, h2, h3]
  simp

theorem dsgnN_hi (j k W : Nat) (hk : j+2 ≤ k+1) (hW : W < 2^(k+1)) :
    dsgnN j (k+2) (W + 2^(k+1)) = - dsgnN j (k+1) W := by
  have h1 : ¬ (k+2 ≤ j+2) := by omega
  have hp : 0 < 2^(k+1) := Nat.two_pow_pos (k+1)
  have h2 : (W + 2^(k+1)) / 2^(k+1) = 1 := by
    rw [Nat.add_div_right W hp, Nat.div_eq_of_lt hW]
  have h3 : (W + 2^(k+1)) % 2^(k+1) = W := by
    rw [Nat.add_mod_right, Nat.mod_eq_of_lt hW]
  rw [dsgnN, if_neg h1, h2, h3]
  simp

/-- **THE COLLAPSE THEOREM.** Off the six degeneracy lines, `Qgen'` at any level is the
    accumulated descent sign times its value at the bottom level `j+2`. The right-hand side does
    not mention `n`. This is the hypothesis-side counterpart of `G_trunc`. -/
theorem collapse (j : Nat) : ∀ (n Y a b : Nat), j+2 ≤ n →
    Y < 2^n → a < 2^n → b < 2^n → NDeg j Y a b →
    Qgen' Y a b n
      = dsgnN j n Y * Qgen (Y % 2^(j+2)) (a % 2^(j+2)) (b % 2^(j+2)) (j+2) := by
  intro n
  induction n with
  | zero => intro Y a b hn; omega
  | succ n ih =>
      intro Y a b hn hY ha hb hnd
      rcases Nat.eq_or_lt_of_le hn with heq | hlt
      · have hne : n = j + 1 := by omega
        subst hne
        obtain ⟨f1, f2, f3, f4, f5, f6⟩ := NDeg_facts hnd
        have eY : Y % 2^(j+2) = Y := Nat.mod_eq_of_lt hY
        have ea : a % 2^(j+2) = a := Nat.mod_eq_of_lt ha
        have eb : b % 2^(j+2) = b := Nat.mod_eq_of_lt hb
        have hs : dsgnN j (j+1+1) Y = 1 := by
          rw [dsgnN, if_pos (by omega : j+1+1 ≤ j+2)]
        rw [eY, ea, eb, hs, Int.one_mul]
        exact (QQ' j (j+1+1) Y a b hY ha hb hnd).symm
      · have hjn : j + 2 ≤ n := by omega
        obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
        have hk : j + 2 ≤ k + 1 := hjn
        have hp : (0:Nat) < 2^(j+2) := Nat.two_pow_pos (j+2)
        have hadd : ∀ x : Nat, (x + 2^(k+1)) % 2^(j+2) = x % 2^(j+2) :=
          fun x => add_pow_mod (j+2) (k+1) x hk
        rcases split_top hY with hYl | ⟨W, hW, rfl⟩ <;>
          rcases split_top ha with hal | ⟨u, hu, rfl⟩ <;>
          rcases split_top hb with hbl | ⟨v, hv, rfl⟩
        -- 1. label low, a low, b low
        · rw [Q'red_low_ll k Y a b hYl hal hbl, dsgnN_low j k Y hk hYl]
          exact ih Y a b hk hYl hal hbl hnd
        -- 2. label low, a low, b upper  (row returns Qgen, arguments SWAPPED)
        · have hnd' : NDeg j Y a v := NDeg_congr rfl rfl (hadd v).symm hnd
          obtain ⟨f1, f2, f3, f4, f5, f6⟩ := NDeg_facts hnd'
          rw [Q'red_low_lu k Y a v hYl hal hv f1 f3, dsgnN_low j k Y hk hYl,
              QQ' j (k+1) Y v a hYl hv hal (NDeg_symm hnd'),
              ih Y v a hk hYl hv hal (NDeg_symm hnd'), hadd v,
              QB_symm j Y a v hnd']
        -- 3. label low, a upper, b low
        · have hnd' : NDeg j Y u b := NDeg_congr rfl (hadd u).symm rfl hnd
          obtain ⟨f1, f2, f3, f4, f5, f6⟩ := NDeg_facts hnd'
          rw [Q'red_low_ul k Y u b hYl hu hbl f2, dsgnN_low j k Y hk hYl,
              QQ' j (k+1) Y u b hYl hu hbl hnd',
              ih Y u b hk hYl hu hbl hnd', hadd u]
        -- 4. label low, a upper, b upper  (SWAPPED)
        · have hnd' : NDeg j Y u v := NDeg_congr rfl (hadd u).symm (hadd v).symm hnd
          obtain ⟨f1, f2, f3, f4, f5, f6⟩ := NDeg_facts hnd'
          rw [Q'red_low_uu k Y u v hYl hu hv f1 f2 f3 f4 f6, dsgnN_low j k Y hk hYl,
              ih Y v u hk hYl hv hu (NDeg_symm hnd'), hadd u, hadd v,
              QB_symm j Y u v hnd']
        -- 5. label HIGH, a low, b low
        · have hnd' : NDeg j W a b := NDeg_congr (hadd W).symm rfl rfl hnd
          obtain ⟨f1, f2, f3, f4, f5, f6⟩ := NDeg_facts hnd'
          rw [Q'red_hi_ll k W a b hW hal hbl f1 f2 f3 f4 f5, dsgnN_hi j k W hk hW,
              ih W a b hk hW hal hbl hnd', hadd W]
          rw [Int.neg_mul]
        -- 6. label HIGH, a low, b upper  (SWAPPED)
        · have hnd' : NDeg j W a v := NDeg_congr (hadd W).symm rfl (hadd v).symm hnd
          obtain ⟨f1, f2, f3, f4, f5, f6⟩ := NDeg_facts hnd'
          rw [Q'red_hi_lu k W a v hW hal hv f2 f3 f4 f5, dsgnN_hi j k W hk hW,
              QQ' j (k+1) W v a hW hv hal (NDeg_symm hnd'),
              ih W v a hk hW hv hal (NDeg_symm hnd'), hadd W, hadd v,
              QB_symm j W a v hnd']
          rw [Int.neg_mul]
        -- 7. label HIGH, a upper, b low
        · have hnd' : NDeg j W u b := NDeg_congr (hadd W).symm (hadd u).symm rfl hnd
          obtain ⟨f1, f2, f3, f4, f5, f6⟩ := NDeg_facts hnd'
          rw [Q'red_hi_ul k W u b hW hu hbl f1 f2 f3 f4 f5, dsgnN_hi j k W hk hW,
              QQ' j (k+1) W u b hW hu hbl hnd',
              ih W u b hk hW hu hbl hnd', hadd W, hadd u]
          rw [Int.neg_mul]
        -- 8. label HIGH, a upper, b upper  (SWAPPED)
        · have hnd' : NDeg j W u v := NDeg_congr (hadd W).symm (hadd u).symm (hadd v).symm hnd
          obtain ⟨f1, f2, f3, f4, f5, f6⟩ := NDeg_facts hnd'
          rw [Q'red_hi_uu k W u v hW hu hv f1 f2 f3 f4 f5 f6, dsgnN_hi j k W hk hW,
              ih W v u hk hW hv hu (NDeg_symm hnd'), hadd W, hadd u, hadd v,
              QB_symm j W u v hnd']
          rw [Int.neg_mul]

/-! ## What the collapse theorem buys: attainment, off the six lines

Two tuples at *different* levels with the same bottom residues and the same accumulated sign
have the **same** `Qgen'` — because both are that sign times the same bottom value. So a witness
at any level `n` transfers down to a witness at level `j+3`, and the only thing that has to be
matched is one bit of the label: at level `j+3` the label may carry bit `j+2` or not, and those
two choices realise `dsgnN = +1` and `dsgnN = −1` respectively. Nothing else about `n` survives.

This is the non-degenerate half of attainment, ∀n. The other half is the six lines themselves,
where `Qgen ≠ Qgen'` and the collapse does not apply; there `REACH` is full and the witnesses are
the two explicit families `N25` records, at level `j+4` — which is what makes the boundary `j+4`
and not `j+3`. -/

theorem dsgnN_pm (j : Nat) : ∀ (n Y : Nat), dsgnN j n Y = 1 ∨ dsgnN j n Y = -1 := by
  intro n
  induction n with
  | zero => intro Y; exact Or.inl rfl
  | succ n ih =>
      intro Y
      by_cases h : n+1 ≤ j+2
      · rw [dsgnN, if_pos h]; exact Or.inl rfl
      · rw [dsgnN, if_neg h]
        by_cases hb : Y / 2^n % 2 = 1
        · rw [if_pos hb]
          rcases ih (Y % 2^n) with e | e <;> rw [e]
          · exact Or.inr (by decide)
          · exact Or.inl (by decide)
        · rw [if_neg hb]
          rcases ih (Y % 2^n) with e | e <;> rw [e]
          · exact Or.inl (by decide)
          · exact Or.inr (by decide)

theorem dsgnN_bot (j x : Nat) : dsgnN j (j+2) x = 1 := by
  rw [dsgnN, if_pos (by omega : j+1+1 ≤ j+2)]

/-- Same bottom residues and same accumulated sign ⟹ same `Qgen'`, across levels. -/
theorem collapse_transfer (j n m Y a b Y' a' b' : Nat)
    (hn : j+2 ≤ n) (hm : j+2 ≤ m)
    (hY : Y < 2^n) (ha : a < 2^n) (hb : b < 2^n) (hnd : NDeg j Y a b)
    (hY' : Y' < 2^m) (ha' : a' < 2^m) (hb' : b' < 2^m)
    (eY : Y' % 2^(j+2) = Y % 2^(j+2)) (ea : a' % 2^(j+2) = a % 2^(j+2))
    (eb : b' % 2^(j+2) = b % 2^(j+2))
    (es : dsgnN j m Y' = dsgnN j n Y) :
    Qgen' Y' a' b' m = Qgen' Y a b n := by
  have hnd' : NDeg j Y' a' b' := NDeg_congr eY ea eb hnd
  rw [collapse j m Y' a' b' hm hY' ha' hb' hnd', collapse j n Y a b hn hY ha hb hnd,
      eY, ea, eb, es]

/-- **Attainment off the six lines, ∀n.** Any level-`n` tuple with a non-degenerate bottom is
    matched, value for value, by a level-`(j+3)` tuple with the same bottom residues. In
    particular a witness stays a witness. -/
theorem attain_nondeg (j n Y a b : Nat) (hn : j+2 ≤ n)
    (hY : Y < 2^n) (ha : a < 2^n) (hb : b < 2^n) (hnd : NDeg j Y a b) :
    ∃ Y', Y' < 2^(j+3) ∧ Y' % 2^(j+2) = Y % 2^(j+2) ∧
      Qgen' Y' (a % 2^(j+2)) (b % 2^(j+2)) (j+3) = Qgen' Y a b n := by
  have hp : (0:Nat) < 2^(j+2) := Nat.two_pow_pos (j+2)
  have hlt : (2:Nat)^(j+2) < 2^(j+3) := Nat.pow_lt_pow_right (by omega) (by omega)
  have hmm : ∀ x : Nat, x % 2^(j+2) % 2^(j+2) = x % 2^(j+2) :=
    fun x => mod_pow_mod (j+2) (j+2) x (Nat.le_refl _)
  have ha0 : a % 2^(j+2) < 2^(j+3) := Nat.lt_trans (Nat.mod_lt _ hp) hlt
  have hb0 : b % 2^(j+2) < 2^(j+3) := Nat.lt_trans (Nat.mod_lt _ hp) hlt
  have hY0 : Y % 2^(j+2) < 2^(j+2) := Nat.mod_lt _ hp
  -- the two candidate labels at level j+3, and their signs
  have key : ∀ Y' : Nat, Y' < 2^(j+3) → Y' % 2^(j+2) = Y % 2^(j+2) →
      dsgnN j (j+3) Y' = dsgnN j n Y →
      Qgen' Y' (a % 2^(j+2)) (b % 2^(j+2)) (j+3) = Qgen' Y a b n := by
    intro Y' h1 h2 h3
    exact collapse_transfer j n (j+3) Y a b Y' (a % 2^(j+2)) (b % 2^(j+2))
      hn (by omega) hY ha hb hnd h1 ha0 hb0 h2 (hmm a) (hmm b) h3
  have hsplit : dsgnN j (j+3) (Y % 2^(j+2)) = 1 := by
    rw [dsgnN, if_neg (by omega : ¬ (j+2+1 ≤ j+2))]
    rw [Nat.div_eq_of_lt hY0, Nat.mod_eq_of_lt hY0, dsgnN_bot]
    simp
  have hsplit' : dsgnN j (j+3) (Y % 2^(j+2) + 2^(j+2)) = -1 := by
    rw [dsgnN, if_neg (by omega : ¬ (j+2+1 ≤ j+2))]
    rw [Nat.add_div_right _ hp, Nat.div_eq_of_lt hY0, Nat.add_mod_right,
        Nat.mod_eq_of_lt hY0, dsgnN_bot]
    simp
  rcases dsgnN_pm j n Y with e | e
  · exact ⟨Y % 2^(j+2), Nat.lt_trans hY0 hlt, hmm Y, key _ (Nat.lt_trans hY0 hlt) (hmm Y)
      (by rw [hsplit, e])⟩
  · refine ⟨Y % 2^(j+2) + 2^(j+2), ?_, ?_, key _ ?_ ?_ (by rw [hsplit', e])⟩
    · have : (2:Nat)^(j+3) = 2^(j+2) * 2 := by rw [Nat.pow_succ]
      omega
    · rw [Nat.add_mod_right, hmm Y]
    · have : (2:Nat)^(j+3) = 2^(j+2) * 2 := by rw [Nat.pow_succ]
      omega
    · rw [Nat.add_mod_right, hmm Y]

/-! ## Attainment ON the six lines — the other half

Off the six lines `collapse` does everything. **On** them `Qgen ≠ Qgen'`, the collapse does not
apply, and `REACH` is *full* — every point of every line is reachable. This section proves that,
by exhibiting the witnesses.

The construction is short for two reasons. First, two of the sixteen rows — `Q'red_low_ul` and
`Q'red_low_lu` — have almost no side conditions, so one step suffices and the descent **stops** at
level `j+3` instead of reaching the bottom. Second, what it stops on has a closed form:

```
Qgen_zero_left : Qgen L 0 t m = −1        Qgen_diag_neg : Qgen L t t m = −1      (L ≠ 0)
```

Contrast `Qgen'_diag = +1`: `Qgen` and `Qgen'` disagree exactly on the diagonal, and that
disagreement is what makes the boundary `j+4` rather than `j+3`.

With `H = 2^{j+2}`, `Y = Y₀ + c·H` (`c ∈ {0,1}` — the caller takes whichever has even weight),
`a*`/`b*` the values `pick_low` supplies, and every second argument written as `v + 2^{j+3}`:

| line | `a` | `b` | lands on |
|---|---|---|---|
| `a₀ = 0`      | `0 + 2^{j+3}`  | `b*`                | `Qgen Y 0 b*` |
| `a₀ = Y₀`     | `Y + 2^{j+3}`  | `b*`                | `Qgen Y Y b*` → `Qgen Y 0 b*` |
| `b₀ = 0`      | `a*`           | `0 + 2^{j+3}`       | `Qgen Y 0 a*` |
| `b₀ = Y₀`     | `a*`           | `Y + 2^{j+3}`       | `Qgen Y Y a*` → `Qgen Y 0 a*` |
| `a₀ = b₀`     | `a*`           | `a* + 2^{j+3}`      | `Qgen Y a* a*` |
| `a₀⊕b₀ = Y₀`  | `a*`           | `(a*⊕Y) + 2^{j+3}`  | `Qgen Y (a*⊕Y) a*` → `Qgen Y a* a*` |

The two `→` steps are `Qgen_coset_left`, which is already in this file and unconditional. No
coset lemma for `Qgen'` is needed. -/

/-- **`Qgen` is `−1` when its first argument is `0`.** -/
theorem Qgen_zero_left (m L t : Nat) (hL : L < 2^m) (ht : t < 2^m) (hL0 : L ≠ 0) :
    Qgen L 0 t m = -1 := by
  have hm : ∃ m', m = m' + 1 := by
    cases m with
    | zero => exact absurd (by omega : L = 0) hL0
    | succ k => exact ⟨k, rfl⟩
  obtain ⟨m', rfl⟩ := hm
  have h := deg_right (m'+1) L t hL ht hL0
  unfold Qgen
  rw [Nat.zero_xor, cdSig0 t m', cdSig0 (t ^^^ L) m']
  simpa using h

/-- **`Qgen` is `−1` on the diagonal** — where `Qgen'` is `+1` (`Qgen'_diag`). That disagreement
    is the whole reason the boundary is `j+4`. -/
theorem Qgen_diag_neg (m L t : Nat) (hL : L < 2^m) (ht : t < 2^m) (hL0 : L ≠ 0) :
    Qgen L t t m = -1 := by
  have hm : ∃ m', m = m' + 1 := by
    cases m with
    | zero => exact absurd (by omega : L = 0) hL0
    | succ k => exact ⟨k, rfl⟩
  obtain ⟨m', rfl⟩ := hm
  have htL : t ^^^ L < 2^(m'+1) := Nat.xor_lt_two_pow ht hL
  by_cases ht0 : t = 0
  · subst ht0; exact Qgen_zero_left (m'+1) L 0 hL ht hL0
  by_cases htl : t = L
  · rw [Qgen_coset_left L t t, htl, Nat.xor_self]
    exact Qgen_zero_left (m'+1) L L hL hL hL0
  · have hxL : t ^^^ L ≠ 0 := by intro h; exact htl (xor_zero_eq t L h)
    have hne : t ≠ t ^^^ L := by
      intro h
      have h2 : t ^^^ t = t ^^^ (t ^^^ L) := by rw [← h]
      rw [Nat.xor_self, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor] at h2
      exact hL0 h2.symm
    have e1 : cdSigma t t (m'+1) = -1 := sigma_self (m'+1) t ht ht0
    have e2 : cdSigma (t ^^^ L) (t ^^^ L) (m'+1) = -1 := sigma_self (m'+1) (t ^^^ L) htL hxL
    have e3 : cdSigma t (t ^^^ L) (m'+1) = - cdSigma (t ^^^ L) t (m'+1) :=
      antisym (m'+1) t (t ^^^ L) ht htL ht0 hxL hne
    unfold Qgen
    rw [e1, e2, e3]
    rcases cdSigma_pm (m'+1) (t ^^^ L) t with h | h <;> rw [h] <;> decide

/-- One `Q'red_low_ul` step, landing on a first argument `Qgen_coset_left` sends to `0`. -/
theorem fam_ul (m W u b : Nat) (hW : W < 2^(m+1)) (hu : u < 2^(m+1)) (hb : b < 2^(m+1))
    (hb0 : b ≠ 0) (hW0 : W ≠ 0) (hpin : u = 0 ∨ u = W) :
    Qgen' W (u + 2^(m+1)) b (m+2) = -1 := by
  rw [Q'red_low_ul m W u b hW hu hb hb0]
  rcases hpin with h | h
  · rw [h]; exact Qgen_zero_left (m+1) W b hW hb hW0
  · rw [h, Qgen_coset_left W W b, Nat.xor_self]
    exact Qgen_zero_left (m+1) W b hW hb hW0

/-- One `Q'red_low_lu` step, landing on `Qgen W 0 a`. -/
theorem fam_lu_zero (m W a v : Nat) (hW : W < 2^(m+1)) (ha : a < 2^(m+1)) (hv : v < 2^(m+1))
    (ha0 : a ≠ 0) (haW : a ^^^ W ≠ 0) (hW0 : W ≠ 0) (hpin : v = 0 ∨ v = W) :
    Qgen' W a (v + 2^(m+1)) (m+2) = -1 := by
  rw [Q'red_low_lu m W a v hW ha hv ha0 haW]
  rcases hpin with h | h
  · rw [h]; exact Qgen_zero_left (m+1) W a hW ha hW0
  · rw [h, Qgen_coset_left W W a, Nat.xor_self]
    exact Qgen_zero_left (m+1) W a hW ha hW0

/-- One `Q'red_low_lu` step, landing on the diagonal. -/
theorem fam_lu_diag (m W a v : Nat) (hW : W < 2^(m+1)) (ha : a < 2^(m+1)) (hv : v < 2^(m+1))
    (ha0 : a ≠ 0) (haW : a ^^^ W ≠ 0) (hW0 : W ≠ 0) (hpin : v = a ∨ v = a ^^^ W) :
    Qgen' W a (v + 2^(m+1)) (m+2) = -1 := by
  rw [Q'red_low_lu m W a v hW ha hv ha0 haW]
  rcases hpin with h | h
  · rw [h]; exact Qgen_diag_neg (m+1) W a hW ha hW0
  · rw [h, Qgen_coset_left W (a ^^^ W) a, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
    exact Qgen_diag_neg (m+1) W a hW ha hW0

/-- The value each family needs: same residue, nonzero, and distinct from the label. -/
theorem pick_low (j Y Y0 x0 : Nat) (hY : Y < 2^(j+3)) (hYmod : Y % 2^(j+2) = Y0)
    (hY00 : Y0 ≠ 0) (hx0 : x0 < 2^(j+2)) :
    ∃ x, x < 2^(j+3) ∧ x % 2^(j+2) = x0 ∧ x ≠ 0 ∧ x ≠ Y := by
  have hpow : (2:Nat)^(j+3) = 2^(j+2) * 2 := by rw [Nat.pow_succ]
  have hp : 0 < 2^(j+2) := Nat.two_pow_pos (j+2)
  by_cases h : x0 + 2^(j+2) = Y
  · refine ⟨x0, by omega, Nat.mod_eq_of_lt hx0, ?_, by omega⟩
    intro hx
    rw [hx, Nat.zero_add] at h
    rw [← h, Nat.mod_self] at hYmod
    exact hY00 hYmod.symm
  · refine ⟨x0 + 2^(j+2), by omega, ?_, by omega, h⟩
    rw [Nat.add_mod_right]; exact Nat.mod_eq_of_lt hx0

/-- **ATTAINMENT ON THE SIX LINES.** Every bottom pair on any of the six degeneracy lines is
    carried by an explicit level-`(j+4)` witness, for **any** label `Y` below `2^{j+3}` with the
    right residue — in particular for whichever of `Y₀`, `Y₀ + 2^{j+2}` has even weight. -/
theorem attain_lines (j Y Y0 a0 b0 : Nat) (hY3 : Y < 2^(j+3)) (hYmod : Y % 2^(j+2) = Y0)
    (hY00 : Y0 ≠ 0) (ha0b : a0 < 2^(j+2)) (hb0b : b0 < 2^(j+2))
    (hline : a0 = 0 ∨ a0 = Y0 ∨ b0 = 0 ∨ b0 = Y0 ∨ a0 = b0 ∨ a0 ^^^ b0 = Y0) :
    ∃ a b, a < 2^(j+4) ∧ b < 2^(j+4) ∧ a % 2^(j+2) = a0 ∧ b % 2^(j+2) = b0 ∧
      a ≠ 0 ∧ b ≠ 0 ∧ b ≠ Y ∧ Qgen' Y a b (j+4) = -1 := by
  have hp : 0 < 2^(j+2) := Nat.two_pow_pos (j+2)
  have e3 : (2:Nat)^(j+3) = 2^(j+2) * 2 := by rw [Nat.pow_succ]
  have e4 : (2:Nat)^(j+4) = 2^(j+3) * 2 := by rw [Nat.pow_succ]
  have hY0' : Y ≠ 0 := by
    intro h; rw [h, Nat.zero_mod] at hYmod; exact hY00 hYmod.symm
  have hp3 : (0:Nat) < 2^(j+3) := Nat.two_pow_pos (j+3)
  have hzmod : ((0:Nat) + 2^(j+3)) % 2^(j+2) = 0 := by
    rw [add_pow_mod (j+2) (j+3) 0 (by omega)]
    exact Nat.zero_mod _
  have hYmod3 : (Y + 2^(j+3)) % 2^(j+2) = Y0 := by
    rw [add_pow_mod (j+2) (j+3) Y (by omega)]; exact hYmod
  obtain ⟨A, hA3, hAmod, hA0, hAY⟩ := pick_low j Y Y0 a0 hY3 hYmod hY00 ha0b
  obtain ⟨B, hB3, hBmod, hB0, hBY⟩ := pick_low j Y Y0 b0 hY3 hYmod hY00 hb0b
  have hAW : A ^^^ Y ≠ 0 := fun h => hAY (xor_zero_eq A Y h)
  rcases hline with h | h | h | h | h | h
  · exact ⟨0 + 2^(j+3), B, by omega, by omega, by rw [hzmod, h], hBmod, by omega, hB0, hBY,
      fam_ul (j+2) Y 0 B hY3 hp3 hB3 hB0 hY0' (Or.inl rfl)⟩
  · exact ⟨Y + 2^(j+3), B, by omega, by omega, by rw [hYmod3, h], hBmod, by omega, hB0, hBY,
      fam_ul (j+2) Y Y B hY3 hY3 hB3 hB0 hY0' (Or.inr rfl)⟩
  · exact ⟨A, 0 + 2^(j+3), by omega, by omega, hAmod, by rw [hzmod, h], hA0, by omega, by omega,
      fam_lu_zero (j+2) Y A 0 hY3 hA3 hp3 hA0 hAW hY0' (Or.inl rfl)⟩
  · exact ⟨A, Y + 2^(j+3), by omega, by omega, hAmod, by rw [hYmod3, h], hA0, by omega, by omega,
      fam_lu_zero (j+2) Y A Y hY3 hA3 hY3 hA0 hAW hY0' (Or.inr rfl)⟩
  · refine ⟨A, A + 2^(j+3), by omega, by omega, hAmod, ?_, hA0, by omega, by omega,
      fam_lu_diag (j+2) Y A A hY3 hA3 hA3 hA0 hAW hY0' (Or.inl rfl)⟩
    rw [add_pow_mod (j+2) (j+3) A (by omega), hAmod, h]
  · have hAYlt : A ^^^ Y < 2^(j+3) := Nat.xor_lt_two_pow hA3 hY3
    refine ⟨A, (A ^^^ Y) + 2^(j+3), by omega, by omega, hAmod, ?_, hA0, by omega, by omega,
      fam_lu_diag (j+2) Y A (A ^^^ Y) hY3 hA3 hAYlt hA0 hAW hY0' (Or.inr rfl)⟩
    rw [add_pow_mod (j+2) (j+3) (A ^^^ Y) (by omega), xor_mod_two_pow, hAmod, hYmod, ← h,
        ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]

/-- **`REACH` contains every degeneracy line at level `j+4`** — the half `attain_nondeg` does not
    cover. Together they give `DEG ∪ ND ⊆ REACH_{j+4}`. -/
theorem ReachD_lines (j Y Y0 a0 b0 : Nat) (hY3 : Y < 2^(j+3)) (hYmod : Y % 2^(j+2) = Y0)
    (hY00 : Y0 ≠ 0) (ha0b : a0 < 2^(j+2)) (hb0b : b0 < 2^(j+2))
    (hline : a0 = 0 ∨ a0 = Y0 ∨ b0 = 0 ∨ b0 = Y0 ∨ a0 = b0 ∨ a0 ^^^ b0 = Y0) :
    ReachD j Y0 (j+4) a0 b0 := by
  obtain ⟨a, b, ha4, hb4, hamod, hbmod, ha0, hb0, hbY, hq⟩ :=
    attain_lines j Y Y0 a0 b0 hY3 hYmod hY00 ha0b hb0b hline
  have e4 : (2:Nat)^(j+4) = 2^(j+3) * 2 := by rw [Nat.pow_succ]
  exact ⟨Y, a, b, by omega, ha4, hb4, hYmod, hamod, hbmod, ha0, hb0, hbY, hq⟩

/-! ## The two halves assembled: `REACH` is attained at `n = j+4`

`ReachD_mono` gives `REACH_{j+4} ⊆ REACH_n`. This is the other inclusion, and it is exactly the
dichotomy: a bottom pair is either **on** one of the six lines — then `ReachD_lines` builds a
level-`(j+4)` witness outright — or it is **off** them, which is precisely `NDeg`, and then
`collapse`/`attain_nondeg` carry the level-`n` witness down to level `j+3`, whence `ReachD_succ`.

Together with `corner_blocked_at_j3`, which says level `j+3` does **not** suffice, that is the
`n = j+4` boundary, proven and sharp. -/

/-- **`REACH` IS ATTAINED AT `n = j+4`.** Every level-`n` witness, `n ≥ j+4`, has its bottom pair
    already carried at level `j+4`. -/
theorem ReachD_attained (j Y0 a0 b0 n : Nat) (hn : j+4 ≤ n) (hY00 : Y0 ≠ 0)
    (hY0b : Y0 < 2^(j+2)) (ha0b : a0 < 2^(j+2)) (hb0b : b0 < 2^(j+2))
    (h : ReachD j Y0 n a0 b0) : ReachD j Y0 (j+4) a0 b0 := by
  have hp : (0:Nat) < 2^(j+2) := Nat.two_pow_pos (j+2)
  have e3 : (2:Nat)^(j+3) = 2^(j+2) * 2 := by rw [Nat.pow_succ]
  by_cases hline : a0 = 0 ∨ a0 = Y0 ∨ b0 = 0 ∨ b0 = Y0 ∨ a0 = b0 ∨ a0 ^^^ b0 = Y0
  · exact ReachD_lines j Y0 Y0 a0 b0 (by omega) (Nat.mod_eq_of_lt hY0b) hY00 ha0b hb0b hline
  · -- off the six lines is exactly `NDeg`
    have f1 : a0 ≠ 0 := fun e => hline (Or.inl e)
    have f2 : a0 ≠ Y0 := fun e => hline (Or.inr (Or.inl e))
    have f3 : b0 ≠ 0 := fun e => hline (Or.inr (Or.inr (Or.inl e)))
    have f4 : b0 ≠ Y0 := fun e => hline (Or.inr (Or.inr (Or.inr (Or.inl e))))
    have f5 : a0 ≠ b0 := fun e => hline (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl e)))))
    have f6 : a0 ^^^ b0 ≠ Y0 :=
      fun e => hline (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr e)))))
    obtain ⟨Y, a, b, hY, ha, hb, eY, ea, eb, ha0, hb0, hbY, hq⟩ := h
    have hnd : NDeg j Y a b := by
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;> simp only [ea, eb, eY] <;> assumption
    obtain ⟨Y', hY'3, hY'mod, hval⟩ :=
      attain_nondeg j n Y a b (by omega) hY ha hb hnd
    rw [hq] at hval
    -- the level-(j+3) witness, then one step up
    have hbY' : b0 ≠ Y' := by
      intro e
      apply f4
      have h1 : b0 % 2^(j+2) = Y0 := by rw [e, hY'mod, eY]
      rw [Nat.mod_eq_of_lt hb0b] at h1
      exact h1
    refine ReachD_succ j Y0 (j+2) a0 b0 ⟨Y', a0, b0, hY'3, by omega, by omega, ?_, ?_, ?_,
      f1, f3, hbY', ?_⟩
    · rw [hY'mod]; exact eY
    · exact Nat.mod_eq_of_lt ha0b
    · exact Nat.mod_eq_of_lt hb0b
    · rw [ea, eb] at hval; exact hval

/-- **`REACH` STABILISES AT `n = j+4`.** The two inclusions: `ReachD_attained` down,
    `ReachD_mono` up. With `corner_blocked_at_j3` — which says `j+3` does *not* suffice — the
    boundary is proven and proven sharp. -/
theorem ReachD_stable (j Y0 a0 b0 n : Nat) (hn : j+4 ≤ n) (hY00 : Y0 ≠ 0)
    (hY0b : Y0 < 2^(j+2)) (ha0b : a0 < 2^(j+2)) (hb0b : b0 < 2^(j+2)) :
    ReachD j Y0 n a0 b0 ↔ ReachD j Y0 (j+4) a0 b0 := by
  constructor
  · exact ReachD_attained j Y0 a0 b0 n hn hY00 hY0b ha0b hb0b
  · intro h
    obtain ⟨d, rfl⟩ : ∃ d, n = j + 4 + d := ⟨n - (j+4), by omega⟩
    exact ReachD_mono j Y0 a0 b0 d (j+3) h

/-! ## `REACH ⊆ {D = +1}` — the last measured link of L2

`REACH = DEG ∪ ND` is now explicit, so (♦) splits in two:

* **on `DEG`** (the six lines) — an *identity*, no hypothesis;
* **on `ND`** — off the lines, `Qgen Y₀ a₀ b₀ = −ε` ⟹ `D = +1`.

For the label class `Y₀ = 2^j` the second half is **vacuous**: `Qgen_pow2` (already in this file)
says `Qgen (2^j) a b m = −1` always, while `ND`'s condition there is `Qgen = +1`. So for that
class (♦) *is* the identity on the lines.

And the identity on the lines collapses to **one** lemma. Writing `D(a,b) =
gdisc j a b · gdisc j (a⊕Y₀) (b⊕Y₀) · psg(a % 2^j) · psg(b % 2^j)`:

| line | reduces to |
|---|---|
| `a₀ = 0`, `a₀ = Y₀` | `gdisc j Y₀ x = psg (x % 2^j)` |
| `b₀ = 0`, `b₀ = Y₀` | the same, by `gdisc_symm` |
| `a₀ = b₀`           | `gdisc j t t = 1` |
| `a₀ ⊕ b₀ = Y₀`      | `gdisc j a b` squared, by `gdisc_symm` |

so everything rests on `gdisc j L x m = psg (x % 2^j)` for `lsb L = j`. This section proves the
pieces, bottom up. -/

/-- `(−1)^{popcount x}`. -/
def psg : Nat → Int
  | 0 => 1
  | (n+1) => (if (n+1) % 2 = 1 then (-1 : Int) else 1) * psg ((n+1) / 2)
decreasing_by exact Nat.div_lt_self (by omega) (by omega)

theorem psg_zero : psg 0 = 1 := by rw [psg]

theorem psg_one : psg 1 = -1 := by
  rw [psg]
  simp [psg_zero]

/-- Peeling the top bit flips the sign. -/
theorem psg_top : ∀ (k u : Nat), u < 2^k → psg (u + 2^k) = - psg u := by
  intro k
  induction k with
  | zero =>
      intro u hu
      have : u = 0 := by simpa using hu
      subst this
      simp [psg_one, psg_zero]
  | succ k ih =>
      intro u hu
      have hpow : (2:Nat)^(k+1) = 2^k * 2 := by rw [Nat.pow_succ]
      have hpos : 0 < 2^k := Nat.two_pow_pos k
      have h1 : (u + 2^(k+1)) % 2 = u % 2 := by omega
      have h2 : (u + 2^(k+1)) / 2 = u / 2 + 2^k := by omega
      have hne : u + 2^(k+1) ≠ 0 := by omega
      obtain ⟨t, ht⟩ : ∃ t, u + 2^(k+1) = t + 1 := ⟨u + 2^(k+1) - 1, by omega⟩
      rw [ht, psg, ← ht, h1, h2, ih (u/2) (by omega)]
      cases u with
      | zero => simp [psg_zero]
      | succ v =>
          rw [psg, Int.mul_neg]

/-- **`σ(w,1) = (−1)^{popcount w}`.** A three-line induction: `R_ll` keeps the value, `R_ul`
    negates it, and that is exactly what adding a bit does to the popcount. -/
theorem sigma_one : ∀ (m w : Nat), w < 2^(m+1) → cdSigma w 1 (m+1) = psg w := by
  intro m
  induction m with
  | zero =>
      intro w hw
      have hp : (2:Nat)^(0+1) = 2 := rfl
      have hw2 : w = 0 ∨ w = 1 := by omega
      rcases hw2 with rfl | rfl
      · rw [cdSig0 1 0, psg_zero]
      · rw [sigma_self 1 1 (by decide) (by decide), psg_one]
  | succ m ih =>
      intro w hw
      have h1 : (1:Nat) < 2^(m+1) := by
        have := Nat.two_pow_pos m; have h := Nat.one_lt_two_pow_iff (n := m+1); omega
      rcases split_top hw with hl | ⟨u, hu, rfl⟩
      · rw [R_ll w 1 m hl h1]; exact ih w hl
      · rw [R_ul u 1 m hu h1, if_neg (by omega), ih u hu, psg_top (m+1) u hu]

theorem psg_pm : ∀ (x : Nat), psg x = 1 ∨ psg x = -1
  | 0 => Or.inl psg_zero
  | (n+1) => by
      rw [psg]
      rcases psg_pm ((n+1)/2) with h | h <;> rw [h] <;>
        by_cases hb : (n+1) % 2 = 1
      · rw [if_pos hb]; exact Or.inr (by decide)
      · rw [if_neg hb]; exact Or.inl (by decide)
      · rw [if_pos hb]; exact Or.inl (by decide)
      · rw [if_neg hb]; exact Or.inr (by decide)
  decreasing_by exact Nat.div_lt_self (by omega) (by omega)

theorem psg_sq (x : Nat) : psg x * psg x = 1 := by
  rcases psg_pm x with h | h <;> rw [h] <;> decide

/-- `gdisc` is `1` when either argument is `0`: `τ0 = 0` and `σ(0,·) = 1`. -/
theorem gdisc_zero_left (j m y : Nat) : gdisc j 0 y (m+1) = 1 := by
  unfold gdisc
  rw [tau_zero j, cdSig0 (tau j y) m, cdSig0 y m]
  decide

/-- `gdisc` is `1` on the diagonal: both factors are self-pairings, and `τt = 0 ↔ t = 0`. -/
theorem gdisc_diag_one (j m t : Nat) (hj : j < m+1) (ht : t < 2^(m+1)) :
    gdisc j t t (m+1) = 1 := by
  unfold gdisc
  by_cases h0 : t = 0
  · subst h0
    rw [tau_zero j, cdSig0 0 m]
    decide
  · have htau : tau j t ≠ 0 := by
      intro h
      exact h0 (tau_inj j t 0 (by rw [h, tau_zero]))
    rw [sigma_self (m+1) t ht h0,
        sigma_self (m+1) (tau j t) (tau_lt j (m+1) t hj ht) htau]
    decide

/-- `Y₀`'s bits below `j` vanish, so `⊕ Y₀` does not move `psg (· % 2^j)`. -/
theorem psg_xor_lsb (j Y0 x : Nat) (hY0 : Y0 % 2^j = 0) :
    psg ((x ^^^ Y0) % 2^j) = psg (x % 2^j) := by
  rw [xor_mod_two_pow, hY0, Nat.xor_zero]

/-- **The `∀m` half of the line identity is free**: `gdisc_trunc` already collapses every level to
    `j+1`, and the `lsb` hypothesis pins the truncated label to `2^j`. Only the level-`(j+1)`
    base is left, and it is taken here as an explicit hypothesis. -/
theorem gdisc_lsb_of_base
    (hbase : ∀ (j y : Nat), y < 2^(j+1) → gdisc j (2^j) y (j+1) = psg (y % 2^j))
    (j d L x : Nat) (hL : L < 2^((j+1)+d)) (hx : x < 2^((j+1)+d))
    (hlsb : L % 2^(j+1) = 2^j) :
    gdisc j L x ((j+1)+d) = psg (x % 2^j) := by
  rw [gdisc_trunc j j (by omega) d L x hL hx, hlsb]
  rw [hbase j (x % 2^(j+1)) (Nat.mod_lt _ (Nat.two_pow_pos (j+1)))]
  rw [mod_pow_mod j (j+1) x (by omega)]

theorem psg_step (z : Nat) : psg z = (if z % 2 = 1 then (-1 : Int) else 1) * psg (z / 2) := by
  cases z with
  | zero => rw [psg_zero]; decide
  | succ n => rw [psg]

theorem xor_one_div (w : Nat) : (w ^^^ 1) / 2 = w / 2 := by
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.testBit_div_two, Nat.testBit_div_two, Nat.testBit_xor,
      Nat.testBit_lt_two_pow (show (1:Nat) < 2^(i+1) by
        have := Nat.one_lt_two_pow_iff (n := i+1); omega)]
  cases w.testBit (i+1) <;> decide

theorem xor_one_mod (w : Nat) : ((w ^^^ 1) % 2 = 1) ↔ ¬ (w % 2 = 1) := by
  have h := Nat.testBit_zero (w ^^^ 1)
  have h2 := Nat.testBit_zero w
  rw [Nat.testBit_xor] at h
  have h1 : (1:Nat).testBit 0 = true := by decide
  rw [h1, h2] at h
  by_cases hw : w % 2 = 1 <;> simp [hw] at h ⊢ <;> simp [h]

/-- Flipping bit `0` flips the popcount parity. -/
theorem psg_xor_one (w : Nat) : psg (w ^^^ 1) = - psg w := by
  rw [psg_step (w ^^^ 1), psg_step w, xor_one_div]
  by_cases hw : w % 2 = 1
  · rw [if_pos hw, if_neg (fun h => ((xor_one_mod w).mp h) hw)]
    simp
  · rw [if_neg hw, if_pos ((xor_one_mod w).mpr hw)]
    simp

/-- The mask `1 ||| 2^j` is `1 ⊕ 2^j` — the two bits are disjoint when `j ≠ 0`. -/
theorem mask_xor (j : Nat) (hj : j ≠ 0) : ((1:Nat) ||| (1 <<< j)) = 1 ^^^ (1 <<< j) := by
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.testBit_or, Nat.testBit_xor, Nat.shiftLeft_eq, Nat.one_mul, Nat.testBit_two_pow]
  have h1 : (1:Nat) = 2^0 := rfl
  rw [h1, Nat.testBit_two_pow]
  by_cases h0 : 0 = i <;> by_cases hji : j = i <;> simp_all

/-- **`τ` sends the top bit `2^j` to `1`.** -/
theorem tau_pow_self (j : Nat) (hj : j ≠ 0) : tau j (2^j) = 1 := by
  have h0 : ((2:Nat)^j).testBit 0 = false := by
    rw [Nat.testBit_two_pow]; simp [hj]
  have hjj : ((2:Nat)^j).testBit j = true := by rw [Nat.testBit_two_pow]; simp
  rw [tau_spec, h0, hjj, if_neg (by decide : ¬ ((false : Bool) = true)),
      mask_xor j hj, Nat.shiftLeft_eq, Nat.one_mul, ← Nat.xor_assoc,
      Nat.xor_comm ((2:Nat)^j) 1, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

/-- Below `2^j`, `τ` fixes the even numbers. -/
theorem tau_low_even (j w : Nat) (hw : w < 2^j) (hp : w % 2 = 0) : tau j w = w := by
  have hbj : w.testBit j = false := Nat.testBit_lt_two_pow hw
  have hb0 : w.testBit 0 = false := by rw [Nat.testBit_zero]; simp [hp]
  rw [tau_spec, hb0, hbj, if_pos rfl]

/-- Below `2^j`, `τ` sends an odd `w` to `(w ⊕ 1) + 2^j`. -/
theorem tau_low_odd (j w : Nat) (hj : j ≠ 0) (hw : w < 2^j) (hp : w % 2 = 1) :
    tau j w = (w ^^^ 1) + 2^j := by
  have hbj : w.testBit j = false := Nat.testBit_lt_two_pow hw
  have hb0 : w.testBit 0 = true := by rw [Nat.testBit_zero]; simp [hp]
  have h1 : (1:Nat) < 2^j := by
    have := Nat.two_pow_pos j
    have h := Nat.one_lt_two_pow_iff (n := j); omega
  have hxlt : w ^^^ 1 < 2^j := Nat.xor_lt_two_pow hw h1
  obtain ⟨i, rfl⟩ : ∃ i, j = i + 1 := ⟨j - 1, by omega⟩
  rw [tau_spec, hb0, hbj, if_neg (by decide : ¬ ((true : Bool) = false)),
      mask_xor (i+1) hj, Nat.shiftLeft_eq, Nat.one_mul, ← Nat.xor_assoc,
      seam_add_xor (w ^^^ 1) i hxlt]

/-- **THE BASE CASE**, and with it the whole six-line identity. -/
theorem gdisc_base (j y : Nat) (hy : y < 2^(j+1)) : gdisc j (2^j) y (j+1) = psg (y % 2^j) := by
  by_cases hj : j = 0
  · subst hj
    unfold gdisc
    rw [tau_id_zero, tau_id_zero, cdSq, Nat.pow_zero, Nat.mod_one, psg_zero]
  · obtain ⟨i, rfl⟩ : ∃ i, j = i + 1 := ⟨j - 1, by omega⟩
    have h1 : (1:Nat) < 2^(i+1) := by
      have := Nat.two_pow_pos i
      have h := Nat.one_lt_two_pow_iff (n := i+1); omega
    have htp : tau (i+1) (2^(i+1)) = 1 := tau_pow_self (i+1) hj
    show gdisc (i+1) (2^(i+1)) y (i+2) = psg (y % 2^(i+1))
    rcases split_top hy with hlow | ⟨w, hw, rfl⟩
    · -- y low
      have hR2 := R_ul 0 y i (Nat.two_pow_pos (i+1)) hlow
      rw [Nat.zero_add] at hR2
      by_cases hp : y % 2 = 0
      · unfold gdisc
        rw [htp, tau_low_even (i+1) y hlow hp, R_ll 1 y i h1 hlow, hR2,
            Nat.mod_eq_of_lt hlow]
        by_cases hy0 : y = 0
        · subst hy0
          rw [cdSig0' 1 i, if_pos rfl, psg_zero]
          decide
        · have hy1 : y ≠ 1 := by intro h; rw [h] at hp; exact absurd hp (by decide)
          rw [if_neg hy0, cdSig0 y i,
              antisym (i+1) 1 y h1 hlow (by decide) hy0 (fun h => hy1 h.symm),
              sigma_one i y hlow]
          rcases psg_pm y with h | h <;> rw [h] <;> decide
      · have hy0 : y ≠ 0 := by intro h; rw [h] at hp; exact hp (by decide)
        have hxlt : y ^^^ 1 < 2^(i+1) := Nat.xor_lt_two_pow hlow h1
        unfold gdisc
        rw [htp, tau_low_odd (i+1) y hj hlow (by omega),
            R_lu 1 (y ^^^ 1) i h1 hxlt, hR2, if_neg hy0, cdSig0 y i,
            sigma_one i (y ^^^ 1) hxlt, psg_xor_one y, Nat.mod_eq_of_lt hlow]
        rcases psg_pm y with h | h <;> rw [h] <;> decide
    · -- y = w + 2^(i+1)
      have hR2 := R_uu 0 w i (Nat.two_pow_pos (i+1)) hw
      rw [Nat.zero_add] at hR2
      have hmod : (w + 2^(i+1)) % 2^(i+1) = w := by
        rw [Nat.add_mod_right]; exact Nat.mod_eq_of_lt hw
      have htauy : tau (i+1) (w + 2^(i+1)) = tau (i+1) w ^^^ 1 := by
        rw [seam_add_xor w i hw, tau_xor, htp]
      by_cases hp : w % 2 = 0
      · have hxlt : w ^^^ 1 < 2^(i+1) := Nat.xor_lt_two_pow hw h1
        unfold gdisc
        rw [htp, htauy, tau_low_even (i+1) w hw hp, R_ll 1 (w ^^^ 1) i h1 hxlt, hR2, hmod]
        by_cases hw0 : w = 0
        · subst hw0
          rw [if_pos rfl, show (0:Nat) ^^^ 1 = 1 from by decide,
              sigma_self (i+1) 1 h1 (by decide), psg_zero]
          decide
        · have hx0 : w ^^^ 1 ≠ 0 := by
            intro h
            have h2 := xor_zero_eq w 1 h
            rw [h2] at hp; exact absurd hp (by decide)
          have hx1 : w ^^^ 1 ≠ 1 := by
            intro h
            apply hw0
            have h2 : (w ^^^ 1) ^^^ 1 = 1 ^^^ 1 := by rw [h]
            rwa [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero] at h2
          rw [if_neg hw0, cdSig0' w i,
              antisym (i+1) 1 (w ^^^ 1) h1 hxlt (by decide) hx0 (fun h => hx1 h.symm),
              sigma_one i (w ^^^ 1) hxlt, psg_xor_one w]
          rcases psg_pm w with h | h <;> rw [h] <;> decide
      · have hw0 : w ≠ 0 := by intro h; rw [h] at hp; exact hp (by decide)
        have hxlt : w ^^^ 1 < 2^(i+1) := Nat.xor_lt_two_pow hw h1
        have hfix : tau (i+1) w ^^^ 1 = w + 2^(i+1) := by
          rw [tau_low_odd (i+1) w hj hw (by omega), seam_add_xor (w ^^^ 1) i hxlt,
              Nat.xor_assoc, Nat.xor_comm ((2:Nat)^(i+1)) 1, ← Nat.xor_assoc,
              Nat.xor_assoc w 1 1, Nat.xor_self, Nat.xor_zero, ← seam_add_xor w i hw]
        unfold gdisc
        rw [htp, htauy, hfix, R_lu 1 w i h1 hw, hR2, if_neg hw0, cdSig0' w i, hmod,
            sigma_one i w hw]
        rcases psg_pm w with h | h <;> rw [h] <;> decide

/-- The defect (♦) asks to be `+1`. -/
def Ddef (j Y0 a b m : Nat) : Int :=
  gdisc j a b m * gdisc j (a ^^^ Y0) (b ^^^ Y0) m * psg (a % 2^j) * psg (b % 2^j)

theorem gdisc_sq (j x y m : Nat) : gdisc j x y m * gdisc j x y m = 1 := by
  rcases gdisc_pm j x y m with h | h <;> rw [h] <;> decide

/-- **`D = +1` ON THE SIX LINES**, given the one base lemma. Every case reduces to
    `gdisc_lsb_of_base`, `gdisc_zero_left`, `gdisc_diag_one` and `gdisc_symm`. -/
theorem D_on_lines
    (hbase : ∀ (j y : Nat), y < 2^(j+1) → gdisc j (2^j) y (j+1) = psg (y % 2^j))
    (j Y0 a b : Nat) (hY0 : Y0 < 2^(j+2)) (hlsb : Y0 % 2^(j+1) = 2^j)
    (ha : a < 2^(j+2)) (hb : b < 2^(j+2))
    (hline : a = 0 ∨ a = Y0 ∨ b = 0 ∨ b = Y0 ∨ a = b ∨ a ^^^ b = Y0) :
    Ddef j Y0 a b (j+2) = 1 := by
  have hj : j < j+2 := by omega
  have hpz : (0:Nat) < 2^(j+2) := Nat.two_pow_pos (j+2)
  have hY0j : Y0 % 2^j = 0 := by
    rw [← mod_pow_mod j (j+1) Y0 (by omega), hlsb, Nat.mod_self]
  have hpsgY0 : psg (Y0 % 2^j) = 1 := by rw [hY0j]; exact psg_zero
  have hxY0 : ∀ x : Nat, psg ((x ^^^ Y0) % 2^j) = psg (x % 2^j) :=
    fun x => psg_xor_lsb j Y0 x hY0j
  have haY : a ^^^ Y0 < 2^(j+2) := Nat.xor_lt_two_pow ha hY0
  have hbY : b ^^^ Y0 < 2^(j+2) := Nat.xor_lt_two_pow hb hY0
  have key : ∀ x : Nat, x < 2^(j+2) → gdisc j Y0 x (j+2) = psg (x % 2^j) := by
    intro x hx
    have e : j + 2 = (j+1) + 1 := by omega
    rw [e] at hx hY0 ⊢
    exact gdisc_lsb_of_base hbase j 1 Y0 x hY0 hx hlsb
  unfold Ddef
  rcases hline with h | h | h | h | h | h
  · -- a = 0
    subst h
    rw [gdisc_zero_left j (j+1) b, Nat.zero_xor, key (b ^^^ Y0) hbY, hxY0 b,
        Nat.zero_mod, psg_zero, Int.one_mul, Int.mul_one]
    exact psg_sq (b % 2^j)
  · -- a = Y0
    rw [h, key b hb, Nat.xor_self, gdisc_zero_left j (j+1) (b ^^^ Y0), hpsgY0,
        Int.mul_one, Int.mul_one]
    exact psg_sq (b % 2^j)
  · -- b = 0
    subst h
    rw [gdisc_symm (j+2) j a 0 hj ha hpz, gdisc_zero_left j (j+1) a, Nat.zero_xor,
        gdisc_symm (j+2) j (a ^^^ Y0) Y0 hj haY hY0, key (a ^^^ Y0) haY, hxY0 a,
        Nat.zero_mod, psg_zero, Int.one_mul, Int.mul_one]
    exact psg_sq (a % 2^j)
  · -- b = Y0
    rw [h, gdisc_symm (j+2) j a Y0 hj ha hY0, key a ha, Nat.xor_self,
        gdisc_symm (j+2) j (a ^^^ Y0) 0 hj haY hpz,
        gdisc_zero_left j (j+1) (a ^^^ Y0), hpsgY0, Int.mul_one, Int.mul_one]
    exact psg_sq (a % 2^j)
  · -- a = b
    rw [h, gdisc_diag_one j (j+1) b hj hb, gdisc_diag_one j (j+1) (b ^^^ Y0) hj hbY,
        Int.one_mul, Int.one_mul]
    exact psg_sq (b % 2^j)
  · -- a ^^^ b = Y0, i.e. a ^^^ Y0 = b  (the form the rest of the lane uses)
    have h' : a ^^^ Y0 = b := by rw [← h, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]
    rw [← h', Nat.xor_assoc, Nat.xor_self, Nat.xor_zero,
        gdisc_symm (j+2) j (a ^^^ Y0) a hj haY ha, hxY0 a,
        gdisc_sq j a (a ^^^ Y0) (j+2), Int.one_mul]
    exact psg_sq (a % 2^j)

/-! ## The `ND` half, `Y₀ = 3·2^j`

`Q` at this label has a **complete closed form**, and it is the shape of the whole remaining
statement:

```
Qgen (3·2^j) a b (j+2) = −1   ⟺   a % 2^j = 0  ∨  b % 2^j = 0  ∨  a % 2^j = b % 2^j
```

— the `mod 2^j` shadow of the six lines. (`Q` at this label depends only on the low parts, which
is what the two-step descent through `Qred_hi_*` then `Q'red_hi_*` says: both steps carry the
label high, so the two signs cancel, and what is left is `Qgen`/`Qgen'` at label `0`, which is
`1`. The side conditions of those rows are exactly `a % 2^j, b % 2^j ≠ 0` and `≠` each other.)

On that set `D = +1`, and the reason is that `g` is *bilinear there*:
`g(x,y) = (−1)^{c(x)p_j(y) + c(y)p_j(x)}` with `c(x) = bit₀(x) ⊕ bit_j(x)`. Since `⊕Y₀` flips
`c` and fixes `p_j`, the exponents cancel in pairs. This section proves the `g` side. -/

/-- **`σ(x⊕1, x) = (−1)^{popcount x}`**, with `x = 1` the single exception (`σ(0,1) = 1` but
    `psg 1 = −1`). Same three-line shape as `sigma_one`. -/
theorem sigma_xor_one : ∀ (m x : Nat), x < 2^(m+1) → x ≠ 1 →
    cdSigma (x ^^^ 1) x (m+1) = psg x := by
  intro m
  induction m with
  | zero =>
      intro x hx hx1
      have hp : (2:Nat)^(0+1) = 2 := rfl
      have hx0 : x = 0 := by omega
      subst hx0
      rw [show (0:Nat) ^^^ 1 = 1 from by decide, cdSig0' 1 0, psg_zero]
  | succ m ih =>
      intro x hx hx1
      have h1 : (1:Nat) < 2^(m+1) := by
        have := Nat.two_pow_pos m
        have h := Nat.one_lt_two_pow_iff (n := m+1); omega
      rcases split_top hx with hl | ⟨v, hv, rfl⟩
      · rw [R_ll (x ^^^ 1) x m (Nat.xor_lt_two_pow hl h1) hl]
        exact ih x hl hx1
      · have hvl : v ^^^ 1 < 2^(m+1) := Nat.xor_lt_two_pow hv h1
        have hseam : (v + 2^(m+1)) ^^^ 1 = (v ^^^ 1) + 2^(m+1) := by
          rw [seam_add_xor v m hv, seam_add_xor (v ^^^ 1) m hvl,
              Nat.xor_assoc, Nat.xor_comm ((2:Nat)^(m+1)) 1, ← Nat.xor_assoc]
        rw [hseam, R_uu (v ^^^ 1) v m hvl hv, psg_top (m+1) v hv]
        by_cases hv0 : v = 0
        · subst hv0; rw [if_pos rfl, psg_zero]
        · rw [if_neg hv0]
          by_cases hv1 : v = 1
          · subst hv1
            rw [show (1:Nat) ^^^ 1 = 0 from by decide, cdSig0' 1 m, psg_one]
            decide
          · have hx0 : v ^^^ 1 ≠ 0 := by
              intro h; exact hv1 (xor_zero_eq v 1 h)
            have hne : v ≠ v ^^^ 1 := by
              intro h
              have h2 : v ^^^ v = v ^^^ (v ^^^ 1) := by rw [← h]
              rw [Nat.xor_self, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor] at h2
              exact absurd h2.symm (by decide)
            rw [antisym (m+1) v (v ^^^ 1) hv hvl hv0 hx0 hne, ih v hv hv1]

/-- **The second base case**: `g` across the seam at equal low parts. -/
theorem gdisc_seam_diag (j t : Nat) (hj : j ≠ 0) (ht : t < 2^j) :
    gdisc j (t + 2^j) t (j+1) = psg t := by
  obtain ⟨i, rfl⟩ : ∃ i, j = i + 1 := ⟨j - 1, by omega⟩
  have h1 : (1:Nat) < 2^(i+1) := by
    have := Nat.two_pow_pos i
    have h := Nat.one_lt_two_pow_iff (n := i+1); omega
  have htp : tau (i+1) (2^(i+1)) = 1 := tau_pow_self (i+1) hj
  have hR2 : cdSigma (t + 2^(i+1)) t (i+2) = 1 := by
    rw [R_ul t t i ht ht]
    by_cases ht0 : t = 0
    · rw [if_pos ht0]
    · rw [if_neg ht0, sigma_self (i+1) t ht ht0]; decide
  have htauy : tau (i+1) (t + 2^(i+1)) = tau (i+1) t ^^^ 1 := by
    rw [seam_add_xor t i ht, tau_xor, htp]
  show gdisc (i+1) (t + 2^(i+1)) t (i+2) = psg t
  unfold gdisc
  rw [htauy, hR2]
  by_cases hp : t % 2 = 0
  · have ht1 : t ≠ 1 := by intro h; rw [h] at hp; exact absurd hp (by decide)
    have hxl : t ^^^ 1 < 2^(i+1) := Nat.xor_lt_two_pow ht h1
    rw [tau_low_even (i+1) t ht hp, R_ll (t ^^^ 1) t i hxl ht,
        sigma_xor_one i t ht ht1, Int.mul_one]
  · have hxl : t ^^^ 1 < 2^(i+1) := Nat.xor_lt_two_pow ht h1
    have htl : tau (i+1) t = (t ^^^ 1) + 2^(i+1) := tau_low_odd (i+1) t hj ht (by omega)
    have hfix : ((t ^^^ 1) + 2^(i+1)) ^^^ 1 = t + 2^(i+1) := by
      rw [seam_add_xor (t ^^^ 1) i hxl, Nat.xor_assoc, Nat.xor_comm ((2:Nat)^(i+1)) 1,
          ← Nat.xor_assoc, Nat.xor_assoc t 1 1, Nat.xor_self, Nat.xor_zero,
          ← seam_add_xor t i ht]
    rw [htl, hfix, R_uu t (t ^^^ 1) i ht hxl]
    by_cases ht1 : t = 1
    · subst ht1
      rw [show (1:Nat) ^^^ 1 = 0 from by decide, if_pos rfl, psg_one, Int.mul_one]
    · have hx0 : t ^^^ 1 ≠ 0 := by intro h; exact ht1 (xor_zero_eq t 1 h)
      rw [if_neg hx0, sigma_xor_one i t ht ht1, Int.mul_one]

/-- A label whose bottom `j+1` bits all vanish makes `gdisc` trivial. -/
theorem gdisc_block_zero (j d u x : Nat) (hu : u < 2^((j+1)+d)) (hx : x < 2^((j+1)+d))
    (h0 : u % 2^(j+1) = 0) : gdisc j u x ((j+1)+d) = 1 := by
  rw [gdisc_trunc j j (by omega) d u x hu hx, h0]
  exact gdisc_zero_left j j (x % 2^(j+1))

/-! ## (♦) for the label class `Y₀ = 2^j`

Everything above assembles. The base case is discharged, so `gdisc_lsb` and `D_on_lines` hold
outright; and for `Y₀ = 2^j` the `ND` half is *empty*, because `collapse` turns the hypothesis
into `dsgnN · Qgen (2^j) a₀ b₀`, and `Qgen_pow2` makes the second factor `−1` unconditionally.
So a non-degenerate bottom can never satisfy (♦)'s hypothesis there, every witness sits on one
of the six lines, and `D_on_lines` finishes.

The even-weight condition enters exactly once, as `dsgnN j n Y = −1` — which is what `N12` says
even weight is, for this label class. -/

theorem gdisc_lsb (j d L x : Nat) (hL : L < 2^((j+1)+d)) (hx : x < 2^((j+1)+d))
    (hlsb : L % 2^(j+1) = 2^j) : gdisc j L x ((j+1)+d) = psg (x % 2^j) :=
  gdisc_lsb_of_base gdisc_base j d L x hL hx hlsb

/-- **`D = +1` on the six lines, unconditionally.** -/
theorem D_lines (j Y0 a b : Nat) (hY0 : Y0 < 2^(j+2)) (hlsb : Y0 % 2^(j+1) = 2^j)
    (ha : a < 2^(j+2)) (hb : b < 2^(j+2))
    (hline : a = 0 ∨ a = Y0 ∨ b = 0 ∨ b = Y0 ∨ a = b ∨ a ^^^ b = Y0) :
    Ddef j Y0 a b (j+2) = 1 :=
  D_on_lines gdisc_base j Y0 a b hY0 hlsb ha hb hline

/-- For the label class `Y₀ = 2^j` there is **no non-degenerate witness**: `collapse` plus
    `Qgen_pow2` force the value to `+1`. -/
theorem no_nondeg_witness_pow2 (j n Y a b : Nat) (hn : j+2 ≤ n)
    (hY : Y < 2^n) (ha : a < 2^n) (hb : b < 2^n) (hnd : NDeg j Y a b)
    (hY0 : Y % 2^(j+2) = 2^j) (hs : dsgnN j n Y = -1) :
    Qgen' Y a b n = 1 := by
  have hp : (0:Nat) < 2^(j+2) := Nat.two_pow_pos (j+2)
  rw [collapse j n Y a b hn hY ha hb hnd, hY0, hs,
      Qgen_pow2 (j+2) j (a % 2^(j+2)) (b % 2^(j+2)) (by omega)
        (Nat.mod_lt _ hp) (Nat.mod_lt _ hp)]
  decide

/-- **(♦) HOLDS FOR THE LABEL CLASS `Y₀ = 2^j`.** Every level-`n` witness has `D = +1` at the
    bottom — and by `G_trunc` the bottom is where `D` lives. -/
theorem diamond_pow2 (j n Y a b : Nat) (hn : j+2 ≤ n)
    (hY : Y < 2^n) (ha : a < 2^n) (hb : b < 2^n)
    (hY0 : Y % 2^(j+2) = 2^j) (hs : dsgnN j n Y = -1) (hw : Qgen' Y a b n = -1) :
    Ddef j (2^j) (a % 2^(j+2)) (b % 2^(j+2)) (j+2) = 1 := by
  have hp : (0:Nat) < 2^(j+2) := Nat.two_pow_pos (j+2)
  have hpj : (0:Nat) < 2^j := Nat.two_pow_pos j
  have hlt : (2:Nat)^j < 2^(j+2) := Nat.pow_lt_pow_right (by omega) (by omega)
  have hlsb : (2:Nat)^j % 2^(j+1) = 2^j :=
    Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by omega) (by omega))
  by_cases hline : a % 2^(j+2) = 0 ∨ a % 2^(j+2) = 2^j ∨ b % 2^(j+2) = 0 ∨
      b % 2^(j+2) = 2^j ∨ a % 2^(j+2) = b % 2^(j+2) ∨
      (a % 2^(j+2)) ^^^ (b % 2^(j+2)) = 2^j
  · exact D_lines j (2^j) (a % 2^(j+2)) (b % 2^(j+2)) hlt hlsb
      (Nat.mod_lt _ hp) (Nat.mod_lt _ hp) hline
  · exfalso
    have f1 : a % 2^(j+2) ≠ 0 := fun e => hline (Or.inl e)
    have f2 : a % 2^(j+2) ≠ 2^j := fun e => hline (Or.inr (Or.inl e))
    have f3 : b % 2^(j+2) ≠ 0 := fun e => hline (Or.inr (Or.inr (Or.inl e)))
    have f4 : b % 2^(j+2) ≠ 2^j := fun e => hline (Or.inr (Or.inr (Or.inr (Or.inl e))))
    have f5 : a % 2^(j+2) ≠ b % 2^(j+2) :=
      fun e => hline (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl e)))))
    have f6 : (a % 2^(j+2)) ^^^ (b % 2^(j+2)) ≠ 2^j :=
      fun e => hline (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr e)))))
    have hnd : NDeg j Y a b := by
      refine ⟨f1, f3, ?_, ?_, f5, ?_⟩ <;> rw [hY0] <;> assumption
    have := no_nondeg_witness_pow2 j n Y a b hn hY ha hb hnd hY0 hs
    rw [this] at hw
    exact absurd hw (by decide)

theorem xor_pow_low (j t : Nat) (hj : j ≠ 0) (ht : t < 2^j) : t ^^^ 2^j = t + 2^j := by
  obtain ⟨i, rfl⟩ : ∃ i, j = i + 1 := ⟨j - 1, by omega⟩
  exact (seam_add_xor t i ht).symm

theorem low_split (k r : Nat) (hr : r < 2^(k+1)) : r = r % 2^k ∨ r = r % 2^k + 2^k := by
  have hp : 0 < 2^k := Nat.two_pow_pos k
  have hpow : (2:Nat)^(k+1) = 2^k * 2 := by rw [Nat.pow_succ]
  have hsplit := Nat.div_add_mod r (2^k)
  have hlt : r / 2^k < 2 := by
    rw [hpow] at hr; exact Nat.div_lt_of_lt_mul (by omega)
  have hq : r / 2^k = 0 ∨ r / 2^k = 1 := by
    cases hd : r / 2^k with
    | zero => exact Or.inl rfl
    | succ q =>
        cases q with
        | zero => exact Or.inr rfl
        | succ q' => rw [hd] at hlt; omega
  rcases hq with h | h <;> rw [h] at hsplit
  · left; omega
  · right; omega

/-- **`D = +1` on the whole set `{Q = −1}` for the label `3·2^j`.** The three low-part
    conditions subsume the six lines, whose `mod 2^j` shadow they are. -/
theorem D_low_cond (j a b : Nat) (hj : j ≠ 0) (ha : a < 2^(j+2)) (hb : b < 2^(j+2))
    (hcond : a % 2^j = 0 ∨ b % 2^j = 0 ∨ a % 2^j = b % 2^j) :
    Ddef j (2^j + 2^(j+1)) a b (j+2) = 1 := by
  have hp : 0 < 2^j := Nat.two_pow_pos j
  have hpow : (2:Nat)^(j+1) = 2^j * 2 := by rw [Nat.pow_succ]
  have hpow2 : (2:Nat)^(j+2) = 2^(j+1) * 2 := by rw [Nat.pow_succ]
  have hlev : (j+1) + 1 = j + 2 := by omega
  have hYlt : (2:Nat)^j + 2^(j+1) < 2^(j+2) := by omega
  have hYmod : ((2:Nat)^j + 2^(j+1)) % 2^(j+1) = 2^j := by
    rw [Nat.add_mod_right]; exact Nat.mod_eq_of_lt (by omega)
  have hYlow : ((2:Nat)^j + 2^(j+1)) % 2^j = 0 := by
    rw [← mod_pow_mod j (j+1) _ (by omega), hYmod, Nat.mod_self]
  have haY : a ^^^ (2^j + 2^(j+1)) < 2^(j+2) := Nat.xor_lt_two_pow ha hYlt
  have hbY : b ^^^ (2^j + 2^(j+1)) < 2^(j+2) := Nat.xor_lt_two_pow hb hYlt
  have hxlow : ∀ x : Nat, (x ^^^ (2^j + 2^(j+1))) % 2^j = x % 2^j := by
    intro x; rw [xor_mod_two_pow, hYlow, Nat.xor_zero]
  have hxmid : ∀ x : Nat, (x ^^^ (2^j + 2^(j+1))) % 2^(j+1) = (x % 2^(j+1)) ^^^ 2^j := by
    intro x; rw [xor_mod_two_pow, hYmod]
  have hsplit : ∀ x : Nat, x % 2^j = 0 → x % 2^(j+1) = 0 ∨ x % 2^(j+1) = 2^j := by
    intro x hx
    have h1 : x % 2^(j+1) % 2^j = 0 := by rw [mod_pow_mod j (j+1) x (by omega)]; exact hx
    rcases low_split j (x % 2^(j+1)) (Nat.mod_lt _ (Nat.two_pow_pos (j+1))) with h | h
    · left; rw [h, h1]
    · right; rw [h, h1, Nat.zero_add]
  have keyA : ∀ x y : Nat, x < 2^(j+2) → y < 2^(j+2) → x % 2^j = 0 →
      gdisc j x y (j+2) * gdisc j (x ^^^ (2^j + 2^(j+1))) (y ^^^ (2^j + 2^(j+1))) (j+2)
        = psg (y % 2^j) := by
    intro x y hx hy hx0
    have hxY : x ^^^ (2^j + 2^(j+1)) < 2^(j+2) := Nat.xor_lt_two_pow hx hYlt
    have hyY : y ^^^ (2^j + 2^(j+1)) < 2^(j+2) := Nat.xor_lt_two_pow hy hYlt
    rw [← hlev] at hx hy hxY hyY
    rcases hsplit x hx0 with h | h
    · have e1 : gdisc j x y ((j+1)+1) = 1 := gdisc_block_zero j 1 x y hx hy h
      have e2 : (x ^^^ (2^j + 2^(j+1))) % 2^(j+1) = 2^j := by
        rw [hxmid x, h, Nat.zero_xor]
      have e3 := gdisc_lsb j 1 (x ^^^ (2^j + 2^(j+1))) (y ^^^ (2^j + 2^(j+1))) hxY hyY e2
      show gdisc j x y ((j+1)+1) * gdisc j _ _ ((j+1)+1) = _
      rw [e1, e3, hxlow y, Int.one_mul]
    · have e1 := gdisc_lsb j 1 x y hx hy h
      have e2 : (x ^^^ (2^j + 2^(j+1))) % 2^(j+1) = 0 := by
        rw [hxmid x, h, Nat.xor_self]
      have e3 : gdisc j (x ^^^ (2^j + 2^(j+1))) (y ^^^ (2^j + 2^(j+1))) ((j+1)+1) = 1 :=
        gdisc_block_zero j 1 _ _ hxY hyY e2
      show gdisc j x y ((j+1)+1) * gdisc j _ _ ((j+1)+1) = _
      rw [e1, e3, Int.mul_one]
  unfold Ddef
  rcases hcond with h | h | h
  · rw [keyA a b ha hb h, h, psg_zero, Int.mul_one]
    exact psg_sq (b % 2^j)
  · rw [gdisc_symm (j+2) j a b (by omega) ha hb,
        gdisc_symm (j+2) j (a ^^^ (2^j + 2^(j+1))) (b ^^^ (2^j + 2^(j+1))) (by omega) haY hbY,
        keyA b a hb ha h, h, psg_zero, Int.mul_one]
    exact psg_sq (a % 2^j)
  · have htlt : a % 2^j < 2^j := Nat.mod_lt _ hp
    have hAm : a % 2^(j+1) % 2^j = a % 2^j := mod_pow_mod j (j+1) a (by omega)
    have hBm : b % 2^(j+1) % 2^j = b % 2^j := mod_pow_mod j (j+1) b (by omega)
    have hAeq := low_split j (a % 2^(j+1)) (Nat.mod_lt _ (Nat.two_pow_pos (j+1)))
    have hBeq := low_split j (b % 2^(j+1)) (Nat.mod_lt _ (Nat.two_pow_pos (j+1)))
    rw [hAm] at hAeq
    rw [hBm, ← h] at hBeq
    have hxA : ∀ z : Nat, z < 2^j → (z + 2^j) ^^^ 2^j = z := by
      intro z hz
      rw [← xor_pow_low j z hj hz, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
    have hjl : j < j + 1 := by omega
    have hGG : gdisc j a b (j+2)
        * gdisc j (a ^^^ (2^j + 2^(j+1))) (b ^^^ (2^j + 2^(j+1))) (j+2) = 1 := by
      have ha' := ha; have hb' := hb; have haY' := haY; have hbY' := hbY
      rw [← hlev] at ha' hb' haY' hbY'
      have e1 : gdisc j a b ((j+1)+1)
          = gdisc j (a % 2^(j+1)) (b % 2^(j+1)) (j+1) :=
        gdisc_trunc j j hjl 1 a b ha' hb'
      have e2 : gdisc j (a ^^^ (2^j + 2^(j+1))) (b ^^^ (2^j + 2^(j+1))) ((j+1)+1)
          = gdisc j ((a % 2^(j+1)) ^^^ 2^j) ((b % 2^(j+1)) ^^^ 2^j) (j+1) := by
        rw [gdisc_trunc j j hjl 1 _ _ haY' hbY', hxmid a, hxmid b]
      show gdisc j a b ((j+1)+1) * gdisc j _ _ ((j+1)+1) = 1
      rw [e1, e2]
      have hlt1 : a % 2^j < 2^(j+1) := by omega
      have hlt2 : a % 2^j + 2^j < 2^(j+1) := by omega
      rcases hAeq with hA' | hA' <;> rcases hBeq with hB' | hB' <;> rw [hA', hB']
      · rw [xor_pow_low j (a % 2^j) hj htlt,
            gdisc_diag_one j j (a % 2^j) hjl hlt1, gdisc_diag_one j j (a % 2^j + 2^j) hjl hlt2]
        decide
      · rw [xor_pow_low j (a % 2^j) hj htlt, hxA (a % 2^j) htlt,
            gdisc_symm (j+1) j (a % 2^j) (a % 2^j + 2^j) hjl hlt1 hlt2,
            gdisc_seam_diag j (a % 2^j) hj htlt]
        exact psg_sq (a % 2^j)
      · rw [xor_pow_low j (a % 2^j) hj htlt, hxA (a % 2^j) htlt,
            gdisc_symm (j+1) j (a % 2^j) (a % 2^j + 2^j) hjl hlt1 hlt2,
            gdisc_seam_diag j (a % 2^j) hj htlt]
        exact psg_sq (a % 2^j)
      · rw [hxA (a % 2^j) htlt, gdisc_diag_one j j (a % 2^j + 2^j) hjl hlt2, gdisc_diag_one j j (a % 2^j) hjl hlt1]
        decide
    rw [hGG, Int.one_mul, h]
    exact psg_sq (b % 2^j)

/-! ### The closed form of `Q` at the label `3·2^j`

Both descent steps carry the label **high**, so the two signs cancel; what is left is `Qgen` or
`Qgen'` at label `0`, which is `1`. The side conditions of the eight `hi` rows involved are
exactly `a % 2^j ≠ 0`, `b % 2^j ≠ 0`, `a % 2^j ≠ b % 2^j` — so off that locus `Q = +1`, and the
locus is precisely where `Q = −1`. -/

theorem Qgen_zero_label (x y m : Nat) : Qgen 0 x y m = 1 := by
  unfold Qgen
  rw [Nat.xor_zero, Nat.xor_zero]
  have h := cdSq x y m
  rcases cdSigma_pm m x y with e | e <;> rw [e] <;> decide

theorem Qgen'_zero_label (x y m : Nat) : Qgen' 0 x y m = 1 := by
  unfold Qgen'
  rw [Nat.xor_zero, Nat.xor_zero]
  rcases cdSigma_pm m x y with e | e <;> rcases cdSigma_pm m y x with e2 | e2 <;>
    rw [e, e2] <;> decide

/-- Step 2 of the descent: at the label `2^j`, level `j+1`. -/
theorem Q'_pow2_level (j x y : Nat) (hj : j ≠ 0) (hx : x < 2^(j+1)) (hy : y < 2^(j+1))
    (hr : x % 2^j ≠ 0) (hs : y % 2^j ≠ 0) (hrs : x % 2^j ≠ y % 2^j) :
    Qgen' (2^j) x y (j+1) = -1 := by
  obtain ⟨i, rfl⟩ : ∃ i, j = i + 1 := ⟨j - 1, by omega⟩
  have hP : (0:Nat) < 2^(i+1) := Nat.two_pow_pos (i+1)
  have hupm : ∀ z : Nat, z < 2^(i+1) → (z + 2^(i+1)) % 2^(i+1) = z := by
    intro z h; rw [Nat.add_mod_right]; exact Nat.mod_eq_of_lt h
  show Qgen' (2^(i+1)) x y (i+2) = -1
  rcases split_top hx with hxl | ⟨r, hrl, rfl⟩ <;> rcases split_top hy with hyl | ⟨s, hsl, rfl⟩
  · rw [Nat.mod_eq_of_lt hxl] at hr hrs
    rw [Nat.mod_eq_of_lt hyl] at hs hrs
    have hrow := Q'red_hi_ll i 0 x y hP hxl hyl hr hs
      (by rwa [Nat.xor_zero]) (by rwa [Nat.xor_zero]) hrs
    rw [Nat.zero_add] at hrow
    rw [hrow, Qgen'_zero_label]
  · rw [Nat.mod_eq_of_lt hxl] at hr hrs
    rw [hupm s hsl] at hs hrs
    have hrow := Q'red_hi_lu i 0 x s hP hxl hsl hs
      (by rwa [Nat.xor_zero]) (by rwa [Nat.xor_zero]) hrs
    rw [Nat.zero_add] at hrow
    rw [hrow, Qgen_zero_label]
  · rw [hupm r hrl] at hr hrs
    rw [Nat.mod_eq_of_lt hyl] at hs hrs
    have hrow := Q'red_hi_ul i 0 r y hP hrl hyl hr hs
      (by rwa [Nat.xor_zero]) (by rwa [Nat.xor_zero]) hrs
    rw [Nat.zero_add] at hrow
    rw [hrow, Qgen_zero_label]
  · rw [hupm r hrl] at hr hrs
    rw [hupm s hsl] at hs hrs
    have hrow := Q'red_hi_uu i 0 r s hP hrl hsl hr hs
      (by rwa [Nat.xor_zero]) (by rwa [Nat.xor_zero]) hrs
      (by rw [Nat.xor_zero]; intro h; exact hrs (xor_zero_eq r s h))
    rw [Nat.zero_add] at hrow
    rw [hrow, Qgen'_zero_label]

/-- **`Q` at the label `3·2^j` is `+1` off the low-part locus.** -/
theorem Q_three_pow2 (j a b : Nat) (hj : j ≠ 0) (ha : a < 2^(j+2)) (hb : b < 2^(j+2))
    (hr : a % 2^j ≠ 0) (hs : b % 2^j ≠ 0) (hrs : a % 2^j ≠ b % 2^j) :
    Qgen (2^j + 2^(j+1)) a b (j+2) = 1 := by
  have hP : (2:Nat)^j < 2^(j+1) := Nat.pow_lt_pow_right (by omega) (by omega)
  have hdrop : ∀ z : Nat, (z + 2^(j+1)) % 2^j = z % 2^j :=
    fun z => add_pow_mod j (j+1) z (by omega)
  have hne0 : ∀ z : Nat, z % 2^j ≠ 0 → z ≠ 0 := by
    intro z h e; rw [e, Nat.zero_mod] at h; exact h rfl
  have hnep : ∀ z : Nat, z % 2^j ≠ 0 → z ^^^ 2^j ≠ 0 := by
    intro z h e
    have h2 := xor_zero_eq z (2^j) e
    rw [h2, Nat.mod_self] at h; exact h rfl
  have hxor3 : ∀ x y : Nat, x % 2^j ≠ y % 2^j → x ^^^ y ^^^ 2^j ≠ 0 := by
    intro x y h e
    apply h
    have h2 := congrArg (fun z => z % 2^j) (xor_zero_eq (x ^^^ y) (2^j) e)
    simp only [xor_mod_two_pow, Nat.mod_self] at h2
    exact xor_zero_eq _ _ h2
  rcases split_top ha with hal | ⟨u, hu, rfl⟩ <;> rcases split_top hb with hbl | ⟨v, hv, rfl⟩
  · rw [Qred_hi_ll j (2^j) a b hP hal hbl (hne0 b hs) (hnep b hs),
        Q'_pow2_level j a b hj hal hbl hr hs hrs]
    decide
  · rw [hdrop v] at hs hrs
    rw [Qred_hi_lu j (2^j) a v hP hal hv (hne0 a hr) (hne0 v hs) (hnep a hr) (hnep v hs)
          (hxor3 a v hrs),
        Q'_pow2_level j v a hj hv hal hs hr (fun e => hrs e.symm)]
    decide
  · rw [hdrop u] at hr hrs
    rw [Qred_hi_ul j (2^j) u b hP hu hbl (hne0 b hs) (hnep b hs),
        Q'_pow2_level j u b hj hu hbl hr hs hrs]
    decide
  · rw [hdrop u] at hr hrs
    rw [hdrop v] at hs hrs
    rw [Qred_hi_uu j (2^j) u v hP hu hv (hne0 u hr) (hne0 v hs) (hnep u hr) (hnep v hs)
          (hxor3 u v hrs),
        Q'_pow2_level j v u hj hv hu hs hr (fun e => hrs e.symm)]
    decide

/-- **(♦) HOLDS FOR THE LABEL CLASS `Y₀ = 3·2^j`, ∀n.** Off the low-part locus `Q = +1`
    (`Q_three_pow2`), so `collapse` makes the hypothesis unsatisfiable there; on it, `D = +1`
    (`D_low_cond`). -/
theorem diamond_three (j n Y a b : Nat) (hj : j ≠ 0) (hn : j+2 ≤ n)
    (hY : Y < 2^n) (ha : a < 2^n) (hb : b < 2^n)
    (hY0 : Y % 2^(j+2) = 2^j + 2^(j+1)) (hsg : dsgnN j n Y = 1) (hw : Qgen' Y a b n = -1) :
    Ddef j (2^j + 2^(j+1)) (a % 2^(j+2)) (b % 2^(j+2)) (j+2) = 1 := by
  have hp2 : (0:Nat) < 2^(j+2) := Nat.two_pow_pos (j+2)
  have hpow : (2:Nat)^(j+1) = 2^j * 2 := by rw [Nat.pow_succ]
  have hpow2 : (2:Nat)^(j+2) = 2^(j+1) * 2 := by rw [Nat.pow_succ]
  have hp : 0 < 2^j := Nat.two_pow_pos j
  have hmm : ∀ x : Nat, x % 2^(j+2) % 2^j = x % 2^j :=
    fun x => mod_pow_mod j (j+2) x (by omega)
  have hYlow : ((2:Nat)^j + 2^(j+1)) % 2^j = 0 := by
    rw [Nat.add_mod, Nat.mod_self, Nat.zero_add]
    have h : (2:Nat)^(j+1) % 2^j = 0 := by rw [hpow]; exact Nat.mul_mod_right _ _
    rw [h, Nat.zero_mod]
  by_cases hcond : a % 2^j = 0 ∨ b % 2^j = 0 ∨ a % 2^j = b % 2^j
  · refine D_low_cond j (a % 2^(j+2)) (b % 2^(j+2)) hj (Nat.mod_lt _ hp2) (Nat.mod_lt _ hp2) ?_
    rw [hmm a, hmm b]; exact hcond
  · exfalso
    have f1 : a % 2^j ≠ 0 := fun e => hcond (Or.inl e)
    have f2 : b % 2^j ≠ 0 := fun e => hcond (Or.inr (Or.inl e))
    have f3 : a % 2^j ≠ b % 2^j := fun e => hcond (Or.inr (Or.inr e))
    have hYj : Y % 2^j = 0 := by rw [← hmm Y, hY0, hYlow]
    have hnd : NDeg j Y a b := by
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
      · intro e; exact f1 (by rw [← hmm a, e, Nat.zero_mod])
      · intro e; exact f2 (by rw [← hmm b, e, Nat.zero_mod])
      · intro e
        apply f1
        have h2 := congrArg (fun z => z % 2^j) e
        simp only [hmm] at h2
        rw [h2]; exact hYj
      · intro e
        apply f2
        have h2 := congrArg (fun z => z % 2^j) e
        simp only [hmm] at h2
        rw [h2]; exact hYj
      · intro e; exact f3 (by rw [← hmm a, ← hmm b, e])
      · intro e
        apply f3
        have h2 := congrArg (fun z => z % 2^j) e
        simp only [xor_mod_two_pow, hmm] at h2
        rw [hYj] at h2
        exact xor_zero_eq _ _ h2
    have hQ := Q_three_pow2 j (a % 2^(j+2)) (b % 2^(j+2)) hj
      (Nat.mod_lt _ hp2) (Nat.mod_lt _ hp2)
      (by rw [hmm a]; exact f1) (by rw [hmm b]; exact f2) (by rw [hmm a, hmm b]; exact f3)
    rw [collapse j n Y a b hn hY ha hb hnd, hY0, hsg, hQ] at hw
    exact absurd hw (by decide)

/-- **(♦), BOTH LABEL CLASSES, ∀n.** `lsb Y = j` leaves exactly two bottom labels, `2^j` and
    `3·2^j`; the sign disjunction is what `N12` says even weight is. -/
theorem diamond_all (j n Y a b : Nat) (hj : j ≠ 0) (hn : j+2 ≤ n)
    (hY : Y < 2^n) (ha : a < 2^n) (hb : b < 2^n)
    (hsign : dsgnN j n Y = -1 ∧ Y % 2^(j+2) = 2^j ∨
             dsgnN j n Y = 1 ∧ Y % 2^(j+2) = 2^j + 2^(j+1))
    (hw : Qgen' Y a b n = -1) :
    Ddef j (Y % 2^(j+2)) (a % 2^(j+2)) (b % 2^(j+2)) (j+2) = 1 := by
  rcases hsign with ⟨hsg, hY0⟩ | ⟨hsg, hY0⟩
  · rw [hY0]
    exact diamond_pow2 j n Y a b hn hY ha hb hY0 hsg hw
  · rw [hY0]
    exact diamond_three j n Y a b hj hn hY ha hb hY0 hsg hw

/-- At `j = 0` the swap is the identity, so the defect is `1` outright. -/
theorem Ddef_zero (Y a b m : Nat) : Ddef 0 Y a b m = 1 := by
  unfold Ddef gdisc
  rw [tau_id_zero, tau_id_zero, tau_id_zero, tau_id_zero, cdSq, cdSq,
      Nat.pow_zero, Nat.mod_one, Nat.mod_one, psg_zero]
  decide

/-- `D` at level `n` **is** `D` at the bottom — this is `G_trunc`, plus the fact that `psg` only
    sees the bottom `j` bits. -/
theorem Ddef_trunc (j d Y a b : Nat) (hY : Y < 2^((j+1+1)+d))
    (ha : a < 2^((j+1+1)+d)) (hb : b < 2^((j+1+1)+d)) :
    Ddef j Y a b ((j+1+1)+d)
      = Ddef j (Y % 2^(j+1+1)) (a % 2^(j+1+1)) (b % 2^(j+1+1)) (j+1+1) := by
  unfold Ddef
  rw [G_trunc j (j+1) d Y a b (by omega) hY ha hb,
      mod_pow_mod j (j+1+1) a (by omega), mod_pow_mod j (j+1+1) b (by omega)]

/-- **(♦), AT LEVEL `n`, BOTH LABEL CLASSES, EVERY `j`.** The defect of every witness is `+1` —
    this is `REACH ⊆ {D = +1}`, the last measured link of L2, at the level the statement lives on.
    The sign disjunction is what `N12` says even weight is. -/
theorem diamond_at_level (j n Y a b : Nat) (hn : j+2 ≤ n)
    (hY : Y < 2^n) (ha : a < 2^n) (hb : b < 2^n)
    (hsign : dsgnN j n Y = -1 ∧ Y % 2^(j+2) = 2^j ∨
             dsgnN j n Y = 1 ∧ Y % 2^(j+2) = 2^j + 2^(j+1))
    (hw : Qgen' Y a b n = -1) :
    Ddef j Y a b n = 1 := by
  by_cases hj : j = 0
  · subst hj; exact Ddef_zero Y a b n
  · obtain ⟨d, rfl⟩ : ∃ d, n = (j+1+1) + d := ⟨n - (j+2), by omega⟩
    rw [Ddef_trunc j d Y a b hY ha hb]
    exact diamond_all j ((j+1+1)+d) Y a b hj hn hY ha hb hsign hw

/-! ## `N12`: even weight **is** the sign disjunction

The last measured step. `dsgnN` is the accumulated descent sign; even weight is `psg Y = 1`. The
bridge is `psg_split`, and its crux is one fact about a single bit:
`(x / 2^k) % 2 = (x % 2^{k+1}) / 2^k`. -/

/-- The bit that `x % 2^{k+1}` carries above `2^k` is `(x / 2^k) % 2`. -/
theorem div_mod_two (k x : Nat) : (x / 2^k) % 2 = (x % 2^(k+1)) / 2^k := by
  have hp : 0 < 2^k := Nat.two_pow_pos k
  have hpow : (2:Nat)^(k+1) = 2^k * 2 := by rw [Nat.pow_succ]
  have hd : 2^k * (2 * (x / 2^(k+1))) + x % 2^(k+1) = x := by
    rw [← Nat.mul_assoc, ← hpow]
    exact Nat.div_add_mod x (2^(k+1))
  have h1 : x / 2^k = 2 * (x / 2^(k+1)) + (x % 2^(k+1)) / 2^k := by
    have hh := Nat.mul_add_div hp (2 * (x / 2^(k+1))) (x % 2^(k+1))
    rw [hd] at hh
    exact hh
  have hrl : x % 2^(k+1) < 2^k * 2 := by
    rw [← hpow]; exact Nat.mod_lt _ (Nat.two_pow_pos (k+1))
  have h2 : (x % 2^(k+1)) / 2^k < 2 := Nat.div_lt_of_lt_mul (by omega)
  rw [h1, Nat.mul_add_mod]
  exact Nat.mod_eq_of_lt h2

/-- **`psg` splits at any level**: `(−1)^{popcount x}` is the product over the low and high parts. -/
theorem psg_split : ∀ (k x : Nat), psg x = psg (x % 2^k) * psg (x / 2^k) := by
  intro k
  induction k with
  | zero =>
      intro x
      rw [Nat.pow_zero, Nat.mod_one, Nat.div_one, psg_zero, Int.one_mul]
  | succ k ih =>
      intro x
      have hp : 0 < 2^k := Nat.two_pow_pos k
      have hmm : x % 2^(k+1) % 2^k = x % 2^k := mod_pow_mod k (k+1) x (by omega)
      have hlt : x % 2^k < 2^k := Nat.mod_lt _ hp
      have hdd : x / 2^(k+1) = (x / 2^k) / 2 := by
        rw [Nat.div_div_eq_div_mul, ← Nat.pow_succ]
      rcases low_split k (x % 2^(k+1)) (Nat.mod_lt _ (Nat.two_pow_pos (k+1))) with h | h <;>
        rw [hmm] at h
      · have hbit : (x / 2^k) % 2 = 0 := by
          rw [div_mod_two k x, h]; exact Nat.div_eq_of_lt hlt
        rw [h, hdd, ih x, psg_step (x / 2^k), hbit, if_neg (by decide), Int.one_mul]
      · have hbit : (x / 2^k) % 2 = 1 := by
          rw [div_mod_two k x, h, Nat.add_div_right _ hp, Nat.div_eq_of_lt hlt]
        rw [h, psg_top k (x % 2^k) hlt, hdd, ih x, psg_step (x / 2^k), hbit,
            if_pos rfl, ← hdd, Int.neg_mul, Int.one_mul, Int.mul_neg, Int.neg_mul]

/-- **The accumulated descent sign is `psg Y` corrected by the bottom label.** -/
theorem dsgnN_eq (j : Nat) : ∀ (n Y : Nat), j+2 ≤ n → Y < 2^n →
    dsgnN j n Y = psg Y * psg (Y % 2^(j+2)) := by
  intro n
  induction n with
  | zero => intro Y hn; omega
  | succ n ih =>
      intro Y hn hY
      rcases Nat.eq_or_lt_of_le hn with heq | hlt
      · have hne : n = j + 1 := by omega
        subst hne
        show dsgnN j (j+2) Y = psg Y * psg (Y % 2^(j+2))
        rw [dsgnN_bot j Y, Nat.mod_eq_of_lt hY]
        exact (psg_sq Y).symm
      · have hjn : j + 2 ≤ n := by omega
        have hbb : ¬ (n+1 ≤ j+2) := by omega
        have hYn : Y % 2^n < 2^n := Nat.mod_lt _ (Nat.two_pow_pos n)
        have hpow : (2:Nat)^(n+1) = 2^n * 2 := by rw [Nat.pow_succ]
        have hq : Y / 2^n < 2 := by
          have h2 : Y < 2^n * 2 := by omega
          exact Nat.div_lt_of_lt_mul h2
        have hq2 : Y / 2^n = 0 ∨ Y / 2^n = 1 := by
          cases hqv : Y / 2^n with
          | zero => exact Or.inl rfl
          | succ q =>
              cases q with
              | zero => exact Or.inr rfl
              | succ q' => rw [hqv] at hq; omega
        have hif : (if Y / 2^n % 2 = 1 then (-1:Int) else 1) = psg (Y / 2^n) := by
          rcases hq2 with h | h <;> rw [h]
          · rw [if_neg (by decide), psg_zero]
          · rw [if_pos (by decide), psg_one]
        rw [dsgnN, if_neg hbb, ih (Y % 2^n) hjn hYn, mod_pow_mod (j+2) n Y hjn, hif,
            psg_split n Y, Int.mul_comm (psg (Y % 2^n)) (psg (Y / 2^n)), Int.mul_assoc]

theorem psg_pow2 (j : Nat) : psg (2^j) = -1 := by
  have h := psg_top j 0 (Nat.two_pow_pos j)
  rw [Nat.zero_add, psg_zero] at h
  exact h

theorem psg_three (j : Nat) : psg (2^j + 2^(j+1)) = 1 := by
  have h := psg_top (j+1) (2^j) (Nat.pow_lt_pow_right (by omega) (by omega))
  rw [h, psg_pow2]
  decide

/-- **`N12`, proven: even weight IS the sign disjunction.** -/
theorem even_weight_sign (j n Y : Nat) (hn : j+2 ≤ n) (hY : Y < 2^n)
    (hlsb : Y % 2^(j+1) = 2^j) (heven : psg Y = 1) :
    dsgnN j n Y = -1 ∧ Y % 2^(j+2) = 2^j ∨
    dsgnN j n Y = 1 ∧ Y % 2^(j+2) = 2^j + 2^(j+1) := by
  have hd := dsgnN_eq j n Y hn hY
  rw [heven, Int.one_mul] at hd
  have hmm : Y % 2^(j+2) % 2^(j+1) = Y % 2^(j+1) := mod_pow_mod (j+1) (j+2) Y (by omega)
  rcases low_split (j+1) (Y % 2^(j+2)) (Nat.mod_lt _ (Nat.two_pow_pos (j+2))) with h | h <;>
    rw [hmm, hlsb] at h
  · exact Or.inl ⟨by rw [hd, h, psg_pow2], h⟩
  · exact Or.inr ⟨by rw [hd, h, psg_three], h⟩

/-- ## ★★★ (♦), PROVEN ∀n
    For even-weight `Y` with `lsb Y = j`, every witness of the resonance predicate has defect
    `+1`. Even weight is `psg Y = 1`, i.e. `(−1)^{popcount Y} = 1`. No hypothesis on `j`, none on
    `a` and `b`. With `l2_reduction` (`L2 ⟸ (♦)`, proven ∀n), **L2 follows**. -/
theorem diamond (j n Y a b : Nat) (hn : j+2 ≤ n)
    (hY : Y < 2^n) (ha : a < 2^n) (hb : b < 2^n)
    (hlsb : Y % 2^(j+1) = 2^j) (heven : psg Y = 1) (hw : Qgen' Y a b n = -1) :
    Ddef j Y a b n = 1 :=
  diamond_at_level j n Y a b hn hY ha hb (even_weight_sign j n Y hn hY hlsb heven) hw

/-- ## ★★★ L2, ∀n
    Composing `diamond` with `l2_reduction_symm`: on the fiber `L = Y + 2^(m+1)`, the
    τ-discrepancy of `σ` **is** the coboundary `λ(a)·λ(b)` with `λ(x) = (−1)^{p_j(x)}`, for every
    even-weight `Y` with `lsb Y = j` on the resonance locus. This is L2's statement, and every
    link in its chain is now a theorem. -/
theorem L2_forall (m j Y a b : Nat) (hj : j+2 ≤ m+1) (hY : Y < 2^(m+1))
    (ha : a < 2^(m+1)) (hb : b < 2^(m+1)) (hbY : b ^^^ Y ≠ 0)
    (hlsb : Y % 2^(j+1) = 2^j) (heven : psg Y = 1)
    (hres : Qgen' Y a b (m+1) = -1) :
    cdSigma (tau j a) (tau j b) (m+2)
        * cdSigma (tau j (a ^^^ (Y + 2^(m+1)))) (tau j (b ^^^ (Y + 2^(m+1)))) (m+2)
        * (cdSigma a b (m+2) * cdSigma (a ^^^ (Y + 2^(m+1))) (b ^^^ (Y + 2^(m+1))) (m+2))
      = psg (a % 2^j) * psg (b % 2^j) := by
  rw [l2_reduction_symm m j Y a b (by omega) hY ha hb hbY]
  have hd := diamond j (m+1) Y a b hj hY ha hb hlsb heven hres
  unfold Ddef at hd
  rcases psg_pm (a % 2^j) with hA | hA <;> rcases psg_pm (b % 2^j) with hB | hB <;>
    rw [hA, hB] at hd ⊢ <;> simp at hd ⊢ <;> omega

/-- **`Qgen'` is `+1` on the coset partner**, `b = a ⊕ W`. Same two-line shape as `Qgen'_diag`:
    two of the four factors coincide and square away by `cdSq`, and the other two are
    self-pairings, pinned to `−1` by `sigma_self`. This is the "never an edge" fact the degree
    count needs: the pair `(a, a⊕W)` is resonance-free for every `a`. -/
theorem Qgen'_coset_partner (m W a : Nat) (hW : W < 2^m) (ha : a < 2^m)
    (ha0 : a ≠ 0) (haW : a ^^^ W ≠ 0) : Qgen' W a (a ^^^ W) m = 1 := by
  have haWlt : a ^^^ W < 2^m := Nat.xor_lt_two_pow ha hW
  have hcancel : (a ^^^ W) ^^^ W = a := by
    rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
  have e1 : cdSigma a a m = -1 := sigma_self m a ha ha0
  have e2 : cdSigma (a ^^^ W) (a ^^^ W) m = -1 := sigma_self m (a ^^^ W) haWlt haW
  unfold Qgen'
  rw [hcancel, e1, e2]
  rcases cdSigma_pm m a (a ^^^ W) with h | h <;> rw [h] <;> decide

/-! ## Tier 21: the value of `Q'` on the whole locus where `Q = -1`

`Qgen_degen` (the six `= 0` degeneracies) and the `Qgen_H_*` family (the six `= H` gap roots)
between them pin `Q = -1` on a twelve-condition locus. `Qgen'_eq_chi` factors `Q'` through `Q`
and two commutation signs, and `chi_char` makes those explicit -- so ONE theorem prices every
degenerate slice of the level descent. This is what the level-constants `10e-18` (low label)
and `6e-10` (high label) are made of: every failure slice of every reduction row lies on that
locus, and its value is read off here. -/

private theorem chi_neg_of (m x y : Nat) (hx : x < 2^m) (hy : y < 2^m)
    (h0 : x ≠ 0) (h1 : y ≠ 0) (h2 : x ≠ y) : chi x y m = -1 := by
  have hc : ¬ (x = 0 ∨ y = 0 ∨ x = y) := by
    intro hc
    rcases hc with h | h | h
    · exact h0 h
    · exact h1 h
    · exact h2 h
  rw [chi_char m x y hx hy, if_neg hc]

private theorem chi_one_left (m y : Nat) (hy : y < 2^m) : chi 0 y m = 1 := by
  rw [chi_char m 0 y (Nat.two_pow_pos m) hy, if_pos (Or.inl rfl)]

private theorem chi_one_right (m x : Nat) (hx : x < 2^m) : chi x 0 m = 1 := by
  rw [chi_char m x 0 hx (Nat.two_pow_pos m), if_pos (Or.inr (Or.inl rfl))]

/-- `W = b ⊕ W` forces `b = 0`. -/
private theorem xor_self_ne (b W : Nat) (hb0 : b ≠ 0) : W ≠ b ^^^ W := by
  intro h
  apply hb0
  have h2 : (b ^^^ W) ^^^ W = W ^^^ W := by rw [← h]
  rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero] at h2
  exact h2

/-- **Wherever `Q = -1`, `Q'` is minus the product of the two commutation signs.** One
    rewrite; the content is entirely `Qgen'_eq_chi`. Combined with `chi_char`, `Q' = +1`
    exactly when precisely one of the two `chi`s is `-1`. -/
theorem Qgen'_on_neg (m W a b : Nat) (hQ : Qgen W a b m = -1) :
    Qgen' W a b m = - (chi (a ^^^ W) (b ^^^ W) m * chi a (b ^^^ W) m) := by
  rw [Qgen'_eq_chi, hQ, Int.neg_one_mul, Int.neg_mul]

/-- Line `a = 0`: `Q' = +1` (while `Q = -1`). -/
theorem Qgen'_zero_left (m W b : Nat) (hW : W < 2^m) (hb : b < 2^m) (hW0 : W ≠ 0)
    (hb0 : b ≠ 0) (hbW : b ≠ W) : Qgen' W 0 b m = 1 := by
  have hbXlt : b ^^^ W < 2^m := Nat.xor_lt_two_pow hb hW
  have hbX0 : b ^^^ W ≠ 0 := fun h => hbW (xor_zero_eq b W h)
  have hQ : Qgen W 0 b m = -1 :=
    Qgen_degen m W 0 b hW (Nat.two_pow_pos m) hb hW0 (Or.inl rfl)
  rw [Qgen'_on_neg m W 0 b hQ, Nat.zero_xor,
      chi_neg_of m W (b ^^^ W) hW hbXlt hW0 hbX0 (xor_self_ne b W hb0),
      chi_one_left m (b ^^^ W) hbXlt]
  decide

/-- Line `a = W`: `Q' = +1`. This is "lemma A, first half" -- the slice the level descent
    removes from the ll/uu quadrants contributes nothing to the resonance count. -/
theorem Qgen'_label_left (m W b : Nat) (hW : W < 2^m) (hb : b < 2^m) (hW0 : W ≠ 0)
    (hb0 : b ≠ 0) (hbW : b ≠ W) : Qgen' W W b m = 1 := by
  have hbXlt : b ^^^ W < 2^m := Nat.xor_lt_two_pow hb hW
  have hbX0 : b ^^^ W ≠ 0 := fun h => hbW (xor_zero_eq b W h)
  have hQ : Qgen W W b m = -1 :=
    Qgen_degen m W W b hW hW hb hW0 (Or.inr (Or.inr (Or.inl (Nat.xor_self W))))
  rw [Qgen'_on_neg m W W b hQ, Nat.xor_self,
      chi_one_left m (b ^^^ W) hbXlt,
      chi_neg_of m W (b ^^^ W) hW hbXlt hW0 hbX0 (xor_self_ne b W hb0)]
  decide

/-- Line `b = 0`: `Q' = -1`. -/
theorem Qgen'_zero_right (m W a : Nat) (hW : W < 2^m) (ha : a < 2^m) (hW0 : W ≠ 0)
    (ha0 : a ≠ 0) (haW : a ≠ W) : Qgen' W a 0 m = -1 := by
  have haXlt : a ^^^ W < 2^m := Nat.xor_lt_two_pow ha hW
  have haX0 : a ^^^ W ≠ 0 := fun h => haW (xor_zero_eq a W h)
  have hQ : Qgen W a 0 m = -1 :=
    Qgen_degen m W a 0 hW ha (Nat.two_pow_pos m) hW0 (Or.inr (Or.inl rfl))
  rw [Qgen'_on_neg m W a 0 hQ, Nat.zero_xor,
      chi_neg_of m (a ^^^ W) W haXlt hW haX0 hW0 (fun h => xor_self_ne a W ha0 h.symm),
      chi_neg_of m a W ha hW ha0 hW0 haW]
  decide

/-- Line `b = W`: `Q' = -1`. "Lemma A, second half" -- this slice contributes its FULL size
    to the resonance count, and that asymmetry with `Qgen'_label_left` is exactly the `(e-2)`
    that the high-label bridge `M = N'` has to absorb. -/
theorem Qgen'_label_right (m W a : Nat) (hW : W < 2^m) (ha : a < 2^m) (hW0 : W ≠ 0) :
    Qgen' W a W m = -1 := by
  have haXlt : a ^^^ W < 2^m := Nat.xor_lt_two_pow ha hW
  have hQ : Qgen W a W m = -1 :=
    Qgen_degen m W a W hW ha hW hW0 (Or.inr (Or.inr (Or.inr (Or.inl (Nat.xor_self W)))))
  rw [Qgen'_on_neg m W a W hQ, Nat.xor_self,
      chi_one_right m (a ^^^ W) haXlt, chi_one_right m a ha]
  decide

/-! ## Tier 22: the base of the descent

The level recursion sends `(m, W)` to `(m-1, W % 2^(m-1))`. For an ODD label -- the lane's
`Llo = 8y+1` -- the reduced label is odd at every level (`odd_stays_odd`), so it is never `0`,
which is exactly the null control the ledger needs; and at level 1 it is `1`. So every chain
bottoms out in the label-`2^k` family, and there `Q'` has a complete closed form
(`Qgen'_pow2_eq`): `+1` on exactly two lines, `-1` on the rest of the box. -/

/-- **Off the two `+1` lines, `Q'` is `-1` wherever `Q` is.** With `a ≠ 0`, `a ≠ W`, `a ≠ b`
    and `b ≠ a ⊕ W`, the two commutation signs AGREE -- both `+1` when `b = W`, both `-1`
    otherwise -- so their product is `+1` either way and `Q' = Q = -1`. The companion of the
    four line-values above: those give `Q'` ON the degeneracy lines, this gives it OFF them. -/
theorem Qgen'_off_lines (m W a b : Nat) (hW : W < 2^m) (ha : a < 2^m) (hb : b < 2^m)
    (hQ : Qgen W a b m = -1) (ha0 : a ≠ 0) (haW : a ≠ W) (hab : a ≠ b)
    (hcos : b ≠ a ^^^ W) : Qgen' W a b m = -1 := by
  have haXlt : a ^^^ W < 2^m := Nat.xor_lt_two_pow ha hW
  have hbXlt : b ^^^ W < 2^m := Nat.xor_lt_two_pow hb hW
  have haX0 : a ^^^ W ≠ 0 := fun h => haW (xor_zero_eq a W h)
  rw [Qgen'_on_neg m W a b hQ]
  by_cases hbW : b = W
  · have hz : b ^^^ W = 0 := by rw [hbW, Nat.xor_self]
    rw [hz, chi_one_right m (a ^^^ W) haXlt, chi_one_right m a ha]
    decide
  · have hbX0 : b ^^^ W ≠ 0 := fun h => hbW (xor_zero_eq b W h)
    have hne1 : a ^^^ W ≠ b ^^^ W := by
      intro h
      apply hab
      have h2 : (a ^^^ W) ^^^ W = (b ^^^ W) ^^^ W := by rw [h]
      rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero, Nat.xor_assoc, Nat.xor_self,
          Nat.xor_zero] at h2
      exact h2
    have hne2 : a ≠ b ^^^ W := by
      intro h
      apply hcos
      have h2 : a ^^^ W = (b ^^^ W) ^^^ W := by rw [h]
      rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero] at h2
      exact h2.symm
    rw [chi_neg_of m (a ^^^ W) (b ^^^ W) haXlt hbXlt haX0 hbX0 hne1,
        chi_neg_of m a (b ^^^ W) ha hbXlt ha0 hbX0 hne2]
    decide

/-- **THE BASE CASE, in closed form at every level.** On the box `1 ≤ a, b < 2^m`, `a ≠ b`,
    the label-`2^k` value of `Q'` is `+1` on exactly two lines -- `a = 2^k` and `b = a ⊕ 2^k`,
    which are disjoint and of size `2^m - 2` each -- and `-1` on everything else. Hence the
    resonance count is `(2^m-1)(2^m-2) - 2(2^m-2) = (2^m-2)(2^m-3)`, independent of `k`.
    Every ingredient is a theorem: `Qgen_pow2` (already in the tree) for `Q = -1`, then
    `Qgen'_label_left`, `Qgen'_coset_partner` and `Qgen'_off_lines`. -/
theorem Qgen'_pow2_eq (m k a b : Nat) (hk : k < m) (ha : a < 2^m) (hb : b < 2^m)
    (ha0 : a ≠ 0) (hb0 : b ≠ 0) (hab : a ≠ b) :
    Qgen' (2^k) a b m = if a = 2^k ∨ b = a ^^^ 2^k then 1 else -1 := by
  have hWlt : (2:Nat)^k < 2^m := Nat.pow_lt_pow_right (by omega) hk
  have hW0 : (2:Nat)^k ≠ 0 := by have := Nat.two_pow_pos k; omega
  by_cases h1 : a = 2^k
  · rw [if_pos (Or.inl h1), h1]
    exact Qgen'_label_left m (2^k) b hWlt hb hW0 hb0 (fun h => hab (h1.trans h.symm))
  · by_cases h2 : b = a ^^^ 2^k
    · rw [if_pos (Or.inr h2), h2]
      exact Qgen'_coset_partner m (2^k) a hWlt ha ha0
        (fun h => h1 (xor_zero_eq a (2^k) h))
    · rw [if_neg (fun h => h.elim h1 h2)]
      exact Qgen'_off_lines m (2^k) a b hWlt ha hb
        (Qgen_pow2 m k a b hk ha hb) ha0 h1 hab h2

/-- The bottom of the descent: at level 1 the box is EMPTY, so the base count is `0` --
    which is what `(2^m-2)(2^m-3)` gives at `m = 1`. -/
theorem base_box_empty (a b : Nat) (ha : a < 2^1) (hb : b < 2^1) (ha0 : a ≠ 0)
    (hb0 : b ≠ 0) : a = b := by
  have h : (2:Nat)^1 = 2 := rfl
  rw [h] at ha hb
  omega

/-- An odd label stays odd all the way down the descent, so the reduced label is NEVER `0` --
    the hypothesis the whole ledger rests on, and the one the `W' = 0` null control violates. -/
theorem odd_stays_odd (W l : Nat) (hl : 1 ≤ l) (hodd : W % 2 = 1) : (W % 2^l) % 2 = 1 := by
  have h2 : (2:Nat)^1 = 2 := rfl
  have := mod_pow_mod 1 l W hl
  rw [h2] at this
  rw [this]
  exact hodd

/-! ## Tier 23: the label invariant `g`

The resonance count `N(m,W)` depends on `W` only through `g(W) = (W &&& (W-1)) >>> 3`. Half of
that is a theorem, and it is this tier: **`Q'` is `tau`-equivariant**, so the count is unchanged
when `tau j` (`j ≤ lsb W`) is applied to the label and to both arguments. At `j = lsb W` the
label-level action is exactly "move the lowest set bit to position 0" (`tau_lsb`), which
NORMALISES every label to an odd one (`tau_lsb_odd`) -- and `g W = (tau (lsb W) W) >>> 3`.

What is NOT proven here: (i) the step from the pointwise equivariance to the equality of
COUNTS -- that needs the bijection-to-cardinality argument, which is Finset territory this
Mathlib-free file does not have; and (ii) the residual collapse, that bits 1 and 2 of an
already-odd label do not matter. `tau` alone is SOUND but not complete: measurement (`W18`)
puts exactly four `tau`-orbits in each block. -/

/-- **`Q'` is `tau`-equivariant.** Three theorems already in the tree do all the work:
    `star_forall` gives it for `Q`, `tau_xor` moves `tau` through the `xor`s, and `chi_tau`
    says the two commutation signs cannot see `tau` at all. -/
theorem Qgen'_tau (m j Y a b : Nat) (hj : j < m) (hY : Y < 2^m) (hY0 : Y ≠ 0)
    (hmod : Y % 2^j = 0) (ha : a < 2^m) (hb : b < 2^m) :
    Qgen' Y a b m = Qgen' (tau j Y) (tau j a) (tau j b) m := by
  have haY : a ^^^ Y < 2^m := Nat.xor_lt_two_pow ha hY
  have hbY : b ^^^ Y < 2^m := Nat.xor_lt_two_pow hb hY
  rw [Qgen'_eq_chi, Qgen'_eq_chi, ← tau_xor j a Y, ← tau_xor j b Y,
      chi_tau m j (a ^^^ Y) (b ^^^ Y) hj haY hbY,
      chi_tau m j a (b ^^^ Y) hj ha hbY,
      ← star_forall m j Y a b hY hY0 hmod ha hb]

/-- At `j = lsb W`, `tau` normalises the label: the result is ODD. Concretely `tau` moves the
    lowest set bit down to position 0, and `g W = (tau (lsb W) W) >>> 3` -- the additive form
    `tau t W + 2^t = W + 1`, hence `tau t W = (W &&& (W-1)) + 1`, is pure bit arithmetic and is
    pinned in `W18` rather than proven here. Oddness is the part the ledger actually needs: it
    is what makes `odd_stays_odd` -- and so `W' ≠ 0` -- available at every level below. -/
theorem tau_lsb_odd (t W : Nat) (ht : 1 ≤ t) (hlsb : W % 2^(t+1) = 2^t) :
    tau t W % 2 = 1 := by
  have hz : (2:Nat)^t % 2 = 0 := by
    obtain ⟨s, rfl⟩ : ∃ s, t = s + 1 := ⟨t - 1, by omega⟩
    rw [Nat.pow_succ]; omega
  have h21 : (2:Nat)^1 = 2 := rfl
  have hw2 : W % 2 = 0 := by
    have hm : W % 2^(t+1) % 2^1 = W % 2^1 := mod_pow_mod 1 (t+1) W (by omega)
    rw [hlsb, h21, hz] at hm
    rw [← h21]; omega
  have hb0 : W.testBit 0 = false := by
    rw [Nat.testBit_zero]; simp [hw2]
  have hbt : W.testBit t = true := by
    have hsplit : (2:Nat)^(t+1) = 2^t * 2 := by rw [Nat.pow_succ]
    have h2 : W % (2^t * 2) / 2^t = W / 2^t % 2 := Nat.mod_mul_right_div_self W (2^t) 2
    have h1 : W % 2^(t+1) / 2^t = 1 := by
      rw [hlsb]; exact Nat.div_self (Nat.two_pow_pos t)
    rw [hsplit] at h1
    have hd : W / 2^t % 2 = 1 := by omega
    have hb := and_one_testBit W t
    rw [Nat.shiftRight_eq_div_pow, Nat.and_one_is_mod, hd] at hb
    cases h : W.testBit t
    · rw [h] at hb; simp at hb
    · rfl
  rw [tau_spec, if_neg (by rw [hb0, hbt]; simp)]
  have hone : ((1:Nat) ||| 1 <<< t).testBit 0 = true := by
    rw [Nat.testBit_or]; simp
  have hres : (W ^^^ (1 ||| 1 <<< t)).testBit 0 = true := by
    rw [Nat.testBit_xor, hb0, hone]; simp
  rw [Nat.testBit_zero] at hres
  simpa using hres

/-! ## Tier 24: the residual factor of four -- a coboundary kills it

`W18` left `tau` SOUND but not complete: each `N`-block is exactly four `tau`-orbits. The
missing mechanism is `GL(3,2)` acting on bits 0,1,2 (identity above) -- order 168, transitive
on the seven nonzero low patterns, so it merges the four odd residues and closes `g`.

Why does it work with NO hypothesis on `W`, when `sigma` itself is not invariant? Because
`sigma` moves by a **coboundary**: `sigma (p x) (p y) = sigma x y * lam x * lam y * lam (x^^^y)`.
`Q` and `Q'` are each a product of FOUR `sigma`s over a coset square, and the six `lam` values
each occur exactly TWICE there, so every one of them squares away. That cancellation is what
this tier proves, `forall n` and for an arbitrary linear `p` and arbitrary sign `lam` -- so the
whole factor of four is reduced to the single `sigma`-level statement `hcob`, which `W19`
verifies (168/168, with a non-linear null control) but which is NOT proven here. -/

/-- `Nat.xor_left_comm` does not exist in this toolchain; supply it so `simp` can
    AC-normalise the xor arguments. -/
private theorem xorLcomm (x y z : Nat) : x ^^^ (y ^^^ z) = y ^^^ (x ^^^ z) := by
  rw [← Nat.xor_assoc, ← Nat.xor_assoc, Nat.xor_comm x y]

private theorem xorCancelL (x y : Nat) : x ^^^ (x ^^^ y) = y := by
  rw [← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]

/-- **A coboundary in `sigma` is invisible to `Q`.** The six `lam` values occur twice each. -/
theorem Qgen_of_coboundary (m W a b : Nat) (p : Nat → Nat) (lam : Nat → Int)
    (hlin : ∀ x y, p (x ^^^ y) = p x ^^^ p y)
    (hpm : ∀ x, lam x = 1 ∨ lam x = -1)
    (hcob : ∀ x y, cdSigma (p x) (p y) m
              = cdSigma x y m * lam x * lam y * lam (x ^^^ y)) :
    Qgen (p W) (p a) (p b) m = Qgen W a b m := by
  have sq : ∀ x : Nat, lam x * lam x = 1 := by
    intro x; rcases hpm x with h | h <;> rw [h] <;> decide
  have h1 : (a ^^^ W) ^^^ (b ^^^ W) = a ^^^ b := by
    simp [Nat.xor_assoc, Nat.xor_comm, xorLcomm, xorCancelL]
  have h2 : a ^^^ (b ^^^ W) = (a ^^^ b) ^^^ W := by
    simp [Nat.xor_assoc, Nat.xor_comm, xorLcomm, xorCancelL]
  have h3 : (a ^^^ W) ^^^ b = (a ^^^ b) ^^^ W := by
    simp [Nat.xor_assoc, Nat.xor_comm, xorLcomm, xorCancelL]
  unfold Qgen
  rw [← hlin a W, ← hlin b W, hcob a b, hcob (a ^^^ W) (b ^^^ W), hcob a (b ^^^ W),
      hcob (a ^^^ W) b, h1, h2, h3]
  calc cdSigma a b m * lam a * lam b * lam (a ^^^ b)
        * (cdSigma (a ^^^ W) (b ^^^ W) m * lam (a ^^^ W) * lam (b ^^^ W) * lam (a ^^^ b))
        * (cdSigma a (b ^^^ W) m * lam a * lam (b ^^^ W) * lam ((a ^^^ b) ^^^ W))
        * (cdSigma (a ^^^ W) b m * lam (a ^^^ W) * lam b * lam ((a ^^^ b) ^^^ W))
      = cdSigma a b m * cdSigma (a ^^^ W) (b ^^^ W) m * cdSigma a (b ^^^ W) m
          * cdSigma (a ^^^ W) b m
        * ((lam a * lam a) * (lam b * lam b) * (lam (a ^^^ b) * lam (a ^^^ b))
           * (lam (a ^^^ W) * lam (a ^^^ W)) * (lam (b ^^^ W) * lam (b ^^^ W))
           * (lam ((a ^^^ b) ^^^ W) * lam ((a ^^^ b) ^^^ W))) := by ac_rfl
    _ = cdSigma a b m * cdSigma (a ^^^ W) (b ^^^ W) m * cdSigma a (b ^^^ W) m
          * cdSigma (a ^^^ W) b m := by
        rw [sq, sq, sq, sq, sq, sq]; simp

/-- The same for `Q'` -- the four factors are transposed but the `lam` multiset is identical. -/
theorem Qgen'_of_coboundary (m W a b : Nat) (p : Nat → Nat) (lam : Nat → Int)
    (hlin : ∀ x y, p (x ^^^ y) = p x ^^^ p y)
    (hpm : ∀ x, lam x = 1 ∨ lam x = -1)
    (hcob : ∀ x y, cdSigma (p x) (p y) m
              = cdSigma x y m * lam x * lam y * lam (x ^^^ y)) :
    Qgen' (p W) (p a) (p b) m = Qgen' W a b m := by
  have sq : ∀ x : Nat, lam x * lam x = 1 := by
    intro x; rcases hpm x with h | h <;> rw [h] <;> decide
  have h1 : (b ^^^ W) ^^^ (a ^^^ W) = a ^^^ b := by
    simp [Nat.xor_assoc, Nat.xor_comm, xorLcomm, xorCancelL]
  have h2 : (b ^^^ W) ^^^ a = (a ^^^ b) ^^^ W := by
    simp [Nat.xor_assoc, Nat.xor_comm, xorLcomm, xorCancelL]
  have h3 : (a ^^^ W) ^^^ b = (a ^^^ b) ^^^ W := by
    simp [Nat.xor_assoc, Nat.xor_comm, xorLcomm, xorCancelL]
  unfold Qgen'
  rw [← hlin a W, ← hlin b W, hcob a b, hcob (b ^^^ W) (a ^^^ W), hcob (b ^^^ W) a,
      hcob (a ^^^ W) b, h1, h2, h3]
  calc cdSigma a b m * lam a * lam b * lam (a ^^^ b)
        * (cdSigma (b ^^^ W) (a ^^^ W) m * lam (b ^^^ W) * lam (a ^^^ W) * lam (a ^^^ b))
        * (cdSigma (b ^^^ W) a m * lam (b ^^^ W) * lam a * lam ((a ^^^ b) ^^^ W))
        * (cdSigma (a ^^^ W) b m * lam (a ^^^ W) * lam b * lam ((a ^^^ b) ^^^ W))
      = cdSigma a b m * cdSigma (b ^^^ W) (a ^^^ W) m * cdSigma (b ^^^ W) a m
          * cdSigma (a ^^^ W) b m
        * ((lam a * lam a) * (lam b * lam b) * (lam (a ^^^ b) * lam (a ^^^ b))
           * (lam (a ^^^ W) * lam (a ^^^ W)) * (lam (b ^^^ W) * lam (b ^^^ W))
           * (lam ((a ^^^ b) ^^^ W) * lam ((a ^^^ b) ^^^ W))) := by ac_rfl
    _ = cdSigma a b m * cdSigma (b ^^^ W) (a ^^^ W) m * cdSigma (b ^^^ W) a m
          * cdSigma (a ^^^ W) b m := by
        rw [sq, sq, sq, sq, sq, sq]; simp

/-! ## Tier 25: the coboundary itself LIFTS -- level 3 decides every level

`W19` left one measured statement behind `g`: that `sigma` moves by a coboundary under the
low-block maps. This tier proves the ∀n half of it. `cdSigma`'s recursion strips the TOP bit
and recurses on the residues, while a map `p` confined to bits 0,1,2 commutes with that split
entirely -- it preserves the `≥ half` tests, the `= 0` tests, and the residues. So the
coboundary property is INHERITED from one level to the next, and the whole ∀n statement
collapses to a check at **level 3**, where everything is finite.

The four branches are exactly `R_ll`, `R_lu`, `R_ul`, `R_uu`, already in the tree. -/

theorem sigma_coboundary_up (p : Nat → Nat) (lam : Nat → Int)
    (hp0 : p 0 = 0) (hpne : ∀ x, x ≠ 0 → p x ≠ 0)
    (hlam0 : lam 0 = 1) (hpm : ∀ x, lam x = 1 ∨ lam x = -1)
    (hplt : ∀ k u, u < 2^(k+3) → p u < 2^(k+3))
    (hpseam : ∀ k u, u < 2^(k+3) → p (u + 2^(k+3)) = p u + 2^(k+3))
    (hlamseam : ∀ k u, lam (u + 2^(k+3)) = lam u)
    (hbase : ∀ x y, x < 2^3 → y < 2^3 →
        cdSigma (p x) (p y) 3 = cdSigma x y 3 * lam x * lam y * lam (x ^^^ y)) :
    ∀ k x y, x < 2^(k+3) → y < 2^(k+3) →
      cdSigma (p x) (p y) (k+3) = cdSigma x y (k+3) * lam x * lam y * lam (x ^^^ y) := by
  intro k
  induction k with
  | zero => exact hbase
  | succ k ih =>
    have sq : ∀ z : Nat, lam z * lam z = 1 := by
      intro z; rcases hpm z with h | h <;> rw [h] <;> decide
    have hhalf : (2:Nat)^(k+1+3) = 2^(k+3) + 2^(k+3) := by rw [Nat.pow_succ]; omega
    intro x y hx hy
    by_cases hxu : x < 2^(k+3)
    · by_cases hyu : y < 2^(k+3)
      · -- both low
        rw [R_ll (p x) (p y) (k+2) (hplt k x hxu) (hplt k y hyu), R_ll x y (k+2) hxu hyu]
        exact ih x y hxu hyu
      · -- x low, y high
        obtain ⟨v, rfl, hv⟩ : ∃ v, y = v + 2^(k+3) ∧ v < 2^(k+3) :=
          ⟨y - 2^(k+3), by omega, by omega⟩
        rw [hpseam k v hv, R_lu (p x) (p v) (k+2) (hplt k x hxu) (hplt k v hv),
            R_lu x v (k+2) hxu hv, xor_seam x v (k+2) hxu hv,
            hlamseam k v, hlamseam k (x ^^^ v), ih v x hv hxu,
            Nat.xor_comm v x]
        ac_rfl
    · obtain ⟨u, rfl, hu⟩ : ∃ u, x = u + 2^(k+3) ∧ u < 2^(k+3) :=
        ⟨x - 2^(k+3), by omega, by omega⟩
      by_cases hyu : y < 2^(k+3)
      · -- x high, y low
        rw [hpseam k u hu, R_ul (p u) (p y) (k+2) (hplt k u hu) (hplt k y hyu),
            R_ul u y (k+2) hu hyu, seam_xor_left u y (k+2) hu hyu,
            hlamseam k u, hlamseam k (u ^^^ y)]
        by_cases hy0 : y = 0
        · subst hy0
          rw [hp0, if_pos rfl, if_pos rfl, Nat.xor_zero, hlam0]
          simp [sq u]
        · rw [if_neg (hpne y hy0), if_neg hy0, ih u y hu hyu]
          simp only [Int.neg_mul]
      · -- both high
        obtain ⟨v, rfl, hv⟩ : ∃ v, y = v + 2^(k+3) ∧ v < 2^(k+3) :=
          ⟨y - 2^(k+3), by omega, by omega⟩
        rw [hpseam k u hu, hpseam k v hv,
            R_uu (p u) (p v) (k+2) (hplt k u hu) (hplt k v hv),
            R_uu u v (k+2) hu hv, xor_seam_cancel u v (k+2) hu hv,
            hlamseam k u, hlamseam k v]
        by_cases hv0 : v = 0
        · subst hv0
          rw [hp0, if_pos rfl, if_pos rfl, Nat.xor_zero, hlam0, Int.mul_one,
              Int.mul_assoc, sq u, Int.mul_one]
        · rw [if_neg (hpne v hv0), if_neg hv0, ih v u hv hu, Nat.xor_comm v u]
          ac_rfl

/-! ### The low-block maps, concretely -- and one generator fully closed

A map confined to bits 0,1,2 is `lowMap t x = 8 * (x / 8) + t (x % 8)`, and a sign confined
to them is `lowSign l x = l (x % 8)`. In that form every hypothesis of `sigma_coboundary_up`
is `omega`-arithmetic rather than bit-fiddling. Instantiating with the TRANSVECTION
`e₂ ↦ e₂ ⊕ e₀` (table `(0,1,2,3,5,4,7,6)`, `lam = -1` exactly on `{5,7}`) closes the
coboundary for that generator at EVERY level -- the level-3 base falls to `decide`. -/

def lowMap (t : Nat → Nat) (x : Nat) : Nat := 8 * (x / 8) + t (x % 8)
def lowSign (l : Nat → Int) (x : Nat) : Int := l (x % 8)

private theorem pow_k3 (k : Nat) : ∃ n, (2:Nat)^(k+3) = 8 * n :=
  ⟨2^k, by rw [Nat.pow_add]; simp [Nat.mul_comm]⟩

theorem lowMap_seam (t : Nat → Nat) (k u : Nat) :
    lowMap t (u + 2^(k+3)) = lowMap t u + 2^(k+3) := by
  obtain ⟨n, hn⟩ := pow_k3 k
  unfold lowMap
  rw [hn, Nat.add_mul_mod_self_left, Nat.add_mul_div_left _ _ (by omega : 0 < 8)]
  omega

theorem lowMap_lt (t : Nat → Nat) (ht : ∀ v, v < 8 → t v < 8) (k u : Nat)
    (hu : u < 2^(k+3)) : lowMap t u < 2^(k+3) := by
  obtain ⟨n, hn⟩ := pow_k3 k
  have h1 : t (u % 8) < 8 := ht _ (Nat.mod_lt _ (by omega))
  rw [hn] at hu ⊢
  unfold lowMap
  omega

theorem lowMap_zero (t : Nat → Nat) (h : t 0 = 0) : lowMap t 0 = 0 := by
  unfold lowMap; simp [h]

theorem lowMap_ne (t : Nat → Nat) (hz : ∀ v, v < 8 → t v = 0 → v = 0) (x : Nat)
    (hx : x ≠ 0) : lowMap t x ≠ 0 := by
  intro h
  unfold lowMap at h
  have h2 : t (x % 8) = 0 := by omega
  have := hz (x % 8) (Nat.mod_lt _ (by omega)) h2
  omega

theorem lowSign_seam (l : Nat → Int) (k u : Nat) :
    lowSign l (u + 2^(k+3)) = lowSign l u := by
  obtain ⟨n, hn⟩ := pow_k3 k
  unfold lowSign
  rw [hn, Nat.add_mul_mod_self_left]

/-- The transvection `e₂ ↦ e₂ ⊕ e₀` on the low block. -/
def tTrans : Nat → Nat := fun v => if v = 4 then 5 else if v = 5 then 4
  else if v = 6 then 7 else if v = 7 then 6 else v

/-- Its coboundary sign: `-1` exactly on `{5,7}`. -/
def lTrans : Nat → Int := fun v => if v = 5 ∨ v = 7 then -1 else 1

private theorem baseTrans : ∀ x < 8, ∀ y < 8,
    cdSigma (lowMap tTrans x) (lowMap tTrans y) 3
      = cdSigma x y 3 * lowSign lTrans x * lowSign lTrans y * lowSign lTrans (x ^^^ y) := by
  decide

/-- **The coboundary, PROVEN ∀n for a generator of `GL(3,2)`.** The level-3 base is `decide`;
    `sigma_coboundary_up` lifts it to every level. -/
theorem sigma_coboundary_trans :
    ∀ k x y, x < 2^(k+3) → y < 2^(k+3) →
      cdSigma (lowMap tTrans x) (lowMap tTrans y) (k+3)
        = cdSigma x y (k+3) * lowSign lTrans x * lowSign lTrans y
            * lowSign lTrans (x ^^^ y) :=
  sigma_coboundary_up (lowMap tTrans) (lowSign lTrans)
    (lowMap_zero tTrans (by decide))
    (lowMap_ne tTrans (by decide))
    (by unfold lowSign lTrans; decide)
    (by
      intro x
      unfold lowSign lTrans
      by_cases h : x % 8 = 5 ∨ x % 8 = 7
      · rw [if_pos h]; exact Or.inr rfl
      · rw [if_neg h]; exact Or.inl rfl)
    (lowMap_lt tTrans (by decide))
    (fun k u _ => lowMap_seam tTrans k u)
    (fun k u => lowSign_seam lTrans k u)
    (fun x y hx hy => baseTrans x hx y hy)

/-- The 7-cycle `e₀↦e₁, e₁↦e₂, e₂↦e₀⊕e₁` on the low block: table `(0,2,4,6,3,1,7,5)`. -/
def tCyc : Nat → Nat := fun v =>
  if v = 1 then 2 else if v = 2 then 4 else if v = 3 then 6
  else if v = 4 then 3 else if v = 5 then 1 else if v = 6 then 7
  else if v = 7 then 5 else 0

/-- Its coboundary sign: `-1` exactly on `{6,7}`. -/
def lCyc : Nat → Int := fun v => if v = 6 ∨ v = 7 then -1 else 1

private theorem baseCyc : ∀ x < 8, ∀ y < 8,
    cdSigma (lowMap tCyc x) (lowMap tCyc y) 3
      = cdSigma x y 3 * lowSign lCyc x * lowSign lCyc y * lowSign lCyc (x ^^^ y) := by
  decide

/-- **The coboundary for the OTHER generator of `GL(3,2)`, also ∀n.** With `tTrans` these two
    generate the whole group, and the coboundary property is closed under composition:
    if `sigma∘p` and `sigma∘q` each move by `lp`, `lq`, then `sigma∘(p∘q)` moves by
    `x ↦ lq x * lp (q x)`, because `q` is linear so `q (x ⊕ y) = q x ⊕ q y`. -/
theorem sigma_coboundary_cyc :
    ∀ k x y, x < 2^(k+3) → y < 2^(k+3) →
      cdSigma (lowMap tCyc x) (lowMap tCyc y) (k+3)
        = cdSigma x y (k+3) * lowSign lCyc x * lowSign lCyc y * lowSign lCyc (x ^^^ y) :=
  sigma_coboundary_up (lowMap tCyc) (lowSign lCyc)
    (lowMap_zero tCyc (by decide))
    (lowMap_ne tCyc (by decide))
    (by unfold lowSign lCyc; decide)
    (by
      intro x
      unfold lowSign lCyc
      by_cases h : x % 8 = 6 ∨ x % 8 = 7
      · rw [if_pos h]; exact Or.inr rfl
      · rw [if_neg h]; exact Or.inl rfl)
    (lowMap_lt tCyc (by decide))
    (fun k u _ => lowMap_seam tCyc k u)
    (fun k u => lowSign_seam lCyc k u)
    (fun x y hx hy => baseCyc x hx y hy)

/-! ### `lowMap` is F2-linear, and the coboundary closes under composition

The last structural gap: `lowMap`'s linearity, which is what lets two coboundaries compose.
It follows from four core bit facts (`shiftRight_xor_distrib`, `shiftLeft_xor_distrib`,
`testBit_mod_two_pow`, `two_pow_add_eq_or_of_lt`) once `8 * a + b` with `b < 8` is recognised
as a disjoint `xor`. -/

private theorem xor_mod8 (x y : Nat) : (x ^^^ y) % 8 = (x % 8) ^^^ (y % 8) := by
  have h8 : (8:Nat) = 2^3 := rfl
  apply Nat.eq_of_testBit_eq
  intro i
  rw [h8, Nat.testBit_mod_two_pow, Nat.testBit_xor, Nat.testBit_xor,
      Nat.testBit_mod_two_pow, Nat.testBit_mod_two_pow]
  by_cases hi : i < 3 <;> simp [hi]

private theorem xor_div8 (x y : Nat) : (x ^^^ y) / 8 = (x / 8) ^^^ (y / 8) := by
  have h : ∀ z : Nat, z / 8 = z >>> 3 := fun z => (Nat.shiftRight_eq_div_pow z 3).symm
  rw [h, h, h, Nat.shiftRight_xor_distrib]

private theorem mul8_xor (a b : Nat) : 8 * (a ^^^ b) = 8 * a ^^^ 8 * b := by
  have h : ∀ z : Nat, 8 * z = z <<< 3 := by
    intro z
    have h2 : (2:Nat)^3 = 8 := rfl
    rw [Nat.shiftLeft_eq, h2]; omega
  rw [h, h, h, Nat.shiftLeft_xor_distrib]

private theorem add8_xor (a b : Nat) (hb : b < 8) : 8 * a + b = 8 * a ^^^ b := by
  have h8 : (2:Nat)^3 = 8 := rfl
  have hor : 2^3 * a + b = 2^3 * a ||| b :=
    Nat.two_pow_add_eq_or_of_lt (by rw [h8]; exact hb) a
  rw [← h8, hor]
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.testBit_or, Nat.testBit_xor]
  by_cases hi : i < 3
  · have hz : (2^3 * a).testBit i = false := by
      have : (2:Nat)^3 * a = a <<< 3 := by rw [Nat.shiftLeft_eq]; omega
      rw [this, Nat.testBit_shiftLeft]
      simp [Nat.not_le.mpr hi]
    rw [hz]; simp
  · have hz : b.testBit i = false := by
      apply Nat.testBit_lt_two_pow
      have : (2:Nat)^3 ≤ 2^i := Nat.pow_le_pow_right (by omega) (by omega)
      omega
    rw [hz]; simp

/-- **`lowMap t` is F2-linear when `t` is.** -/
theorem lowMap_lin (t : Nat → Nat) (ht8 : ∀ v, v < 8 → t v < 8)
    (htlin : ∀ u v, u < 8 → v < 8 → t (u ^^^ v) = t u ^^^ t v) (x y : Nat) :
    lowMap t (x ^^^ y) = lowMap t x ^^^ lowMap t y := by
  have hx8 : x % 8 < 8 := Nat.mod_lt _ (by omega)
  have hy8 : y % 8 < 8 := Nat.mod_lt _ (by omega)
  have h8 : (2:Nat)^3 = 8 := rfl
  have hxy : t (x % 8) ^^^ t (y % 8) < 8 := by
    have := Nat.xor_lt_two_pow (n := 3) (by rw [h8]; exact ht8 _ hx8) (by rw [h8]; exact ht8 _ hy8)
    omega
  unfold lowMap
  rw [xor_div8, xor_mod8, htlin _ _ hx8 hy8,
      add8_xor _ _ hxy, add8_xor _ _ (ht8 _ hx8), add8_xor _ _ (ht8 _ hy8), mul8_xor]
  simp [Nat.xor_assoc, Nat.xor_comm, xorLcomm]

/-- `lowMap` composes on the tables. -/
theorem lowMap_comp (t1 t2 : Nat → Nat) (h2 : ∀ v, v < 8 → t2 v < 8) (x : Nat) :
    lowMap t1 (lowMap t2 x) = lowMap (fun v => t1 (t2 v)) x := by
  have hx8 : x % 8 < 8 := Nat.mod_lt _ (by omega)
  have hlt : t2 (x % 8) < 8 := h2 _ hx8
  have hd : (8 * (x / 8) + t2 (x % 8)) / 8 = x / 8 := by omega
  have hm : (8 * (x / 8) + t2 (x % 8)) % 8 = t2 (x % 8) := by omega
  unfold lowMap
  rw [hd, hm]

/-- `lowSign` reads the composite through `lowMap`. -/
theorem lowSign_comp (l1 : Nat → Int) (t2 : Nat → Nat) (h2 : ∀ v, v < 8 → t2 v < 8) (x : Nat) :
    lowSign l1 (lowMap t2 x) = l1 (t2 (x % 8)) := by
  have hx8 : x % 8 < 8 := Nat.mod_lt _ (by omega)
  have hlt : t2 (x % 8) < 8 := h2 _ hx8
  have hm : (8 * (x / 8) + t2 (x % 8)) % 8 = t2 (x % 8) := by omega
  unfold lowSign lowMap
  rw [hm]

/-- **The coboundary is closed under composition**: `lam_{p∘q} x = lam_q x * lam_p (q x)`. -/
theorem sigma_coboundary_comp (t1 t2 : Nat → Nat) (l1 l2 : Nat → Int)
    (h2 : ∀ v, v < 8 → t2 v < 8)
    (h2lin : ∀ u v, u < 8 → v < 8 → t2 (u ^^^ v) = t2 u ^^^ t2 v)
    (H1 : ∀ k x y, x < 2^(k+3) → y < 2^(k+3) →
        cdSigma (lowMap t1 x) (lowMap t1 y) (k+3)
          = cdSigma x y (k+3) * lowSign l1 x * lowSign l1 y * lowSign l1 (x ^^^ y))
    (H2 : ∀ k x y, x < 2^(k+3) → y < 2^(k+3) →
        cdSigma (lowMap t2 x) (lowMap t2 y) (k+3)
          = cdSigma x y (k+3) * lowSign l2 x * lowSign l2 y * lowSign l2 (x ^^^ y)) :
    ∀ k x y, x < 2^(k+3) → y < 2^(k+3) →
      cdSigma (lowMap (fun v => t1 (t2 v)) x) (lowMap (fun v => t1 (t2 v)) y) (k+3)
        = cdSigma x y (k+3)
            * lowSign (fun v => l2 v * l1 (t2 v)) x
            * lowSign (fun v => l2 v * l1 (t2 v)) y
            * lowSign (fun v => l2 v * l1 (t2 v)) (x ^^^ y) := by
  intro k x y hx hy
  rw [← lowMap_comp t1 t2 h2 x, ← lowMap_comp t1 t2 h2 y,
      H1 k (lowMap t2 x) (lowMap t2 y) (lowMap_lt t2 h2 k x hx) (lowMap_lt t2 h2 k y hy),
      ← lowMap_lin t2 h2 h2lin x y,
      lowSign_comp l1 t2 h2 x, lowSign_comp l1 t2 h2 y, lowSign_comp l1 t2 h2 (x ^^^ y),
      H2 k x y hx hy]
  simp only [lowSign]
  ac_rfl

/-- The class of low-block maps whose coboundary is a THEOREM: the two `GL(3,2)` generators,
    closed under composition. `W20` checks that this class is exactly `GL(3,2)` -- closure of
    the two tables gives 168 elements. -/
inductive LowCob : (Nat → Nat) → (Nat → Int) → Prop where
  | trans : LowCob tTrans lTrans
  | cyc : LowCob tCyc lCyc
  | comp {t1 t2 : Nat → Nat} {l1 l2 : Nat → Int} :
      LowCob t1 l1 → LowCob t2 l2 →
      LowCob (fun v => t1 (t2 v)) (fun v => l2 v * l1 (t2 v))

private theorem tTrans_lin : ∀ u < 8, ∀ v < 8, tTrans (u ^^^ v) = tTrans u ^^^ tTrans v := by
  decide

private theorem tCyc_lin : ∀ u < 8, ∀ v < 8, tCyc (u ^^^ v) = tCyc u ^^^ tCyc v := by
  decide

theorem lowCob_lt {t l} (h : LowCob t l) : ∀ v, v < 8 → t v < 8 := by
  induction h with
  | trans => decide
  | cyc => decide
  | comp _ _ ih1 ih2 => exact fun v hv => ih1 _ (ih2 v hv)

theorem lowCob_lin {t l} (h : LowCob t l) :
    ∀ u v, u < 8 → v < 8 → t (u ^^^ v) = t u ^^^ t v := by
  induction h with
  | trans => exact fun u v hu hv => tTrans_lin u hu v hv
  | cyc => exact fun u v hu hv => tCyc_lin u hu v hv
  | comp h1 h2 ih1 ih2 =>
      intro u v hu hv
      show _ = _
      simp only []
      rw [ih2 u v hu hv, ih1 _ _ (lowCob_lt h2 u hu) (lowCob_lt h2 v hv)]

theorem lowCob_pm {t l} (h : LowCob t l) : ∀ v, l v = 1 ∨ l v = -1 := by
  induction h with
  | trans =>
      intro v
      unfold lTrans
      by_cases hv : v = 5 ∨ v = 7
      · rw [if_pos hv]; exact Or.inr rfl
      · rw [if_neg hv]; exact Or.inl rfl
  | cyc =>
      intro v
      unfold lCyc
      by_cases hv : v = 6 ∨ v = 7
      · rw [if_pos hv]; exact Or.inr rfl
      · rw [if_neg hv]; exact Or.inl rfl
  | comp _ _ ih1 ih2 =>
      intro v
      show _ = _ ∨ _ = _
      simp only []
      rcases ih2 v with h2 | h2 <;> rcases ih1 (_) with h1 | h1 <;>
        rw [h1, h2] <;> decide

/-- **Every map in the generated class carries the coboundary, at every level.** -/
theorem lowCob_sigma {t l} (h : LowCob t l) :
    ∀ k x y, x < 2^(k+3) → y < 2^(k+3) →
      cdSigma (lowMap t x) (lowMap t y) (k+3)
        = cdSigma x y (k+3) * lowSign l x * lowSign l y * lowSign l (x ^^^ y) := by
  induction h with
  | trans => exact sigma_coboundary_trans
  | cyc => exact sigma_coboundary_cyc
  | comp h1 h2 ih1 ih2 =>
      exact sigma_coboundary_comp _ _ _ _ (lowCob_lt h2) (lowCob_lin h2) ih1 ih2

/-- Bounded form of `Qgen'_of_coboundary` -- the coboundary is only available on the box. -/
theorem Qgen'_of_coboundary_lt (m W a b : Nat) (p : Nat → Nat) (lam : Nat → Int)
    (hW : W < 2^m) (ha : a < 2^m) (hb : b < 2^m)
    (hlin : ∀ x y, p (x ^^^ y) = p x ^^^ p y)
    (hpm : ∀ x, lam x = 1 ∨ lam x = -1)
    (hcob : ∀ x y, x < 2^m → y < 2^m →
        cdSigma (p x) (p y) m = cdSigma x y m * lam x * lam y * lam (x ^^^ y)) :
    Qgen' (p W) (p a) (p b) m = Qgen' W a b m := by
  have sq : ∀ x : Nat, lam x * lam x = 1 := by
    intro x; rcases hpm x with h | h <;> rw [h] <;> decide
  have haW : a ^^^ W < 2^m := Nat.xor_lt_two_pow ha hW
  have hbW : b ^^^ W < 2^m := Nat.xor_lt_two_pow hb hW
  have h1 : (b ^^^ W) ^^^ (a ^^^ W) = a ^^^ b := by
    simp [Nat.xor_assoc, Nat.xor_comm, xorLcomm, xorCancelL]
  have h2 : (b ^^^ W) ^^^ a = (a ^^^ b) ^^^ W := by
    simp [Nat.xor_assoc, Nat.xor_comm, xorLcomm, xorCancelL]
  have h3 : (a ^^^ W) ^^^ b = (a ^^^ b) ^^^ W := by
    simp [Nat.xor_assoc, Nat.xor_comm, xorLcomm, xorCancelL]
  unfold Qgen'
  rw [← hlin a W, ← hlin b W, hcob a b ha hb, hcob (b ^^^ W) (a ^^^ W) hbW haW,
      hcob (b ^^^ W) a hbW ha, hcob (a ^^^ W) b haW hb, h1, h2, h3]
  calc cdSigma a b m * lam a * lam b * lam (a ^^^ b)
        * (cdSigma (b ^^^ W) (a ^^^ W) m * lam (b ^^^ W) * lam (a ^^^ W) * lam (a ^^^ b))
        * (cdSigma (b ^^^ W) a m * lam (b ^^^ W) * lam a * lam ((a ^^^ b) ^^^ W))
        * (cdSigma (a ^^^ W) b m * lam (a ^^^ W) * lam b * lam ((a ^^^ b) ^^^ W))
      = cdSigma a b m * cdSigma (b ^^^ W) (a ^^^ W) m * cdSigma (b ^^^ W) a m
          * cdSigma (a ^^^ W) b m
        * ((lam a * lam a) * (lam b * lam b) * (lam (a ^^^ b) * lam (a ^^^ b))
           * (lam (a ^^^ W) * lam (a ^^^ W)) * (lam (b ^^^ W) * lam (b ^^^ W))
           * (lam ((a ^^^ b) ^^^ W) * lam ((a ^^^ b) ^^^ W))) := by ac_rfl
    _ = cdSigma a b m * cdSigma (b ^^^ W) (a ^^^ W) m * cdSigma (b ^^^ W) a m
          * cdSigma (a ^^^ W) b m := by
        rw [sq, sq, sq, sq, sq, sq]; simp

/-- **THE PAYOFF, ∀n: `Q'` is invariant under every map in the class.** With `W20`'s check
    that the class IS `GL(3,2)`, this is the whole residual factor of four. -/
theorem Qgen'_lowCob {t l} (h : LowCob t l) (k W a b : Nat)
    (hW : W < 2^(k+3)) (ha : a < 2^(k+3)) (hb : b < 2^(k+3)) :
    Qgen' (lowMap t W) (lowMap t a) (lowMap t b) (k+3) = Qgen' W a b (k+3) :=
  Qgen'_of_coboundary_lt (k+3) W a b (lowMap t) (lowSign l) hW ha hb
    (lowMap_lin t (lowCob_lt h) (lowCob_lin h))
    (fun x => lowCob_pm h (x % 8))
    (lowCob_sigma h k)

/-! ## Tier 27: the counting step

The last gap. `Qgen'_lowCob` is POINTWISE; the invariant `g` is about the COUNT `N(m,W)`.
Going from one to the other is a reindexing argument, which normally means `Finset` -- absent
here. It is not needed: a plain recursive sum `sumLt` plus the fact that `lowMap t` permutes
each block of eight, and the seam-splitting `sumLt_add`, do the whole job. -/

def sumLt : Nat → (Nat → Nat) → Nat
  | 0, _ => 0
  | (n+1), f => sumLt n f + f n

theorem sumLt_congr (n : Nat) (f g : Nat → Nat) (h : ∀ i, i < n → f i = g i) :
    sumLt n f = sumLt n g := by
  induction n with
  | zero => rfl
  | succ n ih => rw [sumLt, sumLt, ih (fun i hi => h i (by omega)), h n (by omega)]

theorem sumLt_add (n m : Nat) (f : Nat → Nat) :
    sumLt (n + m) f = sumLt n f + sumLt m (fun i => f (n + i)) := by
  induction m with
  | zero => rfl
  | succ m ih =>
      have e : n + (m + 1) = (n + m) + 1 := by omega
      rw [e, sumLt, ih, sumLt]
      omega

/-- The 8-term block sum is invariant under every table in the class. -/
theorem sum8_perm {t l} (h : LowCob t l) : ∀ f : Nat → Nat,
    f (t 0) + f (t 1) + f (t 2) + f (t 3) + f (t 4) + f (t 5) + f (t 6) + f (t 7)
      = f 0 + f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 := by
  induction h with
  | trans => intro f; unfold tTrans; simp; omega
  | cyc => intro f; unfold tCyc; simp; omega
  | @comp t1 t2 l1 l2 _ _ ih1 ih2 =>
      intro f; exact (ih2 (fun v => f (t1 v))).trans (ih1 f)

theorem lowCob_z {t l} (h : LowCob t l) : ∀ v, v < 8 → t v = 0 → v = 0 := by
  induction h with
  | trans => decide
  | cyc => decide
  | comp h1 h2 ih1 ih2 =>
      intro v hv hz
      exact ih2 v hv (ih1 _ (lowCob_lt h2 v hv) hz)

theorem lowCob_t0 {t l} (h : LowCob t l) : t 0 = 0 := by
  have e := lowCob_lin h 0 0 (by omega) (by omega)
  simp [Nat.xor_self] at e
  exact e

theorem lowCob_inj8 {t l} (h : LowCob t l) :
    ∀ u v, u < 8 → v < 8 → t u = t v → u = v := by
  intro u v hu hv he
  have h8 : (2:Nat)^3 = 8 := rfl
  have hlt : u ^^^ v < 8 := by
    have := Nat.xor_lt_two_pow (n := 3) (by rw [h8]; exact hu) (by rw [h8]; exact hv)
    omega
  have hz : t (u ^^^ v) = 0 := by rw [lowCob_lin h u v hu hv, he, Nat.xor_self]
  exact xor_zero_eq u v (lowCob_z h (u ^^^ v) hlt hz)

theorem lowMap_inj {t l} (h : LowCob t l) (x y : Nat) (hxy : lowMap t x = lowMap t y) :
    x = y := by
  by_cases hne : x ^^^ y = 0
  · exact xor_zero_eq x y hne
  · exfalso
    apply lowMap_ne t (lowCob_z h) (x ^^^ y) hne
    rw [lowMap_lin t (lowCob_lt h) (lowCob_lin h) x y, hxy, Nat.xor_self]

/-- **Reindexing a bounded sum by `lowMap t` changes nothing.** Induction on the level: the
    base is the 8-block permutation, the step is `lowMap_seam` plus `sumLt_add`. -/
theorem sumLt_lowMap {t l} (h : LowCob t l) :
    ∀ k f, sumLt (2^(k+3)) (fun x => f (lowMap t x)) = sumLt (2^(k+3)) f := by
  intro k
  induction k with
  | zero =>
      intro f
      have e8 : (2:Nat)^(0+3) = 8 := rfl
      have hl : ∀ i, i < 8 → lowMap t i = t i := by
        intro i hi
        unfold lowMap
        have h1 : i / 8 = 0 := by omega
        have h2 : i % 8 = i := by omega
        rw [h1, h2]
        omega
      rw [e8]
      show sumLt 8 (fun x => f (lowMap t x)) = sumLt 8 f
      rw [sumLt_congr 8 _ (fun x => f (t x)) (fun i hi => by rw [hl i hi])]
      have hs := sum8_perm h f
      simp only [sumLt]
      omega
  | succ k ih =>
      intro f
      have e : (2:Nat)^(k+1+3) = 2^(k+3) + 2^(k+3) := by rw [Nat.pow_succ]; omega
      rw [e, sumLt_add, sumLt_add]
      have h2 : sumLt (2^(k+3)) (fun i => f (lowMap t (2^(k+3) + i)))
              = sumLt (2^(k+3)) (fun i => f (2^(k+3) + i)) := by
        rw [sumLt_congr (2^(k+3)) _ (fun i => (fun x => f (x + 2^(k+3))) (lowMap t i))
              (fun i _ => by
                have : 2^(k+3) + i = i + 2^(k+3) := by omega
                rw [this, lowMap_seam t k i])]
        rw [ih (fun x => f (x + 2^(k+3)))]
        exact sumLt_congr _ _ _ (fun i _ => by rw [Nat.add_comm])
      rw [ih f, h2]

/-- The resonance count, as a plain double sum. -/
def Ncnt (W m : Nat) : Nat :=
  sumLt (2^m) (fun a => sumLt (2^m) (fun b =>
    if a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b ∧ Qgen' W a b m = -1 then 1 else 0))

/-- **THE COUNTING STEP, ∀n.** `N` is invariant under every map in the class -- so the
    invariant `g` is now proven end to end, from `sigma`'s coboundary to the count. -/
theorem Ncnt_lowCob {t l} (h : LowCob t l) (k W : Nat) (hW : W < 2^(k+3)) :
    Ncnt (lowMap t W) (k+3) = Ncnt W (k+3) := by
  unfold Ncnt
  rw [← sumLt_lowMap h k (fun a => sumLt (2^(k+3)) (fun b =>
        if a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b ∧ Qgen' (lowMap t W) a b (k+3) = -1 then 1 else 0))]
  apply sumLt_congr
  intro a ha
  rw [← sumLt_lowMap h k (fun b =>
        if lowMap t a ≠ 0 ∧ b ≠ 0 ∧ lowMap t a ≠ b
           ∧ Qgen' (lowMap t W) (lowMap t a) b (k+3) = -1 then 1 else 0)]
  apply sumLt_congr
  intro b hb
  have hQ := Qgen'_lowCob h k W a b hW ha hb
  have hz0 : lowMap t 0 = 0 := lowMap_zero t (lowCob_t0 h)
  have hiff : (lowMap t a ≠ 0 ∧ lowMap t b ≠ 0 ∧ lowMap t a ≠ lowMap t b
                ∧ Qgen' (lowMap t W) (lowMap t a) (lowMap t b) (k+3) = -1)
            ↔ (a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b ∧ Qgen' W a b (k+3) = -1) := by
    rw [hQ]
    constructor
    · rintro ⟨p1, p2, p3, p4⟩
      exact ⟨fun hzz => p1 (by rw [hzz]; exact hz0),
             fun hzz => p2 (by rw [hzz]; exact hz0),
             fun hzz => p3 (by rw [hzz]), p4⟩
    · rintro ⟨p1, p2, p3, p4⟩
      exact ⟨lowMap_ne t (lowCob_z h) a p1, lowMap_ne t (lowCob_z h) b p2,
             fun hzz => p3 (lowMap_inj h a b hzz), p4⟩
  by_cases hc : a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b ∧ Qgen' W a b (k+3) = -1
  · rw [if_pos (hiff.mpr hc), if_pos hc]
  · rw [if_neg (fun hx => hc (hiff.mp hx)), if_neg hc]

section GLClosure
-- scoped so the raised limit does not leak past this block
set_option maxRecDepth 100000

/-! ### `LowCob` IS `GL(3,2)`, both directions

Soundness is the easy half and is already assembled (`lowCob_lt`, `lowCob_lin`, `lowCob_t0`,
`lowCob_inj8`): every member restricts to an injective linear endomorphism of `F2^3`, which is
exactly an element of `GL(3,2)`. Completeness is the finite half `W20` had been carrying: for
each of the 168 elements, an explicit WORD in the two generators, found by breadth-first search
(longest word: 12). -/

/-- A linear map on the low block, from the images of `e0, e1, e2`. -/
def linMap (a b c : Nat) : Nat → Nat := fun v =>
  (if v % 2 = 1 then a else 0) ^^^ (if v / 2 % 2 = 1 then b else 0)
    ^^^ (if v / 4 % 2 = 1 then c else 0)

/-- `(a,b,c)` are an `F2`-basis of the low block: exactly the `GL(3,2)` condition. -/
def glIndep (a b c : Nat) : Bool :=
  decide (a < 8) && decide (b < 8) && decide (c < 8) &&
  decide (a ≠ 0) && decide (b ≠ 0) && decide (c ≠ 0) &&
  decide (a ≠ b) && decide (a ≠ c) && decide (b ≠ c) && decide (a ^^^ b ≠ c)

/-- SOUNDNESS: every member of the class is an element of `GL(3,2)`. -/
theorem lowCob_isGL {t l} (h : LowCob t l) :
    (∀ v, v < 8 → t v < 8) ∧ t 0 = 0
      ∧ (∀ u v, u < 8 → v < 8 → t (u ^^^ v) = t u ^^^ t v)
      ∧ (∀ u v, u < 8 → v < 8 → t u = t v → u = v) :=
  ⟨lowCob_lt h, lowCob_t0 h, lowCob_lin h, lowCob_inj8 h⟩

def glList : List (Nat × Nat × Nat) :=
  [(1,2,4), (1,2,5), (1,2,6), (1,2,7), (1,3,4), (1,3,5), (1,3,6), (1,3,7), (1,4,2), (1,4,3), (1
   ,4,6), (1,4,7), (1,5,2), (1,5,3), (1,5,6), (1,5,7), (1,6,2), (1,6,3), (1,6,4), (1,6,5), (1,7
   ,2), (1,7,3), (1,7,4), (1,7,5), (2,1,4), (2,1,5), (2,1,6), (2,1,7), (2,3,4), (2,3,5), (2,3,6
   ), (2,3,7), (2,4,1), (2,4,3), (2,4,5), (2,4,7), (2,5,1), (2,5,3), (2,5,4), (2,5,6), (2,6,1),
    (2,6,3), (2,6,5), (2,6,7), (2,7,1), (2,7,3), (2,7,4), (2,7,6), (3,1,4), (3,1,5), (3,1,6), (
   3,1,7), (3,2,4), (3,2,5), (3,2,6), (3,2,7), (3,4,1), (3,4,2), (3,4,5), (3,4,6), (3,5,1), (3,
   5,2), (3,5,4), (3,5,7), (3,6,1), (3,6,2), (3,6,4), (3,6,7), (3,7,1), (3,7,2), (3,7,5), (3,7,
   6), (4,1,2), (4,1,3), (4,1,6), (4,1,7), (4,2,1), (4,2,3), (4,2,5), (4,2,7), (4,3,1), (4,3,2)
   , (4,3,5), (4,3,6), (4,5,2), (4,5,3), (4,5,6), (4,5,7), (4,6,1), (4,6,3), (4,6,5), (4,6,7), 
   (4,7,1), (4,7,2), (4,7,5), (4,7,6), (5,1,2), (5,1,3), (5,1,6), (5,1,7), (5,2,1), (5,2,3), (5
   ,2,4), (5,2,6), (5,3,1), (5,3,2), (5,3,4), (5,3,7), (5,4,2), (5,4,3), (5,4,6), (5,4,7), (5,6
   ,1), (5,6,2), (5,6,4), (5,6,7), (5,7,1), (5,7,3), (5,7,4), (5,7,6), (6,1,2), (6,1,3), (6,1,4
   ), (6,1,5), (6,2,1), (6,2,3), (6,2,5), (6,2,7), (6,3,1), (6,3,2), (6,3,4), (6,3,7), (6,4,1),
    (6,4,3), (6,4,5), (6,4,7), (6,5,1), (6,5,2), (6,5,4), (6,5,7), (6,7,2), (6,7,3), (6,7,4), (
   6,7,5), (7,1,2), (7,1,3), (7,1,4), (7,1,5), (7,2,1), (7,2,3), (7,2,4), (7,2,6), (7,3,1), (7,
   3,2), (7,3,5), (7,3,6), (7,4,1), (7,4,2), (7,4,5), (7,4,6), (7,5,1), (7,5,3), (7,5,4), (7,5,
   6), (7,6,2), (7,6,3), (7,6,4), (7,6,5)]

def glTable (i : Nat) : Nat × Nat × Nat := glList.getD i (0, 0, 0)

/-- The 168 listed triples are EXACTLY the `GL(3,2)` bases: each is independent, ... -/
theorem glList_indep : ∀ i, i < 168 →
    glIndep (glTable i).1 (glTable i).2.1 (glTable i).2.2 = true := by decide

/-- ... and every independent triple is listed, at a computable index. -/
def glIdx (a b c : Nat) : Nat := glList.findIdx (fun p => p == (a, b, c))

theorem glIdx_lt : ∀ a < 8, ∀ b < 8, ∀ c < 8,
    glIndep a b c = true → glIdx a b c < 168 := by decide

theorem glIdx_eq : ∀ a < 8, ∀ b < 8, ∀ c < 8, glIndep a b c = true →
    (glTable (glIdx a b c)).1 = a ∧ (glTable (glIdx a b c)).2.1 = b
      ∧ (glTable (glIdx a b c)).2.2 = c := by decide

/-- COMPLETENESS: every element of `GL(3,2)` is realised by a word in the two generators.
    The words were found by breadth-first search; the longest has length 12. -/
theorem lowCob_covers : ∀ i, i < 168 →
    ∃ t l, LowCob t l ∧ ∀ v, v < 8 →
      t v = linMap (glTable i).1 (glTable i).2.1 (glTable i).2.2 v := by
  intro i hi
  match i, hi with
  | 0, _ => exact ⟨_, _, (LowCob.comp LowCob.trans LowCob.trans)
, by decide⟩
  | 1, _ => exact ⟨_, _, LowCob.trans
, by decide⟩
  | 2, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc)))))))
, by decide⟩
  | 3, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans))))))))
, by decide⟩
  | 4, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc)))))))
, by decide⟩
  | 5, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))))))
, by decide⟩
  | 6, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))))))))
, by decide⟩
  | 7, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc)))))))))
, by decide⟩
  | 8, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc)))))))
, by decide⟩
  | 9, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))))))
, by decide⟩
  | 10, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))))))
, by decide⟩
  | 11, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc)))))))
, by decide⟩
  | 12, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))))))))
, by decide⟩
  | 13, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))))))
, by decide⟩
  | 14, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))))))
, by decide⟩
  | 15, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))))
, by decide⟩
  | 16, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc)))))))))
, by decide⟩
  | 17, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc)))))))))
, by decide⟩
  | 18, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))
, by decide⟩
  | 19, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc)))
, by decide⟩
  | 20, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc))))))))
, by decide⟩
  | 21, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))))))))
, by decide⟩
  | 22, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc))))
, by decide⟩
  | 23, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))
, by decide⟩
  | 24, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))))))
, by decide⟩
  | 25, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))))))
, by decide⟩
  | 26, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))))))))
, by decide⟩
  | 27, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc)))))))
, by decide⟩
  | 28, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))))))))
, by decide⟩
  | 29, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))))))))
, by decide⟩
  | 30, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))))))
, by decide⟩
  | 31, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))))))
, by decide⟩
  | 32, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc LowCob.trans)
, by decide⟩
  | 33, _ => exact ⟨_, _, LowCob.cyc
, by decide⟩
  | 34, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans)))))))))
, by decide⟩
  | 35, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc))))))))
, by decide⟩
  | 36, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans))
, by decide⟩
  | 37, _ => exact ⟨_, _, (LowCob.comp LowCob.trans LowCob.cyc)
, by decide⟩
  | 38, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc)))))))))
, by decide⟩
  | 39, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc)))))))))
, by decide⟩
  | 40, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc)))))
, by decide⟩
  | 41, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))))
, by decide⟩
  | 42, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))))))))))
, by decide⟩
  | 43, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))))))))
, by decide⟩
  | 44, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc))))
, by decide⟩
  | 45, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))
, by decide⟩
  | 46, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc))))))))))
, by decide⟩
  | 47, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))))))))))
, by decide⟩
  | 48, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc))))))
, by decide⟩
  | 49, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc)))))))
, by decide⟩
  | 50, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))))))
, by decide⟩
  | 51, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))))
, by decide⟩
  | 52, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans))))
, by decide⟩
  | 53, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans)))))
, by decide⟩
  | 54, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc))))
, by decide⟩
  | 55, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc)))
, by decide⟩
  | 56, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))))))
, by decide⟩
  | 57, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))))))))
, by decide⟩
  | 58, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc))))))))
, by decide⟩
  | 59, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))))))
, by decide⟩
  | 60, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc)))))))))
, by decide⟩
  | 61, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc)))))))))
, by decide⟩
  | 62, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc)))))))
, by decide⟩
  | 63, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))))))
, by decide⟩
  | 64, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc))))))))))
, by decide⟩
  | 65, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans)))))))))))
, by decide⟩
  | 66, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))
, by decide⟩
  | 67, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))
, by decide⟩
  | 68, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc))))))))))
, by decide⟩
  | 69, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))))))))))
, by decide⟩
  | 70, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))
, by decide⟩
  | 71, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc)))
, by decide⟩
  | 72, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans)))
, by decide⟩
  | 73, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc)))))))
, by decide⟩
  | 74, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc))
, by decide⟩
  | 75, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans))))))))
, by decide⟩
  | 76, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc)))))))
, by decide⟩
  | 77, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))))))))
, by decide⟩
  | 78, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))))))
, by decide⟩
  | 79, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc)))))))))
, by decide⟩
  | 80, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans))))))))))
, by decide⟩
  | 81, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))
, by decide⟩
  | 82, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc)))))))))
, by decide⟩
  | 83, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc LowCob.cyc)
, by decide⟩
  | 84, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc)))))
, by decide⟩
  | 85, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))))))))
, by decide⟩
  | 86, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))))
, by decide⟩
  | 87, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc)))))))))
, by decide⟩
  | 88, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc)))))))))
, by decide⟩
  | 89, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))))))
, by decide⟩
  | 90, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc)))))))))
, by decide⟩
  | 91, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))))
, by decide⟩
  | 92, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc))))))))
, by decide⟩
  | 93, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc))))))
, by decide⟩
  | 94, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))))))))
, by decide⟩
  | 95, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))))
, by decide⟩
  | 96, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans))))
, by decide⟩
  | 97, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc))))))
, by decide⟩
  | 98, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans)))))))
, by decide⟩
  | 99, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc)))
, by decide⟩
  | 100, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))))))
, by decide⟩
  | 101, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))))))))
, by decide⟩
  | 102, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))))
, by decide⟩
  | 103, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc))))))))))
, by decide⟩
  | 104, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans)))))))))))
, by decide⟩
  | 105, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))
, by decide⟩
  | 106, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc))))))))))
, by decide⟩
  | 107, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc))
, by decide⟩
  | 108, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc))))))
, by decide⟩
  | 109, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))))))
, by decide⟩
  | 110, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))))))))
, by decide⟩
  | 111, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))))
, by decide⟩
  | 112, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc)))))))
, by decide⟩
  | 113, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc))))))
, by decide⟩
  | 114, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc)))))))
, by decide⟩
  | 115, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans)))))))
, by decide⟩
  | 116, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans)))))))))
, by decide⟩
  | 117, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc)))))
, by decide⟩
  | 118, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc))))))))
, by decide⟩
  | 119, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))))
, by decide⟩
  | 120, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))))
, by decide⟩
  | 121, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc))))))))
, by decide⟩
  | 122, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))))))
, by decide⟩
  | 123, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))))))
, by decide⟩
  | 124, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc))))))))
, by decide⟩
  | 125, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc)))))))
, by decide⟩
  | 126, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))))))
, by decide⟩
  | 127, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))))))
, by decide⟩
  | 128, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc))))))
, by decide⟩
  | 129, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc)))))))
, by decide⟩
  | 130, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc)))))))
, by decide⟩
  | 131, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans)))))))
, by decide⟩
  | 132, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc)))))
, by decide⟩
  | 133, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans)))))
, by decide⟩
  | 134, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc))))
, by decide⟩
  | 135, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc)))))
, by decide⟩
  | 136, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))
, by decide⟩
  | 137, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc))))))))))
, by decide⟩
  | 138, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))))))))
, by decide⟩
  | 139, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))))
, by decide⟩
  | 140, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc))))))))
, by decide⟩
  | 141, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))
, by decide⟩
  | 142, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans)))))))))
, by decide⟩
  | 143, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc)))
, by decide⟩
  | 144, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))))
, by decide⟩
  | 145, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc)))))))))
, by decide⟩
  | 146, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))))))))
, by decide⟩
  | 147, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc)))))
, by decide⟩
  | 148, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc)))))))))
, by decide⟩
  | 149, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc)))))))
, by decide⟩
  | 150, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))))))
, by decide⟩
  | 151, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc)))))))))
, by decide⟩
  | 152, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc)))))
, by decide⟩
  | 153, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))))))
, by decide⟩
  | 154, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc))))))
, by decide⟩
  | 155, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans))))))
, by decide⟩
  | 156, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))))
, by decide⟩
  | 157, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.cyc)))))))))
, by decide⟩
  | 158, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans))))))))))
, by decide⟩
  | 159, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc)))))
, by decide⟩
  | 160, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))))
, by decide⟩
  | 161, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans))))))
, by decide⟩
  | 162, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc)))))
, by decide⟩
  | 163, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))
, by decide⟩
  | 164, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans LowCob.cyc)))))))
, by decide⟩
  | 165, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.trans)))))
, by decide⟩
  | 166, _ => exact ⟨_, _, (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc LowCob.cyc))))
, by decide⟩
  | 167, _ => exact ⟨_, _, (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.cyc (LowCob.comp LowCob.trans (LowCob.comp LowCob.cyc LowCob.trans)))))))), by decide⟩
  | (n+168), h => omega


/-- **`LowCob` IS `GL(3,2)`.** Soundness by `lowCob_isGL`, completeness by `lowCob_covers`
    together with `glList_indep` / `glList_onto`. Nothing about the invariant `g` is left
    outside Lean. -/
theorem lowCob_eq_GL (a b c : Nat) (ha : a < 8) (hb : b < 8) (hc : c < 8)
    (h : glIndep a b c = true) :
    ∃ t l, LowCob t l ∧ ∀ v, v < 8 → t v = linMap a b c v := by
  obtain ⟨t, l, hL, hv⟩ := lowCob_covers (glIdx a b c) (glIdx_lt a ha b hb c hc h)
  obtain ⟨e1, e2, e3⟩ := glIdx_eq a ha b hb c hc h
  rw [e1, e2, e3] at hv
  exact ⟨t, l, hL, hv⟩

end GLClosure


/-! ## Tier 28: why `tr(A³)` is the FINER invariant

`A_sig`'s entry is not the resonance predicate but the SIGN `-P1 = -σ(a,b)·σ(a⊕L,b⊕L)`.
Under a class member the coboundary does NOT cancel there -- only `λ(a⊕b)` squares away, and
what survives is a factor depending on `a` alone times one depending on `b` alone:

    P1 (p a) (p b) = P1 a b * μ a * μ b,        μ x = λ x * λ (x ⊕ L).

That is a DIAGONAL SIMILARITY `A' = D A D` with `D = diag μ` and `D² = I`, so `tr(A'^k) =
tr(A^k)` for every `k` -- `tr(A³)` is `GL(3,2)`-invariant. `tau` has no such factorisation, and
measurement (`W24`) confirms it genuinely CHANGES `tr(A³)`. That asymmetry is exactly why the
pair `(tr A², tr A³)` separates strictly more than `tr A²` alone: `tr(A²)` is invariant under
BOTH symmetries, `tr(A³)` only under one. -/

theorem P1_of_coboundary (m L a b : Nat) (p : Nat → Nat) (lam : Nat → Int)
    (hL : L < 2^m) (ha : a < 2^m) (hb : b < 2^m)
    (hlin : ∀ x y, p (x ^^^ y) = p x ^^^ p y)
    (hpm : ∀ x, lam x = 1 ∨ lam x = -1)
    (hcob : ∀ x y, x < 2^m → y < 2^m →
        cdSigma (p x) (p y) m = cdSigma x y m * lam x * lam y * lam (x ^^^ y)) :
    cdSigma (p a) (p b) m * cdSigma (p a ^^^ p L) (p b ^^^ p L) m
      = cdSigma a b m * cdSigma (a ^^^ L) (b ^^^ L) m
          * (lam a * lam (a ^^^ L)) * (lam b * lam (b ^^^ L)) := by
  have sq : ∀ x : Nat, lam x * lam x = 1 := by
    intro x; rcases hpm x with h | h <;> rw [h] <;> decide
  have haL : a ^^^ L < 2^m := Nat.xor_lt_two_pow ha hL
  have hbL : b ^^^ L < 2^m := Nat.xor_lt_two_pow hb hL
  have hx : (a ^^^ L) ^^^ (b ^^^ L) = a ^^^ b := by
    simp [Nat.xor_assoc, Nat.xor_comm, xorLcomm, xorCancelL]
  rw [← hlin a L, ← hlin b L, hcob a b ha hb, hcob (a ^^^ L) (b ^^^ L) haL hbL, hx]
  calc cdSigma a b m * lam a * lam b * lam (a ^^^ b)
        * (cdSigma (a ^^^ L) (b ^^^ L) m * lam (a ^^^ L) * lam (b ^^^ L) * lam (a ^^^ b))
      = cdSigma a b m * cdSigma (a ^^^ L) (b ^^^ L) m
          * (lam a * lam (a ^^^ L)) * (lam b * lam (b ^^^ L))
          * (lam (a ^^^ b) * lam (a ^^^ b)) := by ac_rfl
    _ = _ := by rw [sq]; simp

/-- The class-member form: the `A_sig` entry sign transforms by a diagonal similarity. -/
theorem P1_lowCob {t l} (h : LowCob t l) (k L a b : Nat)
    (hL : L < 2^(k+3)) (ha : a < 2^(k+3)) (hb : b < 2^(k+3)) :
    cdSigma (lowMap t a) (lowMap t b) (k+3)
        * cdSigma (lowMap t a ^^^ lowMap t L) (lowMap t b ^^^ lowMap t L) (k+3)
      = cdSigma a b (k+3) * cdSigma (a ^^^ L) (b ^^^ L) (k+3)
          * (lowSign l a * lowSign l (a ^^^ L)) * (lowSign l b * lowSign l (b ^^^ L)) :=
  P1_of_coboundary (k+3) L a b (lowMap t) (lowSign l) hL ha hb
    (lowMap_lin t (lowCob_lt h) (lowCob_lin h))
    (fun x => lowCob_pm h (x % 8))
    (lowCob_sigma h k)

/-! ## Tier 29: the counting recursion, in Lean

`W15`'s ledger -- the step that turns `Ncnt` at level `m+1` into `Ncnt` at level `m` -- has been
carried on paper and pinned by contract. This tier formalises it. The whole mechanism is
**split by predicate, extract singletons, evaluate constants**; no `Finset`, exactly as Tier 27
avoided cardinality.

Stage 1 is the toolkit. -/

theorem sumLt_zero (n : Nat) : sumLt n (fun _ => 0) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => rw [sumLt, ih]

theorem sumLt_const (n c : Nat) : sumLt n (fun _ => c) = n * c := by
  induction n with
  | zero => simp [sumLt]
  | succ n ih => rw [sumLt, ih, Nat.succ_mul]

/-- A bounded sum is additive in the summand. -/
theorem sumLt_pair (n : Nat) (f g : Nat → Nat) :
    sumLt n (fun i => f i + g i) = sumLt n f + sumLt n g := by
  induction n with
  | zero => rfl
  | succ n ih => rw [sumLt, sumLt, sumLt, ih]; omega

/-- Split a bounded sum along a decidable predicate. -/
theorem sumLt_split_if (n : Nat) (p : Nat → Prop) [DecidablePred p] (f g : Nat → Nat) :
    sumLt n (fun i => if p i then f i else g i)
      = sumLt n (fun i => if p i then f i else 0)
        + sumLt n (fun i => if p i then 0 else g i) := by
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [sumLt, sumLt, sumLt, ih]
      by_cases h : p n
      · rw [if_pos h, if_pos h, if_pos h]; omega
      · rw [if_neg h, if_neg h, if_neg h]; omega

/-- A single index contributes its own value. -/
theorem sumLt_single (n j c : Nat) (hj : j < n) :
    sumLt n (fun i => if i = j then c else 0) = c := by
  induction n with
  | zero => omega
  | succ n ih =>
      rw [sumLt]
      by_cases h : n = j
      · subst h
        rw [if_pos rfl]
        have : sumLt n (fun i => if i = n then c else 0) = sumLt n (fun _ => 0) :=
          sumLt_congr n _ _ (fun i hi => by rw [if_neg (by omega)])
        rw [this, sumLt_zero, Nat.zero_add]
      · rw [if_neg h, ih (by omega), Nat.add_zero]

/-- A single index, with the guard written the other way round. -/
theorem sumLt_single' (n j c : Nat) (hj : j < n) :
    sumLt n (fun i => if j = i then c else 0) = c := by
  rw [sumLt_congr n _ (fun i => if i = j then c else 0)
        (fun i _ => by by_cases h : j = i
                       · rw [if_pos h, if_pos h.symm]
                       · rw [if_neg h, if_neg (fun hh => h hh.symm)]),
      sumLt_single n j c hj]

/-! ### Stage 2: the quadrant split, and the `ll` quadrant

Everything is stated at level `m+2` so the seam is `2^(m+1)` and the reduction rows apply
verbatim. -/

/-- The summand of `Ncnt`, named so the quadrant proofs stay readable. -/
def nInd (W m a b : Nat) : Nat :=
  if a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b ∧ Qgen' W a b m = -1 then 1 else 0

theorem Ncnt_eq (W m : Nat) :
    Ncnt W m = sumLt (2^m) (fun a => sumLt (2^m) (fun b => nInd W m a b)) := rfl

/-- The four quadrants of the level-`m+2` box, split at the seam `2^(m+1)`. -/
theorem Ncnt_quad (W m : Nat) :
    Ncnt W (m+2)
      = (sumLt (2^(m+1)) (fun a => sumLt (2^(m+1)) (fun b => nInd W (m+2) a b))
          + sumLt (2^(m+1)) (fun a =>
              sumLt (2^(m+1)) (fun v => nInd W (m+2) a (2^(m+1) + v))))
        + (sumLt (2^(m+1)) (fun u =>
              sumLt (2^(m+1)) (fun b => nInd W (m+2) (2^(m+1) + u) b))
          + sumLt (2^(m+1)) (fun u =>
              sumLt (2^(m+1)) (fun v => nInd W (m+2) (2^(m+1) + u) (2^(m+1) + v)))) := by
  have hsplit : (2:Nat)^(m+2) = 2^(m+1) + 2^(m+1) := by rw [Nat.pow_succ]; omega
  rw [Ncnt_eq, hsplit, sumLt_add]
  rw [sumLt_congr (2^(m+1)) (fun a => sumLt (2^(m+1) + 2^(m+1)) (fun b => nInd W (m+2) a b))
        (fun a => sumLt (2^(m+1)) (fun b => nInd W (m+2) a b)
                  + sumLt (2^(m+1)) (fun v => nInd W (m+2) a (2^(m+1) + v)))
        (fun a _ => sumLt_add _ _ _)]
  rw [sumLt_congr (2^(m+1))
        (fun i => sumLt (2^(m+1) + 2^(m+1)) (fun b => nInd W (m+2) (2^(m+1) + i) b))
        (fun i => sumLt (2^(m+1)) (fun b => nInd W (m+2) (2^(m+1) + i) b)
                  + sumLt (2^(m+1)) (fun v => nInd W (m+2) (2^(m+1) + i) (2^(m+1) + v)))
        (fun i _ => sumLt_add _ _ _)]
  rw [sumLt_pair, sumLt_pair]

/-- `ll`: `Q'red_low_ll` is UNCONDITIONAL, so this quadrant is the level-`m+1` count outright. -/
theorem Ncnt_ll_low (W m : Nat) (hW : W < 2^(m+1)) :
    sumLt (2^(m+1)) (fun a => sumLt (2^(m+1)) (fun b => nInd W (m+2) a b))
      = Ncnt W (m+1) := by
  rw [Ncnt_eq]
  apply sumLt_congr
  intro a ha
  apply sumLt_congr
  intro b hb
  unfold nInd
  rw [Q'red_low_ll m W a b hW ha hb]

/-! ### Stage 2b: `ul` and `lu` reduce to the UNPRIMED count

The low `ul` and `lu` rows land on `Qgen`, not `Qgen'`, so both quadrants reduce to a count of
the unprimed form. `lu`'s one row-failure, `a = W`, contributes NOTHING: `Qgen'_label_left`
makes it `+1` there. -/

/-- The unprimed summand, in the shape the `ul` row produces. -/
def qInd (W m u b : Nat) : Nat := if b ≠ 0 ∧ Qgen W u b m = -1 then 1 else 0

/-- The unprimed count over `u ∈ [0,2^m)`, `b ∈ [1,2^m)`. -/
def Mcnt (W m : Nat) : Nat :=
  sumLt (2^m) (fun u => sumLt (2^m) (fun b => qInd W m u b))

theorem Ncnt_ul_low (W m : Nat) (hW : W < 2^(m+1)) :
    sumLt (2^(m+1)) (fun u => sumLt (2^(m+1)) (fun b => nInd W (m+2) (2^(m+1) + u) b))
      = Mcnt W (m+1) := by
  have hp : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  unfold Mcnt
  apply sumLt_congr
  intro u hu
  apply sumLt_congr
  intro b hb
  unfold nInd qInd
  have hne0 : (2:Nat)^(m+1) + u ≠ 0 := by omega
  have hneb : (2:Nat)^(m+1) + u ≠ b := by omega
  have hcomm : (2:Nat)^(m+1) + u = u + 2^(m+1) := by omega
  by_cases hb0 : b = 0
  · rw [if_neg (by rintro ⟨-, h, -⟩; exact h hb0), if_neg (by rintro ⟨h, -⟩; exact h hb0)]
  · rw [hcomm, Q'red_low_ul m W u b hW hu hb hb0]
    by_cases hq : Qgen W u b (m+1) = -1
    · rw [if_pos ⟨by omega, hb0, by omega, hq⟩, if_pos ⟨hb0, hq⟩]
    · rw [if_neg (by rintro ⟨-, -, -, h⟩; exact hq h), if_neg (by rintro ⟨-, h⟩; exact hq h)]

theorem Ncnt_lu_low (W m : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0) :
    sumLt (2^(m+1)) (fun a => sumLt (2^(m+1)) (fun v => nInd W (m+2) a (2^(m+1) + v)))
      = sumLt (2^(m+1)) (fun a =>
          sumLt (2^(m+1)) (fun v =>
            if a ≠ 0 ∧ a ≠ W ∧ Qgen W v a (m+1) = -1 then 1 else 0)) := by
  have hp : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  apply sumLt_congr
  intro a ha
  apply sumLt_congr
  intro v hv
  unfold nInd
  have hcomm : (2:Nat)^(m+1) + v = v + 2^(m+1) := by omega
  have hne0 : v + (2:Nat)^(m+1) ≠ 0 := by omega
  have hnea : a ≠ v + (2:Nat)^(m+1) := by omega
  by_cases ha0 : a = 0
  · rw [if_neg (by rintro ⟨h, -⟩; exact h ha0), if_neg (by rintro ⟨h, -⟩; exact h ha0)]
  · by_cases haW : a = W
    · -- the row fails here, and `Qgen'_label_left` makes the value +1: contributes nothing
      have hv0 : v + (2:Nat)^(m+1) ≠ 0 := hne0
      have hvW : v + (2:Nat)^(m+1) ≠ W := by omega
      have hlt : v + (2:Nat)^(m+1) < 2^(m+2) := by
        have : (2:Nat)^(m+2) = 2^(m+1) + 2^(m+1) := by rw [Nat.pow_succ]; omega
        omega
      have hWlt : W < 2^(m+2) := by
        have : (2:Nat)^(m+2) = 2^(m+1) + 2^(m+1) := by rw [Nat.pow_succ]; omega
        omega
      rw [hcomm, haW, Qgen'_label_left (m+2) W (v + 2^(m+1)) hWlt hlt hW0 hv0 hvW]
      rw [if_neg (by rintro ⟨-, -, -, h⟩; exact absurd h (by decide)),
          if_neg (by rintro ⟨-, h, -⟩; exact h rfl)]
    · rw [hcomm, Q'red_low_lu m W a v hW ha hv ha0 (fun h => haW (xor_zero_eq a W h))]
      by_cases hq : Qgen W v a (m+1) = -1
      · rw [if_pos ⟨ha0, hne0, hnea, hq⟩, if_pos ⟨ha0, haW, hq⟩]
      · rw [if_neg (by rintro ⟨-, -, -, h⟩; exact hq h),
            if_neg (by rintro ⟨-, -, h⟩; exact hq h)]

/-! ### Stage 2c: the `uu` quadrant

Five side conditions, but they collapse. On EVERY failure slice `Qgen = -1` -- `u = 0` and
`v = 0` are the gap roots `a = H` and `b = H`, `u = W` and `v = W` are `a ⊕ W = H` and
`b ⊕ W = H` -- so `Qgen'_off_lines` converts all four at once. The fifth, `v = u ⊕ W`, is
exactly `Qgen'_coset_partner`, which gives `+1` and so contributes nothing. -/

/-- The level-`m+1` summand the `uu` quadrant reduces to. -/
def uuInd (W m u v : Nat) : Nat :=
  if u ≠ v ∧ v ≠ u ^^^ W ∧
     (u = 0 ∨ v = 0 ∨ u = W ∨ v = W ∨ Qgen' W v u (m+1) = -1) then 1 else 0

theorem Ncnt_uu_low (W m u v : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (hu : u < 2^(m+1)) (hv : v < 2^(m+1)) :
    nInd W (m+2) (2^(m+1) + u) (2^(m+1) + v) = uuInd W m u v := by
  have hp : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have h2 : (2:Nat)^(m+2) = 2^(m+1) + 2^(m+1) := by rw [Nat.pow_succ]; omega
  have hcu : (2:Nat)^(m+1) + u = u + 2^(m+1) := by omega
  have hcv : (2:Nat)^(m+1) + v = v + 2^(m+1) := by omega
  have halt : u + (2:Nat)^(m+1) < 2^(m+2) := by omega
  have hblt : v + (2:Nat)^(m+1) < 2^(m+2) := by omega
  have hWlt : W < (2:Nat)^(m+2) := by omega
  have hxa : (u + 2^(m+1)) ^^^ W = (u ^^^ W) + 2^(m+1) := seam_xor_left u W m hu hW
  have ha0 : u + (2:Nat)^(m+1) ≠ 0 := by omega
  have haW : u + (2:Nat)^(m+1) ≠ W := by omega
  have hb0 : v + (2:Nat)^(m+1) ≠ 0 := by omega
  unfold nInd uuInd
  rw [hcu, hcv]
  by_cases huv : u = v
  · rw [if_neg (by rintro ⟨-, -, h, -⟩; exact h (by rw [huv])),
        if_neg (by rintro ⟨h, -⟩; exact h huv)]
  · have hab : u + (2:Nat)^(m+1) ≠ v + 2^(m+1) := by omega
    have hcosEq : (u + 2^(m+1)) ^^^ W = (u ^^^ W) + 2^(m+1) := hxa
    by_cases hcos : v = u ^^^ W
    · -- b = a ⊕ W : the coset partner, value +1, contributes nothing
      have hbe : v + (2:Nat)^(m+1) = (u + 2^(m+1)) ^^^ W := by rw [hcosEq, hcos]
      rw [hbe, Qgen'_coset_partner (m+2) W (u + 2^(m+1)) hWlt halt ha0
            (by rw [hcosEq]; omega)]
      rw [if_neg (by rintro ⟨-, -, -, h⟩; exact absurd h (by decide)),
          if_neg (by rintro ⟨-, h, -⟩; exact h hcos)]
    · have hcosNe : v + (2:Nat)^(m+1) ≠ (u + 2^(m+1)) ^^^ W := by
        rw [hcosEq]; omega
      by_cases hrow : u ≠ 0 ∧ v ≠ 0 ∧ u ≠ W ∧ v ≠ W
      · obtain ⟨hu0, hv0, huW, hvW⟩ := hrow
        rw [Q'red_low_uu m W u v hW hu hv hu0 hv0
              (fun h => huW (xor_zero_eq u W h)) (fun h => hvW (xor_zero_eq v W h))
              (by
                intro h
                apply hcos
                have e1 : u ^^^ v = W := xor_zero_eq (u ^^^ v) W h
                have e2 : u ^^^ (u ^^^ v) = u ^^^ W := by rw [e1]
                rw [xorCancelL] at e2
                exact e2)]
        by_cases hq : Qgen' W v u (m+1) = -1
        · rw [if_pos ⟨ha0, hb0, hab, hq⟩,
              if_pos ⟨huv, hcos, Or.inr (Or.inr (Or.inr (Or.inr hq)))⟩]
        · have hno : ¬ (u = 0 ∨ v = 0 ∨ u = W ∨ v = W ∨ Qgen' W v u (m+1) = -1) := by
            rintro (h | h | h | h | h)
            · exact hu0 h
            · exact hv0 h
            · exact huW h
            · exact hvW h
            · exact hq h
          rw [if_neg (by rintro ⟨-, -, -, h⟩; exact hq h),
              if_neg (by rintro ⟨-, -, h⟩; exact hno h)]
      · -- a row failure: Qgen = -1 on every one of them, so `Qgen'_off_lines` applies
        have hQ : Qgen W (u + 2^(m+1)) (v + 2^(m+1)) (m+2) = -1 := by
          by_cases hu0 : u = 0
          · rw [hu0, Nat.zero_add]
            exact Qgen_H_left_low m W (v + 2^(m+1)) hW hW0 hblt
          · by_cases hv0 : v = 0
            · rw [hv0, Nat.zero_add]
              exact Qgen_H_right_low m W (u + 2^(m+1)) hW hW0 halt
            · by_cases huW : u = W
              · exact Qgen_H_left_low' m W (u + 2^(m+1)) (v + 2^(m+1)) hW hW0 hblt
                  (by rw [hxa, huW, Nat.xor_self, Nat.zero_add])
              · have hvW : v = W := by
                  by_cases hvW : v = W
                  · exact hvW
                  · exact absurd ⟨hu0, hv0, huW, hvW⟩ hrow
                exact Qgen_H_right_low' m W (u + 2^(m+1)) (v + 2^(m+1)) hW hW0 halt
                  (by rw [seam_xor_left v W m hv hW, hvW, Nat.xor_self, Nat.zero_add])
        rw [Qgen'_off_lines (m+2) W (u + 2^(m+1)) (v + 2^(m+1)) hWlt halt hblt hQ
              ha0 haW hab hcosNe]
        rw [if_pos ⟨ha0, hb0, hab, rfl⟩]
        have hdis : u = 0 ∨ v = 0 ∨ u = W ∨ v = W := by
          by_cases h1 : u = 0
          · exact Or.inl h1
          · by_cases h2 : v = 0
            · exact Or.inr (Or.inl h2)
            · by_cases h3 : u = W
              · exact Or.inr (Or.inr (Or.inl h3))
              · by_cases h4 : v = W
                · exact Or.inr (Or.inr (Or.inr h4))
                · exact absurd ⟨h1, h2, h3, h4⟩ hrow
        rcases hdis with h | h | h | h
        · rw [if_pos ⟨huv, hcos, Or.inl h⟩]
        · rw [if_pos ⟨huv, hcos, Or.inr (Or.inl h)⟩]
        · rw [if_pos ⟨huv, hcos, Or.inr (Or.inr (Or.inl h))⟩]
        · rw [if_pos ⟨huv, hcos, Or.inr (Or.inr (Or.inr (Or.inl h)))⟩]

/-! ### Stage 3: the bridge, via a shared core

All four quadrants differ from `Ncnt` only on the six lines, so they all factor through ONE
quantity: the count OFF the lines. Proving `Ncnt` and each quadrant equal that core plus an
explicit constant avoids inclusion-exclusion entirely -- the sequential
`sumLt_split_if` chains keep the pieces disjoint by construction. -/

/-- Counting a predicate and its complement exhausts the range. -/
theorem sumLt_compl (n : Nat) (p : Nat → Prop) [DecidablePred p] :
    sumLt n (fun i => if p i then 1 else 0) + sumLt n (fun i => if p i then 0 else 1) = n := by
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [sumLt, sumLt]
      by_cases h : p n
      · rw [if_pos h, if_pos h]; omega
      · rw [if_neg h, if_neg h]; omega

/-- Two distinct indices below `n` contribute exactly `2`. -/
theorem sumLt_two (n W : Nat) (hW : W < n) (hW0 : W ≠ 0) (hn : 0 < n) :
    sumLt n (fun i => if i = 0 ∨ i = W then 1 else 0) = 2 := by
  have hsplit := sumLt_split_if n (fun i => i = 0) (fun _ => 1)
      (fun i => if i = 0 ∨ i = W then 1 else 0)
  have e1 : sumLt n (fun i => if i = 0 ∨ i = W then 1 else 0)
      = sumLt n (fun i => if i = 0 then 1 else if i = 0 ∨ i = W then 1 else 0) := by
    apply sumLt_congr
    intro i _
    by_cases h : i = 0
    · rw [if_pos h, if_pos (Or.inl h)]
    · rw [if_neg h]
  rw [e1, hsplit]
  have e2 : sumLt n (fun i => if i = 0 then 1 else 0) = 1 := by
    rw [sumLt_congr n _ (fun i => if i = 0 then 1 else 0) (fun i _ => rfl)]
    exact sumLt_single n 0 1 hn
  have e3 : sumLt n (fun i => if i = 0 then 0 else if i = 0 ∨ i = W then 1 else 0)
      = sumLt n (fun i => if i = W then 1 else 0) := by
    apply sumLt_congr
    intro i _
    by_cases h : i = 0
    · rw [if_pos h, if_neg (by omega : ¬ i = W)]
    · rw [if_neg h]
      by_cases h2 : i = W
      · rw [if_pos (Or.inr h2), if_pos h2]
      · rw [if_neg (fun hc => hc.elim h h2), if_neg h2]
  rw [e2, e3, sumLt_single n W 1 hW]

/-- The core: the count OFF all six lines. -/
def OffCnt (W m : Nat) : Nat :=
  sumLt (2^m) (fun a => sumLt (2^m) (fun b =>
    if a ≠ 0 ∧ a ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ a ≠ b ∧ b ≠ a ^^^ W ∧ Qgen' W a b m = -1
    then 1 else 0))

/-- Pointwise: `Ncnt`'s summand is `OffCnt`'s plus the `b = W` column. The `a = W` row and the
    coset diagonal `b = a ⊕ W` contribute NOTHING -- `Qgen'_label_left` and
    `Qgen'_coset_partner` are `+1` there -- while `Qgen'_label_right` makes the `b = W` column
    `-1` throughout. -/
theorem nInd_split (W M a b : Nat) (hW : W < 2^M) (hW0 : W ≠ 0)
    (ha : a < 2^M) (hb : b < 2^M) :
    nInd W M a b
      = (if b = W ∧ a ≠ 0 ∧ a ≠ W then 1 else 0)
        + (if a ≠ 0 ∧ a ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ a ≠ b ∧ b ≠ a ^^^ W ∧ Qgen' W a b M = -1
           then 1 else 0) := by
  unfold nInd
  by_cases hbW : b = W
  · have hb0 : b ≠ 0 := by rw [hbW]; exact hW0
    have hO : (if a ≠ 0 ∧ a ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ a ≠ b ∧ b ≠ a ^^^ W
                  ∧ Qgen' W a b M = -1 then (1:Nat) else 0) = 0 :=
      if_neg (fun hc => hc.2.2.2.1 hbW)
    by_cases ha0 : a = 0
    · have hL : (if a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b ∧ Qgen' W a b M = -1 then (1:Nat) else 0) = 0 :=
        if_neg (fun hc => hc.1 ha0)
      have hD : (if b = W ∧ a ≠ 0 ∧ a ≠ W then (1:Nat) else 0) = 0 :=
        if_neg (fun hc => hc.2.1 ha0)
      rw [hL, hD, hO]
    · by_cases haW : a = W
      · have hab : a = b := by rw [haW, hbW]
        have hL : (if a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b ∧ Qgen' W a b M = -1 then (1:Nat) else 0) = 0 :=
          if_neg (fun hc => hc.2.2.1 hab)
        have hD : (if b = W ∧ a ≠ 0 ∧ a ≠ W then (1:Nat) else 0) = 0 :=
          if_neg (fun hc => hc.2.2 haW)
        rw [hL, hD, hO]
      · have hab : a ≠ b := by rw [hbW]; exact haW
        have hq : Qgen' W a b M = -1 := by
          rw [hbW]; exact Qgen'_label_right M W a hW ha hW0
        have hL : (if a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b ∧ Qgen' W a b M = -1 then (1:Nat) else 0) = 1 :=
          if_pos ⟨ha0, hb0, hab, hq⟩
        have hD : (if b = W ∧ a ≠ 0 ∧ a ≠ W then (1:Nat) else 0) = 1 :=
          if_pos ⟨hbW, ha0, haW⟩
        rw [hL, hD, hO]
  · have hD : (if b = W ∧ a ≠ 0 ∧ a ≠ W then (1:Nat) else 0) = 0 :=
      if_neg (fun hc => hbW hc.1)
    rw [hD, Nat.zero_add]
    by_cases ha0 : a = 0
    · rw [if_neg (fun hc => hc.1 ha0), if_neg (fun hc => hc.1 ha0)]
    · by_cases hb0 : b = 0
      · rw [if_neg (fun hc => hc.2.1 hb0), if_neg (fun hc => hc.2.2.1 hb0)]
      · by_cases haW : a = W
        · have hq : Qgen' W a b M = 1 := by
            rw [haW]; exact Qgen'_label_left M W b hW hb hW0 hb0 hbW
          rw [if_neg (fun hc => by rw [hq] at hc; exact absurd hc.2.2.2 (by decide)),
              if_neg (fun hc => hc.2.1 haW)]
        · by_cases hcos : b = a ^^^ W
          · have haX : a ^^^ W ≠ 0 := fun h => haW (xor_zero_eq a W h)
            have hq : Qgen' W a b M = 1 := by
              rw [hcos]; exact Qgen'_coset_partner M W a hW ha ha0 haX
            rw [if_neg (fun hc => by rw [hq] at hc; exact absurd hc.2.2.2 (by decide)),
                if_neg (fun hc => hc.2.2.2.2.2.1 hcos)]
          · by_cases hab : a = b
            · rw [if_neg (fun hc => hc.2.2.1 hab), if_neg (fun hc => hc.2.2.2.2.1 hab)]
            · by_cases hq : Qgen' W a b M = -1
              · rw [if_pos ⟨ha0, hb0, hab, hq⟩, if_pos ⟨ha0, haW, hb0, hbW, hab, hcos, hq⟩]
              · rw [if_neg (fun hc => hq hc.2.2.2),
                    if_neg (fun hc => hq hc.2.2.2.2.2.2)]

/-- The `b = W` column contributes exactly one, and only when `a` is off the two lines. -/
private theorem col_W (W M a : Nat) (hW : W < 2^M) :
    sumLt (2^M) (fun b => if b = W ∧ a ≠ 0 ∧ a ≠ W then 1 else 0)
      = if a ≠ 0 ∧ a ≠ W then 1 else 0 := by
  by_cases h : a ≠ 0 ∧ a ≠ W
  · rw [if_pos h,
        sumLt_congr (2^M) _ (fun b => if b = W then 1 else 0)
          (fun b _ => by
            by_cases hb : b = W
            · rw [if_pos ⟨hb, h.1, h.2⟩, if_pos hb]
            · rw [if_neg (fun hc => hb hc.1), if_neg hb])]
    exact sumLt_single (2^M) W 1 hW
  · rw [if_neg h,
        sumLt_congr (2^M) _ (fun _ => 0)
          (fun b _ => if_neg (fun hc => h ⟨hc.2.1, hc.2.2⟩))]
    exact sumLt_zero _

/-- **The bridge.** `Ncnt` is the off-lines core plus the `b = W` column, whose size is
    `2^M - 2`. Stated additively to stay in `Nat`. -/
theorem Ncnt_eq_OffCnt (W M : Nat) (hW : W < 2^M) (hW0 : W ≠ 0) :
    Ncnt W M + 2 = OffCnt W M + 2^M := by
  have hp : (0:Nat) < 2^M := Nat.two_pow_pos M
  have hstep : Ncnt W M
      = sumLt (2^M) (fun a => if a ≠ 0 ∧ a ≠ W then 1 else 0) + OffCnt W M := by
    rw [Ncnt_eq]
    unfold OffCnt
    rw [← sumLt_pair]
    apply sumLt_congr
    intro a hA
    rw [sumLt_congr (2^M) _
          (fun b => (if b = W ∧ a ≠ 0 ∧ a ≠ W then 1 else 0)
            + (if a ≠ 0 ∧ a ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ a ≠ b ∧ b ≠ a ^^^ W
                 ∧ Qgen' W a b M = -1 then 1 else 0))
          (fun b hB => nInd_split W M a b hW hW0 hA hB),
        sumLt_pair, col_W W M a hW]
  have hcount : sumLt (2^M) (fun a => if a ≠ 0 ∧ a ≠ W then 1 else 0) + 2 = 2^M := by
    have hc := sumLt_compl (2^M) (fun i => i = 0 ∨ i = W)
    have h2 := sumLt_two (2^M) W hW hW0 hp
    have he : sumLt (2^M) (fun i => if i = 0 ∨ i = W then 0 else 1)
        = sumLt (2^M) (fun a => if a ≠ 0 ∧ a ≠ W then 1 else 0) := by
      apply sumLt_congr
      intro i _
      by_cases h : i = 0 ∨ i = W
      · rw [if_pos h, if_neg (fun hc2 => h.elim hc2.1 hc2.2)]
      · rw [if_neg h, if_pos ⟨fun hh => h (Or.inl hh), fun hh => h (Or.inr hh)⟩]
    rw [he] at hc
    omega
  omega

/-- Swapping the order of a square double sum. -/
theorem sumLt_swap (n : Nat) (f : Nat → Nat → Nat) :
    sumLt n (fun i => sumLt n (fun j => f i j))
      = sumLt n (fun j => sumLt n (fun i => f i j)) := by
  induction n with
  | zero => rfl
  | succ n ih =>
      have hL : sumLt (n+1) (fun i => sumLt (n+1) (fun j => f i j))
          = (sumLt n (fun i => sumLt n (fun j => f i j)) + sumLt n (fun i => f i n))
            + (sumLt n (fun j => f n j) + f n n) := by
        rw [sumLt]
        rw [sumLt_congr n (fun i => sumLt (n+1) (fun j => f i j))
              (fun i => sumLt n (fun j => f i j) + f i n) (fun i _ => by rw [sumLt]),
            sumLt_pair, sumLt]
      have hR : sumLt (n+1) (fun j => sumLt (n+1) (fun i => f i j))
          = (sumLt n (fun j => sumLt n (fun i => f i j)) + sumLt n (fun j => f n j))
            + (sumLt n (fun i => f i n) + f n n) := by
        rw [sumLt]
        rw [sumLt_congr n (fun j => sumLt (n+1) (fun i => f i j))
              (fun j => sumLt n (fun i => f i j) + f n j) (fun j _ => by rw [sumLt]),
            sumLt_pair, sumLt]
      rw [hL, hR, ih]
      omega

/-- Pointwise: `uuInd` is `OffCnt`'s summand (with the arguments swapped, as the `uu` row
    produces them) plus the four boundary lines. -/
theorem uuInd_split (W m u v : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (hu : u < 2^(m+1)) (hv : v < 2^(m+1)) :
    uuInd W m u v
      = (if u ≠ v ∧ v ≠ u ^^^ W ∧ (u = 0 ∨ v = 0 ∨ u = W ∨ v = W) then 1 else 0)
        + (if v ≠ 0 ∧ v ≠ W ∧ u ≠ 0 ∧ u ≠ W ∧ v ≠ u ∧ u ≠ v ^^^ W
              ∧ Qgen' W v u (m+1) = -1 then 1 else 0) := by
  have hxc : (v = u ^^^ W) ↔ (u = v ^^^ W) := by
    constructor
    · intro hh; rw [hh, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
    · intro hh; rw [hh, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
  unfold uuInd
  by_cases hd : u = v
  · rw [if_neg (fun hc => hc.1 hd), if_neg (fun hc => hc.1 hd),
        if_neg (fun hc => hc.2.2.2.2.1 hd.symm), Nat.add_zero]
  · by_cases hc : v = u ^^^ W
    · rw [if_neg (fun h => h.2.1 hc), if_neg (fun h => h.2.1 hc),
          if_neg (fun h => h.2.2.2.2.2.1 (hxc.mp hc)), Nat.add_zero]
    · by_cases hline : u = 0 ∨ v = 0 ∨ u = W ∨ v = W
      · have hline5 : u = 0 ∨ v = 0 ∨ u = W ∨ v = W ∨ Qgen' W v u (m+1) = -1 := by
          rcases hline with h | h | h | h
          · exact Or.inl h
          · exact Or.inr (Or.inl h)
          · exact Or.inr (Or.inr (Or.inl h))
          · exact Or.inr (Or.inr (Or.inr (Or.inl h)))
        have hoff : ¬ (v ≠ 0 ∧ v ≠ W ∧ u ≠ 0 ∧ u ≠ W ∧ v ≠ u ∧ u ≠ v ^^^ W
                        ∧ Qgen' W v u (m+1) = -1) := by
          intro h
          rcases hline with hh | hh | hh | hh
          · exact h.2.2.1 hh
          · exact h.1 hh
          · exact h.2.2.2.1 hh
          · exact h.2.1 hh
        have hLv : (if u ≠ v ∧ v ≠ u ^^^ W ∧
                      (u = 0 ∨ v = 0 ∨ u = W ∨ v = W ∨ Qgen' W v u (m+1) = -1)
                    then (1:Nat) else 0) = 1 := if_pos ⟨hd, hc, hline5⟩
        have hDv : (if u ≠ v ∧ v ≠ u ^^^ W ∧ (u = 0 ∨ v = 0 ∨ u = W ∨ v = W)
                    then (1:Nat) else 0) = 1 := if_pos ⟨hd, hc, hline⟩
        have hOv : (if v ≠ 0 ∧ v ≠ W ∧ u ≠ 0 ∧ u ≠ W ∧ v ≠ u ∧ u ≠ v ^^^ W
                      ∧ Qgen' W v u (m+1) = -1 then (1:Nat) else 0) = 0 := if_neg hoff
        rw [hLv, hDv, hOv]
      · have h0 : u ≠ 0 := fun hh => hline (Or.inl hh)
        have h1 : v ≠ 0 := fun hh => hline (Or.inr (Or.inl hh))
        have h2 : u ≠ W := fun hh => hline (Or.inr (Or.inr (Or.inl hh)))
        have h3 : v ≠ W := fun hh => hline (Or.inr (Or.inr (Or.inr hh)))
        have hDv : (if u ≠ v ∧ v ≠ u ^^^ W ∧ (u = 0 ∨ v = 0 ∨ u = W ∨ v = W)
                    then (1:Nat) else 0) = 0 := if_neg (fun h => hline h.2.2)
        rw [hDv, Nat.zero_add]
        by_cases hq : Qgen' W v u (m+1) = -1
        · rw [if_pos ⟨hd, hc, Or.inr (Or.inr (Or.inr (Or.inr hq)))⟩,
              if_pos ⟨h1, h3, h0, h2, fun hh => hd hh.symm,
                      fun hh => hc (hxc.mpr hh), hq⟩]
        · have hno : ¬ (u ≠ v ∧ v ≠ u ^^^ W ∧
                        (u = 0 ∨ v = 0 ∨ u = W ∨ v = W ∨ Qgen' W v u (m+1) = -1)) := by
            intro h
            rcases h.2.2 with hh | hh | hh | hh | hh
            · exact h0 hh
            · exact h1 hh
            · exact h2 hh
            · exact h3 hh
            · exact hq hh
          rw [if_neg hno, if_neg (fun h => hq h.2.2.2.2.2.2)]

/-! ### Stage 3b: `ul` and `lu` factor too

Both land on the UNPRIMED `Qgen`, so besides the line values they need `Qgen_eq_Qgen'` OFF the
lines. Every line value is already in the tree: `Qgen_zero_left`, `Qgen_diag_neg`,
`Qgen_coset_left`/`_right` and `Qgen_degen`. -/

/-- Adding one fresh value to a disjunction adds exactly one to the count. -/
theorem sumLt_cons (n j : Nat) (p : Nat → Prop) [DecidablePred p] (hj : j < n) (hnp : ¬ p j) :
    sumLt n (fun i => if i = j ∨ p i then 1 else 0)
      = 1 + sumLt n (fun i => if p i then 1 else 0) := by
  have e1 : sumLt n (fun i => if i = j ∨ p i then 1 else 0)
      = sumLt n (fun i => if i = j then 1 else 0)
        + sumLt n (fun i => if i = j then 0 else if p i then 1 else 0) := by
    rw [← sumLt_split_if]
    apply sumLt_congr
    intro i _
    by_cases h : i = j
    · rw [if_pos h, if_pos (Or.inl h)]
    · rw [if_neg h]
      by_cases h2 : p i
      · rw [if_pos h2, if_pos (Or.inr h2)]
      · rw [if_neg h2, if_neg (fun hc => hc.elim h h2)]
  have e2 : sumLt n (fun i => if p i then 1 else 0)
      = sumLt n (fun i => if i = j then 0 else if p i then 1 else 0) := by
    apply sumLt_congr
    intro i _
    by_cases h : i = j
    · rw [if_pos h, h, if_neg hnp]
    · rw [if_neg h]
  rw [e1, e2, sumLt_single n j 1 hj]

/-- Pointwise: the `ul` (unprimed) summand is `OffCnt`'s plus FIVE boundary lines. -/
theorem qInd_split (W M u b : Nat) (hW : W < 2^M) (hW0 : W ≠ 0)
    (hu : u < 2^M) (hb : b < 2^M) :
    qInd W M u b
      = (if b ≠ 0 ∧ (u = 0 ∨ u = W ∨ b = W ∨ u = b ∨ b = u ^^^ W) then 1 else 0)
        + (if u ≠ 0 ∧ u ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ u ≠ b ∧ b ≠ u ^^^ W
              ∧ Qgen' W u b M = -1 then 1 else 0) := by
  unfold qInd
  by_cases hb0 : b = 0
  · have hL : (if b ≠ 0 ∧ Qgen W u b M = -1 then (1:Nat) else 0) = 0 :=
      if_neg (fun hc => hc.1 hb0)
    have hD : (if b ≠ 0 ∧ (u = 0 ∨ u = W ∨ b = W ∨ u = b ∨ b = u ^^^ W)
               then (1:Nat) else 0) = 0 := if_neg (fun hc => hc.1 hb0)
    have hO : (if u ≠ 0 ∧ u ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ u ≠ b ∧ b ≠ u ^^^ W
                  ∧ Qgen' W u b M = -1 then (1:Nat) else 0) = 0 :=
      if_neg (fun hc => hc.2.2.1 hb0)
    rw [hL, hD, hO]
  · by_cases hline : u = 0 ∨ u = W ∨ b = W ∨ u = b ∨ b = u ^^^ W
    · have hq : Qgen W u b M = -1 := by
        rcases hline with h | h | h | h | h
        · rw [h]; exact Qgen_zero_left M W b hW hb hW0
        · rw [h, Qgen_coset_left, Nat.xor_self]
          exact Qgen_zero_left M W b hW hb hW0
        · exact Qgen_degen M W u b hW hu hb hW0
            (Or.inr (Or.inr (Or.inr (Or.inl (by rw [h, Nat.xor_self])))))
        · rw [h]; exact Qgen_diag_neg M W b hW hb hW0
        · rw [h, ← Qgen_coset_right]
          exact Qgen_diag_neg M W u hW hu hW0
      have hO : (if u ≠ 0 ∧ u ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ u ≠ b ∧ b ≠ u ^^^ W
                    ∧ Qgen' W u b M = -1 then (1:Nat) else 0) = 0 := by
        apply if_neg
        intro hc
        rcases hline with h | h | h | h | h
        · exact hc.1 h
        · exact hc.2.1 h
        · exact hc.2.2.2.1 h
        · exact hc.2.2.2.2.1 h
        · exact hc.2.2.2.2.2.1 h
      rw [if_pos ⟨hb0, hq⟩, if_pos ⟨hb0, hline⟩, hO]
    · have h0 : u ≠ 0 := fun h => hline (Or.inl h)
      have h1 : u ≠ W := fun h => hline (Or.inr (Or.inl h))
      have h2 : b ≠ W := fun h => hline (Or.inr (Or.inr (Or.inl h)))
      have h3 : u ≠ b := fun h => hline (Or.inr (Or.inr (Or.inr (Or.inl h))))
      have h4 : b ≠ u ^^^ W := fun h => hline (Or.inr (Or.inr (Or.inr (Or.inr h))))
      have heq : Qgen W u b M = Qgen' W u b M :=
        Qgen_eq_Qgen' W u b M hu hb hW
          (fun h => h1 (xor_zero_eq u W h))
          (fun h => h2 (xor_zero_eq b W h))
          (by intro h
              apply h3
              have e1 : (u ^^^ W) = (b ^^^ W) := xor_zero_eq (u ^^^ W) (b ^^^ W) h
              have e2 : (u ^^^ W) ^^^ W = (b ^^^ W) ^^^ W := by rw [e1]
              rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero, Nat.xor_assoc, Nat.xor_self,
                  Nat.xor_zero] at e2
              exact e2)
          h0
          (by intro h
              apply h4
              have e1 : u = b ^^^ W := xor_zero_eq u (b ^^^ W) h
              rw [e1, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero])
      have hD : (if b ≠ 0 ∧ (u = 0 ∨ u = W ∨ b = W ∨ u = b ∨ b = u ^^^ W)
                 then (1:Nat) else 0) = 0 := if_neg (fun hc => hline hc.2)
      rw [hD, Nat.zero_add, heq]
      by_cases hq : Qgen' W u b M = -1
      · rw [if_pos ⟨hb0, hq⟩, if_pos ⟨h0, h1, hb0, h2, h3, h4, hq⟩]
      · rw [if_neg (fun hc => hq hc.2), if_neg (fun hc => hq hc.2.2.2.2.2.2)]

/-- Pointwise: the `lu` (unprimed, transposed) summand is `OffCnt`'s plus FOUR boundary lines.
    Only four, because the fifth -- `Qgen`'s `b ⊕ W = 0` degeneracy -- is `a = W`, which the
    quadrant's own guard already excludes. -/
theorem luInd_split (W M v a : Nat) (hW : W < 2^M) (hW0 : W ≠ 0)
    (hv : v < 2^M) (ha : a < 2^M) :
    (if a ≠ 0 ∧ a ≠ W ∧ Qgen W v a M = -1 then 1 else 0)
      = (if a ≠ 0 ∧ a ≠ W ∧ (v = 0 ∨ v = W ∨ v = a ∨ a = v ^^^ W) then 1 else 0)
        + (if v ≠ 0 ∧ v ≠ W ∧ a ≠ 0 ∧ a ≠ W ∧ v ≠ a ∧ a ≠ v ^^^ W
              ∧ Qgen' W v a M = -1 then 1 else 0) := by
  by_cases hg : a ≠ 0 ∧ a ≠ W
  · obtain ⟨ha0, haW⟩ := hg
    by_cases hline : v = 0 ∨ v = W ∨ v = a ∨ a = v ^^^ W
    · have hq : Qgen W v a M = -1 := by
        rcases hline with h | h | h | h
        · rw [h]; exact Qgen_zero_left M W a hW ha hW0
        · rw [h, Qgen_coset_left, Nat.xor_self]
          exact Qgen_zero_left M W a hW ha hW0
        · rw [h]; exact Qgen_diag_neg M W a hW ha hW0
        · rw [h, ← Qgen_coset_right]
          exact Qgen_diag_neg M W v hW hv hW0
      have hO : (if v ≠ 0 ∧ v ≠ W ∧ a ≠ 0 ∧ a ≠ W ∧ v ≠ a ∧ a ≠ v ^^^ W
                    ∧ Qgen' W v a M = -1 then (1:Nat) else 0) = 0 := by
        apply if_neg
        intro hc
        rcases hline with h | h | h | h
        · exact hc.1 h
        · exact hc.2.1 h
        · exact hc.2.2.2.2.1 h
        · exact hc.2.2.2.2.2.1 h
      rw [if_pos ⟨ha0, haW, hq⟩, if_pos ⟨ha0, haW, hline⟩, hO]
    · have h0 : v ≠ 0 := fun h => hline (Or.inl h)
      have h1 : v ≠ W := fun h => hline (Or.inr (Or.inl h))
      have h2 : v ≠ a := fun h => hline (Or.inr (Or.inr (Or.inl h)))
      have h3 : a ≠ v ^^^ W := fun h => hline (Or.inr (Or.inr (Or.inr h)))
      have heq : Qgen W v a M = Qgen' W v a M :=
        Qgen_eq_Qgen' W v a M hv ha hW
          (fun h => h1 (xor_zero_eq v W h))
          (fun h => haW (xor_zero_eq a W h))
          (by intro h
              apply h2
              have e1 : (v ^^^ W) = (a ^^^ W) := xor_zero_eq (v ^^^ W) (a ^^^ W) h
              have e2 : (v ^^^ W) ^^^ W = (a ^^^ W) ^^^ W := by rw [e1]
              rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero, Nat.xor_assoc, Nat.xor_self,
                  Nat.xor_zero] at e2
              exact e2)
          h0
          (by intro h
              apply h3
              have e1 : v = a ^^^ W := xor_zero_eq v (a ^^^ W) h
              rw [e1, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero])
      have hD : (if a ≠ 0 ∧ a ≠ W ∧ (v = 0 ∨ v = W ∨ v = a ∨ a = v ^^^ W)
                 then (1:Nat) else 0) = 0 := if_neg (fun hc => hline hc.2.2)
      rw [hD, Nat.zero_add, heq]
      by_cases hq : Qgen' W v a M = -1
      · rw [if_pos ⟨ha0, haW, hq⟩, if_pos ⟨h0, h1, ha0, haW, h2, h3, hq⟩]
      · rw [if_neg (fun hc => hq hc.2.2), if_neg (fun hc => hq hc.2.2.2.2.2.2)]
  · have hL : (if a ≠ 0 ∧ a ≠ W ∧ Qgen W v a M = -1 then (1:Nat) else 0) = 0 :=
      if_neg (fun hc => hg ⟨hc.1, hc.2.1⟩)
    have hD : (if a ≠ 0 ∧ a ≠ W ∧ (v = 0 ∨ v = W ∨ v = a ∨ a = v ^^^ W)
               then (1:Nat) else 0) = 0 := if_neg (fun hc => hg ⟨hc.1, hc.2.1⟩)
    have hO : (if v ≠ 0 ∧ v ≠ W ∧ a ≠ 0 ∧ a ≠ W ∧ v ≠ a ∧ a ≠ v ^^^ W
                  ∧ Qgen' W v a M = -1 then (1:Nat) else 0) = 0 :=
      if_neg (fun hc => hg ⟨hc.2.2.1, hc.2.2.2.1⟩)
    rw [hL, hD, hO]

/-! ### Stage 3c: counting the boundary sets -/

theorem sumLt_scale (n c : Nat) (p : Nat → Prop) [DecidablePred p] :
    sumLt n (fun i => if p i then c else 0)
      = c * sumLt n (fun i => if p i then 1 else 0) := by
  induction n with
  | zero => simp [sumLt]
  | succ n ih =>
      rw [sumLt, sumLt, ih]
      by_cases h : p n
      · rw [if_pos h, if_pos h, Nat.mul_add, Nat.mul_one]
      · rw [if_neg h, if_neg h, Nat.mul_add]; simp

/-- Four distinct values below `n` contribute exactly `4`. -/
theorem sumLt_four (n W a : Nat) (hW : W < n) (ha : a < n) (haW : a ^^^ W < n)
    (hW0 : W ≠ 0) (ha0 : a ≠ 0) (haWne : a ≠ W) :
    sumLt n (fun i => if i = 0 ∨ i = W ∨ i = a ∨ i = a ^^^ W then 1 else 0) = 4 := by
  have hax : a ^^^ W ≠ 0 := fun h => haWne (xor_zero_eq a W h)
  have haxW : a ^^^ W ≠ W := by
    intro h
    apply ha0
    have e : (a ^^^ W) ^^^ W = W ^^^ W := by rw [h]
    rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero] at e
    exact e
  have haxa : a ^^^ W ≠ a := by
    intro h
    apply hW0
    have e : a ^^^ (a ^^^ W) = a ^^^ a := by rw [h]
    rw [xorCancelL, Nat.xor_self] at e
    exact e
  have hn0 : 0 < n := by omega
  rw [sumLt_cons n 0 (fun i => i = W ∨ i = a ∨ i = a ^^^ W) hn0
        (by rintro (h | h | h)
            · exact hW0 h.symm
            · exact ha0 h.symm
            · exact hax h.symm),
      sumLt_cons n W (fun i => i = a ∨ i = a ^^^ W) hW
        (by rintro (h | h)
            · exact haWne h.symm
            · exact haxW h.symm),
      sumLt_cons n a (fun i => i = a ^^^ W) ha (fun h => haxa h.symm),
      sumLt_single n (a ^^^ W) 1 haW]

/-- The two-lines-removed count, extracted for reuse. -/
theorem count_off2 (W M : Nat) (hW : W < 2^M) (hW0 : W ≠ 0) :
    sumLt (2^M) (fun a => if a ≠ 0 ∧ a ≠ W then 1 else 0) + 2 = 2^M := by
  have hp : (0:Nat) < 2^M := Nat.two_pow_pos M
  have hc := sumLt_compl (2^M) (fun i => i = 0 ∨ i = W)
  have h2 := sumLt_two (2^M) W hW hW0 hp
  have he : sumLt (2^M) (fun i => if i = 0 ∨ i = W then 0 else 1)
      = sumLt (2^M) (fun a => if a ≠ 0 ∧ a ≠ W then 1 else 0) := by
    apply sumLt_congr
    intro i _
    by_cases h : i = 0 ∨ i = W
    · rw [if_pos h, if_neg (fun hc2 => h.elim hc2.1 hc2.2)]
    · rw [if_neg h, if_pos ⟨fun hh => h (Or.inl hh), fun hh => h (Or.inr hh)⟩]
  rw [he] at hc
  omega

/-- **The `lu` boundary count.** Four distinct values per surviving row, `2^M - 2` rows. -/
theorem lu_boundary (W M : Nat) (hW : W < 2^M) (hW0 : W ≠ 0) :
    sumLt (2^M) (fun a => sumLt (2^M) (fun v =>
      if a ≠ 0 ∧ a ≠ W ∧ (v = 0 ∨ v = W ∨ v = a ∨ a = v ^^^ W) then 1 else 0)) + 8
      = 4 * 2^M := by
  have hinner : ∀ a, a < 2^M →
      sumLt (2^M) (fun v =>
        if a ≠ 0 ∧ a ≠ W ∧ (v = 0 ∨ v = W ∨ v = a ∨ a = v ^^^ W) then 1 else 0)
        = if a ≠ 0 ∧ a ≠ W then 4 else 0 := by
    intro a hA
    by_cases hg : a ≠ 0 ∧ a ≠ W
    · rw [if_pos hg]
      have hxlt : a ^^^ W < 2^M := Nat.xor_lt_two_pow hA hW
      rw [sumLt_congr (2^M) _
            (fun v => if v = 0 ∨ v = W ∨ v = a ∨ v = a ^^^ W then 1 else 0)
            (fun v _ => by
              have hiff : (a = v ^^^ W) ↔ (v = a ^^^ W) := by
                constructor
                · intro hh; rw [hh, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
                · intro hh; rw [hh, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
              by_cases hd : v = 0 ∨ v = W ∨ v = a ∨ v = a ^^^ W
              · have hd' : v = 0 ∨ v = W ∨ v = a ∨ a = v ^^^ W := by
                  rcases hd with h | h | h | h
                  · exact Or.inl h
                  · exact Or.inr (Or.inl h)
                  · exact Or.inr (Or.inr (Or.inl h))
                  · exact Or.inr (Or.inr (Or.inr (hiff.mpr h)))
                rw [if_pos ⟨hg.1, hg.2, hd'⟩, if_pos hd]
              · rw [if_neg hd, if_neg (fun hc => by
                    rcases hc.2.2 with h | h | h | h
                    · exact hd (Or.inl h)
                    · exact hd (Or.inr (Or.inl h))
                    · exact hd (Or.inr (Or.inr (Or.inl h)))
                    · exact hd (Or.inr (Or.inr (Or.inr (hiff.mp h)))))])]
      exact sumLt_four (2^M) W a hW hA hxlt hW0 hg.1 hg.2
    · rw [if_neg hg,
          sumLt_congr (2^M) _ (fun _ => 0)
            (fun v _ => if_neg (fun hc => hg ⟨hc.1, hc.2.1⟩))]
      exact sumLt_zero _
  rw [sumLt_congr (2^M) _ (fun a => if a ≠ 0 ∧ a ≠ W then 4 else 0) hinner,
      sumLt_scale]
  have := count_off2 W M hW hW0
  omega

/-- Three distinct values below `n` contribute exactly `3`. -/
theorem sumLt_three (n x y z : Nat) (hx : x < n) (hy : y < n) (hz : z < n)
    (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) :
    sumLt n (fun i => if i = x ∨ i = y ∨ i = z then 1 else 0) = 3 := by
  rw [sumLt_cons n x (fun i => i = y ∨ i = z) hx
        (by rintro (h | h)
            · exact hxy h
            · exact hxz h),
      sumLt_cons n y (fun i => i = z) hy (fun h => hyz h),
      sumLt_single n z 1 hz]

/-- One line removed. -/
theorem count_off1 (M : Nat) :
    sumLt (2^M) (fun b => if b ≠ 0 then 1 else 0) + 1 = 2^M := by
  have hp : (0:Nat) < 2^M := Nat.two_pow_pos M
  have hc := sumLt_compl (2^M) (fun i => i = 0)
  have h1 : sumLt (2^M) (fun i => if i = 0 then 1 else 0) = 1 :=
    sumLt_single (2^M) 0 1 hp
  have he : sumLt (2^M) (fun i => if i = 0 then 0 else 1)
      = sumLt (2^M) (fun b => if b ≠ 0 then 1 else 0) := by
    apply sumLt_congr
    intro i _
    by_cases h : i = 0
    · rw [if_pos h, if_neg (fun hc2 => hc2 h)]
    · rw [if_neg h, if_pos h]
  rw [he, h1] at hc
  omega

/-- **The `ul` boundary count.** Two full rows of `2^M - 1`, then three values on each of the
    remaining `2^M - 2` rows. -/
theorem ul_boundary (W M : Nat) (hW : W < 2^M) (hW0 : W ≠ 0) :
    sumLt (2^M) (fun u => sumLt (2^M) (fun b =>
      if b ≠ 0 ∧ (u = 0 ∨ u = W ∨ b = W ∨ u = b ∨ b = u ^^^ W) then 1 else 0)) + 8
      = 5 * 2^M := by
  have hp : (0:Nat) < 2^M := Nat.two_pow_pos M
  have hinner : ∀ u, u < 2^M →
      sumLt (2^M) (fun b =>
        if b ≠ 0 ∧ (u = 0 ∨ u = W ∨ b = W ∨ u = b ∨ b = u ^^^ W) then 1 else 0)
        = if u = 0 ∨ u = W then 2^M - 1 else 3 := by
    intro u hU
    by_cases hg : u = 0 ∨ u = W
    · rw [if_pos hg,
          sumLt_congr (2^M) _ (fun b => if b ≠ 0 then 1 else 0)
            (fun b _ => by
              by_cases hb : b ≠ 0
              · rw [if_pos ⟨hb, hg.elim (fun h => Or.inl h) (fun h => Or.inr (Or.inl h))⟩,
                    if_pos hb]
              · rw [if_neg (fun hc => hb hc.1), if_neg hb])]
      have := count_off1 M
      omega
    · have h0 : u ≠ 0 := fun h => hg (Or.inl h)
      have h1 : u ≠ W := fun h => hg (Or.inr h)
      have hxlt : u ^^^ W < 2^M := Nat.xor_lt_two_pow hU hW
      have hx0 : u ^^^ W ≠ 0 := fun h => h1 (xor_zero_eq u W h)
      have hxW : u ^^^ W ≠ W := by
        intro h
        apply h0
        have e : (u ^^^ W) ^^^ W = W ^^^ W := by rw [h]
        rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero] at e
        exact e
      have hxu : u ^^^ W ≠ u := by
        intro h
        apply hW0
        have e : u ^^^ (u ^^^ W) = u ^^^ u := by rw [h]
        rw [xorCancelL, Nat.xor_self] at e
        exact e
      rw [if_neg hg,
          sumLt_congr (2^M) _ (fun b => if b = W ∨ b = u ∨ b = u ^^^ W then 1 else 0)
            (fun b _ => by
              by_cases hd : b = W ∨ b = u ∨ b = u ^^^ W
              · have hb0 : b ≠ 0 := by
                  rcases hd with h | h | h
                  · rw [h]; exact hW0
                  · rw [h]; exact h0
                  · rw [h]; exact hx0
                have hd' : u = 0 ∨ u = W ∨ b = W ∨ u = b ∨ b = u ^^^ W := by
                  rcases hd with h | h | h
                  · exact Or.inr (Or.inr (Or.inl h))
                  · exact Or.inr (Or.inr (Or.inr (Or.inl h.symm)))
                  · exact Or.inr (Or.inr (Or.inr (Or.inr h)))
                rw [if_pos ⟨hb0, hd'⟩, if_pos hd]
              · rw [if_neg hd, if_neg (fun hc => by
                    rcases hc.2 with h | h | h | h | h
                    · exact h0 h
                    · exact h1 h
                    · exact hd (Or.inl h)
                    · exact hd (Or.inr (Or.inl h.symm))
                    · exact hd (Or.inr (Or.inr h)))])]
      exact sumLt_three (2^M) W u (u ^^^ W) hW hU hxlt (fun h => h1 h.symm)
        (fun h => hxW h.symm) (fun h => hxu h.symm)
  rw [sumLt_congr (2^M) _ (fun u => if u = 0 ∨ u = W then 2^M - 1 else 3) hinner,
      sumLt_split_if]
  have e1 : sumLt (2^M) (fun u => if u = 0 ∨ u = W then 2^M - 1 else 0)
      = (2^M - 1) * 2 := by
    rw [sumLt_scale, sumLt_two (2^M) W hW hW0 hp]
  have e2 : sumLt (2^M) (fun u => if u = 0 ∨ u = W then 0 else 3)
      = 3 * sumLt (2^M) (fun a => if a ≠ 0 ∧ a ≠ W then 1 else 0) := by
    rw [← sumLt_scale]
    apply sumLt_congr
    intro i _
    by_cases h : i = 0 ∨ i = W
    · rw [if_pos h, if_neg (fun hc => h.elim hc.1 hc.2)]
    · rw [if_neg h, if_pos ⟨fun hh => h (Or.inl hh), fun hh => h (Or.inr hh)⟩]
  rw [e1, e2]
  have := count_off2 W M hW hW0
  omega

/-- **The `uu` boundary count.** Two rows of `2^M - 2`, then two values on each of the
    remaining `2^M - 2` rows. -/
theorem uu_boundary (W M : Nat) (hW : W < 2^M) (hW0 : W ≠ 0) :
    sumLt (2^M) (fun u => sumLt (2^M) (fun v =>
      if u ≠ v ∧ v ≠ u ^^^ W ∧ (u = 0 ∨ v = 0 ∨ u = W ∨ v = W) then 1 else 0)) + 8
      = 4 * 2^M := by
  have hp : (0:Nat) < 2^M := Nat.two_pow_pos M
  have hinner : ∀ u, u < 2^M →
      sumLt (2^M) (fun v =>
        if u ≠ v ∧ v ≠ u ^^^ W ∧ (u = 0 ∨ v = 0 ∨ u = W ∨ v = W) then 1 else 0)
        = if u = 0 ∨ u = W then 2^M - 2 else 2 := by
    intro u hU
    by_cases hg : u = 0 ∨ u = W
    · rw [if_pos hg,
          sumLt_congr (2^M) _ (fun v => if v ≠ 0 ∧ v ≠ W then 1 else 0)
            (fun v _ => by
              rcases hg with h | h
              · have hx : u ^^^ W = W := by rw [h, Nat.zero_xor]
                by_cases hd : v ≠ 0 ∧ v ≠ W
                · rw [if_pos ⟨fun hh => hd.1 (by rw [← hh, h]), by rw [hx]; exact hd.2,
                        Or.inl h⟩, if_pos hd]
                · rw [if_neg hd, if_neg (fun hc => by
                      apply hd
                      exact ⟨fun hh => hc.1 (by rw [h, hh]), by
                        have := hc.2.1; rw [hx] at this; exact this⟩)]
              · have hx : u ^^^ W = 0 := by rw [h, Nat.xor_self]
                by_cases hd : v ≠ 0 ∧ v ≠ W
                · rw [if_pos ⟨fun hh => hd.2 (by rw [← hh, h]), by rw [hx]; exact hd.1,
                        Or.inr (Or.inr (Or.inl h))⟩, if_pos hd]
                · rw [if_neg hd, if_neg (fun hc => by
                      apply hd
                      refine ⟨?_, fun hh => hc.1 (by rw [h, hh])⟩
                      have := hc.2.1; rw [hx] at this; exact this)])]
      have := count_off2 W M hW hW0
      omega
    · have h0 : u ≠ 0 := fun h => hg (Or.inl h)
      have h1 : u ≠ W := fun h => hg (Or.inr h)
      have hx0 : u ^^^ W ≠ 0 := fun h => h1 (xor_zero_eq u W h)
      have hxW : u ^^^ W ≠ W := by
        intro h
        apply h0
        have e : (u ^^^ W) ^^^ W = W ^^^ W := by rw [h]
        rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero] at e
        exact e
      rw [if_neg hg,
          sumLt_congr (2^M) _ (fun v => if v = 0 ∨ v = W then 1 else 0)
            (fun v _ => by
              by_cases hd : v = 0 ∨ v = W
              · have hne : u ≠ v := by
                  rcases hd with h | h
                  · rw [h]; exact h0
                  · rw [h]; exact h1
                have hnx : v ≠ u ^^^ W := by
                  rcases hd with h | h
                  · rw [h]; exact fun hh => hx0 hh.symm
                  · rw [h]; exact fun hh => hxW hh.symm
                have hdis : u = 0 ∨ v = 0 ∨ u = W ∨ v = W := by
                  rcases hd with h | h
                  · exact Or.inr (Or.inl h)
                  · exact Or.inr (Or.inr (Or.inr h))
                rw [if_pos ⟨hne, hnx, hdis⟩, if_pos hd]
              · rw [if_neg hd, if_neg (fun hc => by
                    rcases hc.2.2 with h | h | h | h
                    · exact h0 h
                    · exact hd (Or.inl h)
                    · exact h1 h
                    · exact hd (Or.inr h))])]
      exact sumLt_two (2^M) W hW hW0 hp
  rw [sumLt_congr (2^M) _ (fun u => if u = 0 ∨ u = W then 2^M - 2 else 2) hinner,
      sumLt_split_if]
  have e1 : sumLt (2^M) (fun u => if u = 0 ∨ u = W then 2^M - 2 else 0)
      = (2^M - 2) * 2 := by
    rw [sumLt_scale, sumLt_two (2^M) W hW hW0 hp]
  have e2 : sumLt (2^M) (fun u => if u = 0 ∨ u = W then 0 else 2)
      = 2 * sumLt (2^M) (fun a => if a ≠ 0 ∧ a ≠ W then 1 else 0) := by
    rw [← sumLt_scale]
    apply sumLt_congr
    intro i _
    by_cases h : i = 0 ∨ i = W
    · rw [if_pos h, if_neg (fun hc => h.elim hc.1 hc.2)]
    · rw [if_neg h, if_pos ⟨fun hh => h (Or.inl hh), fun hh => h (Or.inr hh)⟩]
  rw [e1, e2]
  have := count_off2 W M hW hW0
  omega

/-! ### Stage 3d: the assembly -/

theorem Mcnt_eq (W M : Nat) (hW : W < 2^M) (hW0 : W ≠ 0) :
    Mcnt W M + 8 = OffCnt W M + 5 * 2^M := by
  have hstep : Mcnt W M
      = sumLt (2^M) (fun u => sumLt (2^M) (fun b =>
          if b ≠ 0 ∧ (u = 0 ∨ u = W ∨ b = W ∨ u = b ∨ b = u ^^^ W) then 1 else 0))
        + OffCnt W M := by
    unfold Mcnt OffCnt
    rw [← sumLt_pair]
    apply sumLt_congr
    intro u hU
    rw [sumLt_congr (2^M) _
          (fun b => (if b ≠ 0 ∧ (u = 0 ∨ u = W ∨ b = W ∨ u = b ∨ b = u ^^^ W)
                     then 1 else 0)
            + (if u ≠ 0 ∧ u ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ u ≠ b ∧ b ≠ u ^^^ W
                 ∧ Qgen' W u b M = -1 then 1 else 0))
          (fun b hB => qInd_split W M u b hW hW0 hU hB),
        sumLt_pair]
  have := ul_boundary W M hW hW0
  omega

theorem LuSum_eq (W M : Nat) (hW : W < 2^M) (hW0 : W ≠ 0) :
    sumLt (2^M) (fun a => sumLt (2^M) (fun v =>
        if a ≠ 0 ∧ a ≠ W ∧ Qgen W v a M = -1 then 1 else 0)) + 8
      = OffCnt W M + 4 * 2^M := by
  have hstep : sumLt (2^M) (fun a => sumLt (2^M) (fun v =>
        if a ≠ 0 ∧ a ≠ W ∧ Qgen W v a M = -1 then 1 else 0))
      = sumLt (2^M) (fun a => sumLt (2^M) (fun v =>
          if a ≠ 0 ∧ a ≠ W ∧ (v = 0 ∨ v = W ∨ v = a ∨ a = v ^^^ W) then 1 else 0))
        + OffCnt W M := by
    unfold OffCnt
    rw [sumLt_swap (2^M) (fun x y => if x ≠ 0 ∧ x ≠ W ∧ y ≠ 0 ∧ y ≠ W ∧ x ≠ y
                                       ∧ y ≠ x ^^^ W ∧ Qgen' W x y M = -1 then 1 else 0)]
    rw [← sumLt_pair]
    apply sumLt_congr
    intro a hA
    rw [sumLt_congr (2^M) _
          (fun v => (if a ≠ 0 ∧ a ≠ W ∧ (v = 0 ∨ v = W ∨ v = a ∨ a = v ^^^ W)
                     then 1 else 0)
            + (if v ≠ 0 ∧ v ≠ W ∧ a ≠ 0 ∧ a ≠ W ∧ v ≠ a ∧ a ≠ v ^^^ W
                 ∧ Qgen' W v a M = -1 then 1 else 0))
          (fun v hV => luInd_split W M v a hW hW0 hV hA),
        sumLt_pair]
  have := lu_boundary W M hW hW0
  omega

theorem UuSum_eq (W m : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0) :
    sumLt (2^(m+1)) (fun u => sumLt (2^(m+1)) (fun v => uuInd W m u v)) + 8
      = OffCnt W (m+1) + 4 * 2^(m+1) := by
  have hstep : sumLt (2^(m+1)) (fun u => sumLt (2^(m+1)) (fun v => uuInd W m u v))
      = sumLt (2^(m+1)) (fun u => sumLt (2^(m+1)) (fun v =>
          if u ≠ v ∧ v ≠ u ^^^ W ∧ (u = 0 ∨ v = 0 ∨ u = W ∨ v = W) then 1 else 0))
        + OffCnt W (m+1) := by
    unfold OffCnt
    rw [sumLt_swap (2^(m+1)) (fun x y => if x ≠ 0 ∧ x ≠ W ∧ y ≠ 0 ∧ y ≠ W ∧ x ≠ y
                                           ∧ y ≠ x ^^^ W ∧ Qgen' W x y (m+1) = -1
                                         then 1 else 0)]
    rw [← sumLt_pair]
    apply sumLt_congr
    intro u hU
    rw [sumLt_congr (2^(m+1)) _
          (fun v => (if u ≠ v ∧ v ≠ u ^^^ W ∧ (u = 0 ∨ v = 0 ∨ u = W ∨ v = W)
                     then 1 else 0)
            + (if v ≠ 0 ∧ v ≠ W ∧ u ≠ 0 ∧ u ≠ W ∧ v ≠ u ∧ u ≠ v ^^^ W
                 ∧ Qgen' W v u (m+1) = -1 then 1 else 0))
          (fun v hV => uuInd_split W m u v hW hW0 hU hV),
        sumLt_pair]
  have := uu_boundary W (m+1) hW hW0
  omega

/-- **THE LOW RECURSION, ∀n.** The `W15` ledger is now a Lean theorem. -/
theorem Ncnt_low (W m : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0) :
    Ncnt W (m+2) + 18 = 4 * Ncnt W (m+1) + 10 * 2^(m+1) := by
  have huu : sumLt (2^(m+1)) (fun u =>
        sumLt (2^(m+1)) (fun v => nInd W (m+2) (2^(m+1) + u) (2^(m+1) + v)))
      = sumLt (2^(m+1)) (fun u => sumLt (2^(m+1)) (fun v => uuInd W m u v)) := by
    apply sumLt_congr
    intro u hU
    apply sumLt_congr
    intro v hV
    exact Ncnt_uu_low W m u v hW hW0 hU hV
  rw [Ncnt_quad, Ncnt_ll_low W m hW, Ncnt_ul_low W m hW, Ncnt_lu_low W m hW hW0, huu]
  have h1 := Ncnt_eq_OffCnt W (m+1) hW hW0
  have h2 := Mcnt_eq W (m+1) hW hW0
  have h3 := LuSum_eq W (m+1) hW hW0
  have h4 := UuSum_eq W m hW hW0
  have hp : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  omega

/-! ## Tier 30: the HIGH branch

Every `hi` row carries a MINUS sign, so "count the `-1`s at level `m+2`" becomes "count the
`+1`s at level `m+1`". That reflection is the only structural difference from the LOW branch,
and it needs one new object: the POSITIVE core `OffCntP`. -/

theorem Qgen'_pm (W a b m : Nat) : Qgen' W a b m = 1 ∨ Qgen' W a b m = -1 := by
  unfold Qgen'
  rcases cdSigma_pm m a b with h1 | h1 <;>
    rcases cdSigma_pm m (b ^^^ W) (a ^^^ W) with h2 | h2 <;>
    rcases cdSigma_pm m (b ^^^ W) a with h3 | h3 <;>
    rcases cdSigma_pm m (a ^^^ W) b with h4 | h4 <;>
    rw [h1, h2, h3, h4] <;> decide

/-- The positive core: the count OFF all six lines where `Q'` is `+1`. -/
def OffCntP (W m : Nat) : Nat :=
  sumLt (2^m) (fun a => sumLt (2^m) (fun b =>
    if a ≠ 0 ∧ a ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ a ≠ b ∧ b ≠ a ^^^ W ∧ Qgen' W a b m = 1
    then 1 else 0))

/-- `OffCnt` and `OffCntP` exhaust the off-lines domain, which has `(2^M-2)(2^M-4)` points. -/
theorem OffCnt_add_OffCntP (W M : Nat) (hW : W < 2^M) (hW0 : W ≠ 0) :
    OffCnt W M + OffCntP W M + 6 * 2^M = 2^M * 2^M + 8 := by
  have hp : (0:Nat) < 2^M := Nat.two_pow_pos M
  have hdom : OffCnt W M + OffCntP W M
      = sumLt (2^M) (fun a => sumLt (2^M) (fun b =>
          if a ≠ 0 ∧ a ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ a ≠ b ∧ b ≠ a ^^^ W then 1 else 0)) := by
    unfold OffCnt OffCntP
    rw [← sumLt_pair]
    apply sumLt_congr
    intro a _
    rw [← sumLt_pair]
    apply sumLt_congr
    intro b _
    by_cases hc : a ≠ 0 ∧ a ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ a ≠ b ∧ b ≠ a ^^^ W
    · rw [if_pos hc]
      rcases Qgen'_pm W a b M with h | h
      · rw [if_neg (fun hh => by rw [h] at hh; exact absurd hh.2.2.2.2.2.2 (by decide)),
            if_pos ⟨hc.1, hc.2.1, hc.2.2.1, hc.2.2.2.1, hc.2.2.2.2.1, hc.2.2.2.2.2, h⟩]
      · rw [if_pos ⟨hc.1, hc.2.1, hc.2.2.1, hc.2.2.2.1, hc.2.2.2.2.1, hc.2.2.2.2.2, h⟩,
            if_neg (fun hh => by rw [h] at hh; exact absurd hh.2.2.2.2.2.2 (by decide))]
    · rw [if_neg hc,
          if_neg (fun hh => hc ⟨hh.1, hh.2.1, hh.2.2.1, hh.2.2.2.1, hh.2.2.2.2.1,
                                hh.2.2.2.2.2.1⟩),
          if_neg (fun hh => hc ⟨hh.1, hh.2.1, hh.2.2.1, hh.2.2.2.1, hh.2.2.2.2.1,
                                hh.2.2.2.2.2.1⟩)]
  have hinner : ∀ a, a < 2^M →
      sumLt (2^M) (fun b =>
        if a ≠ 0 ∧ a ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ a ≠ b ∧ b ≠ a ^^^ W then 1 else 0)
        = if a ≠ 0 ∧ a ≠ W then 2^M - 4 else 0 := by
    intro a hA
    by_cases hg : a ≠ 0 ∧ a ≠ W
    · rw [if_pos hg]
      have hxlt : a ^^^ W < 2^M := Nat.xor_lt_two_pow hA hW
      have hf := sumLt_four (2^M) W a hW hA hxlt hW0 hg.1 hg.2
      have hc := sumLt_compl (2^M) (fun i => i = 0 ∨ i = W ∨ i = a ∨ i = a ^^^ W)
      have he : sumLt (2^M) (fun i => if i = 0 ∨ i = W ∨ i = a ∨ i = a ^^^ W then 0 else 1)
          = sumLt (2^M) (fun b =>
              if a ≠ 0 ∧ a ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ a ≠ b ∧ b ≠ a ^^^ W then 1 else 0) := by
        apply sumLt_congr
        intro b _
        by_cases hd : b = 0 ∨ b = W ∨ b = a ∨ b = a ^^^ W
        · rw [if_pos hd]
          refine (if_neg ?_).symm
          intro hh
          rcases hd with h | h | h | h
          · exact hh.2.2.1 h
          · exact hh.2.2.2.1 h
          · exact hh.2.2.2.2.1 h.symm
          · exact hh.2.2.2.2.2 h
        · rw [if_neg hd]
          refine (if_pos ⟨hg.1, hg.2, ?_, ?_, ?_, ?_⟩).symm
          · exact fun h => hd (Or.inl h)
          · exact fun h => hd (Or.inr (Or.inl h))
          · exact fun h => hd (Or.inr (Or.inr (Or.inl h.symm)))
          · exact fun h => hd (Or.inr (Or.inr (Or.inr h)))
      rw [he] at hc
      omega
    · rw [if_neg hg,
          sumLt_congr (2^M) _ (fun _ => 0)
            (fun b _ => if_neg (fun hh => hg ⟨hh.1, hh.2.1⟩))]
      exact sumLt_zero _
  rw [hdom, sumLt_congr (2^M) _ (fun a => if a ≠ 0 ∧ a ≠ W then 2^M - 4 else 0) hinner,
      sumLt_scale]
  have hco := count_off2 W M hW hW0
  -- A nonempty label forces `2 <= 2^M`, and a power of two that is not `2` is at least `4`.
  -- The `2^M = 2` box is the `m = 0` bottom: it holds, but by Nat truncation rather than by
  -- the `(e-2)(e-4)` expansion, so it needs its own line.
  have hcase : 2^M = 2 ∨ 4 ≤ 2^M := by
    cases M with
    | zero => exact absurd (by have h1 : (2:Nat)^0 = 1 := rfl; omega) hW0
    | succ j =>
        cases j with
        | zero => exact Or.inl rfl
        | succ i =>
            refine Or.inr ?_
            have hle : (2:Nat)^2 ≤ 2^(i+1+1) := Nat.pow_le_pow_right (by omega) (by omega)
            have he : (2:Nat)^2 = 4 := rfl
            omega
  rcases hcase with h2 | h4
  · have hs : sumLt (2^M) (fun a => if a ≠ 0 ∧ a ≠ W then 1 else 0) = 0 := by omega
    rw [hs, Nat.mul_zero, h2]
  · obtain ⟨k, hk4⟩ : ∃ k, 2^M = k + 4 := ⟨2^M - 4, by omega⟩
    have hcnt : sumLt (2^M) (fun a => if a ≠ 0 ∧ a ≠ W then 1 else 0) = k + 2 := by omega
    rw [hcnt, hk4]
    have hsub : k + 4 - 4 = k := by omega
    have e1 : (k + 4 - 4) * (k + 2) = k * k + k * 2 := by rw [hsub, Nat.mul_add]
    have e2 : (k + 4) * (k + 4) = k * k + k * 4 + (4 * k + 4 * 4) := by
      rw [Nat.add_mul, Nat.mul_add, Nat.mul_add]
    omega

/-- **ll boundary, `a = W`.** The `Q'red_hi_ll` row needs `a ^^^ W ≠ 0`; on the excluded
    locus the same reduction still runs, but the `R_uu` branch flips and the level-`(m+1)` value
    is the LABEL value, which `Qgen'_label_left` already pins at `+1`. -/
theorem Qgen'_hi_ll_aW (m W b : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (hb : b < 2^(m+1)) (hb0 : b ≠ 0) (hbW : b ≠ W) :
    Qgen' (W + 2^(m+1)) W b (m+2) = -1 := by
  have hpos : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have hself : W ^^^ W = 0 := Nat.xor_self W
  have hlab : Qgen' W W b (m+1) = 1 := Qgen'_label_left (m+1) W b hW hb hW0 hb0 hbW
  unfold Qgen' at hlab
  rw [hself, cdSig0' (b ^^^ W) m, cdSig0 b m] at hlab
  have hxa : W ^^^ (W + 2^(m+1)) = (W ^^^ W) + 2^(m+1) := xor_seam W W m hW hW
  have hxb : b ^^^ (W + 2^(m+1)) = (b ^^^ W) + 2^(m+1) := xor_seam b W m hb hW
  unfold Qgen'
  rw [hxa, hxb, hself, R_ll W b m hW hb,
      R_uu (b ^^^ W) 0 m (xorlt hb hW) hpos, if_pos rfl,
      R_ul (b ^^^ W) W m (xorlt hb hW) hW, if_neg hW0,
      R_ul 0 b m hpos hb, if_neg hb0, cdSig0 b m]
  rcases cdSigma_pm (m+1) W b with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (b ^^^ W) W with h2 | h2 <;>
    rw [h1, h2] at hlab ⊢ <;> revert hlab <;> decide

/-- The `chi` value on a coset pair `(u, u ^^^ W)`: `-1` whenever `u` avoids `0` and `W`. -/
private theorem chi_uW (m W u : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0) (hu : u < 2^(m+1))
    (hu0 : u ≠ 0) (huW : u ≠ W) :
    cdSigma u (u ^^^ W) (m+1) * cdSigma (u ^^^ W) u (m+1) = -1 := by
  have huX : u ^^^ W ≠ 0 := fun h => huW (xor_zero_eq u W h)
  have hne : u ≠ u ^^^ W := by
    intro h
    apply hW0
    have h2 : u ^^^ u = u ^^^ (u ^^^ W) := by rw [← h]
    rw [Nat.xor_self, xorCancelL] at h2
    exact h2.symm
  have h := chi_char (m+1) u (u ^^^ W) hu (xorlt hu hW)
  unfold chi at h
  rw [if_neg (fun hc => hc.elim hu0 (fun hc2 => hc2.elim huX hne))] at h
  exact h

/-- **ll boundary, `b = W`.** -/
theorem Qgen'_hi_ll_bW (m W a : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (ha : a < 2^(m+1)) (ha0 : a ≠ 0) (haW : a ≠ W) :
    Qgen' (W + 2^(m+1)) a W (m+2) = -1 := by
  have hpos : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have haX : a ^^^ W ≠ 0 := fun h => haW (xor_zero_eq a W h)
  have hlab : Qgen' W a W (m+1) = -1 := Qgen'_label_right (m+1) W a hW ha hW0
  unfold Qgen' at hlab
  rw [Nat.xor_self, cdSig0 (a ^^^ W) m, cdSig0 a m] at hlab
  have hxa : a ^^^ (W + 2^(m+1)) = (a ^^^ W) + 2^(m+1) := xor_seam a W m ha hW
  have hxb : W ^^^ (W + 2^(m+1)) = (W ^^^ W) + 2^(m+1) := xor_seam W W m hW hW
  unfold Qgen'
  rw [hxa, hxb, Nat.xor_self, R_ll a W m ha hW,
      R_uu 0 (a ^^^ W) m hpos (xorlt ha hW), if_neg haX, cdSig0' (a ^^^ W) m,
      R_ul 0 a m hpos ha, if_neg ha0, cdSig0 a m,
      R_ul (a ^^^ W) W m (xorlt ha hW) hW, if_neg hW0]
  rcases cdSigma_pm (m+1) a W with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (a ^^^ W) W with h2 | h2 <;>
    rw [h1, h2] at hlab ⊢ <;> revert hlab <;> decide

/-- **ul boundary, `u = 0`, `b ∉ {0, W}`.** -/
theorem Qgen'_hi_ul_u0 (m W b : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (hb : b < 2^(m+1)) (hb0 : b ≠ 0) (hbW : b ≠ W) :
    Qgen' (W + 2^(m+1)) (0 + 2^(m+1)) b (m+2) = -1 := by
  have hpos : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have hlab : Qgen' W W b (m+1) = 1 := Qgen'_label_left (m+1) W b hW hb hW0 hb0 hbW
  unfold Qgen' at hlab
  rw [Nat.xor_self, cdSig0' (b ^^^ W) m, cdSig0 b m] at hlab
  have hxa : (0 + 2^(m+1)) ^^^ (W + 2^(m+1)) = 0 ^^^ W := xor_seam_cancel 0 W m hpos hW
  have hxb : b ^^^ (W + 2^(m+1)) = (b ^^^ W) + 2^(m+1) := xor_seam b W m hb hW
  unfold Qgen'
  rw [hxa, hxb, Nat.zero_xor, R_ul 0 b m hpos hb, if_neg hb0, cdSig0 b m,
      R_ul (b ^^^ W) W m (xorlt hb hW) hW, if_neg hW0,
      R_uu (b ^^^ W) 0 m (xorlt hb hW) hpos, if_pos rfl,
      R_ll W b m hW hb]
  rcases cdSigma_pm (m+1) W b with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (b ^^^ W) W with h2 | h2 <;>
    rw [h1, h2] at hlab ⊢ <;> revert hlab <;> decide

/-- **ul boundary, `u = 0`, `b = W`** -- the one point of that slice that does NOT count. -/
theorem Qgen'_hi_ul_u0W (m W : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0) :
    Qgen' (W + 2^(m+1)) (0 + 2^(m+1)) W (m+2) = 1 := by
  have hpos : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have hxa : (0 + 2^(m+1)) ^^^ (W + 2^(m+1)) = 0 ^^^ W := xor_seam_cancel 0 W m hpos hW
  have hxb : W ^^^ (W + 2^(m+1)) = (W ^^^ W) + 2^(m+1) := xor_seam W W m hW hW
  unfold Qgen'
  rw [hxa, hxb, Nat.zero_xor, Nat.xor_self,
      R_ul 0 W m hpos hW, if_neg hW0, cdSig0 W m,
      R_uu 0 0 m hpos hpos, if_pos rfl,
      R_ll W W m hW hW, sigma_self (m+1) W hW hW0]
  decide

/-- **ul boundary, `b = W`, `u ∉ {0, W}`.** -/
theorem Qgen'_hi_ul_bW (m W u : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (hu : u < 2^(m+1)) (hu0 : u ≠ 0) (huW : u ≠ W) :
    Qgen' (W + 2^(m+1)) (u + 2^(m+1)) W (m+2) = -1 := by
  have hpos : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have huX : u ^^^ W ≠ 0 := fun h => huW (xor_zero_eq u W h)
  have hlab : Qgen' W u W (m+1) = -1 := Qgen'_label_right (m+1) W u hW hu hW0
  unfold Qgen' at hlab
  rw [Nat.xor_self, cdSig0 (u ^^^ W) m, cdSig0 u m] at hlab
  have hxa : (u + 2^(m+1)) ^^^ (W + 2^(m+1)) = u ^^^ W := xor_seam_cancel u W m hu hW
  have hxb : W ^^^ (W + 2^(m+1)) = (W ^^^ W) + 2^(m+1) := xor_seam W W m hW hW
  unfold Qgen'
  rw [hxa, hxb, Nat.xor_self, R_ul u W m hu hW, if_neg hW0,
      R_ul 0 (u ^^^ W) m hpos (xorlt hu hW), if_neg huX, cdSig0 (u ^^^ W) m,
      R_uu 0 u m hpos hu, if_neg hu0, cdSig0' u m,
      R_ll (u ^^^ W) W m (xorlt hu hW) hW]
  rcases cdSigma_pm (m+1) u W with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (u ^^^ W) W with h2 | h2 <;>
    rw [h1, h2] at hlab ⊢ <;> revert hlab <;> decide

/-- **ul boundary, the diagonal `b = u`, `u ∉ {0, W}`.** -/
theorem Qgen'_hi_ul_ub (m W u : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (hu : u < 2^(m+1)) (hu0 : u ≠ 0) (huW : u ≠ W) :
    Qgen' (W + 2^(m+1)) (u + 2^(m+1)) u (m+2) = -1 := by
  have huX : u ^^^ W ≠ 0 := fun h => huW (xor_zero_eq u W h)
  have hchi := chi_uW m W u hW hW0 hu hu0 huW
  have hxa : (u + 2^(m+1)) ^^^ (W + 2^(m+1)) = u ^^^ W := xor_seam_cancel u W m hu hW
  have hxb : u ^^^ (W + 2^(m+1)) = (u ^^^ W) + 2^(m+1) := xor_seam u W m hu hW
  unfold Qgen'
  rw [hxa, hxb, R_ul u u m hu hu, if_neg hu0, sigma_self (m+1) u hu hu0,
      R_ul (u ^^^ W) (u ^^^ W) m (xorlt hu hW) (xorlt hu hW), if_neg huX,
      sigma_self (m+1) (u ^^^ W) (xorlt hu hW) huX,
      R_uu (u ^^^ W) u m (xorlt hu hW) hu, if_neg hu0,
      R_ll (u ^^^ W) u m (xorlt hu hW) hu]
  rcases cdSigma_pm (m+1) u (u ^^^ W) with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (u ^^^ W) u with h2 | h2 <;>
    rw [h1, h2] at hchi ⊢ <;> revert hchi <;> decide

/-- **lu boundary, `v = 0`, `a ∉ {0, W}`.** -/
theorem Qgen'_hi_lu_v0 (m W a : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (ha : a < 2^(m+1)) (ha0 : a ≠ 0) (haW : a ≠ W) :
    Qgen' (W + 2^(m+1)) a (0 + 2^(m+1)) (m+2) = -1 := by
  have hpos : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have hlab : Qgen' W W a (m+1) = 1 := Qgen'_label_left (m+1) W a hW ha hW0 ha0 haW
  unfold Qgen' at hlab
  rw [Nat.xor_self, cdSig0' (a ^^^ W) m, cdSig0 a m] at hlab
  have hxa : a ^^^ (W + 2^(m+1)) = (a ^^^ W) + 2^(m+1) := xor_seam a W m ha hW
  have hxb : (0 + 2^(m+1)) ^^^ (W + 2^(m+1)) = 0 ^^^ W := xor_seam_cancel 0 W m hpos hW
  unfold Qgen'
  rw [hxa, hxb, Nat.zero_xor, R_lu a 0 m ha hpos, cdSig0 a m,
      R_lu W (a ^^^ W) m hW (xorlt ha hW),
      R_ll W a m hW ha,
      R_uu (a ^^^ W) 0 m (xorlt ha hW) hpos, if_pos rfl]
  rcases cdSigma_pm (m+1) W a with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (a ^^^ W) W with h2 | h2 <;>
    rw [h1, h2] at hlab ⊢ <;> revert hlab <;> decide

/-- **lu boundary, `v = 0`, `a = W`** -- the one point of that slice that does NOT count. -/
theorem Qgen'_hi_lu_v0W (m W : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0) :
    Qgen' (W + 2^(m+1)) W (0 + 2^(m+1)) (m+2) = 1 := by
  have hpos : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have hxa : W ^^^ (W + 2^(m+1)) = (W ^^^ W) + 2^(m+1) := xor_seam W W m hW hW
  have hxb : (0 + 2^(m+1)) ^^^ (W + 2^(m+1)) = 0 ^^^ W := xor_seam_cancel 0 W m hpos hW
  unfold Qgen'
  rw [hxa, hxb, Nat.zero_xor, Nat.xor_self,
      R_lu W 0 m hW hpos, cdSig0 W m,
      R_ll W W m hW hW, sigma_self (m+1) W hW hW0,
      R_uu 0 0 m hpos hpos, if_pos rfl]
  decide

/-- **lu boundary, `a = W`, `v ≠ 0`.** -/
theorem Qgen'_hi_lu_aW (m W v : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (hv : v < 2^(m+1)) (hv0 : v ≠ 0) :
    Qgen' (W + 2^(m+1)) W (v + 2^(m+1)) (m+2) = -1 := by
  have hpos : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have hlab : Qgen' W v W (m+1) = -1 := Qgen'_label_right (m+1) W v hW hv hW0
  unfold Qgen' at hlab
  rw [Nat.xor_self, cdSig0 (v ^^^ W) m, cdSig0 v m] at hlab
  have hxa : W ^^^ (W + 2^(m+1)) = (W ^^^ W) + 2^(m+1) := xor_seam W W m hW hW
  have hxb : (v + 2^(m+1)) ^^^ (W + 2^(m+1)) = v ^^^ W := xor_seam_cancel v W m hv hW
  unfold Qgen'
  rw [hxa, hxb, Nat.xor_self, R_lu W v m hW hv,
      R_lu (v ^^^ W) 0 m (xorlt hv hW) hpos, cdSig0 (v ^^^ W) m,
      R_ll (v ^^^ W) W m (xorlt hv hW) hW,
      R_uu 0 v m hpos hv, if_neg hv0, cdSig0' v m]
  rcases cdSigma_pm (m+1) v W with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (v ^^^ W) W with h2 | h2 <;>
    rw [h1, h2] at hlab ⊢ <;> revert hlab <;> decide

/-- **lu boundary, the diagonal `a = v`, `v ∉ {0, W}`.** -/
theorem Qgen'_hi_lu_av (m W v : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (hv : v < 2^(m+1)) (hv0 : v ≠ 0) (hvW : v ≠ W) :
    Qgen' (W + 2^(m+1)) v (v + 2^(m+1)) (m+2) = -1 := by
  have hvX : v ^^^ W ≠ 0 := fun h => hvW (xor_zero_eq v W h)
  have hchi := chi_uW m W v hW hW0 hv hv0 hvW
  have hxa : v ^^^ (W + 2^(m+1)) = (v ^^^ W) + 2^(m+1) := xor_seam v W m hv hW
  have hxb : (v + 2^(m+1)) ^^^ (W + 2^(m+1)) = v ^^^ W := xor_seam_cancel v W m hv hW
  unfold Qgen'
  rw [hxa, hxb, R_lu v v m hv hv, sigma_self (m+1) v hv hv0,
      R_lu (v ^^^ W) (v ^^^ W) m (xorlt hv hW) (xorlt hv hW),
      sigma_self (m+1) (v ^^^ W) (xorlt hv hW) hvX,
      R_ll (v ^^^ W) v m (xorlt hv hW) hv,
      R_uu (v ^^^ W) v m (xorlt hv hW) hv, if_neg hv0]
  rcases cdSigma_pm (m+1) v (v ^^^ W) with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (v ^^^ W) v with h2 | h2 <;>
    rw [h1, h2] at hchi ⊢ <;> revert hchi <;> decide

/-- **uu boundary, `u = 0`, `v ≠ 0`.** -/
theorem Qgen'_hi_uu_u0 (m W v : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (hv : v < 2^(m+1)) (hv0 : v ≠ 0) :
    Qgen' (W + 2^(m+1)) (0 + 2^(m+1)) (v + 2^(m+1)) (m+2) = -1 := by
  have hpos : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have hlab : Qgen' W v W (m+1) = -1 := Qgen'_label_right (m+1) W v hW hv hW0
  unfold Qgen' at hlab
  rw [Nat.xor_self, cdSig0 (v ^^^ W) m, cdSig0 v m] at hlab
  have hxa : (0 + 2^(m+1)) ^^^ (W + 2^(m+1)) = 0 ^^^ W := xor_seam_cancel 0 W m hpos hW
  have hxb : (v + 2^(m+1)) ^^^ (W + 2^(m+1)) = v ^^^ W := xor_seam_cancel v W m hv hW
  unfold Qgen'
  rw [hxa, hxb, Nat.zero_xor, R_uu 0 v m hpos hv, if_neg hv0, cdSig0' v m,
      R_ll (v ^^^ W) W m (xorlt hv hW) hW,
      R_lu (v ^^^ W) 0 m (xorlt hv hW) hpos, cdSig0 (v ^^^ W) m,
      R_lu W v m hW hv]
  rcases cdSigma_pm (m+1) v W with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (v ^^^ W) W with h2 | h2 <;>
    rw [h1, h2] at hlab ⊢ <;> revert hlab <;> decide

/-- **uu boundary, `v = 0`, `u ∉ {0, W}`.** -/
theorem Qgen'_hi_uu_v0 (m W u : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (hu : u < 2^(m+1)) (hu0 : u ≠ 0) (huW : u ≠ W) :
    Qgen' (W + 2^(m+1)) (u + 2^(m+1)) (0 + 2^(m+1)) (m+2) = -1 := by
  have hpos : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have huX : u ^^^ W ≠ 0 := fun h => huW (xor_zero_eq u W h)
  have hbW : u ^^^ W ≠ W := by
    intro h
    apply hu0
    have h2 : (u ^^^ W) ^^^ W = W ^^^ W := by rw [h]
    rw [xor_cancel u W, Nat.xor_self] at h2
    exact h2
  have hlab : Qgen' W W (u ^^^ W) (m+1) = 1 :=
    Qgen'_label_left (m+1) W (u ^^^ W) hW (xorlt hu hW) hW0 huX hbW
  unfold Qgen' at hlab
  rw [Nat.xor_self, xor_cancel u W, cdSig0' u m, cdSig0 (u ^^^ W) m] at hlab
  have hxa : (u + 2^(m+1)) ^^^ (W + 2^(m+1)) = u ^^^ W := xor_seam_cancel u W m hu hW
  have hxb : (0 + 2^(m+1)) ^^^ (W + 2^(m+1)) = 0 ^^^ W := xor_seam_cancel 0 W m hpos hW
  unfold Qgen'
  rw [hxa, hxb, Nat.zero_xor, R_uu u 0 m hu hpos, if_pos rfl,
      R_ll W (u ^^^ W) m hW (xorlt hu hW),
      R_lu W u m hW hu,
      R_lu (u ^^^ W) 0 m (xorlt hu hW) hpos, cdSig0 (u ^^^ W) m]
  rcases cdSigma_pm (m+1) W (u ^^^ W) with h1 | h1 <;>
    rcases cdSigma_pm (m+1) u W with h2 | h2 <;>
    rw [h1, h2] at hlab ⊢ <;> revert hlab <;> decide

/-- **uu boundary, the coset diagonal `v = u ^^^ W`, `u ∉ {0, W}`.** -/
theorem Qgen'_hi_uu_cos (m W u : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (hu : u < 2^(m+1)) (hu0 : u ≠ 0) (huW : u ≠ W) :
    Qgen' (W + 2^(m+1)) (u + 2^(m+1)) ((u ^^^ W) + 2^(m+1)) (m+2) = -1 := by
  have huX : u ^^^ W ≠ 0 := fun h => huW (xor_zero_eq u W h)
  have hchi := chi_uW m W u hW hW0 hu hu0 huW
  have hxa : (u + 2^(m+1)) ^^^ (W + 2^(m+1)) = u ^^^ W := xor_seam_cancel u W m hu hW
  have hxb : ((u ^^^ W) + 2^(m+1)) ^^^ (W + 2^(m+1)) = (u ^^^ W) ^^^ W :=
    xor_seam_cancel (u ^^^ W) W m (xorlt hu hW) hW
  unfold Qgen'
  rw [hxa, hxb, xor_cancel u W,
      R_uu u (u ^^^ W) m hu (xorlt hu hW), if_neg huX,
      R_ll u (u ^^^ W) m hu (xorlt hu hW),
      R_lu u u m hu hu, sigma_self (m+1) u hu hu0,
      R_lu (u ^^^ W) (u ^^^ W) m (xorlt hu hW) (xorlt hu hW),
      sigma_self (m+1) (u ^^^ W) (xorlt hu hW) huX]
  rcases cdSigma_pm (m+1) u (u ^^^ W) with h1 | h1 <;>
    rcases cdSigma_pm (m+1) (u ^^^ W) u with h2 | h2 <;>
    rw [h1, h2] at hchi ⊢ <;> revert hchi <;> decide

/-! ### The four HIGH quadrants -/

/-- Pointwise split of the HIGH `ll` quadrant: the `a = W` row, the `b = W` column, and the
    row-domain remainder, on which `Q'red_hi_ll` REFLECTS the level-`(m+1)` sign -- so the
    remainder counts `Q' = +1`, not `-1`. -/
theorem hi_ll_split (m W a b : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (ha : a < 2^(m+1)) (hb : b < 2^(m+1)) :
    nInd (W + 2^(m+1)) (m+2) a b
      = (if a = W ∧ b ≠ 0 ∧ b ≠ W then 1 else 0)
        + ((if b = W ∧ a ≠ 0 ∧ a ≠ W then 1 else 0)
           + (if a ≠ 0 ∧ a ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ a ≠ b ∧ Qgen' W a b (m+1) = 1
              then 1 else 0)) := by
  unfold nInd
  by_cases haW : a = W
  · have ha0 : a ≠ 0 := by rw [haW]; exact hW0
    have h2 : (if b = W ∧ a ≠ 0 ∧ a ≠ W then (1:Nat) else 0) = 0 :=
      if_neg (fun hc => hc.2.2 haW)
    have h3 : (if a ≠ 0 ∧ a ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ a ≠ b ∧ Qgen' W a b (m+1) = 1
                then (1:Nat) else 0) = 0 := if_neg (fun hc => hc.2.1 haW)
    rw [h2, h3]
    by_cases hb0 : b = 0
    · rw [if_neg (fun hc => hc.2.1 hb0), if_neg (fun hc => hc.2.1 hb0)]
    · by_cases hbW : b = W
      · have hab : a = b := by rw [haW, hbW]
        rw [if_neg (fun hc => hc.2.2.1 hab), if_neg (fun hc => hc.2.2 hbW)]
      · have hq : Qgen' (W + 2^(m+1)) a b (m+2) = -1 := by
          rw [haW]; exact Qgen'_hi_ll_aW m W b hW hW0 hb hb0 hbW
        have hab : a ≠ b := by rw [haW]; exact fun h => hbW h.symm
        rw [if_pos ⟨ha0, hb0, hab, hq⟩, if_pos ⟨haW, hb0, hbW⟩]
  · have h1 : (if a = W ∧ b ≠ 0 ∧ b ≠ W then (1:Nat) else 0) = 0 :=
      if_neg (fun hc => haW hc.1)
    rw [h1, Nat.zero_add]
    by_cases ha0 : a = 0
    · rw [if_neg (fun hc => hc.1 ha0), if_neg (fun hc => hc.2.1 ha0),
          if_neg (fun hc => hc.1 ha0)]
    · by_cases hbW : b = W
      · have hb0 : b ≠ 0 := by rw [hbW]; exact hW0
        have hab : a ≠ b := by rw [hbW]; exact haW
        have hq : Qgen' (W + 2^(m+1)) a b (m+2) = -1 := by
          rw [hbW]; exact Qgen'_hi_ll_bW m W a hW hW0 ha ha0 haW
        rw [if_pos ⟨ha0, hb0, hab, hq⟩, if_pos ⟨hbW, ha0, haW⟩,
            if_neg (fun hc => hc.2.2.2.1 hbW)]
      · have h2 : (if b = W ∧ a ≠ 0 ∧ a ≠ W then (1:Nat) else 0) = 0 :=
          if_neg (fun hc => hbW hc.1)
        rw [h2, Nat.zero_add]
        by_cases hb0 : b = 0
        · rw [if_neg (fun hc => hc.2.1 hb0), if_neg (fun hc => hc.2.2.1 hb0)]
        · by_cases hab : a = b
          · rw [if_neg (fun hc => hc.2.2.1 hab), if_neg (fun hc => hc.2.2.2.2.1 hab)]
          · have haX : a ^^^ W ≠ 0 := fun h => haW (xor_zero_eq a W h)
            have hbX : b ^^^ W ≠ 0 := fun h => hbW (xor_zero_eq b W h)
            have hrow : Qgen' (W + 2^(m+1)) a b (m+2) = - Qgen' W a b (m+1) :=
              Q'red_hi_ll m W a b hW ha hb ha0 hb0 haX hbX hab
            by_cases hq : Qgen' W a b (m+1) = 1
            · have hq2 : Qgen' (W + 2^(m+1)) a b (m+2) = -1 := by rw [hrow, hq]
              rw [if_pos ⟨ha0, hb0, hab, hq2⟩, if_pos ⟨ha0, haW, hb0, hbW, hab, hq⟩]
            · have hq2 : Qgen' (W + 2^(m+1)) a b (m+2) ≠ -1 := by
                rw [hrow]
                rcases Qgen'_pm W a b (m+1) with h | h
                · exact absurd h hq
                · rw [h]; decide
              rw [if_neg (fun hc => hq2 hc.2.2.2), if_neg (fun hc => hq hc.2.2.2.2.2)]

/-- The coset diagonal `b = a ^^^ W` splits off the `ll` remainder, and there `Q' = +1`
    (`Qgen'_coset_partner`) -- so it COUNTS, which is why the `ll` quadrant carries `3(e-2)`
    rather than `2(e-2)`. -/
theorem hi_ll_cos_split (m W a b : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (ha : a < 2^(m+1)) :
    (if a ≠ 0 ∧ a ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ a ≠ b ∧ Qgen' W a b (m+1) = 1 then (1:Nat) else 0)
      = (if a ≠ 0 ∧ a ≠ W ∧ b = a ^^^ W then 1 else 0)
        + (if a ≠ 0 ∧ a ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ a ≠ b ∧ b ≠ a ^^^ W ∧ Qgen' W a b (m+1) = 1
           then 1 else 0) := by
  by_cases hg : a ≠ 0 ∧ a ≠ W
  · have haX : a ^^^ W ≠ 0 := fun h => hg.2 (xor_zero_eq a W h)
    by_cases hcos : b = a ^^^ W
    · have hb0 : b ≠ 0 := by rw [hcos]; exact haX
      have hbW : b ≠ W := by
        rw [hcos]
        intro h
        apply hg.1
        have h2 : (a ^^^ W) ^^^ W = W ^^^ W := by rw [h]
        rw [xor_cancel a W, Nat.xor_self] at h2
        exact h2
      have hab : a ≠ b := by
        rw [hcos]
        intro h
        apply hW0
        have h2 : a ^^^ a = a ^^^ (a ^^^ W) := by rw [← h]
        rw [Nat.xor_self, xorCancelL] at h2
        exact h2.symm
      have hq : Qgen' W a b (m+1) = 1 := by
        rw [hcos]; exact Qgen'_coset_partner (m+1) W a hW ha hg.1 haX
      rw [if_pos ⟨hg.1, hg.2, hb0, hbW, hab, hq⟩, if_pos ⟨hg.1, hg.2, hcos⟩,
          if_neg (fun hc => hc.2.2.2.2.2.1 hcos)]
    · have hz : (if a ≠ 0 ∧ a ≠ W ∧ b = a ^^^ W then (1:Nat) else 0) = 0 :=
        if_neg (fun hc => hcos hc.2.2)
      rw [hz, Nat.zero_add]
      by_cases hc2 : b ≠ 0 ∧ b ≠ W ∧ a ≠ b ∧ Qgen' W a b (m+1) = 1
      · rw [if_pos ⟨hg.1, hg.2, hc2.1, hc2.2.1, hc2.2.2.1, hc2.2.2.2⟩,
            if_pos ⟨hg.1, hg.2, hc2.1, hc2.2.1, hc2.2.2.1, hcos, hc2.2.2.2⟩]
      · rw [if_neg (fun hc => hc2 ⟨hc.2.2.1, hc.2.2.2.1, hc.2.2.2.2.1, hc.2.2.2.2.2⟩),
            if_neg (fun hc => hc2 ⟨hc.2.2.1, hc.2.2.2.1, hc.2.2.2.2.1, hc.2.2.2.2.2.2⟩)]
  · rw [if_neg (fun hc => hg ⟨hc.1, hc.2.1⟩), if_neg (fun hc => hg ⟨hc.1, hc.2.1⟩),
        if_neg (fun hc => hg ⟨hc.1, hc.2.1⟩)]

/-- The `a = W` row of the `ll` quadrant: `2^(m+1) - 2` points, all counting. -/
private theorem hi_ll_row_a (m W a : Nat) (hW : W < 2^(m+1)) :
    sumLt (2^(m+1)) (fun b => if a = W ∧ b ≠ 0 ∧ b ≠ W then 1 else 0)
      = if a = W then sumLt (2^(m+1)) (fun b => if b ≠ 0 ∧ b ≠ W then 1 else 0) else 0 := by
  by_cases hg : a = W
  · rw [if_pos hg]
    apply sumLt_congr
    intro b _
    by_cases hc : b ≠ 0 ∧ b ≠ W
    · rw [if_pos ⟨hg, hc.1, hc.2⟩, if_pos hc]
    · rw [if_neg (fun hh => hc ⟨hh.2.1, hh.2.2⟩), if_neg hc]
  · rw [if_neg hg,
        sumLt_congr (2^(m+1)) _ (fun _ => 0) (fun b _ => if_neg (fun hh => hg hh.1))]
    exact sumLt_zero _

/-- The coset diagonal of the `ll` remainder: one point per `a` off the two lines. -/
private theorem hi_ll_cos_row (m W a : Nat) (hW : W < 2^(m+1)) (ha : a < 2^(m+1)) :
    sumLt (2^(m+1)) (fun b => if a ≠ 0 ∧ a ≠ W ∧ b = a ^^^ W then 1 else 0)
      = if a ≠ 0 ∧ a ≠ W then 1 else 0 := by
  by_cases hg : a ≠ 0 ∧ a ≠ W
  · rw [if_pos hg,
        sumLt_congr (2^(m+1)) _ (fun b => if b = a ^^^ W then 1 else 0)
          (fun b _ => by
            by_cases hc : b = a ^^^ W
            · rw [if_pos ⟨hg.1, hg.2, hc⟩, if_pos hc]
            · rw [if_neg (fun hh => hc hh.2.2), if_neg hc])]
    exact sumLt_single (2^(m+1)) (a ^^^ W) 1 (xorlt ha hW)
  · rw [if_neg hg,
        sumLt_congr (2^(m+1)) _ (fun _ => 0)
          (fun b _ => if_neg (fun hh => hg ⟨hh.1, hh.2.1⟩))]
    exact sumLt_zero _

/-- **The `ll` quadrant of the HIGH box.** Three copies of the off-lines count `2^(m+1) - 2`
    -- the `a = W` row, the `b = W` column, and the coset diagonal -- on top of the positive
    core. -/
theorem Ncnt_ll_hi (m W : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0) :
    sumLt (2^(m+1)) (fun a => sumLt (2^(m+1)) (fun b => nInd (W + 2^(m+1)) (m+2) a b)) + 6
      = OffCntP W (m+1) + 3 * 2^(m+1) := by
  have hsplit : sumLt (2^(m+1)) (fun a =>
        sumLt (2^(m+1)) (fun b => nInd (W + 2^(m+1)) (m+2) a b))
      = sumLt (2^(m+1)) (fun a =>
            sumLt (2^(m+1)) (fun b => if a = W ∧ b ≠ 0 ∧ b ≠ W then 1 else 0))
        + (sumLt (2^(m+1)) (fun a =>
              sumLt (2^(m+1)) (fun b => if b = W ∧ a ≠ 0 ∧ a ≠ W then 1 else 0))
           + (sumLt (2^(m+1)) (fun a =>
                 sumLt (2^(m+1)) (fun b => if a ≠ 0 ∧ a ≠ W ∧ b = a ^^^ W then 1 else 0))
              + sumLt (2^(m+1)) (fun a =>
                  sumLt (2^(m+1)) (fun b =>
                    if a ≠ 0 ∧ a ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ a ≠ b ∧ b ≠ a ^^^ W
                       ∧ Qgen' W a b (m+1) = 1 then 1 else 0)))) := by
    rw [sumLt_congr (2^(m+1)) _ (fun a =>
          sumLt (2^(m+1)) (fun b => if a = W ∧ b ≠ 0 ∧ b ≠ W then 1 else 0)
          + (sumLt (2^(m+1)) (fun b => if b = W ∧ a ≠ 0 ∧ a ≠ W then 1 else 0)
             + (sumLt (2^(m+1)) (fun b => if a ≠ 0 ∧ a ≠ W ∧ b = a ^^^ W then 1 else 0)
                + sumLt (2^(m+1)) (fun b =>
                    if a ≠ 0 ∧ a ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ a ≠ b ∧ b ≠ a ^^^ W
                       ∧ Qgen' W a b (m+1) = 1 then 1 else 0))))
        (fun a hA => by
          rw [sumLt_congr (2^(m+1)) _ _ (fun b hB => by
                rw [hi_ll_split m W a b hW hW0 hA hB, hi_ll_cos_split m W a b hW hW0 hA]),
              sumLt_pair, sumLt_pair, sumLt_pair]),
        sumLt_pair, sumLt_pair, sumLt_pair]
  rw [hsplit,
      sumLt_congr (2^(m+1)) _ _ (fun a hA => hi_ll_row_a m W a hW),
      sumLt_congr (2^(m+1)) _ _ (fun a hA => col_W W (m+1) a hW),
      sumLt_congr (2^(m+1)) _ _ (fun a hA => hi_ll_cos_row m W a hW hA),
      sumLt_single (2^(m+1)) W
        (sumLt (2^(m+1)) (fun b => if b ≠ 0 ∧ b ≠ W then 1 else 0)) hW]
  unfold OffCntP
  have hc2 := count_off2 W (m+1) hW hW0
  omega

/-- Generic row: a guard times a single point. -/
private theorem row_at (n c : Nat) (hc : c < n) (p : Prop) [Decidable p] :
    sumLt n (fun b => if p ∧ b = c then 1 else 0) = if p then 1 else 0 := by
  by_cases hg : p
  · rw [if_pos hg,
        sumLt_congr n _ (fun b => if b = c then 1 else 0)
          (fun b _ => by
            by_cases hb : b = c
            · rw [if_pos ⟨hg, hb⟩, if_pos hb]
            · rw [if_neg (fun hh => hb hh.2), if_neg hb])]
    exact sumLt_single n c 1 hc
  · rw [if_neg hg, sumLt_congr n _ (fun _ => 0) (fun b _ => if_neg (fun hh => hg hh.1))]
    exact sumLt_zero _

/-- Generic row: a guard times the off-lines count. -/
private theorem row_off (n W : Nat) (p : Prop) [Decidable p] :
    sumLt n (fun b => if p ∧ b ≠ 0 ∧ b ≠ W then 1 else 0)
      = if p then sumLt n (fun b => if b ≠ 0 ∧ b ≠ W then 1 else 0) else 0 := by
  by_cases hg : p
  · rw [if_pos hg]
    apply sumLt_congr
    intro b _
    by_cases hb : b ≠ 0 ∧ b ≠ W
    · rw [if_pos ⟨hg, hb.1, hb.2⟩, if_pos hb]
    · rw [if_neg (fun hh => hb ⟨hh.2.1, hh.2.2⟩), if_neg hb]
  · rw [if_neg hg, sumLt_congr n _ (fun _ => 0) (fun b _ => if_neg (fun hh => hg hh.1))]
    exact sumLt_zero _

/-- Pointwise split of the HIGH `ul` quadrant. The `u = W` row is the whole `label_left` row of
    the level-`(m+2)` label `W + 2^(m+1)`, so it contributes NOTHING and never appears below. -/
theorem hi_ul_split (m W u b : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (hu : u < 2^(m+1)) (hb : b < 2^(m+1)) :
    nInd (W + 2^(m+1)) (m+2) (2^(m+1) + u) b
      = (if u = 0 ∧ b ≠ 0 ∧ b ≠ W then 1 else 0)
        + ((if (u ≠ 0 ∧ u ≠ W) ∧ b = W then 1 else 0)
           + ((if (u ≠ 0 ∧ u ≠ W) ∧ b = u then 1 else 0)
              + (if u ≠ 0 ∧ u ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ u ≠ b ∧ b ≠ u ^^^ W
                    ∧ Qgen' W u b (m+1) = 1 then 1 else 0))) := by
  have hp : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have hcu : (2:Nat)^(m+1) + u = u + 2^(m+1) := by omega
  unfold nInd
  rw [hcu]
  have ha0 : u + 2^(m+1) ≠ 0 := by omega
  have hab : u + 2^(m+1) ≠ b := by omega
  by_cases hu0 : u = 0
  · have h2 : (if (u ≠ 0 ∧ u ≠ W) ∧ b = W then (1:Nat) else 0) = 0 :=
      if_neg (fun hc => hc.1.1 hu0)
    have h3 : (if (u ≠ 0 ∧ u ≠ W) ∧ b = u then (1:Nat) else 0) = 0 :=
      if_neg (fun hc => hc.1.1 hu0)
    have h4 : (if u ≠ 0 ∧ u ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ u ≠ b ∧ b ≠ u ^^^ W
                  ∧ Qgen' W u b (m+1) = 1 then (1:Nat) else 0) = 0 := if_neg (fun hc => hc.1 hu0)
    rw [h2, h3, h4]
    subst hu0
    by_cases hb0 : b = 0
    · rw [if_neg (fun hc => hc.2.1 hb0), if_neg (fun hc => hc.2.1 hb0)]
    · by_cases hbW : b = W
      · have hq : Qgen' (W + 2^(m+1)) (0 + 2^(m+1)) b (m+2) = 1 := by
          rw [hbW]; exact Qgen'_hi_ul_u0W m W hW hW0
        rw [if_neg (fun hc => by rw [hq] at hc; exact absurd hc.2.2.2 (by decide)),
            if_neg (fun hc => hc.2.2 hbW)]
      · rw [if_pos ⟨ha0, hb0, hab, Qgen'_hi_ul_u0 m W b hW hW0 hb hb0 hbW⟩,
            if_pos ⟨rfl, hb0, hbW⟩]
  · have h1 : (if u = 0 ∧ b ≠ 0 ∧ b ≠ W then (1:Nat) else 0) = 0 := if_neg (fun hc => hu0 hc.1)
    rw [h1, Nat.zero_add]
    by_cases huW : u = W
    · have h2 : (if (u ≠ 0 ∧ u ≠ W) ∧ b = W then (1:Nat) else 0) = 0 :=
        if_neg (fun hc => hc.1.2 huW)
      have h3 : (if (u ≠ 0 ∧ u ≠ W) ∧ b = u then (1:Nat) else 0) = 0 :=
        if_neg (fun hc => hc.1.2 huW)
      have h4 : (if u ≠ 0 ∧ u ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ u ≠ b ∧ b ≠ u ^^^ W
                    ∧ Qgen' W u b (m+1) = 1 then (1:Nat) else 0) = 0 :=
        if_neg (fun hc => hc.2.1 huW)
      rw [h2, h3, h4]
      have hlt : W + 2^(m+1) < 2^(m+2) := by
        have : (2:Nat)^(m+2) = 2^(m+1) + 2^(m+1) := by rw [Nat.pow_succ]; omega
        omega
      have hblt : b < 2^(m+2) := by
        have : (2:Nat)^(m+2) = 2^(m+1) + 2^(m+1) := by rw [Nat.pow_succ]; omega
        omega
      by_cases hb0 : b = 0
      · exact if_neg (fun hc => hc.2.1 hb0)
      · have hq : Qgen' (W + 2^(m+1)) (u + 2^(m+1)) b (m+2) = 1 := by
          rw [huW]
          exact Qgen'_label_left (m+2) (W + 2^(m+1)) b hlt hblt (by omega) hb0 (by omega)
        exact if_neg (fun hc => by rw [hq] at hc; exact absurd hc.2.2.2 (by decide))
    · have huX : u ^^^ W ≠ 0 := fun h => huW (xor_zero_eq u W h)
      by_cases hbW : b = W
      · have hb0 : b ≠ 0 := by rw [hbW]; exact hW0
        have hbu : b ≠ u := by rw [hbW]; exact fun h => huW h.symm
        have hq : Qgen' (W + 2^(m+1)) (u + 2^(m+1)) b (m+2) = -1 := by
          rw [hbW]; exact Qgen'_hi_ul_bW m W u hW hW0 hu hu0 huW
        rw [if_pos ⟨ha0, hb0, hab, hq⟩, if_pos ⟨⟨hu0, huW⟩, hbW⟩,
            if_neg (fun hc => hc.1.2 (hc.2.symm.trans hbW)),
            if_neg (fun hc => hc.2.2.2.1 hbW)]
      · have h2 : (if (u ≠ 0 ∧ u ≠ W) ∧ b = W then (1:Nat) else 0) = 0 :=
          if_neg (fun hc => hbW hc.2)
        rw [h2, Nat.zero_add]
        by_cases hbu : b = u
        · have hb0 : b ≠ 0 := by rw [hbu]; exact hu0
          have hq : Qgen' (W + 2^(m+1)) (u + 2^(m+1)) b (m+2) = -1 := by
            rw [hbu]; exact Qgen'_hi_ul_ub m W u hW hW0 hu hu0 huW
          rw [if_pos ⟨ha0, hb0, hab, hq⟩, if_pos ⟨⟨hu0, huW⟩, hbu⟩,
              if_neg (fun hc => hc.2.2.2.2.1 hbu.symm)]
        · have h3 : (if (u ≠ 0 ∧ u ≠ W) ∧ b = u then (1:Nat) else 0) = 0 :=
            if_neg (fun hc => hbu hc.2)
          rw [h3, Nat.zero_add]
          by_cases hb0 : b = 0
          · rw [if_neg (fun hc => hc.2.1 hb0), if_neg (fun hc => hc.2.2.1 hb0)]
          · have hbX : b ^^^ W ≠ 0 := fun h => hbW (xor_zero_eq b W h)
            have hub : u ≠ b := fun h => hbu h.symm
            have hrow : Qgen' (W + 2^(m+1)) (u + 2^(m+1)) b (m+2) = - Qgen W u b (m+1) :=
              Q'red_hi_ul m W u b hW hu hb hu0 hb0 huX hbX hub
            by_cases hcos : b = u ^^^ W
            · have hz : u ^^^ b ^^^ W = 0 := by
                rw [hcos, xorCancelL, Nat.xor_self]
              have hqm : Qgen W u b (m+1) = -1 :=
                Qgen_degen (m+1) W u b hW hu hb hW0
                  (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr hz)))))
              have hq : Qgen' (W + 2^(m+1)) (u + 2^(m+1)) b (m+2) = 1 := by
                rw [hrow, hqm]; decide
              rw [if_neg (fun hc => by rw [hq] at hc; exact absurd hc.2.2.2 (by decide)),
                  if_neg (fun hc => hc.2.2.2.2.2.1 hcos)]
            · have h3ne : (u ^^^ W) ^^^ (b ^^^ W) ≠ 0 := by
                intro h
                apply hub
                have : u ^^^ b = 0 := by
                  rw [Nat.xor_assoc, ← Nat.xor_assoc W b W, Nat.xor_comm W b,
                      Nat.xor_assoc b W W, Nat.xor_self, Nat.xor_zero] at h
                  exact h
                exact xor_zero_eq u b this
              have h5ne : u ^^^ (b ^^^ W) ≠ 0 := by
                intro h
                apply hcos
                have hu2 : u = b ^^^ W := xor_zero_eq u (b ^^^ W) h
                rw [hu2, xor_cancel b W]
              have hbridge : Qgen W u b (m+1) = Qgen' W u b (m+1) :=
                Qgen_eq_Qgen' W u b (m+1) hu hb hW huX hbX h3ne hu0 h5ne
              by_cases hq1 : Qgen' W u b (m+1) = 1
              · have hq : Qgen' (W + 2^(m+1)) (u + 2^(m+1)) b (m+2) = -1 := by
                  rw [hrow, hbridge, hq1]
                rw [if_pos ⟨ha0, hb0, hab, hq⟩,
                    if_pos ⟨hu0, huW, hb0, hbW, hub, hcos, hq1⟩]
              · have hq : Qgen' (W + 2^(m+1)) (u + 2^(m+1)) b (m+2) ≠ -1 := by
                  rw [hrow, hbridge]
                  rcases Qgen'_pm W u b (m+1) with h | h
                  · exact absurd h hq1
                  · rw [h]; decide
                rw [if_neg (fun hc => hq hc.2.2.2),
                    if_neg (fun hc => hq1 hc.2.2.2.2.2.2)]

/-- **The `ul` quadrant of the HIGH box.** Same total as `ll`, but from three DIFFERENT slices:
    the `u = 0` row, the `b = W` column, and the diagonal `b = u`. The `u = W` row is the label
    row and contributes nothing. -/
theorem Ncnt_ul_hi (m W : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0) :
    sumLt (2^(m+1)) (fun u =>
        sumLt (2^(m+1)) (fun b => nInd (W + 2^(m+1)) (m+2) (2^(m+1) + u) b)) + 6
      = OffCntP W (m+1) + 3 * 2^(m+1) := by
  have hp : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have hsplit : sumLt (2^(m+1)) (fun u =>
        sumLt (2^(m+1)) (fun b => nInd (W + 2^(m+1)) (m+2) (2^(m+1) + u) b))
      = sumLt (2^(m+1)) (fun u =>
            sumLt (2^(m+1)) (fun b => if u = 0 ∧ b ≠ 0 ∧ b ≠ W then 1 else 0))
        + (sumLt (2^(m+1)) (fun u =>
              sumLt (2^(m+1)) (fun b => if (u ≠ 0 ∧ u ≠ W) ∧ b = W then 1 else 0))
           + (sumLt (2^(m+1)) (fun u =>
                 sumLt (2^(m+1)) (fun b => if (u ≠ 0 ∧ u ≠ W) ∧ b = u then 1 else 0))
              + sumLt (2^(m+1)) (fun u =>
                  sumLt (2^(m+1)) (fun b =>
                    if u ≠ 0 ∧ u ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ u ≠ b ∧ b ≠ u ^^^ W
                       ∧ Qgen' W u b (m+1) = 1 then 1 else 0)))) := by
    rw [sumLt_congr (2^(m+1)) _ (fun u =>
          sumLt (2^(m+1)) (fun b => if u = 0 ∧ b ≠ 0 ∧ b ≠ W then 1 else 0)
          + (sumLt (2^(m+1)) (fun b => if (u ≠ 0 ∧ u ≠ W) ∧ b = W then 1 else 0)
             + (sumLt (2^(m+1)) (fun b => if (u ≠ 0 ∧ u ≠ W) ∧ b = u then 1 else 0)
                + sumLt (2^(m+1)) (fun b =>
                    if u ≠ 0 ∧ u ≠ W ∧ b ≠ 0 ∧ b ≠ W ∧ u ≠ b ∧ b ≠ u ^^^ W
                       ∧ Qgen' W u b (m+1) = 1 then 1 else 0))))
        (fun u hU => by
          rw [sumLt_congr (2^(m+1)) _ _ (fun b hB => hi_ul_split m W u b hW hW0 hU hB),
              sumLt_pair, sumLt_pair, sumLt_pair]),
        sumLt_pair, sumLt_pair, sumLt_pair]
  rw [hsplit,
      sumLt_congr (2^(m+1)) _ _ (fun u _ => row_off (2^(m+1)) W (u = 0)),
      sumLt_congr (2^(m+1)) _ _ (fun u _ => row_at (2^(m+1)) W hW (u ≠ 0 ∧ u ≠ W)),
      sumLt_congr (2^(m+1)) _ _ (fun u hU => row_at (2^(m+1)) u hU (u ≠ 0 ∧ u ≠ W)),
      sumLt_single (2^(m+1)) 0
        (sumLt (2^(m+1)) (fun b => if b ≠ 0 ∧ b ≠ W then 1 else 0)) hp]
  unfold OffCntP
  have hc2 := count_off2 W (m+1) hW hW0
  omega

/-- Generic row: a guard times the nonzero count. -/
private theorem row_nz (n : Nat) (p : Prop) [Decidable p] :
    sumLt n (fun v => if p ∧ v ≠ 0 then 1 else 0)
      = if p then sumLt n (fun v => if v ≠ 0 then 1 else 0) else 0 := by
  by_cases hg : p
  · rw [if_pos hg]
    apply sumLt_congr
    intro v _
    by_cases hv : v ≠ 0
    · rw [if_pos ⟨hg, hv⟩, if_pos hv]
    · rw [if_neg (fun hh => hv hh.2), if_neg hv]
  · rw [if_neg hg, sumLt_congr n _ (fun _ => 0) (fun v _ => if_neg (fun hh => hg hh.1))]
    exact sumLt_zero _

/-- The transposed core: `lu` and `uu` evaluate `Q'` with its arguments SWAPPED, so their
    remainders are `OffCntP` only after `sumLt_swap`. -/
private theorem swapped_core (m W : Nat) :
    sumLt (2^(m+1)) (fun a => sumLt (2^(m+1)) (fun v =>
      if a ≠ 0 ∧ a ≠ W ∧ v ≠ 0 ∧ v ≠ W ∧ a ≠ v ∧ a ≠ v ^^^ W ∧ Qgen' W v a (m+1) = 1
      then 1 else 0)) = OffCntP W (m+1) := by
  rw [sumLt_swap]
  unfold OffCntP
  apply sumLt_congr
  intro v _
  apply sumLt_congr
  intro a _
  by_cases h : v ≠ 0 ∧ v ≠ W ∧ a ≠ 0 ∧ a ≠ W ∧ v ≠ a ∧ a ≠ v ^^^ W ∧ Qgen' W v a (m+1) = 1
  · rw [if_pos ⟨h.2.2.1, h.2.2.2.1, h.1, h.2.1, fun hh => h.2.2.2.2.1 hh.symm,
          h.2.2.2.2.2.1, h.2.2.2.2.2.2⟩, if_pos h]
  · rw [if_neg (fun hc => h ⟨hc.2.2.1, hc.2.2.2.1, hc.1, hc.2.1,
          fun hh => hc.2.2.2.2.1 hh.symm, hc.2.2.2.2.2.1, hc.2.2.2.2.2.2⟩), if_neg h]

/-- Pointwise split of the HIGH `lu` quadrant. The `v = W` column is the whole `label_right`
    column of the level-`(m+2)` label, so it is `-1` throughout and counts `2^(m+1) - 1` -- one
    MORE than every other slice. That single extra point is what makes `lu` and `uu` carry
    `4e - 7` rather than `4(e-2)`. -/
theorem hi_lu_split (m W a v : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (ha : a < 2^(m+1)) (hv : v < 2^(m+1)) :
    nInd (W + 2^(m+1)) (m+2) a (2^(m+1) + v)
      = (if (a ≠ 0 ∧ a ≠ W) ∧ v = 0 then 1 else 0)
        + ((if (a ≠ 0) ∧ v = W then 1 else 0)
           + ((if a = W ∧ v ≠ 0 ∧ v ≠ W then 1 else 0)
              + ((if (a ≠ 0 ∧ a ≠ W) ∧ v = a then 1 else 0)
                 + (if a ≠ 0 ∧ a ≠ W ∧ v ≠ 0 ∧ v ≠ W ∧ a ≠ v ∧ a ≠ v ^^^ W
                       ∧ Qgen' W v a (m+1) = 1 then 1 else 0)))) := by
  have hp : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have hcv : (2:Nat)^(m+1) + v = v + 2^(m+1) := by omega
  unfold nInd
  rw [hcv]
  have hb0 : v + 2^(m+1) ≠ 0 := by omega
  have hab : a ≠ v + 2^(m+1) := by omega
  have hpow : (2:Nat)^(m+2) = 2^(m+1) + 2^(m+1) := by rw [Nat.pow_succ]; omega
  by_cases hv0 : v = 0
  · have e2 : (if (a ≠ 0) ∧ v = W then (1:Nat) else 0) = 0 :=
      if_neg (fun hc => hW0 (hv0 ▸ hc.2).symm)
    have e3 : (if a = W ∧ v ≠ 0 ∧ v ≠ W then (1:Nat) else 0) = 0 :=
      if_neg (fun hc => hc.2.1 hv0)
    have e5 : (if a ≠ 0 ∧ a ≠ W ∧ v ≠ 0 ∧ v ≠ W ∧ a ≠ v ∧ a ≠ v ^^^ W
                  ∧ Qgen' W v a (m+1) = 1 then (1:Nat) else 0) = 0 :=
      if_neg (fun hc => hc.2.2.1 hv0)
    rw [e2, e3, e5]
    by_cases ha0 : a = 0
    · rw [if_neg (fun hc => hc.1 ha0), if_neg (fun hc => hc.1.1 ha0),
          if_neg (fun hc => hc.1.1 ha0)]
    · have e4 : (if (a ≠ 0 ∧ a ≠ W) ∧ v = a then (1:Nat) else 0) = 0 :=
        if_neg (fun hc => hc.1.1 (hv0 ▸ hc.2).symm)
      rw [e4]
      subst hv0
      by_cases haW : a = W
      · have hq : Qgen' (W + 2^(m+1)) a (0 + 2^(m+1)) (m+2) = 1 := by
          rw [haW]; exact Qgen'_hi_lu_v0W m W hW hW0
        rw [if_neg (fun hc => by rw [hq] at hc; exact absurd hc.2.2.2 (by decide)),
            if_neg (fun hc => hc.1.2 haW)]
      · rw [if_pos ⟨ha0, hb0, hab, Qgen'_hi_lu_v0 m W a hW hW0 ha ha0 haW⟩,
            if_pos ⟨⟨ha0, haW⟩, rfl⟩]
  · have e1 : (if (a ≠ 0 ∧ a ≠ W) ∧ v = 0 then (1:Nat) else 0) = 0 :=
      if_neg (fun hc => hv0 hc.2)
    rw [e1, Nat.zero_add]
    by_cases hvW : v = W
    · have e3 : (if a = W ∧ v ≠ 0 ∧ v ≠ W then (1:Nat) else 0) = 0 :=
        if_neg (fun hc => hc.2.2 hvW)
      have e5 : (if a ≠ 0 ∧ a ≠ W ∧ v ≠ 0 ∧ v ≠ W ∧ a ≠ v ∧ a ≠ v ^^^ W
                    ∧ Qgen' W v a (m+1) = 1 then (1:Nat) else 0) = 0 :=
        if_neg (fun hc => hc.2.2.2.1 hvW)
      rw [e3, e5]
      have hlt : W + 2^(m+1) < 2^(m+2) := by omega
      have halt : a < 2^(m+2) := by omega
      have hq : Qgen' (W + 2^(m+1)) a (v + 2^(m+1)) (m+2) = -1 := by
        rw [hvW]; exact Qgen'_label_right (m+2) (W + 2^(m+1)) a hlt halt (by omega)
      by_cases ha0 : a = 0
      · rw [if_neg (fun hc => hc.1 ha0), if_neg (fun hc => hc.1 ha0),
            if_neg (fun hc => hc.1.1 ha0)]
      · have e4 : (if (a ≠ 0 ∧ a ≠ W) ∧ v = a then (1:Nat) else 0) = 0 :=
          if_neg (fun hc => hc.1.2 (hvW ▸ hc.2).symm)
        rw [e4, if_pos ⟨ha0, hb0, hab, hq⟩, if_pos ⟨ha0, hvW⟩]
    · have e2 : (if (a ≠ 0) ∧ v = W then (1:Nat) else 0) = 0 := if_neg (fun hc => hvW hc.2)
      rw [e2, Nat.zero_add]
      have hvX : v ^^^ W ≠ 0 := fun h => hvW (xor_zero_eq v W h)
      by_cases ha0 : a = 0
      · rw [if_neg (fun hc => hc.1 ha0), if_neg (fun hc => ha0 ▸ hW0 <| hc.1.symm),
            if_neg (fun hc => hc.1.1 ha0), if_neg (fun hc => hc.1 ha0)]
      · by_cases haW : a = W
        · have hq : Qgen' (W + 2^(m+1)) a (v + 2^(m+1)) (m+2) = -1 := by
            rw [haW]; exact Qgen'_hi_lu_aW m W v hW hW0 hv hv0
          rw [if_pos ⟨ha0, hb0, hab, hq⟩, if_pos ⟨haW, hv0, hvW⟩,
              if_neg (fun hc => hc.1.2 haW), if_neg (fun hc => hc.2.1 haW)]
        · have e3 : (if a = W ∧ v ≠ 0 ∧ v ≠ W then (1:Nat) else 0) = 0 :=
            if_neg (fun hc => haW hc.1)
          rw [e3, Nat.zero_add]
          have haX : a ^^^ W ≠ 0 := fun h => haW (xor_zero_eq a W h)
          by_cases hav : v = a
          · have hq : Qgen' (W + 2^(m+1)) a (v + 2^(m+1)) (m+2) = -1 := by
              rw [hav]; exact Qgen'_hi_lu_av m W a hW hW0 ha ha0 haW
            rw [if_pos ⟨ha0, hb0, hab, hq⟩, if_pos ⟨⟨ha0, haW⟩, hav⟩,
                if_neg (fun hc => hc.2.2.2.2.1 hav.symm)]
          · have e4 : (if (a ≠ 0 ∧ a ≠ W) ∧ v = a then (1:Nat) else 0) = 0 :=
              if_neg (fun hc => hav hc.2)
            rw [e4, Nat.zero_add]
            have hav' : a ≠ v := fun h => hav h.symm
            have hrow : Qgen' (W + 2^(m+1)) a (v + 2^(m+1)) (m+2) = - Qgen W v a (m+1) :=
              Q'red_hi_lu m W a v hW ha hv hv0 haX hvX hav'
            by_cases hcos : a = v ^^^ W
            · have hz : v ^^^ a ^^^ W = 0 := by rw [hcos, xorCancelL, Nat.xor_self]
              have hqm : Qgen W v a (m+1) = -1 :=
                Qgen_degen (m+1) W v a hW hv ha hW0
                  (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr hz)))))
              have hq : Qgen' (W + 2^(m+1)) a (v + 2^(m+1)) (m+2) = 1 := by
                rw [hrow, hqm]; decide
              rw [if_neg (fun hc => by rw [hq] at hc; exact absurd hc.2.2.2 (by decide)),
                  if_neg (fun hc => hc.2.2.2.2.2.1 hcos)]
            · have h3ne : (v ^^^ W) ^^^ (a ^^^ W) ≠ 0 := by
                intro h
                apply hav
                have h2 : v ^^^ a = 0 := by
                  rw [Nat.xor_assoc, ← Nat.xor_assoc W a W, Nat.xor_comm W a,
                      Nat.xor_assoc a W W, Nat.xor_self, Nat.xor_zero] at h
                  exact h
                exact xor_zero_eq v a h2
              have h5ne : v ^^^ (a ^^^ W) ≠ 0 := by
                intro h
                apply hcos
                have hv2 : v = a ^^^ W := xor_zero_eq v (a ^^^ W) h
                rw [hv2, xor_cancel a W]
              have hbridge : Qgen W v a (m+1) = Qgen' W v a (m+1) :=
                Qgen_eq_Qgen' W v a (m+1) hv ha hW hvX haX h3ne hv0 h5ne
              by_cases hq1 : Qgen' W v a (m+1) = 1
              · have hq : Qgen' (W + 2^(m+1)) a (v + 2^(m+1)) (m+2) = -1 := by
                  rw [hrow, hbridge, hq1]
                rw [if_pos ⟨ha0, hb0, hab, hq⟩,
                    if_pos ⟨ha0, haW, hv0, hvW, hav', hcos, hq1⟩]
              · have hq : Qgen' (W + 2^(m+1)) a (v + 2^(m+1)) (m+2) ≠ -1 := by
                  rw [hrow, hbridge]
                  rcases Qgen'_pm W v a (m+1) with h | h
                  · exact absurd h hq1
                  · rw [h]; decide
                rw [if_neg (fun hc => hq hc.2.2.2),
                    if_neg (fun hc => hq1 hc.2.2.2.2.2.2)]

/-- **The `lu` quadrant of the HIGH box.** Four slices, one of which (the `v = W` label column)
    has `2^(m+1) - 1` points instead of `2^(m+1) - 2`. -/
theorem Ncnt_lu_hi (m W : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0) :
    sumLt (2^(m+1)) (fun a =>
        sumLt (2^(m+1)) (fun v => nInd (W + 2^(m+1)) (m+2) a (2^(m+1) + v))) + 7
      = OffCntP W (m+1) + 4 * 2^(m+1) := by
  have hp : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have hsplit : sumLt (2^(m+1)) (fun a =>
        sumLt (2^(m+1)) (fun v => nInd (W + 2^(m+1)) (m+2) a (2^(m+1) + v)))
      = sumLt (2^(m+1)) (fun a =>
            sumLt (2^(m+1)) (fun v => if (a ≠ 0 ∧ a ≠ W) ∧ v = 0 then 1 else 0))
        + (sumLt (2^(m+1)) (fun a =>
              sumLt (2^(m+1)) (fun v => if (a ≠ 0) ∧ v = W then 1 else 0))
           + (sumLt (2^(m+1)) (fun a =>
                 sumLt (2^(m+1)) (fun v => if a = W ∧ v ≠ 0 ∧ v ≠ W then 1 else 0))
              + (sumLt (2^(m+1)) (fun a =>
                    sumLt (2^(m+1)) (fun v => if (a ≠ 0 ∧ a ≠ W) ∧ v = a then 1 else 0))
                 + sumLt (2^(m+1)) (fun a =>
                     sumLt (2^(m+1)) (fun v =>
                       if a ≠ 0 ∧ a ≠ W ∧ v ≠ 0 ∧ v ≠ W ∧ a ≠ v ∧ a ≠ v ^^^ W
                          ∧ Qgen' W v a (m+1) = 1 then 1 else 0))))) := by
    rw [sumLt_congr (2^(m+1)) _ (fun a =>
          sumLt (2^(m+1)) (fun v => if (a ≠ 0 ∧ a ≠ W) ∧ v = 0 then 1 else 0)
          + (sumLt (2^(m+1)) (fun v => if (a ≠ 0) ∧ v = W then 1 else 0)
             + (sumLt (2^(m+1)) (fun v => if a = W ∧ v ≠ 0 ∧ v ≠ W then 1 else 0)
                + (sumLt (2^(m+1)) (fun v => if (a ≠ 0 ∧ a ≠ W) ∧ v = a then 1 else 0)
                   + sumLt (2^(m+1)) (fun v =>
                       if a ≠ 0 ∧ a ≠ W ∧ v ≠ 0 ∧ v ≠ W ∧ a ≠ v ∧ a ≠ v ^^^ W
                          ∧ Qgen' W v a (m+1) = 1 then 1 else 0)))))
        (fun a hA => by
          rw [sumLt_congr (2^(m+1)) _ _ (fun v hV => hi_lu_split m W a v hW hW0 hA hV),
              sumLt_pair, sumLt_pair, sumLt_pair, sumLt_pair]),
        sumLt_pair, sumLt_pair, sumLt_pair, sumLt_pair]
  rw [hsplit,
      sumLt_congr (2^(m+1)) _ _ (fun a _ => row_at (2^(m+1)) 0 hp (a ≠ 0 ∧ a ≠ W)),
      sumLt_congr (2^(m+1)) _ _ (fun a _ => row_at (2^(m+1)) W hW (a ≠ 0)),
      sumLt_congr (2^(m+1)) _ _ (fun a _ => row_off (2^(m+1)) W (a = W)),
      sumLt_congr (2^(m+1)) _ _ (fun a hA => row_at (2^(m+1)) a hA (a ≠ 0 ∧ a ≠ W)),
      swapped_core m W,
      sumLt_single (2^(m+1)) W
        (sumLt (2^(m+1)) (fun b => if b ≠ 0 ∧ b ≠ W then 1 else 0)) hW]
  have hc2 := count_off2 W (m+1) hW hW0
  have hc1 := count_off1 (m+1)
  omega

/-- Pointwise split of the HIGH `uu` quadrant. Like `lu` it carries one asymmetric slice --
    here the `u = 0` row, which has `2^(m+1) - 1` points because `u = 0` already forces `u ≠ v`
    for every `v ≠ 0`. -/
theorem hi_uu_split (m W u v : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0)
    (hu : u < 2^(m+1)) (hv : v < 2^(m+1)) :
    nInd (W + 2^(m+1)) (m+2) (2^(m+1) + u) (2^(m+1) + v)
      = (if u = 0 ∧ v ≠ 0 then 1 else 0)
        + ((if (u ≠ 0 ∧ u ≠ W) ∧ v = 0 then 1 else 0)
           + ((if (u ≠ 0 ∧ u ≠ W) ∧ v = W then 1 else 0)
              + ((if (u ≠ 0 ∧ u ≠ W) ∧ v = u ^^^ W then 1 else 0)
                 + (if u ≠ 0 ∧ u ≠ W ∧ v ≠ 0 ∧ v ≠ W ∧ u ≠ v ∧ u ≠ v ^^^ W
                       ∧ Qgen' W v u (m+1) = 1 then 1 else 0)))) := by
  have hp : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have hcu : (2:Nat)^(m+1) + u = u + 2^(m+1) := by omega
  have hcv : (2:Nat)^(m+1) + v = v + 2^(m+1) := by omega
  have hpow : (2:Nat)^(m+2) = 2^(m+1) + 2^(m+1) := by rw [Nat.pow_succ]; omega
  unfold nInd
  rw [hcu, hcv]
  have ha0 : u + 2^(m+1) ≠ 0 := by omega
  have hb0 : v + 2^(m+1) ≠ 0 := by omega
  by_cases huv : u = v
  · have hab : ¬ (u + 2^(m+1) ≠ v + 2^(m+1)) := by omega
    rw [if_neg (fun hc => hab hc.2.2.1),
        if_neg (fun hc => hc.2 (by rw [← huv]; exact hc.1)),
        if_neg (fun hc => hc.1.1 (by rw [huv]; exact hc.2)),
        if_neg (fun hc => hc.1.2 (by rw [huv]; exact hc.2)),
        if_neg (fun hc => hW0 (by
          have h2 : u ^^^ u = u ^^^ (u ^^^ W) := by rw [huv] at hc ⊢; rw [← hc.2]
          rw [Nat.xor_self, xorCancelL] at h2
          exact h2.symm)),
        if_neg (fun hc => hc.2.2.2.2.1 huv)]
  · have hab : u + 2^(m+1) ≠ v + 2^(m+1) := by omega
    by_cases hu0 : u = 0
    · have hv0 : v ≠ 0 := by rw [← hu0]; exact fun h => huv h.symm
      have hq : Qgen' (W + 2^(m+1)) (u + 2^(m+1)) (v + 2^(m+1)) (m+2) = -1 := by
        rw [hu0]; exact Qgen'_hi_uu_u0 m W v hW hW0 hv hv0
      rw [if_pos ⟨ha0, hb0, hab, hq⟩, if_pos ⟨hu0, hv0⟩,
          if_neg (fun hc => hc.1.1 hu0), if_neg (fun hc => hc.1.1 hu0),
          if_neg (fun hc => hc.1.1 hu0), if_neg (fun hc => hc.1 hu0)]
    · have h1 : (if u = 0 ∧ v ≠ 0 then (1:Nat) else 0) = 0 := if_neg (fun hc => hu0 hc.1)
      rw [h1, Nat.zero_add]
      by_cases huW : u = W
      · have hvW : v ≠ W := by rw [← huW]; exact fun h => huv h.symm
        have hlt : W + 2^(m+1) < 2^(m+2) := by omega
        have hblt : v + 2^(m+1) < 2^(m+2) := by omega
        have hq : Qgen' (W + 2^(m+1)) (u + 2^(m+1)) (v + 2^(m+1)) (m+2) = 1 := by
          rw [huW]
          exact Qgen'_label_left (m+2) (W + 2^(m+1)) (v + 2^(m+1)) hlt hblt (by omega) hb0
            (by omega)
        rw [if_neg (fun hc => by rw [hq] at hc; exact absurd hc.2.2.2 (by decide)),
            if_neg (fun hc => hc.1.2 huW), if_neg (fun hc => hc.1.2 huW),
            if_neg (fun hc => hc.1.2 huW), if_neg (fun hc => hc.2.1 huW)]
      · have huX : u ^^^ W ≠ 0 := fun h => huW (xor_zero_eq u W h)
        by_cases hv0 : v = 0
        · have hq : Qgen' (W + 2^(m+1)) (u + 2^(m+1)) (v + 2^(m+1)) (m+2) = -1 := by
            rw [hv0]; exact Qgen'_hi_uu_v0 m W u hW hW0 hu hu0 huW
          rw [if_pos ⟨ha0, hb0, hab, hq⟩, if_pos ⟨⟨hu0, huW⟩, hv0⟩,
              if_neg (fun hc => hW0 (by rw [← hc.2]; exact hv0)),
              if_neg (fun hc => huX (by rw [← hc.2]; exact hv0)),
              if_neg (fun hc => hc.2.2.1 hv0)]
        · have h2 : (if (u ≠ 0 ∧ u ≠ W) ∧ v = 0 then (1:Nat) else 0) = 0 :=
            if_neg (fun hc => hv0 hc.2)
          rw [h2, Nat.zero_add]
          by_cases hvW : v = W
          · have hlt : W + 2^(m+1) < 2^(m+2) := by omega
            have halt : u + 2^(m+1) < 2^(m+2) := by omega
            have hq : Qgen' (W + 2^(m+1)) (u + 2^(m+1)) (v + 2^(m+1)) (m+2) = -1 := by
              rw [hvW]
              exact Qgen'_label_right (m+2) (W + 2^(m+1)) (u + 2^(m+1)) hlt halt (by omega)
            rw [if_pos ⟨ha0, hb0, hab, hq⟩, if_pos ⟨⟨hu0, huW⟩, hvW⟩,
                if_neg (fun hc => hu0 (by
                  have h3 : (u ^^^ W) ^^^ W = W ^^^ W := by rw [← hc.2, hvW]
                  rw [xor_cancel u W, Nat.xor_self] at h3
                  exact h3)),
                if_neg (fun hc => hc.2.2.2.1 hvW)]
          · have h3 : (if (u ≠ 0 ∧ u ≠ W) ∧ v = W then (1:Nat) else 0) = 0 :=
              if_neg (fun hc => hvW hc.2)
            rw [h3, Nat.zero_add]
            have hvX : v ^^^ W ≠ 0 := fun h => hvW (xor_zero_eq v W h)
            by_cases hcos : v = u ^^^ W
            · have hcos' : u = v ^^^ W := by rw [hcos, xor_cancel u W]
              have hq : Qgen' (W + 2^(m+1)) (u + 2^(m+1)) (v + 2^(m+1)) (m+2) = -1 := by
                rw [hcos]; exact Qgen'_hi_uu_cos m W u hW hW0 hu hu0 huW
              rw [if_pos ⟨ha0, hb0, hab, hq⟩, if_pos ⟨⟨hu0, huW⟩, hcos⟩,
                  if_neg (fun hc => hc.2.2.2.2.2.1 hcos')]
            · have h4 : (if (u ≠ 0 ∧ u ≠ W) ∧ v = u ^^^ W then (1:Nat) else 0) = 0 :=
                if_neg (fun hc => hcos hc.2)
              rw [h4, Nat.zero_add]
              have hcos2 : u ≠ v ^^^ W := by
                intro h
                apply hcos
                rw [h, xor_cancel v W]
              have hxor3 : u ^^^ v ^^^ W ≠ 0 := by
                intro h
                apply hcos
                have h2 : u ^^^ v = W := xor_zero_eq (u ^^^ v) W h
                rw [← h2, xorCancelL]
              have hrow : Qgen' (W + 2^(m+1)) (u + 2^(m+1)) (v + 2^(m+1)) (m+2)
                  = - Qgen' W v u (m+1) :=
                Q'red_hi_uu m W u v hW hu hv hu0 hv0 huX hvX huv hxor3
              by_cases hq1 : Qgen' W v u (m+1) = 1
              · have hq : Qgen' (W + 2^(m+1)) (u + 2^(m+1)) (v + 2^(m+1)) (m+2) = -1 := by
                  rw [hrow, hq1]
                rw [if_pos ⟨ha0, hb0, hab, hq⟩,
                    if_pos ⟨hu0, huW, hv0, hvW, huv, hcos2, hq1⟩]
              · have hq : Qgen' (W + 2^(m+1)) (u + 2^(m+1)) (v + 2^(m+1)) (m+2) ≠ -1 := by
                  rw [hrow]
                  rcases Qgen'_pm W v u (m+1) with h | h
                  · exact absurd h hq1
                  · rw [h]; decide
                rw [if_neg (fun hc => hq hc.2.2.2),
                    if_neg (fun hc => hq1 hc.2.2.2.2.2.2)]

/-- **The `uu` quadrant of the HIGH box.** -/
theorem Ncnt_uu_hi (m W : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0) :
    sumLt (2^(m+1)) (fun u =>
        sumLt (2^(m+1)) (fun v =>
          nInd (W + 2^(m+1)) (m+2) (2^(m+1) + u) (2^(m+1) + v))) + 7
      = OffCntP W (m+1) + 4 * 2^(m+1) := by
  have hp : (0:Nat) < 2^(m+1) := Nat.two_pow_pos (m+1)
  have hsplit : sumLt (2^(m+1)) (fun u =>
        sumLt (2^(m+1)) (fun v => nInd (W + 2^(m+1)) (m+2) (2^(m+1) + u) (2^(m+1) + v)))
      = sumLt (2^(m+1)) (fun u =>
            sumLt (2^(m+1)) (fun v => if u = 0 ∧ v ≠ 0 then 1 else 0))
        + (sumLt (2^(m+1)) (fun u =>
              sumLt (2^(m+1)) (fun v => if (u ≠ 0 ∧ u ≠ W) ∧ v = 0 then 1 else 0))
           + (sumLt (2^(m+1)) (fun u =>
                 sumLt (2^(m+1)) (fun v => if (u ≠ 0 ∧ u ≠ W) ∧ v = W then 1 else 0))
              + (sumLt (2^(m+1)) (fun u =>
                    sumLt (2^(m+1)) (fun v =>
                      if (u ≠ 0 ∧ u ≠ W) ∧ v = u ^^^ W then 1 else 0))
                 + sumLt (2^(m+1)) (fun u =>
                     sumLt (2^(m+1)) (fun v =>
                       if u ≠ 0 ∧ u ≠ W ∧ v ≠ 0 ∧ v ≠ W ∧ u ≠ v ∧ u ≠ v ^^^ W
                          ∧ Qgen' W v u (m+1) = 1 then 1 else 0))))) := by
    rw [sumLt_congr (2^(m+1)) _ (fun u =>
          sumLt (2^(m+1)) (fun v => if u = 0 ∧ v ≠ 0 then 1 else 0)
          + (sumLt (2^(m+1)) (fun v => if (u ≠ 0 ∧ u ≠ W) ∧ v = 0 then 1 else 0)
             + (sumLt (2^(m+1)) (fun v => if (u ≠ 0 ∧ u ≠ W) ∧ v = W then 1 else 0)
                + (sumLt (2^(m+1)) (fun v => if (u ≠ 0 ∧ u ≠ W) ∧ v = u ^^^ W then 1 else 0)
                   + sumLt (2^(m+1)) (fun v =>
                       if u ≠ 0 ∧ u ≠ W ∧ v ≠ 0 ∧ v ≠ W ∧ u ≠ v ∧ u ≠ v ^^^ W
                          ∧ Qgen' W v u (m+1) = 1 then 1 else 0)))))
        (fun u hU => by
          rw [sumLt_congr (2^(m+1)) _ _ (fun v hV => hi_uu_split m W u v hW hW0 hU hV),
              sumLt_pair, sumLt_pair, sumLt_pair, sumLt_pair]),
        sumLt_pair, sumLt_pair, sumLt_pair, sumLt_pair]
  rw [hsplit,
      sumLt_congr (2^(m+1)) _ _ (fun u _ => row_nz (2^(m+1)) (u = 0)),
      sumLt_congr (2^(m+1)) _ _ (fun u _ => row_at (2^(m+1)) 0 hp (u ≠ 0 ∧ u ≠ W)),
      sumLt_congr (2^(m+1)) _ _ (fun u _ => row_at (2^(m+1)) W hW (u ≠ 0 ∧ u ≠ W)),
      sumLt_congr (2^(m+1)) _ _ (fun u hU =>
        row_at (2^(m+1)) (u ^^^ W) (xorlt hU hW) (u ≠ 0 ∧ u ≠ W)),
      swapped_core m W,
      sumLt_single (2^(m+1)) 0
        (sumLt (2^(m+1)) (fun v => if v ≠ 0 then 1 else 0)) hp]
  have hc2 := count_off2 W (m+1) hW hW0
  have hc1 := count_off1 (m+1)
  omega

/-! ### Tier 31: THE HIGH RECURSION

The four quadrants are now theorems. What remains is bookkeeping, and it is stated with NO Nat
subtraction anywhere so that `omega` stays linear with `2^(m+1) * 2^(m+1)` as a single atom. -/

/-- The positive core, priced against the level-`(m+1)` count. -/
theorem OffCntP_eq (m W : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0) :
    OffCntP W (m+1) + 5 * 2^(m+1) + Ncnt W (m+1) = 2^(m+1) * 2^(m+1) + 6 := by
  have h1 := OffCnt_add_OffCntP W (m+1) hW hW0
  have h2 := Ncnt_eq_OffCnt W (m+1) hW hW0
  omega

/-- **THE HIGH RECURSION, forall n.** With `e = 2^(m+1)`, this is
    `N(m+2, W+e) = 4e^2 - 6e - 2 - 4*N(m+1, W)`, i.e. the paper's
    `4P' - 4N' + 6e - 10` with `P' = (e-1)(e-2)` -- stated additively to stay in `Nat`. -/
theorem Ncnt_hi (m W : Nat) (hW : W < 2^(m+1)) (hW0 : W ≠ 0) :
    Ncnt (W + 2^(m+1)) (m+2) + 6 * 2^(m+1) + 2 + 4 * Ncnt W (m+1)
      = 4 * (2^(m+1) * 2^(m+1)) := by
  have hll := Ncnt_ll_hi m W hW hW0
  have hul := Ncnt_ul_hi m W hW hW0
  have hlu := Ncnt_lu_hi m W hW hW0
  have huu := Ncnt_uu_hi m W hW hW0
  have hP := OffCntP_eq m W hW hW0
  rw [Ncnt_quad (W + 2^(m+1)) m]
  omega

/-- **The `m = 0` bottom**, which the old `1 ≤ m` hypothesis excluded. This is the smallest box
    (`e = 2`, so the only label is `W = 1`), and it is exactly the step an odd non-power-of-two
    label's descent lands on: `W = 3` uses LOW down to level 2 and then needs HIGH here. Stated
    as a closed instance so that the removal of the hypothesis is CHECKED, not just asserted. -/
theorem Ncnt_hi_bottom : Ncnt 3 2 + 6 * 2 + 2 + 4 * Ncnt 1 1 = 4 * (2 * 2) :=
  Ncnt_hi 0 1 (by decide) (by decide)

/-! ## Tier 32: unrolling the two recursions

`Ncnt_low` and `Ncnt_hi` determine `Ncnt` completely once the descent has a floor. This tier
supplies the floor and then removes `Qgen'` from the picture entirely. -/

/-- **The positive core is EMPTY at a power-of-two label.** `OffCntP` already excludes `a = W`
    and `b = a ^^^ W`, and by `Qgen'_pow2_eq` those two lines are *exactly* where `Q' = +1` when
    `W = 2^k`. So nothing is left to count -- the power-of-two labels are the SATURATED case, and
    every other label is that value minus a deficit. -/
theorem OffCntP_pow2 (m k : Nat) (hk : k < m) : OffCntP (2^k) m = 0 := by
  have hinner : ∀ a, a < 2^m →
      sumLt (2^m) (fun b =>
        if a ≠ 0 ∧ a ≠ 2^k ∧ b ≠ 0 ∧ b ≠ 2^k ∧ a ≠ b ∧ b ≠ a ^^^ 2^k
           ∧ Qgen' (2^k) a b m = 1 then 1 else 0) = 0 := by
    intro a hA
    have h2 : ∀ b, b < 2^m →
        (if a ≠ 0 ∧ a ≠ 2^k ∧ b ≠ 0 ∧ b ≠ 2^k ∧ a ≠ b ∧ b ≠ a ^^^ 2^k
            ∧ Qgen' (2^k) a b m = 1 then (1:Nat) else 0) = 0 := by
      intro b hB
      refine if_neg ?_
      rintro ⟨ha0, haW, hb0, hbW, hab, hcos, hq⟩
      rw [Qgen'_pow2_eq m k a b hk hA hB ha0 hb0 hab,
          if_neg (fun hc => hc.elim haW hcos)] at hq
      exact absurd hq (by decide)
    rw [sumLt_congr (2^m) _ (fun _ => 0) h2]
    exact sumLt_zero _
  unfold OffCntP
  rw [sumLt_congr (2^m) _ (fun _ => 0) hinner]
  exact sumLt_zero _

/-- **The floor of the descent.** `Ncnt (2^k) m = (2^m - 2)(2^m - 3)`, independent of `k`.
    `Qgen'_pow2_eq` was pointwise only; this is the COUNT, and it is what every chain bottoms
    out on. Stated additively to keep `omega` linear with `2^m * 2^m` as one atom. -/
theorem Ncnt_pow2 (m k : Nat) (hk : k < m) :
    Ncnt (2^k) m + 5 * 2^m = 2^m * 2^m + 6 := by
  have hWlt : (2:Nat)^k < 2^m := Nat.pow_lt_pow_right (by omega) hk
  have hW0 : (2:Nat)^k ≠ 0 := by have := Nat.two_pow_pos k; omega
  have h1 := OffCnt_add_OffCntP (2^k) m hWlt hW0
  have h2 := Ncnt_eq_OffCnt (2^k) m hWlt hW0
  have h3 := OffCntP_pow2 m k hk
  omega

/-- The two smallest boxes are empty: at level 0 the only index is `0`, at level 1 the only two
    admissible indices coincide (`base_box_empty`). -/
theorem Ncnt_zero_level (W : Nat) : Ncnt W 0 = 0 := by
  unfold Ncnt
  rw [show (2:Nat)^0 = 1 from rfl,
      sumLt_congr 1 _ (fun _ => 0) (fun a ha => by
        rw [sumLt_congr 1 _ (fun _ => 0)
              (fun b _ => if_neg (fun hc => hc.1 (by omega)))]
        exact sumLt_zero _)]
  exact sumLt_zero _

theorem Ncnt_one_level (W : Nat) : Ncnt W 1 = 0 := by
  unfold Ncnt
  rw [sumLt_congr (2^1) _ (fun _ => 0) (fun a ha => by
        rw [sumLt_congr (2^1) _ (fun _ => 0)
              (fun b hb => if_neg (fun hc => hc.2.2.1 (base_box_empty a b ha hb hc.1 hc.2.1)))]
        exact sumLt_zero _)]
  exact sumLt_zero _

/-- **The unrolled count.** A self-contained evaluator: structural recursion on the LEVEL, with
    no reference to `Qgen'`, `cdSigma`, or any sum. The three branches are exactly `Ncnt_low`,
    `Ncnt_pow2` and `Ncnt_hi`.

    There is deliberately NO power-of-two test. A label `2^k` with `k < n+1` is already below the
    seam, so the LOW branch handles it, and the two agree because
    `(2e-2)(2e-3) = 4(e-2)(e-3) + 10e - 18` identically. Only `W = 2^(n+1)` needs its own branch,
    and that is decidable `Nat` equality -- which is what lets this file avoid characterising
    powers of two by bit arithmetic.

    `Int` codomain (the HIGH branch subtracts), but every product and power is formed on the
    `Nat` side and cast, so `omega` only ever sees atoms it shares with `Ncnt_hi`. -/
def Nclosed : Nat → Nat → Int
  | 0, _ => 0
  | 1, _ => 0
  | (n+2), W =>
      if W < 2^(n+1) then
        4 * Nclosed (n+1) W + 10 * ((2^(n+1) : Nat) : Int) - 18
      else if W = 2^(n+1) then
        ((2^(n+2) * 2^(n+2) : Nat) : Int) + 6 - 5 * ((2^(n+2) : Nat) : Int)
      else
        4 * ((2^(n+1) * 2^(n+1) : Nat) : Int) - 6 * ((2^(n+1) : Nat) : Int) - 2
          - 4 * Nclosed (n+1) (W - 2^(n+1))

theorem Nclosed_zero (W : Nat) : Nclosed 0 W = 0 := rfl
theorem Nclosed_one (W : Nat) : Nclosed 1 W = 0 := rfl
theorem Nclosed_step (n W : Nat) :
    Nclosed (n+2) W =
      if W < 2^(n+1) then
        4 * Nclosed (n+1) W + 10 * ((2^(n+1) : Nat) : Int) - 18
      else if W = 2^(n+1) then
        ((2^(n+2) * 2^(n+2) : Nat) : Int) + 6 - 5 * ((2^(n+2) : Nat) : Int)
      else
        4 * ((2^(n+1) * 2^(n+1) : Nat) : Int) - 6 * ((2^(n+1) : Nat) : Int) - 2
          - 4 * Nclosed (n+1) (W - 2^(n+1)) := rfl

/-- **THE UNROLLING, forall n.** The counting recursion is now solved: `Ncnt` equals a closed
    evaluator that never mentions `Qgen'`. Every step is one of the three theorems
    `Ncnt_low` / `Ncnt_pow2` / `Ncnt_hi`. -/
theorem Ncnt_eq_Nclosed : ∀ (m W : Nat), W < 2^m → W ≠ 0 → (Ncnt W m : Int) = Nclosed m W := by
  intro m
  induction m with
  | zero =>
      intro W hW hW0
      have h : (2:Nat)^0 = 1 := rfl
      omega
  | succ n ih =>
      cases n with
      | zero =>
          intro W hW hW0
          rw [Ncnt_one_level, Nclosed_one]
          rfl
      | succ k =>
          intro W hW hW0
          -- Normalise the level index: the induction leaves `k+1+1`, but every recursion
          -- theorem is stated with `k+2`, and `omega` treats those as DIFFERENT atoms even
          -- though they are definitionally equal.
          show (Ncnt W (k+2) : Int) = Nclosed (k+2) W
          replace hW : W < 2^(k+2) := hW
          have hpow : (2:Nat)^(k+2) = 2^(k+1) + 2^(k+1) := by rw [Nat.pow_succ]; omega
          have hp : (0:Nat) < 2^(k+1) := Nat.two_pow_pos (k+1)
          rw [Nclosed_step]
          by_cases hlt : W < 2^(k+1)
          · rw [if_pos hlt]
            have hrec := Ncnt_low W k hlt hW0
            have hih := ih W hlt hW0
            omega
          · rw [if_neg hlt]
            by_cases heq : W = 2^(k+1)
            · rw [if_pos heq, heq]
              have hb := Ncnt_pow2 (k+2) (k+1) (by omega)
              omega
            · rw [if_neg heq]
              obtain ⟨W', hWe, hW'lt⟩ : ∃ W', W = W' + 2^(k+1) ∧ W' < 2^(k+1) :=
                ⟨W - 2^(k+1), by omega, by omega⟩
              have hW'0 : W' ≠ 0 := by omega
              subst hWe
              rw [Nat.add_sub_cancel]
              have hrec := Ncnt_hi k W' hW'lt hW'0
              have hih := ih W' hW'lt hW'0
              omega

/-! ## Tier 33: the explicit digit sum

`Nclosed` solved the recursion but is still recursive. What the paper calls the CLOSED FORM is a
finite, non-recursive sum over the set bits of `W`. This tier builds it and proves it.

The form proved here is stated on `W`'s OWN bits, not on the normalised label `8*g(W)+1`. That is
the whole trick: the descent stops at the LOWEST SET BIT of `W`, so the term at that bit must be
omitted -- and omitting it is exactly what the normalisation was compensating for. W16's negative
("the raw label FAILS on every seam") is precisely the missing exclusion, not a missing
normalisation. So this route never meets W17's unproven residual. -/

def sumLtI : Nat → (Nat → Int) → Int
  | 0, _ => 0
  | (n+1), f => sumLtI n f + f n

theorem sumLtI_congr (n : Nat) (f g : Nat → Int) (h : ∀ i, i < n → f i = g i) :
    sumLtI n f = sumLtI n g := by
  induction n with
  | zero => rfl
  | succ n ih => rw [sumLtI, sumLtI, ih (fun i hi => h i (by omega)), h n (by omega)]

theorem sumLtI_zero (n : Nat) : sumLtI n (fun _ => 0) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => rw [sumLtI, ih]; decide

theorem sumLtI_mul (n : Nat) (c : Int) (f : Nat → Int) :
    sumLtI n (fun i => c * f i) = c * sumLtI n f := by
  induction n with
  | zero => simp [sumLtI]
  | succ n ih => rw [sumLtI, sumLtI, ih, Int.mul_add]

/-- The magnitude of the level-`i` digit, `(2^i-4)(2^i-8)*4^(m-i)`, in `Nat`.

    Truncated subtraction is EXACTLY right here, which is what keeps every product on the `Nat`
    side: at `i = 2` and `i = 3` the true coefficient is genuinely `0` and truncation gives `0`;
    at `i = 0, 1` truncation gives the wrong value but those indices are always EXCLUDED by
    `dterm`'s guard. So no index is both included and mis-valued. -/
def dcoef (m i : Nat) : Nat := (2^i - 4) * (2^i - 8) * 4^(m - i)

/-- The level-`i` digit. The guard says a HIGH step happens at level `i`: bit `i-1` of `W` is set
    (`2^(i-1) ≤ W % 2^i`) AND something remains below it (`W % 2^(i-1) ≠ 0`), the second
    conjunct being where the descent stops. The sign is `(-1)^popcount` of the bits above,
    which is `psg`. -/
def dterm (m W i : Nat) : Int :=
  if 2^(i-1) ≤ W % 2^i ∧ W % 2^(i-1) ≠ 0 then (dcoef m i : Int) * psg (W >>> i) else 0

/-- **The signed base-4 digit sum `E`.** Finite, non-recursive, determined by `W`'s set bits. -/
def Ddig (m W : Nat) : Int := sumLtI (m+1) (fun i => dterm m W i)

end SounioZDFiberAntisym










