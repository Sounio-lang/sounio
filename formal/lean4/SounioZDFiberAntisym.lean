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

end SounioZDFiberAntisym




