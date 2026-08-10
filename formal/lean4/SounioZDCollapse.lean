/-
  SounioZDCollapse — (c), the parity-collapse law: the REDUCTION formalised.

  What this file proves, kernel-checked, ∀n:

      (*)  ∧  (L2-closed)   ⇒   Φ is an isomorphism of the signed annihilation graph

  i.e. the two identities the numerical rungs of 2026-07-31 isolated are TOGETHER sufficient for
  the collapse law. They enter as explicit hypotheses; the implication is the theorem.

  UPDATE 2026-08-01 -- (*) IS NO LONGER MEASURED. It is proven, for every level, as
    `SounioZDFiberAntisym.star_forall` (formal/lean4/SounioZDFiberAntisym.lean, commit
    256bdbda4), kernel-checked, axioms [propext, Classical.choice, Quot.sound], no sorryAx:

        star_forall : Y < 2^m -> Y != 0 -> Y % 2^j = 0 -> a < 2^m -> b < 2^m ->
                      Qgen Y a b m = Qgen (tau j Y) (tau j a) (tau j b) m

    and its hypothesis `Y % 2^j = 0` (j <= lsb Y) is WEAKER than the `j = lsb Y` this file's
    `hres` needs, so it covers it. The Lean objects are pinned entrywise to the measured ones
    by K21a/b/c of the l1_reduction contract.

    WIRED 2026-08-01. This file now `import`s SounioZDFiberAntisym and DISCHARGES (*). The two
    obstacles were real and both are paid for explicitly, not waved away:

      * this file carries its own copy of `cdSigma` -- identical body, DIFFERENT CONSTANT, so
        nothing is defeq at a symbolic level. `cdSigma_eq` proves the two agree, by the same
        structural induction the definition uses. (`tau` is not recursive; `tau_eq` is rfl.)
      * `hres` here quantifies over UNBOUNDED a, b, where (*) says nothing. Rather than weaken
        another lane's general theorem, the discharged instances are NEW theorems --
        `Phi_preserves_adj_star` / `Phi_reflects_adj_star` -- carrying `p.1, q.1 < 2^m`. That
        bound is LOAD-BEARING, not convenience: `cdSigma` is total, so the out-of-range
        question is meaningful, and out of range (*) IS FALSE -- 19 200 violations at levels
        4 and 5 (C7 of the collapse contract). The general `hres` is therefore not merely
        unproven but WRONG, and nothing could ever have discharged it. In context the bound
        costs nothing: the vertices of a level-m fiber are lo-labels < 2^m by construction.

    So of the two hypotheses below, (*) IS DISCHARGED and L2 (`hdisc`) is not. The collapse law
    is now blocked on L2 alone. Read the header text below with that correction applied.

  What this file does NOT prove: either identity. As written in 2026-07-31, both were measured:
    (*)          `Q Y a b = Q (τY) (τa) (τb)` for seam Y   -- levels 5..8, 0 violations
                 (scripts/research/cd_tower_zd_fiber_l1_reduction_contract.py, K1)
                 ** SUPERSEDED: now proven forall n, see the UPDATE above **
    (L2-closed)  `disc a b = λ a * λ b` with λ a = ±(-1)^(p j a), p j = parity of bits below j
                 -- all even-weight seams, n = 6..9, 0 violations
                 (scripts/research/cd_tower_zd_fiber_l2_switching_contract.py, M1/M3)

  UPDATE 2026-08-02 -- (c) IS CLOSED. The other leg, L2, is proven too:
    `SounioZDFiberAntisym.L2_forall`, kernel-checked, no sorryAx. So `hdisc` need not be
    assumed either, and `parity_collapse` below discharges BOTH. One adjustment was needed and
    is the same correction `hres` needed: `hdisc` as written quantifies over ALL a, b, and L2
    holds ON THE RESONANCE GRAPH -- off it the identity is false, so nothing could discharge
    the universal form. The resonance is already in hand: `adj Y m p q` carries
    `res Y p.1 q.1 m`, which IS L2's hypothesis, so the discharged instances take `hdisc`
    POINTWISE at the pair the proof uses. The two remaining side conditions (`q.1 != 0`,
    `q.1 ^^^ W != 0`, inherited from `l2_reduction`'s branch condition) exclude NO edge of the
    graph -- 0 of 1920 at n = 6,7 (C8 of the collapse contract).

  So the honest reading, before 2026-08-02, was: **(c) is one machine-checked implication away
  from two explicit sign identities**, neither of which was proven. Both are now proven.

  Why bother formalising the implication and not the identities: the identities are genuine
  inductions -- (*) has a case A4_sub never faced (when Y is a lone power of two, j is the level's
  TOP bit and τ moves it, which is exactly what the four branch reductions hold fixed), and the
  L2 identity additionally carries the weight-parity hypothesis. Stating them precisely and
  discharging everything around them is the part that can be done correctly in one sitting.

  Axioms: [propext, Quot.sound] / [propext, Classical.choice, Quot.sound]. No `sorry`, no
  `native_decide`. Verify with `#print axioms` -- `grep error` is not sufficient.
-/

import SounioZDFiberAntisym

namespace SounioZDCollapse

/-! ## The sign cocycle (carried; identical to SounioZDFiberAntisym) -/

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

theorem cdSq (a b m : Nat) : cdSigma a b m * cdSigma a b m = 1 := by
  rcases cdSigma_pm m a b with h | h <;> rw [h] <;> decide

/-! ## The fiber, the two products, resonance, and the edge sign -/

/-- `P1 L a b = σ(a,b)·σ(a⊕L,b⊕L)`. -/
def P1 (L a b m : Nat) : Int := cdSigma a b m * cdSigma (a ^^^ L) (b ^^^ L) m

/-- `P3 L a b = σ(a,b⊕L)·σ(a⊕L,b)`. -/
def P3 (L a b m : Nat) : Int := cdSigma a (b ^^^ L) m * cdSigma (a ^^^ L) b m

/-- The coset-square product. `res ↔ Q = 1`, since `P1`, `P3` are `±1`. -/
def Q (L a b m : Nat) : Int := P1 L a b m * P3 L a b m

/-- Resonance, in the form the rungs measure it. -/
def res (L a b m : Nat) : Prop := P1 L a b m = P3 L a b m

/-- The edge sign carried by a resonant pair. -/
def eps (L a b m : Nat) : Int := - P1 L a b m

theorem P1_pm (L a b m : Nat) : P1 L a b m = 1 ∨ P1 L a b m = -1 := by
  unfold P1
  rcases cdSigma_pm m a b with h1 | h1 <;>
    rcases cdSigma_pm m (a ^^^ L) (b ^^^ L) with h2 | h2 <;> rw [h1, h2] <;> decide

theorem eps_sq (L a b m : Nat) : eps L a b m * eps L a b m = 1 := by
  unfold eps
  rcases P1_pm L a b m with h | h <;> rw [h] <;> decide

/-- `res` is exactly `Q = 1`: the form the numerical rungs use. -/
theorem res_iff_Q (L a b m : Nat) : res L a b m ↔ Q L a b m = 1 := by
  unfold res Q P1 P3
  constructor
  · intro h
    rw [h]
    rcases cdSigma_pm m a (b ^^^ L) with h1 | h1 <;>
      rcases cdSigma_pm m (a ^^^ L) b with h2 | h2 <;> rw [h1, h2] <;> decide
  · intro h
    rcases cdSigma_pm m a b with e1 | e1 <;>
      rcases cdSigma_pm m (a ^^^ L) (b ^^^ L) with e2 | e2 <;>
      rcases cdSigma_pm m a (b ^^^ L) with e3 | e3 <;>
      rcases cdSigma_pm m (a ^^^ L) b with e4 | e4 <;>
      rw [e1, e2, e3, e4] at h ⊢ <;> revert h <;> decide

/-! ## The collapse map -/

/-- `τ = swap(bit 0, bit j)`. -/
def tau (j x : Nat) : Nat := if (x &&& 1) == ((x >>> j) &&& 1) then x else x ^^^ (1 ||| (1 <<< j))

/-- A vertex of the signed annihilation graph: a lo-label with a sign. -/
abbrev Vtx := Nat × Int

/-- Adjacency in fiber `L`: resonant, and the signs multiply to the edge sign. -/
def adj (L m : Nat) (p q : Vtx) : Prop :=
  res L p.1 q.1 m ∧ p.2 * q.2 = eps L p.1 q.1 m

/-- The collapse map `Φ(lo,s) = (τ lo, λ(lo)·s)`. -/
def Phi (j : Nat) (lam : Nat → Int) (p : Vtx) : Vtx := (tau j p.1, lam p.1 * p.2)

/-! ## The reduction: the two identities suffice -/

/-- Cancellation by two independent signs. Mathlib-free: `ac_rfl` only, no `ring`. -/
theorem sign_cancel {u v w z e : Int} (hu : u * u = 1) (hv : v * v = 1)
    (h : u * w * (v * z) = u * v * e) : w * z = e := by
  have h2 : (u * v) * (u * w * (v * z)) = (u * v) * (u * v * e) := by rw [h]
  have l : (u * v) * (u * w * (v * z)) = (u * u) * (v * v) * (w * z) := by ac_rfl
  have r : (u * v) * (u * v * e) = (u * u) * (v * v) * e := by ac_rfl
  rw [l, r, hu, hv] at h2
  simpa using h2

/-- The forward form of the same cancellation. -/
theorem sign_build {u v w z es ef : Int} (hes : es * es = 1)
    (hwz : w * z = es) (hd : ef * es = u * v) : u * w * (v * z) = ef := by
  have l : u * w * (v * z) = (u * v) * (w * z) := by ac_rfl
  rw [l, hwz, ← hd]
  calc ef * es * es = ef * (es * es) := by ac_rfl
    _ = ef := by rw [hes, Int.mul_one]

/--
**(c) reduced, forall n.** Given
* `hres`  -- this is `(*)`: resonance is `tau`-equivariant between the seam fiber and its Fano
  partner;
* `hdisc` -- this is the L2 closed form: the edge-sign discrepancy is the coboundary of `lam`;
* `hlam`  -- `lam` takes values in `{+-1}` (true for `lam a = +-(-1)^(p j a)`),

then `Phi` carries adjacency in the seam fiber to adjacency in the Fano fiber, and back. That
two-way statement is the content of the parity-collapse law.

NEITHER HYPOTHESIS IS PROVEN HERE -- this is the GENERAL implication, kept with both free.
For the instance with (*) DISCHARGED, see `Phi_preserves_adj_star` / `Phi_reflects_adj_star`
below, where only `hdisc` (L2) remains an assumption.
-/
theorem Phi_preserves_adj
    (Ls Lf j m : Nat) (lam : Nat -> Int)
    (hres : forall a b, res Lf (tau j a) (tau j b) m <-> res Ls a b m)
    (hdisc : forall a b, eps Lf (tau j a) (tau j b) m * eps Ls a b m = lam a * lam b)
    (p q : Vtx) (h : adj Ls m p q) :
    adj Lf m (Phi j lam p) (Phi j lam q) := by
  obtain ⟨hr, hs⟩ := h
  refine ⟨(hres p.1 q.1).mpr hr, ?_⟩
  exact sign_build (eps_sq Ls p.1 q.1 m) hs (hdisc p.1 q.1)

theorem Phi_reflects_adj
    (Ls Lf j m : Nat) (lam : Nat -> Int)
    (hlam : forall x, lam x = 1 ∨ lam x = -1)
    (hres : forall a b, res Lf (tau j a) (tau j b) m <-> res Ls a b m)
    (hdisc : forall a b, eps Lf (tau j a) (tau j b) m * eps Ls a b m = lam a * lam b)
    (p q : Vtx) (h : adj Lf m (Phi j lam p) (Phi j lam q)) :
    adj Ls m p q := by
  obtain ⟨hr, hs⟩ := h
  refine ⟨(hres p.1 q.1).mp hr, ?_⟩
  have hu : lam p.1 * lam p.1 = 1 := by rcases hlam p.1 with h1 | h1 <;> rw [h1] <;> decide
  have hv : lam q.1 * lam q.1 = 1 := by rcases hlam q.1 with h1 | h1 <;> rw [h1] <;> decide
  have hsq : eps Ls p.1 q.1 m * eps Ls p.1 q.1 m = 1 := eps_sq Ls p.1 q.1 m
  have hd := hdisc p.1 q.1
  -- eps_f = lam a * lam b * eps_s
  have hf : eps Lf (tau j p.1) (tau j q.1) m = lam p.1 * lam q.1 * eps Ls p.1 q.1 m := by
    have e2 : eps Lf (tau j p.1) (tau j q.1) m * (eps Ls p.1 q.1 m * eps Ls p.1 q.1 m)
            = lam p.1 * lam q.1 * eps Ls p.1 q.1 m := by
      calc eps Lf (tau j p.1) (tau j q.1) m * (eps Ls p.1 q.1 m * eps Ls p.1 q.1 m)
          = (eps Lf (tau j p.1) (tau j q.1) m * eps Ls p.1 q.1 m) * eps Ls p.1 q.1 m := by ac_rfl
        _ = lam p.1 * lam q.1 * eps Ls p.1 q.1 m := by rw [hd]
    rw [hsq, Int.mul_one] at e2
    exact e2
  have hs' : lam p.1 * p.2 * (lam q.1 * q.2) = lam p.1 * lam q.1 * eps Ls p.1 q.1 m := by
    rw [← hf]; exact hs
  exact sign_cancel hu hv hs'

/-! ## (*) DISCHARGED — the bridge to `SounioZDFiberAntisym`

This file was written with `hres` as a hypothesis because (*) was measured, not proven. It is
proven now (`SounioZDFiberAntisym.star_forall`, 256bdbda4). Wiring it in costs three bridge
lemmas, because this file carries its own copies of `cdSigma` and `tau`: identical bodies, but
DIFFERENT CONSTANTS, so nothing is defeq for a symbolic level and the equality has to be proved
by the same induction the definition uses.

`tau` is not recursive, so its bridge is `rfl`. `cdSigma` is, so its is not. -/

/-- The two carried copies of the cocycle agree. Structural induction on the level; the only
    content is that the four branch bodies are the same up to the recursive call. -/
theorem cdSigma_eq : ∀ (m a b : Nat), cdSigma a b m = SounioZDFiberAntisym.cdSigma a b m
  | 0, _, _ => rfl
  | 1, _, _ => rfl
  | (n+2), a, b => by
      unfold cdSigma SounioZDFiberAntisym.cdSigma
      simp only [cdSigma_eq (n+1)]

/-- `tau` is not recursive: the two copies are definitionally the same map. -/
theorem tau_eq (j x : Nat) : tau j x = SounioZDFiberAntisym.tau j x := rfl

/-- This file's `Q` (as `P1 * P3`) and the sibling's `Qgen` (a flat four-fold product) are the
    same number -- the association differs, nothing else. -/
theorem Q_eq_Qgen (L a b m : Nat) : Q L a b m = SounioZDFiberAntisym.Qgen L a b m := by
  unfold Q P1 P3 SounioZDFiberAntisym.Qgen
  simp only [cdSigma_eq]
  ac_rfl

/-- **`hres` discharged.** For the fiber pair `(Y, tau j Y)` this is exactly (*), and (*) is a
    theorem. The hypothesis `Y % 2^j = 0` is `j <= lsb Y`, which the intended `j = lsb Y`
    satisfies; `a, b < 2^m` is the bound `star_forall` needs and this file's `hres` lacked. -/
theorem res_tau_of_star (m j Y a b : Nat) (hY : Y < 2^m) (hY0 : Y ≠ 0) (hj : Y % 2^j = 0)
    (ha : a < 2^m) (hb : b < 2^m) :
    res (tau j Y) (tau j a) (tau j b) m ↔ res Y a b m := by
  rw [res_iff_Q, res_iff_Q, Q_eq_Qgen, Q_eq_Qgen, tau_eq, tau_eq, tau_eq,
      ← SounioZDFiberAntisym.star_forall m j Y a b hY hY0 hj ha hb]

/-! ### The collapse law with (*) discharged

`Phi_preserves_adj` / `Phi_reflects_adj` above are unchanged -- they are the general
implication and another lane may want them with both hypotheses free. These two are the
instance that matters: the seam fiber `Y`, its partner `tau j Y`, and **`hdisc` (L2) as the
only remaining assumption.**

The bounds `hp`, `hq` are the price of the wiring and are not a weakening in context: the
vertices of the annihilation graph on a level-`m` fiber are lo-labels `< 2^m` by construction.
This is why the general theorems could not simply be re-stated -- their `hres` quantifies over
unbounded `a, b`, where (*) says nothing. -/

theorem Phi_preserves_adj_star (Y j m : Nat) (lam : Nat → Int)
    (hY : Y < 2^m) (hY0 : Y ≠ 0) (hj : Y % 2^j = 0)
    (hdisc : ∀ a b, eps (tau j Y) (tau j a) (tau j b) m * eps Y a b m = lam a * lam b)
    (p q : Vtx) (hp : p.1 < 2^m) (hq : q.1 < 2^m) (h : adj Y m p q) :
    adj (tau j Y) m (Phi j lam p) (Phi j lam q) := by
  obtain ⟨hr, hs⟩ := h
  refine ⟨(res_tau_of_star m j Y p.1 q.1 hY hY0 hj hp hq).mpr hr, ?_⟩
  exact sign_build (eps_sq Y p.1 q.1 m) hs (hdisc p.1 q.1)

theorem Phi_reflects_adj_star (Y j m : Nat) (lam : Nat → Int)
    (hlam : ∀ x, lam x = 1 ∨ lam x = -1)
    (hY : Y < 2^m) (hY0 : Y ≠ 0) (hj : Y % 2^j = 0)
    (hdisc : ∀ a b, eps (tau j Y) (tau j a) (tau j b) m * eps Y a b m = lam a * lam b)
    (p q : Vtx) (hp : p.1 < 2^m) (hq : q.1 < 2^m)
    (h : adj (tau j Y) m (Phi j lam p) (Phi j lam q)) :
    adj Y m p q := by
  obtain ⟨hr, hs⟩ := h
  refine ⟨(res_tau_of_star m j Y p.1 q.1 hY hY0 hj hp hq).mp hr, ?_⟩
  have hu : lam p.1 * lam p.1 = 1 := by rcases hlam p.1 with h1 | h1 <;> rw [h1] <;> decide
  have hv : lam q.1 * lam q.1 = 1 := by rcases hlam q.1 with h1 | h1 <;> rw [h1] <;> decide
  have hsq : eps Y p.1 q.1 m * eps Y p.1 q.1 m = 1 := eps_sq Y p.1 q.1 m
  have hd := hdisc p.1 q.1
  have hf : eps (tau j Y) (tau j p.1) (tau j q.1) m = lam p.1 * lam q.1 * eps Y p.1 q.1 m := by
    have e2 : eps (tau j Y) (tau j p.1) (tau j q.1) m * (eps Y p.1 q.1 m * eps Y p.1 q.1 m)
            = lam p.1 * lam q.1 * eps Y p.1 q.1 m := by
      calc eps (tau j Y) (tau j p.1) (tau j q.1) m * (eps Y p.1 q.1 m * eps Y p.1 q.1 m)
          = (eps (tau j Y) (tau j p.1) (tau j q.1) m * eps Y p.1 q.1 m) * eps Y p.1 q.1 m := by
            ac_rfl
        _ = lam p.1 * lam q.1 * eps Y p.1 q.1 m := by rw [hd]
    rw [hsq, Int.mul_one] at e2
    exact e2
  have hs' : lam p.1 * p.2 * (lam q.1 * q.2) = lam p.1 * lam q.1 * eps Y p.1 q.1 m := by
    rw [← hf]; exact hs
  exact sign_cancel hu hv hs'

/-! ## The intended `lam`: the L2 closed form -/

/-- Parity of the set bits of `x`. -/
def bitParity (x : Nat) : Bool :=
  if h : x = 0 then false
  else Bool.xor (decide (x % 2 = 1)) (bitParity (x / 2))
decreasing_by exact Nat.div_lt_self (Nat.pos_of_ne_zero h) (by decide)

/-- `lam a = -(-1)^(p_j a)`, `p_j` = parity of the bits of `a` BELOW `j` -- the closed form the
    L2 rung measured (M1/M3). Only its `{+-1}`-valuedness is used above. -/
def lamClosed (j a : Nat) : Int :=
  if bitParity (a &&& ((1 <<< j) - 1)) then 1 else -1

theorem lamClosed_pm (j a : Nat) : lamClosed j a = 1 ∨ lamClosed j a = -1 := by
  unfold lamClosed
  split
  · exact Or.inl rfl
  · exact Or.inr rfl

/-! ## ★★★ (c) WITH BOTH LEGS DISCHARGED

`(*)` was wired in on 2026-08-01 (`star_forall`). The other leg, L2, is now proven too —
`SounioZDFiberAntisym.L2_forall` — so `hdisc` need not be assumed: it can be *derived*, at the
pair where the proof actually uses it.

The one adjustment: `hdisc` as written quantifies over all `a, b`, and L2 holds **on the
resonance graph**. That is the same shape as the `hres` correction — an over-strong hypothesis
that nothing could discharge. Here the resonance is already in hand: `adj Y m p q` carries
`res Y p.1 q.1 m`, which is exactly L2's hypothesis. So the discharged instances take it
pointwise. -/

/-- `psg` and `bitParity` are the same recursion, read into `Int` and `Bool`. -/
theorem psg_bitParity : ∀ (x : Nat),
    SounioZDFiberAntisym.psg x = if bitParity x then -1 else 1
  | 0 => by rw [SounioZDFiberAntisym.psg_zero]; unfold bitParity; simp
  | (n+1) => by
      rw [SounioZDFiberAntisym.psg_step]
      rw [bitParity, dif_neg (by omega : ¬ (n+1 = 0))]
      rw [psg_bitParity ((n+1)/2)]
      by_cases hm : (n+1) % 2 = 1
      · have hd : decide ((n+1) % 2 = 1) = true := by simp [hm]
        rw [if_pos hm, hd]
        cases hb : bitParity ((n+1)/2) <;> decide
      · have hd : decide ((n+1) % 2 = 1) = false := by simp [hm]
        rw [if_neg hm, hd]
        cases hb : bitParity ((n+1)/2) <;> decide
  decreasing_by exact Nat.div_lt_self (by omega) (by decide)

/-- The closed-form `λ` of the L2 rung, in the sibling's `psg` language. -/
theorem lamClosed_psg (j a : Nat) :
    lamClosed j a = - SounioZDFiberAntisym.psg (a % 2^j) := by
  unfold lamClosed
  rw [Nat.shiftLeft_eq, Nat.one_mul, Nat.and_two_pow_sub_one_eq_mod,
      psg_bitParity (a % 2^j)]
  cases bitParity (a % 2^j) <;> simp

/-- **The L2 leg, discharged.** `hdisc` at a resonant pair, from `L2_forall`. -/
theorem disc_of_L2 (m j W a b : Nat) (hj : j + 2 ≤ m + 1)
    (hW : W < 2^(m+1)) (ha : a < 2^(m+1)) (hb : b < 2^(m+1))
    (hb0 : b ≠ 0) (hbW : b ^^^ W ≠ 0)
    (hlsb : W % 2^(j+1) = 2^j) (heven : SounioZDFiberAntisym.psg W = 1)
    (hr : res (W + 2^(m+1)) a b (m+2)) :
    eps (tau j (W + 2^(m+1))) (tau j a) (tau j b) (m+2)
        * eps (W + 2^(m+1)) a b (m+2)
      = lamClosed j a * lamClosed j b := by
  -- the resonance hypothesis, in the sibling's form
  have hQ : SounioZDFiberAntisym.Qgen (W + 2^(m+1)) a b (m+2) = 1 := by
    rw [← Q_eq_Qgen]; exact (res_iff_Q _ _ _ _).mp hr
  have hQ' : SounioZDFiberAntisym.Qgen' W a b (m+1) = -1 := by
    rw [SounioZDFiberAntisym.Qred_hi_ll m W a b hW ha hb hb0 hbW] at hQ
    omega
  have hL2 := SounioZDFiberAntisym.L2_forall m j W a b hj hW ha hb hbW hlsb heven hQ'
  -- unfold this file's `eps`/`P1` and match
  unfold eps P1
  rw [tau_eq, tau_eq, tau_eq, ← SounioZDFiberAntisym.tau_xor,
      ← SounioZDFiberAntisym.tau_xor, cdSigma_eq, cdSigma_eq, cdSigma_eq, cdSigma_eq,
      Int.neg_mul_neg]
  rw [hL2, lamClosed_psg, lamClosed_psg, Int.neg_mul_neg]

/-- Bounds and the label's shape, shared by both directions. -/
theorem seam_facts (m j W : Nat) (hj : j + 2 ≤ m + 1) (hW : W < 2^(m+1)) (hjW : W % 2^j = 0) :
    W + 2^(m+1) < 2^(m+2) ∧ W + 2^(m+1) ≠ 0 ∧ (W + 2^(m+1)) % 2^j = 0 := by
  have h2 : (2:Nat)^(m+2) = 2^(m+1) * 2 := by rw [Nat.pow_succ]
  have hpos := Nat.two_pow_pos (m+1)
  refine ⟨by omega, by omega, ?_⟩
  have h1 : (2:Nat)^(m+1) % 2^j = 0 := by
    obtain ⟨c, hc⟩ := Nat.pow_dvd_pow 2 (show j ≤ m+1 by omega)
    rw [hc]; exact Nat.mul_mod_right _ _
  rw [Nat.add_mod, hjW, h1]
  simp

/-- **★★★ (c), BOTH LEGS DISCHARGED — forward.** `Φ` carries adjacency in the seam fiber to
    adjacency in the Fano fiber. `(*)` via `star_forall`, L2 via `L2_forall`; nothing assumed. -/
theorem Phi_preserves_adj_L2 (m j W : Nat) (hj : j + 2 ≤ m + 1)
    (hW : W < 2^(m+1))
    (hlsb : W % 2^(j+1) = 2^j) (heven : SounioZDFiberAntisym.psg W = 1)
    (p q : Vtx) (hp : p.1 < 2^(m+1)) (hq : q.1 < 2^(m+1))
    (hq0 : q.1 ≠ 0) (hqW : q.1 ^^^ W ≠ 0)
    (h : adj (W + 2^(m+1)) (m+2) p q) :
    adj (tau j (W + 2^(m+1))) (m+2) (Phi j (lamClosed j) p) (Phi j (lamClosed j) q) := by
  have hjW : W % 2^j = 0 := by
    rw [← SounioZDFiberAntisym.mod_pow_mod j (j+1) W (by omega), hlsb, Nat.mod_self]
  obtain ⟨hL, hL0, hLj⟩ := seam_facts m j W hj hW hjW
  obtain ⟨hr, hs⟩ := h
  have hpl : p.1 < 2^(m+2) := by
    have h2 : (2:Nat)^(m+2) = 2^(m+1) * 2 := by rw [Nat.pow_succ]
    have := Nat.two_pow_pos (m+1); omega
  have hql : q.1 < 2^(m+2) := by
    have h2 : (2:Nat)^(m+2) = 2^(m+1) * 2 := by rw [Nat.pow_succ]
    have := Nat.two_pow_pos (m+1); omega
  refine ⟨(res_tau_of_star (m+2) j (W + 2^(m+1)) p.1 q.1 hL hL0 hLj hpl hql).mpr hr, ?_⟩
  exact sign_build (eps_sq (W + 2^(m+1)) p.1 q.1 (m+2)) hs
    (disc_of_L2 m j W p.1 q.1 hj hW hp hq hq0 hqW hlsb heven hr)

/-- **★★★ (c), BOTH LEGS DISCHARGED — backward.** Mirrors `Phi_reflects_adj_star`, with
    `hdisc` taken pointwise from `disc_of_L2` instead of assumed. -/
theorem Phi_reflects_adj_L2 (m j W : Nat) (hj : j + 2 ≤ m + 1)
    (hW : W < 2^(m+1))
    (hlsb : W % 2^(j+1) = 2^j) (heven : SounioZDFiberAntisym.psg W = 1)
    (p q : Vtx) (hp : p.1 < 2^(m+1)) (hq : q.1 < 2^(m+1))
    (hq0 : q.1 ≠ 0) (hqW : q.1 ^^^ W ≠ 0)
    (h : adj (tau j (W + 2^(m+1))) (m+2) (Phi j (lamClosed j) p) (Phi j (lamClosed j) q)) :
    adj (W + 2^(m+1)) (m+2) p q := by
  have hjW : W % 2^j = 0 := by
    rw [← SounioZDFiberAntisym.mod_pow_mod j (j+1) W (by omega), hlsb, Nat.mod_self]
  obtain ⟨hL, hL0, hLj⟩ := seam_facts m j W hj hW hjW
  have hpl : p.1 < 2^(m+2) := by
    have h2 : (2:Nat)^(m+2) = 2^(m+1) * 2 := by rw [Nat.pow_succ]
    have := Nat.two_pow_pos (m+1); omega
  have hql : q.1 < 2^(m+2) := by
    have h2 : (2:Nat)^(m+2) = 2^(m+1) * 2 := by rw [Nat.pow_succ]
    have := Nat.two_pow_pos (m+1); omega
  obtain ⟨hr, hs⟩ := h
  have hr' : res (W + 2^(m+1)) p.1 q.1 (m+2) :=
    (res_tau_of_star (m+2) j (W + 2^(m+1)) p.1 q.1 hL hL0 hLj hpl hql).mp hr
  refine ⟨hr', ?_⟩
  have hu : lamClosed j p.1 * lamClosed j p.1 = 1 := by
    rcases lamClosed_pm j p.1 with h1 | h1 <;> rw [h1] <;> decide
  have hv : lamClosed j q.1 * lamClosed j q.1 = 1 := by
    rcases lamClosed_pm j q.1 with h1 | h1 <;> rw [h1] <;> decide
  have hsq : eps (W + 2^(m+1)) p.1 q.1 (m+2) * eps (W + 2^(m+1)) p.1 q.1 (m+2) = 1 :=
    eps_sq (W + 2^(m+1)) p.1 q.1 (m+2)
  have hd := disc_of_L2 m j W p.1 q.1 hj hW hp hq hq0 hqW hlsb heven hr'
  have hf : eps (tau j (W + 2^(m+1))) (tau j p.1) (tau j q.1) (m+2)
          = lamClosed j p.1 * lamClosed j q.1 * eps (W + 2^(m+1)) p.1 q.1 (m+2) := by
    have e2 : eps (tau j (W + 2^(m+1))) (tau j p.1) (tau j q.1) (m+2)
              * (eps (W + 2^(m+1)) p.1 q.1 (m+2) * eps (W + 2^(m+1)) p.1 q.1 (m+2))
            = lamClosed j p.1 * lamClosed j q.1 * eps (W + 2^(m+1)) p.1 q.1 (m+2) := by
      calc eps (tau j (W + 2^(m+1))) (tau j p.1) (tau j q.1) (m+2)
              * (eps (W + 2^(m+1)) p.1 q.1 (m+2) * eps (W + 2^(m+1)) p.1 q.1 (m+2))
          = (eps (tau j (W + 2^(m+1))) (tau j p.1) (tau j q.1) (m+2)
              * eps (W + 2^(m+1)) p.1 q.1 (m+2)) * eps (W + 2^(m+1)) p.1 q.1 (m+2) := by ac_rfl
        _ = lamClosed j p.1 * lamClosed j q.1 * eps (W + 2^(m+1)) p.1 q.1 (m+2) := by rw [hd]
    rw [hsq, Int.mul_one] at e2
    exact e2
  have hs' : lamClosed j p.1 * p.2 * (lamClosed j q.1 * q.2)
           = lamClosed j p.1 * lamClosed j q.1 * eps (W + 2^(m+1)) p.1 q.1 (m+2) := by
    rw [← hf]; exact hs
  exact sign_cancel hu hv hs'

/-- **★★★ (c): THE PARITY-COLLAPSE LAW, ∀n, nothing assumed.** `Φ` is an isomorphism of the
    signed annihilation graph between the seam fiber and its Fano partner. Both legs are
    theorems: `(*)` = `star_forall`, L2 = `L2_forall`. -/
theorem parity_collapse (m j W : Nat) (hj : j + 2 ≤ m + 1)
    (hW : W < 2^(m+1))
    (hlsb : W % 2^(j+1) = 2^j) (heven : SounioZDFiberAntisym.psg W = 1)
    (p q : Vtx) (hp : p.1 < 2^(m+1)) (hq : q.1 < 2^(m+1))
    (hq0 : q.1 ≠ 0) (hqW : q.1 ^^^ W ≠ 0) :
    adj (W + 2^(m+1)) (m+2) p q ↔
      adj (tau j (W + 2^(m+1))) (m+2) (Phi j (lamClosed j) p) (Phi j (lamClosed j) q) :=
  ⟨Phi_preserves_adj_L2 m j W hj hW hlsb heven p q hp hq hq0 hqW,
   Phi_reflects_adj_L2 m j W hj hW hlsb heven p q hp hq hq0 hqW⟩

end SounioZDCollapse
