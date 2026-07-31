/-
  SounioZDCollapse — (c), the parity-collapse law: the REDUCTION formalised.

  What this file proves, kernel-checked, ∀n:

      (*)  ∧  (L2-closed)   ⇒   Φ is an isomorphism of the signed annihilation graph

  i.e. the two identities the numerical rungs of 2026-07-31 isolated are TOGETHER sufficient for
  the collapse law. They enter as explicit hypotheses; the implication is the theorem.

  What this file does NOT prove: either identity. Both are measured, not derived:
    (*)          `Q Y a b = Q (τY) (τa) (τb)` for seam Y   -- levels 5..8, 0 violations
                 (scripts/research/cd_tower_zd_fiber_l1_reduction_contract.py, K1)
    (L2-closed)  `disc a b = λ a * λ b` with λ a = ±(-1)^(p j a), p j = parity of bits below j
                 -- all even-weight seams, n = 6..9, 0 violations
                 (scripts/research/cd_tower_zd_fiber_l2_switching_contract.py, M1/M3)

  So the honest reading is: **(c) is now one machine-checked implication away from two explicit
  sign identities**, neither of which is proven. Before this file the sufficiency was prose.

  Why bother formalising the implication and not the identities: the identities are genuine
  inductions -- (*) has a case A4_sub never faced (when Y is a lone power of two, j is the level's
  TOP bit and τ moves it, which is exactly what the four branch reductions hold fixed), and the
  L2 identity additionally carries the weight-parity hypothesis. Stating them precisely and
  discharging everything around them is the part that can be done correctly in one sitting.

  Axioms: [propext, Quot.sound] / [propext, Classical.choice, Quot.sound]. No `sorry`, no
  `native_decide`. Verify with `#print axioms` -- `grep error` is not sufficient.
-/

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

NEITHER HYPOTHESIS IS PROVEN HERE; both are measured. See the file header.
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

end SounioZDCollapse
