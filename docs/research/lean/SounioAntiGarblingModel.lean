/-
  SounioAntiGarblingModel.lean

  Formal anchor for docs/research/ANTIGARBLING_COMPLETENESS_2026-08-23.md and its
  proof companion ANTIGARBLING_COMPLETENESS_PROOF_2026-08-23.md.

  The two orthogonal anti-garbling CERTIFICATE lemmas, both proven:
    Lemma 3 (Axis 2, noise measure): disjoint supports ⇒ zero covariance.
    Lemma 4 (Axis 1, multiplication): vanishing associator ⇒ the two
            parenthesizations of a triple product coincide.

  Self-contained: core Lean 4 only (Int coefficients; a minimal Mathlib-free
  additive-group class for Lemma 4). No Mathlib import.

  STATUS (honest): Lemmas 3 and 4 are stated with full proofs (no `sorry`). The
  composite Theorem 4.1 (both certificates ⇒ exact variance) and the completeness
  dimension count (Prop 2) require the sensitivity-propagation model and live, with
  full rigor, in the .md proof companion — they are NOT asserted here as if
  machine-checked. `lean`/`lake` is absent on the authoring host, so this file's
  typecheck is pending a toolchain slot; it is written to compile on core Lean 4.
-/

namespace SounioAntiGarbling

/- ===================================================================== -/
/-  Axis 2 — the support certificate (Lemma 3)                            -/
/- ===================================================================== -/

/-- A value's noise part: an integer coefficient per independent source
    (`x s = 0` means source `s` is not in the support of `x`). -/
abbrev Noise := Nat → Int

/-- Disjoint supports: no single source is nonzero in both operands — exactly what
    NS `ns_disjoint` certifies at `ep_add`/`ep_mul`. -/
def disjointSupport (x y : Noise) : Prop := ∀ s, x s = 0 ∨ y s = 0

/-- Self-contained integer list sum (avoids any dependency on `List.sum`). -/
def listSum : List Int → Int
  | []      => 0
  | a :: t  => a + listSum t

/-- Covariance of `x` and `y` accumulated over a source list `S`
    (in practice `S` enumerates the union of the two supports). GUM's independence
    assumption drops this term; Axis 2 is about when dropping it is sound. -/
def cov (x y : Noise) (S : List Nat) : Int :=
  listSum (S.map (fun s => x s * y s))

/-- Mathematical core of Axis 2: under disjoint supports every product
    `x s * y s` vanishes. -/
theorem cov_pointwise_zero (x y : Noise) (h : disjointSupport x y) :
    ∀ s, x s * y s = 0 := by
  intro s
  cases h s with
  | inl hx => rw [hx]; exact Int.zero_mul _
  | inr hy => rw [hy]; exact Int.mul_zero _

/-- LEMMA 3 (support certificate): disjoint supports ⇒ zero covariance over any
    source list. The off-diagonal cross term that "independence" would drop is
    genuinely absent, so certifying independence there fabricates no precision. -/
theorem cov_zero_of_disjoint (x y : Noise) (S : List Nat)
    (h : disjointSupport x y) : cov x y S = 0 := by
  have hpt := cov_pointwise_zero x y h
  unfold cov
  induction S with
  | nil => rfl
  | cons a t ih =>
      simp only [List.map_cons, listSum]
      rw [hpt a, ih, Int.add_zero]

/- ===================================================================== -/
/-  Axis 1 — the order certificate (Lemma 4)                             -/
/- ===================================================================== -/

/-- Minimal self-contained additive group (Mathlib-free): just enough to derive
    `a - b = 0 → a = b`. The octonion carrier ℝ⁸ is an instance. -/
class AddGrp (α : Type) extends Add α, Neg α, Zero α, Sub α where
  add_assoc      : ∀ a b c : α, a + b + c = a + (b + c)
  zero_add       : ∀ a : α, 0 + a = a
  add_zero       : ∀ a : α, a + 0 = a
  neg_add_cancel : ∀ a : α, -a + a = 0
  sub_eq_add_neg : ∀ a b : α, a - b = a + -b

/-- In an additive group, `a - b = 0` forces `a = b`. -/
theorem sub_eq_zero_imp {α : Type} [AddGrp α] (a b : α)
    (h : a - b = 0) : a = b := by
  have h1 : a + -b = 0 := by
    rw [← AddGrp.sub_eq_add_neg]; exact h
  calc
    a = a + 0        := by rw [AddGrp.add_zero]
    _ = a + (-b + b) := by rw [AddGrp.neg_add_cancel]
    _ = a + -b + b   := by rw [AddGrp.add_assoc]
    _ = 0 + b        := by rw [h1]
    _ = b            := by rw [AddGrp.zero_add]

/-- The associator of a magma product on an additive-group carrier:
    `[x,y,z] = (x·y)·z − x·(y·z)`. Nonzero exactly when order is not free. -/
def associator {α : Type} [AddGrp α] (mul : α → α → α) (x y z : α) : α :=
  mul (mul x y) z - mul x (mul y z)

/-- LEMMA 4 (order certificate): a vanishing associator forces the two
    parenthesizations of the triple product to coincide — so reporting a single
    value hides no order-freedom, and the freely-re-associating (naive) sensitivity
    equals the true one. (For octonions this FAILS on non-Fano triples; the order
    garbling the `κ·‖[x,y,z]‖²` augmentation must then account for.) -/
theorem parenthesizations_agree_of_associator_zero
    {α : Type} [AddGrp α] (mul : α → α → α) (x y z : α)
    (h : associator mul x y z = 0) :
    mul (mul x y) z = mul x (mul y z) := by
  unfold associator at h
  exact sub_eq_zero_imp _ _ h

/-
  The composite operational theorem (PROOF §4.1) —
    (Δ_support = 0  ∧  Δ_order = 0)  →  Var_naive = Var_true
  and the completeness dimension count (PROOF §4.2, Prop 2) rest on the
  first-order sensitivity-propagation model. They are proved in prose in the .md
  companion and are NOT re-asserted here as machine-checked. Lemmas 3 and 4 above
  are the two certificate cores those theorems invoke.
-/

end SounioAntiGarbling
