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

  STATUS (verified): MACHINE-CHECKED by `lean` (Lean 4.33.0,
  leanprover/lean4:v4.33.0) — typechecks clean: exit 0, zero warnings, zero `sorry`
  in code (the only "sorry" token is in this comment). Verified here:
    * Lemma 3 (Axis 2)  — `cov_zero_of_disjoint`.
    * Lemma 4 (Axis 1)  — `parenthesizations_agree_of_associator_zero`.
    * THEOREM 4.1, Axis-2 (sum-node) instance — `antigarbling_sound_sum`: disjoint
      supports ⇒ the naive independence-variance equals the true variance of the
      sum (no fabricated precision), proved through `varSum_expand` and Lemma 3.
    * `var_eq_of_sensitivity_eq` — the congruence shape of Theorem 4.1 for both
      axes (equal sensitivities ⇒ equal variance).
  Still prose-only in the .md companion (NOT asserted machine-checked): the
  Axis-1/product-node instance of Theorem 4.1, the completeness dimension count
  (Prop 2), and the §5 non-separability caveat — these need the full
  sensitivity-propagation model over the algebra.
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

/- ===================================================================== -/
/-  Theorem 4.1 (operational core) — the sum-node / Axis-2 instance,       -/
/-  machine-checked, using Lemma 3.                                        -/
/- ===================================================================== -/

/-- `listSum` distributes over pointwise addition of the mapped function. -/
theorem listSum_map_add (f g : Nat → Int) (S : List Nat) :
    listSum (S.map (fun s => f s + g s))
      = listSum (S.map f) + listSum (S.map g) := by
  induction S with
  | nil => rfl
  | cons a t ih =>
      simp only [List.map_cons, listSum]
      rw [ih]; omega

/-- `listSum` pulls out a scalar factor. -/
theorem listSum_map_smul (c : Int) (f : Nat → Int) (S : List Nat) :
    listSum (S.map (fun s => c * f s)) = c * listSum (S.map f) := by
  induction S with
  | nil => simp only [List.map_nil, listSum, Int.mul_zero]
  | cons a t ih =>
      simp only [List.map_cons, listSum]
      rw [ih, Int.mul_add]

/-- First-order variance of a scalar sensitivity vector over its support list:
    `Var = Σ (d s)²`  (Lemma 0 specialised to scalars). -/
def varN (d : Nat → Int) (S : List Nat) : Int :=
  listSum (S.map (fun s => d s * d s))

/-- The `(dL+dR)² = dL² + dR² + 2·(dL·dR)` expansion, summed:
    true variance of a sum (sensitivities add) = the two naive variances plus
    twice the covariance. -/
theorem varSum_expand (dL dR : Nat → Int) (S : List Nat) :
    listSum (S.map (fun s => (dL s + dR s) * (dL s + dR s)))
      = varN dL S + varN dR S + 2 * cov dL dR S := by
  unfold varN cov
  have hpt : (fun s => (dL s + dR s) * (dL s + dR s))
      = (fun s => dL s * dL s + (dR s * dR s + 2 * (dL s * dR s))) := by
    funext s
    rw [Int.add_mul, Int.mul_add, Int.mul_add, Int.mul_comm (dR s) (dL s)]
    omega
  rw [hpt]
  rw [listSum_map_add (fun s => dL s * dL s)
        (fun s => dR s * dR s + 2 * (dL s * dR s)) S]
  rw [listSum_map_add (fun s => dR s * dR s) (fun s => 2 * (dL s * dR s)) S]
  rw [listSum_map_smul 2 (fun s => dL s * dR s) S]
  omega

/-- THEOREM 4.1 (operational core, sum node): if the operand supports are disjoint
    (Lemma 3 ⇒ zero covariance), the NAIVE variance computed under the independence
    assumption equals the TRUE variance of the sum — no fabricated precision.
    This is the Axis-2 instance of `(Δ_support = 0) → Var_naive = Var_true`. -/
theorem antigarbling_sound_sum (dL dR : Noise) (S : List Nat)
    (hdisj : disjointSupport dL dR) :
    listSum (S.map (fun s => (dL s + dR s) * (dL s + dR s)))
      = varN dL S + varN dR S := by
  rw [varSum_expand]
  rw [cov_zero_of_disjoint dL dR S hdisj]
  omega

/-- Composite congruence (the shape of Theorem 4.1 for BOTH axes): if the true and
    naive propagators agree on the whole sensitivity vector, the variances agree.
    The certificates deliver the hypothesis — Lemma 3 (disjoint ⇒ the naive
    independence bookkeeping matches the shared-symbol sensitivity) and Lemma 4
    (vanishing associator ⇒ the freely-re-associated sensitivity matches the true
    one). The remaining content (that each certificate makes its own correction
    vanish, and the §5 non-separability caveat) is the prose proof companion. -/
theorem var_eq_of_sensitivity_eq (dTrue dNaive : Noise) (S : List Nat)
    (h : ∀ s, dTrue s = dNaive s) : varN dTrue S = varN dNaive S := by
  unfold varN
  have hfun : (fun s => dTrue s * dTrue s) = (fun s => dNaive s * dNaive s) := by
    funext s; rw [h s]
  rw [hfun]

end SounioAntiGarbling
