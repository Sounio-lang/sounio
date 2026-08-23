/-
  SounioAntiGarblingModel.lean

  Formal anchor for docs/research/ANTIGARBLING_COMPLETENESS_2026-08-23.md.

  The affine noise model and the TWO orthogonal anti-garbling axes:
    Axis 2 (noise measure): disjoint supports  ⇒  zero covariance
            (the cross term "independence" drops is genuinely absent — NS certificate).
    Axis 1 (multiplication): vanishing associator ⇒ the two parenthesizations agree
            (no order-freedom is hidden — the associator certificate).

  Self-contained: core Lean 4, Int coefficients, no Mathlib.

  STATUS (honest):
    * `cov_pointwise_zero`         — PROVEN (the mathematical core of Axis 2).
    * `cov_zero_of_disjoint`       — statement final; the list-fold step is `sorry`
                                     (mechanical; discharge on `lake build`).
    * `parenthesizations_agree_of_associator_zero` — statement final; `sorry`
                                     (needs `a - b = 0 → a = b` for the carrier).
    * `AntiGarblingComplete`       — the §2.1 dimension-count; stated as a `sorry`
                                     CONJECTURE for the bilinear fragment, not a theorem.
  Do NOT report this file as sorry-free; it is a skeleton with one proven core lemma.
-/

namespace SounioAntiGarbling

/-- A value's noise part: an integer coefficient per independent measurement source.
    `x s = 0` means source `s` is not in the support of `x`. -/
abbrev Noise := Nat → Int

/-- Source `s` is in the support of `x` iff its coefficient is nonzero. -/
def inSupport (x : Noise) (s : Nat) : Prop := x s ≠ 0

/-- Disjoint supports: no single source is nonzero in both operands.
    This is exactly what NS `ns_disjoint` certifies at `ep_add`/`ep_mul`. -/
def disjointSupport (x y : Noise) : Prop := ∀ s, x s = 0 ∨ y s = 0

/-- Covariance of `x` and `y` accumulated over an explicit source list `S`
    (in practice `S` enumerates the union of the two supports).
    GUM's independence assumption drops this term; Axis 2 is about when that is sound. -/
def cov (x y : Noise) (S : List Nat) : Int :=
  (S.map (fun s => x s * y s)).sum

/-- The mathematical core of Axis 2: under disjoint supports every product
    `x s * y s` vanishes. PROVEN. -/
theorem cov_pointwise_zero (x y : Noise) (h : disjointSupport x y) :
    ∀ s, x s * y s = 0 := by
  intro s
  cases h s with
  | inl hx => rw [hx]; exact Int.zero_mul _
  | inr hy => rw [hy]; exact Int.mul_zero _

/-- Axis-2 certificate: disjoint supports ⇒ zero covariance over any source list.
    (The off-diagonal cross term that "independence" would drop is genuinely absent,
    so certifying independence there does not fabricate precision.) -/
theorem cov_zero_of_disjoint (x y : Noise) (S : List Nat)
    (h : disjointSupport x y) : cov x y S = 0 := by
  have hpt := cov_pointwise_zero x y h
  unfold cov
  -- every element of the mapped list is 0 (hpt); the sum of an all-zero list is 0.
  sorry

/-- The associator of a magma operation `mul` on a carrier with subtraction:
    `[x,y,z] = (x·y)·z − x·(y·z)`. Nonzero exactly when order is not free. -/
def associator {α : Type} [Sub α] (mul : α → α → α) (x y z : α) : α :=
  mul (mul x y) z - mul x (mul y z)

/-- Axis-1 certificate: a vanishing associator ⇒ the two parenthesizations of the
    triple product coincide, so reporting a single value hides no order-freedom.
    (For octonions this fails on non-Fano triples — the order garbling the
    `κ·‖[x,y,z]‖²` augmentation must then account for.) -/
theorem parenthesizations_agree_of_associator_zero
    {α : Type} [Sub α] (mul : α → α → α) (x y z : α)
    (h : associator mul x y z = 0) :
    mul (mul x y) z = mul x (mul y z) := by
  -- from `a - b = 0` conclude `a = b` in the additive group carrier.
  sorry

/-- Anti-Garbling Completeness (§2.1), CONJECTURE for the bilinear fragment:
    for product/bilinear propagation over a normed algebra, Axis 1 (associativity)
    and Axis 2 (measure diagonality) are the ONLY structural sources of
    identity-induced variance understatement — a dimension count, since a bilinear
    value's variance depends on the propagator only through reassociation
    (the associator) and the second-moment pairing (the covariance). Curvature is
    the next order (non-bilinear maps), ceded to the `C_H`/Monte-Carlo axis.
    Left as a documented `sorry`; the falsifier is F1 in the companion note. -/
theorem AntiGarblingComplete : True := by
  -- placeholder for the bilinear-fragment completeness statement; see F1.
  trivial

end SounioAntiGarbling
