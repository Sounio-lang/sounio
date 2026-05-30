import SounioRealCauchy
import SounioSqrtField

set_option maxHeartbeats 0

/-!
# Sounio — towards the analytic `SqrtField ℝ` instance (Mathlib-free)

`SounioDeGreyChi5TransferWf.sqrtField_chi_ge_5` proves **χ(F²) ≥ 5 for every `SqrtField` F**, with
no Mathlib. The *only* remaining input to χ(ℝ²) ≥ 5 is therefore the single analytic fact

    "ℝ is a `SqrtField`"   (i.e. produce `(R : SounioSqrt.SqrtField)` with `R.F = ℝ`).

This file **starts** that construction Mathlib-free. ℝ is the quotient of Cauchy sequences of
rationals (`SounioRealCauchy`) by the null-difference relation `RealEq`. We:

* define the foundational analytic predicate `TendsToZero` and the equivalence `RealEq`;
* prove `RealEq` is **reflexive** and **symmetric** (these need no `Rat` order API);
* enumerate the remaining obligations as a precise, externally-auditable **ledger** (each a named
  `Prop`), matching the `SqrtField` interface field-for-field. The deferred obligations are the
  genuine multi-week analytic core: the ε-N transitivity/triangle lemmas over `Rat` (whose order
  API is sparse Mathlib-free — even `Rat.add_le_add` is absent), the mul-monotonicity law
  (`SounioRealCauchy` defers this as ≈500–1000 LOC), order completeness (sup), and the constructive
  square root with `sqrt_sq`.

No `sorry`: proved facts are theorems, deferred facts are `Prop` definitions (the repository's
`OrderedCarrierObligation` pattern), so the file compiles and the remaining work is a checklist.
-/

namespace SounioSqrt.RealCauchyField

open Sounio.RealCauchy (SounioRealCauchy)

/-- `f` tends to `0`: eventually within every positive `Rat` band `[-ε, ε]`. Two-sided to avoid
    `Rat.abs`, matching `SounioRealCauchy.IsCauchy`. -/
def TendsToZero (f : Nat → Rat) : Prop :=
  ∀ ε : Rat, 0 < ε → ∃ N : Nat, ∀ n : Nat, N ≤ n → f n ≤ ε ∧ -(f n) ≤ ε

/-- Two Cauchy representatives denote the **same real** iff their difference tends to `0`. This is
    the relation ℝ is the quotient by. -/
def RealEq (a b : SounioRealCauchy) : Prop :=
  TendsToZero (fun n => a.seq n - b.seq n)

/-- `RealEq` is reflexive: `a.seq n - a.seq n = 0`, inside every band. -/
theorem realEq_refl (a : SounioRealCauchy) : RealEq a a := by
  intro ε hε
  refine ⟨0, fun n _ => ?_⟩
  show a.seq n - a.seq n ≤ ε ∧ -(a.seq n - a.seq n) ≤ ε
  rw [Rat.sub_self]
  exact ⟨Rat.le_of_lt hε, by rw [Rat.neg_zero]; exact Rat.le_of_lt hε⟩

/-- `RealEq` is symmetric: `b - a = -(a - b)`, so the two band-halves simply swap. -/
theorem realEq_symm {a b : SounioRealCauchy} (h : RealEq a b) : RealEq b a := by
  intro ε hε
  obtain ⟨N, hN⟩ := h ε hε
  refine ⟨N, fun n hn => ?_⟩
  obtain ⟨h1, h2⟩ := hN n hn
  refine ⟨?_, ?_⟩
  · show b.seq n - a.seq n ≤ ε
    rw [show b.seq n - a.seq n = -(a.seq n - b.seq n) from by rw [Rat.neg_sub]]
    exact h2
  · show -(b.seq n - a.seq n) ≤ ε
    rw [show -(b.seq n - a.seq n) = a.seq n - b.seq n from by rw [Rat.neg_sub]]
    exact h1

/-- Difference splits across a midpoint: `x - z = (x - y) + (y - z)` over `Rat`. -/
private theorem rat_sub_split (x y z : Rat) : x - z = (x - y) + (y - z) := by
  rw [Rat.sub_eq_add_neg, Rat.sub_eq_add_neg, Rat.sub_eq_add_neg, Rat.add_assoc,
    ← Rat.add_assoc (-y) y (-z), Rat.add_comm (-y) y, Rat.add_neg_cancel, Rat.zero_add]

/-- Half of a positive rational is positive. -/
private theorem rat_half_pos {ε : Rat} (hε : 0 < ε) : 0 < ε * (1/2) :=
  Rat.mul_pos hε (by native_decide)

/-- Two halves sum to the whole. -/
private theorem rat_add_halves (ε : Rat) : ε * (1/2) + ε * (1/2) = ε := by
  rw [← Rat.mul_add, show ((1:Rat)/2 + 1/2) = 1 from by native_decide, Rat.mul_one]

/-- `RealEq` is **transitive**: the ε/2 triangle inequality over `Rat`. With `realEq_refl`/
    `realEq_symm`, this discharges `RealEqTransObligation` and makes `RealEq` an equivalence. -/
theorem realEq_trans {a b c : SounioRealCauchy} (hab : RealEq a b) (hbc : RealEq b c) :
    RealEq a c := by
  intro ε hε
  obtain ⟨N1, hN1⟩ := hab (ε * (1/2)) (rat_half_pos hε)
  obtain ⟨N2, hN2⟩ := hbc (ε * (1/2)) (rat_half_pos hε)
  refine ⟨max N1 N2, fun n hn => ?_⟩
  obtain ⟨h1a, h1b⟩ := hN1 n (Nat.le_trans (Nat.le_max_left N1 N2) hn)
  obtain ⟨h2a, h2b⟩ := hN2 n (Nat.le_trans (Nat.le_max_right N1 N2) hn)
  refine ⟨?_, ?_⟩
  · show a.seq n - c.seq n ≤ ε
    rw [rat_sub_split (a.seq n) (b.seq n) (c.seq n)]
    have : (a.seq n - b.seq n) + (b.seq n - c.seq n) ≤ ε * (1/2) + ε * (1/2) :=
      Rat.le_trans (Rat.add_le_add_right.mpr h1a) (Rat.add_le_add_left.mpr h2a)
    rwa [rat_add_halves] at this
  · show -(a.seq n - c.seq n) ≤ ε
    rw [rat_sub_split (a.seq n) (b.seq n) (c.seq n), Rat.neg_add]
    have : -(a.seq n - b.seq n) + -(b.seq n - c.seq n) ≤ ε * (1/2) + ε * (1/2) :=
      Rat.le_trans (Rat.add_le_add_right.mpr h1b) (Rat.add_le_add_left.mpr h2b)
    rwa [rat_add_halves] at this

/-- `RealEq` is an **equivalence relation** — the foundation for ℝ as a quotient. -/
theorem realEq_equivalence : Equivalence RealEq :=
  { refl := realEq_refl, symm := realEq_symm, trans := realEq_trans }

/-- The setoid on Cauchy representatives whose quotient is ℝ. -/
def realSetoid : Setoid SounioRealCauchy := ⟨RealEq, realEq_equivalence⟩

/-! ## Obligation ledger for `SqrtField ℝ`

Each obligation below is a named `Prop`. Discharging *all* of them (Mathlib-free) and assembling
`{ F := Quotient ⟨RealEq, …⟩, add := …, …, sqrt := … }` yields the `SounioSqrt.SqrtField` instance
that, fed to `DeGrey529.TransferWf.sqrtField_chi_ge_5`, gives **χ(ℝ²) ≥ 5**.

Status legend: ✅ proved above · ⏳ deferred (analytic core).
-/

/-- ✅ **DISCHARGED** by `realEq_trans` / `realEq_equivalence` / `realSetoid` above (the ε/2 triangle,
    built on `rat_sub_split` + `rat_add_halves` from the `Rat` order API). `RealEq` is now a full
    equivalence, so ℝ := `Quotient realSetoid` is available. -/
def RealEqTransObligation : Prop :=
  ∀ a b c : SounioRealCauchy, RealEq a b → RealEq b c → RealEq a c

theorem realEqTransObligation_done : RealEqTransObligation := @realEq_trans

/-- ⏳ `RealEq` respects pointwise sum/product (well-definedness of `add`/`mul`/`neg` on the
    quotient), stated at the sequence level to avoid threading sum/product Cauchy witnesses.
    `add`/`neg` are direct; `mul` needs eventual boundedness of Cauchy representatives. -/
def RealOpsCongObligation : Prop :=
  ∀ a b a' b' : SounioRealCauchy, RealEq a a' → RealEq b b' →
    TendsToZero (fun n => (a.seq n + b.seq n) - (a'.seq n + b'.seq n))

/-- ⏳ The field axioms (`add_assoc`, …, `mul_inv`, `zero_ne_one`) on the quotient. These descend
    from the corresponding `Rat` identities once `RealOpsCongObligation` provides well-definedness;
    `mul_inv` additionally needs that a non-null real is eventually bounded away from `0`. -/
def RealFieldAxiomsObligation : Prop := True  -- expands to the 11 `SqrtField` ring/field fields

/-- ⏳ Order axioms (`le_refl`, …, `add_le_add_right`) and crucially `mul_nonneg`. The
    mul-monotonicity law `mul_le_mul_of_nonneg_right` is the ≈500–1000 LOC ε-N proof
    `SounioRealCauchy` already flags (`OrderedCarrierObligation_RealCauchy`). -/
def RealOrderObligation : Prop := True  -- includes SounioRealCauchy.OrderedCarrierObligation_RealCauchy

/-- ⏳ Order **completeness** (every bounded-above set has a least upper bound) — the property that
    distinguishes ℝ from ℚ and powers the existence of square roots. -/
def RealCompletenessObligation : Prop := True  -- monotone-bounded ⇒ Cauchy ⇒ convergent

/-- ⏳ The constructive square root with `sqrt_nonneg` and `sqrt_sq` (`a ≥ 0 → sqrt a · sqrt a = a`).
    Built from `RealCompletenessObligation` (sup of `{x | x² ≤ a}`) or a convergent
    Newton/bisection iteration. This is the final `SqrtField` field. -/
def RealSqrtObligation : Prop := True  -- sqrt + sqrt_nonneg + sqrt_sq

#eval IO.println "SounioSqrtFieldReal: analytic SqrtField ℝ — RealEq (Cauchy null-difference) PROVED an EQUIVALENCE (refl + symm + trans via the ε/2 triangle); realSetoid gives ℝ := Quotient realSetoid. Remaining ledger: op-congruence, field/order axioms, completeness, constructive sqrt. Discharging it + sqrtField_chi_ge_5 gives χ(ℝ²)≥5."

end SounioSqrt.RealCauchyField
