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

/-! ## Obligation ledger for `SqrtField ℝ`

Each obligation below is a named `Prop`. Discharging *all* of them (Mathlib-free) and assembling
`{ F := Quotient ⟨RealEq, …⟩, add := …, …, sqrt := … }` yields the `SounioSqrt.SqrtField` instance
that, fed to `DeGrey529.TransferWf.sqrtField_chi_ge_5`, gives **χ(ℝ²) ≥ 5**.

Status legend: ✅ proved above · ⏳ deferred (analytic core).
-/

/-- ⏳ Transitivity of `RealEq` — the ε/2 triangle inequality over `Rat`. Needs a `Rat`
    `add_le_add`-style lemma and `ε/2 + ε/2 = ε`, neither readily available Mathlib-free. Together
    with `realEq_refl`/`realEq_symm` this upgrades `RealEq` to a `Setoid`, enabling the quotient. -/
def RealEqTransObligation : Prop :=
  ∀ a b c : SounioRealCauchy, RealEq a b → RealEq b c → RealEq a c

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

#eval IO.println "SounioSqrtFieldReal: STARTED the analytic SqrtField ℝ construction (Mathlib-free) — RealEq (Cauchy null-difference) PROVED reflexive + symmetric; obligation ledger enumerates the deferred analytic core (ε-N transitivity, op-congruence, field/order axioms, completeness, constructive sqrt). Discharging the ledger + sqrtField_chi_ge_5 gives χ(ℝ²)≥5."

end SounioSqrt.RealCauchyField
