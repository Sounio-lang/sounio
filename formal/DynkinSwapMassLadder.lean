/-!
# Sounio.DynkinSwapMassLadder — obligation for the J₃(𝕆_ℂ) fermion-mass ladder

Formal companion to `experiments/non_assoc_connectomics/octonion_arc_findings.md`
(§4.4–4.8) and the Sounio artifact `examples/physics/octonion_mass_delta.sio`.

## What is established vs. assumed

The empirical finding: real PDG fermion masses select the single algebraic constant
`δ² = 3/8` of the exceptional Jordan algebra J₃(𝕆_ℂ), with sector eigenvalue-centers
equal to the electric charges. This file separates the **algebraic identities**
(proved here, over ℝ) from the one **representation-theoretic input** (stated only as
a prose obligation — NOT encoded as a proof or a fake axiom):

  * PROVED: the Koide closed form of the arithmetic spectrum `(c−δ, c, c+δ)`;
    that Koide = 2/3 ⟺ δ² = (3/2)c² (so δ² = 3/8 at the diagonal center c = 1/2);
    and that the μ/e "Dynkin-swap" factor `(δ+⅓)/(δ−⅓)` is exactly the `c/a` edge
    ratio with the roles of *center* and *spread* exchanged.

  * OBLIGATION (NOT discharged, see §3): that the E₆ Dynkin Z₂ outer automorphism
    acts on the down→lepton ladder as this center↔spread exchange to the charge
    partner (1 ↔ ⅓). Deriving it from E₆/Spin(8)-triality representation theory is
    a research-paper result, left open.

## Verification status

No Lean toolchain is on the path used for the empirical work, so this file is
**not machine-checked here** (same caveat as `OctonionAssociator.lean`). Every
algebraic identity is independently verified with SymPy in
`experiments/non_assoc_connectomics/scripts/dynkin_swap_symbolic.py` and numerically
(forward, zero free parameter) in `centers_from_charges.py` and the Sounio artifact.
-/

import Mathlib.Tactic

namespace Sounio.DynkinSwapMassLadder

/-- Ascending `c/a` edge ratio for a sector with eigenvalue center `c`, spread `δ`. -/
noncomputable def caEdge (c δ : ℝ) : ℝ := (c + δ) / (c - δ)

/-- `c/b` edge ratio. -/
noncomputable def cbEdge (c δ : ℝ) : ℝ := (c + δ) / c

/-- Koide quantity of the arithmetic √m-spectrum `(c−δ, c, c+δ)`. -/
noncomputable def koideQ (c δ : ℝ) : ℝ :=
  ((c - δ)^2 + c^2 + (c + δ)^2) / ((c - δ) + c + (c + δ))^2

-- §1. Koide closed form and the δ²=3/8 point ---------------------------------

theorem koideQ_eq (c δ : ℝ) (hc : c ≠ 0) :
    koideQ c δ = 2 * δ^2 / (9 * c^2) + 1/3 := by
  have h3 : (c - δ) + c + (c + δ) = 3 * c := by ring
  unfold koideQ
  rw [h3]
  field_simp
  ring

/-- Koide is exactly `2/3` iff `δ² = (3/2)·c²`. -/
theorem koide_two_thirds_iff (c δ : ℝ) (hc : c ≠ 0) :
    koideQ c δ = 2/3 ↔ δ^2 = (3/2) * c^2 := by
  have hc2 : (9 : ℝ) * c^2 ≠ 0 := mul_ne_zero (by norm_num) (pow_ne_zero 2 hc)
  rw [koideQ_eq c δ hc]
  rw [div_add' _ _ _ hc2, div_eq_iff hc2]
  constructor <;> intro h <;> nlinarith [h]

/-- At the diagonal-spectrum center `c = 1/2`: Koide = 2/3 ⟺ `δ² = 3/8`. -/
theorem delta_sq_eq_three_eighths (δ : ℝ) :
    koideQ (1/2) δ = 2/3 ↔ δ^2 = 3/8 := by
  rw [koide_two_thirds_iff (1/2) δ (by norm_num)]
  constructor <;> intro h <;> nlinarith [h]

-- §2. The swap factor is the center↔spread exchange (algebraic, provable) -----

/-- The Dynkin-swap factor `(δ+q)/(δ−q)` is the `c/a` edge ratio with the roles of
    center and spread **exchanged**: `caEdge δ q`. Definitional. -/
theorem swap_factor_is_center_spread_exchange (δ q : ℝ) :
    (δ + q) / (δ - q) = caEdge δ q := rfl

/-- μ/e closed form: the center-1 `c/a` step times the swapped `caEdge δ (1/3)`. -/
noncomputable def muOverE (δ : ℝ) : ℝ := caEdge 1 δ * caEdge δ (1/3)

/-- b/s closed form: two center-1 edges (`c/a` then `c/b`). -/
noncomputable def bOverS (δ : ℝ) : ℝ := caEdge 1 δ * cbEdge 1 δ

/-- μ/e matches Table I: `((1+δ)/(1−δ))·((δ+⅓)/(δ−⅓))`. -/
theorem muOverE_expand (δ : ℝ) :
    muOverE δ = ((1 + δ) / (1 - δ)) * ((δ + 1/3) / (δ - 1/3)) := by
  unfold muOverE caEdge

/-- b/s matches Table I: `((1+δ)/(1−δ))·(1+δ)`. -/
theorem bOverS_expand (δ : ℝ) :
    bOverS δ = ((1 + δ) / (1 - δ)) * (1 + δ) := by
  unfold bOverS caEdge cbEdge
  ring

/-! ## §3. The open obligation (representation theory — NOT proved here)

The single step this file does **not** discharge: that the E₆ Dynkin Z₂ outer
automorphism, restricted to the residual SU(3)_F acting on the Sym³(3) ladder,
sends the down ladder to the lepton ladder by multiplying by the center↔spread
exchanged edge `caEdge δ (1/3)` at the charge partner `1 ↔ 1/3`. Equivalently:
the physical μ/e step equals the physical s/d-style `c/a` step times that factor.

`muOverE` above *defines* the predicted combination and `muOverE_expand` proves it
equals the Table-I form; `swap_factor_is_center_spread_exchange` proves the factor
is the role-exchanged edge. What remains — the content of the obligation — is the
group-theoretic claim that triality *realises* this exchange. We deliberately do
NOT encode that as an axiom: it is an external research result, and asserting it
here would defeat the purpose of separating proof from assumption. -/

end Sounio.DynkinSwapMassLadder
