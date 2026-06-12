/-
# Sounio — de Grey χ(ℝ²) ≥ 5: single-entry showcase ("vitrine")

This file is the **single citation/review entry point** for the Mathlib-free, self-hosted
formalisation of de Grey's lower bound χ(ℝ²) ≥ 5, together with its algebraic *sharpening*.
It introduces no new mathematics: it re-states the abstract sharp theorem as a one-liner and
provides a build-time manifest (`#check` / `#print axioms`) of the four load-bearing results,
each proved elsewhere with `0 sorryAx` and reviewed by an orthogonal LLM (see
`.claude/llm_offload_log.md`, 2026-05-30).

The full arc, end to end:

  1. EMPIRICAL  (`_radical_support_probe`): the G₅₂₉ unit-distance graph touches only the
     radicals of S = {3, 5, 11}. √7 appears in 0 of 529 vertices and 0 of 2670 edges.
     ⇒ the minimal witness field is the multiquadratic ℚ(√3, √5, √11), a degree-8 extension.

  2. ALGEBRAIC TOWER  (`SounioSqrt.RealCauchyField.indep8`): the 8 radicals
     {1, √3, √5, √15, √11, √33, √55, √165} (the basis of ℚ(√3,√5,√11)) are ℚ-linearly
     independent in the from-scratch ℝ. Proved through the tower ℚ ⊂ ℚ(√3) ⊂ ℚ(√3,√5) ⊂
     ℚ(√3,√5,√11), with an iterated subfield inverse (no Galois theory, no flat degree-4 norm).
     ⇒ the witness field is genuinely degree 8; there is no hidden algebraic collapse.
     **Parametric re-framing:** `SounioDeGreyChi5Param` packages the same support as
     `MultiquadField [3,5,11]` (`indep_3_5_11`, `radS3511_to_rR`, `q3511_plane_needs_5_colours`).

  3. FAITHFULNESS  (`SounioSqrt.RealCauchyField.evalNum_faithful_on_support`): the abstract
     QF → ℝ numerator map `evalNum` (the numerator of the fraction homomorphism `φ` of
     `SounioMultiquadHom`) is injective on the √7-free support. ⇒ the symbolic QF calculus
     realises faithfully in the real plane.

  4. RESULT  (`DeGrey529.RealPlane.chi_R2_ge_5_unconditional`): the unit-distance graph on the
     Mathlib-free real plane ℝ × ℝ admits no proper 4-colouring — both legs discharged (the
     geometry/transfer leg + the SAT leg via souc_sat's CDCL LRAT certificate re-checked in
     Lean core + WLOG triangle-precolour). No remaining hypotheses, no Mathlib.

The sharp abstract theorem below packages (4) for an ARBITRARY `RootedField` — i.e. any ordered
field equipped with √3, √5, √7, √11 — and (2)+(3) certify that the field it factors through is
genuinely degree 8, with the witness ℝ a concrete, faithful instance.

NOTE: heavy import (transitively pulls in the ~30 MB G₅₂₉ certificate re-checked under
`native_decide`). NOT a default target; build on demand with `lake build DeGreyChi5Vitrine`.
-/
import SounioDeGreyChi5Real
import SounioMultiquadIndep
import SounioMultiquadFaithful

set_option maxHeartbeats 0

namespace DeGrey529.Showcase

open UnitDistanceChromatic

/-- **Sharp abstract theorem: χ(F²) ≥ 5 for every `RootedField F`.**

For any `SounioSqrt.RootedField F` — an ordered field equipped with square roots of 3, 5, 7, 11 —
the unit-distance graph on `F × F` (with squared distance measured through the de Grey transfer)
admits no proper 4-colouring. The de Grey chromatic obstruction is therefore *purely algebraic*:
it holds in every ordered field containing the relevant radicals, with the Mathlib-free real
numbers (`SounioSqrt.RealCauchyField.rootedFieldReal`) merely one concrete witness.

The empirical support analysis (S = {3,5,11}, √7 unused) plus `indep8` further pin the minimal
witness field as the genuinely degree-8 ℚ(√3,√5,√11); `evalNum_faithful_on_support` certifies the
QF → ℝ map is faithful there, so no algebraic collapse hides behind the bound. -/
theorem deGrey_chi_ge_5_any_rootedField (F : SounioSqrt.RootedField) :
    ¬ Nonempty (PlaneColouring
        ((DeGrey529.TransferWf.rootedTransfer F).F × (DeGrey529.TransferWf.rootedTransfer F).F)
        (DeGrey529.TransferWf.rootedTransfer F).unit 4) :=
  DeGrey529.TransferWf.rootedField_chi_ge_5 F DeGrey529.Closed.not_VColourable

/-! ## Build-time manifest of the four load-bearing results (each proved elsewhere, `0 sorryAx`). -/

-- (4) χ(ℝ²) ≥ 5, unconditional, over the Mathlib-free Cauchy-quotient ℝ.
#check @DeGrey529.RealPlane.chi_R2_ge_5_unconditional
-- (2) ℚ-linear independence of the 8 radicals of ℚ(√3,√5,√11) in the from-scratch ℝ.
#check @SounioSqrt.RealCauchyField.indep8
-- (3) faithfulness of the abstract QF → ℝ numerator map on the √7-free support.
#check @SounioSqrt.RealCauchyField.evalNum_faithful_on_support
-- SAT leg: G₅₂₉ is not 4-colourable (souc_sat LRAT re-checked in Lean core).
#check @DeGrey529.Closed.not_VColourable

#print axioms deGrey_chi_ge_5_any_rootedField

#eval IO.println "DeGreyChi5Vitrine: single-entry showcase. deGrey_chi_ge_5_any_rootedField (χ(F²) ≥ 5 for every RootedField F) + manifest of the four pillars (chi_R2_ge_5_unconditional, indep8, evalNum_faithful_on_support, not_VColourable). Mathlib-free, 0 sorryAx."

end DeGrey529.Showcase
