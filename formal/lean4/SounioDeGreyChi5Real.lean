/-
# Sounio — χ(ℝ²) ≥ 5 UNCONDITIONAL over the Mathlib-free Cauchy-quotient ℝ

`SounioRootedFieldReal.lean` assembled the Mathlib-free real numbers
`Real := Quotient realSetoid` into a `SounioSqrt.RootedField` and fired the abstract
de Grey transfer to obtain `chi_R2_ge_5`, *conditional* on the SAT leg
`¬ DeGrey529.Concrete.VColourable`.

`SounioDeGreyChi5Closed.lean` discharged that SAT leg unconditionally
(`DeGrey529.Closed.not_VColourable`), via souc_sat's CDCL LRAT certificate re-checked
by Lean core's verified LRAT checker plus the WLOG triangle-precolour leg.

This file feeds the second into the first, closing the QF↪ℝ gap: the unit-distance
graph on the Mathlib-free real plane ℝ×ℝ has NO proper 4-colouring, with NO remaining
hypotheses — χ(ℝ²) ≥ 5.

NOTE: heavy import (pulls in the ~30 MB G₅₂₉ certificate via SounioDeGreyChi5Closed,
re-checked under native_decide). NOT a default_target; build on demand with
`lake build SounioDeGreyChi5Real`.
-/
import SounioRootedFieldReal
import SounioDeGreyChi5Closed

set_option maxHeartbeats 0

namespace DeGrey529.RealPlane

/-- **χ(ℝ²) ≥ 5 — UNCONDITIONAL, Mathlib-free.**

The unit-distance graph on the Mathlib-free real plane `ℝ × ℝ`
(`ℝ := SounioSqrt.RealCauchyField.Real = Quotient realSetoid`, the Cauchy-sequence
quotient assembled into a `SounioSqrt.RootedField`) admits **no** proper 4-colouring.
Both legs are discharged: the geometry/transfer leg (`chi_R2_ge_5`, conditional on the
SAT leg) and the SAT leg (`DeGrey529.Closed.not_VColourable`, souc_sat LRAT re-checked
in Lean core + WLOG triangle-precolour). No remaining hypotheses, no Mathlib. -/
theorem chi_R2_ge_5_unconditional :
    ¬ Nonempty (UnitDistanceChromatic.PlaneColouring
      (SounioSqrt.RealCauchyField.Real × SounioSqrt.RealCauchyField.Real)
      (DeGrey529.TransferWf.rootedTransfer SounioSqrt.RealCauchyField.rootedFieldReal).unit 4) :=
  SounioSqrt.RealCauchyField.chi_R2_ge_5 DeGrey529.Closed.not_VColourable

#print axioms chi_R2_ge_5_unconditional

#eval IO.println "SounioDeGreyChi5Real: χ(ℝ²) ≥ 5 UNCONDITIONAL over the Mathlib-free Cauchy-quotient ℝ. Both legs discharged (geometry/transfer via chi_R2_ge_5, SAT via DeGrey529.Closed.not_VColourable); no remaining hypotheses, no Mathlib, no sorry."

end DeGrey529.RealPlane
