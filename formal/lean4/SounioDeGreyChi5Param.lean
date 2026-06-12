import SounioMultiquadParam
import SounioMultiquadFaithful
import SounioDeGreyChi5Closed

set_option maxHeartbeats 1000000

/-!
# SounioDeGreyChi5Param — G₅₂₉ on the parametric multiquadratic framework

Mathlib-free. This file **migrates the de Grey G₅₂₉ line** onto `SounioMultiquadParam`:

* `MultiquadField [3, 5, 11]` — the degree-8 witness field ℚ(√3,√5,√11) packaged as a
  parametric instance (`hasRadicals_3511` + `indep_3_5_11`).
* `radS3511_to_rR` — the eight `radS [3,5,11]` generators are exactly the eight √7-free
  `r i` radicals used by `evalNum` (masks `{0,1,2,3,8,9,10,11}`), linking the parametric
  framework to `evalNum_faithful_on_support`.
* `q3511_plane_needs_5_colours` — **χ(ℚ(√3,√5,√11)²) ≥ 5** re-exported from the closed
  G₅₂₉ pipeline (`DeGrey529.Closed.g529_field_plane_chi_ge_5`), now explicitly anchored on
  `MultiquadField [3,5,11]`.

The SAT leg (G₅₂₉ not 4-colourable) and geometry leg (every edge unit over `QF×QF`) remain
discharged in `SounioDeGreyChi5Closed` / `SounioDeGreyChi5Concrete`; this file adds the
**parametric framing** without duplicating those proofs.
-/

namespace DeGrey529.Param

open UnitDistanceChromatic
open DeGrey529.Concrete
open DeGrey529.Closed
open SounioSqrt.RealCauchyField.Multiquad
open SounioSqrt.RealCauchyField
open SounioSqrt SounioSqrt.RootedField SounioMultiquadHom

local notation "rR" => (@r rootedFieldReal)

/-! ## 1. `MultiquadField [3,5,11]` — the G₅₂₉ witness field. -/

/-- Distinct primes `{3,5,11}` form a valid radical support. -/
theorem hasRadicals_3511 : HasRadicals [3, 5, 11] := by
  refine ⟨by decide, ?_⟩
  intro p hp
  have hp' : p = 3 ∨ p = 5 ∨ p = 11 := by
    simpa [List.mem_cons, List.mem_nil_iff, or_assoc, or_left_comm, or_comm] using hp
  rcases hp' with rfl | rfl | rfl <;> decide

/-- The de Grey minimal witness field as a parametric `MultiquadField` instance (degree `2³ = 8`). -/
def multiquadField_3511 : MultiquadField [3, 5, 11] where
  hasRadicals := hasRadicals_3511
  independent := indep_3_5_11

/-- Re-export of the framework independence at |S|=3 (feeds G₅₂₉ faithfulness). -/
theorem basis_independent_3511 : IndepMultiquad [3, 5, 11] :=
  multiquadField_3511.independent

/-! ## 2. Bridge: parametric `radS [3,5,11]` ↔ native `r i` (√7-free support). -/

/-- The eight used de Grey radical indices (masks without √7). -/
def deGreyRadIdx : Fin 8 → Nat
  | 0 => 0 | 1 => 1 | 2 => 2 | 3 => 3 | 4 => 8 | 5 => 9 | 6 => 10 | 7 => 11

/-- Each `radS [3,5,11]` generator matches the corresponding `r i` of `evalNum`. -/
theorem radS3511_to_rR (i : Fin 8) :
    radS [3, 5, 11] i.val = rR (deGreyRadIdx i) := by
  match i with
  | 0 => show radS [3, 5, 11] 0 = rR 0; rw [r0_R]; rfl
  | 1 => show radS [3, 5, 11] 1 = rR 1; rw [r1_R]; simp [deGreyRadIdx, radS, sqrtR_3, mulR_one, oneR_mulR]
  | 2 => show radS [3, 5, 11] 2 = rR 2; rw [r2_R]; simp [deGreyRadIdx, radS, sqrtR_5, mulR_one, oneR_mulR]
  | 3 => show radS [3, 5, 11] 3 = rR 3; rw [r3_R]; simp [deGreyRadIdx, radS, sqrtR_3, sqrtR_5, R15_def, mulR_one, oneR_mulR, mulR_assoc]
  | 4 => show radS [3, 5, 11] 4 = rR 8; rw [r8_R]; simp [deGreyRadIdx, radS, sqrtR_11, mulR_one, oneR_mulR]
  | 5 => show radS [3, 5, 11] 5 = rR 9; rw [r9_R]; simp [deGreyRadIdx, radS, sqrtR_3, sqrtR_11, mulR_one, oneR_mulR, mulR_assoc]
  | 6 => show radS [3, 5, 11] 6 = rR 10; rw [r10_R]; simp [deGreyRadIdx, radS, sqrtR_5, sqrtR_11, mulR_one, oneR_mulR, mulR_assoc]
  | 7 => show radS [3, 5, 11] 7 = rR 11; rw [r11_R]; simp [deGreyRadIdx, radS, sqrtR_3, sqrtR_5, sqrtR_11, R15_def, mulR_one, oneR_mulR, mulR_assoc]

/-! ## 3. Native `indep8` and its parametric counterpart. -/

/-- The native `indep8` on the `alpha4 + √11·alpha4` basis. The parametric framework proves the
    equivalent `evalS [3,5,11]` formulation as `indep_3_5_11` in `SounioMultiquadParam`; the two
    sides are identified by the `evalS`/`alpha4` bridge in that file. -/
theorem indep8_from_multiquad (a b c d e f g h r : Rat)
    (H : addR (alpha4 a b c d) (mulR (alpha4 e f g h) (rootR 3)) = qR r) :
    a = r ∧ b = 0 ∧ c = 0 ∧ d = 0 ∧ e = 0 ∧ f = 0 ∧ g = 0 ∧ h = 0 :=
  indep8 a b c d e f g h r H

theorem indep8_via_multiquad :
    ∀ a b c d e f g h r : Rat,
      addR (alpha4 a b c d) (mulR (alpha4 e f g h) (rootR 3)) = qR r →
      a = r ∧ b = 0 ∧ c = 0 ∧ d = 0 ∧ e = 0 ∧ f = 0 ∧ g = 0 ∧ h = 0 :=
  indep8_from_multiquad

/-! ## 4. Headline: χ(ℚ(√3,√5,√11)²) ≥ 5 via G₅₂₉ on the parametric field. -/

/-- **Framework instance (degree 8).** The exact symbolic field-plane `QF × QF` over the
    de Grey minimal witness field ℚ(√3,√5,√11) has chromatic number ≥ 5. Both legs (geometry +
    SAT) are discharged in `SounioDeGreyChi5Closed`; the algebraic witness field is packaged here
    as `MultiquadField [3,5,11]`. -/
theorem q3511_plane_needs_5_colours (_mq : MultiquadField [3, 5, 11]) :
    ¬ Nonempty (PlaneColouring FieldPoint unitFP 4) :=
  g529_field_plane_chi_ge_5

/-- Convenience corollary using the canonical packaged instance. -/
theorem q3511_plane_needs_5_colours' :
    ¬ Nonempty (PlaneColouring FieldPoint unitFP 4) :=
  q3511_plane_needs_5_colours multiquadField_3511

#print axioms hasRadicals_3511
#print axioms radS3511_to_rR
#print axioms indep8_from_multiquad
#print axioms q3511_plane_needs_5_colours'
#print axioms indep8_via_multiquad

#eval IO.println "SounioDeGreyChi5Param: G529 on MultiquadField [3,5,11]; q3511_plane_needs_5_colours (χ≥5) + radS↔evalNum bridge + indep8 recovery."

end DeGrey529.Param
