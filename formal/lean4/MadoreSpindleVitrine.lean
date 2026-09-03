/-
# Sounio — Madore χ(ℝ²) ≥ 4: single-entry showcase ("vitrine")

This file is the **single citation/review entry point** for the Mathlib-free, self-hosted
formalisation of Madore's base-case lower bound χ(ℝ²) ≥ 4 via the Moser spindle over
ℚ(√3,√11). It introduces no new mathematics: it re-states the load-bearing results as a
build-time manifest (`#check` / `#print axioms`), each proved elsewhere with `0 sorryAx`.

The full arc, end to end:

  1. PARAMETRIC FRAMEWORK  (`SounioMultiquadParam.indep_3_11`): the four radicals
     {1, √3, √11, √33} (basis of ℚ(√3,√11), degree 4) are ℚ-linearly independent in the
     from-scratch ℝ, bridged from the de Grey `indep8` restricted to masks {0,1,8,9}.

  2. SPINDLE GEOMETRY  (`SounioMoserSpindleQ311`): 7 vertices (×12 integer scale), 11 unit
     edges with exact `dist² = 144` (all surd parts cancel), `¬3-colourable` + explicit
     4-colouring witness ⇒ χ(spindle) = 4.

  3. FIELD-PLANE REDUCTION  (`q311_plane_needs_4_colours`): generic `UnitDistanceChromatic`
     transfer ⇒ χ(ℚ(√3,√11)²) ≥ 4 — Madore's Lemma 2.4 line (arXiv:1509.07023).

  4. REAL PLANE  (`MoserSpindleQ311.RealPlane.chi_R2_ge_4`): ring homomorphism `phi311 : Qf → Real`
     (`E_mul311`) embeds the spindle in `ℝ × ℝ`; edges carry `dist² = 144` at the native integer
     scale (Euclidean unit after global rescale `1/12`, documented in §7 of the spindle file).

NOTE: light import (no SAT certificate). NOT a default target; build on demand with
`lake build MadoreSpindleVitrine`.
-/
import SounioMoserSpindleQ311Real
import SounioMultiquadParam

set_option maxHeartbeats 1000000

namespace MadoreSpindle.Showcase

open UnitDistanceChromatic
open MoserSpindleQ311
open SounioSqrt.RealCauchyField.Multiquad

/-! ## Build-time manifest of the four load-bearing results (each proved elsewhere, `0 sorryAx`). -/

-- (1) Degree-4 multiquadratic independence {1,√3,√11,√33}.
#check @indep_3_11
-- (2) Spindle not 3-colourable (11 edge disequalities, omega).
#check @spindle_not_3_colourable
-- (3) χ(ℚ(√3,√11)²) ≥ 4 via generic unit-distance reduction.
#check @q311_plane_needs_4_colours
-- (4) χ(ℝ²) ≥ 4 over the Mathlib-free Cauchy-quotient ℝ (phi311 embedding).
#check @MoserSpindleQ311.RealPlane.chi_R2_ge_4

#print axioms MoserSpindleQ311.RealPlane.chi_R2_ge_4

#eval IO.println "MadoreSpindleVitrine: Madore χ(ℝ²) ≥ 4 showcase (indep_3_11, spindle_not_3_colourable, q311_plane_needs_4_colours, chi_R2_ge_4). Mathlib-free, 0 sorryAx."

end MadoreSpindle.Showcase
