/-
# Sounio — de Grey field-plane χ ≥ 5 CLOSED: both legs discharged, ZERO hypotheses

`SounioDeGreyChi5Concrete.lean` discharged the *geometry* leg (every G₅₂₉ edge is a
unit pair over the exact symbolic field-plane QF×QF, QF = ℚ(√3,√5,√7,√11)) but left
the *SAT* leg — "G₅₂₉ is not 4-colourable" — as an explicit hypothesis, previously
backed only by the external `cake_lpr` checker.

That hypothesis is now a **theorem**: `SounioSatG529.g529_not_colourable` re-checks
souc_sat's own CDCL LRAT certificate inside Lean core's *verified* LRAT checker
(file-loaded reflection), and the WLOG triangle-precolour leg lifts it to the
unconditional bound. This file wires the two together, closing

    g529_field_plane_chi_ge_5 : ¬ ∃ proper 4-colouring of the QF×QF unit-distance graph

with **no remaining hypotheses, no Mathlib, no `sorry`, no external trust anchor**.

Remaining (clearly isolated): the isometric embedding QF×QF ↪ ℝ² (the field χ≥5 ⇒
Euclidean χ(ℝ²)≥5 step), which needs `Real.sqrt`/`ring` over Mathlib's ℝ.
-/
import SounioDeGreyChi5Concrete
import SounioSatG529

set_option maxRecDepth 1000000
set_option maxHeartbeats 0

namespace DeGrey529.Closed

open UnitDistanceChromatic
open DeGrey529
open DeGrey529.Concrete

/-- The geometry file's edge list and souc_sat's edge list are literally equal
(both are `data/degrey_529.edge`, 0-based, in the same order). -/
theorem edges_eq : DeGrey529.edges.toList = g529_edges := by native_decide

/-- **SAT leg DISCHARGED in Lean.** From `g529_not_colourable` (souc_sat's LRAT
re-checked by Lean core's verified checker), G₅₂₉'s edge graph has no proper
4-colouring `c : Nat → Fin 4`. -/
theorem not_VColourable : ¬ VColourable := by
  rintro ⟨c, hc⟩
  apply g529_not_colourable
  refine ⟨fun i => c i.val, ?_⟩
  intro e he h1 h2
  exact hc e (by rw [edges_eq]; exact he)

/-- **FIELD-PLANE χ(QF²) ≥ 5 — BOTH LEGS DISCHARGED, ZERO HYPOTHESES.**

The exact symbolic field-plane `QF × QF` (QF = ℚ(√3,√5,√7,√11)) with the algebraic
unit-distance relation `unitFP` admits **no** proper 4-colouring — chromatic number
≥ 5. Geometry leg (every G₅₂₉ edge is a unit pair) and SAT leg (G₅₂₉ is not
4-colourable, souc_sat + Lean-core verified LRAT checker via file-loaded reflection,
plus the WLOG triangle-precolour) are **both machine-checked inside Lean**, with no
Mathlib, no external checker as a trust anchor, and no `sorry`. -/
theorem g529_field_plane_chi_ge_5 :
    ¬ Nonempty (PlaneColouring FieldPoint unitFP 4) :=
  g529_field_plane_needs_5_colours not_VColourable

#print axioms g529_field_plane_chi_ge_5

#eval IO.println "SounioDeGreyChi5Closed: field-plane χ(QF²) ≥ 5 fully discharged in Lean core (both legs), no Mathlib; only QF↪ℝ Euclidean embedding remains."

end DeGrey529.Closed
