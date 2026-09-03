import SounioDeGreyUnitDistance
import SounioDeGreyChi5

set_option maxRecDepth 100000
set_option maxHeartbeats 0

/-!
# Sounio — de Grey χ(ℝ²) ≥ 5: GEOMETRY LEG DISCHARGED over the symbolic field-plane

This file instantiates the abstract reduction of `SounioDeGreyChi5.lean` on the *concrete*
de Grey core G₅₂₉, turning the **geometry leg from an external hypothesis into a PROVED
fact** — entirely inside Lean, with no Mathlib and no real analysis.

The point set is the **exact symbolic field-plane** `QF × QF`, where `QF` is the
ℚ(√3,√5,√7,√11) 16-tuple model from `SounioDeGreyUnitDistance.lean`. The unit relation
`unitFP p q` is "the exact algebraic squared distance between `p` and `q` equals 1".
Every edge of G₅₂₉ satisfies it (discharged by the existing `native_decide` certificate),
so the reduction of `SounioDeGreyChi5` applies with the geometry hypothesis **closed**.

What is now machine-checked end-to-end (modulo the single remaining SAT hypothesis):
  G₅₂₉ embeds in the exact field-plane with every edge at symbolic unit distance, and if
  G₅₂₉ is not 4-colourable (SAT leg, cake_lpr) then the field-plane unit-distance graph has
  **no proper 4-colouring**, i.e. its chromatic number is ≥ 5.

REMAINING GAP (the only one, clearly isolated): the field-plane `QF × QF` with `unitFP`
embeds isometrically into the Euclidean plane ℝ² via `b_mask ↦ √(∏ primes)`. That embedding
is a ring homomorphism `QF → ℝ` sending the basis radicals to actual real radicals; proving
it (`√3·√5 = √15` in ℝ, etc., and `eval` is a ring hom) needs real analysis — i.e. Mathlib's
ℝ, `Real.sqrt`, and the `ring` tactic, none of which are available in this core-Lean project.
That is the standard multiquadratic-field fact and is staged as the `ℚ(√…)↪ℝ` bridge.
-/

namespace DeGrey529.Concrete

open UnitDistanceChromatic
open DeGrey529

/-- A point of the exact symbolic field-plane: a pair of ℚ(√3,√5,√7,√11) elements. -/
abbrev FieldPoint := QF × QF

/-- Embedding of vertex index `v` to its exact symbolic coordinates `(X v, Y v)`. -/
def emb (v : Nat) : FieldPoint := (X.getD v ([], 1), Y.getD v ([], 1))

/-- Exact symbolic unit relation: the algebraic squared distance equals 1.
    By construction this matches `dist2`/`edgeUnit` definitionally on embedded points. -/
def unitFP (p q : FieldPoint) : Prop :=
  isOne (qadd (qmul (qsub p.1 q.1) (qsub p.1 q.1))
              (qmul (qsub p.2 q.2) (qsub p.2 q.2))) = true

/-- `unitFP` on embedded vertices is exactly `edgeUnit`. -/
theorem unitFP_emb (a b : Nat) : unitFP (emb a) (emb b) ↔ edgeUnit (a, b) = true := by
  unfold unitFP emb edgeUnit dist2
  exact Iff.rfl

/-- A vertex `4`-colouring of G₅₂₉'s edge graph. -/
def VColourable : Prop :=
  ∃ c : Nat → Fin 4, ∀ e ∈ edges.toList, c e.1 ≠ c e.2

/-- **Geometry leg — PROVED.** Every edge of G₅₂₉ is a `unitFP` pair under `emb`.
    Discharged by the same exact computation as `g529_all_edges_unit_distance`. -/
theorem geom_all_edges_unitFP : ∀ e ∈ edges.toList, unitFP (emb e.1) (emb e.2) := by
  have hall : ∀ e ∈ edges.toList, edgeUnit e = true := by native_decide
  intro e he
  have h := hall e he
  -- `unitFP (emb e.1) (emb e.2)` is defeq to `edgeUnit e = true`
  exact (unitFP_emb e.1 e.2).2 (by simpa using h)

/-- **Reduction (geometry leg discharged).** Any proper 4-colouring of the field-plane
    unit-distance graph restricts, along `emb`, to a proper 4-colouring of G₅₂₉. -/
theorem plane_colouring_gives_vcolouring
    (pc : PlaneColouring FieldPoint unitFP 4) : VColourable := by
  obtain ⟨κ, hκ⟩ := pc
  refine ⟨fun v => κ (emb v), ?_⟩
  intro e he
  exact hκ (emb e.1) (emb e.2) (geom_all_edges_unitFP e he)

/-- **Main result of this file.** With the geometry leg proved, the SAT leg (G₅₂₉ not
    4-colourable, verified by `cake_lpr`; see `examples/erdos/CAKE_LPR_RESULT.md`) is the
    *only* remaining hypothesis. Under it, the exact field-plane unit-distance graph has no
    proper 4-colouring — chromatic number ≥ 5. -/
theorem g529_field_plane_needs_5_colours
    (h_sat : ¬ VColourable) : ¬ Nonempty (PlaneColouring FieldPoint unitFP 4) := by
  intro ⟨pc⟩
  exact h_sat (plane_colouring_gives_vcolouring pc)

#print axioms geom_all_edges_unitFP
#print axioms g529_field_plane_needs_5_colours

#eval IO.println "SounioDeGreyChi5Concrete: geometry leg DISCHARGED over the exact field-plane QF×QF; only the SAT leg remains hypothetical; R-embedding (QF->R) staged."

end DeGrey529.Concrete
