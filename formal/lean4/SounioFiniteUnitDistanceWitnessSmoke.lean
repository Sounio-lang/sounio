import Init.Omega
import SounioFiniteUnitDistanceWitness
import SounioSatK65Reflect

/-!
# SounioFiniteUnitDistanceWitnessSmoke — tiny `k = 5` contract smoke

This file is deliberately **not** a Euclidean unit-distance claim. It is a small
finite complete-graph smoke test for the `NoFiveColourWitness` interface: if a
future search produces exact geometry plus a verified no-5-colouring certificate,
the generic no-5 obstruction plumbing compiles end-to-end. Public chi>=6 promotion
is reserved for the Euclidean squared-distance API.
-/

namespace UnitDistanceChromatic.Smoke

open UnitDistanceChromatic
open Std.Sat

/-- The five positive colour literals used by the `k = 5` colouring CNF for one vertex. -/
def vertexColourClause5 (v : Nat) : CNF.Clause Nat :=
  (List.range 5).map (fun c => (v * 5 + c, true))

/-- Complete graph K₆ as a Nat edge list. -/
def k6Edges : List (Nat × Nat) :=
  [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
   (1, 2), (1, 3), (1, 4), (1, 5),
   (2, 3), (2, 4), (2, 5),
   (3, 4), (3, 5),
   (4, 5)]

/-- The finite complete relation on the six distinguished Nat points.
This is a smoke-test relation, not the Euclidean unit-distance relation. -/
def k6Unit (a b : Nat) : Prop :=
  a < 6 ∧ b < 6 ∧ a ≠ b

theorem k6_edges_endpoints_lt :
    ∀ e ∈ k6Edges, e.1 < 6 ∧ e.2 < 6 := by
  native_decide

theorem k6_edges_unit :
    ∀ e ∈ k6Edges, k6Unit e.1 e.2 := by
  intro e he
  rcases (by simpa [k6Edges] using he) with h | h | h | h | h | h | h | h | h | h | h | h | h | h | h
  all_goals cases h
  all_goals simp [k6Unit]

/-- K₆ has no proper 5-colouring. This is a finite smoke fact, discharged by exhaustive
computation, used only to exercise the witness interface without an external LRAT file. -/
theorem k6_not_5_colourable :
    ¬ ∃ c : Fin 6 → Fin 5,
        ∀ e ∈ k6Edges, ∀ (h1 : e.1 < 6) (h2 : e.2 < 6),
          c ⟨e.1, h1⟩ ≠ c ⟨e.2, h2⟩ := by
  rintro ⟨c, hc⟩
  let v0 : Fin 6 := ⟨0, by decide⟩
  let v1 : Fin 6 := ⟨1, by decide⟩
  let v2 : Fin 6 := ⟨2, by decide⟩
  let v3 : Fin 6 := ⟨3, by decide⟩
  let v4 : Fin 6 := ⟨4, by decide⟩
  let v5 : Fin 6 := ⟨5, by decide⟩
  have e01 : c v0 ≠ c v1 := by simpa [v0, v1] using hc (0, 1) (by decide) (by decide) (by decide)
  have e02 : c v0 ≠ c v2 := by simpa [v0, v2] using hc (0, 2) (by decide) (by decide) (by decide)
  have e03 : c v0 ≠ c v3 := by simpa [v0, v3] using hc (0, 3) (by decide) (by decide) (by decide)
  have e04 : c v0 ≠ c v4 := by simpa [v0, v4] using hc (0, 4) (by decide) (by decide) (by decide)
  have e05 : c v0 ≠ c v5 := by simpa [v0, v5] using hc (0, 5) (by decide) (by decide) (by decide)
  have e12 : c v1 ≠ c v2 := by simpa [v1, v2] using hc (1, 2) (by decide) (by decide) (by decide)
  have e13 : c v1 ≠ c v3 := by simpa [v1, v3] using hc (1, 3) (by decide) (by decide) (by decide)
  have e14 : c v1 ≠ c v4 := by simpa [v1, v4] using hc (1, 4) (by decide) (by decide) (by decide)
  have e15 : c v1 ≠ c v5 := by simpa [v1, v5] using hc (1, 5) (by decide) (by decide) (by decide)
  have e23 : c v2 ≠ c v3 := by simpa [v2, v3] using hc (2, 3) (by decide) (by decide) (by decide)
  have e24 : c v2 ≠ c v4 := by simpa [v2, v4] using hc (2, 4) (by decide) (by decide) (by decide)
  have e25 : c v2 ≠ c v5 := by simpa [v2, v5] using hc (2, 5) (by decide) (by decide) (by decide)
  have e34 : c v3 ≠ c v4 := by simpa [v3, v4] using hc (3, 4) (by decide) (by decide) (by decide)
  have e35 : c v3 ≠ c v5 := by simpa [v3, v5] using hc (3, 5) (by decide) (by decide) (by decide)
  have e45 : c v4 ≠ c v5 := by simpa [v4, v5] using hc (4, 5) (by decide) (by decide) (by decide)
  have b0 := (c v0).isLt
  have b1 := (c v1).isLt
  have b2 := (c v2).isLt
  have b3 := (c v3).isLt
  have b4 := (c v4).isLt
  have b5 := (c v5).isLt
  have v01 : (c v0).val ≠ (c v1).val := fun h => e01 (Fin.eq_of_val_eq h)
  have v02 : (c v0).val ≠ (c v2).val := fun h => e02 (Fin.eq_of_val_eq h)
  have v03 : (c v0).val ≠ (c v3).val := fun h => e03 (Fin.eq_of_val_eq h)
  have v04 : (c v0).val ≠ (c v4).val := fun h => e04 (Fin.eq_of_val_eq h)
  have v05 : (c v0).val ≠ (c v5).val := fun h => e05 (Fin.eq_of_val_eq h)
  have v12 : (c v1).val ≠ (c v2).val := fun h => e12 (Fin.eq_of_val_eq h)
  have v13 : (c v1).val ≠ (c v3).val := fun h => e13 (Fin.eq_of_val_eq h)
  have v14 : (c v1).val ≠ (c v4).val := fun h => e14 (Fin.eq_of_val_eq h)
  have v15 : (c v1).val ≠ (c v5).val := fun h => e15 (Fin.eq_of_val_eq h)
  have v23 : (c v2).val ≠ (c v3).val := fun h => e23 (Fin.eq_of_val_eq h)
  have v24 : (c v2).val ≠ (c v4).val := fun h => e24 (Fin.eq_of_val_eq h)
  have v25 : (c v2).val ≠ (c v5).val := fun h => e25 (Fin.eq_of_val_eq h)
  have v34 : (c v3).val ≠ (c v4).val := fun h => e34 (Fin.eq_of_val_eq h)
  have v35 : (c v3).val ≠ (c v5).val := fun h => e35 (Fin.eq_of_val_eq h)
  have v45 : (c v4).val ≠ (c v5).val := fun h => e45 (Fin.eq_of_val_eq h)
  omega

/-- Deterministically choose one of five true colour variables for vertex `v`.
If no colour variable is true, this returns `4`; the accompanying lemma is used only
when the CNF's at-least-one clause is known true. -/
def choose5FromAssignment (a : Nat → Bool) (v : Nat) : Fin 5 :=
  if a (v * 5) then 0
  else if a (v * 5 + 1) then 1
  else if a (v * 5 + 2) then 2
  else if a (v * 5 + 3) then 3
  else 4

theorem choose5FromAssignment_true {a : Nat → Bool} {v : Nat}
    (hcl : CNF.Clause.eval a (vertexColourClause5 v) = true) :
    a (v * 5 + (choose5FromAssignment a v).val) = true := by
  have hrange5 : List.range 5 = [0, 1, 2, 3, 4] := by native_decide
  have hcl' : CNF.Clause.eval a
      [(v * 5, true), (v * 5 + 1, true), (v * 5 + 2, true),
       (v * 5 + 3, true), (v * 5 + 4, true)] = true := by
    simpa [vertexColourClause5, hrange5] using hcl
  by_cases h0 : a (v * 5) = true
  · have hchosen : choose5FromAssignment a v = 0 := by
      simp [choose5FromAssignment, h0]
    simpa [hchosen] using h0
  · have h0f : a (v * 5) = false := by cases h : a (v * 5) <;> simp_all
    by_cases h1 : a (v * 5 + 1) = true
    · have hchosen : choose5FromAssignment a v = 1 := by
        simp [choose5FromAssignment, h0f, h1]
      simpa [hchosen] using h1
    · have h1f : a (v * 5 + 1) = false := by cases h : a (v * 5 + 1) <;> simp_all
      by_cases h2 : a (v * 5 + 2) = true
      · have hchosen : choose5FromAssignment a v = 2 := by
          simp [choose5FromAssignment, h0f, h1f, h2]
        simpa [hchosen] using h2
      · have h2f : a (v * 5 + 2) = false := by cases h : a (v * 5 + 2) <;> simp_all
        by_cases h3 : a (v * 5 + 3) = true
        · have hchosen : choose5FromAssignment a v = 3 := by
            simp [choose5FromAssignment, h0f, h1f, h2f, h3]
          simpa [hchosen] using h3
        · have h3f : a (v * 5 + 3) = false := by cases h : a (v * 5 + 3) <;> simp_all
          have hchosen : choose5FromAssignment a v = 4 := by
            simp [choose5FromAssignment, h0f, h1f, h2f, h3f]
          simp [CNF.Clause.eval, h0f, h1f, h2f, h3f] at hcl'
          simpa [hchosen] using hcl'

/-- K₆/no-5 through the generic `colourCNF` bridge.
This is still a finite complete-relation smoke, but it exercises the same
plain colouring-CNF shape a generated LRAT certificate would inhabit. -/
theorem k6_colourCNF5_unsat :
    (SounioSatColouring.colourCNF 6 5 k6Edges).Unsat := by
  intro a
  by_cases hsat : CNF.eval a (SounioSatColouring.colourCNF 6 5 k6Edges) = true
  · exfalso
    have hall :
        ∀ cl ∈ (SounioSatColouring.colourCNF 6 5 k6Edges).clauses,
          CNF.Clause.eval a cl = true := by
      simpa [CNF.eval] using (Array.all_eq_true_iff_forall_mem.mp hsat)
    have vertexClauseTrue (v : Fin 6) :
        CNF.Clause.eval a (vertexColourClause5 v.val) = true := by
      have hmemALO :
          vertexColourClause5 v.val ∈ SounioSatColouring.atLeastOne 6 5 := by
        have hv : v.val ∈ List.range 6 := List.mem_range.mpr v.isLt
        unfold SounioSatColouring.atLeastOne vertexColourClause5
        exact List.mem_map_of_mem hv
      exact hall _ (by
        simp [SounioSatColouring.colourCNF, hmemALO])
    have edgeClauseTrue (e : Nat × Nat) (he : e ∈ k6Edges) (c : Fin 5) :
        CNF.Clause.eval a
          [(e.1 * 5 + c.val, false), (e.2 * 5 + c.val, false)] = true := by
      have hmemEdge :
          [(e.1 * 5 + c.val, false), (e.2 * 5 + c.val, false)] ∈
            SounioSatColouring.edgeClauses 5 k6Edges := by
        have hc : c.val ∈ List.range 5 := List.mem_range.mpr c.isLt
        have hcolor :
            [(e.1 * 5 + c.val, false), (e.2 * 5 + c.val, false)] ∈
              (List.range 5).map
                (fun c => [(e.1 * 5 + c, false), (e.2 * 5 + c, false)]) :=
          List.mem_map_of_mem hc
        unfold SounioSatColouring.edgeClauses
        exact List.mem_flatMap.mpr ⟨e, he, hcolor⟩
      exact hall _ (by
        simp [SounioSatColouring.colourCNF, hmemEdge])
    apply k6_not_5_colourable
    refine ⟨fun v => choose5FromAssignment a v.val, ?_⟩
    intro e he h1 h2 hsame
    have hleft :
        a (e.1 * 5 + (choose5FromAssignment a e.1).val) = true := by
      exact choose5FromAssignment_true (vertexClauseTrue ⟨e.1, h1⟩)
    have hright :
        a (e.2 * 5 + (choose5FromAssignment a e.2).val) = true := by
      exact choose5FromAssignment_true (vertexClauseTrue ⟨e.2, h2⟩)
    have hcval :
        (choose5FromAssignment a e.1).val = (choose5FromAssignment a e.2).val :=
      congrArg Fin.val hsame
    have hright' :
        a (e.2 * 5 + (choose5FromAssignment a e.1).val) = true := by
      simpa [hcval] using hright
    have hforbid := edgeClauseTrue e he (choose5FromAssignment a e.1)
    simp [CNF.Clause.eval, hleft, hright'] at hforbid
  · exact Bool.of_not_eq_true hsat

/-- The same K₆/no-5 smoke through the actual `k = 5` triangle-precolour CNF hook.
This remains a tiny finite complete-relation smoke; a real chi>=6 witness must replace this
finite hand proof with a generated DRAT/LRAT proof for a planar unit-distance graph. -/
theorem k6_colourCNFsb5_unsat :
    (SounioSatColouringSB.colourCNFsb5 0 1 2 6 k6Edges).Unsat := by
  intro a
  by_cases hsat : CNF.eval a (SounioSatColouringSB.colourCNFsb5 0 1 2 6 k6Edges) = true
  · exfalso
    have hall :
        ∀ cl ∈ (SounioSatColouringSB.colourCNFsb5 0 1 2 6 k6Edges).clauses,
          CNF.Clause.eval a cl = true := by
      simpa [CNF.eval] using (Array.all_eq_true_iff_forall_mem.mp hsat)
    have vertexClauseTrue (v : Fin 6) :
        CNF.Clause.eval a (vertexColourClause5 v.val) = true := by
      have hmemALO :
          vertexColourClause5 v.val ∈ SounioSatColouring.atLeastOne 6 5 := by
        have hv : v.val ∈ List.range 6 := List.mem_range.mpr v.isLt
        unfold SounioSatColouring.atLeastOne vertexColourClause5
        exact List.mem_map_of_mem hv
      exact hall _ (by
        simp [SounioSatColouringSB.colourCNFsb5, hmemALO])
    have edgeClauseTrue (e : Nat × Nat) (he : e ∈ k6Edges) (c : Fin 5) :
        CNF.Clause.eval a
          [(e.1 * 5 + c.val, false), (e.2 * 5 + c.val, false)] = true := by
      have hmemEdge :
          [(e.1 * 5 + c.val, false), (e.2 * 5 + c.val, false)] ∈
            SounioSatColouring.edgeClauses 5 k6Edges := by
        have hc : c.val ∈ List.range 5 := List.mem_range.mpr c.isLt
        have hcolor :
            [(e.1 * 5 + c.val, false), (e.2 * 5 + c.val, false)] ∈
              (List.range 5).map
                (fun c => [(e.1 * 5 + c, false), (e.2 * 5 + c, false)]) :=
          List.mem_map_of_mem hc
        unfold SounioSatColouring.edgeClauses
        exact List.mem_flatMap.mpr ⟨e, he, hcolor⟩
      exact hall _ (by
        simp [SounioSatColouringSB.colourCNFsb5, hmemEdge])
    apply k6_not_5_colourable
    refine ⟨fun v => choose5FromAssignment a v.val, ?_⟩
    intro e he h1 h2 hsame
    have hleft :
        a (e.1 * 5 + (choose5FromAssignment a e.1).val) = true := by
      exact choose5FromAssignment_true (vertexClauseTrue ⟨e.1, h1⟩)
    have hright :
        a (e.2 * 5 + (choose5FromAssignment a e.2).val) = true := by
      exact choose5FromAssignment_true (vertexClauseTrue ⟨e.2, h2⟩)
    have hcval :
        (choose5FromAssignment a e.1).val = (choose5FromAssignment a e.2).val :=
      congrArg Fin.val hsame
    have hright' :
        a (e.2 * 5 + (choose5FromAssignment a e.1).val) = true := by
      simpa [hcval] using hright
    have hforbid := edgeClauseTrue e he (choose5FromAssignment a e.1)
    simp [CNF.Clause.eval, hleft, hright'] at hforbid
  · exact Bool.of_not_eq_true hsat

/-- K₆ packaged as a `NoFiveColourWitness` smoke object. -/
def k6NoFiveWitness :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness 6 Nat k6Unit where
  edges := k6Edges
  emb := id
  endpoints := k6_edges_endpoints_lt
  unit_edges := k6_edges_unit
  not_colourable := k6_not_5_colourable

/-- K₆ packaged through the same `colourCNFsb5` constructor a future chi>=6 witness should use. -/
def k6NoFiveWitnessViaSB5 :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness 6 Nat k6Unit :=
  NatEdgeUnitDistanceCertificate.noFiveWitnessOfColourCNFsb5UnsatTri
    (n := 6) (P := Nat) (unit := k6Unit)
    k6Edges id 0 1 2
    (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide)
    k6_edges_endpoints_lt
    k6_edges_unit
    k6_colourCNFsb5_unsat

/-- K₆ packaged through the generic `colourCNF` constructor.
This names the non-SB LRAT target shape for future generated `k = 5` modules. -/
def k6NoFiveWitnessViaPlainCNF :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness 6 Nat k6Unit :=
  NatEdgeUnitDistanceCertificate.ofColourCNFUnsat
    (n := 6) (k := 5) (P := Nat) (unit := k6Unit)
    (by decide) k6Edges id
    k6_edges_endpoints_lt
    k6_edges_unit
    k6_colourCNF5_unsat

/-- The generated K₆/5 edge list is definitionally the same complete graph as `k6Edges`. -/
theorem k65_edges_eq_k6Edges : k65_edges = k6Edges := by
  native_decide

/-- K₆ packaged through a generated reflected LRAT certificate checked by Lean's LRAT checker.
This is still a finite complete-relation smoke, not a planar Euclidean witness. -/
def k6NoFiveWitnessViaReflectedLRAT :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness 6 Nat k6Unit :=
  NatEdgeUnitDistanceCertificate.ofColourCNFUnsat
    (n := 6) (k := 5) (P := Nat) (unit := k6Unit)
    (by decide) k65_edges id
    (by
      rw [k65_edges_eq_k6Edges]
      exact k6_edges_endpoints_lt)
    (by
      rw [k65_edges_eq_k6Edges]
      exact k6_edges_unit)
    k65_unsat

/-- K₆ packaged through the cube-cover constructor. The five leaf UNSAT facts here
are derived from the hand-proved base CNF UNSAT only to smoke-test the API; a real
candidate supplies these leaves from independent LRAT artifacts. -/
def k6NoFiveWitnessViaCubeCover :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness 6 Nat k6Unit :=
  NatEdgeUnitDistanceCertificate.noFiveWitnessOfSplitVertex5Unsat
    (n := 6) (P := Nat) (unit := k6Unit)
    k6Edges id 0
    (by decide)
    k6_edges_endpoints_lt
    k6_edges_unit
    (SounioSatCubeCover.unsat_with_unit_of_unsat
      (n := 6) (k := 5) (edges := k6Edges) (v := 0) (c := 0)
      k6_colourCNF5_unsat)
    (SounioSatCubeCover.unsat_with_unit_of_unsat
      (n := 6) (k := 5) (edges := k6Edges) (v := 0) (c := 1)
      k6_colourCNF5_unsat)
    (SounioSatCubeCover.unsat_with_unit_of_unsat
      (n := 6) (k := 5) (edges := k6Edges) (v := 0) (c := 2)
      k6_colourCNF5_unsat)
    (SounioSatCubeCover.unsat_with_unit_of_unsat
      (n := 6) (k := 5) (edges := k6Edges) (v := 0) (c := 3)
      k6_colourCNF5_unsat)
    (SounioSatCubeCover.unsat_with_unit_of_unsat
      (n := 6) (k := 5) (edges := k6Edges) (v := 0) (c := 4)
      k6_colourCNF5_unsat)

/-- End-to-end smoke for the generic no-5 obstruction plumbing.
Again: this is a finite complete-relation smoke, not a planar Euclidean theorem. -/
theorem k6_smoke_no_five_plane_colouring :
    ¬ Nonempty (PlaneColouring Nat k6Unit 5) :=
  NatEdgeUnitDistanceCertificate.generic_no_five_colour_obstruction k6NoFiveWitness

/-- End-to-end smoke through the triangle-precoloured `k = 5` SAT hook. -/
theorem k6_smoke_no_five_plane_colouring_via_sb5 :
    ¬ Nonempty (PlaneColouring Nat k6Unit 5) :=
  NatEdgeUnitDistanceCertificate.generic_no_five_colour_obstruction k6NoFiveWitnessViaSB5

/-- End-to-end smoke through the generic `k = 5` colouring-CNF hook. -/
theorem k6_smoke_no_five_plane_colouring_via_plain_cnf :
    ¬ Nonempty (PlaneColouring Nat k6Unit 5) :=
  NatEdgeUnitDistanceCertificate.generic_no_five_colour_obstruction k6NoFiveWitnessViaPlainCNF

/-- End-to-end smoke through the generated reflected LRAT certificate. -/
theorem k6_smoke_no_five_plane_colouring_via_reflected_lrat :
    ¬ Nonempty (PlaneColouring Nat k6Unit 5) :=
  NatEdgeUnitDistanceCertificate.generic_no_five_colour_obstruction k6NoFiveWitnessViaReflectedLRAT

/-- End-to-end smoke through the cube-cover witness hook. -/
theorem k6_smoke_no_five_plane_colouring_via_cube_cover :
    ¬ Nonempty (PlaneColouring Nat k6Unit 5) :=
  NatEdgeUnitDistanceCertificate.generic_no_five_colour_obstruction k6NoFiveWitnessViaCubeCover

#print axioms k6_not_5_colourable
#print axioms k6_colourCNF5_unsat
#print axioms k6_colourCNFsb5_unsat
#print axioms k6NoFiveWitness
#print axioms k6NoFiveWitnessViaSB5
#print axioms k6NoFiveWitnessViaPlainCNF
#print axioms k6NoFiveWitnessViaReflectedLRAT
#print axioms k6NoFiveWitnessViaCubeCover
#print axioms k6_smoke_no_five_plane_colouring
#print axioms k6_smoke_no_five_plane_colouring_via_sb5
#print axioms k6_smoke_no_five_plane_colouring_via_plain_cnf
#print axioms k6_smoke_no_five_plane_colouring_via_reflected_lrat
#print axioms k6_smoke_no_five_plane_colouring_via_cube_cover

#eval IO.println "SounioFiniteUnitDistanceWitnessSmoke: K6/no-5-colouring smoke for the chi>=6 witness interface, including plain colourCNF5, colourCNFsb5, generated reflected-LRAT, and cube-cover hooks; not a Euclidean unit-distance claim."

end UnitDistanceChromatic.Smoke
