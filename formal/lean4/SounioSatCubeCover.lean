/-
# Sounio — finite cube-cover composition for graph-colouring SAT

This module is the Lean-side composition rung for cube-and-conquer SAT
artifacts. The generic theorem `unsat_of_cube_cover` says: if a finite cube list
covers every satisfying assignment of the base colouring CNF, and every
cube-augmented leaf is UNSAT, then the base CNF is UNSAT.

The first calibration route remains deliberately small: the five one-literal
cubes obtained by splitting a vertex over colours `0..4` form a cover by the
at-least-one colour clause already present in `colourCNF`; no at-most-one
colour constraints are needed. This module does not parse LRAT files. Generated
modules should supply the cube-augmented `Unsat` leaves from Lean-checked LRAT
artifacts plus, for generic searches, the cover proof.
-/
import SounioSatColouringBridge

open Std.Sat
open SounioSatColouring

namespace SounioSatCubeCover

/-- The positive colour-literal clause for vertex `v` in the `k = 5` encoding. -/
def vertexColourClause5 (v : Nat) : CNF.Clause Nat :=
  [(v * 5 + 0, true), (v * 5 + 1, true), (v * 5 + 2, true),
   (v * 5 + 3, true), (v * 5 + 4, true)]

private theorem vertexColourClause5_eq_range (v : Nat) :
    vertexColourClause5 v = (List.range 5).map (fun c => (v * 5 + c, true)) := by
  rfl

/-- The graph-colouring CNF augmented with one cube unit `v ↦ c`.
Clause order matches `souc_sat`: base colourCNF clauses, then cube units. -/
def colourCNFWithUnit (n k : Nat) (edges : List (Nat × Nat)) (v c : Nat) : CNF Nat :=
  { clauses := (atLeastOne n k ++ edgeClauses k edges ++ [[(v * k + c, true)]]).toArray }

/-! ## Generic cube-cover adapter

The single-vertex split theorem below is useful for the first K6 smoke, but real
cube-and-conquer needs a proof object for an arbitrary finite list of cubes. The
search producer remains untrusted: it must provide both Lean-checked UNSAT leaves
and a Lean proof of `CubeCover`.
-/

/-- A SAT cube is a list of literals that are all asserted as unit clauses. -/
abbrev Cube := List (Nat × Bool)

/-- Encode every cube literal as a singleton CNF clause. -/
def cubeClauses (cube : Cube) : List (CNF.Clause Nat) :=
  cube.map (fun lit => [lit])

/-- The CNF containing only the unit clauses of a cube. -/
def cubeCNF (cube : Cube) : CNF Nat :=
  { clauses := (cubeClauses cube).toArray }

/-- Flip a SAT literal. -/
def negLit (lit : Nat × Bool) : Nat × Bool :=
  (lit.1, !lit.2)

/-- A clause blocking one cube: at least one asserted cube literal must fail. -/
def cubeBlockingClause (cube : Cube) : CNF.Clause Nat :=
  cube.map negLit

/-- The complement-cover CNF for arbitrary cubes.

It is satisfiable exactly when the base colouring CNF has a satisfying
assignment that satisfies none of the cubes. A Lean-checked UNSAT proof of this
CNF is therefore a generic finite cover certificate, independent of how the
cubes were produced. -/
def cubeCoverComplementCNF
    (n k : Nat) (edges : List (Nat × Nat)) (cubes : List Cube) : CNF Nat :=
  { clauses := (atLeastOne n k ++ edgeClauses k edges ++
      cubes.map cubeBlockingClause).toArray }

/-- The graph-colouring CNF augmented with an arbitrary cube. -/
def colourCNFWithCube (n k : Nat) (edges : List (Nat × Nat)) (cube : Cube) : CNF Nat :=
  { clauses := (atLeastOne n k ++ edgeClauses k edges ++ cubeClauses cube).toArray }

/-- A list of cubes covers all satisfying assignments of the base colouring CNF. -/
def CubeCover (n k : Nat) (edges : List (Nat × Nat)) (cubes : List Cube) : Prop :=
  ∀ a : Nat → Bool,
    CNF.eval a (colourCNF n k edges) = true →
      ∃ cube, cube ∈ cubes ∧ CNF.eval a (cubeCNF cube) = true

/-- The five one-literal cubes induced by splitting vertex `v` over the colours
`0..4`. This is the current K6 calibration route as data for the generic cover
adapter. -/
def splitVertex5Cubes (v : Nat) : List Cube :=
  [[(v * 5 + 0, true)], [(v * 5 + 1, true)], [(v * 5 + 2, true)],
   [(v * 5 + 3, true)], [(v * 5 + 4, true)]]

/-- Product cover obtained by splitting every vertex in `vs` over all `k`
colours. Each cube contains one positive colour literal for each listed vertex.
The cubes may overlap semantically, because `colourCNF` has no at-most-one
clauses; the cover theorem below only needs that every base satisfying Boolean
assignment satisfies at least one cube. -/
def splitVerticesCubes (k : Nat) : List Nat → List Cube
  | [] => [[]]
  | v :: vs =>
      (List.range k).flatMap (fun c =>
        (splitVerticesCubes k vs).map (fun cube => (v * k + c, true) :: cube))

/-- A unit assignment is the one-literal special case of a generic cube. -/
theorem colourCNFWithUnit_eq_cube (n k : Nat) (edges : List (Nat × Nat)) (v c : Nat) :
    colourCNFWithUnit n k edges v c =
      colourCNFWithCube n k edges [(v * k + c, true)] := by
  rfl

private theorem vertexColourClause5_mem_atLeastOne {n v : Nat} (hv : v < n) :
    vertexColourClause5 v ∈ atLeastOne n 5 := by
  have hvRange : v ∈ List.range n := List.mem_range.mpr hv
  have hmem :
      (List.range 5).map (fun c => (v * 5 + c, true)) ∈ atLeastOne n 5 := by
    unfold atLeastOne
    exact List.mem_map.mpr ⟨v, hvRange, rfl⟩
  simpa [vertexColourClause5_eq_range] using hmem

private theorem vertexColourClause_mem_atLeastOne {n k v : Nat} (hv : v < n) :
    (List.range k).map (fun c => (v * k + c, true)) ∈ atLeastOne n k := by
  unfold atLeastOne
  exact List.mem_map.mpr ⟨v, List.mem_range.mpr hv, rfl⟩

private theorem true_lit_of_vertex_clause_true
    {a : Nat → Bool} {k v : Nat}
    (hcl : CNF.Clause.eval a ((List.range k).map (fun c => (v * k + c, true))) = true) :
    ∃ c, c < k ∧ a (v * k + c) = true := by
  simp only [CNF.Clause.eval, List.any_map, List.any_eq_true, List.mem_range, Function.comp] at hcl
  rcases hcl with ⟨c, hc, hbeq⟩
  have htrue : a (v * k + c) = true := by
    simpa only [beq_true] using hbeq
  exact ⟨c, hc, htrue⟩

private theorem cubeCNF_singleton_pos_true
    {a : Nat → Bool} {x : Nat}
    (hx : a x = true) :
    CNF.eval a (cubeCNF [(x, true)]) = true := by
  simp [cubeCNF, cubeClauses, CNF.eval, CNF.Clause.eval, hx]

private theorem cubeCNF_nil_true {a : Nat → Bool} :
    CNF.eval a (cubeCNF []) = true := by
  simp [cubeCNF, cubeClauses, CNF.eval]

private theorem cubeCNF_cons_pos_true
    {a : Nat → Bool} {x : Nat} {cube : Cube}
    (hx : a x = true)
    (hcube : CNF.eval a (cubeCNF cube) = true) :
    CNF.eval a (cubeCNF ((x, true) :: cube)) = true := by
  have hallCube : ∀ cl ∈ (cubeCNF cube).clauses, CNF.Clause.eval a cl = true := by
    simpa [CNF.eval] using (Array.all_eq_true_iff_forall_mem.mp hcube)
  simp only [cubeCNF, cubeClauses, CNF.eval]
  rw [Array.all_eq_true_iff_forall_mem]
  intro cl hcl
  rw [List.mem_toArray] at hcl
  simp only [List.map_cons, List.mem_cons] at hcl
  rcases hcl with rfl | hcl
  · simp [CNF.Clause.eval, hx]
  · exact hallCube cl (by simpa [cubeCNF, cubeClauses, List.mem_toArray] using hcl)

private theorem cubeBlockingClause_true_of_cubeCNF_false
    {a : Nat → Bool} {cube : Cube}
    (hcube : CNF.eval a (cubeCNF cube) = false) :
    CNF.Clause.eval a (cubeBlockingClause cube) = true := by
  induction cube with
  | nil =>
      simp [cubeCNF, cubeClauses, CNF.eval] at hcube
  | cons lit cube ih =>
      rcases lit with ⟨x, b⟩
      cases hx : a x <;> cases b <;>
        simp_all [cubeCNF, cubeClauses, cubeBlockingClause, negLit,
          CNF.eval, CNF.Clause.eval]

/-- Any assignment satisfying a cube-augmented CNF also satisfies the base CNF. -/
theorem eval_base_of_cube_aug
    {n k : Nat} {edges : List (Nat × Nat)} {cube : Cube} {a : Nat → Bool}
    (haug : CNF.eval a (colourCNFWithCube n k edges cube) = true) :
    CNF.eval a (colourCNF n k edges) = true := by
  have hallAug :
      ∀ cl ∈ (colourCNFWithCube n k edges cube).clauses, CNF.Clause.eval a cl = true := by
    simpa [CNF.eval] using (Array.all_eq_true_iff_forall_mem.mp haug)
  simp only [colourCNF, CNF.eval]
  rw [Array.all_eq_true_iff_forall_mem]
  intro cl hcl
  have hbaseList : cl ∈ atLeastOne n k ++ edgeClauses k edges := by
    simpa [List.mem_toArray] using hcl
  have haugList :
      cl ∈ atLeastOne n k ++ edgeClauses k edges ++ cubeClauses cube :=
    List.mem_append_left _ hbaseList
  exact hallAug cl (by
    simpa [colourCNFWithCube, List.mem_toArray] using haugList)

/-- Base UNSAT remains UNSAT after adding an arbitrary cube. -/
theorem unsat_with_cube_of_unsat
    {n k : Nat} {edges : List (Nat × Nat)} {cube : Cube}
    (hbase : (colourCNF n k edges).Unsat) :
    (colourCNFWithCube n k edges cube).Unsat := by
  intro a
  by_cases haug : CNF.eval a (colourCNFWithCube n k edges cube) = true
  · have hbaseTrue := eval_base_of_cube_aug (n := n) (k := k)
      (edges := edges) (cube := cube) (a := a) haug
    have hfalse := hbase a
    rw [hbaseTrue] at hfalse
    exact False.elim (Bool.noConfusion hfalse)
  · exact Bool.of_not_eq_true haug

private theorem eval_cube_aug_of_base_and_cube
    {n k : Nat} {edges : List (Nat × Nat)} {cube : Cube} {a : Nat → Bool}
    (hbase : CNF.eval a (colourCNF n k edges) = true)
    (hcube : CNF.eval a (cubeCNF cube) = true) :
    CNF.eval a (colourCNFWithCube n k edges cube) = true := by
  have hbaseClauses :
      ∀ cl ∈ atLeastOne n k ++ edgeClauses k edges, CNF.Clause.eval a cl = true := by
    have hall :
        ∀ cl ∈ (colourCNF n k edges).clauses, CNF.Clause.eval a cl = true := by
      simpa [CNF.eval] using (Array.all_eq_true_iff_forall_mem.mp hbase)
    intro cl hcl
    exact hall cl (by simpa [colourCNF, List.mem_toArray] using hcl)
  have hcubeClauses :
      ∀ cl ∈ cubeClauses cube, CNF.Clause.eval a cl = true := by
    have hall :
        ∀ cl ∈ (cubeCNF cube).clauses, CNF.Clause.eval a cl = true := by
      simpa [CNF.eval] using (Array.all_eq_true_iff_forall_mem.mp hcube)
    intro cl hcl
    exact hall cl (by simpa [cubeCNF, List.mem_toArray] using hcl)
  simp only [colourCNFWithCube, CNF.eval]
  rw [Array.all_eq_true_iff_forall_mem]
  intro cl hcl
  rw [List.mem_toArray, List.mem_append] at hcl
  rcases hcl with hcl | hcl
  · exact hbaseClauses cl hcl
  · exact hcubeClauses cl hcl

/-- Generic cube-cover composition: if every base satisfying assignment is covered
by some cube and every cube-augmented CNF is UNSAT, then the base CNF is UNSAT.
This is the Lean-side contract needed for multi-cube cube-and-conquer. -/
theorem unsat_of_cube_cover
    {n k : Nat} {edges : List (Nat × Nat)} {cubes : List Cube}
    (hcover : CubeCover n k edges cubes)
    (hunsat : ∀ cube, cube ∈ cubes → (colourCNFWithCube n k edges cube).Unsat) :
    (colourCNF n k edges).Unsat := by
  intro a
  by_cases hbase : CNF.eval a (colourCNF n k edges) = true
  · obtain ⟨cube, hmem, hcube⟩ := hcover a hbase
    have haug := eval_cube_aug_of_base_and_cube (n := n) (k := k)
      (edges := edges) (cube := cube) (a := a) hbase hcube
    have hfalse := hunsat cube hmem a
    rw [haug] at hfalse
    exact False.elim (Bool.noConfusion hfalse)
  · exact Bool.of_not_eq_true hbase

/-- Turn a checked UNSAT proof of the complement-cover CNF into a generic
`CubeCover` proof. This is the promotion path for arbitrary cube-and-conquer
partitions: the producer may emit any finite cube family, but Lean only trusts a
refutation of `base ∧ ⋀ cube, block(cube)`. -/
theorem cube_cover_of_complement_unsat
    {n k : Nat} {edges : List (Nat × Nat)} {cubes : List Cube}
    (hcomp : (cubeCoverComplementCNF n k edges cubes).Unsat) :
    CubeCover n k edges cubes := by
  intro a hbase
  by_cases hcov : ∃ cube, cube ∈ cubes ∧ CNF.eval a (cubeCNF cube) = true
  · exact hcov
  have hbaseClauses :
      ∀ cl ∈ atLeastOne n k ++ edgeClauses k edges, CNF.Clause.eval a cl = true := by
    have hall :
        ∀ cl ∈ (colourCNF n k edges).clauses, CNF.Clause.eval a cl = true := by
      simpa [CNF.eval] using (Array.all_eq_true_iff_forall_mem.mp hbase)
    intro cl hcl
    exact hall cl (by simpa [colourCNF, List.mem_toArray] using hcl)
  have hblocking :
      ∀ cl ∈ cubes.map cubeBlockingClause, CNF.Clause.eval a cl = true := by
    intro cl hcl
    rcases List.mem_map.mp hcl with ⟨cube, hmem, rfl⟩
    have hcubeFalse : CNF.eval a (cubeCNF cube) = false := by
      cases hcube : CNF.eval a (cubeCNF cube) with
      | false => rfl
      | true =>
          exact False.elim (hcov ⟨cube, hmem, hcube⟩)
    exact cubeBlockingClause_true_of_cubeCNF_false hcubeFalse
  have hcompSat : CNF.eval a (cubeCoverComplementCNF n k edges cubes) = true := by
    simp only [cubeCoverComplementCNF, CNF.eval]
    rw [Array.all_eq_true_iff_forall_mem]
    intro cl hcl
    rw [List.mem_toArray, List.mem_append] at hcl
    rcases hcl with hcl | hcl
    · exact hbaseClauses cl hcl
    · exact hblocking cl hcl
  have hfalse := hcomp a
  rw [hcompSat] at hfalse
  exact False.elim (Bool.noConfusion hfalse)

/-- The five split-vertex cubes cover every satisfying assignment of the base
`k = 5` colouring CNF, by the existing at-least-one colour clause for `v`. -/
theorem split_vertex5_cubes_cover
    {n : Nat} {edges : List (Nat × Nat)} {v : Nat} (hv : v < n) :
    CubeCover n 5 edges (splitVertex5Cubes v) := by
  intro a hbase
  have hall :
      ∀ cl ∈ (colourCNF n 5 edges).clauses, CNF.Clause.eval a cl = true := by
    simpa [CNF.eval] using (Array.all_eq_true_iff_forall_mem.mp hbase)
  have hvertex : CNF.Clause.eval a (vertexColourClause5 v) = true := by
    exact hall _ (by
      have hmem := vertexColourClause5_mem_atLeastOne (n := n) (v := v) hv
      simp [colourCNF, List.mem_toArray, hmem])
  by_cases h0lit : a (v * 5 + 0) = true
  · exact ⟨[(v * 5 + 0, true)], by simp [splitVertex5Cubes],
      cubeCNF_singleton_pos_true h0lit⟩
  · have h0f : a (v * 5 + 0) = false := by cases h : a (v * 5 + 0) <;> simp_all
    by_cases h1lit : a (v * 5 + 1) = true
    · exact ⟨[(v * 5 + 1, true)], by simp [splitVertex5Cubes],
        cubeCNF_singleton_pos_true h1lit⟩
    · have h1f : a (v * 5 + 1) = false := by cases h : a (v * 5 + 1) <;> simp_all
      by_cases h2lit : a (v * 5 + 2) = true
      · exact ⟨[(v * 5 + 2, true)], by simp [splitVertex5Cubes],
          cubeCNF_singleton_pos_true h2lit⟩
      · have h2f : a (v * 5 + 2) = false := by cases h : a (v * 5 + 2) <;> simp_all
        by_cases h3lit : a (v * 5 + 3) = true
        · exact ⟨[(v * 5 + 3, true)], by simp [splitVertex5Cubes],
            cubeCNF_singleton_pos_true h3lit⟩
        · have h3f : a (v * 5 + 3) = false := by cases h : a (v * 5 + 3) <;> simp_all
          have h0f0 : a (v * 5) = false := by simpa using h0f
          have h4lit : a (v * 5 + 4) = true := by
            simpa [vertexColourClause5, CNF.Clause.eval, h0f0, h1f, h2f, h3f] using hvertex
          exact ⟨[(v * 5 + 4, true)], by simp [splitVertex5Cubes],
            cubeCNF_singleton_pos_true h4lit⟩

/-- Splitting any finite list of vertices over all colours gives a genuine
`CubeCover`: each satisfying Boolean assignment of `colourCNF` has at least one
true colour literal at every listed vertex, so choosing one such colour per
listed vertex yields a satisfied cube. -/
theorem split_vertices_cubes_cover
    {n k : Nat} {edges : List (Nat × Nat)} {vs : List Nat}
    (hverts : ∀ v, v ∈ vs → v < n) :
    CubeCover n k edges (splitVerticesCubes k vs) := by
  induction vs with
  | nil =>
      intro a _hbase
      exact ⟨[], by simp [splitVerticesCubes], cubeCNF_nil_true⟩
  | cons v vs ih =>
      intro a hbase
      have hv : v < n := hverts v (by simp)
      have hvertsTail : ∀ w, w ∈ vs → w < n := by
        intro w hw
        exact hverts w (by simp [hw])
      have hall :
          ∀ cl ∈ (colourCNF n k edges).clauses, CNF.Clause.eval a cl = true := by
        simpa [CNF.eval] using (Array.all_eq_true_iff_forall_mem.mp hbase)
      have hvertex :
          CNF.Clause.eval a ((List.range k).map (fun c => (v * k + c, true))) = true := by
        exact hall _ (by
          have hmem := vertexColourClause_mem_atLeastOne (n := n) (k := k) (v := v) hv
          simp [colourCNF, List.mem_toArray, hmem])
      obtain ⟨c, hc, hlit⟩ :=
        true_lit_of_vertex_clause_true (a := a) (k := k) (v := v) hvertex
      obtain ⟨tailCube, htailMem, htailEval⟩ := ih hvertsTail a hbase
      refine ⟨(v * k + c, true) :: tailCube, ?_, ?_⟩
      · simp only [splitVerticesCubes, List.mem_flatMap, List.mem_map, List.mem_range]
        exact ⟨c, hc, tailCube, htailMem, rfl⟩
      · exact cubeCNF_cons_pos_true hlit htailEval

/-- The old single-vertex split route, expressed through the generic cube-cover
theorem. This is useful for generated certificates that already traffic in cube
lists rather than five named hypotheses. -/
theorem unsat_of_split_vertex5_cube_cover
    {n : Nat} {edges : List (Nat × Nat)} {v : Nat} (hv : v < n)
    (hunsat : ∀ cube, cube ∈ splitVertex5Cubes v →
      (colourCNFWithCube n 5 edges cube).Unsat) :
    (colourCNF n 5 edges).Unsat :=
  unsat_of_cube_cover
    (n := n) (k := 5) (edges := edges) (cubes := splitVertex5Cubes v)
    (split_vertex5_cubes_cover (n := n) (edges := edges) (v := v) hv)
    hunsat

/-- Any assignment satisfying the unit-augmented CNF also satisfies the base colouring CNF. -/
theorem eval_base_of_unit_aug
    {n k : Nat} {edges : List (Nat × Nat)} {v c : Nat} {a : Nat → Bool}
    (haug : CNF.eval a (colourCNFWithUnit n k edges v c) = true) :
    CNF.eval a (colourCNF n k edges) = true := by
  have hallAug :
      ∀ cl ∈ (colourCNFWithUnit n k edges v c).clauses, CNF.Clause.eval a cl = true := by
    simpa [CNF.eval] using (Array.all_eq_true_iff_forall_mem.mp haug)
  simp only [colourCNF, CNF.eval]
  rw [Array.all_eq_true_iff_forall_mem]
  intro cl hcl
  have hbaseList : cl ∈ atLeastOne n k ++ edgeClauses k edges := by
    simpa [List.mem_toArray] using hcl
  have haugList :
      cl ∈ atLeastOne n k ++ edgeClauses k edges ++ [[(v * k + c, true)]] :=
    List.mem_append_left _ hbaseList
  exact hallAug cl (by
    simpa [colourCNFWithUnit, List.mem_toArray] using haugList)

/-- Base UNSAT remains UNSAT after adding one positive cube unit. -/
theorem unsat_with_unit_of_unsat
    {n k : Nat} {edges : List (Nat × Nat)} {v c : Nat}
    (hbase : (colourCNF n k edges).Unsat) :
    (colourCNFWithUnit n k edges v c).Unsat := by
  intro a
  by_cases haug : CNF.eval a (colourCNFWithUnit n k edges v c) = true
  · have hbaseTrue := eval_base_of_unit_aug (n := n) (k := k)
      (edges := edges) (v := v) (c := c) (a := a) haug
    have hfalse := hbase a
    rw [hbaseTrue] at hfalse
    exact False.elim (Bool.noConfusion hfalse)
  · exact Bool.of_not_eq_true haug

private theorem eval_unit_aug_of_base_and_lit
    {n k : Nat} {edges : List (Nat × Nat)} {v c : Nat} {a : Nat → Bool}
    (hbase : CNF.eval a (colourCNF n k edges) = true)
    (hunit : a (v * k + c) = true) :
    CNF.eval a (colourCNFWithUnit n k edges v c) = true := by
  have hbaseClauses :
      ∀ cl ∈ atLeastOne n k ++ edgeClauses k edges, CNF.Clause.eval a cl = true := by
    have hall :
        ∀ cl ∈ (colourCNF n k edges).clauses, CNF.Clause.eval a cl = true := by
      simpa [CNF.eval] using (Array.all_eq_true_iff_forall_mem.mp hbase)
    intro cl hcl
    exact hall cl (by simpa [colourCNF, List.mem_toArray] using hcl)
  simp only [colourCNFWithUnit, CNF.eval]
  rw [Array.all_eq_true_iff_forall_mem]
  intro cl hcl
  rw [List.mem_toArray, List.mem_append] at hcl
  rcases hcl with hcl | hcl
  · exact hbaseClauses cl hcl
  · simp only [List.mem_singleton] at hcl
    subst cl
    simp [CNF.Clause.eval, hunit]

/-- If all five one-literal split leaves for vertex `v` are UNSAT, then the base
`k = 5` colouring CNF is UNSAT. This is the finite SAT cover step; downstream
graph/geometry meaning still comes from the ordinary `colourCNF` bridge. -/
theorem unsat_of_split_vertex5
    {n : Nat} {edges : List (Nat × Nat)} {v : Nat} (hv : v < n)
    (h0 : (colourCNFWithUnit n 5 edges v 0).Unsat)
    (h1 : (colourCNFWithUnit n 5 edges v 1).Unsat)
    (h2 : (colourCNFWithUnit n 5 edges v 2).Unsat)
    (h3 : (colourCNFWithUnit n 5 edges v 3).Unsat)
    (h4 : (colourCNFWithUnit n 5 edges v 4).Unsat) :
    (colourCNF n 5 edges).Unsat := by
  intro a
  by_cases hbase : CNF.eval a (colourCNF n 5 edges) = true
  · have hall :
        ∀ cl ∈ (colourCNF n 5 edges).clauses, CNF.Clause.eval a cl = true := by
      simpa [CNF.eval] using (Array.all_eq_true_iff_forall_mem.mp hbase)
    have hvertex : CNF.Clause.eval a (vertexColourClause5 v) = true := by
      exact hall _ (by
        have hmem := vertexColourClause5_mem_atLeastOne (n := n) (v := v) hv
        simp [colourCNF, List.mem_toArray, hmem])
    by_cases h0lit : a (v * 5 + 0) = true
    · have haug := eval_unit_aug_of_base_and_lit (n := n) (k := 5)
        (edges := edges) (v := v) (c := 0) hbase h0lit
      have hfalse := h0 a
      rw [haug] at hfalse
      exact False.elim (Bool.noConfusion hfalse)
    · have h0f : a (v * 5 + 0) = false := by cases h : a (v * 5 + 0) <;> simp_all
      by_cases h1lit : a (v * 5 + 1) = true
      · have haug := eval_unit_aug_of_base_and_lit (n := n) (k := 5)
          (edges := edges) (v := v) (c := 1) hbase h1lit
        have hfalse := h1 a
        rw [haug] at hfalse
        exact False.elim (Bool.noConfusion hfalse)
      · have h1f : a (v * 5 + 1) = false := by cases h : a (v * 5 + 1) <;> simp_all
        by_cases h2lit : a (v * 5 + 2) = true
        · have haug := eval_unit_aug_of_base_and_lit (n := n) (k := 5)
            (edges := edges) (v := v) (c := 2) hbase h2lit
          have hfalse := h2 a
          rw [haug] at hfalse
          exact False.elim (Bool.noConfusion hfalse)
        · have h2f : a (v * 5 + 2) = false := by cases h : a (v * 5 + 2) <;> simp_all
          by_cases h3lit : a (v * 5 + 3) = true
          · have haug := eval_unit_aug_of_base_and_lit (n := n) (k := 5)
              (edges := edges) (v := v) (c := 3) hbase h3lit
            have hfalse := h3 a
            rw [haug] at hfalse
            exact False.elim (Bool.noConfusion hfalse)
          · have h3f : a (v * 5 + 3) = false := by cases h : a (v * 5 + 3) <;> simp_all
            have h0f0 : a (v * 5) = false := by simpa using h0f
            have h4lit : a (v * 5 + 4) = true := by
              simpa [vertexColourClause5, CNF.Clause.eval, h0f0, h1f, h2f, h3f] using hvertex
            have haug := eval_unit_aug_of_base_and_lit (n := n) (k := 5)
              (edges := edges) (v := v) (c := 4) hbase h4lit
            have hfalse := h4 a
            rw [haug] at hfalse
            exact False.elim (Bool.noConfusion hfalse)
  · exact Bool.of_not_eq_true hbase

#print axioms unsat_of_split_vertex5
#print axioms eval_base_of_unit_aug
#print axioms unsat_with_unit_of_unsat
#print axioms unsat_of_cube_cover
#print axioms eval_base_of_cube_aug
#print axioms unsat_with_cube_of_unsat
#print axioms split_vertex5_cubes_cover
#print axioms split_vertices_cubes_cover
#print axioms unsat_of_split_vertex5_cube_cover
#print axioms cube_cover_of_complement_unsat

end SounioSatCubeCover
