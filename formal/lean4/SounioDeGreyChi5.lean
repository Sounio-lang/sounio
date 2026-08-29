/-!
# Sounio — de Grey χ(ℝ²) ≥ 5: the REDUCTION leg (SAT + geometry → plane)

This file proves the **logical reduction** only: if a finite graph `G` embeds in a
point set with every edge at unit relation, and `G` is not `k`-colourable, then no
proper `k`-colouring of that unit-relation plane graph exists (equivalently, its
chromatic number is **> k**, hence **≥ k + 1** when `k = 4`).

The two inputs are discharged **outside** this file:

| Leg | Fact | Where verified |
|-----|------|----------------|
| Geometry | Every edge of G_529 embeds at unit distance | `SounioDeGreyUnitDistance.lean`: `DeGrey529.g529_all_edges_unit_distance` |
| SAT | G_529 is **not** 4-colourable | cake_lpr LRAT check; see `examples/erdos/CAKE_LPR_RESULT.md` |

No `sorry`, no `native_decide` on the SAT fact. The reduction (steps 4–5) is pure logic.
-/

namespace UnitDistanceChromatic

/-!
## 1. Abstract finite graph on `V = Fin n`
-/

/-- Vertex type for an `n`-vertex graph. -/
abbrev V (n : Nat) := Fin n

/-- A vertex colouring with `k` colours. -/
def Coloring (n k : Nat) := V n → Fin k

/-- A list of undirected edges (ordered pairs of vertices). -/
structure Graph (n : Nat) where
  edges : List (Fin n × Fin n)

namespace Graph

variable {n k : Nat} (G : Graph n)

/-- No edge is monochromatic under `c`. -/
def IsProper (c : Coloring n k) : Prop :=
  ∀ e ∈ G.edges, c e.1 ≠ c e.2

/-- `G` admits a proper `k`-colouring. -/
def Colourable (k : Nat) : Prop :=
  ∃ c : Coloring n k, IsProper G c

end Graph

/-!
## 2. Abstract “plane”: points with a unit relation
-/

/-- A proper `k`-colouring of the unit-relation graph on `P`. -/
def PlaneColouring (P : Type) (unit : P → P → Prop) (k : Nat) : Type :=
  { κ : P → Fin k // ∀ p q, unit p q → κ p ≠ κ q }

/-!
## 3. Unit-distance embedding and the reduction
-/

variable {n k : Nat} (G : Graph n) {P : Type} (unit : P → P → Prop) (emb : V n → P)

/-- **Reduction (forward).** Any proper plane `k`-colouring pulls back along a
unit-distance embedding to a proper `k`-colouring of `G`. -/
theorem reduction
    (h_emb : ∀ e ∈ G.edges, unit (emb e.1) (emb e.2))
    (h_planecol : PlaneColouring P unit k) : Graph.Colourable G k := by
  obtain ⟨κ, hκ⟩ := h_planecol
  refine ⟨fun v => κ (emb v), ?_⟩
  intro e he
  exact hκ (emb e.1) (emb e.2) (h_emb e he)

/-- **Contrapositive.** If `G` is not `k`-colourable, no proper plane `k`-colouring exists
    (chromatic number of the unit-relation graph on `P` is **> k**). -/
theorem not_colourable_implies_plane_chromatic_gt
    (h_emb : ∀ e ∈ G.edges, unit (emb e.1) (emb e.2))
    (h_not : ¬ Graph.Colourable G k) : ¬ Nonempty (PlaneColouring P unit k) := by
  intro ⟨pc⟩
  exact h_not (reduction G unit emb h_emb pc)

end UnitDistanceChromatic

/-!
## 4. Instantiate for Heule's G_529 and `k = 4` → plane needs ≥ 5 colours
-/

namespace DeGrey529.Chi5

open UnitDistanceChromatic

abbrev numVertices : Nat := 529
abbrev refutedColours : Nat := 4

/-- Heule's de Grey core graph G_529 (529 vertices, 2670 edges).
    Concrete edge list: `DeGrey529.edges` in `SounioDeGreyUnitDistance.lean` /
    `examples/erdos/data/degrey_529.edge`. This file keeps `G` abstract. -/
abbrev G529 (G : Graph numVertices) : Graph numVertices := G

/-- **Geometry leg (external).** Embedding of G_529 into the de Grey field point set
    such that every graph edge maps to a unit pair. Discharged by
    `DeGrey529.g529_all_edges_unit_distance` in `SounioDeGreyUnitDistance.lean`. -/
abbrev GeometryLeg {P : Type} (unit : P → P → Prop) (emb : V numVertices → P)
    (G : Graph numVertices) :=
  ∀ e ∈ G.edges, unit (emb e.1) (emb e.2)

/-- **SAT leg (external, cake_lpr).** G_529 is not 4-colourable. Verified by the
    CakeML machine-code LRAT checker; see `examples/erdos/CAKE_LPR_RESULT.md`. -/
abbrev SatLegNot4Colourable (G : Graph numVertices) :=
  ¬ Graph.Colourable G refutedColours

/-- **Main corollary.** Under the geometry and SAT legs, the de Grey unit-distance
    point set has **no** proper 4-colouring — equivalently, it needs **at least 5**
    colours (chromatic number ≥ 5). -/
theorem degrey_plane_needs_5_colours
    {P : Type} (unit : P → P → Prop) (emb : V numVertices → P)
    (G : Graph numVertices)
    (h_emb_g529 : GeometryLeg unit emb G)
    (h_sat : SatLegNot4Colourable G) :
    ¬ Nonempty (PlaneColouring P unit refutedColours) :=
  not_colourable_implies_plane_chromatic_gt G unit emb h_emb_g529 h_sat

#print axioms DeGrey529.Chi5.degrey_plane_needs_5_colours

#eval IO.println "SounioDeGreyChi5: reduction leg OK — G_529 not 4-colourable + unit embedding ⇒ plane needs ≥ 5 colours (hypotheses external)"

end DeGrey529.Chi5
