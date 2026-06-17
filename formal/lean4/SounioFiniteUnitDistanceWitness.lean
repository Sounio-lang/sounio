import SounioDeGreyChi5
import SounioSatColouringBridge
import SounioSatColouringSB
import SounioSatCubeCover

/-!
# SounioFiniteUnitDistanceWitness — a reusable obstruction certificate

This file factors out the tiny piece of logic shared by the Moser spindle,
G529, and any future exact unit-distance witness:

* a listed graph embeds into a point type with a unit relation;
* the listed graph is not `k`-colourable;
* therefore the ambient unit-relation graph has no proper `k`-colouring.

For a future chi>=6 attempt, the plug-in shape is the same certificate with
`k = 5`: exact unit-distance geometry plus a verified no-5-colouring theorem.
-/

namespace UnitDistanceChromatic

/-- A finite listed unit-distance obstruction over an arbitrary vertex type.

`k` is the number of colours being refuted. A certificate with `k = 5` is the
logical core needed for a chi>=6 witness once the exact geometry is supplied. -/
structure ListedUnitDistanceObstruction
    (Vert P : Type) (unit : P → P → Prop) (k : Nat) where
  edges : List (Vert × Vert)
  emb : Vert → P
  edges_unit : ∀ e ∈ edges, unit (emb e.1) (emb e.2)
  not_colourable : ¬ ∃ c : Vert → Fin k, ∀ e ∈ edges, c e.1 ≠ c e.2

namespace ListedUnitDistanceObstruction

variable {Vert P : Type} {unit : P → P → Prop} {k : Nat}

/-- Any proper ambient plane colouring pulls back along the witness embedding,
contradicting the finite no-`k`-colouring certificate. -/
theorem no_plane_colouring
    (W : ListedUnitDistanceObstruction Vert P unit k) :
    ¬ Nonempty (PlaneColouring P unit k) := by
  intro hpc
  rcases hpc with ⟨pc⟩
  rcases pc with ⟨κ, hκ⟩
  exact W.not_colourable ⟨fun v => κ (W.emb v), by
    intro e he
    exact hκ (W.emb e.1) (W.emb e.2) (W.edges_unit e he)⟩

/-- Build a listed obstruction from the older `Graph` API. -/
def ofGraph {n k : Nat} {P : Type} {unit : P → P → Prop}
    (G : Graph n) (emb : V n → P)
    (h_emb : ∀ e ∈ G.edges, unit (emb e.1) (emb e.2))
    (h_not : ¬ Graph.Colourable G k) :
    ListedUnitDistanceObstruction (V n) P unit k where
  edges := G.edges
  emb := emb
  edges_unit := h_emb
  not_colourable := by
    simpa [Graph.Colourable, Graph.IsProper, Coloring] using h_not

/-- Certificate shape for a future exact chi>=6 obstruction: no proper 5-colouring
of a finite listed unit-distance graph embedded in the ambient point type. -/
abbrev NoFiveColourWitness (Vert P : Type) (unit : P → P → Prop) :=
  ListedUnitDistanceObstruction Vert P unit 5

/-- The logical conclusion supplied by a no-5-colouring witness. This theorem is only
an interface: it does not assert that such a planar witness is currently present. -/
theorem no_five_colour_witness_refutes_plane_five_colouring
    (W : NoFiveColourWitness Vert P unit) :
    ¬ Nonempty (PlaneColouring P unit 5) :=
  W.no_plane_colouring

end ListedUnitDistanceObstruction

/-! ## Nat-indexed certificate shape for generated SAT/geometry witnesses. -/

/-- A finite unit-distance obstruction whose edge list uses raw `Nat` vertex ids.

This is the shape emitted naturally by graph/SAT generators. The `endpoints` field keeps
the finite vertex bound explicit, and the `not_colourable` field matches
`SounioSatColouring.not_colourable_of_unsat`. -/
structure NatEdgeUnitDistanceCertificate
    (n k : Nat) (P : Type) (unit : P → P → Prop) where
  edges : List (Nat × Nat)
  emb : Nat → P
  endpoints : ∀ e ∈ edges, e.1 < n ∧ e.2 < n
  unit_edges : ∀ e ∈ edges, unit (emb e.1) (emb e.2)
  not_colourable :
    ¬ ∃ c : Fin n → Fin k,
        ∀ e ∈ edges, ∀ (h1 : e.1 < n) (h2 : e.2 < n),
          c ⟨e.1, h1⟩ ≠ c ⟨e.2, h2⟩

namespace NatEdgeUnitDistanceCertificate

variable {n k : Nat} {P : Type} {unit : P → P → Prop}

/-- Pull back an ambient plane colouring along a Nat-indexed exact embedding. -/
theorem no_plane_colouring
    (W : NatEdgeUnitDistanceCertificate n k P unit) :
    ¬ Nonempty (PlaneColouring P unit k) := by
  intro hpc
  rcases hpc with ⟨pc⟩
  rcases pc with ⟨κ, hκ⟩
  exact W.not_colourable ⟨fun v => κ (W.emb v.val), by
    intro e he h1 h2
    exact hκ (W.emb e.1) (W.emb e.2) (W.unit_edges e he)⟩

/-- Nat-edge certificate shape specialized to a no-5-colouring witness. -/
abbrev NoFiveColourWitness (n : Nat) (P : Type) (unit : P → P → Prop) :=
  NatEdgeUnitDistanceCertificate n 5 P unit

/-- Logical conclusion of a Nat-indexed no-5-colouring witness. This is the exact
certificate target for a future planar chi>=6 graph, but asserts no such graph exists here. -/
theorem no_five_plane_colouring
    (W : NoFiveColourWitness n P unit) :
    ¬ Nonempty (PlaneColouring P unit 5) :=
  W.no_plane_colouring

/-- Generic no-5-colouring obstruction: exact unit-relation geometry plus a verified
no-5-colouring certificate refutes every ambient 5-colouring. This name is deliberately
neutral because `P` and `unit` are arbitrary; Euclidean chi>=6 promotion uses
`EuclideanNatEdgeExactGeometry.chi_ge_6_euclidean_plugin_contract`. -/
theorem generic_no_five_colour_obstruction
    (W : NoFiveColourWitness n P unit) :
    ¬ Nonempty (PlaneColouring P unit 5) :=
  no_five_plane_colouring W

/-- Direct Nat-edge constructor from an already-verified finite no-`k`-colouring theorem. -/
theorem no_plane_colouring_of_nat_edges
    (edges : List (Nat × Nat)) (emb : Nat → P)
    (hedges : ∀ e ∈ edges, e.1 < n ∧ e.2 < n)
    (hunit : ∀ e ∈ edges, unit (emb e.1) (emb e.2))
    (hnot : ¬ ∃ c : Fin n → Fin k,
        ∀ e ∈ edges, ∀ (h1 : e.1 < n) (h2 : e.2 < n),
          c ⟨e.1, h1⟩ ≠ c ⟨e.2, h2⟩) :
    ¬ Nonempty (PlaneColouring P unit k) := by
  let W : NatEdgeUnitDistanceCertificate n k P unit :=
    { edges := edges
      emb := emb
      endpoints := hedges
      unit_edges := hunit
      not_colourable := hnot }
  exact W.no_plane_colouring

/-- Package a raw colouring-CNF `Unsat` plus exact geometry as a Nat-edge obstruction
certificate. This is the reusable object form of `no_plane_colouring_of_colourCNF_unsat`. -/
def ofColourCNFUnsat
    (hk : 0 < k)
    (edges : List (Nat × Nat)) (emb : Nat → P)
    (hedges : ∀ e ∈ edges, e.1 < n ∧ e.2 < n)
    (hunit : ∀ e ∈ edges, unit (emb e.1) (emb e.2))
    (hunsat : (SounioSatColouring.colourCNF n k edges).Unsat) :
    NatEdgeUnitDistanceCertificate n k P unit where
  edges := edges
  emb := emb
  endpoints := hedges
  unit_edges := hunit
  not_colourable := SounioSatColouring.not_colourable_of_unsat hk hedges hunsat

/-- Direct SAT-wired constructor: an LRAT-checked colouring-CNF `Unsat`, plus exact unit
geometry, yields the ambient no-`k`-colouring theorem. -/
theorem no_plane_colouring_of_colourCNF_unsat
    (hk : 0 < k)
    (edges : List (Nat × Nat)) (emb : Nat → P)
    (hedges : ∀ e ∈ edges, e.1 < n ∧ e.2 < n)
    (hunit : ∀ e ∈ edges, unit (emb e.1) (emb e.2))
    (hunsat : (SounioSatColouring.colourCNF n k edges).Unsat) :
    ¬ Nonempty (PlaneColouring P unit k) := by
  exact (ofColourCNFUnsat (n := n) (k := k) (P := P) (unit := unit)
    hk edges emb hedges hunit hunsat).no_plane_colouring

/-- Package a `k = 4` triangle-precoloured LRAT result as a reusable Nat-edge certificate. -/
def ofColourCNFsb4UnsatTri
    (edges : List (Nat × Nat)) (emb : Nat → P)
    (a b c : Nat)
    (ha : a < n) (hb : b < n) (hc : c < n)
    (hab : (a, b) ∈ edges) (hac : (a, c) ∈ edges) (hbc : (b, c) ∈ edges)
    (hedges : ∀ e ∈ edges, e.1 < n ∧ e.2 < n)
    (hunit : ∀ e ∈ edges, unit (emb e.1) (emb e.2))
    (hunsat : (SounioSatColouringSB.colourCNFsb a b c n edges).Unsat) :
    NatEdgeUnitDistanceCertificate n 4 P unit where
  edges := edges
  emb := emb
  endpoints := hedges
  unit_edges := hunit
  not_colourable := SounioSatColouringSB.not_colourable_of_unsat_tri
    ha hb hc hab hac hbc hedges hunsat

/-- SAT-wired constructor for the existing `k = 4` triangle-precolour symmetry break. -/
theorem no_plane_colouring_of_colourCNFsb4_unsat_tri
    (edges : List (Nat × Nat)) (emb : Nat → P)
    (a b c : Nat)
    (ha : a < n) (hb : b < n) (hc : c < n)
    (hab : (a, b) ∈ edges) (hac : (a, c) ∈ edges) (hbc : (b, c) ∈ edges)
    (hedges : ∀ e ∈ edges, e.1 < n ∧ e.2 < n)
    (hunit : ∀ e ∈ edges, unit (emb e.1) (emb e.2))
    (hunsat : (SounioSatColouringSB.colourCNFsb a b c n edges).Unsat) :
    ¬ Nonempty (PlaneColouring P unit 4) :=
  (ofColourCNFsb4UnsatTri (n := n) (P := P) (unit := unit)
    edges emb a b c ha hb hc hab hac hbc hedges hunit hunsat).no_plane_colouring

/-- Package a future `k = 5` triangle-precoloured LRAT result as the exact
`NoFiveColourWitness` object expected by the chi>=6 search lane. -/
def noFiveWitnessOfColourCNFsb5UnsatTri
    (edges : List (Nat × Nat)) (emb : Nat → P)
    (a b c : Nat)
    (ha : a < n) (hb : b < n) (hc : c < n)
    (hab : (a, b) ∈ edges) (hac : (a, c) ∈ edges) (hbc : (b, c) ∈ edges)
    (hedges : ∀ e ∈ edges, e.1 < n ∧ e.2 < n)
    (hunit : ∀ e ∈ edges, unit (emb e.1) (emb e.2))
    (hunsat : (SounioSatColouringSB.colourCNFsb5 a b c n edges).Unsat) :
    NoFiveColourWitness n P unit where
  edges := edges
  emb := emb
  endpoints := hedges
  unit_edges := hunit
  not_colourable := SounioSatColouringSB.not_colourable5_of_unsat_tri
    ha hb hc hab hac hbc hedges hunsat

/-- SAT-wired constructor for a future `k = 5` triangle-precolour symmetry break. -/
theorem no_plane_colouring_of_colourCNFsb5_unsat_tri
    (edges : List (Nat × Nat)) (emb : Nat → P)
    (a b c : Nat)
    (ha : a < n) (hb : b < n) (hc : c < n)
    (hab : (a, b) ∈ edges) (hac : (a, c) ∈ edges) (hbc : (b, c) ∈ edges)
    (hedges : ∀ e ∈ edges, e.1 < n ∧ e.2 < n)
    (hunit : ∀ e ∈ edges, unit (emb e.1) (emb e.2))
    (hunsat : (SounioSatColouringSB.colourCNFsb5 a b c n edges).Unsat) :
    ¬ Nonempty (PlaneColouring P unit 5) :=
  (noFiveWitnessOfColourCNFsb5UnsatTri (n := n) (P := P) (unit := unit)
    edges emb a b c ha hb hc hab hac hbc hedges hunit hunsat).no_plane_colouring

/-- Package a five-leaf cube-cover LRAT result as the exact `NoFiveColourWitness`
object expected by the chi>=6 search lane. The split cover is justified in Lean by
`SounioSatCubeCover.unsat_of_split_vertex5`; the GPU/search producer only supplies
the five leaf UNSAT facts. -/
def noFiveWitnessOfSplitVertex5Unsat
    (edges : List (Nat × Nat)) (emb : Nat → P)
    (v : Nat) (hv : v < n)
    (hedges : ∀ e ∈ edges, e.1 < n ∧ e.2 < n)
    (hunit : ∀ e ∈ edges, unit (emb e.1) (emb e.2))
    (h0 : (SounioSatCubeCover.colourCNFWithUnit n 5 edges v 0).Unsat)
    (h1 : (SounioSatCubeCover.colourCNFWithUnit n 5 edges v 1).Unsat)
    (h2 : (SounioSatCubeCover.colourCNFWithUnit n 5 edges v 2).Unsat)
    (h3 : (SounioSatCubeCover.colourCNFWithUnit n 5 edges v 3).Unsat)
    (h4 : (SounioSatCubeCover.colourCNFWithUnit n 5 edges v 4).Unsat) :
    NoFiveColourWitness n P unit :=
  ofColourCNFUnsat
    (n := n) (k := 5) (P := P) (unit := unit)
    (by decide) edges emb hedges hunit
    (SounioSatCubeCover.unsat_of_split_vertex5
      (n := n) (edges := edges) (v := v) hv h0 h1 h2 h3 h4)

/-- Direct ambient no-5-colouring theorem from exact geometry and five cube-cover
UNSAT leaves. -/
theorem no_plane_colouring_of_split_vertex5_unsat
    (edges : List (Nat × Nat)) (emb : Nat → P)
    (v : Nat) (hv : v < n)
    (hedges : ∀ e ∈ edges, e.1 < n ∧ e.2 < n)
    (hunit : ∀ e ∈ edges, unit (emb e.1) (emb e.2))
    (h0 : (SounioSatCubeCover.colourCNFWithUnit n 5 edges v 0).Unsat)
    (h1 : (SounioSatCubeCover.colourCNFWithUnit n 5 edges v 1).Unsat)
    (h2 : (SounioSatCubeCover.colourCNFWithUnit n 5 edges v 2).Unsat)
    (h3 : (SounioSatCubeCover.colourCNFWithUnit n 5 edges v 3).Unsat)
    (h4 : (SounioSatCubeCover.colourCNFWithUnit n 5 edges v 4).Unsat) :
    ¬ Nonempty (PlaneColouring P unit 5) :=
  (noFiveWitnessOfSplitVertex5Unsat (n := n) (P := P) (unit := unit)
    edges emb v hv hedges hunit h0 h1 h2 h3 h4).no_plane_colouring

/-- Package a generic cube-cover LRAT result as the exact `NoFiveColourWitness`
object expected by the chi>=6 search lane. The search producer supplies the
cube list, a Lean proof that those cubes cover all base-CNF satisfying
assignments, and one Lean-checked UNSAT fact per cube. -/
def noFiveWitnessOfCubeCoverUnsat
    (edges : List (Nat × Nat)) (emb : Nat → P)
    (cubes : List SounioSatCubeCover.Cube)
    (hcover : SounioSatCubeCover.CubeCover n 5 edges cubes)
    (hedges : ∀ e ∈ edges, e.1 < n ∧ e.2 < n)
    (hunit : ∀ e ∈ edges, unit (emb e.1) (emb e.2))
    (hunsat : ∀ cube, cube ∈ cubes →
      (SounioSatCubeCover.colourCNFWithCube n 5 edges cube).Unsat) :
    NoFiveColourWitness n P unit :=
  ofColourCNFUnsat
    (n := n) (k := 5) (P := P) (unit := unit)
    (by decide) edges emb hedges hunit
    (SounioSatCubeCover.unsat_of_cube_cover
      (n := n) (k := 5) (edges := edges) (cubes := cubes) hcover hunsat)

/-- Direct ambient no-5-colouring theorem from exact geometry and a generic
cube-cover UNSAT certificate. -/
theorem no_plane_colouring_of_cube_cover_unsat
    (edges : List (Nat × Nat)) (emb : Nat → P)
    (cubes : List SounioSatCubeCover.Cube)
    (hcover : SounioSatCubeCover.CubeCover n 5 edges cubes)
    (hedges : ∀ e ∈ edges, e.1 < n ∧ e.2 < n)
    (hunit : ∀ e ∈ edges, unit (emb e.1) (emb e.2))
    (hunsat : ∀ cube, cube ∈ cubes →
      (SounioSatCubeCover.colourCNFWithCube n 5 edges cube).Unsat) :
    ¬ Nonempty (PlaneColouring P unit 5) :=
  (noFiveWitnessOfCubeCoverUnsat (n := n) (P := P) (unit := unit)
    edges emb cubes hcover hedges hunit hunsat).no_plane_colouring

end NatEdgeUnitDistanceCertificate

/-! ## Geometry-first adapter for generated candidates. -/

/-- Exact Nat-indexed finite geometry without a SAT certificate attached yet.

Future search modules can populate this object from generated exact coordinates, then attach either
a plain `colourCNF` UNSAT proof or a triangle-precoloured `colourCNFsb5` proof to obtain the public
`NoFiveColourWitness` object. This deliberately contains no chromatic conclusion by itself. -/
structure NatEdgeExactGeometry
    (n : Nat) (P : Type) (unit : P → P → Prop) where
  edges : List (Nat × Nat)
  emb : Nat → P
  emb_injective : ∀ {i j}, i < n → j < n → emb i = emb j → i = j
  endpoints : ∀ e ∈ edges, e.1 < n ∧ e.2 < n
  unit_edges : ∀ e ∈ edges, unit (emb e.1) (emb e.2)

namespace NatEdgeExactGeometry

variable {n k : Nat} {P : Type} {unit : P → P → Prop}

/-- Attach an already-verified finite no-`k`-colouring theorem to exact geometry. -/
def toCertificate
    (G : NatEdgeExactGeometry n P unit)
    (hnot : ¬ ∃ c : Fin n → Fin k,
        ∀ e ∈ G.edges, ∀ (h1 : e.1 < n) (h2 : e.2 < n),
          c ⟨e.1, h1⟩ ≠ c ⟨e.2, h2⟩) :
    NatEdgeUnitDistanceCertificate n k P unit where
  edges := G.edges
  emb := G.emb
  endpoints := G.endpoints
  unit_edges := G.unit_edges
  not_colourable := hnot

/-- Direct ambient-colouring refutation from exact geometry plus a finite no-`k` theorem. -/
theorem noPlaneColouringOfNotColourable
    (G : NatEdgeExactGeometry n P unit)
    (hnot : ¬ ∃ c : Fin n → Fin k,
        ∀ e ∈ G.edges, ∀ (h1 : e.1 < n) (h2 : e.2 < n),
          c ⟨e.1, h1⟩ ≠ c ⟨e.2, h2⟩) :
    ¬ Nonempty (PlaneColouring P unit k) :=
  (G.toCertificate hnot).no_plane_colouring

/-- Attach a reflected plain colouring-CNF UNSAT proof to exact geometry. -/
def certificateOfColourCNFUnsat
    (G : NatEdgeExactGeometry n P unit)
    (hk : 0 < k)
    (hunsat : (SounioSatColouring.colourCNF n k G.edges).Unsat) :
    NatEdgeUnitDistanceCertificate n k P unit :=
  NatEdgeUnitDistanceCertificate.ofColourCNFUnsat
    hk G.edges G.emb G.endpoints G.unit_edges hunsat

/-- Attach a reflected plain no-5-colouring proof to exact geometry. -/
def noFiveWitnessOfColourCNFUnsat
    (G : NatEdgeExactGeometry n P unit)
    (hunsat : (SounioSatColouring.colourCNF n 5 G.edges).Unsat) :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness n P unit :=
  G.certificateOfColourCNFUnsat (k := 5) (by decide) hunsat

/-- Direct ambient no-5-colouring theorem from exact geometry plus plain `colourCNF n 5` UNSAT. -/
theorem noFivePlaneColouringOfColourCNFUnsat
    (G : NatEdgeExactGeometry n P unit)
    (hunsat : (SounioSatColouring.colourCNF n 5 G.edges).Unsat) :
    ¬ Nonempty (PlaneColouring P unit 5) :=
  (G.noFiveWitnessOfColourCNFUnsat hunsat).no_plane_colouring

/-- Attach a reflected triangle-precoloured `k = 5` UNSAT proof to exact geometry. -/
def noFiveWitnessOfColourCNFsb5UnsatTri
    (G : NatEdgeExactGeometry n P unit)
    (a b c : Nat)
    (ha : a < n) (hb : b < n) (hc : c < n)
    (hab : (a, b) ∈ G.edges) (hac : (a, c) ∈ G.edges) (hbc : (b, c) ∈ G.edges)
    (hunsat : (SounioSatColouringSB.colourCNFsb5 a b c n G.edges).Unsat) :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness n P unit :=
  NatEdgeUnitDistanceCertificate.noFiveWitnessOfColourCNFsb5UnsatTri
    G.edges G.emb a b c ha hb hc hab hac hbc G.endpoints G.unit_edges hunsat

/-- Direct ambient no-5-colouring theorem from exact geometry plus SB5 UNSAT. -/
theorem noFivePlaneColouringOfColourCNFsb5UnsatTri
    (G : NatEdgeExactGeometry n P unit)
    (a b c : Nat)
    (ha : a < n) (hb : b < n) (hc : c < n)
    (hab : (a, b) ∈ G.edges) (hac : (a, c) ∈ G.edges) (hbc : (b, c) ∈ G.edges)
    (hunsat : (SounioSatColouringSB.colourCNFsb5 a b c n G.edges).Unsat) :
    ¬ Nonempty (PlaneColouring P unit 5) :=
  (G.noFiveWitnessOfColourCNFsb5UnsatTri a b c ha hb hc hab hac hbc hunsat).no_plane_colouring

/-- Attach five reflected cube-cover leaf UNSAT proofs to exact geometry. -/
def noFiveWitnessOfSplitVertex5Unsat
    (G : NatEdgeExactGeometry n P unit)
    (v : Nat) (hv : v < n)
    (h0 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.edges v 0).Unsat)
    (h1 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.edges v 1).Unsat)
    (h2 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.edges v 2).Unsat)
    (h3 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.edges v 3).Unsat)
    (h4 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.edges v 4).Unsat) :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness n P unit :=
  NatEdgeUnitDistanceCertificate.noFiveWitnessOfSplitVertex5Unsat
    G.edges G.emb v hv G.endpoints G.unit_edges h0 h1 h2 h3 h4

/-- Direct ambient no-5-colouring theorem from exact geometry and cube-cover UNSAT leaves. -/
theorem noFivePlaneColouringOfSplitVertex5Unsat
    (G : NatEdgeExactGeometry n P unit)
    (v : Nat) (hv : v < n)
    (h0 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.edges v 0).Unsat)
    (h1 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.edges v 1).Unsat)
    (h2 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.edges v 2).Unsat)
    (h3 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.edges v 3).Unsat)
    (h4 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.edges v 4).Unsat) :
    ¬ Nonempty (PlaneColouring P unit 5) :=
  (G.noFiveWitnessOfSplitVertex5Unsat v hv h0 h1 h2 h3 h4).no_plane_colouring

/-- Attach a generic cube-cover proof and reflected leaf UNSAT proofs to exact geometry. -/
def noFiveWitnessOfCubeCoverUnsat
    (G : NatEdgeExactGeometry n P unit)
    (cubes : List SounioSatCubeCover.Cube)
    (hcover : SounioSatCubeCover.CubeCover n 5 G.edges cubes)
    (hunsat : ∀ cube, cube ∈ cubes →
      (SounioSatCubeCover.colourCNFWithCube n 5 G.edges cube).Unsat) :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness n P unit :=
  NatEdgeUnitDistanceCertificate.noFiveWitnessOfCubeCoverUnsat
    G.edges G.emb cubes hcover G.endpoints G.unit_edges hunsat

/-- Direct ambient no-5-colouring theorem from exact geometry and a generic cube cover. -/
theorem noFivePlaneColouringOfCubeCoverUnsat
    (G : NatEdgeExactGeometry n P unit)
    (cubes : List SounioSatCubeCover.Cube)
    (hcover : SounioSatCubeCover.CubeCover n 5 G.edges cubes)
    (hunsat : ∀ cube, cube ∈ cubes →
      (SounioSatCubeCover.colourCNFWithCube n 5 G.edges cube).Unsat) :
    ¬ Nonempty (PlaneColouring P unit 5) :=
  (G.noFiveWitnessOfCubeCoverUnsat cubes hcover hunsat).no_plane_colouring

end NatEdgeExactGeometry

/-! ## Squared-distance geometry adapter for real Euclidean candidates. -/

/-- Minimal exact scalar-field law bundle for promoted squared-distance geometry.

This is intentionally local and Mathlib-free. It is strong enough to rule out
degenerate "all distances are one" scalars, while still letting a future
candidate choose the exact arithmetic domain used by its generated coordinates. -/
structure ExactFieldLike (α : Type) where
  zero : α
  one : α
  add : α → α → α
  neg : α → α
  sub : α → α → α
  mul : α → α → α
  inv : α → α
  ofNat : Nat → α
  add_assoc : ∀ a b c, add (add a b) c = add a (add b c)
  add_comm : ∀ a b, add a b = add b a
  zero_add : ∀ a, add zero a = a
  add_zero : ∀ a, add a zero = a
  add_left_neg : ∀ a, add (neg a) a = zero
  sub_eq_add_neg : ∀ a b, sub a b = add a (neg b)
  mul_assoc : ∀ a b c, mul (mul a b) c = mul a (mul b c)
  mul_comm : ∀ a b, mul a b = mul b a
  one_mul : ∀ a, mul one a = a
  mul_one : ∀ a, mul a one = a
  left_distrib : ∀ a b c, mul a (add b c) = add (mul a b) (mul a c)
  right_distrib : ∀ a b c, mul (add a b) c = add (mul a c) (mul b c)
  zero_ne_one : zero ≠ one
  inv_mul_cancel : ∀ a, a ≠ zero → mul (inv a) a = one
  ofNat_zero : ofNat 0 = zero
  ofNat_one : ofNat 1 = one
  ofNat_add : ∀ m n, ofNat (m + n) = add (ofNat m) (ofNat n)
  ofNat_mul : ∀ m n, ofNat (m * n) = mul (ofNat m) (ofNat n)
  ofNat_inj : ∀ {m n}, ofNat m = ofNat n → m = n

/-- Exact coordinate data proving that an ambient unit relation is induced by a
two-coordinate squared-distance formula over a characteristic-zero field-like
scalar.

The candidate chooses its exact scalar domain and proves both the scalar laws and
that its `unit` relation is precisely `(x₁-x₂)² + (y₁-y₂)² = 1` in that domain.
It also supplies the metric sanity facts used by the promotion contract:
zero distance is equality, unit distance is symmetric, and no point is unit
distant from itself.
The structure is a promotion fuse for future chi>=6 candidates; finite smoke
relations should not inhabit it. -/
structure ExactSquaredDistancePlane
    (P : Type) (unit : P → P → Prop) where
  Scalar : Type
  scalar : ExactFieldLike Scalar
  x : P → Scalar
  y : P → Scalar
  dist2 : P → P → Scalar
  dist2_formula :
    ∀ p q,
      dist2 p q =
        scalar.add
          (scalar.mul (scalar.sub (x p) (x q)) (scalar.sub (x p) (x q)))
          (scalar.mul (scalar.sub (y p) (y q)) (scalar.sub (y p) (y q)))
  unit_iff_dist2_eq_one :
    ∀ p q, unit p q ↔ dist2 p q = scalar.one
  dist2_zero_iff_eq :
    ∀ p q, dist2 p q = scalar.zero ↔ p = q
  unit_symm :
    ∀ p q, unit p q → unit q p
  unit_irrefl :
    ∀ p, ¬ unit p p

/-- Geometry-first package for a candidate that claims exact Euclidean squared-distance
geometry. It is still only geometry; a no-5 SAT/LRAT certificate must be attached separately. -/
structure EuclideanNatEdgeExactGeometry
    (n : Nat) (P : Type) (unit : P → P → Prop) where
  exact : NatEdgeExactGeometry n P unit
  plane : ExactSquaredDistancePlane P unit

namespace EuclideanNatEdgeExactGeometry

variable {n : Nat} {P : Type} {unit : P → P → Prop}

/-- Attach a reflected plain no-5-colouring proof to exact squared-distance geometry. -/
def noFiveWitnessOfColourCNFUnsat
    (G : EuclideanNatEdgeExactGeometry n P unit)
    (hunsat : (SounioSatColouring.colourCNF n 5 G.exact.edges).Unsat) :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness n P unit :=
  G.exact.noFiveWitnessOfColourCNFUnsat hunsat

/-- Attach a reflected SB5 no-5-colouring proof to exact squared-distance geometry. -/
def noFiveWitnessOfColourCNFsb5UnsatTri
    (G : EuclideanNatEdgeExactGeometry n P unit)
    (a b c : Nat)
    (ha : a < n) (hb : b < n) (hc : c < n)
    (hab : (a, b) ∈ G.exact.edges)
    (hac : (a, c) ∈ G.exact.edges)
    (hbc : (b, c) ∈ G.exact.edges)
    (hunsat : (SounioSatColouringSB.colourCNFsb5 a b c n G.exact.edges).Unsat) :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness n P unit :=
  G.exact.noFiveWitnessOfColourCNFsb5UnsatTri a b c ha hb hc hab hac hbc hunsat

/-- Public Euclidean chi>=6 plug-in contract. A candidate reaches this theorem only after
supplying exact squared-distance geometry and a verified no-5-colouring certificate. -/
theorem chi_ge_6_euclidean_plugin_contract
    (G : EuclideanNatEdgeExactGeometry n P unit)
    (W : NatEdgeUnitDistanceCertificate.NoFiveColourWitness n P unit)
    (_hedges : W.edges = G.exact.edges)
    (_hemb : W.emb = G.exact.emb) :
    ¬ Nonempty (PlaneColouring P unit 5) :=
  NatEdgeUnitDistanceCertificate.generic_no_five_colour_obstruction W

/-- Direct theorem for the plain `colourCNF n 5` route. -/
theorem noFivePlaneColouringOfColourCNFUnsat
    (G : EuclideanNatEdgeExactGeometry n P unit)
    (hunsat : (SounioSatColouring.colourCNF n 5 G.exact.edges).Unsat) :
    ¬ Nonempty (PlaneColouring P unit 5) :=
  chi_ge_6_euclidean_plugin_contract G
    (G.noFiveWitnessOfColourCNFUnsat hunsat) rfl rfl

/-- Direct theorem for the triangle-precoloured `colourCNFsb5` route. -/
theorem noFivePlaneColouringOfColourCNFsb5UnsatTri
    (G : EuclideanNatEdgeExactGeometry n P unit)
    (a b c : Nat)
    (ha : a < n) (hb : b < n) (hc : c < n)
    (hab : (a, b) ∈ G.exact.edges)
    (hac : (a, c) ∈ G.exact.edges)
    (hbc : (b, c) ∈ G.exact.edges)
    (hunsat : (SounioSatColouringSB.colourCNFsb5 a b c n G.exact.edges).Unsat) :
    ¬ Nonempty (PlaneColouring P unit 5) :=
  chi_ge_6_euclidean_plugin_contract G
    (G.noFiveWitnessOfColourCNFsb5UnsatTri a b c ha hb hc hab hac hbc hunsat) rfl rfl

/-- Attach reflected cube-cover leaf UNSAT proofs to exact squared-distance geometry. -/
def noFiveWitnessOfSplitVertex5Unsat
    (G : EuclideanNatEdgeExactGeometry n P unit)
    (v : Nat) (hv : v < n)
    (h0 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.exact.edges v 0).Unsat)
    (h1 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.exact.edges v 1).Unsat)
    (h2 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.exact.edges v 2).Unsat)
    (h3 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.exact.edges v 3).Unsat)
    (h4 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.exact.edges v 4).Unsat) :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness n P unit :=
  G.exact.noFiveWitnessOfSplitVertex5Unsat v hv h0 h1 h2 h3 h4

/-- Direct Euclidean plug-in theorem for the cube-cover route. -/
theorem noFivePlaneColouringOfSplitVertex5Unsat
    (G : EuclideanNatEdgeExactGeometry n P unit)
    (v : Nat) (hv : v < n)
    (h0 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.exact.edges v 0).Unsat)
    (h1 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.exact.edges v 1).Unsat)
    (h2 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.exact.edges v 2).Unsat)
    (h3 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.exact.edges v 3).Unsat)
    (h4 : (SounioSatCubeCover.colourCNFWithUnit n 5 G.exact.edges v 4).Unsat) :
    ¬ Nonempty (PlaneColouring P unit 5) :=
  chi_ge_6_euclidean_plugin_contract G
    (G.noFiveWitnessOfSplitVertex5Unsat v hv h0 h1 h2 h3 h4) rfl rfl

/-- Attach a generic cube-cover proof and reflected leaf UNSAT proofs to exact
squared-distance geometry. -/
def noFiveWitnessOfCubeCoverUnsat
    (G : EuclideanNatEdgeExactGeometry n P unit)
    (cubes : List SounioSatCubeCover.Cube)
    (hcover : SounioSatCubeCover.CubeCover n 5 G.exact.edges cubes)
    (hunsat : ∀ cube, cube ∈ cubes →
      (SounioSatCubeCover.colourCNFWithCube n 5 G.exact.edges cube).Unsat) :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness n P unit :=
  G.exact.noFiveWitnessOfCubeCoverUnsat cubes hcover hunsat

/-- Direct Euclidean plug-in theorem for the generic cube-cover route. -/
theorem noFivePlaneColouringOfCubeCoverUnsat
    (G : EuclideanNatEdgeExactGeometry n P unit)
    (cubes : List SounioSatCubeCover.Cube)
    (hcover : SounioSatCubeCover.CubeCover n 5 G.exact.edges cubes)
    (hunsat : ∀ cube, cube ∈ cubes →
      (SounioSatCubeCover.colourCNFWithCube n 5 G.exact.edges cube).Unsat) :
    ¬ Nonempty (PlaneColouring P unit 5) :=
  chi_ge_6_euclidean_plugin_contract G
    (G.noFiveWitnessOfCubeCoverUnsat cubes hcover hunsat) rfl rfl

end EuclideanNatEdgeExactGeometry

#print axioms ListedUnitDistanceObstruction.no_plane_colouring
#print axioms ListedUnitDistanceObstruction.no_five_colour_witness_refutes_plane_five_colouring
#print axioms NatEdgeUnitDistanceCertificate.no_plane_colouring
#print axioms NatEdgeUnitDistanceCertificate.no_five_plane_colouring
#print axioms NatEdgeUnitDistanceCertificate.generic_no_five_colour_obstruction
#print axioms NatEdgeUnitDistanceCertificate.no_plane_colouring_of_nat_edges
#print axioms NatEdgeUnitDistanceCertificate.ofColourCNFUnsat
#print axioms NatEdgeUnitDistanceCertificate.no_plane_colouring_of_colourCNF_unsat
#print axioms NatEdgeUnitDistanceCertificate.ofColourCNFsb4UnsatTri
#print axioms NatEdgeUnitDistanceCertificate.no_plane_colouring_of_colourCNFsb4_unsat_tri
#print axioms NatEdgeUnitDistanceCertificate.noFiveWitnessOfColourCNFsb5UnsatTri
#print axioms NatEdgeUnitDistanceCertificate.no_plane_colouring_of_colourCNFsb5_unsat_tri
#print axioms NatEdgeUnitDistanceCertificate.noFiveWitnessOfSplitVertex5Unsat
#print axioms NatEdgeUnitDistanceCertificate.no_plane_colouring_of_split_vertex5_unsat
#print axioms NatEdgeUnitDistanceCertificate.noFiveWitnessOfCubeCoverUnsat
#print axioms NatEdgeUnitDistanceCertificate.no_plane_colouring_of_cube_cover_unsat
#print axioms NatEdgeExactGeometry.toCertificate
#print axioms NatEdgeExactGeometry.noPlaneColouringOfNotColourable
#print axioms NatEdgeExactGeometry.certificateOfColourCNFUnsat
#print axioms NatEdgeExactGeometry.noFiveWitnessOfColourCNFUnsat
#print axioms NatEdgeExactGeometry.noFivePlaneColouringOfColourCNFUnsat
#print axioms NatEdgeExactGeometry.noFiveWitnessOfColourCNFsb5UnsatTri
#print axioms NatEdgeExactGeometry.noFiveWitnessOfCubeCoverUnsat
#print axioms NatEdgeExactGeometry.noFivePlaneColouringOfCubeCoverUnsat
#print axioms NatEdgeExactGeometry.noFivePlaneColouringOfColourCNFsb5UnsatTri
#print axioms NatEdgeExactGeometry.noFiveWitnessOfSplitVertex5Unsat
#print axioms NatEdgeExactGeometry.noFivePlaneColouringOfSplitVertex5Unsat
#print axioms ExactFieldLike.zero_ne_one
#print axioms ExactSquaredDistancePlane.unit_irrefl
#print axioms EuclideanNatEdgeExactGeometry.chi_ge_6_euclidean_plugin_contract
#print axioms EuclideanNatEdgeExactGeometry.noFivePlaneColouringOfColourCNFUnsat
#print axioms EuclideanNatEdgeExactGeometry.noFivePlaneColouringOfColourCNFsb5UnsatTri
#print axioms EuclideanNatEdgeExactGeometry.noFiveWitnessOfSplitVertex5Unsat
#print axioms EuclideanNatEdgeExactGeometry.noFivePlaneColouringOfSplitVertex5Unsat
#print axioms EuclideanNatEdgeExactGeometry.noFiveWitnessOfCubeCoverUnsat
#print axioms EuclideanNatEdgeExactGeometry.noFivePlaneColouringOfCubeCoverUnsat

#eval IO.println "SounioFiniteUnitDistanceWitness: reusable finite unit-distance obstruction interface; chi>=6 plug-in shape is k=5, with an exact squared-distance geometry adapter for future Euclidean candidates."

end UnitDistanceChromatic
