import SounioDeGreyChi5
import SounioMultiquadParam

set_option maxHeartbeats 1000000

/-!
# SounioMoserSpindleQ311 — the Moser spindle over `ℚ(√3,√11)` (Madore's χ ≥ 4 line)

Mathlib-free. This is the **base case of Madore's multiquadratic line** (arXiv:1509.07023):
the Moser spindle is realised with *exact* coordinates in `ℚ(√3,√11)`, every one of its 11 unit
edges has squared distance exactly the rational `1` (here scaled `×12`, target `144`, all
`√3 / √11 / √33` components cancelling to `0`), and the graph is **not 3-colourable**, hence
`χ ≥ 4` for the unit-distance graph of the `ℚ(√3,√11)`-plane.

Pipeline:
1. exact field arithmetic `Qf = a + b√3 + c√11 + d√33` (integer 4-tuples, `×12` scaled);
2. the 7 spindle vertices and `dist2`, with the 11 unit edges verified by `decide`
   (`dist2 = ⟨144,0,0,0⟩`, i.e. rational with zero surd part);
3. `¬ 3-colourable` (the 11 disequalities over `Fin 3` are refuted by `omega`) and a `4`-colouring
   witness ⇒ `χ(spindle) = 4`;
4. the generic unit-distance reduction of `SounioDeGreyChi5` ⇒ the `ℚ(√3,√11)`-plane needs `≥ 4`
   colours.

The independence of the basis `{1,√3,√11,√33}` is `Multiquad.indep_3_11`
(`SounioMultiquadParam`); see §6 for the cross-check.
-/

namespace MoserSpindleQ311

open UnitDistanceChromatic

/-! ## 1. Exact `ℚ(√3,√11)` arithmetic on integer 4-tuples `(a, b√3, c√11, d√33)`. -/

/-- An element `a + b√3 + c√11 + d√33` of `ℚ(√3,√11)` as an integer 4-tuple (`×12` scaled). -/
structure Qf where
  a : Int
  b : Int
  c : Int
  d : Int
deriving DecidableEq, Repr

def qadd (x y : Qf) : Qf := ⟨x.a + y.a, x.b + y.b, x.c + y.c, x.d + y.d⟩
def qsub (x y : Qf) : Qf := ⟨x.a - y.a, x.b - y.b, x.c - y.c, x.d - y.d⟩

/-- Multiplication via `√3·√3=3, √11·√11=11, √33·√33=33, √3·√11=√33, √3·√33=3√11, √11·√33=11√3`. -/
def qmul (x y : Qf) : Qf :=
  ⟨ x.a*y.a + 3*x.b*y.b + 11*x.c*y.c + 33*x.d*y.d,
    x.a*y.b + y.a*x.b + 11*(x.c*y.d + y.c*x.d),
    x.a*y.c + y.a*x.c + 3*(x.b*y.d + y.b*x.d),
    x.a*y.d + y.a*x.d + x.b*y.c + y.b*x.c ⟩

def qsq (x : Qf) : Qf := qmul x x

/-! ## 2. The 7 spindle vertices (`A..G`), scaled `×12`. -/

/-- `x`-coordinate of each vertex (`A=0 … G=6`). -/
def vx : Fin 7 → Qf
  | 0 => ⟨0, 0, 0, 0⟩      -- A = (0,0)
  | 1 => ⟨0, 6, 0, 0⟩      -- B = (6√3, 6)
  | 2 => ⟨0, 12, 0, 0⟩     -- C = (12√3, 0)
  | 3 => ⟨0, 6, 0, 0⟩      -- D = (6√3, -6)
  | 4 => ⟨0, 5, -1, 0⟩     -- E = (5√3 - √11, 5 + √33)
  | 5 => ⟨0, 10, 0, 0⟩     -- F = (10√3, 2√33)
  | 6 => ⟨0, 5, 1, 0⟩      -- G = (5√3 + √11, √33 - 5)

/-- `y`-coordinate of each vertex. -/
def vy : Fin 7 → Qf
  | 0 => ⟨0, 0, 0, 0⟩
  | 1 => ⟨6, 0, 0, 0⟩
  | 2 => ⟨0, 0, 0, 0⟩
  | 3 => ⟨-6, 0, 0, 0⟩
  | 4 => ⟨5, 0, 0, 1⟩
  | 5 => ⟨0, 0, 0, 2⟩
  | 6 => ⟨-5, 0, 0, 1⟩

/-- Squared Euclidean distance of two field-plane points, exact in `ℚ(√3,√11)`. -/
def dist2pt (p q : Qf × Qf) : Qf :=
  qadd (qsq (qsub p.1 q.1)) (qsq (qsub p.2 q.2))

/-- Squared distance between two spindle vertices. -/
def dist2 (i j : Fin 7) : Qf := dist2pt (vx i, vy i) (vx j, vy j)

/-- A field-plane pair is a *unit* pair iff its squared distance is the rational `144` (`= 12²`),
    i.e. the `√3 / √11 / √33` components all vanish. -/
def isUnitQ (d : Qf) : Prop := d = ⟨144, 0, 0, 0⟩

instance : DecidablePred isUnitQ := fun d => by unfold isUnitQ; infer_instance

/-! ## 3. The spindle graph: 7 vertices, 11 unit edges. -/

/-- The 11 unit edges of the Moser spindle (computed: exactly the pairs with `dist2 = 144`). -/
def edgesL : List (Fin 7 × Fin 7) :=
  [(0,1),(0,3),(0,4),(0,6),(1,2),(1,3),(2,3),(2,5),(4,5),(4,6),(5,6)]

/-- The spindle as an abstract `Graph` on `Fin 7`. -/
def spindleG : Graph 7 := ⟨edgesL⟩

/-- **Geometry leg.** Every one of the 11 edges has squared distance exactly the rational `144`
    — all surd components cancel. Verified by `decide` over the integer 4-tuples. -/
theorem edges_unit : ∀ e ∈ edgesL, dist2 e.1 e.2 = ⟨144, 0, 0, 0⟩ := by decide

/-- The long diagonals `A–C`, `A–F` are *not* unit (`dist² = 432 = 144·3`): a sanity check that
    the `isUnitQ` test is discriminating, not vacuous. -/
theorem long_diagonals_not_unit :
    dist2 0 2 = ⟨432, 0, 0, 0⟩ ∧ dist2 0 5 = ⟨432, 0, 0, 0⟩ := by decide

/-! ## 4. Chromatic number: `χ(spindle) = 4`. -/

/-- **Lower bound.** The spindle is **not** 3-colourable. The 11 edge-disequalities over `Fin 3`
    contradict each other; `omega` refutes the resulting integer system (no `native_decide`). -/
theorem spindle_not_3_colourable : ¬ Graph.Colourable spindleG 3 := by
  rintro ⟨c, hc⟩
  have e01 : c 0 ≠ c 1 := hc (0,1) (by decide)
  have e03 : c 0 ≠ c 3 := hc (0,3) (by decide)
  have e04 : c 0 ≠ c 4 := hc (0,4) (by decide)
  have e06 : c 0 ≠ c 6 := hc (0,6) (by decide)
  have e12 : c 1 ≠ c 2 := hc (1,2) (by decide)
  have e13 : c 1 ≠ c 3 := hc (1,3) (by decide)
  have e23 : c 2 ≠ c 3 := hc (2,3) (by decide)
  have e25 : c 2 ≠ c 5 := hc (2,5) (by decide)
  have e45 : c 4 ≠ c 5 := hc (4,5) (by decide)
  have e46 : c 4 ≠ c 6 := hc (4,6) (by decide)
  have e56 : c 5 ≠ c 6 := hc (5,6) (by decide)
  have b0 := (c 0).isLt; have b1 := (c 1).isLt; have b2 := (c 2).isLt
  have b3 := (c 3).isLt; have b4 := (c 4).isLt; have b5 := (c 5).isLt; have b6 := (c 6).isLt
  have v01 : (c 0).val ≠ (c 1).val := fun h => e01 (Fin.eq_of_val_eq h)
  have v03 : (c 0).val ≠ (c 3).val := fun h => e03 (Fin.eq_of_val_eq h)
  have v04 : (c 0).val ≠ (c 4).val := fun h => e04 (Fin.eq_of_val_eq h)
  have v06 : (c 0).val ≠ (c 6).val := fun h => e06 (Fin.eq_of_val_eq h)
  have v12 : (c 1).val ≠ (c 2).val := fun h => e12 (Fin.eq_of_val_eq h)
  have v13 : (c 1).val ≠ (c 3).val := fun h => e13 (Fin.eq_of_val_eq h)
  have v23 : (c 2).val ≠ (c 3).val := fun h => e23 (Fin.eq_of_val_eq h)
  have v25 : (c 2).val ≠ (c 5).val := fun h => e25 (Fin.eq_of_val_eq h)
  have v45 : (c 4).val ≠ (c 5).val := fun h => e45 (Fin.eq_of_val_eq h)
  have v46 : (c 4).val ≠ (c 6).val := fun h => e46 (Fin.eq_of_val_eq h)
  have v56 : (c 5).val ≠ (c 6).val := fun h => e56 (Fin.eq_of_val_eq h)
  omega

/-- A proper 4-colouring of the spindle. -/
def col4 : Fin 7 → Fin 4
  | 0 => 0 | 1 => 1 | 2 => 2 | 3 => 3 | 4 => 1 | 5 => 0 | 6 => 2

/-- **Upper bound.** The spindle is 4-colourable (`χ ≤ 4`). -/
theorem spindle_4_colourable : Graph.Colourable spindleG 4 := by
  refine ⟨col4, ?_⟩
  show ∀ e ∈ edgesL, col4 e.1 ≠ col4 e.2
  decide

/-! ## 5. Transfer: the `ℚ(√3,√11)`-plane needs at least 4 colours. -/

/-- The unit relation of the `ℚ(√3,√11)`-plane: two points are *adjacent* iff their exact squared
    distance is the rational `1` (here `144` after the `×12` scaling). -/
def unitFP (p q : Qf × Qf) : Prop := isUnitQ (dist2pt p q)

/-- The spindle embedding into the field plane. -/
def emb : Fin 7 → Qf × Qf := fun i => (vx i, vy i)

/-- Each edge of the spindle maps to a unit pair of the field plane. -/
theorem emb_unit : ∀ e ∈ spindleG.edges, unitFP (emb e.1) (emb e.2) := by
  intro e he
  have := edges_unit e he
  show isUnitQ (dist2pt (emb e.1) (emb e.2))
  unfold isUnitQ emb
  exact this

/-- **Madore base case (χ ≥ 4 over `ℚ(√3,√11)`).** The unit-distance graph of the `ℚ(√3,√11)`-plane
    has **no proper 3-colouring** — its chromatic number is `≥ 4`. Pure logic on top of the
    geometry (`emb_unit`) and the SAT leg (`spindle_not_3_colourable`). -/
theorem q311_plane_needs_4_colours :
    ¬ Nonempty (PlaneColouring (Qf × Qf) unitFP 3) :=
  not_colourable_implies_plane_chromatic_gt spindleG unitFP emb emb_unit spindle_not_3_colourable

/-! ## 6. Cross-check: the basis `{1,√3,√11,√33}` is ℚ-linearly independent.

The geometry above is purely symbolic (integer 4-tuples). It models *genuine* points of
`ℚ(√3,√11) ⊂ ℝ` precisely because the four basis radicals `{1, √3, √11, √33}` are
ℚ-linearly independent — so distinct 4-tuples are distinct reals and the surd-cancellation
`dist² = ⟨144,0,0,0⟩` is the *real* statement `dist² = 144`. That independence is
`Multiquad.indep_3_11` (proved in `SounioMultiquadParam` by bridging to the de Grey `indep8`
restricted to the masks `{0,1,8,9} = {1,√3,√11,√33}`). -/

/-- Re-export of the |S|=2 independence underpinning the faithfulness of the embedding. -/
theorem basis_independent :
    SounioSqrt.RealCauchyField.Multiquad.IndepMultiquad [3, 11] :=
  SounioSqrt.RealCauchyField.Multiquad.indep_3_11

/-! ## 7. Corollary for `ℝ²`.

The exact field-plane bound `q311_plane_needs_4_colours` is the Madore base case over
`Qf × Qf`.  The ring homomorphism `phi311 : Qf → Real` (`E_mul311` in
`SounioMoserSpindleQ311Real`) embeds the spindle into `Real × Real`; spindle edges have
`dist² = 144` at the native `×12` integer scale (`MoserSpindleQ311.RealPlane.chi_R2_ge_4`).
Rescaling the plane by `1/12` turns those into Euclidean unit edges, so χ(ℝ²) ≥ 4.
Showcase entry point: `MadoreSpindleVitrine`. -/

#print axioms edges_unit
#print axioms spindle_not_3_colourable
#print axioms spindle_4_colourable
#print axioms q311_plane_needs_4_colours
#print axioms basis_independent

end MoserSpindleQ311

