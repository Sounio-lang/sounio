import Init.Data.Rat.Lemmas
import SounioMoserSpindleQ311Real

set_option maxHeartbeats 1000000

/-!
# SounioMoserSpindleQ311EuclideanGeometry

Geometry-only adapter for the Madore/Moser `{3,11}` base case.

The existing `SounioMoserSpindleQ311Real` file already proves the chromatic
lower bound over the Mathlib-free real plane.  This file packages the same
normalized seven-point embedding in the Nat-edge geometry interface used by
future generated candidates.  It also closes the finite zero-distance separation
check needed for the full `EuclideanNatEdgeExactGeometry` object.
-/

namespace MoserSpindleQ311.RealPlane

open MoserSpindleQ311
open UnitDistanceChromatic
open SounioSqrt.RealCauchyField
open SounioSqrt.RealCauchyField.Multiquad

/-! ## Real scalar package for exact squared-distance geometry -/

/-- The Mathlib-free Cauchy-quotient reals satisfy the local scalar laws required by
`ExactSquaredDistancePlane`. -/
noncomputable def realExactFieldLike : ExactFieldLike Real where
  zero := zeroR'
  one := oneR
  add := addR
  neg := negR
  sub := subR
  mul := mulR
  inv := invR
  ofNat := fun n => qR (n : Rat)
  add_assoc := addR_assoc
  add_comm := addR_comm
  zero_add := by intro a; rw [addR_comm, addR_zero]
  add_zero := addR_zero
  add_left_neg := by intro a; rw [addR_comm, addR_neg]
  sub_eq_add_neg := by intro _a _b; rfl
  mul_assoc := mulR_assoc
  mul_comm := mulR_comm
  one_mul := by intro a; rw [mulR_comm, mulR_one]
  mul_one := mulR_one
  left_distrib := leftDistribR
  right_distrib := rightDistribR
  zero_ne_one := zeroNeOneR
  inv_mul_cancel := by
    intro a ha
    rw [mulR_comm]
    exact mulR_inv a ha
  ofNat_zero := by simpa using qR_zero
  ofNat_one := by simpa using qR_one
  ofNat_add := by
    intro m n
    rw [qR_add]
    congr 1
    exact Rat.natCast_add m n
  ofNat_mul := by
    intro m n
    rw [qR_mul]
    congr 1
    exact Rat.natCast_mul m n
  ofNat_inj := by
    intro m n h
    have hr : (m : Rat) = (n : Rat) := qR_inj h
    exact Rat.natCast_inj.mp hr

/-! ## Normalized finite Moser geometry -/

/-- The seven normalized real-plane Moser points, indexed as finite vertices. -/
noncomputable def moserQ311Point (i : Fin 7) : Real × Real :=
  embRealParamUnit i

/-- Exact squared distance between normalized Moser vertices. -/
noncomputable def moserQ311Dist2 (i j : Fin 7) : Real :=
  dist2Real (moserQ311Point i) (moserQ311Point j)

/-- Unit relation induced by normalized squared distance `1` on the seven sampled points. -/
noncomputable def moserQ311Unit (i j : Fin 7) : Prop :=
  moserQ311Dist2 i j = oneR

/-- Nat-indexed embedding used by generated-candidate APIs. Only vertices `< 7`
matter to the contract; out-of-range values are sent to `0`. -/
def moserQ311NatEmb (v : Nat) : Fin 7 :=
  if h : v < 7 then ⟨v, h⟩ else ⟨0, by decide⟩

/-- The Moser spindle edge list as raw Nat pairs, matching `edgesL`. -/
def moserQ311NatEdges : List (Nat × Nat) :=
  [(0, 1), (0, 3), (0, 4), (0, 6), (1, 2), (1, 3),
   (2, 3), (2, 5), (4, 5), (4, 6), (5, 6)]

theorem moserQ311NatEmb_injective :
    ∀ {i j}, i < 7 → j < 7 → moserQ311NatEmb i = moserQ311NatEmb j → i = j := by
  intro _i _j hi hj h
  simp [moserQ311NatEmb, hi, hj] at h
  exact h

theorem moserQ311NatEndpoints :
    ∀ e ∈ moserQ311NatEdges, e.1 < 7 ∧ e.2 < 7 := by
  intro e he
  simp [moserQ311NatEdges] at he
  rcases he with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    exact ⟨by decide, by decide⟩

theorem moserQ311NatUnitEdges :
    ∀ e ∈ moserQ311NatEdges, moserQ311Unit (moserQ311NatEmb e.1) (moserQ311NatEmb e.2) := by
  intro e he
  simp [moserQ311NatEdges] at he
  rcases he with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    first
      | simpa [moserQ311Unit, moserQ311Dist2, moserQ311Point, moserQ311NatEmb, unitReal1,
          qR_one] using embRealParamUnit_unit1 (e := ((0 : Fin 7), (1 : Fin 7))) (by decide)
      | simpa [moserQ311Unit, moserQ311Dist2, moserQ311Point, moserQ311NatEmb, unitReal1,
          qR_one] using embRealParamUnit_unit1 (e := ((0 : Fin 7), (3 : Fin 7))) (by decide)
      | simpa [moserQ311Unit, moserQ311Dist2, moserQ311Point, moserQ311NatEmb, unitReal1,
          qR_one] using embRealParamUnit_unit1 (e := ((0 : Fin 7), (4 : Fin 7))) (by decide)
      | simpa [moserQ311Unit, moserQ311Dist2, moserQ311Point, moserQ311NatEmb, unitReal1,
          qR_one] using embRealParamUnit_unit1 (e := ((0 : Fin 7), (6 : Fin 7))) (by decide)
      | simpa [moserQ311Unit, moserQ311Dist2, moserQ311Point, moserQ311NatEmb, unitReal1,
          qR_one] using embRealParamUnit_unit1 (e := ((1 : Fin 7), (2 : Fin 7))) (by decide)
      | simpa [moserQ311Unit, moserQ311Dist2, moserQ311Point, moserQ311NatEmb, unitReal1,
          qR_one] using embRealParamUnit_unit1 (e := ((1 : Fin 7), (3 : Fin 7))) (by decide)
      | simpa [moserQ311Unit, moserQ311Dist2, moserQ311Point, moserQ311NatEmb, unitReal1,
          qR_one] using embRealParamUnit_unit1 (e := ((2 : Fin 7), (3 : Fin 7))) (by decide)
      | simpa [moserQ311Unit, moserQ311Dist2, moserQ311Point, moserQ311NatEmb, unitReal1,
          qR_one] using embRealParamUnit_unit1 (e := ((2 : Fin 7), (5 : Fin 7))) (by decide)
      | simpa [moserQ311Unit, moserQ311Dist2, moserQ311Point, moserQ311NatEmb, unitReal1,
          qR_one] using embRealParamUnit_unit1 (e := ((4 : Fin 7), (5 : Fin 7))) (by decide)
      | simpa [moserQ311Unit, moserQ311Dist2, moserQ311Point, moserQ311NatEmb, unitReal1,
          qR_one] using embRealParamUnit_unit1 (e := ((4 : Fin 7), (6 : Fin 7))) (by decide)
      | simpa [moserQ311Unit, moserQ311Dist2, moserQ311Point, moserQ311NatEmb, unitReal1,
          qR_one] using embRealParamUnit_unit1 (e := ((5 : Fin 7), (6 : Fin 7))) (by decide)

/-- Nat-edge exact geometry package for the normalized Moser spindle. This is geometry only:
the chromatic no-3 proof remains the existing `spindle_not_3_colourable`, and no no-5/χ≥6
claim is attached here. -/
noncomputable def moserQ311NatExactGeometry :
    NatEdgeExactGeometry 7 (Fin 7) moserQ311Unit where
  edges := moserQ311NatEdges
  emb := moserQ311NatEmb
  emb_injective := moserQ311NatEmb_injective
  endpoints := moserQ311NatEndpoints
  unit_edges := moserQ311NatUnitEdges

/-! ## Metric facts and the one remaining Euclidean-plane obligation -/

private theorem negR_negR_local (x : Real) : negR (negR x) = x := by
  refine Quotient.inductionOn x (fun a => ?_)
  show mkR (negC (negC a)) = mkR a
  exact mk_eq_of_seq_eq (fun n => Rat.neg_neg _)

private theorem subR_swap_neg (x y : Real) : subR y x = negR (subR x y) := by
  unfold subR
  rw [negR_addR, negR_negR_local, addR_comm]

private theorem mulR_negR_right_moser (x y : Real) : mulR x (negR y) = negR (mulR x y) := by
  rw [mulR_comm x (negR y), ← negR_mulR_left y x, mulR_comm y x]

private theorem sqR_negR (x : Real) : sqR (negR x) = sqR x := by
  unfold sqR
  rw [← negR_mulR_left x (negR x), mulR_negR_right_moser, negR_negR_local]

theorem dist2Real_symm (p q : Real × Real) : dist2Real p q = dist2Real q p := by
  unfold dist2Real
  rw [subR_swap_neg p.1 q.1, subR_swap_neg p.2 q.2, sqR_negR, sqR_negR]

theorem moserQ311Dist2_symm (i j : Fin 7) : moserQ311Dist2 i j = moserQ311Dist2 j i := by
  exact dist2Real_symm (moserQ311Point i) (moserQ311Point j)

theorem moserQ311Unit_symm (i j : Fin 7) : moserQ311Unit i j → moserQ311Unit j i := by
  intro h
  unfold moserQ311Unit
  rw [← moserQ311Dist2_symm]
  exact h

private theorem mulR_zero_right_moser (x : Real) : mulR x zeroR' = zeroR' := by
  rw [mulR_comm, zeroR'_mul]

private theorem subR_self_zero (x : Real) : subR x x = zeroR' := by
  unfold subR
  exact addR_neg x

private theorem sqR_zero : sqR zeroR' = zeroR' := by
  unfold sqR
  exact zeroR'_mul zeroR'

theorem dist2Real_self (p : Real × Real) : dist2Real p p = zeroR' := by
  unfold dist2Real
  rw [subR_self_zero, subR_self_zero]
  simp only [sqR_zero, addR_zero]

theorem moserQ311Dist2_self (i : Fin 7) : moserQ311Dist2 i i = zeroR' := by
  exact dist2Real_self (moserQ311Point i)

theorem moserQ311Unit_irrefl (i : Fin 7) : ¬ moserQ311Unit i i := by
  intro h
  unfold moserQ311Unit at h
  rw [moserQ311Dist2_self] at h
  exact zeroNeOneR h

/-- Exact finite separation in the symbolic field-plane model. -/
theorem dist2_qf_zero_iff_eq :
    ∀ i j : Fin 7, dist2 i j = zeroQf ↔ i = j := by
  decide

/-- The square of the global `1/12` unit scale is nonzero. -/
theorem unitScale_sq_ne_zero : mulR unitScale unitScale ≠ zeroR' := by
  intro h
  have hbad : qR (1 : Rat) = zeroR' := by
    rw [← unitScale_sq_144, h, zeroR'_mul]
  have hrat : (1 : Rat) = 0 := by
    apply qR_inj
    rw [hbad, qR_zero]
  exact (by native_decide : ¬ ((1 : Rat) = 0)) hrat

/-- Cancel a nonzero left factor from a product equal to zero. -/
theorem mulR_eq_zero_of_left_ne_zero {a b : Real}
    (ha : a ≠ zeroR') (h : mulR a b = zeroR') : b = zeroR' := by
  calc
    b = mulR oneR b := by rw [mulR_comm, mulR_one]
    _ = mulR (mulR (invR a) a) b := by rw [mulR_comm (invR a) a, mulR_inv a ha]
    _ = mulR (invR a) (mulR a b) := by rw [mulR_assoc]
    _ = mulR (invR a) zeroR' := by rw [h]
    _ = zeroR' := by rw [mulR_zero_right_moser]

/-- Normalized real squared distance is the native Q311 squared distance multiplied by
the square of the global unit scale. -/
theorem moserQ311Dist2_eq_scaled_phi311Param (i j : Fin 7) :
    moserQ311Dist2 i j =
      mulR (mulR unitScale unitScale) (phi311Param (dist2 i j)) := by
  unfold moserQ311Dist2 moserQ311Point embRealParamUnit
  rw [dist2Real_scalePoint, dist2Real_emb_param]

/-- The finite metric fact upgrading the Nat-edge package to the stricter Euclidean geometry
interface: zero squared distance separates the seven normalized `{3,11}` coordinates.

Proof uses:
- `moserQ311NatExactGeometry` (the Nat-edge faithful embedding for the spindle's unit graph)
- `tabela finita de dist2 (decide)`: `dist2_qf_zero_iff_eq` (the 7×7 table over the parametric Qf model in the evalS [3,11] field, proved by `decide`)
- `qR_inj` (in `ofNat_inj`, `phi311Param_eq_zero_iff_qf_zero` lifts, `unitScale_sq_ne_zero` etc to connect Qf zeros to Real)
- scaling by the global unit (unitScale^2 * phi311Param(distQf)) + mul-nonzero lemma to reduce Real dist2==0 to the Qf table.
- reverse direction by self-distance = 0.
This is the incondicional Mathlib-free closure for the embedding faithfulness (distinct points for distinct Fin 7 verts). -/
def MoserQ311Dist2ZeroSeparatesPoints : Prop :=
  ∀ i j : Fin 7, moserQ311Dist2 i j = zeroR' ↔ i = j

/-- Closed proof of the finite zero-distance separation obligation (NatExact + finite dist2 decide table + qR_inj). -/
theorem moserQ311Dist2_zero_separates : MoserQ311Dist2ZeroSeparatesPoints := by
  intro i j
  constructor
  · intro h
    rw [moserQ311Dist2_eq_scaled_phi311Param i j] at h
    have hPhi : phi311Param (dist2 i j) = zeroR' :=
      mulR_eq_zero_of_left_ne_zero unitScale_sq_ne_zero h
    have hQ : dist2 i j = zeroQf :=
      (phi311Param_eq_zero_iff_qf_zero (dist2 i j)).mp hPhi
    exact (dist2_qf_zero_iff_eq i j).mp hQ
  · intro h
    subst h
    exact moserQ311Dist2_self i

/-- All fields of the exact squared-distance plane except the explicit finite
zero-distance/equality proof are already available. -/
noncomputable def moserQ311ExactSquaredDistancePlane_of_zero_separates
    (hzero : MoserQ311Dist2ZeroSeparatesPoints) :
    ExactSquaredDistancePlane (Fin 7) moserQ311Unit where
  Scalar := Real
  scalar := realExactFieldLike
  x := fun i => (moserQ311Point i).1
  y := fun i => (moserQ311Point i).2
  dist2 := moserQ311Dist2
  dist2_formula := by
    intro _i _j
    unfold moserQ311Dist2 moserQ311Point dist2Real sqR subR
    rfl
  unit_iff_dist2_eq_one := by
    intro _i _j
    rfl
  dist2_zero_iff_eq := hzero
  unit_symm := moserQ311Unit_symm
  unit_irrefl := moserQ311Unit_irrefl

/-- The full Euclidean geometry object, isolated behind the one remaining finite
zero-distance/equality proof. -/
noncomputable def moserQ311EuclideanGeometry_of_zero_separates
    (hzero : MoserQ311Dist2ZeroSeparatesPoints) :
    EuclideanNatEdgeExactGeometry 7 (Fin 7) moserQ311Unit where
  exact := moserQ311NatExactGeometry
  plane := moserQ311ExactSquaredDistancePlane_of_zero_separates hzero

/-- Exact squared-distance plane for the normalized Madore/Moser Q311 embedding. -/
noncomputable def moserQ311ExactSquaredDistancePlane :
    ExactSquaredDistancePlane (Fin 7) moserQ311Unit :=
  moserQ311ExactSquaredDistancePlane_of_zero_separates moserQ311Dist2_zero_separates

/-- Full Euclidean geometry object for the normalized Madore/Moser Q311 embedding. -/
noncomputable def moserQ311EuclideanGeometry :
    EuclideanNatEdgeExactGeometry 7 (Fin 7) moserQ311Unit :=
  moserQ311EuclideanGeometry_of_zero_separates moserQ311Dist2_zero_separates

#check realExactFieldLike
#check moserQ311NatExactGeometry
#check MoserQ311Dist2ZeroSeparatesPoints
#check moserQ311Dist2_zero_separates
#check moserQ311ExactSquaredDistancePlane_of_zero_separates
#check moserQ311EuclideanGeometry_of_zero_separates
#check moserQ311ExactSquaredDistancePlane
#check moserQ311EuclideanGeometry

#print axioms realExactFieldLike
#print axioms moserQ311NatExactGeometry
#print axioms moserQ311Unit_symm
#print axioms moserQ311Unit_irrefl
#print axioms moserQ311Dist2_zero_separates
#print axioms moserQ311EuclideanGeometry_of_zero_separates
#print axioms moserQ311EuclideanGeometry

#eval IO.println "SounioMoserSpindleQ311EuclideanGeometry: Moser/Q311 Nat-edge exact geometry plus full Euclidean adapter packaged; finite zero-distance separation closed."

end MoserSpindleQ311.RealPlane
