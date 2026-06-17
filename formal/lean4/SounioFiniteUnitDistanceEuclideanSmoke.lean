import SounioFiniteUnitDistanceWitness
import Init.Data.Rat.Lemmas

/-!
# SounioFiniteUnitDistanceEuclideanSmoke — Euclidean geometry contract smoke

This file is a small calibration gate for the chi>=6 candidate interface.
It inhabits `EuclideanNatEdgeExactGeometry` with honest two-coordinate
squared-distance relations over `Rat`, first for the single unit edge
`(0,0) -- (1,0)` and then for the four-edge unit square.

It deliberately does not attach a no-5-colouring certificate and does not claim
any chromatic lower bound. Its only purpose is to prove that the future
candidate geometry contract is a real exact-Euclidean Lean object, not merely a
manifest string check.
-/

namespace UnitDistanceChromatic
namespace Chi6EuclideanGeometrySmoke

/-- A tiny exact scalar-law package over `Rat`, used only to smoke-test the
Euclidean chi>=6 geometry contract on concrete coordinates. -/
def ratExactFieldLike : ExactFieldLike Rat where
  zero := 0
  one := 1
  add := (· + ·)
  neg := Neg.neg
  sub := (· - ·)
  mul := (· * ·)
  inv := Inv.inv
  ofNat := fun n => (n : Rat)
  add_assoc := Rat.add_assoc
  add_comm := Rat.add_comm
  zero_add := Rat.zero_add
  add_zero := Rat.add_zero
  add_left_neg := by intro a; grind
  sub_eq_add_neg := Rat.sub_eq_add_neg
  mul_assoc := Rat.mul_assoc
  mul_comm := Rat.mul_comm
  one_mul := Rat.one_mul
  mul_one := Rat.mul_one
  left_distrib := Rat.mul_add
  right_distrib := Rat.add_mul
  zero_ne_one := by native_decide
  inv_mul_cancel := Rat.inv_mul_cancel
  ofNat_zero := rfl
  ofNat_one := rfl
  ofNat_add := Rat.natCast_add
  ofNat_mul := Rat.natCast_mul
  ofNat_inj := by
    intro m n h
    exact Rat.natCast_inj.mp h

/-- Two exact rational points: vertex `0` is `(0,0)`, vertex `1` is `(1,0)`. -/
def twoPointX (p : Fin 2) : Rat := if p.val = 0 then 0 else 1

/-- Both smoke points live on the rational x-axis. -/
def twoPointY (_p : Fin 2) : Rat := 0

/-- Exact squared Euclidean distance for the two-point rational smoke plane. -/
def twoPointDist2 (p q : Fin 2) : Rat :=
  ((twoPointX p - twoPointX q) * (twoPointX p - twoPointX q)) +
    ((twoPointY p - twoPointY q) * (twoPointY p - twoPointY q))

/-- Unit relation induced by exact squared distance equal to one. -/
def twoPointUnit (p q : Fin 2) : Prop := twoPointDist2 p q = 1

theorem twoPointDist2_zero_iff_eq (p q : Fin 2) : twoPointDist2 p q = 0 ↔ p = q := by
  cases p with
  | mk pv hp =>
    cases q with
    | mk qv hq =>
      have hp_cases : pv = 0 ∨ pv = 1 := by omega
      have hq_cases : qv = 0 ∨ qv = 1 := by omega
      rcases hp_cases with rfl | rfl <;> rcases hq_cases with rfl | rfl <;>
        simp [twoPointDist2, twoPointX, twoPointY] <;> native_decide

theorem twoPointUnit_symm (p q : Fin 2) : twoPointUnit p q → twoPointUnit q p := by
  cases p with
  | mk pv hp =>
    cases q with
    | mk qv hq =>
      have hp_cases : pv = 0 ∨ pv = 1 := by omega
      have hq_cases : qv = 0 ∨ qv = 1 := by omega
      rcases hp_cases with rfl | rfl <;> rcases hq_cases with rfl | rfl <;>
        simp [twoPointUnit, twoPointDist2, twoPointX, twoPointY] <;> native_decide

theorem twoPointUnit_irrefl (p : Fin 2) : ¬ twoPointUnit p p := by
  cases p with
  | mk pv hp =>
    have hp_cases : pv = 0 ∨ pv = 1 := by omega
    rcases hp_cases with rfl | rfl <;>
      simp [twoPointUnit, twoPointDist2, twoPointX, twoPointY] <;> native_decide

/-- Exact squared-distance plane for the rational two-point smoke geometry. -/
def twoPointPlane : ExactSquaredDistancePlane (Fin 2) twoPointUnit where
  Scalar := Rat
  scalar := ratExactFieldLike
  x := twoPointX
  y := twoPointY
  dist2 := twoPointDist2
  dist2_formula := by intro _p _q; rfl
  unit_iff_dist2_eq_one := by intro _p _q; rfl
  dist2_zero_iff_eq := twoPointDist2_zero_iff_eq
  unit_symm := twoPointUnit_symm
  unit_irrefl := twoPointUnit_irrefl

/-- Nat-indexed embedding used by generated-candidate APIs. Only vertices `< 2`
matter to the contract; out-of-range values are sent to `0`. -/
def twoPointEmb (v : Nat) : Fin 2 := if h : v < 2 then ⟨v, h⟩ else ⟨0, by decide⟩

theorem twoPointEmb_injective :
    ∀ {i j}, i < 2 → j < 2 → twoPointEmb i = twoPointEmb j → i = j := by
  intro _i _j hi hj h
  simp [twoPointEmb, hi, hj] at h
  exact h

/-- One listed unit edge, using the same Nat-edge shape as generated candidates. -/
def twoPointEdges : List (Nat × Nat) := [(0, 1)]

theorem twoPointEndpoints : ∀ e ∈ twoPointEdges, e.1 < 2 ∧ e.2 < 2 := by
  intro e he
  simp [twoPointEdges] at he
  subst e
  exact ⟨by decide, by decide⟩

theorem twoPointUnitEdges :
    ∀ e ∈ twoPointEdges, twoPointUnit (twoPointEmb e.1) (twoPointEmb e.2) := by
  intro e he
  simp [twoPointEdges] at he
  subst e
  simp [twoPointEmb, twoPointUnit, twoPointDist2, twoPointX, twoPointY]
  native_decide

/-- Geometry-only exact Nat-edge package for the unit segment smoke. -/
def twoPointExactGeometry : NatEdgeExactGeometry 2 (Fin 2) twoPointUnit where
  edges := twoPointEdges
  emb := twoPointEmb
  emb_injective := twoPointEmb_injective
  endpoints := twoPointEndpoints
  unit_edges := twoPointUnitEdges

/-- Euclidean squared-distance geometry package for the unit segment smoke.
No SAT/no-colouring certificate is attached here. -/
def twoPointEuclideanGeometry : EuclideanNatEdgeExactGeometry 2 (Fin 2) twoPointUnit where
  exact := twoPointExactGeometry
  plane := twoPointPlane

/-- Contract calibration theorem: the Euclidean chi>=6 geometry interface is inhabited
by a concrete exact squared-distance object over `Rat`. -/
theorem twoPointGeometryHasEuclideanContract :
    ∃ G : EuclideanNatEdgeExactGeometry 2 (Fin 2) twoPointUnit, G.exact.edges = [(0, 1)] :=
  ⟨twoPointEuclideanGeometry, rfl⟩

/-- The unit square vertices in cyclic order:
`0=(0,0)`, `1=(1,0)`, `2=(1,1)`, `3=(0,1)`. -/
def squarePointX (p : Fin 4) : Rat :=
  if p.val = 0 then 0 else if p.val = 1 then 1 else if p.val = 2 then 1 else 0

/-- The unit square y-coordinates in cyclic order. -/
def squarePointY (p : Fin 4) : Rat :=
  if p.val = 0 then 0 else if p.val = 1 then 0 else if p.val = 2 then 1 else 1

/-- Exact squared Euclidean distance for the rational unit-square smoke plane. -/
def squareDist2 (p q : Fin 4) : Rat :=
  ((squarePointX p - squarePointX q) * (squarePointX p - squarePointX q)) +
    ((squarePointY p - squarePointY q) * (squarePointY p - squarePointY q))

/-- Unit relation induced by exact squared distance equal to one on the square vertices. -/
def squareUnit (p q : Fin 4) : Prop := squareDist2 p q = 1

theorem squareDist2_zero_iff_eq (p q : Fin 4) : squareDist2 p q = 0 ↔ p = q := by
  cases p with
  | mk pv hp =>
    cases q with
    | mk qv hq =>
      have hp_cases : pv = 0 ∨ pv = 1 ∨ pv = 2 ∨ pv = 3 := by omega
      have hq_cases : qv = 0 ∨ qv = 1 ∨ qv = 2 ∨ qv = 3 := by omega
      rcases hp_cases with rfl | rfl | rfl | rfl <;>
        rcases hq_cases with rfl | rfl | rfl | rfl <;>
        simp [squareDist2, squarePointX, squarePointY] <;> native_decide

theorem squareUnit_symm (p q : Fin 4) : squareUnit p q → squareUnit q p := by
  cases p with
  | mk pv hp =>
    cases q with
    | mk qv hq =>
      have hp_cases : pv = 0 ∨ pv = 1 ∨ pv = 2 ∨ pv = 3 := by omega
      have hq_cases : qv = 0 ∨ qv = 1 ∨ qv = 2 ∨ qv = 3 := by omega
      rcases hp_cases with rfl | rfl | rfl | rfl <;>
        rcases hq_cases with rfl | rfl | rfl | rfl <;>
        simp [squareUnit, squareDist2, squarePointX, squarePointY] <;> native_decide

theorem squareUnit_irrefl (p : Fin 4) : ¬ squareUnit p p := by
  cases p with
  | mk pv hp =>
    have hp_cases : pv = 0 ∨ pv = 1 ∨ pv = 2 ∨ pv = 3 := by omega
    rcases hp_cases with rfl | rfl | rfl | rfl <;>
      simp [squareUnit, squareDist2, squarePointX, squarePointY] <;> native_decide

/-- Exact squared-distance plane for the rational unit square smoke geometry. -/
def squarePlane : ExactSquaredDistancePlane (Fin 4) squareUnit where
  Scalar := Rat
  scalar := ratExactFieldLike
  x := squarePointX
  y := squarePointY
  dist2 := squareDist2
  dist2_formula := by intro _p _q; rfl
  unit_iff_dist2_eq_one := by intro _p _q; rfl
  dist2_zero_iff_eq := squareDist2_zero_iff_eq
  unit_symm := squareUnit_symm
  unit_irrefl := squareUnit_irrefl

/-- Nat-indexed embedding for the four square vertices. -/
def squareEmb (v : Nat) : Fin 4 := if h : v < 4 then ⟨v, h⟩ else ⟨0, by decide⟩

theorem squareEmb_injective :
    ∀ {i j}, i < 4 → j < 4 → squareEmb i = squareEmb j → i = j := by
  intro _i _j hi hj h
  simp [squareEmb, hi, hj] at h
  exact h

/-- Four unit edges of the rational square, in cyclic order. -/
def squareEdges : List (Nat × Nat) := [(0, 1), (1, 2), (2, 3), (3, 0)]

theorem squareEndpoints : ∀ e ∈ squareEdges, e.1 < 4 ∧ e.2 < 4 := by
  intro e he
  simp [squareEdges] at he
  rcases he with rfl | rfl | rfl | rfl <;> exact ⟨by decide, by decide⟩

theorem squareUnitEdges :
    ∀ e ∈ squareEdges, squareUnit (squareEmb e.1) (squareEmb e.2) := by
  intro e he
  simp [squareEdges] at he
  rcases he with rfl | rfl | rfl | rfl <;>
    simp [squareEmb, squareUnit, squareDist2, squarePointX, squarePointY] <;> native_decide

/-- Geometry-only exact Nat-edge package for the four-edge unit square smoke. -/
def squareExactGeometry : NatEdgeExactGeometry 4 (Fin 4) squareUnit where
  edges := squareEdges
  emb := squareEmb
  emb_injective := squareEmb_injective
  endpoints := squareEndpoints
  unit_edges := squareUnitEdges

/-- Euclidean squared-distance geometry package for the four-edge unit square smoke.
No SAT/no-colouring certificate is attached here. -/
def squareEuclideanGeometry : EuclideanNatEdgeExactGeometry 4 (Fin 4) squareUnit where
  exact := squareExactGeometry
  plane := squarePlane

/-- Contract calibration theorem for a multi-edge exact Euclidean smoke graph. -/
theorem squareGeometryHasEuclideanContract :
    ∃ G : EuclideanNatEdgeExactGeometry 4 (Fin 4) squareUnit,
      G.exact.edges = [(0, 1), (1, 2), (2, 3), (3, 0)] :=
  ⟨squareEuclideanGeometry, rfl⟩

#print axioms ratExactFieldLike
#print axioms twoPointDist2_zero_iff_eq
#print axioms twoPointUnit_symm
#print axioms twoPointUnit_irrefl
#print axioms twoPointExactGeometry
#print axioms twoPointEuclideanGeometry
#print axioms twoPointGeometryHasEuclideanContract
#print axioms squareDist2_zero_iff_eq
#print axioms squareUnit_symm
#print axioms squareUnit_irrefl
#print axioms squareExactGeometry
#print axioms squareEuclideanGeometry
#print axioms squareGeometryHasEuclideanContract

#eval IO.println "SounioFiniteUnitDistanceEuclideanSmoke: exact Euclidean geometry contract smoke over Rat^2 (unit segment + square); no chi>=6/no-5 claim."

end Chi6EuclideanGeometrySmoke
end UnitDistanceChromatic
