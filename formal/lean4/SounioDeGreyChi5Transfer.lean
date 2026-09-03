import SounioDeGreyChi5Concrete

set_option maxHeartbeats 0

/-!
# Sounio — de Grey χ ≥ 5: the ABSTRACT TRANSFER leg (QF → any receiving ring)

The reduction of `SounioDeGreyChi5.lean` is already polymorphic in the plane type `P`
and its unit relation. This file makes the **algebraic transfer** explicit and isolates
the *only* place the real numbers are needed.

We package the data of "a commutative-ring-like target `F` that receives QF through a
homomorphism" into `QFTransfer`, and prove **once and for all** that

    χ(F², T.unit) ≥ 5     for every `T : QFTransfer`,

with **no Mathlib, no real analysis**. The proof uses *only* the homomorphism equations
(`φ` commutes with `qadd`/`qmul`/`qsub`) and the unit-detection law (`φ` sends any QF value
the solver certified `= 1` to a unit of `F`). No ring axioms of `F` itself are required —
the squared-distance over `F` collapses through `φ` onto the exact QF squared-distance the
`native_decide` geometry certificate already pinned at `1` for every edge.

Two instances close the picture:

* **`qfSelf` (proved here, core Lean).** `F = QF`, `φ = id`, `isUnitVal = isOne · = true`.
  This recovers `g529_field_plane_needs_5_colours` verbatim, so the abstraction is faithful
  and non-vacuous in core Lean.
* **`ℝ` (the single remaining Mathlib step).** `F = ℝ`, `φ` = evaluate the 16-tuple by
  sending `basis m ↦ ∏√pⱼ` (real radicals), `isUnitVal x := x = 1`. Discharging
  `hadd/hmul/hsub` is exactly "`φ` is a ring hom", whose multiplicative core is the real
  image of `SounioMultiquadRing.basis_mul_law` (`√a·√b = √(ab)`); `hunit` is `φ 1 = 1`.
  That instance — and *only* that instance — needs `Real.sqrt` and `ring`, i.e. Mathlib's ℝ.
  Once provided, `chi_ge_5 (ℝ-instance) g529_not_colourable` is **χ(ℝ²) ≥ 5**.
-/

namespace DeGrey529.Transfer

open UnitDistanceChromatic
open DeGrey529
open DeGrey529.Concrete

/-- The data witnessing "`F` receives `QF` through a homomorphism that detects units".
    `isUnitVal x` means "the element `x : F` counts as a unit-squared-distance". -/
structure QFTransfer where
  F : Type
  add : F → F → F
  mul : F → F → F
  sub : F → F → F
  phi : QF → F
  isUnitVal : F → Prop
  hadd : ∀ a b, phi (qadd a b) = add (phi a) (phi b)
  hmul : ∀ a b, phi (qmul a b) = mul (phi a) (phi b)
  hsub : ∀ a b, phi (qsub a b) = sub (phi a) (phi b)
  /-- Solver-certified `= 1` QF values map to `F`-units (for ℝ this is `φ 1 = 1`). -/
  hunit : ∀ d, isOne d = true → isUnitVal (phi d)

namespace QFTransfer

variable (T : QFTransfer)

/-- Squared-distance unit relation transported to `F²`. -/
def unit (p q : T.F × T.F) : Prop :=
  T.isUnitVal (T.add (T.mul (T.sub p.1 q.1) (T.sub p.1 q.1))
                     (T.mul (T.sub p.2 q.2) (T.sub p.2 q.2)))

/-- `φ`-image of the exact symbolic embedding of vertex `v`. -/
def embF (v : Nat) : T.F × T.F := (T.phi (emb v).1, T.phi (emb v).2)

/-- **Geometry-leg transfer (PROVED, no Mathlib).** Every G₅₂₉ edge lands at `T.unit`.
    The `F`-squared-distance collapses through the homomorphism equations onto the exact
    QF squared-distance, which the `native_decide` certificate fixed at `1` for each edge. -/
theorem geom_transfer (e : Nat × Nat) (he : e ∈ edges.toList) :
    T.unit (T.embF e.1) (T.embF e.2) := by
  have h : unitFP (emb e.1) (emb e.2) := geom_all_edges_unitFP e he
  unfold unitFP at h
  unfold unit embF
  simp only [← T.hsub, ← T.hmul, ← T.hadd]
  exact T.hunit _ h

/-- **Abstract χ ≥ 5 transfer.** For any `QFTransfer T`, if G₅₂₉ is not 4-colourable then
    the unit-distance graph on `F²` has no proper 4-colouring — chromatic number ≥ 5. -/
theorem chi_ge_5 (h_sat : ¬ VColourable) :
    ¬ Nonempty (PlaneColouring (T.F × T.F) T.unit 4) := by
  rintro ⟨κ, hκ⟩
  apply h_sat
  refine ⟨fun v => κ (T.embF v), ?_⟩
  intro e he
  exact hκ (T.embF e.1) (T.embF e.2) (T.geom_transfer e he)

end QFTransfer

/-! ## Sanity instance: `F = QF`, `φ = id` — recovers the field-plane result in core Lean. -/

/-- The identity transfer onto the QF field-plane itself. -/
def qfSelf : QFTransfer where
  F := QF
  add := qadd
  mul := qmul
  sub := qsub
  phi := id
  isUnitVal := fun x => isOne x = true
  hadd := fun _ _ => rfl
  hmul := fun _ _ => rfl
  hsub := fun _ _ => rfl
  hunit := fun _ h => h

/-- `qfSelf.unit` is *definitionally* the field-plane relation `unitFP`. -/
theorem qfSelf_unit_eq (p q : FieldPoint) : qfSelf.unit p q = unitFP p q := rfl

/-- The abstract transfer, instantiated at `qfSelf`, **is** the concrete field-plane
    theorem `g529_field_plane_needs_5_colours` — confirming the abstraction is faithful. -/
theorem qfSelf_field_plane_needs_5_colours (h_sat : ¬ VColourable) :
    ¬ Nonempty (PlaneColouring FieldPoint unitFP 4) :=
  qfSelf.chi_ge_5 h_sat

#print axioms QFTransfer.geom_transfer
#print axioms QFTransfer.chi_ge_5
#print axioms qfSelf_field_plane_needs_5_colours

#eval IO.println "SounioDeGreyChi5Transfer: abstract χ≥5 transfer PROVED for any QF-receiving ring (no Mathlib); qfSelf instance recovers the field-plane result; only the ℝ instance (Real.sqrt) remains."

end DeGrey529.Transfer
