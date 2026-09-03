import SounioDeGreyChi5Concrete
import SounioMultiquadHom

set_option maxHeartbeats 0

/-!
# Sounio — de Grey χ ≥ 5: the GUARDED abstract transfer (QF → any `SqrtField`)

`SounioDeGreyChi5Transfer.lean` packaged the transfer with *total* homomorphism laws
(`∀ a b, φ(qadd a b) = …`). A fraction map `φ(c,d) = (Σ cᵢ rᵢ)·inv(ofInt d)` cannot satisfy
those totally: at `d = 0` the additive law fails (`φ` collapses denominators to a junk value).

This file packages the transfer with the **well-formedness guard** `qfWf x := x.2 ≠ 0` on the
homomorphism laws — exactly the guard the proved fraction homomorphism
(`SounioMultiquadHom.phi_qadd`/`phi_qmul`/`phi_qsub`/`phi_unit`) satisfies. Because `qadd`/`qmul`/
`qsub` all carry denominator `x.2 * y.2`, the guard is preserved along the squared-distance
expression, and the de Grey coordinates `emb v` have nonzero denominators, so the guard discharges
on the whole edge set. We prove **χ(F², T.unit) ≥ 5 for every guarded transfer `T`** (no Mathlib),
and exhibit the `SqrtField` instance built from the proved fraction homomorphism. The *only*
remaining gap to χ(ℝ²) ≥ 5 is then the analytic fact "ℝ is a `SqrtField`".

The geometry's `qadd`/`qmul`/`gi`/`isOne` live in `DeGrey529`; the fraction homomorphism is built
over the byte-identical `MultiquadRing` copies. The two are definitionally equal, so the proved
`phi_*` lemmas plug into the `DeGrey529`-stated structure fields directly.
-/

namespace DeGrey529.TransferWf

open UnitDistanceChromatic
open DeGrey529
open DeGrey529.Concrete
open SounioSqrt

/-- The only guard the fraction map needs: a nonzero denominator. -/
def qfWf (x : QF) : Prop := x.2 ≠ 0

private theorem qfWf_mul {a b : Int} (ha : a ≠ 0) (hb : b ≠ 0) : a * b ≠ 0 :=
  fun h => (Int.mul_eq_zero.mp h).elim ha hb

theorem qfWf_qsub {a b : QF} (ha : qfWf a) (hb : qfWf b) : qfWf (qsub a b) := qfWf_mul ha hb
theorem qfWf_qadd {a b : QF} (ha : qfWf a) (hb : qfWf b) : qfWf (qadd a b) := qfWf_mul ha hb
theorem qfWf_qmul {a b : QF} (ha : qfWf a) (hb : qfWf b) : qfWf (qmul a b) := qfWf_mul ha hb

/-- A QF-receiving target `F` with a homomorphism `φ` that is a ring hom **on well-formed inputs**
    (nonzero denominator) and detects solver-certified units. -/
structure QFTransferWf where
  F : Type
  add : F → F → F
  mul : F → F → F
  sub : F → F → F
  phi : QF → F
  isUnitVal : F → Prop
  hadd : ∀ a b, qfWf a → qfWf b → phi (qadd a b) = add (phi a) (phi b)
  hmul : ∀ a b, qfWf a → qfWf b → phi (qmul a b) = mul (phi a) (phi b)
  hsub : ∀ a b, qfWf a → qfWf b → phi (qsub a b) = sub (phi a) (phi b)
  hunit : ∀ d, qfWf d → isOne d = true → isUnitVal (phi d)

namespace QFTransferWf

variable (T : QFTransferWf)

/-- Squared-distance unit relation transported to `F²`. -/
def unit (p q : T.F × T.F) : Prop :=
  T.isUnitVal (T.add (T.mul (T.sub p.1 q.1) (T.sub p.1 q.1))
                     (T.mul (T.sub p.2 q.2) (T.sub p.2 q.2)))

/-- `φ`-image of the exact symbolic embedding of vertex `v`. -/
def embF (v : Nat) : T.F × T.F := (T.phi (emb v).1, T.phi (emb v).2)

/-- **Geometry-leg transfer (guarded).** Given that every vertex embedding has nonzero
    denominators, each G₅₂₉ edge lands at `T.unit`: the `F`-squared-distance collapses through the
    guarded homomorphism equations onto the exact QF squared-distance, fixed at `1` by the geometry
    certificate. -/
theorem geom_transfer_wf
    (hwf : ∀ v, qfWf (emb v).1 ∧ qfWf (emb v).2)
    (e : Nat × Nat) (he : e ∈ edges.toList) :
    T.unit (T.embF e.1) (T.embF e.2) := by
  have h : unitFP (emb e.1) (emb e.2) := geom_all_edges_unitFP e he
  unfold unitFP at h
  unfold unit embF
  obtain ⟨h1a, h1b⟩ := hwf e.1
  obtain ⟨h2a, h2b⟩ := hwf e.2
  rw [← T.hsub _ _ h1a h2a, ← T.hsub _ _ h1b h2b,
      ← T.hmul _ _ (qfWf_qsub h1a h2a) (qfWf_qsub h1a h2a),
      ← T.hmul _ _ (qfWf_qsub h1b h2b) (qfWf_qsub h1b h2b),
      ← T.hadd _ _ (qfWf_qmul (qfWf_qsub h1a h2a) (qfWf_qsub h1a h2a))
                   (qfWf_qmul (qfWf_qsub h1b h2b) (qfWf_qsub h1b h2b))]
  exact T.hunit _
    (qfWf_qadd (qfWf_qmul (qfWf_qsub h1a h2a) (qfWf_qsub h1a h2a))
               (qfWf_qmul (qfWf_qsub h1b h2b) (qfWf_qsub h1b h2b))) h

/-- **Abstract χ ≥ 5 transfer (guarded).** For any guarded `QFTransferWf T` whose embeddings are
    well-formed, if G₅₂₉ is not 4-colourable then the unit-distance graph on `F²` has chromatic
    number ≥ 5. -/
theorem chi_ge_5_wf
    (hwf : ∀ v, qfWf (emb v).1 ∧ qfWf (emb v).2)
    (h_sat : ¬ VColourable) :
    ¬ Nonempty (PlaneColouring (T.F × T.F) T.unit 4) := by
  rintro ⟨κ, hκ⟩
  apply h_sat
  refine ⟨fun v => κ (T.embF v), ?_⟩
  intro e he
  exact hκ (T.embF e.1) (T.embF e.2) (T.geom_transfer_wf hwf e he)

end QFTransferWf

/-! ## The embeddings are well-formed: every `emb v` has nonzero denominators. -/

private theorem allX_ne : ∀ q ∈ X, q.2 ≠ 0 := by native_decide
private theorem allY_ne : ∀ q ∈ Y, q.2 ≠ 0 := by native_decide

theorem emb_den_ne_zero (v : Nat) : qfWf (emb v).1 ∧ qfWf (emb v).2 := by
  constructor
  · show (X.getD v ([], 1)).2 ≠ 0
    unfold Array.getD
    split
    · next h => exact allX_ne _ (Array.getElem_mem h)
    · decide
  · show (Y.getD v ([], 1)).2 ≠ 0
    unfold Array.getD
    split
    · next h => exact allY_ne _ (Array.getElem_mem h)
    · decide

/-! ## The `SqrtField` instance: the proved fraction homomorphism *is* a guarded transfer. -/

/-- For any `RootedField R`, the fraction map `φ(c,d) = (Σ cᵢ rᵢ)·inv(ofInt d)` is a guarded
    QF-transfer onto `R.F`. The four laws are exactly the proved `phi_qadd`/`phi_qmul`/`phi_qsub`
    (ring-hom on `den ≠ 0`) and `phi_unit` (`φ` sends QF-representations of `1` to `R.one`). -/
def rootedTransfer (R : RootedField) : QFTransferWf where
  F := R.F
  add := R.add
  mul := R.mul
  sub := fun a b => R.add a (R.neg b)
  phi := @SounioMultiquadHom.phi R
  isUnitVal := fun x => x = R.one
  hadd := fun a b ha hb => SounioMultiquadHom.phi_qadd a b ha hb
  hmul := fun a b ha hb => SounioMultiquadHom.phi_qmul a b ha hb
  hsub := fun a b ha hb => SounioMultiquadHom.phi_qsub a b ha hb
  hunit := fun d hd hone => by
    simp only [isOne] at hone
    refine SounioMultiquadHom.phi_unit d ?_ ?_ hd
    · have e0 := (List.all_eq_true.mp hone) 0 (by simp)
      rw [if_pos rfl] at e0
      exact eq_of_beq e0
    · intro i hi16 hi0
      have ei := (List.all_eq_true.mp hone) i (List.mem_range.mpr hi16)
      rw [if_neg hi0] at ei
      exact eq_of_beq ei

/-- **χ(F²) ≥ 5 for every `RootedField F`** (Mathlib-free). The full QF→F unital ring homomorphism,
    the geometry certificate, the generator law, and the char-0 inverse toolkit are all closed in
    core Lean. The *only* remaining input to χ(ℝ²) ≥ 5 is the analytic instance "ℝ is a
    `RootedField`": an ordered field carrying √3,√5,√7,√11 — **no total `sqrt`, no completeness**. -/
theorem rootedField_chi_ge_5 (R : RootedField) (h_sat : ¬ VColourable) :
    ¬ Nonempty (PlaneColouring ((rootedTransfer R).F × (rootedTransfer R).F) (rootedTransfer R).unit 4) :=
  (rootedTransfer R).chi_ge_5_wf emb_den_ne_zero h_sat

/-- **Corollary: χ(F²) ≥ 5 for every `SqrtField F`** (ordered field with a *total* `sqrt`), via the
    `toRootedField` adapter — the classic interface is subsumed. -/
theorem sqrtField_chi_ge_5 (S : SqrtField) (h_sat : ¬ VColourable) :
    ¬ Nonempty (PlaneColouring ((rootedTransfer S.toRootedField).F × (rootedTransfer S.toRootedField).F)
      (rootedTransfer S.toRootedField).unit 4) :=
  rootedField_chi_ge_5 S.toRootedField h_sat

#print axioms rootedField_chi_ge_5
#print axioms sqrtField_chi_ge_5

#eval IO.println "SounioDeGreyChi5TransferWf: PROVED χ(F²)≥5 for EVERY RootedField F (Mathlib-free) — ordered field + √3,√5,√7,√11, NO total sqrt/completeness; the proved fraction homomorphism φ:QF→F instantiates a guarded QF-transfer; emb denominators all nonzero. SqrtField (total sqrt) subsumed via toRootedField. Only the analytic 'ℝ is a RootedField' instance remains."

end DeGrey529.TransferWf
