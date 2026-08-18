import EpistemicEffects
open Sounio.EpistemicEffects

/-
  Positive control for V2 consumption. This is the measure-Nat witness
  written against the *refuted* calculus. V1's `kraw` has no payload
  slot (`KCell → Expr`, always `Knowledge<Real>`), so the statement
  cannot elaborate. If `lake env lean` on this file exits 0, the
  consumer arm is measuring mention, not use, and must not be added.
-/

theorem measure_nat_reduct_stays_know_nat
    (m : KMeta) (hm : kvalid m) :
    HasTy [] (.measure (.lit_nat 0) m) (.tknow .tnat) (singleE .eObserve)
    ∧ ((.measure (.lit_nat 0) m) ⇒ (.kraw (.lit_nat 0) m))
    ∧ HasTy [] (.kraw (.lit_nat 0) m) (.tknow .tnat) emptyE
