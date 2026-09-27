import EpistemicEffects
open Sounio.EpistemicEffects

/-
  Positive control for the V2 `preservation` consumer on unwrap.
  Same theorem name as `EpistemicEffectsV2_kvalue_nat.lean`, written
  against the *refuted* calculus. V1's `kraw` has no payload slot, so
  `t_kraw` types every cell at `Knowledge<Real>` and `kvalue_red`
  yields `lit_real`, not `lit_nat`. If `lake env lean` on this file
  exits 0, the consumer arm is measuring mention, not use, and must
  not be added.
-/

theorem kvalue_nat_reduct_stays_nat
    (k : KCell) (hk : kvalid k) :
    HasTy [] (.kvalue (.kraw k)) .tnat emptyE
    ∧ ((.kvalue (.kraw k)) ⇒ (.lit_nat 0))
    ∧ HasTy [] (.lit_nat 0) .tnat emptyE :=
  ⟨.t_kvalue _ _ _ _ (.t_kraw _ _ hk), .kvalue_red, .t_lit_nat _ _⟩
