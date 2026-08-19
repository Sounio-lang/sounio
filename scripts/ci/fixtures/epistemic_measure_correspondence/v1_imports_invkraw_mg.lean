import EpistemicEffects
open Sounio.EpistemicEffects

/-
  Positive control for the V2 `invKraw` consumer at `Knowledge<mg>`.
  Same theorem name as `EpistemicEffectsV2_invkraw_mg.lean`, written
  against the *refuted* calculus. `tmg` exists on the shared `Ty`
  spine so this file does not fail by a missing constructor. V1's
  `kraw` has no payload slot, so `t_kraw` types every cell at
  `Knowledge<Real>` and `f (kraw k)` with
  `f : Knowledge<mg> → Knowledge<mg>` cannot elaborate — `t_app`
  wants `Knowledge<mg>`, `t_kraw` supplies `Knowledge<Real>`.
  If `lake env lean` on this file exits 0, the consumer arm is
  measuring mention, not use, and must not be added.
-/

theorem kraw_mg_inverts_and_is_usable
    (k : KCell) (hk : kvalid k) :
    HasTy [] (.app (.lam (.tknow .tmg) emptyE (.var 0)) (.kraw k))
            (.tknow .tmg) emptyE
    ∧ IsValue (.lam (.tknow .tmg) emptyE (.var 0))
    ∧ (∃ T' : Ty, T' = Ty.tmg) :=
  ⟨.t_app _ _ _ _ _ _ _ _
      (.t_lam _ _ _ _ _ (.t_var _ _ _ (by simp [lookupCtx])))
      (.t_kraw _ _ hk)
      (emptyE_sub _)
      (subE_refl _),
   .v_lam _ _ _,
   ⟨Ty.tmg, rfl⟩⟩
