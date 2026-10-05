import EpistemicEffects
open Sounio.EpistemicEffects

/-
  Positive control for the V2 `invKraw` consumer on the propagation
  witness. Same theorem name as `EpistemicEffectsV2_invkraw_nat.lean`,
  written against the *refuted* calculus. V1's `kraw` has no payload
  slot, so `t_kraw` types every cell at `Knowledge<Real>` and
  `f (kraw k)` with `f : Knowledge<Nat> → Knowledge<Nat>` cannot
  elaborate. If `lake env lean` on this file exits 0, the consumer
  arm is measuring mention, not use, and must not be added.
-/

theorem kraw_nat_inverts_and_is_usable
    (k : KCell) (hk : kvalid k) :
    HasTy [] (.app (.lam (.tknow .tnat) emptyE (.var 0)) (.kraw k))
            (.tknow .tnat) emptyE
    ∧ IsValue (.lam (.tknow .tnat) emptyE (.var 0))
    ∧ (∃ T', T' = .tnat ∧ HasTy [] (.lit_nat 0) T' emptyE) :=
  ⟨.t_app _ _ _ _ _ _ _ _
      (.t_lam _ _ _ _ _ (.t_var _ _ _ (by simp [lookupCtx])))
      (.t_kraw _ _ hk)
      (emptyE_sub _)
      (subE_refl _),
   .v_lam _ _ _,
   ⟨.tnat, rfl, .t_lit_nat _ _⟩⟩
