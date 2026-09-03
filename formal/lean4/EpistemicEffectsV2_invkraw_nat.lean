import EpistemicEffectsV2

/-!
# V2 consumer — `invKraw` on the V1 propagation witness

This file *cites* `invKraw`. Consumers 1 and 2 show that `measure` of a
Nat *is* `Knowledge<Nat>` and *unwraps* to `Nat`. They do not show the
reduct can be *used* as `Knowledge<Nat>`. V1's second refutation
(`effect_preservation_existential_is_false`) is exactly that hole:
`f (kraw _)` with `f : Knowledge<Nat> → Knowledge<Nat>` is untypable.

`invKraw` is the V2 dual of V1's `genKraw`: recover `T` from a typed
`kraw`, rather than pin `T = Real`. The instance is the identity
applied to `kraw (.lit_nat 0) m`. The statement cannot be proved
under `import EpistemicEffects` — the fixture
`scripts/ci/fixtures/epistemic_measure_correspondence/v1_imports_invkraw_nat.lean`
is that attempt, and must fail.

No `sorry`. No `axiom`. No Mathlib.
-/

namespace Sounio.EpistemicEffectsV2

open Sounio.EpistemicEffects (emptyE lookupCtx emptyE_sub subE_refl)

def idKnowNat : Expr := .lam (.tknow .tnat) emptyE (.var 0)

theorem id_know_nat_typed :
    HasTy [] idKnowNat (.tarrow (.tknow .tnat) emptyE (.tknow .tnat)) emptyE :=
  .t_lam _ _ _ _ _ (.t_var _ _ _ (by simp [lookupCtx]))

theorem kraw_nat_typed (m : KMeta) (hm : kvalid m) :
    HasTy [] (.kraw (.lit_nat 0) m) (.tknow .tnat) emptyE :=
  .t_kraw _ _ _ _ (.t_lit_nat _ _) (.v_nat 0) hm

theorem id_know_nat_is_value : IsValue idKnowNat :=
  .v_lam _ _ _

theorem kraw_nat_is_value (m : KMeta) :
    IsValue (.kraw (.lit_nat 0) m) :=
  .v_kraw m (.v_nat 0)

/-- Cites `invKraw`. The payload of `kraw (.lit_nat 0) m` is `Nat`,
    and the identity applied to that value is typed `Knowledge<Nat>`. -/
theorem kraw_nat_inverts_and_is_usable
    (m : KMeta) (hm : kvalid m) :
    HasTy [] (.app idKnowNat (.kraw (.lit_nat 0) m)) (.tknow .tnat) emptyE
    ∧ IsValue idKnowNat
    ∧ IsValue (.kraw (.lit_nat 0) m)
    ∧ (∃ T', T' = .tnat ∧ HasTy [] (.lit_nat 0) T' emptyE
        ∧ IsValue (.lit_nat 0)) := by
  have hk := kraw_nat_typed m hm
  rcases invKraw hk rfl with ⟨T', hT, hv, hval, _⟩
  injection hT with hTeq
  subst hTeq
  refine ⟨?app, id_know_nat_is_value, kraw_nat_is_value m,
    ⟨.tnat, rfl, hv, hval⟩⟩
  exact .t_app _ _ _ _ _ _ _ _ id_know_nat_typed hk (emptyE_sub _) (subE_refl _)

end Sounio.EpistemicEffectsV2
