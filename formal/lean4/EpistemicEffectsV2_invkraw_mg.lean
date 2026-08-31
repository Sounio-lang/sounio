import EpistemicEffectsV2

/-!
# V2 consumer — `invKraw` at `Knowledge<mg>`

Consumers 1–3 used `Knowledge<Nat>`. The checker is generic
(`ty_knowledge(v_ty)`); Nat was the only payload we consumed. V1's
second refutation (`effect_preservation_existential_is_false`) is
payload-blind: `f (kraw _)` with `f : Knowledge<T> → Knowledge<T>`
is untypable for any `T ≠ Real`. The language actually ships a unit
payload — `let m: Knowledge<mg> = measure(500.0, uncertainty: 2.5)`.

This file cites `invKraw` on that payload. `tmg` lives on the shared
`Ty` spine so a V1 mutant can write `Knowledge<mg>` and still fail
for the Real pin, not a missing constructor. The instance is the
identity applied to `kraw (.lit_mg 500) m`. The statement cannot
be proved under `import EpistemicEffects` — the fixture
`scripts/ci/fixtures/epistemic_measure_correspondence/v1_imports_invkraw_mg.lean`
is that attempt, and must fail.

This is not a fourth Nat-shaped client. It does not clone measure
or unwrap at mg: those would fail in V1 for the same Real pin
already scored. It closes the "T is only Nat" hole.

No `sorry`. No `axiom`. No Mathlib.
-/

namespace Sounio.EpistemicEffectsV2

open Sounio.EpistemicEffects (emptyE lookupCtx emptyE_sub subE_refl)

def idKnowMg : Expr := .lam (.tknow .tmg) emptyE (.var 0)

theorem id_know_mg_typed :
    HasTy [] idKnowMg (.tarrow (.tknow .tmg) emptyE (.tknow .tmg)) emptyE :=
  .t_lam _ _ _ _ _ (.t_var _ _ _ (by simp [lookupCtx]))

theorem kraw_mg_typed (m : KMeta) (hm : kvalid m) :
    HasTy [] (.kraw (.lit_mg 500) m) (.tknow .tmg) emptyE :=
  .t_kraw _ _ _ _ (.t_lit_mg _ _) (.v_mg 500) hm

theorem id_know_mg_is_value : IsValue idKnowMg :=
  .v_lam _ _ _

theorem kraw_mg_is_value (m : KMeta) :
    IsValue (.kraw (.lit_mg 500) m) :=
  .v_kraw m (.v_mg 500)

/-- Cites `invKraw`. The payload of `kraw (.lit_mg 500) m` is `mg`,
    and the identity applied to that value is typed `Knowledge<mg>`. -/
theorem kraw_mg_inverts_and_is_usable
    (m : KMeta) (hm : kvalid m) :
    HasTy [] (.app idKnowMg (.kraw (.lit_mg 500) m)) (.tknow .tmg) emptyE
    ∧ IsValue idKnowMg
    ∧ IsValue (.kraw (.lit_mg 500) m)
    ∧ (∃ T', T' = .tmg ∧ HasTy [] (.lit_mg 500) T' emptyE
        ∧ IsValue (.lit_mg 500)) := by
  have hk := kraw_mg_typed m hm
  rcases invKraw hk rfl with ⟨T', hT, hv, hval, _⟩
  injection hT with hTeq
  subst hTeq
  refine ⟨?app, id_know_mg_is_value, kraw_mg_is_value m,
    ⟨.tmg, rfl, hv, hval⟩⟩
  exact .t_app _ _ _ _ _ _ _ _ id_know_mg_typed hk (emptyE_sub _) (subE_refl _)

end Sounio.EpistemicEffectsV2
