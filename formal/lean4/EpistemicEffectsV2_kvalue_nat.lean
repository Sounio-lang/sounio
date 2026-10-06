import EpistemicEffectsV2

/-!
# V2 consumer — `kvalue` of `Knowledge<Nat>` stays `Nat`

This file *cites* `preservation`. The first importer reconstructed the
measure-Nat path from constructors and cited zero of V2's 28 theorems.
That is an import edge, not a verifier.

The first named theorem worth a client is not `preservation` applied
to that same measure-Nat path (no new compiler contact) and not
`effect_progress` (V1 proves progress; a V1 mutant of progress would
likely elaborate, so the positive control would not fire).

It is `preservation` applied to `kvalue_red` on `Knowledge<Nat>`.
The live checker path is `check_knowledge_unwrap` in
`self-hosted/check/epistemic.sio`: `Knowledge<T>` unwraps to `T`.
V1's `kvalue_red` always yields `lit_real`. The statement cannot be
proved under `import EpistemicEffects` — the fixture
`scripts/ci/fixtures/epistemic_measure_correspondence/v1_imports_kvalue_nat.lean`
is that attempt, and must fail.

No `sorry`. No `axiom`. No Mathlib.
-/

namespace Sounio.EpistemicEffectsV2

open Sounio.EpistemicEffects (emptyE)

theorem kvalue_nat_typed (m : KMeta) (hm : kvalid m) :
    HasTy [] (.kvalue (.kraw (.lit_nat 0) m)) .tnat emptyE :=
  .t_kvalue _ _ _ _ (.t_kraw _ _ _ _ (.t_lit_nat _ _) (.v_nat 0) hm)

theorem kvalue_nat_steps (m : KMeta) :
    (.kvalue (.kraw (.lit_nat 0) m)) ⇒ (.lit_nat 0) :=
  .kvalue_red

/-- Cites `preservation`. The reduct of `kvalue` on a `Knowledge<Nat>`
    value is still `Nat`. -/
theorem kvalue_nat_reduct_stays_nat
    (m : KMeta) (hm : kvalid m) :
    HasTy [] (.kvalue (.kraw (.lit_nat 0) m)) .tnat emptyE
    ∧ ((.kvalue (.kraw (.lit_nat 0) m)) ⇒ (.lit_nat 0))
    ∧ HasTy [] (.lit_nat 0) .tnat emptyE :=
  ⟨kvalue_nat_typed m hm,
   kvalue_nat_steps m,
   preservation (kvalue_nat_typed m hm) (kvalue_nat_steps m)⟩

end Sounio.EpistemicEffectsV2
