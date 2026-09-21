import EpistemicEffectsV2

/-!
# V2 consumer — `measure` of a Nat stays `Knowledge<Nat>`

This is the first Lean module that *imports* `EpistemicEffectsV2`.
One importer stops V2 from being a leaf. It does not cover V2:
this file cites **0 of the 28** named theorems in
`EpistemicEffectsV2.lean` (including `preservation` and
`effect_progress`). It reconstructs one path from constructors
(`t_lit_nat`, `t_measure`, `t_kraw`, `meas_red`, `v_nat`).
Count lives in `docs/audit/epistemic_calculus_spec_divergence/DISPATCH.md` §5 R1.

V1 proves the opposite statement (`preservation_is_false` /
`kraw_not_nat`): `measure (lit_nat 0)` steps to a `kraw` that cannot
be typed at `Knowledge<Nat>`. That is the payload-erasing calculus.

Here the same witness is stated in the value-carrying calculus. The
reduct is `.kraw (.lit_nat 0) m`, and `t_kraw` binds the payload type
variable, so the reduct is `Knowledge<Nat>`. The statement is
inexpressible under `import EpistemicEffects` — the fixture
`scripts/ci/fixtures/epistemic_measure_correspondence/v1_imports_measure_nat.lean`
is that attempt, and must fail to elaborate.

No `sorry`. No `axiom`. No Mathlib.
-/

namespace Sounio.EpistemicEffectsV2

open Sounio.EpistemicEffects (emptyE singleE)

theorem measure_nat_typed (m : KMeta) (hm : kvalid m) :
    HasTy [] (.measure (.lit_nat 0) m) (.tknow .tnat) (singleE .eObserve) :=
  .t_measure _ _ _ _ (.t_lit_nat _ _) hm

theorem measure_nat_steps (m : KMeta) :
    (.measure (.lit_nat 0) m) ⇒ (.kraw (.lit_nat 0) m) :=
  .meas_red (.v_nat 0)

theorem measure_nat_kraw_typed (m : KMeta) (hm : kvalid m) :
    HasTy [] (.kraw (.lit_nat 0) m) (.tknow .tnat) emptyE :=
  .t_kraw _ _ _ _ (.t_lit_nat _ _) (.v_nat 0) hm

/-- The V1 counterexample, inverted. Same starting term, opposite
    conclusion: the reduct remains `Knowledge<Nat>`. -/
theorem measure_nat_reduct_stays_know_nat
    (m : KMeta) (hm : kvalid m) :
    HasTy [] (.measure (.lit_nat 0) m) (.tknow .tnat) (singleE .eObserve)
    ∧ ((.measure (.lit_nat 0) m) ⇒ (.kraw (.lit_nat 0) m))
    ∧ HasTy [] (.kraw (.lit_nat 0) m) (.tknow .tnat) emptyE :=
  ⟨measure_nat_typed m hm, measure_nat_steps m, measure_nat_kraw_typed m hm⟩

end Sounio.EpistemicEffectsV2
