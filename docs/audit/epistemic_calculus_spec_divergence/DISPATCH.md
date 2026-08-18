<!-- docs:meta
topic_id: repo.docs.audit.epistemic-calculus-spec-divergence.dispatch
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.epistemic-calculus-spec-divergence.dispatch
-->

# DISPATCH — the refuted epistemic calculus diverges from the `measure` spec

**Opened.** 2026-08-17, on founder instruction, after `#1772` merged at
`1bb2db46fc` carrying `EpistemicPreservationWIP_counterexample.lean`.

**Class.** Formalisation defect in `formal/lean4/`. **Not** a compiler defect
and **not** a soundness bug in Sounio. The repair has already landed; what
remains open is correspondence and consumption, enumerated in §5.

**Priority.** P3. No shipping code is wrong. The exposure is citation exposure:
`EpistemicEffects.lean` is the file a reader reaches for when asking what the
metatheory of `Knowledge<T>` is, and that file's calculus is provably unsound.
The dissertation defends Aug–Sep 2026 with `Knowledge<T>` as contribution
material, so a citation of the refuted module is a live risk with a date on it.

**Founder ruling.** "isso vai contra a spec" — the divergence is a defect in
the model, not a design choice in the language. §2 records the spec text and
the implementation that ruling rests on.

---

## §0 — Scope constraint

This dispatch is confined to `formal/lean4/` plus one new gate under
`scripts/ci/`. It authorises **no** edit to `self-hosted/`, because §2
establishes that the compiler is the side that is already correct.

The `formal/` contract holds without exception: **no Mathlib, no `sorry`,
no `axiom`.** Any repair that needs Mathlib — HALT and report.

---

## §1 — What was proven, and the receipt

`EpistemicEffects.lean` §9.1 previously carried this file's only `sorry`, on
full subject reduction, annotated "straightforward but verbose". That
annotation was false. The statement is not merely unproven; it is **untrue**,
and both it and its nearest weakenings are now machine-checked refutations.

| Theorem | Statement refuted |
|---|---|
| `effect_preservation_is_false` | `HasTy Γ e T E → e ⇒ e' → HasTy Γ e' T E` — the reduct cannot be retyped at the original type, **at any effect** |
| `effect_preservation_existential_is_false` | even `∃ T', HasTy [] e' T' E` — the corruption propagates through application, and `f (kraw k0)` with `f : Knowledge<Nat> → Knowledge<Nat>` is untypable at **every** type and effect |
| `preservation_is_false` (companion file, `#1772`) | the same root result, restated standalone as an existential |

The minimal witness is three lines:

```lean
theorem meas_nat_typed : HasTy [] (.measure (.lit_nat 0) k0) (.tknow .tnat) (singleE .eObserve)
theorem meas_nat_steps : (Expr.measure (.lit_nat 0) k0) ⇒ (.kraw k0)
theorem kraw_not_nat   : ¬ ∃ E, HasTy [] (.kraw k0) (.tknow .tnat) E
```

**Receipt.** `Lean Proofs` = `success` on run `32039008956` at head
`1bb2db46fc` (current `main`). The job is `cd formal/lean4 && lake build` over
all default targets, and both `EpistemicEffects` and `EpistemicEffectsV2` are
declared `@[default_target]` (`formal/lean4/lakefile.lean:637`, `:643`), so the
build reaches them. Zero real `sorry` in either file — all five occurrences of
the word in `EpistemicEffects.lean` are prose inside comments, which was
checked rather than assumed.

Correction on the record: this refutation is **not** new in `#1772`. It already
stood in `EpistemicEffects.lean` §9.1 dated 2026-08-16, in both forms. `#1772`
adds a standalone restatement, which is useful as a citable minimal unit and is
not an independent discovery. This dispatch was opened after an initial report
that overstated `#1772`'s novelty.

---

## §2 — The spec contract the V1 model violates

Three sources agree, and all three disagree with V1.

**The compiler.** `self-hosted/check/check.sio:6888`,
`checker_check_measure_expr_inplace` — the live in-place checker path:

```
// measure(v: T, uncertainty: f64) -> Knowledge<T>.
let first_arg = expr_list_head(e.args)
let v_ty = checker_check_opt_expr_inplace(c, first_arg)
ty_knowledge(v_ty, 0.0 - 1.0)
```

The argument's own type `v_ty` is propagated into the `Knowledge` constructor.
`measure` of a `Nat` is `Knowledge<Nat>`. There is no coercion to real anywhere
on this path. `checker_check_knowledge_ctor_expr_inplace` (`:6898`) is the same
shape, deliberately.

**The language reference.** `CLAUDE.md` §7 states the surface contract as
`let m: Knowledge<mg> = measure(500.0, uncertainty: 2.5)` — a payload carrying
a *unit* type, which is not real and cannot survive a collapse to real.

**The semantics document.** `stdlib/epistemic/SEMANTICS.md` separates Channel A
(uncertainty of the measured quantity) from Channel B (confidence in the
claim), and treats both as metadata *about* a quantity. The quantity is the
subject; the GUM data describes it. V1 inverts that relationship.

---

## §3 — Root cause, one line

V1's runtime Knowledge value has **no payload slot**:

```lean
| kraw : KCell → Expr                       -- EpistemicEffects.lean:158
structure KCell where value : Int; gumVar : Int; conf : Int   -- :134
| t_kraw : ∀ Γ k, kvalid k →
    HasTy Γ (.kraw k) (.tknow .treal) emptyE -- :222  — ALWAYS Real
| meas_red : IsValue v → Step (.measure v k) (.kraw k)  -- :329  — DISCARDS v
```

`meas_red` throws the measured value away and keeps only the cell. Since the
cell is a fixed scalar triple, the payload type is erased to `Real` for every
`T`, and `kvalue_red` then reads the cell's `Int` back out as `.lit_real`. The
type-level promise `measure : T → Knowledge<T>` and the operational rule
`measure v k ⇒ kraw k` cannot both hold. `t_measure` keeps the promise for any
`T`; `meas_red` breaks it for every `T` except `Real`.

This is a modelling error with a two-year-old smell to it: the model was
written around the *uncertainty triple*, which is what makes GUM interesting,
and the value being measured was treated as one more field of that triple
rather than as the thing the triple is about.

---

## §4 — What is already repaired

`formal/lean4/EpistemicEffectsV2.lean`, 629 lines, is the value-carrying
calculus:

```lean
| kraw : Expr → KMeta → Expr                -- payload value + metadata
structure KMeta where gumVar : Int; conf : Int   -- the value LEFT the metadata
| t_kraw : ∀ Γ T v m, HasTy Γ v T emptyE → IsValue v → kvalid m →
    HasTy Γ (.kraw v m) (.tknow T) emptyE   -- payload type PRESERVED
| meas_red : IsValue v → Step (.measure v m) (.kraw v m)   -- v RETAINED
```

and it proves the full statement, not a weakening:

```lean
theorem preservation {e T E} (h : HasTy [] e T E) {e'} (hs : e ⇒ e') : HasTy [] e' T E
theorem effect_progress {e T E} (ht : HasTy [] e T E) : IsValue e ∨ ∃ e', e ⇒ e'
```

Same type, same effect set, closed terms, `Γ = []`. Zero `sorry`, zero `axiom`,
Mathlib-free, and green under the §1 receipt.

V2 is therefore in exact correspondence with §2's `ty_knowledge(v_ty, …)`. The
repair is not owed — it is done. Which makes the residuals in §5 the whole of
this dispatch.

---

## §5 — What remains open

### R1 — The dependency graph is inverted

Measured on `main`:

| Module | Imported by |
|---|---|
| `EpistemicEffects` (refuted) | `EpistemicEffectsV2`, `EpistemicPreservationWIP`, `EpistemicPreservationWIP_counterexample` — 3 dependents |
| `EpistemicEffectsV2` (proven) | **nothing** |

The refuted calculus is the one with dependents; the proven one is a leaf. V2
imports V1 for shared definitions (`Effect`, `Ty`, `EffectSet`, `emptyE_sub`,
`TyCtx`), which is legitimate reuse of the parts that were never in question —
but it means the module a newcomer imports by name is the unsound one, and no
mechanism nudges them elsewhere.

**Landed (banner).** V1's header now says `REFUTED MODEL — USE EpistemicEffectsV2`.
That closes the dispatch wording of R1. It does not close the fact: a banner
is not an import edge.

**Landed (consumer).** `EpistemicEffectsV2_measure_nat.lean` imports V2 and
proves `measure_nat_reduct_stays_know_nat` — the V1 counterexample inverted.
The correspondence gate's `--lean-consume` arm builds that module only after
the V1 mutant (`v1_imports_measure_nat.lean`) fails to elaborate. A grep of
`import EpistemicEffectsV2` without that mutant would measure mention, not use.

**Deferred — extract the shared spine.** A third module holding `Effect`, `Ty`,
`EffectSet`, `TyCtx` would stop V2 from being a *client* of the refuted file.
It would not give V2 a *consumer*. Doing the extraction now would trade
orphanhood for orphanhood, with fewer visible edges: V2 would import the
spine instead of V1, and still have no importer of its own. Revisit only
after the consumer above has been the import edge for a measured stretch.

### R2 — The correspondence is prose, not a gate

The claim "V2 models what the checker implements" currently exists only as
English, in this document and in a lakefile comment. Nothing fails if
`checker_check_measure_expr_inplace` is edited tomorrow to erase the payload
type, or if V2's `t_kraw` drifts.

This is the residual that matters, and it is the one this project has been
burned by before: a proof about a model is not a statement about the
implementation until the correspondence is checked by something that can fail.

**Wanted.** `scripts/ci/epistemic_measure_correspondence_gate.sh` asserting, on
both sides, that (a) the checker's `measure`/`Knowledge` constructor paths pass
the argument type through to `ty_knowledge` and do not substitute a fixed
scalar type, and (b) V2's `t_kraw` binds the payload type variable rather than a
literal. A grep-shaped gate is acceptable here provided it ships with a
**positive control** — a deliberately-broken copy of each side that the gate
must reject. Per house rule, a gate without a firing positive control measures
nothing.

### R3 — ε is unspecified at the type level

The checker writes `ty_knowledge(v_ty, 0.0 - 1.0)` — the sentinel `-1` for
"epsilon unspecified". V2's `t_kraw` instead demands `kvalid m`, i.e. metadata
present and well-formed with `conf ∈ [0, 1000]`.

So V2 is *stricter* than the implementation on the confidence channel while
being exactly right on the payload channel. That gap is not a refutation of
anything, but it is a known correspondence hole, and it is the same hole already
tracked as the unauthored `Knowledge<T>` ε/provenance work. Record it here;
do not attempt to close it under this dispatch.

---

## §6 — Acceptance criteria

1. R1 and R2 landed. R3 recorded and explicitly deferred, not silently dropped.
   R1's remaining *fact* (V2 had no importer) is closed by
   `EpistemicEffectsV2_measure_nat.lean`, not by the banner. Spine extraction
   stays deferred — see §5 R1.
2. `Lean Proofs` green, with V1, V2, and `EpistemicEffectsV2_measure_nat` all
   `@[default_target]`. Deleting V1 to make the problem disappear is **out of
   scope** — the refutation is a result worth keeping, and §1's theorems are
   its statement.
3. The new gate's positive control demonstrated firing, with the output pasted
   into the PR body. A green gate whose control was never shown to fail is not
   evidence.
4. No `sorry`, no `axiom`, no Mathlib anywhere under `formal/`.
5. Docs registry synced **after** the doc commit, never before — the registry is
   filesystem-derived and syncing first leaves the gate red.

---

## §7 — What NOT to do

- **Do not "fix" the compiler.** §2 establishes the compiler is the correct
  side. An edit to `self-hosted/check/check.sio` under this dispatch is a
  category error.
- **Do not delete or weaken `EpistemicEffects.lean` §9.1.** A refutation is a
  result. The honest record of a false claim is more valuable here than a clean
  file, and this repository's thesis is exactly that.
- **Do not restate the refutation a fourth time.** Three statements of it
  already exist. A fourth adds citation ambiguity, not confidence.
- **Do not claim V2 "proves Sounio sound."** It proves subject reduction and
  progress for a calculus that models a fragment of `Knowledge<T>`. The step
  from that to a statement about Madaros is exactly what R2 does not yet
  establish, and overstating it would repeat the §1 `sorry` annotation's error
  in the opposite direction.
