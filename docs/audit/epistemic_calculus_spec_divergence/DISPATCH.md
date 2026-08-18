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

Measured on `main` before the consumer (still the citation risk if this
PR is unread):

| Module | Imported by |
|---|---|
| `EpistemicEffects` (refuted) | `EpistemicEffectsV2`, `EpistemicPreservationWIP`, `EpistemicPreservationWIP_counterexample` — 3 dependents |
| `EpistemicEffectsV2` (proven) | **nothing** |

On this branch the second row is two importers:
`EpistemicEffectsV2_measure_nat` and `EpistemicEffectsV2_kvalue_nat`.
That stops V2 from being a leaf. It is not coverage. See the fraction
under **Coverage**.

The refuted calculus is still the one with more dependents. V2 imports V1
for shared definitions (`Effect`, `Ty`, `EffectSet`, `emptyE_sub`,
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

**Coverage (measured 2026-08-18, union of V2 importers, after #1883 /
#1892).** One importer ≠ V2 covered. Re-derive with the names in
`EpistemicEffectsV2.lean` against every `EpistemicEffectsV2_*.lean`
consumer, comments stripped:

| Layer | After measure-Nat | After + kvalue-Nat | Remainder |
|---|---:|---:|---:|
| Named theorems (28) | 0 | **1** (`preservation`) | 27 |
| `HasTy` rules (14) | 3 | **4** (`+ t_kvalue`) | 10 |
| `Step` rules (19) | 1 | **2** (`+ kvalue_red`) | 17 |
| `IsValue` rules (4) | 1 (`v_nat`) | **1** (`v_nat`) | 3 |

kvalue did **not** move `IsValue`. `t_kraw` needs the *payload* to be a
value (`v_nat`); `kvalue_red` does not require proving the outer `kraw`
is a value. `v_kraw`, `v_lam`, `v_real` still have no client. A reader
who treats "V2 has a theorem client" as "4 of 4 values are covered" is
repeating the mention/use error on a new noun.

**Next proposition (round 3) — `invKraw` on the V1 propagation witness.**
Not the next name in the file. V1's second refutation
(`effect_preservation_existential_is_false`) is the unused compiler
surface: `f (kraw _)` with `f : Knowledge<Nat> → Knowledge<Nat>` is
untypable after `meas_red` in argument position. Madaros users pass
`Knowledge<T>` into functions. Consumers 1 and 2 show the reduct *is*
`Knowledge<Nat>` and *unwraps* to `Nat`; they do not show it can be
*used* as `Knowledge<Nat>`. That is the remaining payload bug that
shows up as a type error (or a silent coerce-to-Real) on a call.

The named theorem to cite is `invKraw`, not a second application of
`preservation` (already cited; the unique count would stay 1 of 28)
and not `canon_know` (V1 has it; a V1 mutant would likely elaborate).
`invKraw` is the V2 dual of V1's `genKraw`: recover `T` from a typed
`kraw`, rather than pin `T = Real`. The instance is the identity
applied to `kraw (.lit_nat 0) m` — the inverted existential. Expected
new constructors if that consumer is written: `t_lam`, `t_app`,
`t_var`, `v_lam`, and `v_kraw` if the reduct is used as a CBV value.
That would move `IsValue` to 3 of 4 (`v_real` still unused) and
theorems to 2 of 28. It would not make 4 of 4, and 4 of 4 is not the
stop (see below).

Rejected for round 3:

| Candidate | Why not |
|---|---|
| `effect_progress` | V1 proves progress. Mutant would likely pass. |
| `preservation` on measure-Nat again | Same path as consumer 1. No new compiler surface. |
| `canon_know` | V1 has it. Mutant would likely pass. |
| `kunc` / `kconf` | Both calculi project metadata to `Real`. Mutant would pass. Compiler `check_knowledge_epsilon` agrees. |
| `kadd` / `kmul` on `Knowledge<Nat>` | V2 *rejects* this (`t_kadd` pins `Knowledge<Real>`). The checker *keeps* `left_inner` (`epistemic.sio` Knowledge binop → `ty_knowledge(left_inner, …)`). That is a correspondence hole, not a consumer. Recorded next to R3; do not "cover" it by forcing a Real-only client. |
| `v_real` via a Real literal | Would make 4 of 4 `IsValue` if stacked with `v_lam`/`v_kraw`. Catches nothing about `Knowledge<T>`. |

**When to stop.** Not when all 28 theorems have a client. A surface is
owed a consumer only if it is *discriminating*: a V1 mutant of the
same statement fails for a payload/Real reason, not a missing `:=`.
Coverage is sufficient when every discriminating Madaros Knowledge
operation is either consumed or classified as a hole/glue.

| Surface | Compiler | Discriminates V1 vs V2? | Status |
|---|---|---|---|
| Introduce (`measure` / ctor) | `ty_knowledge(v_ty)` | Yes — V1 discards `v` | Consumed (round 1) |
| Eliminate payload | `check_knowledge_unwrap` | Yes — V1 → `lit_real` | Consumed (round 2) |
| Use as `Knowledge<T>` | pass / apply / let | Yes — V1 existential | Round 3 (`invKraw` on that witness) |
| Eliminate metadata | `.epsilon` / `.confidence` | No — both → `Real` | Do not consume |
| GUM `+` `*` | `ty_knowledge(left_inner, …)` | Hole — V2 pins Real, checker keeps `T` | Do not consume; see R3-adjacent |
| Progress, weakening, lookup, shift, `int_sq_nonneg`, `gAddMeta_valid` | none | Glue | Do not consume |

After round 3 the discriminating set is empty. The leftover 26
theorems and `v_real` are not a backlog. 4 of 4 `IsValue` is a
possible *measurement*, not a stop. If a later consumer is proposed
and its V1 mutant elaborates, that is the signal to stop, not to
weaken the mutant.

**Deferred — extract the shared spine (end condition).** A third
module holding `Effect`, `Ty`, `EffectSet`, `TyCtx` would stop V2 from
being a *client* of the refuted file. It would not give V2 a
*consumer*. Two CI-enforced importers have now been that import edge
for a measured stretch, so the original "wait until there is a
consumer" reason is spent. That does **not** trigger extraction.
Extract on the first of these, not on a consumer count:

1. A new Lean module needs `Ty` / `Effect` / `EffectSet` / `TyCtx` and
   would otherwise `import EpistemicEffects` for those names — the next
   spine client is about to be born as a V1 dependent.
2. V1 gains a fourth importer that is not a refutation restatement.
3. An external-facing citation (dissertation, paper, registry claim)
   names `EpistemicEffects.lean` as the Knowledge metatheory.

Until one of those fires, V2→V1 for the spine is the known ugliness.
Extracting now still trades a visible edge for a hidden one.

**Landed (consumer 2).** `EpistemicEffectsV2_kvalue_nat.lean` cites
`preservation` and proves `kvalue_nat_reduct_stays_nat`. The
`--lean-consume-kvalue` arm builds that module only after
`v1_imports_kvalue_nat.lean` fails. `--lean-consume` stays the
measure-Nat pair only — one named step, one pair.

**Acceptance for the named kvalue step (written before the step is
scored).** A green Lean Proofs *job* is not evidence. The PR head must
show this step as `success` and not `skipped`:

```
success  V2 kvalue-Nat consumer (V1 mutant must fail first)
```

and that step's log must contain:

```
POSITIVE_CONTROL_FIRED: v1_imports_kvalue_nat rejected
V2_CONSUMED: EpistemicEffectsV2_kvalue_nat built
```

A job whose this step is skipped is the 390-gate class. The #1883
verification (kvalue lines inside the measure-Nat step) does not
transfer: that head did not have this step. If the kvalue mutant
elaborates, the consumer must not be scored.

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
being exactly right on the payload channel. The same shape of hole sits
on GUM arithmetic: V2's `t_kadd` / `t_kmul` pin `Knowledge<Real>`; the
checker's Knowledge binop returns `ty_knowledge(left_inner, …)` and so
keeps `T`. Neither hole is a refutation of the payload repair. Record
them here; do not attempt to close them under this dispatch, and do not
consume `kadd` as if it were the next coverage target.

---

## §6 — Acceptance criteria

1. R1 and R2 landed. R3 recorded and explicitly deferred, not silently dropped.
   R1's remaining *fact* (V2 had no importer) is closed by
   `EpistemicEffectsV2_measure_nat.lean`, not by the banner. That close is
   an import edge, not metatheory coverage. After consumer 2 the
   fraction is **1 of 28** named theorems, **4 of 14** `HasTy`,
   **2 of 19** `Step`, **1 of 4** `IsValue` — kvalue did not move
   `IsValue`. Stop when the discriminating set in §5 R1 is empty, not
   when 28/28 or 4/4. Spine extraction stays deferred until one of the
   three end conditions in §5 R1, not until N consumers.
2. `Lean Proofs` green, with V1, V2, `EpistemicEffectsV2_measure_nat`,
   and `EpistemicEffectsV2_kvalue_nat` all `@[default_target]`. Deleting
   V1 to make the problem disappear is **out of scope** — the refutation
   is a result worth keeping, and §1's theorems are its statement.
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
