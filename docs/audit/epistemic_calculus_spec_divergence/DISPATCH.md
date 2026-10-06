<!-- docs:meta
topic_id: repo.docs.audit.epistemic-calculus-spec-divergence.dispatch
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.epistemic-calculus-spec-divergence.dispatch
-->

# DISPATCH — the refuted epistemic calculus diverges from the `measure` spec

**Opened.** 2026-08-17, on founder instruction, after `#1772` merged at
`1bb2db46fc` carrying `EpistemicPreservationWIP_counterexample.lean`.

**Class.** Formalisation defect in `formal/lean4/`, now read as a **language
specification** defect. The payload repair has landed. Consumption of the
discriminating Nat/`mg` surfaces has landed. What remains is not another
consumer: it is whether `EpistemicEffectsV2` is the specification of
`Knowledge<T>` — the feature that makes Sounio not Rust — or an exercise
that is silent where the language is wrong.

**Priority.** P0 for epistemic honesty (2). The model-as-exercise reading
is closed. A citation of V2 as the metatheory of `Knowledge<T>` without a
proposition about GUM across a compiled call is an overclaim.

**Founder ruling (payload, 2026-08-17).** "isso vai contra a spec" — V1's
discard of the payload was a defect in the model, not a design choice.
§2 still holds for `measure` / `ty_knowledge(v_ty)`.

**Founder ruling (PL, 2026-08-19).** All lanes in Sounio as a programming
language. V2 is no longer "a formalisation with consumers". It is the
specification of `Knowledge<T>`. The discriminating set being empty at
Nat/`mg` does not end the work. R3 — correspondence for GUM
propagation — is the target. Claims-Forbidden: "V2 models `Knowledge<T>`"
without a proposition about boundary crossing.

---

## §0 — Scope constraint

This dispatch is confined to `formal/lean4/` plus gates under
`scripts/ci/`. It authorises **no** edit to `self-hosted/ir/lower.sio`.
§2 still holds for the *payload* of `measure`: the checker is the
correct side of that disagreement. It does **not** hold for
first-order GUM across a compiled call — that is a language defect
the model does not mention. The FO-call-boundary lane owns the
compiler object (`docs/audit/MADAROS_FO_CALL_BOUNDARY_DISPATCH_2026-08-18.md`).
This dispatch's job is to stop V2 from being silent about it.

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

**Coverage (measured 2026-08-18, union of V2 importers, after
consumer 3).** One importer ≠ V2 covered. Re-derive with the names in
`EpistemicEffectsV2.lean` against every `EpistemicEffectsV2_*.lean`
consumer, comments stripped:

| Layer | After measure | After + kvalue | After + invKraw | After + mg | Remainder |
|---|---:|---:|---:|---:|---:|
| Named theorems | 0 | 1 (`preservation`) | **2** (`+ invKraw`) | **2** (same; `invKraw` again) | glue |
| `HasTy` rules | 3 | 4 | **7** | **8** (`+ t_lit_mg`; now 15) | 7 |
| `Step` rules (19) | 1 | 2 | **2** | **2** | 17 |
| `IsValue` rules | 1 (`v_nat`) | 1 | **3** | **4** (`+ v_mg`; now 5) | 1 (`v_real`) |

kvalue did **not** move `IsValue`. invKraw on the propagation witness
does: the identity is `v_lam` and the reduct used as a CBV argument is
`v_kraw`. `v_real` is still unused, and that is correct — it does not
discriminate `Knowledge<T>`. 3 of 4 is not 4 of 4, and 4 of 4 is not
the stop.

**Why `invKraw` (round 3), not the next name in the file.** V1's
second refutation (`effect_preservation_existential_is_false`) was
the unused compiler surface: `f (kraw _)` with
`f : Knowledge<Nat> → Knowledge<Nat>` is untypable after `meas_red`
in argument position. Madaros users pass `Knowledge<T>` into
functions. Consumers 1 and 2 showed the reduct *is* `Knowledge<Nat>`
and *unwraps* to `Nat`; they did not show it can be *used* as
`Knowledge<Nat>`.

The named theorem is `invKraw`, not a second application of
`preservation` (already cited; the unique count would have stayed
1 of 28) and not `canon_know` (V1 has it; a V1 mutant would likely
elaborate). `invKraw` is the V2 dual of V1's `genKraw`: recover `T`
from a typed `kraw`, rather than pin `T = Real`. The instance is the
identity applied to `kraw (.lit_nat 0) m`. That moved `IsValue` to
3 of 4 and theorems to 2 of 28. It did not make 4 of 4, and 4 of 4
is not the stop (see below).

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
| Use as `Knowledge<T>` | pass / apply / let | Yes — V1 existential | Consumed (round 3); T≠Nat in round 4 |
| Eliminate metadata | `.epsilon` / `.confidence` | No — both → `Real` | Do not consume |
| GUM `+` `*` | `ty_knowledge(left_inner, …)` | Hole — V2 pins Real, checker keeps `T` | Do not consume; see R3-adjacent |
| Progress, weakening, lookup, shift, `int_sq_nonneg`, `gAddMeta_valid` | none | Glue | Do not consume |

**Conclusion after rounds 1–3 — discriminating set empty at Nat.**
Fraction beside it was **2/28** theorems, **7/14** `HasTy`,
**2/19** `Step`, **3/4** `IsValue`. Read 2/28 without that sentence
and it looks like 93% is missing. The other theorems are ones any
implementation would satisfy. They do not separate V2 from V1.
Three consumers exhausted the Nat-shaped surfaces that do.

**What reopened the set (round 4, landed).** Not a fourth
`Knowledge<Nat>` consumer. The second reopening clause: a client
that uses `Knowledge` at a type other than Nat. The checker is
generic (`ty_knowledge(v_ty)`). We had only consumed Nat. The
payload the language ships is `Knowledge<mg>`.
`EpistemicEffectsV2_invkraw_mg.lean` cites `invKraw` on
`kraw (.lit_mg 500) m`. `tmg` was added to the shared `Ty` spine
so the V1 mutant can write `Knowledge<mg>` and still fail because
`t_kraw` pins Real, not because `tmg` is missing. `lit_mg` /
`t_lit_mg` / `v_mg` are the V2 intro; they are new constructors,
not new discriminating *reasons*. The unique named-theorem count
stays 2 (`preservation`, `invKraw`).

Do **not** clone measure or unwrap at mg. Those V1 mutants would
fail for the same Real pin already scored at Nat. That would be
the 93%-left mistake at the type-instance level.

**What would make the *consumer* set non-empty again.** Not
another unit literal (`Knowledge<kg>`, `Knowledge<L>`). A new
V2 proposition whose V1 mutant fails for a payload/Real reason.
Glue (`genMg`) is still glue. `t_kadd` pins Real (R3b) is still
not a consumer.

**What is owed as *specification*, not as a consumer.** R3c:
`gum_across_compiled_call`. That is the PL target. It does not
reopen the Nat/`mg` consumer table. See §5 R3.

Until one of those appears, do not add a consumer. If a proposed
client's V1 mutant elaborates, that is the signal to stop, not to
weaken the mutant. `tmg` on the shared spine is not spine
extraction: V2 still imports `Ty` from the refuted module.
The three extraction triggers in the deferred note below have
not fired.

**Landed (consumer 3).** `EpistemicEffectsV2_invkraw_nat.lean` cites
`invKraw` and proves `kraw_nat_inverts_and_is_usable`. The
`--lean-consume-invkraw` arm builds that module only after
`v1_imports_invkraw_nat.lean` fails.

**Acceptance for the named invKraw step (written before the step is
scored).** A green Lean Proofs *job* is not evidence. The PR head must
show this step as `success` and not `skipped`:

```
success  V2 invKraw-Nat consumer (V1 mutant must fail first)
```

and that step's log must contain:

```
POSITIVE_CONTROL_FIRED: v1_imports_invkraw_nat rejected
V2_CONSUMED: EpistemicEffectsV2_invkraw_nat built
```

A job whose this step is skipped is the 390-gate class. The #1892 /
#1883 verifications do not transfer: those heads did not have this
step. If the invKraw mutant elaborates, the consumer must not be
scored.

**Landed (consumer 4).** `EpistemicEffectsV2_invkraw_mg.lean` cites
`invKraw` at `Knowledge<mg>` and proves `kraw_mg_inverts_and_is_usable`.
The `--lean-consume-invkraw-mg` arm builds that module only after
`v1_imports_invkraw_mg.lean` fails.

**Acceptance for the named invKraw-mg step (written before the step
is scored).** A green Lean Proofs *job* is not evidence. The PR
head must show this step as `success` and not `skipped`:

```
success  V2 invKraw-mg consumer (V1 mutant must fail first)
```

and that step's log must contain:

```
POSITIVE_CONTROL_FIRED: v1_imports_invkraw_mg rejected
V2_CONSUMED: EpistemicEffectsV2_invkraw_mg built
```

and the mutant rejection must mention `Knowledge` / `tmg` / `treal`
(payload/Real), not an unknown identifier `tmg`. A job whose this
step is skipped is the 390-gate class. The #1909 / #1892 / #1883
verifications do not transfer: those heads did not have this step.
If the mg mutant elaborates, the consumer must not be scored.

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

### R3 — GUM correspondence (the target, 2026-08-19)

R3 was recorded as two type-level holes and deferred. Under the PL
ruling they are still holes, and they are **not** the target. The
target is the hole the language actually has and the model does not
state.

**R3a — ε unspecified (still deferred).** The checker writes
`ty_knowledge(v_ty, 0.0 - 1.0)` — sentinel `-1`. V2's `t_kraw`
demands `kvalid m` (`conf ∈ [0, 1000]`). V2 is stricter on
confidence, right on payload.

**R3b — `kadd` pins Real (still not a consumer).** V2's `t_kadd` /
`t_kmul` require `Knowledge<Real>`. The checker's Knowledge binop
returns `ty_knowledge(left_inner, …)` and keeps `T`. Do not "cover"
this by forcing a Real-only client.

**R3c — GUM across a compiled call (the target).** Madaros loses
first-order variance at a user `fn` the FO catalog does not fire
for. The catalog in `fo_register_pure_fn_transfer` registers 1 or 2
parameters; a third parameter returns without registering. Thesis
pin: `tests/run-pass/gum_fo_across_call.sio` — `rhs(c, cl, fu)` is
three arguments plus `OpSub`/`OpDiv`; `variance_of(c)` after the
loop is exactly `0`. Independent matrix: same-file `ADD2` is live
on Madaros; `ADD3`/`ADD4` print `0.000000` (`docs/audit/FO_VARIANCE_ACROSS_FN_INDEPENDENT_VERIFY_2026-08-18.md`).
This is calculation, not print. lean_single is the honesty oracle
on that additive matrix, not a universal GUM-magnitude oracle.

The model says nothing about this. `kadd_red` combines metadata on
two `kraw` cells in the same term. That is not the path the
language takes, and it is not the path that zeros.

#### Question 1 — can V2 express propagation across a call?

**No**, not the call the language has.

| Object | V2 | Language (Madaros today) |
|---|---|---|
| Application | `app f a` — CBV **beta**, one argument, substitution of a whole `kraw` | compiled `fn`, catalog lookup, silent `0` on miss |
| Arity | curried; no third-parameter abort | third parameter **does not register** |
| Payload of the call | `Knowledge<T>` cell (`kraw v m`) | peeled `f64` (`.value`) then `variance_of` on an `f64` |
| GUM rule in play | `gAddMeta` / `gMulMeta` on cells | FO transfer on unwrapped floats |
| Import / method | absent | catalog miss → `0` |

V2 can express "a lambda receives a `kraw` and beta substitutes it".
Metadata rides along because it is a field of the term. That is
source-level substitution. The defect is FO after a peel through a
*compiled* function. Identifying V2 `app` with that call is a
category error.

So the missing proposition is not another `HasTy` client. It is a
judgment **distinct from `Step.beta`**: uncertainty on either side
of a compiled application of an unwrapped payload. Working name:
`gum_across_compiled_call`. Until that object exists in V2, V2
does not specify the feature that makes Sounio not Rust. It
specifies a cell calculus that the checker uses for `measure` /
unwrap / apply-of-`Knowledge<T>`, and that is already consumed.

#### Question 2 — if it could, would the theorem be true of today's Madaros?

The primary answer is question 1. The counterfactual is still
binding, because it is the failure mode of writing the wrong
theorem:

- `id1` / `ADD2` (catalog, ≤2 args, same file): Madaros **keeps**
  FO. A V2 theorem "metadata survives beta of identity" would be
  true in both, and would **not** discriminate the defect.
- `ADD3` / `rhs(c,cl,fu)` (third parameter): Madaros **zeros**.
  Curried V2 beta would still substitute and keep metadata. A
  theorem stated of V2 `app` and read as the language would be
  **true in the model and false in Madaros** — a specification of
  a compiler that does not exist.

Do not write that theorem. Do not close R3 by citing `preservation`
on `app`. `preservation` says the reduct stays well-typed. The
defect is a live `0` that is well-typed.

#### Semantic-Lane declaration (before any Lean)

```text
Semantic-Lane-ID: v2-is-language-spec-r3
Owner: cursor-3
Concept-IDs: SOUNIO-EPISTEMIC-NUMERIC-VALUE
Intent-Preserved: Knowledge<T> is a language feature (payload +
  uncertainty), not a Lean coverage exercise. Uncertainty is not
  a value and not an arithmetic error.
Transformation: V2 is declared the specification of Knowledge<T>.
  The next owed object is gum_across_compiled_call, a judgment
  distinct from Step.beta. No constructor added in this turn.
Types-Changed: none (declaration only)
Effects-Changed: none
IR-Changed: none
Claims-Introduced: V2 does not yet specify GUM across a compiled
  call; that silence is now a named hole (R3c), not a deferred
  remark beside epsilon.
Claims-Forbidden: "V2 models Knowledge<T>" ; "the discriminating
  set is empty, so the spec is done" ; "preservation on app is
  GUM-across-call" ; "R3 is only kadd-pins-Real"
Assumptions: fo_register_pure_fn_transfer still aborts at the
  third parameter; gum_fo_across_call.sio still zeros; this lane
  does not edit lower.sio.
Write-Set: docs/audit/epistemic_calculus_spec_divergence/DISPATCH.md
Read-Set: formal/lean4/EpistemicEffectsV2.lean ;
  tests/run-pass/gum_fo_across_call.sio ;
  docs/audit/MADAROS_FO_CALL_BOUNDARY_DISPATCH_2026-08-18.md
Positive-Witness: a V2 judgment whose Madaros reading can fail
  (ADD3 / rhs arity 3 → 0)
Negative-Witness: V2.preservation on app (true, does not mention
  variance); V2.kadd_red (inline cells, not a compiled call)
Acceptance-Gate: this declaration, in this file, on a head that
  adds no Lean theorem about app/kadd. The next Lean edit is
  owed the named judgment or an explicit halt that it cannot be
  stated Mathlib-free.
Integration-Target: V2 as language spec, not a fifth consumer
Authoritative-Only-If: a proposition about compiled-call GUM
  exists in V2 and a correspondence gate can fail when Madaros
  zeros
```

**What this turn does not do.** No `lit`, no `HasTy`, no consumer,
no `--lean-consume-*`. Writing Lean now would be either glue
(`gen` for a new constructor) or the forbidden theorem (beta
read as call). Halt is the deliverable until the judgment is
designed so that it can be false of today's Madaros.

---

## §6 — Acceptance criteria

1. R1 and R2 landed. R3 recorded and explicitly deferred, not silently dropped.
   R1's remaining *fact* (V2 had no importer) is closed by
   `EpistemicEffectsV2_measure_nat.lean`, not by the banner. That close is
   an import edge, not metatheory coverage. After consumer 4 the
   cited named theorems are still **2** (`preservation`, `invKraw` —
   the mg client re-cites `invKraw`, it does not add a third name),
   **8 of 15** `HasTy` (`+ t_lit_mg`), **2 of 19** `Step`,
   **4 of 5** `IsValue` (`+ v_mg`). The Nat-shaped discriminating
   set stayed empty; the `T ≠ Nat` hole closed at `Knowledge<mg>`.
   The Nat/`mg` discriminating set is empty as a *consumer*
   question. Under the PL ruling that is not "the spec is done".
   R3c (`gum_across_compiled_call`) is the owed proposition. It
   is not `kadd`-pins-Real and it is not `Step.beta`. Spine
   extraction stays deferred until one of the three end
   conditions in §5 R1 — none has fired. Adding `tmg` to the
   shared `Ty` is not extraction. This declaration does not
   fire extraction: no new Lean module is being born.
2. `Lean Proofs` green, with V1, V2, `EpistemicEffectsV2_measure_nat`,
   `EpistemicEffectsV2_kvalue_nat`, `EpistemicEffectsV2_invkraw_nat`,
   and `EpistemicEffectsV2_invkraw_mg`
   all `@[default_target]`. Deleting
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

- **Do not "fix" the compiler under this dispatch.** The payload side of
  §2 still forbids editing `self-hosted/check/check.sio` here. The
  FO-across-call defect is real and is owned by the call-boundary
  lane (`lower.sio`), not by a silent patch from this file.
- **Do not write a Lean theorem about `Step.beta` / `t_app` and call
  it GUM-across-call.** That theorem is true of V2 and, read as the
  language, false of Madaros at arity ≥ 3. It would specify a
  compiler that does not exist.
- **Do not claim "V2 models `Knowledge<T>`"** until
  `gum_across_compiled_call` exists as an object distinct from beta.
- **Do not delete or weaken `EpistemicEffects.lean` §9.1.** A refutation is a
  result. The honest record of a false claim is more valuable here than a clean
  file, and this repository's thesis is exactly that.
- **Do not restate the refutation a fourth time.** Three statements of it
  already exist. A fourth adds citation ambiguity, not confidence.
- **Do not claim V2 "proves Sounio sound."** It proves subject reduction and
  progress for a cell calculus. The step from that to a statement about
  Madaros is R2 (payload grep) plus R3c (compiled-call GUM). Overstating
  either repeats the §1 `sorry` annotation's error in the opposite
  direction.
