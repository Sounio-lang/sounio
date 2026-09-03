<!-- docs:meta
topic_id: repo.docs.spec.s06-effects-rows
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.s06-effects-rows
-->

# §6 — Effects: rows and subtyping

Spec-Section: `SOUNIO-SPEC-06`
Frame: `docs/spec/E2E_SPECIFICATION_FRAME.md`

Status: **Hypothesis.** One normative statement is ruled (6.0). No conformance
test exists for it. Everything else in this section is measured state and owed
rulings, not specification.

## 6.0 Normative

> **A function type carries the effects of the function.**

Founder ruling, 2026-08-19. A function type is not merely its parameter and
result types; the effects the function may perform are part of it.

Measured state at the time of the ruling, corrected: in live `.sio` source
**381 function types occur in parameter position; 165 carry an effect clause and
216 do not** (`docs/audit/FN_TYPE_EFFECT_CLAUSE_CENSUS_2026-08-19.md`).

The surface syntax already supports the ruling. `self-hosted/parser/types.sio:717`
documents the grammar as `fn(T, U) -> V with E1, E2`, and it is used, e.g.

    fn filter8(arr: [i64; 8], pred: fn(i64) -> bool with Div, Panic) -> ...

So 6.0 is closer to **codifying existing practice** than to introducing a
capability. What it makes normative is that the clause is not optional: the
question *"what does this function argument do?"* must be answerable at every
function type, not at 165 of 381.

> **Correction, 2026-08-19.** An earlier revision of this section stated *"559
> function types occur in live `.sio` source and not one of them declares an
> effect"*. Both halves were wrong. 559 was a raw line count including
> `archive/`, `bootstrap/`, comments and string literals; and the "not one" came
> from a pattern whose return-type character class excluded the shapes that
> actually occur. The ruling stands; the state it was ruled against was
> misreported, by me, and the misreport made the ruling look more expensive than
> it is.

Two things follow immediately and are recorded in 6.6 rather than assumed here:
what a function type with no effect clause means, and whether a function may
abstract over the effects of its argument. The second is untouched by the
correction above: **no effect variable occurs in live source.** The one apparent
instance, `fn(T, U) -> V with E`, is the parser comment quoted above.

`scripts/ci/fn_type_effect_ratchet_gate.sh` freezes the bare count at 216. It
does not implement 6.0; it stops the gap from widening while 6.0 is
unimplemented.

## 6.1 What runs

Measured on `origin/main`, 2026-08-19.

The effect system that executes is a **flat set of integer identifiers**. The
live operations in `self-hosted/check/effects.sio` are:

| function | external call sites |
|---|---:|
| `has_effect_id` | 29 |
| `effect_name_to_id` | 6 |
| `effects_subset` | 2 |
| `find_missing_effects` | 2 |
| `print_effect_name` | 2 |

Membership and subset. There is no composition operation in the live path:
`extract_effects` and `merge_effects` are the file's only two functions with no
caller anywhere, internal or external.

`Confidence` is not a distinct effect. `effect_name_to_id` returns id 8 for it,
with the comment *"Confidence is an alias of Epistemic (id 8). Not a new
variant."*

### 6.1.0 CORRECTION — thirty names, twenty-nine ids, and this is not a row

Raised by an independent review of this section on 2026-08-20 (fable-1, consulted
for disagreement rather than agreement). Four criticisms; three land.

**The count. §6.1 says 23 names. It is stale.** Derivation, from
`self-hosted/check/effects.sio`:

    fn effect_named_id_max() -> i64 { 28 }

so ids run `0..=28` — **29 named effects** — and `effect_name_to_id` opens with a
hardcoded alias, *"Confidence is an alias of Epistemic (id 8). Not a new
variant."* That is **30 names over 29 identities**, and it matches the closed-list
gate (`scripts/ci/effect_name_closed_list_gate.sh`) exactly. The two numbers were
never in conflict about the world; this document was simply behind the table.

**This is not a row, and this document's own measurement says so.** A row's
defining property is a variable, unbounded tail. `current_effects: [i64; 8]` is a
fixed width. Whatever §6 specifies, the thing implemented today is a **bounded
set** with an inclusion check at the call boundary, and the title's word "rows"
describes an intention rather than the mechanism. The review put it exactly:
*writing `[i64; 8]` is already the answer to the question of whether this is a
row.*

**The two silent-drop holes are one mechanism.** The unknown-name hole and the
ninth-effect hole are both `if eff_id >= 0 && c.current_effect_count < 8`. §6.1.1
already unified them; a later dispatch of mine re-separated them, and that was a
regression against this section.

**The fourth criticism does not survive measurement, and is recorded because
deference to a reviewer is the same failure as deference to an instrument.** The
review read the `lean_single` confidence gate as `== 400` hardcoded, i.e.
validation rather than algebra. Measured on
`tests/run-pass/dissertation_pbpk28_confidence_gate.sio`, varying only N:

| `Epistemic(N)` | lean_single |
|---|---|
| 350 | passes |
| 399 | passes |
| 400 | passes |
| **401** | **refused** |

The source is `if call_cmin > 0 && call_conf > 0 && call_conf < call_cmin`
(`lean_single.sio:26489`) — an inequality against a `call_conf` computed by
`ety_conf_product(arg_conf, FN_EFF_CONF[...])`, a multiplicative composition along
the call. That is an algebra, not an equality test.

**Where the payload question belongs.** The same review makes a point this section
should adopt: `Epistemic(N)` is a **belief** gate, and §8.2.6 ruled that belief is
the value layer. So whether `Epistemic(950)` satisfies a requirement of
`Epistemic(400)` is not a §6 question about effect algebra — it is §8 wearing an
effect's syntax, and its ordering must be ruled there.

### 6.1.1 Four of the effects are bookkeeping with no consumer

Measured 2026-08-19. Of the 23 names, the ones any `has_effect_id` **decision**
consults are `ZD` (8 call sites), `Epistemic` (4), `Observe` (2), `Chaotic`,
`MultiTest`, `Hypothesis`, `Witness`, `Temporal` and `NonAssoc`.

**No decision consults `Mut`, `Alloc`, `Panic`, `Div` or `IO`.** As string
literals in `self-hosted/check/` and `self-hosted/ir/` they occur 0, 1, 0, 1 and
0 times respectively — positive control on the same command, `ZD` occurs 16.

They participate only in the generic propagation: `effects_subset` /
`find_missing_effects` require a caller to declare whatever its callee declares,
for any effect alike. Nothing ever asks *"does this function divide?"*.

`#1995` measured the other side: of **57,752** signatures, **46,219 — 80.03% —
declare only `Mut`/`Div`/`Panic`/`Alloc`**. `stdlib/math/pure.sio:245` is
`fn sin(x: f64) -> f64 with Mut, Div, Panic` — a mathematically pure function, in
a file named `pure.sio`, carrying three effects because of its loop and its
division. `Network`, `Sensor` and `Render` are recognised names with **zero**
uses. Maximum arity measured is **6**, so the eight-slot cap does not bite today.

So four fifths of all effect declarations are propagated up the call graph,
consume slots against the cap, and are read by nothing. That is the measured
shape of 6.6's "set or row" question, and it is why 6.6 now carries a third
option., and the ninth is dropped in silence

`self-hosted/check/check.sio:285` declares

    current_effects: [i64; 8],

and the guard that admits an effect reads

    if eff_id >= 0 && c.current_effect_count < 8 {

Two silent losses follow from that one line. A function that already carries
eight effects **discards the ninth without a diagnostic**. And an effect name
that does not resolve yields `eff_id = -1`, which fails the same guard and is
**also discarded without a diagnostic** — so `with SomeEffectThatDoesNotExist`
contributes nothing and says nothing.

Sounio names more than eight effects. The cap is therefore reachable by ordinary
code, not a theoretical bound.

## 6.3 Row polymorphism exists in the production tree and has no entry point

`self-hosted/check/effects_row.sio` is 84 lines in the **live** checker
directory — not in the orphan tree. Its header states the design:

> The row tail is encoded as the `inner` field of `TyEffectRow`; an empty tail
> is represented by `TyUnit`, and `TyEffectVar` represents an open row.

It defines four functions. Three of them — `effect_row_label_eq`,
`effect_row_contains`, `effect_row_subset` — call **one another** and have
**zero callers outside the file**. The chain is closed and has no entrance. The
fourth, `check_handler_coverage`, is the file's stated purpose and has no caller
at all, not even internally.

So row polymorphism is not "designed but unimplemented". It is **implemented and
unreachable**.

## 6.4 The gate that guards it tests that the file exists

`scripts/archive/sprint17a_row_poly_gate.sh`, case 8:

    if [ -f self-hosted/check/effects_row.sio ]; then
      record "effects_row_sio_exists" "pass" "found"

The check is file existence. It would pass on an empty file. This is recorded
here rather than in an audit because it bears on the section directly: the
strongest evidence that row polymorphism was *believed* to be in place is a gate
that never tested it.

## 6.5 The orphan tree

`self-hosted/effects/` — `checker.sio` (2,056 lines), `handlers.sio` (2,991),
`types.sio` (375), `mod.sio` (15) — totals **5,437 lines with zero importers**.
Instrument: `^use effects::`, validated in the same command against
`^use parser::` (155) and `^use ir::` (117).

## 6.6 Rulings owed

- **What does a bare function type mean under 6.0?** If `fn(f64) -> f64` denotes
  a *pure* function, then 559 existing types become purity claims overnight, and
  the claim is already false at live call sites: of the functions actually passed
  to higher-order functions, `sin`, `cos`, `exp` and `sqrt` are pure while
  `my_sin`, `tc_sin` and `tc_cos` declare `with Mut, Panic, Div` — and `my_sin`
  is passed six times. If instead a bare type denotes *any* effects, the ruling
  buys nothing, because the type still cannot be relied upon. A third option is
  that a bare type is **refused**, making the clause mandatory.

- **May a function abstract over its argument's effects?** 6.0 makes the question
  live; it does not answer it. Measured pressure: `my_sin` is mathematically a
  pure `f64 -> f64` and carries three effects **because of how it is
  implemented** — a loop (`Mut`) and division (`Div`). Sounio's effect set does
  not separate observable effect from implementation mechanism, so effectful
  numeric functions are the norm rather than the exception. Under 6.0 without
  abstraction, one `deriv` cannot serve both `sin` and `my_sin`.

- **Are these one system or two?** 6.1.1 measures that `Mut`, `Div`, `Panic` and
  `Alloc` gate no decision and account for 80% of declarations, while the effects
  that *are* consulted are the observable ones. Separating the two axes would
  make effect abstraction rare rather than mandatory. It is **not free**: the
  four must then live somewhere (inferred — which costs interprocedural
  inference; a second clause — which is two annotation surfaces forever, and
  every future feature must choose a side; or removed — which loses their E035
  propagation), and the boundary is not crisp (`Alloc` is plausibly *observable*
  on a GPU) while being expensive to revise once assignments are made. The
  measured cost of not deciding is currently low: max arity 6 against a cap of 8.
  **Correction of an earlier claim in session:** splitting does *not* close the
  silent-drop hole. Validating names against a closed list closes it, and that is
  independent of any split.

- **Is the effect annotation a set or a row?** A flat set with subset is what
  runs. A row discipline with an open tail is what `effects_row.sio` implements
  and nothing calls. These are different type systems, not two maturities of
  one.
- **Is eight a designed bound or an implementation limit?** If designed, the
  specification says so and the ninth effect must be **refused with a named
  diagnostic** rather than dropped. If not designed, the cap is a silent
  truncation in the subsystem the language is named for.
- **Must an unresolvable effect name be refused?** Today `with X` for unknown
  `X` type-checks and contributes nothing. Under `SOUNIO-EFFECT-DECLARATION`
  this is already answerable, and the answer is not the current behaviour.

## Claims forbidden

- Do not describe Sounio as having row-polymorphic effects. The code exists; no
  caller reaches it.
- Do not quote an effect count as a capability without naming the eight-slot
  cap.
