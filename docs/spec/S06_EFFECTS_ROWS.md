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

Measured state at the time of the ruling: **559 function types occur in live
`.sio` source and not one of them declares an effect.** The ruling is therefore a
change of kind, not a description. What it buys is that the question *"what does
this function argument do?"* becomes **askable**; today it is not formulable at
all, because the type has nowhere to say it.

Two things follow immediately and are recorded in 6.6 rather than assumed here:
what a function type with no effect clause means, and whether a function may
abstract over the effects of its argument.

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

## 6.2 The set is capped at eight, and the ninth is dropped in silence

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
