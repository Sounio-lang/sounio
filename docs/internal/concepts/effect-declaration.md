<!-- docs:meta
topic_id: repo.docs.internal.concepts.effect-declaration
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.effect-declaration
-->

# Effect Declaration

Concept-ID: `SOUNIO-EFFECT-DECLARATION`

Status: **Hypothesis** — founder ruling of 2026-08-19. Not implemented, and it
**may not be implemented before the tail census below completes**.

## Founder Intent

> `with X` requires `X` to be built in, or declared in scope.

An effect name the compiler does not know is refused by name. Today it is
accepted in silence.

## What the measurement established

The ruling was not taken on principle. It was taken because a measurement
closed off every other answer.

**The bodies cannot distinguish.** Asked whether the 2,789 functions declaring
`with Mod` mutate — in which case `Mod` was a misspelling of `Mut` — or perform
modular arithmetic — in which case `Mod` is a real effect that never landed —
the classification came back **`INDETERMINATE`**: *"multi-modal; collapsing it
to a single hypothesis is not supported by the data"*. No inspection of what
the code *does* separates the two.

If the distinction is not in the body, it must come from a **declaration**.

**And Sounio already has the mechanism.** `effect Choice { fn pick() -> bool }`
parses, typechecks, and is usable in `with` (`tests/frontend/effect_user_defined_basic.sio`,
`tests/run-pass/effect_handler_basic.sio`). Six user-defined effects exist in
the tree: `Choice`, `Counter`, `Fail`, `Fetch`, `Logger`, `Storage`.

**So the hole was never "there is no way to declare an effect". It is that
declaring is not required** — and therefore nobody did.

## The negative control that settles it

```
fn f() with NoSuchEffectX     ->  rc=0, check: OK
```

An entirely invented name compiles. `effect_name_to_id` returns `-1` and both
collection paths drop it. This is not about `Mod`: **the `with` line accepts any
unknown name in silence**, so a declaration that does nothing is
indistinguishable from one that works.

## Measured scale

Counting names appearing after `with` across `stdlib/`, `self-hosted/` and
`examples/` on `origin/main`:

| recognised (the 29 ids) | | not recognised | |
|---|---:|---|---:|
| `Mut` | 33418 | **`Mod`** | **2261** |
| `Panic` | 32785 | **`GUM`** | **91** |
| `Div` | 30490 | **`Uncertainty`** | **20** |
| `IO` | 6735 | **`GetTid`** | **13** |
| `Alloc` | 4586 | *(tail unmeasured)* | ? |
| `Epistemic` | 316 | | |

`GUM` and `Uncertainty` are used **more** than `Observe` (33), `Witness` (21),
`Learn` (13) or `Temporal` (11) — all of which have a place in the vocabulary.
And all three of `GUM`, `Uncertainty` and `Epistemic` first appear in the same
commit, `b6d03ae18a`, **2025-12-25 — day one**. They were designed together and
only one reached the compiler. Nothing recorded that the other two had not.

## Sequencing — this may not land yet

The refusal cannot be implemented before the **tail census**: the complete set
of names appearing after `with` that are neither among the 29 ids nor declared
by a visible `effect X { }`. That measurement is in flight. Landing the refusal
first would reject names nobody has counted, and the resulting breakage would
be indistinguishable from the defect it is meant to expose.

When it lands, each affected site resolves one of two ways, and **only its
author can say which**: write `effect Mod { }`, or discover the intent was
`Mut`. That is the point — the measurement proved nobody can decide it for them.

## Required Invariants

- An unknown effect name is refused by name, never accepted silently. The
  diagnostic states whether the name is unknown or merely undeclared, since the
  fix differs.
- Declaring must remain cheap. If declaring an effect were expensive, the
  refusal would push people to delete the annotation rather than write it — and
  a deleted `with` is worse than an unread one.
- The refusal does not decide intent. The compiler says the name is undeclared;
  it never guesses that `Mod` meant `Mut`. Proximity may be *suggested*, never
  applied.
- A name reaching a count in the thousands without recognition is evidence
  about the compiler, not about the people who wrote it.

## Claims Forbidden

- Do not describe this as implemented. `with NoSuchEffectX` compiles today.
- Do not treat `Mod` as decided. Its measurement returned `INDETERMINATE`; the
  founder held it out of the phase-2b enum for exactly that reason (#1963).
- Do not read the counts above as the full set. The tail is **unmeasured**, and
  the table says so.
- Do not present the `GUM`/`Uncertainty` finding as abandonment. Nothing
  recorded a decision to drop them; the absence of a record **is** the finding.

## Related

- `SOUNIO-NO-IMPLICIT-DEGRADATION` — the same principle one layer up: silence is
  not consent
- `SOUNIO-ADMISSIBILITY` — needs the effect row to say what happened upstream,
  which an unread `with` cannot do
