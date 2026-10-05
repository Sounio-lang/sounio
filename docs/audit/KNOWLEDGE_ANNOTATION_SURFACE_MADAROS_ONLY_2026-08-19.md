<!-- docs:meta
topic_id: repo.docs.audit.knowledge-annotation-surface-madaros-only-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.knowledge-annotation-surface-madaros-only-2026-08-19
-->

# The rich `Knowledge<…>` annotation surface exists only in Madaros

## The question that produced this

Founder ruling 2026-08-19 gives `Input` a keyword. Before implementing, the risk
to check was: **if only Madaros gains the keyword, a valid program compiles on one
engine and not the other** — the divergence that already costs 437 of 1,527
greens (`#1985`).

## Answer

**`lean_single` does not need the keyword, because it has no provenance parsing
to extend.** Adding `Input` to Madaros creates **no new divergence**: this axis
has been fully divergent from the start.

Measured on `origin/main`, 2026-08-19, over
`self-hosted/compiler/lean_single.sio`:

| symbol | occurrences |
|---|---:|
| `Knowledge` | 118 |
| `measure` | 66 |
| `variance_of` | 5 |
| `ExactlyPrivate` | 3 |
| **`KnowledgeTypeInfo`** | **0** |
| **`EpsilonBound`** | **0** |
| **`ValidityCondition`** | **0** |
| **`AstProvenanceKind`** | **0** |

The first four are the positive control: the seed carries plenty of epistemic
content, so a zero in the second group is a measurement and not a dead command.
`Derived`, `Computed`, `Measured` and `Input` are likewise **0** in that file —
the three keywords that already work are already Madaros-only.

## The larger finding

The seed knows `Knowledge<T>` as a **bare** constructor. It has none of the
annotation-component machinery: no epsilon bound, no validity condition, no
provenance kind.

So every epistemic claim expressed through an annotation —
`Knowledge<f64 measured>`, an `ε` bound, a `valid_while(…)` — is **invisible to
the engine CI actually runs** (`#1978`: the Full Test Suite runs `souc-stage2`,
which is lean_single).

This changes how `#1985`'s 437-of-1,527 should be read. Part of what CI does not
test is not a *difference in result between two engines*. It is a **language
surface one engine does not have**. A test exercising an annotated `Knowledge`
type cannot disagree between the engines; it can only fail to parse on one.

## What this does and does not license

- **Licenses:** implementing the `Input` keyword in Madaros alone, without
  waiting on a seed decision. Nothing regresses, because nothing on this axis
  currently agrees.
- **Does not license:** describing annotated `Knowledge` types as a property of
  *Sounio*. They are a property of *Madaros*. Under
  `SOUNIO-GATING-ENGINE`, a claim about them must name the engine.
- **Does not license:** reading the divergence census as fully explained. This
  accounts for the annotation surface; it says nothing about the rest of the 437.

## Instrument

`git grep -c <symbol> origin/main -- self-hosted/compiler/lean_single.sio`, with
the four-symbol positive control run in the same command. Word-boundary matching
was not needed: the symbols are distinctive identifiers.
