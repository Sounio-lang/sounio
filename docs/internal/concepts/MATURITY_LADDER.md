<!-- docs:meta
topic_id: repo.docs.internal.concepts.maturity-ladder
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.maturity-ladder
-->

# Maturity Ladder — five states, and how to measure one

The evidence progression in `FOUNDER_INTENT.md` and `docs/internal/garden/README.md`
is `Garden -> Hypothesis -> Executable -> Claim-ready`. This document adds the
rule that makes the ladder measurable, and the fifth state the archaeology of
2026-08-19 found by failing.

## The ladder is monotone

    Claim-ready  =>  Executable  =>  Hypothesis  =>  Garden

Each rung **requires** every rung beneath it. A kind cannot be `Claim-ready`
without being `Executable`. If no program constructs it and passes, the
highest reachable position is `Hypothesis`, however many refusals the
compiler emits.

This was not stated in the first archaeology protocol. A lane applied the
four definitions independently, correctly, and reached a position the
definitions permitted and the ladder does not: `TyF128` marked `Claim-ready`
because Madaros refuses every use of it. The defect was in the protocol.

## Beside the ladder: Reserved splits in two

A single `Reserved` state confused a **promise** with a **marker**. They must
not look the same.

### `reserved-owed` — the name is taken and someone owes the landing

    reserved-owed   refuse every use with a NAMED diagnostic; semantics
                    unimplemented; a named owner is on the hook for a named
                    technical condition.

Required fields on the contract:

| field | meaning |
|---|---|
| `Reserved-Owner` | who signs |
| `Reserved-Since` | ISO date the debt was declared |
| `Reserved-Blocked-On` | the **technical condition** that is missing — not a calendar deadline |

Age is visible (same rule as `Evidence-Does-Not-Count`): derived from
`Reserved-Since`, not git log; **does not expire**; does not fail by age; the
gate always prints the owed roster oldest-first on green and red.

### `reserved-taken` — the name is taken so nobody else defines it

    reserved-taken   refuse every use with a NAMED diagnostic; **owes nothing**.
                     Staying forever is correct behaviour.

Requires only `Reserved-Reason` (non-empty text).

### Shared refuse rule

Both forms still require negative evidence (a refuse surface). A name whose
use is not refused is not reserved — it is merely unimplemented.

### Bare `reserved`

Invalid vocabulary. Pick `reserved-owed` or `reserved-taken`. When this split
landed, **zero** concept contracts declared `Status: reserved`, so migration
cost is zero; the gate rejects bare `reserved` rather than guessing.

### Not Hypothesis, not Claim-ready

In `Hypothesis` the system is passive. In either reserved form it knows, acts,
and says why it refuses. Refusing everything is still not `Claim-ready`
(discriminating types need a correct program that passes).

### Measured instances (typekind archaeology, not yet concept Status)

`TyF128` / `TyF256` under Madaros — both refuse with `E218` (see
`tests/typekind/index.tsv`). Both are **promises** (scientific surface under
`SOUNIO-PRECISION-PRESERVATION`), so they map to **`reserved-owed`**.

**Proposed `Reserved-Blocked-On` for founder confirmation** (proposal, not fact):

| kind | Reserved-Blocked-On (proposal) |
|---|---|
| TyF128 | Madaros/native x86-64 path where a *correct* `f128` bind constructs and runs while a wrong program still refuses with a typed diagnostic — today E218 refuses all surface use including correct binds |
| TyF256 | same class for `f256` (or an explicit qd-family surface if that is the chosen representation) |

Owner/Since: founder assigns when Status is written on a concept or typekind index row.


## The test that decides a position

Write **two** programs: one that must pass and one that must fail.

| correct program | wrong program | position |
|---|---|---|
| passes | passes | label — nothing is being typed |
| **fails** | **fails** | **reserved-owed** or **reserved-taken** |
| passes | fails | **Claim-ready** — the only case that is |
| passes | passes | `Executable`, not `Claim-ready` |

**Without both programs there is no position above `Hypothesis`.** One
program never decides. A single accepting witness cannot distinguish a type
from a label; a single refusal cannot distinguish a type from a wall.

## Why this belongs in governance, not in an audit

`FOUNDER_INTENT.md` asks what information was made invisible so the system
could look simpler. A four-state ladder was simpler than the language, and
the price was a kind recorded at the wrong rung. The state existed and had
even been observed — `f128/f256` had already been called "seed reservations"
in the known-failure census hours earlier — but had no standing, so the
observation could not be carried.

When a measurement produces an impossible position, suspect the ruler before
the measurement. The impossible position is a site.
