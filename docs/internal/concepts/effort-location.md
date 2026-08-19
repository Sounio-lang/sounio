<!-- docs:meta
topic_id: repo.docs.internal.concepts.effort-location
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.effort-location
-->

# Effort Location

Concept-ID: `SOUNIO-EFFORT-LOCATION`

Status: **Hypothesis** — a governing principle, stated after twelve concepts
were written without it. It explains why they take the shape they do.

## Founder Intent

> A record puts the effort on whoever reads. A gate puts it on whoever acts.

The difference between the two is **not rigour**. It is where the work lands.
And since whoever reads is always someone in the future — with less context and
more haste — **anything that depends on being read degrades over time by
construction**.

## Why this had to be written

Every substantial defect measured on 2026-08-19 was a defect of effort
location, not of knowledge. In each case the information was already available
and already correct:

| defect | how long | the information was |
|---|---|---|
| `with GUM` inert, `effect_name_to_id` returns `-1` | 8 months | in the source since day one (`b6d03ae18a`) |
| 14 of 17 ontology gates named by no workflow | unknown | in `scripts/ci/`, readable |
| 2,800 `with Mod` that no compiler reads | years | in 360 files |
| `serialize.sio` sized `[IrFunction; 1024]` against `IR_MAX_FUNCS` 16384 | 5 weeks | one `git grep` away |
| shipped `bin/madaros` cannot compile the shipped tree | 2 days | printed by the compiler itself |
| `main` red for 9 hours on a stale `known-failure` label | 9 hours | in the job log, first line: `Fail: 0` |

None of these was hidden. Every one was a record that nothing forced anybody to
read.

## The demonstration that produced it

The author of this document built a dispatch ledger to detect stacking a second
task on a busy agent lane — an error made twice that day. The ledger worked: it
reported `SEM_ECO` correctly, in real time, on screen.

**The same error was then made twice more, while the ledger was reporting it.**

Not a failure of information. The information was on the screen. It was a
failure of *where the effort sat*: the ledger asked to be read, and reading is
what does not happen under load. The fix was not better detection — it was a
`livre <lane>` check that **refuses before dispatch**.

Four occurrences in one hour, by the person who built the detector, is the
cleanest available evidence that reports do not change behaviour and refusals do.

## The relation to the rest of the corpus

This is why the founder chose **blocking in both directions** for the
concept-status gate rather than warning; why `SOUNIO-NO-IMPLICIT-DEGRADATION`
requires a written act rather than a logged note; why
`SOUNIO-EFFECT-DECLARATION` refuses an unknown name rather than counting it;
why a malformed `Evidence-Does-Not-Count` must be **redder than its absence**.

Each of those looked like a separate severity choice. They are one choice, made
repeatedly: **move the effort from the reader to the actor.**

## Required Invariants

- A guarantee that depends on someone reading is not a guarantee. It is a hope
  with documentation.
- When a defect class recurs, the fix is not a better report. Ask what would
  have refused, and build that instead.
- The escape hatch must cost more than compliance. If declaring badly is cheaper
  than declaring well, everyone declares badly — which is why malformed is
  redder than absent throughout this corpus.
- Records remain necessary. This is not an argument against writing things down:
  it is an argument that writing them down is **not sufficient**, and that every
  record should name what would enforce it, or say plainly that nothing does.

## Claims Forbidden

- Do not read this as "documentation is useless". Every concept in this registry
  is a record; the claim is that a record without an enforcer decays, not that
  it is worthless.
- Do not treat "add a gate" as universally correct. A gate on an unmeasured
  criterion refuses correct work, and refusing correct work costs more than a
  report nobody reads.
- Do not cite this to bypass measurement. The order is unchanged: measure, then
  decide, then enforce. A gate built before its criterion is understood is the
  failure this concept would otherwise cause.
- Do not present the table above as complete. It lists what was measured in one
  day by one session.

## Related

- `MATURITY_LADDER` — a rung claim nothing enforces is exactly this defect
- `SOUNIO-SIGNAL-DIRECTION` — an enforcer that fires in the wrong direction
- `SOUNIO-NO-IMPLICIT-DEGRADATION` — the first concept to place the effort on
  the actor, before this principle had a name
