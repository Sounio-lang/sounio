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
| 14 of 17 ontology gates named by no workflow¹ | unknown | in `scripts/ci/`, readable |
| 2,800 `with Mod` that no compiler reads | years | in 360 files |
| `serialize.sio` sized `[IrFunction; 1024]` against `IR_MAX_FUNCS` 16384 | 5 weeks | one `git grep` away |
| shipped `bin/madaros` cannot compile the shipped tree | 2 days | printed by the compiler itself |
| `main` red for 9 hours on a stale `known-failure` label | 9 hours | in the job log, first line: `Fail: 0` |

None of these was hidden. Every one was a record that nothing forced anybody to
read.

**¹ How that row was measured, and why it understates.** By `git grep -c
"<basename>" -- .github/` — that is **direct invocation**, not coverage. A gate
can be reached without being named: called from inside another gate, from an
aggregator, from an umbrella. A same-day census of the 443 workflow-unnamed
scripts found **45 covered by a running parent** — not dead, not disconnected,
*included*. So "named by no workflow" is a lower bound on what runs, and any
figure derived from it is understated by an unknown amount. A transitive-closure
measurement is in flight.

The correction belongs in this document rather than in an erratum, because the
error is an instance of what the document is about: a number crossed a boundary —
from `git grep` to argument — and **lost its measurement conditions on the way**.
Nobody wrote "104 of 547 by direct invocation, which understates coverage"; the
round figure was written, and from there it was treated as fact.

That is `SOUNIO-EPISTEMIC-ERASURE` outside the compiler, committed by the author
of this corpus on the same day it was specified. And it is the harder kind to
catch, because **the wrong number supported a conclusion that remains true** —
there really are far more gates written than wired. A faulty measurement backing
a sound conclusion has nothing in its result that screams.

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

## The criterion — mechanical

What decides whether something needs a gate is **asymmetry, not importance**. A
critical thing that fails loudly does not need one; a small thing that fails
silently does.

Three yes/no questions about any invariant `I`:

    S  SILENT     if I is violated, does the system continue and produce
                  output that looks valid?
    G  GROWING    does the cost of finding and repairing it grow with
                  elapsed time or with accumulated work?
    R  REACHABLE  is there a cheap mechanical check that would refuse?

And the decision:

| S | G | R | verdict |
|---|---|---|---|
| ✓ | ✓ | ✓ | **gate required** |
| ✓ | ✓ | ✗ | **record, and name what would enforce it** — the honest gap |
| ✓ | ✗ | — | record suffices: the cost is flat, someone finds it eventually |
| ✗ | — | — | **no gate**: it announces itself. If the announcement is being *misread*, the defect is in the signal, not in the absence of a gate |

Importance appears nowhere. It is not a term.

### Worked against the day that produced it

| finding | S | G | R | verdict | what was actually done |
|---|:-:|:-:|:-:|---|---|
| `with GUM` inert (8 months) | ✓ | ✓ | ✓ | gate | `SOUNIO-EFFECT-DECLARATION` — refuse undeclared names |
| `serialize.sio` `[IrFunction; 1024]` vs 16384 | ✓ | ✓ | ✓ | gate | capacity coherence, wired to the constant |
| shipped `bin/madaros` cannot build the tree | ✓ | ✓ | ✓ | gate | `shipped_compiler_selfhost_gate.sh` |
| 14 ontology gates outside CI | ✓ | ✓ | ✓ | gate | reachability, one workflow line each |
| concept status vs evidence | ✓ | ✓ | ✓ | gate | `concept_status_gate.sh`, both directions |
| `main` red 9h on a stale label | **✗** | — | — | **no gate** | `SOUNIO-SIGNAL-DIRECTION` — the signal fired correctly and was misread |

The last row is the test that matters. The criterion **declines** to prescribe a
gate for the loudest, most expensive incident of the day, because nothing was
silent: the log said `Fail: 0` on its first line. Adding a gate there would have
been effort spent where the system was already working, and the real repair —
making direction legible — is a different kind of fix. A criterion that
prescribed a gate for every painful event would not be a criterion.

### Where it refuses to answer

`R` is the honest failure mode. When loss is silent and the cost grows but no
cheap mechanical check exists, the verdict is **not** "build an expensive gate".
It is: record it, and state in the record that nothing enforces it. That row is
the reason `Claims-Forbidden` exists throughout this corpus — it is what a
record does when it cannot become a gate yet.

## Required Invariants

- A guarantee that depends on someone reading is not a guarantee. It is a hope
  with documentation.
- When a defect class recurs, the fix is not a better report. Ask what would
  have refused, and build that instead.
- The escape hatch must cost more than compliance. If declaring badly is cheaper
  than declaring well, everyone declares badly — which is why malformed is
  redder than absent throughout this corpus.
- A number carries how it was measured, or it is not evidence. `19%` and
  `19% by direct invocation, transitive coverage unmeasured` are different
  claims; only the second can be checked, and only the second degrades
  visibly when its method turns out to be wrong.
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
