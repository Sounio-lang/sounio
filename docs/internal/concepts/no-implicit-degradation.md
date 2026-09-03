<!-- docs:meta
topic_id: repo.docs.internal.concepts.no-implicit-degradation
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.no-implicit-degradation
-->

# No Implicit Degradation

Concept-ID: `SOUNIO-NO-IMPLICIT-DEGRADATION`

Status: **Hypothesis** — stated by the founder in session on 2026-08-19. It is
the generating principle behind several concepts that already exist, but
nothing in the compiler enforces it as a principle.

## Founder Intent

> No epistemic degradation is implicit. Every step down the knowledge ladder
> must be written by a person, with a reason.

Not a coercion, not a default, not an automatic combination rule. Where
knowledge becomes less than it was, a named act stands at that point and
carries a justification.

## Why this is the generator

The founder's specification names types and effects as first principles, and
the effect vocabulary was repeatedly felt to be incomplete without the missing
entries being nameable. This document records why.

Every epistemic effect in the language today — `Observe`, `Learn`, `Witness`,
`Prob`, `Audit` — names an **acquisition**. Enumerating effects by what a
computation *does* has no generator and does not close.

Enumerating **the ways knowledge degrades** does. Each degradation requires
one named act. That list is finite, inspectable, and generates the missing
vocabulary directly.

## The degradation table

Decided in session 2026-08-19. "Act" is the named operation that must stand at
the point of loss; absence of an act is the gap this table exists to expose.

| Degradation | Act | State |
|---|---|---|
| Uncertainty discarded by projection (`.value`) | mark propagates; see `SOUNIO-EPISTEMIC-ERASURE` | decided, unimplemented |
| Uncertainty restored without measuring | `attest(v, uncertainty:, because:)` | decided, unimplemented |
| Origins of different classes combined | `misturar(a, b, because:)` | decided, unimplemented |
| Exact result narrowed to floating point | — | `SOUNIO-EXACTNESS` states the invariant; no act named |
| Precision narrowed (`f256` → `f64`) | — | **no act** |
| Independence assumed where a common ancestor exists | — | **no act**; see `SOUNIO-PROVENANCE` |
| Evaluation outside the validated domain | — | **no act** |
| A computed value presented as measured | — | **no act**; the class distinction exists in `ProvEntity`, unenforced |
| A witness or proof discarded | — | **no act** |

The four rows marked **no act** are effects the founder designed and could not
name. They were not forgotten: the vocabulary pointed entirely at acquisition,
so there was nowhere for a loss to be written.

## Required Invariants

- A degradation without a named act is a gap in the language, not a
  convenience. Adding the operation silently is the defect this concept names.
- An act carries a justification. An act that only records *that* the loss
  happened, without *why*, has recorded the wrong half.
- The act is the only door. If the degraded state is also reachable by
  constructing a value by hand, the act is decorative and the guarantee is
  void. This is the same class as an unenforced ledger.
- Saturation is failure. A mark that every value acquires after two operations
  no longer discriminates at the point where discrimination was the purpose.

## Claims Forbidden

- Do not describe this principle as enforced. No compiler surface reads it.
- Do not present the three decided rows as implemented; they are recorded
  designs, and their own documents say so.
- Do not treat the degradation table as complete. It is the list as of
  2026-08-19; the generator is the principle, not the table.

## Related

- `SOUNIO-EPISTEMIC-ERASURE` — the first row, worked out in full
- `SOUNIO-PROVENANCE` — supplies the sixth and eighth rows
- `SOUNIO-EXACTNESS` — the fourth row, stated as an invariant before the
  principle behind it was named
