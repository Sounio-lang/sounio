<!-- docs:meta
topic_id: repo.docs.internal.concepts.provenance
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.provenance
-->

# Provenance

Concept-ID: `SOUNIO-PROVENANCE`

Status: **Hypothesis** — a substantial implementation exists and is
disconnected; the design recorded here was decided in session on 2026-08-19
and is not implemented.

## Founder Intent

Provenance is not bookkeeping. It is load-bearing for the arithmetic.

```sounio
let cl = medido(...)          // measured clearance
let a  = derivar_a(cl)
let b  = derivar_b(cl)
let r  = combinar(a, b)       // var(a) + var(b)
```

Adding variances assumes independence. `a` and `b` share an ancestor, so the
sum **underestimates** the true uncertainty — and the more a pipeline branches
and rejoins, the more confident the result appears, in the opposite direction
from the truth. Without provenance the compiler has no way to know they are
related. With it, that is a query.

This is what makes GUM propagation correct rather than optimistic.

## Measured state (2026-08-19, `origin/main`)

A provenance subsystem exists: `stdlib/epistemic/prov.sio` (11 KB),
`ledger.sio` (17 KB), `slsa.sio` (14 KB), `merkle.sio` (8 KB),
`claim_ast.sio`, `audit_runtime.sio`, plus **11** `proof_carrying_*.sio`
files. `ProvEntity` carries a W3C-PROV-shaped record: an origin class
(measured / literature / computed / input), value, uncertainty, confidence,
timestamp and reference hash.

**The only importers are three demos** — `examples/epistemic/prov_demo.sio`,
`ledger_demo.sio`, `slsa_demo.sio`. No stdlib module, no compiler surface and
no dissertation surface imports any of it.

**`Knowledge` has no provenance field.** It holds `value`, `variance`,
`confidence`. `stdlib/epistemic/README.md` draws `└── provenance: Provenance`,
but that is the document, not the struct. Provenance therefore exists as a
**parallel record with nothing forcing it to agree with the values**.

## Design

**Two carriers, because they answer different questions.**

- **Class travels in the type.** `Knowledge<T, Origem>` — measured, literature,
  computed, input. Checked before a binary exists, zero runtime cost.
- **Instance travels as an id.** A `ProvId` in the value answers the question
  the type cannot reach: *is this measurement THAT measurement* — the
  correlation query above.

**Mixing is an act, not a rule.** There is no automatic combination rule for
`Origem`. Combining values of different origin classes is refused, and passes
only through an explicit `misturar(a, b, because:)` that assumes the mixture,
exactly as `attest` assumes restored uncertainty. This is chosen over a lattice
(`weakest element wins`) or a set-valued `Misto` specifically because nothing
then saturates on its own: each mixture is written down.

**The constructor is the only door.** If a `Knowledge` can be built by hand
with a `ProvId` filled in, the id is an unsourced assertion and every
correlation query becomes decorative — returning correct answers about a graph
that no longer corresponds to the values. This is a precondition, not a
refinement: without it neither carrier is worth implementing.

**Ancestor bitset over graph query (proposal, not founder-decided).** Carrying
a fixed-width root-ancestor bitset in the value reduces "do these share an
origin?" to one `AND`. A model exceeding the width must **refuse detectably**;
a silently truncated ancestor set reports independence that does not hold,
which is the same failure class as a capacity literal that drops past its cap.

## Required Invariants

- Provenance that cannot be trusted is worse than none: it converts an unknown
  into a false assurance. A ledger nothing forces to agree with the values is
  in this state today.
- Variances may not be summed as independent without an ancestor check, once
  the check exists.
- Origin class is not confidence. A literature value may be more reliable than
  a bad measurement; the class records where it came from, not how good it is.

## Claims Forbidden

- Do not describe the existing subsystem as integrated. It is imported by three
  demos and by nothing else.
- Do not cite `README.md`'s `provenance: Provenance` as the shape of
  `Knowledge`. The struct has no such field.
- Do not present the ancestor-bitset proposal as founder-decided; it is an
  engineering suggestion recorded for evaluation.
- Do not claim any correlation-aware GUM propagation exists.

## Related

- `SOUNIO-NO-IMPLICIT-DEGRADATION` — the principle `misturar` instantiates
- `SOUNIO-EPISTEMIC-ERASURE` — `attest` needs provenance, which is why these
  two were designed in the same session
