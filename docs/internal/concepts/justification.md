<!-- docs:meta
topic_id: repo.docs.internal.concepts.justification
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.justification
-->

# Justification

Concept-ID: `SOUNIO-JUSTIFICATION`

Status: **Hypothesis** — decided by the founder in session on 2026-08-19. The
mechanism that would connect a Sounio call site to a Lean obligation does not
exist.

## Founder Intent

`SOUNIO-NO-IMPLICIT-DEGRADATION` requires that every epistemic loss carry a
justification. This document fixes what a justification *is*.

Founder ruling: the floor is a **proof obligation**. `because:` names a
theorem in `formal/`, and the build fails if that theorem does not exist or
does not close.

The consequence is the ruling, not a side effect:

> A justification that cannot be discharged as a proof does not belong in
> `because:` at all.

## Why prose was the wrong shape

A justification written as a string is inert. The compiler cannot read it,
cannot check it, and cannot tell when it stopped being true. It is a parallel
record with nothing forcing it to agree with the code — the same failure class
as a provenance ledger no value is bound to.

The apparent objection is that some justifications are empirical
(*"Kp is not measurable in vivo"*) and no compiler will ever decide them. The
ruling dissolves this rather than conceding it: **that claim was never a
justification for the mixture.** It explains why *that value* has
literature origin. It belongs at the construction of the value, in its
provenance, with citation, owner and date — not at the site where the value is
combined.

What remains at the act is whether *this combination* is valid. That is a
theorem.

## Why this is feasible

The `formal/` contract — Lean proves **algorithmic** invariants; the paper
proves probabilistic fidelity — is what makes the ruling implementable.

The obligation for combining two epistemic values is **not** "GUM propagation
is valid". That requires measure theory, requires Mathlib, and is the paper's
work. The obligation is *"these two ancestor sets are disjoint"* — a finite,
combinatorial, decidable fact **about the program**, provable with no Mathlib
dependency.

The theorem is about the program, not about the world. A rule written for a
different reason turns out to be the precondition for this one.

Measured 2026-08-19 on `origin/main`: `formal/` holds 182 `.lean` files;
`sorry` appears as an actual tactic **3 times in one file**
(`formal/lean4/SounioSedenionBipartite.lean`) — the other 165 hits are prose in
docstrings stating "Zero sorry". Mathlib is imported by 5 files.

## Required Invariants

- The floor is a discharged proof. A justification that names no obligation,
  or names one that does not close, is not a justification.
- Empirical claims live in provenance, not in `because:`. Moving one into a
  justification slot re-creates the inert string this concept exists to remove.
- The obligation is about the program. An obligation that requires modelling
  the world has been stated at the wrong layer and belongs to the paper.
- A missing theorem fails the build. A justification whose obligation is
  absent must be indistinguishable from no justification at all — silence here
  would make the whole mechanism decorative.

## What does not exist

The gap is the connection, and it is the whole implementation:

- No mechanism binds a Sounio call site to a Lean theorem name.
- `formal/` builds separately; no gate makes a Sounio compile depend on a Lean
  obligation closing.
- No obligation of this family has been written. The disjoint-ancestor
  theorem is described here and does not exist.
- The empirical half — owner, date, citation, review — has no carrier in the
  value either; see `SOUNIO-PROVENANCE`, which also does not exist in
  `Knowledge`.

## Claims Forbidden

- Do not describe any justification mechanism as implemented. None is.
- Do not cite `formal/`'s 182 files as evidence that this concept is underway.
  They prove unrelated invariants and predate the decision.
- Do not present the disjoint-ancestor obligation as proved, drafted, or
  scheduled.
- Do not read "the floor is a proof obligation" as a claim that Sounio verifies
  scientific validity. It verifies a program property; validity is the paper's.

## Related

- `SOUNIO-NO-IMPLICIT-DEGRADATION` — requires the justification this defines
- `SOUNIO-PROVENANCE` — receives the empirical half, and supplies the ancestor
  sets the obligation is stated over
- `SOUNIO-EPISTEMIC-ERASURE` — `attest(because:)` inherits this floor
