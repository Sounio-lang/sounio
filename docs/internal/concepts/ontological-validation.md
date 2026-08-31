<!-- docs:meta
topic_id: repo.docs.internal.concepts.ontological-validation
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.ontological-validation
-->

# Ontological Validation

Concept-ID: `SOUNIO-ONTOLOGICAL-VALIDATION`

Status: **Hypothesis** — as a *stated principle*. Substantial machinery exists
and part of it runs; what does not exist is this principle written down anywhere
a reader or a gate can find it.

## Founder Intent

Stated in session, 2026-08-19:

> **"Practically everything must be ontologically validated — that is part of
> Sounio's soul."**
>
> <sub>Founder, in Portuguese: *"praticamente tudo deve ser ontologicamente
> validado… isso faz parte da alma do Sounio"*. The English rendering above is
> the normative text; the original is retained as attribution.</sub>

Not a feature. A commitment about what it means for a Sounio program to be
correct: a term is not merely well-typed, it is **answerable to a model of what
exists**. A quantity has a dimension because dimensions are real, not because
the compiler was taught a table. A relation holds because an axiom permits it.

This document exists because the principle was, until now, **written nowhere**.
It appears in passing across fifteen other concept documents —
`endogenous-observability`, `nonassociative-order`, `ordered-path-provenance`,
`rebracketing-authority`, `reflexive-inquiry`, the `proof-carrying-*` family and
others — and in no place that governs them. That is the same dispersion
`SOUNIO-ADMISSIBILITY` was written to end: the state from which two documents
contradict each other and nobody notices.

## Measured state (2026-08-19, `origin/main`)

This is not aspiration. It is built.

| where | how much |
|---|---:|
| compiler files mentioning ontology | **42** |
| — including | `check/ontology_side_table_cache.sio`, `ci/ontology_validation_driver.sio`, `ci/ontology_validation_debug_driver.sio` |
| stdlib files | **101** |
| CI gate scripts named `*ontolog*` | **17** |
| concept documents mentioning it | 15 |

The gates cover axiom enforcement, generated-ontology manifests, cache
compilation and frontend composition, a **typed bridge**, a **query compiler**,
a **reasoner**, unit metadata, bundle directives, CLI smoke and a hash
benchmark. A language with an ontological reasoner in its compiler is not a
language with a units table.

## The gap this document exists to record

**Three of the seventeen gates are named by any workflow. What the other
fourteen do is not established by that number.**

The count measures **direct invocation** — whether a workflow names the script —
not **coverage**, which would require establishing that no running parent invokes
it. `SOUNIO-EFFORT-LOCATION` measured the difference elsewhere and found 45
scripts that no workflow names but a running parent covers. Under its invariant
— *a number carries how it was measured, or it is not evidence* — the earlier
wording "fourteen never run" claimed more than the instrument supported and is
withdrawn here. The coverage of the fourteen is **unmeasured**.

Among the fourteen: the reasoner, the query compiler, the typed bridge, the
cache, the generated-ontology manifest, and `ontology_unit_metadata_gate.sh`.

The soul of the language is verified by infrastructure the CI does not read. If
the reasoner rots tomorrow, nothing says so.

**And there is a measured aggravator.** `madaros_ontology_enforcement_gate.sh`
exists *because* Madaros accepts axiom violations that `lean_single` refuses —
a cross-engine divergence in ontological validation that is already recorded.
So divergence in this exact area is known, and most of the gates that would
watch it are switched off.

## Required Invariants

- Ontological validation that does not run is not validation. A gate outside
  every workflow is a document, and this concept is about what the compiler
  *enforces*, not what it contains.
- The two engines must agree on what an ontology permits. A term Madaros accepts
  and `lean_single` refuses is not a tolerance difference; one of them is wrong
  about what exists.
- Units are a case of this principle, not a parallel feature. A dimension is an
  ontological commitment; `stdlib/units/qudt.sio` anchors it in a standard
  vocabulary rather than a compiler convention.
- "Practically everything" is a direction, not a measured claim. What is
  actually validated is a subset, and that subset must be nameable.

## Claims Forbidden

- Do not describe ontological validation as enforced across the language. Three
  of seventeen gates are reachable; the coverage is unmeasured and this document
  says so.
- Do not read the file counts as coverage. Counting files that mention a word
  measures reach, never the two-program test.
- Do not present the reasoner, query compiler or typed bridge as working. They
  compile-gate outside CI; whether they pass today is being measured and is not
  yet known.
- Do not quote the founder's "practically everything" as a description of the
  current state. It is the intent this concept records, and the distance between
  it and the tree is the open question.
- Do not treat this document as a specification of the ontology itself. It
  records that ontological validation is a first principle and that its
  enforcement is mostly unreachable; the model, its axioms and its bridge to the
  type system are not defined here and are not defined anywhere.

## Related

- `SOUNIO-ADMISSIBILITY` — the same dispersion problem, ended the same way
- `SOUNIO-EFFECT-DECLARATION` — silence is not consent, one layer down
- `MATURITY_LADDER` — why a gate outside CI cannot move anything up a rung
