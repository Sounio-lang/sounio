<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-graph-identity-composition
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pireus-graph-identity-composition
-->

# Pireus Graph Identity Composition

Concept-ID: `SOUNIO-PIREUS-GRAPH-IDENTITY-COMPOSITION`

Status: `SEMANTICS_FROZEN`

Semantic-Lane-ID: `pireus-graph-identity-20260827`

## Intent

Compose the independently frozen Pireus stores without equating unrelated
terms that happen to use the same local integer and without inflating queries
with copied parent closures.

The semantic authority is the Sounio pair:

```text
stdlib/hardware/pireus/graph_identity.sio
examples/pireus_graph_identity_composition.sio
```

## Identity Rule

A local term is interpreted only through a graph term-reference entry:

```text
(producer graph, local integer, term sort)
    -> namespace owner
    -> lifted identity
```

Graph production and term ownership are separate. The language-owned RDF type
term, Pireus core terms, parent-owned foreign references, producer-owned terms,
and literal vocabulary identities remain distinct registry classes.

## Composition Rule

Every child store must begin with a bit-exact copy of its declared parent
store. The projection retains every copied triple as an inherited source
occurrence, but canonical composed queries range over the deduplicated lifted
graph. Source graph, parent graph, producer concept, source index, local terms,
source hash, literal value bits, lifted terms, and canonical-triple index remain
on each occurrence.

## Frozen Result

The first Sounio execution admitted seven producers, eight namespace owners,
805 graph term references, 650 canonical lifted triples, and 1,621 source
occurrences. Of those occurrences, 971 are inherited parent copies. The
canonical graph contains 290 triples with more than one retained occurrence.

The complete store-level collision census contains 24 distinct local IRI keys
and 24 owner pairs. Six keys are in the derived `7033xx` subject space. Every
collision is between the execution-engine owner and the AARCHMRS owner; the
lifted identities remain distinct.

## Evidence Boundary

This concept establishes graph identity, parent-closure verification,
collision-free composition, occurrence provenance, literal-shape preservation,
and sort-safe composed joins. It establishes no instruction equivalence,
processor observation, capability inheritance, lowering choice, or performance
claim.

Lean, Koka, C++, Haskell, and external review cannot redefine this registry or
its expected result. `PARITY_OPEN` remains closed.
