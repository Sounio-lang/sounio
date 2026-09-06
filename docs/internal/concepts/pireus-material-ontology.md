<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-material-ontology
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pireus-material-ontology
-->

# Pireus Material Ontology

Status: proposed executable concept

Authority: founder

Concept-ID: `SOUNIO-PIREUS-MATERIAL-ONTOLOGY`

Registry-Status: pending registration by the active registry owner

## Semantic Lane Declaration

Semantic-Lane-ID: `pireus-v0-material-ontology-20260827`

Owner: `codex/session-01a040f3-2b73-76e2-bbf7-`

Concept-IDs: proposed `SOUNIO-PIREUS-MATERIAL-ONTOLOGY`; existing
`SOUNIO-SCIENCE-RESEARCH-BOUNDARY`; existing
`SOUNIO-NONASSOCIATIVE-ORDER`

Intent-Preserved: processor-specific material differences must become
queryable without allowing a backend, parity language, or cost model to define
or erase the source program's semantics.

Transformation: introduce a hardware-domain ontology over Sounio's existing
triple-store and query substrate, beginning with synthetic capability
individuals and exact requirement matching.

Types-Changed: none in v0; Pireus concepts are represented as ontology IRIs,
properties, individuals, and literal identifiers.

Effects-Changed: none.

IR-Changed: none.

Claims-Introduced: a passing v0 witness establishes only that Sounio can store
and query a synthetic material-capability vocabulary with positive and negative
matches.

Claims-Forbidden: real instruction support; correct vendor encoding; measured
latency or throughput; optimal scheduling; complete ISA coverage; Apple
microarchitecture knowledge; parity output used as semantic authority.

Assumptions: the live `ontology::query` triple store and SPARQL-like executor
preserve exact IRI and literal identifiers for the v0 finite witness.

Write-Set: the Pireus Garden seed, this contract,
`stdlib/hardware/pireus/model.sio`, the first executable example, its frozen
semantics, and its receipt.

Read-Set: `stdlib/ontology/query.sio`, the Cayley-Dickson SIMD DAG receipt, the
semantic lane contract, and the canonical compiler resolver.

Positive-Witness: an exact `f64x8`, one-source, intrachunk permutation query
returns the one matching synthetic capability.

Negative-Witness: the same query returns no two-source capability and returns
no capability for an unsupported lane count.

Acceptance-Gate: the Garden seed exists before execution; Madaros checks and
runs the Sounio witness twice; outputs are byte-identical; source, semantics,
and output hashes are frozen in a Sounio-authority receipt.

Integration-Target: `stdlib/hardware/pireus` after concept registration and
canonical Loom acceptance.

Authoritative-Only-If: Sounio is the producing language with role
`SEMANTIC_AUTHORITY`, all receipt hashes match, and no parity producer supplied
expected query results.

## Core Boundary

Pireus classifies evidence as part of the ontology. A capability individual
without sufficient evidence remains queryable as declared, synthetic, or
unknown, but cannot be promoted silently to material fact.

```text
semantic capability != material observation
material observation != optimal schedule
missing evidence     != negative evidence
```
