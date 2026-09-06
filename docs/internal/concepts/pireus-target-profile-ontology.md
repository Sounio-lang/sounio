<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-target-profile-ontology
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pireus-target-profile-ontology
-->

# Pireus Target And Material Profile Ontology

Status: proposed executable extension

Authority: founder

Concept-ID: proposed extension of `SOUNIO-PIREUS-MATERIAL-ONTOLOGY`

Registry-Status: pending registration by the active registry owner

## Semantic Lane Declaration

Semantic-Lane-ID: `pireus-v01-target-profile-20260827`

Owner: `codex/session-01a040f3-2b73-76e2-bbf7-`

Concept-IDs: proposed `SOUNIO-PIREUS-MATERIAL-ONTOLOGY`; existing
`SOUNIO-SCIENCE-RESEARCH-BOUNDARY`; existing
`SOUNIO-NONASSOCIATIVE-ORDER`

Intent-Preserved: Pireus must encompass Darwin Xeon, Apple Silicon, and DGX
without treating a named target as an observed machine, an observed feature as
an executable instruction, or a material profile as source-language semantics.

Transformation: extend the frozen v0 triple store with canonical targets,
observed machines, processor profiles, kernel-reported features, operand roles,
and destination access modes. Replace the ambiguous derived notion of one
`source_count` with separate data, selector, and mask source cardinalities.

Types-Changed: none; the extension is represented using the existing Sounio
ontology IRIs, literals, triples, patterns, and solution rows.

Effects-Changed: none.

IR-Changed: none.

Claims-Introduced: a passing v0.1 witness establishes that Sounio can distinguish
three canonical targets from observed material machines; represent the five
live Darwin Xeon profiles; query selected kernel-reported feature groups; and
distinguish a selector operand from a second payload data source.

Claims-Forbidden: Darwin as multi-ISA; fresh Apple or DGX observation; usable
instruction support from a CPU flag alone; vendor encoding correctness;
latency, throughput, frequency, scheduling, or lowering optimality; complete
target or ISA coverage; parity output used as semantic authority.

Assumptions: `/proc/cpuinfo` and `lscpu` strings observed in the five live Slurm
worker pods identify the material inputs recorded on 2026-08-27. They do not
establish OS-enabled execution or cost.

Write-Set: the v0.1 Garden seed, this extension contract,
`stdlib/hardware/pireus/target_profile.sio`, its Sounio witness, frozen semantics,
and receipt. Frozen Pireus v0 files are read-only inputs.

Read-Set: `stdlib/hardware/pireus/model.sio`, `stdlib/ontology/query.sio`, live
Slurm worker identity output, repository Apple/DGX target routing, the semantic
lane contract, and the canonical compiler resolver.

Positive-Witness: Sounio returns three canonical targets, five observed Darwin
Xeon machines, feature counts `5/4/3/3`, and exactly one synthetic `f64x8`
permutation form with one payload data source, one selector, zero mask sources,
and a write-only destination.

Negative-Witness: Apple Silicon and DGX remain canonical while each has zero
observed machines; the one-data-source query rejects the two-data-source form;
and a zero-selector query returns no form.

Acceptance-Gate: the Garden seed predates execution; the rebuilt current-source
ontology wrapper checks the Sounio import closure unanimously; two Sounio runs
are byte-identical; source, semantic, output, and toolchain hashes are frozen in
the receipt; Loom accepts the receipt before parity opens.

Integration-Target: `stdlib/hardware/pireus` after concept registration and
canonical Loom acceptance.

Authoritative-Only-If: `examples/pireus_target_profile_query.sio` is the
producer with role `SEMANTIC_AUTHORITY`, its frozen hashes match, and no parity
language, vendor corpus, shell extractor, or external LLM supplied expected
query results.

## Core Boundary

```text
canonical target         != observed machine
observed machine         != executable capability
kernel-reported feature  != measured cost
selector source          != payload data source
material profile         != source-language semantics
```

