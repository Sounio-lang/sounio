<!-- docs:meta
topic_id: repo.docs.research.pireus-v0-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-v0-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus v0 Semantic Contract

Semantic-Lane-ID: `pireus-v0-material-ontology-20260827`

Owner: `founder`

Concept-IDs: proposed `SOUNIO-PIREUS-MATERIAL-ONTOLOGY`,
`SOUNIO-SCIENCE-RESEARCH-BOUNDARY`, `SOUNIO-NONASSOCIATIVE-ORDER`

Stage: `SEMANTICS_FROZEN`

## Authority

`examples/pireus_vector_capability_query.sio` is the first executable and sole
semantic producer for Pireus v0. Its domain model is
`stdlib/hardware/pireus/model.sio`; its storage and query substrate is the
existing `stdlib/ontology/query.sio`.

No parity language, vendor database, LLVM metadata, disposable-language parser,
or external LLM supplied a capability, expected match, or negative result.

## Existing Ontology Substrate

Pireus does not introduce another ontology engine. It uses the existing Sounio
types and operations:

```text
TripleStore
Triple
SparqlQuery
TriplePattern
SolutionRow
sparql_execute
```

The v0 model adds only domain vocabulary and query construction.

## Vocabulary

Pireus v0 represents these ontology classes:

```text
Capability
Architecture
Operation
LaneScope
EvidenceRole
```

It represents these capability properties:

```text
hasArchitecture
hasOperation
hasElementBits
hasLanes
hasSourceCount
hasLaneScope
hasEvidenceRole
```

Every v0 capability carries an evidence role. The only accepted role in the
first store is `SyntheticSounioWitness`.

## Synthetic Catalog

The store contains two capability individuals:

```text
SyntheticF64x8OneSourcePermutation
SyntheticF64x8TwoSourcePermutation
```

Both belong to a synthetic architecture. Neither denotes a real instruction,
encoding, processor, ABI, latency, throughput, or compiler lowering.

The two-source individual is not unused decoration. It is the negative control
that makes rejection by an exact one-source query observable.

## Exact Query Contract

The first client requirement is derived from the frozen Cayley-Dickson SIMD
DAG but is represented here without a processor assignment:

```text
operation     = lane permutation
element bits  = 64
lanes         = 8
sources       = 1
lane scope    = intrachunk
evidence role = synthetic Sounio witness
```

The positive witness requires exactly one result and requires it to be the
one-source individual. It separately checks that the two-source control is not
present in that result.

The control query changes only `sources` from one to two and requires exactly
the two-source individual. The unsupported-shape query changes `lanes` from
eight to sixteen and requires zero results.

## Compiler Routing

The repository's canonical `ontology_query_compile_gate.sh` concatenates the
query kernel with an exercising Sounio main and passes under the public Madaros
resolver. Direct `souc check` of an importing module on this branch reports
pre-existing cross-module `Seq` method/index errors in `ontology::query`.

Following the ontology-specific repository contract, Pireus v0 uses the
rebuilt/current-source ontology validation wrapper. That wrapper reports
unanimous `verdict=ok` between its rebuilt checker driver and fallback and then
compiles and executes the imported module closure.

This is classified as a validation-wrapper routing distinction, not a Pireus
ontology-kernel failure. The default concatenated query gate remains green.

## Semantic Boundary

Intent-Preserved: material differences become explicit and queryable without
allowing material data to redefine a frozen semantic DAG.

Transformation: instantiate Sounio ontology individuals for synthetic vector
capabilities and execute exact multi-pattern queries.

Types-Changed: none.

Effects-Changed: none.

IR-Changed: none.

Claims-Introduced: Sounio can represent and exactly query the Pireus v0
synthetic capability vocabulary using its existing ontology query kernel.

Claims-Forbidden: real ISA coverage; opcode semantics; encoding correctness;
instruction availability; Apple-specific support; measured cost; optimal
schedule; sub-quadratic Cayley-Dickson multiplication.

Assumptions: exact IRI and literal-identifier matching in the live finite query
kernel; all v0 capability facts are synthetic by construction.

Positive-Witness: one exact `f64x8`, one-source permutation match.

Negative-Witness: zero two-source false positives and zero matches for the
unsupported sixteen-lane shape.

Acceptance-Gate: two byte-identical extracted Sounio outputs and hashes binding
the executable, Pireus model, ontology query kernel, this semantics document,
output, toolchain, hardware, command, language, and role.

Integration-Target: remain in the Sounio monorepo through schema stabilization;
consider `sounio-lang/pireus` only for a later independently versioned corpus.

Authoritative-Only-If: Sounio is `SEMANTIC_AUTHORITY`, hashes match, and all
capabilities remain classified by evidence role.

## Next Boundary

No vendor ingestion begins before this v0 contract is frozen and accepted by
the canonical Loom authority gate. The first vendor slice should add declared
facts separately from material observations and must retain source version,
license, content hash, architecture, feature set, and evidence role.
