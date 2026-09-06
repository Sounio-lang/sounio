<!-- docs:meta
topic_id: repo.docs.research.pireus-aarchmrs-tbl-import-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-aarchmrs-tbl-import-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus AARCHMRS TBL/TBX Import Semantics v0

Date: `2026-08-27`

Stage: `SEMANTICS_FROZEN`

Language-Producer: `Sounio`

Language-Role: `SEMANTIC_AUTHORITY`

## Mandatory Order

The Arm corpus Garden seed was committed as
`19e3019da30d40640b8697d0c99b0032a322ce55` before this parser or result
existed.

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
```

Parity and claim promotion are closed.

## Complete-Stream Semantics

The normative input is the 115,312,065-byte `Instructions.json` from the open
AARCHMRS A-profile FAT 2025-12 package. Ordered chunk files are a bootstrap
transport only. The Sounio executable recomputes the SHA-256 of their complete
concatenation and accepts only:

```text
bedf5f8fc142d6232f15caaa170b9fab996a732db0b04bf4604e91fb10c3244b
```

The parser state survives arbitrary chunk boundaries, including boundaries
inside strings, escapes, primitives, objects, and arrays. Hash acceptance alone
does not select a form; structural JSON recognition and exact direct-field
pairs are both required.

## Sounio-Produced Inventory

The complete stream contains:

| Structural query | Result |
| --- | ---: |
| JSON objects | 615,423 |
| JSON strings | 2,994,616 |
| maximum nesting depth | 24 |
| `Instruction.Instruction` objects | 6,569 |
| admitted `TBL`/`TBX` forms | 11 |

The admitted forms partition as:

| Query | Result |
| --- | ---: |
| `TBL` family | 6 |
| `TBX` family | 5 |
| `TBL_advsimd` | 4 |
| `TBX_advsimd` | 4 |
| `tbl_z_zz` | 2 |
| `tbx_z_zz` | 1 |
| fixed 128-bit shape | 8 |
| scalable shape | 3 |
| one table register | 4 |
| two table registers | 3 |
| three table registers | 2 |
| four table registers | 2 |

The Pireus store contains 259 triples and all 11 forms are recovered by the
corpus query. Operation, vector-shape, table-cardinality, and family queries
agree with the inventory computed before ontology materialization.

## Fail-Closed Controls

Four Sounio-owned negative controls pass:

1. a selected record with a duplicate `operation_id` is rejected;
2. a selected operation paired with an unknown name is rejected;
3. an unterminated JSON object is rejected;
4. the SHA-256 of an empty stream is rejected as the pinned corpus digest.

Missing files, wrong chunk sizes, a wrong total length, wrong complete digest,
invalid JSON state, excessive nesting, or more than 16 admitted forms also have
stable nonzero importer errors.

## Semantic Restraint

`TBL` and `TBX` remain vendor-derived family names. Fixed and scalable are raw
shape partitions derived from exact admitted operation/name pairs. The table
cardinality is retained from the exact encoding names.

The importer deliberately emits:

```text
PIREUS_ARM_SEMANTIC_ROLE_ASSIGNMENTS count=0
```

No selector, payload, merge-source, destination-access, or lowering role is
inferred in v0.

## Validation Boundary

The rebuilt/current-source ontology wrapper was unanimous with
`provenance=rebuilt_direct`. The explicit `lean_single` Sounio path typechecked
and executed the 115 MB witness repeatedly with byte-identical output.

The semantics do not claim Apple or DGX observation, instruction availability,
instruction equivalence, material cost, backend selection, or Cayley-Dickson
acceleration.
