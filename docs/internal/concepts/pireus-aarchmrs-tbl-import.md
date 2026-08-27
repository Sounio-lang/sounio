<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-aarchmrs-tbl-import
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pireus-aarchmrs-tbl-import
-->

# Pireus AARCHMRS TBL/TBX Import

Proposed Concept-ID: `SOUNIO-PIREUS-AARCHMRS-IMPORT`

Status: `SEMANTICS_FROZEN_PENDING_REGISTRY_AND_LOOM_ACCEPTANCE`

Semantic-Lane-ID: `pireus-aarchmrs-tbl-import-20260827`

## Intent

Import the first Arm instruction family from the official open AARCHMRS corpus
without allowing a vendor example, text search, parity implementation, or
chunking command to define Pireus semantics or expected results.

The semantic authority is the Sounio pair:

```text
stdlib/hardware/pireus/aarchmrs_import.sio
examples/pireus_aarchmrs_tbl_import.sio
```

## Pinned Authority Input

```text
package=AARCHMRS_OPENSOURCE_A_profile_FAT-2025-12.tar.gz
archive_sha256=4dc5da62a5c856d7b1086b895075f54807f821ea21a333049cb0f40f9479cecc
file=Instructions.json
file_bytes=115312065
file_sha256=bedf5f8fc142d6232f15caaa170b9fab996a732db0b04bf4604e91fb10c3244b
schema=2.7.4
build=518
ref=2025-12_rel
license=BSD-3-Clause
```

The package is vendor evidence. Sounio owns byte reconstruction, hash
verification, JSON recognition, selected-record admission, Pireus projection,
ontology queries, negatives, and the canonical output.

## Transport Boundary

The frozen `lean_single` bootstrap `read_file` primitive is bounded below the
size of `Instructions.json`. A mechanical byte split creates 116 ordered files:

```text
part-000.part ... part-115.part
```

The first 115 contain 1,000,000 bytes each; the last contains 312,065 bytes.
Sounio reads them in numeric order and accepts them only when the reconstructed
length and SHA-256 equal the pinned complete file. The split cannot select
records, count forms, assign roles, or produce the result.

## Accepted Grammar

The importer carries a streaming JSON state across chunk boundaries. It
validates containers, key/value states, strings and escapes, literals, number
grammar, root completion, and a maximum depth. It distinguishes direct object
fields from identically named fields in descendants.

An object enters the first Pireus slice only when it has exactly one direct
`_type`, `name`, and `operation_id`, with:

```text
_type=Instruction.Instruction
```

and one of these exact name/operation pairs:

| Names | Operation ID |
| --- | --- |
| `TBL_asimdtbl_L1_1` through `TBL_asimdtbl_L4_4` | `TBL_advsimd` |
| `TBX_asimdtbl_L1_1` through `TBX_asimdtbl_L4_4` | `TBX_advsimd` |
| `tbl_z_zz_1`, `tbl_z_zz_2` | `tbl_z_zz` |
| `tbx_z_zz_` | `tbx_z_zz` |

A selected operation with an unknown name, a selected name with a mismatched
operation, duplicate required fields, malformed JSON, hash drift, missing
chunks, or capacity overflow fails closed.

## Frozen Projection

The Sounio result contains 11 forms:

```text
family:       TBL=6, TBX=5
operation:    TBL_advsimd=4, TBX_advsimd=4, tbl_z_zz=2, tbx_z_zz=1
vector shape: fixed_128=8, scalable=3
table count:  one=4, two=3, three=2, four=2
```

These are raw structured distinctions tied to exact vendor fields. No operand
is assigned a Pireus selector, payload, merge, destination, or lowering role.

## Evidence Boundary

This concept establishes corpus identity, exact selected shapes, structural
counts, and an ontology projection. It does not establish:

- equivalence to an Intel XED permutation form;
- equivalence between Advanced SIMD and SVE/SME operations;
- availability on Apple Silicon or DGX hardware;
- an Apple GPU instruction set;
- instruction execution, cost, throughput, or latency;
- a Cayley-Dickson lowering or acceleration claim.

`PARITY_OPEN` remains closed until Loom accepts the frozen Sounio receipt and
the proposed Concept-ID is registered.
