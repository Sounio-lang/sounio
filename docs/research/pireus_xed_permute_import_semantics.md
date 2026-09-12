<!-- docs:meta
topic_id: repo.docs.research.pireus-xed-permute-import-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-xed-permute-import-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus XED Permutation Import Semantics v0

Date: `2026-08-27`

Stage: `SEMANTICS_FROZEN`

Language-Producer: `Sounio`

Language-Role: `SEMANTIC_AUTHORITY`

## Order

The committed Garden inventory
`docs/internal/garden/seeds/2026-08-27-pireus-canonical-corpora.md` existed
before the importer was written or executed. The first successful Sounio
execution existed before this document and its receipt were created.

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
```

`PARITY_OPEN` and `CLAIM_READY` are closed.

## Input Semantics

The input is the exact Intel XED `avx512-foundation-isa.xed.txt` file from
release `v2026.08.23`, commit
`0bcb6237345c5066726dcc08b3d87928df3b5b26`. Before interpreting any record,
Sounio streams the complete 458,470 bytes through the repository's pure-Sounio
SHA-256 implementation and requires digest:

```text
e9bc0522be4c1a3a3d938eab334c47e306fe759cccf376b9dfb2b9cf7aee0038
```

A same-size mutation therefore remains a failure rather than a new corpus
version.

## Record Semantics

Only records whose first field is one of these `ICLASS` values enter the slice:

```text
VPERMI2PD
VPERMPD
VPERMT2PD
```

For those records, the parser accepts exactly eleven named fields and exact
512-bit `f64` `PATTERN`, `OPERANDS`, and `IFORM` combinations. The three raw
surfaces must agree on one of eight record kinds.

The resulting inventory is:

| Raw property | Count |
| --- | ---: |
| records | 8 |
| `VPERMI2PD` | 2 |
| `VPERMPD` | 4 |
| `VPERMT2PD` | 2 |
| destination `w` | 4 |
| destination `rw` | 4 |
| mask reads | 8 |
| register storage forms | 4 |
| memory storage forms | 4 |
| `UIMM8` selector syntax | 2 |
| register-index selector syntax | 2 |
| selector syntax deliberately unassigned | 4 |
| total operands | 32 |
| read operands | 24 |
| destination `zf64` operands | 8 |
| register-read `zf64` operands | 10 |
| memory-read `f64` operands | 4 |
| immediate reads | 2 |

The four unassigned selector-syntax records are the two `VPERMI2PD` and two
`VPERMT2PD` forms. This is intentional: raw XED operand order alone does not
establish semantic operand roles.

## Ontology Semantics

The accepted forms extend the v0.1 `TripleStore` with 82 triples, producing 246
triples total. Each record is typed as an instruction form and linked to the
pinned corpus, `ICLASS`, ISA set, raw destination access, raw selector syntax,
storage kind, and normative-vendor evidence role.

Sounio queries recover all eight forms and the `2/4/2` family partition. The
vendor corpus remains distinct from material machine observation and from
compiler-lowering evidence.

## Failure Semantics

The importer returns a stable nonzero error category for:

- missing or oversized input;
- full-file SHA-256 mismatch;
- record-brace or duplicate-field syntax error;
- unknown field in a selected record;
- missing selected-record field;
- accepted-field value or cross-field drift;
- inventory capacity exhaustion.

The executable's three negative controls independently exercise hash mismatch,
unknown field, and missing field. All are denied before an ontology result can
be accepted.

## Toolchain Boundary

The `lean_single` Sounio compiler path typechecked and executed the witness.
The rebuilt/current-source ontology checker parsed it successfully, but the
current default Madaros fallback rejected imported struct field access and
caused the wrapper to classify the combined check as `mixed/unknown`.

That disagreement is recorded as compiler/harness evidence. It does not allow
Madaros to create a replacement result, and it is not hidden as a passing
unanimous check.

## Non-Claims

No instruction semantics, encoding theorem, hardware availability, cost,
lowering, equivalence, or performance result is frozen here. Apple Silicon,
AArch64, PTX, SASS, DGX, and Darwin measurements are outside this slice.
