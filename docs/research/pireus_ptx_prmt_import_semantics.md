<!-- docs:meta
topic_id: repo.docs.research.pireus-ptx-prmt-import-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-ptx-prmt-import-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus PTX `prmt` Import Semantics

**Semantic bundle:** `pireus-ptx-prmt.v0`
**Date:** 2026-08-27
**Producing language:** Sounio
**Role:** `SEMANTIC_AUTHORITY`

## Authority Input

The sole normative byte input is the official archived CUDA 13.2.0 PTX ISA
9.2 HTML at:

```text
https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html
```

The accepted stream is 3,428,895 bytes with SHA-256
`fd013df0c9560d9f86672c379b57b30a6d5efb2eccbb0c6c487950032e6d3457`.
Sounio reads four ordered sub-1-MiB chunks, updates SHA-256 over every byte, and
admits the projection only when chunk count, total length, and digest all match.

The PDF rendering and the observed live PTX 9.3 document are provenance
references only. They do not vote on or replace this semantic input.

## Structural Grammar

The importer is an incremental HTML scanner with explicit states for tags,
quoted attributes, comments, selected-section nesting, paragraphs, code spans,
and preformatted blocks. It:

1. tracks balanced `section` nesting over the complete document;
2. parses tag names and attribute boundaries;
3. requires exactly one `section` whose `id` is
   `data-movement-and-conversion-instructions-prmt`;
4. normalizes whitespace only inside paragraph and code text nodes;
5. classifies rubric labels only after a structural `p class="rubric"` match;
6. classifies raw mode tokens only after a structurally closed `code` node;
7. counts non-empty lines only inside the syntax, semantics, and example `pre`
   blocks selected by the preceding rubric;
8. rejects unsupported selected shapes instead of dropping or guessing fields.

Substring counts do not create records or expected results.

## Sounio-Produced Projection

The frozen authority stream is:

```text
SOUNIO_AUTHORITY schema=pireus-ptx-prmt.v0 role=SEMANTIC_AUTHORITY
PIREUS_PTX_CORPUS release=CUDA-13.2.0 ptx_isa=9.2 bytes=3428895 chunks=4 error=0 sha256=fd013df0c9560d9f86672c379b57b30a6d5efb2eccbb0c6c487950032e6d3457 digest_match=1
PIREUS_PTX_SECTION id=data-movement-and-conversion-instructions-prmt selected=1 headings=1 paragraphs=166 rubrics=7
PIREUS_PTX_BLOCKS pre=3 syntax_lines=2 semantics_lines=13 example_lines=2
PIREUS_PTX_TABLES tables=2 rows=27 code_tokens=28
PIREUS_PTX_MODES f4e=1 b4e=1 rc8=1 ecl=1 ecr=1 rc16=1
PIREUS_PTX_NOTES introduced_ptx_2_0=1 target_sm_20_or_higher=1
PIREUS_PTX_ONTOLOGY triples=190 forms=1 raw_modes=6
PIREUS_PTX_NEGATIVE duplicate_section=1 selected_shape=1 malformed_html=1 capacity=1 digest=1
PIREUS_PTX_BOUNDARY sass_links=0 material_capabilities=0 lowering_claims=0 semantic_role_assignments=0
PIREUS_PTX_SUMMARY failures=0
```

The actual stream contains one blank line after each `print_int`-terminated
record because the current bootstrap printing ABI emits a newline for integer
output. Its exact byte representation, including those blank lines, hashes to
`a2276391cb7a188727fee27881334eb48c03f7c51075c2a6b9c689e822ad4cac`.

## Ontology Projection

The projection adds these evidence-level classes:

- `InstructionForm`;
- `VendorCorpus`;
- `VirtualISA`;
- `RawMode`;
- `TargetRequirement`.

The `prmt` form is linked to the pinned corpus, PTX ISA 9.2, the raw
`sm_20-or-higher` requirement, and six raw modes. The ontology contains 190
triples when layered on the frozen Pireus v0.1 store. SPARQL witnesses return
one form and six raw modes.

## Non-Claims

This bundle freezes no assertion that:

- PTX `prmt` is a vector-lane operation;
- any PTX form maps one-to-one to SASS;
- a particular DGX GPU physically implements a corresponding instruction;
- PTX acceptance establishes material availability, latency, or throughput;
- `prmt` is equivalent to the pinned x86 or Arm permutation forms;
- `prmt` lowers the Cayley-Dickson XOR formulation correctly or optimally.

Those relations remain absent, not false-by-default. A later material or
lowering lane must introduce new typed evidence and pass the mandatory order.

## Stage Boundary

This bundle is produced by `SOUNIO_EXECUTABLE` and is submitted for
`SEMANTICS_FROZEN`. It does not open `PARITY_OPEN` or `CLAIM_READY` by itself.
