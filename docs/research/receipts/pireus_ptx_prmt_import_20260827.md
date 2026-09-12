<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-ptx-prmt-import-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-ptx-prmt-import-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus PTX `prmt` Import Receipt

**Date:** 2026-08-27
**Concept-ID:** `SOUNIO-PIREUS-PTX-PRMT-IMPORT`
**Language:** Sounio
**Role:** `SEMANTIC_AUTHORITY`
**Requested stage:** `SEMANTICS_FROZEN`

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

The Garden source pin is commit `71fcaa020171a9baf404a05c6ff537abf11db8f9`.
This receipt covers the next two stages only. `PARITY_OPEN=false` and
`CLAIM_READY=false` remain explicit.

## Authority Source

The Sounio authority surfaces are:

| File | SHA-256 |
| --- | --- |
| `stdlib/hardware/pireus/ptx_import.sio` | `ca2760d539c4602c85841ac8475a9ffd8a2f760313a8169faf99a32956063bba` |
| `examples/pireus_ptx_prmt_import.sio` | `a21248170c321b55423db8cc4afbe0965dc1ddb180fc5afa2a8b339bacbf29fa` |
| `docs/internal/concepts/pireus-ptx-prmt-import.md` | `cf84fb2dd5d168d09fdfc8472928a27b7cf4528b2b6fb544c6f96b07b6950579` |
| `docs/research/pireus_ptx_prmt_import_semantics.md` | `1454e6a212f320fbf4194b3cbb220a30abed56fbf5e8041ce076b7dee5cae697` |

The executable source hash, not this prose receipt, is the source field bound
to Loom frame `9020`. The paired semantics document is the frozen-semantics
field.

## Vendor Corpus

```text
release=CUDA 13.2.0
ptx_isa=9.2
html_url=https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html
html_bytes=3428895
html_sha256=fd013df0c9560d9f86672c379b57b30a6d5efb2eccbb0c6c487950032e6d3457
html_last_modified=Sat, 04 Apr 2026 19:38:39 GMT
html_etag="1cd98e8eb716453c209c1e34fad90980"
pdf_url=https://docs.nvidia.com/cuda/archive/13.2.0/pdf/ptx_isa_9.2.pdf
pdf_bytes=20208675
pdf_sha256=6d136dbaa3f72bc82e42593c5a1a214977cfc4eeba88282b76f284c06f26e254
```

The NVIDIA notice restricts reproduction and does not grant a document IP
license. The raw HTML and PDF are not stored in Git. The PDF is a provenance
cross-reference, not a second semantic source.

## Byte Transport

The native bootstrap `read_file` limit required a mechanical split below 1
MiB:

```bash
split -d -a 3 --additional-suffix=.part -b 1000000 \
  /tmp/pireus-ptx-13.2.0/index.html \
  /tmp/pireus-ptx-13.2.0/chunks-v1/part-
```

This produced four chunks: three at 1,000,000 bytes and one at 428,895 bytes.
Shell concatenation was only a transport-integrity observation. The Sounio
executable independently read every chunk in order, reconstructed the full byte
count, computed SHA-256, parsed the HTML structure, selected the section,
materialized the ontology, and produced the expected result.

## Toolchain

```text
public_wrapper=bin/souc
public_wrapper_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
execution_engine=lean_single
compiler=bin/souc-lean-single-x86_64
compiler_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
rebuilt_checker_wrapper=/tmp/pireus-v01-ontology-validation-souc
rebuilt_checker_wrapper_sha256=ac705d2a14710b7034d08bd742c159acd1129b4f3760d69a522ee05fb7933395
ontology_query_kernel=stdlib/ontology/query.sio
ontology_query_kernel_sha256=e36f9d7bb4e16dd7c68a69dd51ae5f2db96d9bd8209bf61483c9b3ee88ac8cbb
sha256_kernel=stdlib/crypto/sha256.sio
sha256_kernel_sha256=6c5c6895f2d3b094ea114ee3ba894c535cb12e7822e7c902fbd52771aac7537a
```

## Execution Hardware

```text
os=Linux 7.0.2-5-pve
architecture=x86_64
cpu_model=Intel Xeon Gold 6526Y
sockets=2
cores_per_socket=16
threads_per_core=2
logical_cpus=64
```

## Commands

```bash
/tmp/pireus-v01-ontology-validation-souc check \
  examples/pireus_ptx_prmt_import.sio

/tmp/pireus-v01-ontology-validation-souc run \
  examples/pireus_ptx_prmt_import.sio -- \
  /tmp/pireus-ptx-13.2.0/chunks-v1/part-

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check \
  examples/pireus_ptx_prmt_import.sio

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  examples/pireus_ptx_prmt_import.sio \
  /tmp/pireus-ptx-13.2.0/chunks-v1/part-
```

The rebuilt check returned `verdict=ok`, `provenance=rebuilt_direct`, and an
unanimous driver/fallback witness. Its run log includes compiler diagnostics;
the program-output suffix beginning at `SOUNIO_AUTHORITY` was byte-identical to
the `lean_single` authority stream. Two direct authority runs were also
byte-identical.

## Sounio-Produced Result

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

The exact stream, including bootstrap integer-printing newlines, hashes to:

```text
a2276391cb7a188727fee27881334eb48c03f7c51075c2a6b9c689e822ad4cac
```

## Negative Evidence

All five deliberate parser negatives are Sounio functions and return `1`:

- a duplicate selected section;
- an unknown selected-section rubric/shape;
- malformed and unbalanced HTML;
- fixed-capacity tag overflow;
- the empty SHA-256 digest against the pinned corpus digest.

The first draft of the duplicate-section negative scanned two complete
synthetic selected sections through a helper returning the full parser state.
The corpus import and ontology queries completed, but that negative triggered
`rc=139` before a receipt was emitted. The run was discarded. The final witness
seeds the already-seen section state and feeds a second structural opening tag,
exercising the same duplicate guard without the unstable large-state return.
All final runs use `./bin/souc run`; no raw ELF is invoked.

## Evidence Boundary

This receipt establishes one pinned PTX vendor-document record, six raw mode
tokens, one raw virtual-ISA link, and one raw target-requirement link. It does
not establish SASS identity, DGX material availability, cost, semantic operand
roles, cross-ISA equivalence, lowering correctness, or Cayley-Dickson speedup.

Lean, Koka, C++, Haskell, and external LLM parity/review were not invoked.
External LLM offload reviews invoked: none; this internal ontology importer adds
no mathematical, clinical, or external-facing claim.

## Canonical Acceptance Normalization

The source-lane handoff declared `diff-check=PASS`, but an independent
`git diff --check` found Markdown hard-break whitespace in this receipt, the
concept contract, and the semantics document. Canonical Loom acceptance removed
only that trailing whitespace. The Sounio sources and produced result did not
change. Consequently, the accepted contract and semantics hashes above replace
the source-lane hashes `37c74ae8...7921ed8` and `cc6a8484...1372e34c`.
The canonical acceptance receipt records both histories and binds the normalized
semantics hash to a fresh Loom frame.

## Loom Admission

The frozen semantics were submitted to Loom frame `9020` with:

```text
stage=2 SOUNIO_EXECUTABLE
action=3 FREEZE_SEMANTICS
language=1 Sounio
role=1 SEMANTIC_AUTHORITY
policy_state=1 available
semantic_write=1
expected_result_write=1
parity_receipt_valid=0
review_promoted=0
exception_and_waiver_fields=0
guardian_fields=0
parent_semantics_sha256=absent
waiver_sha256=absent
```

The complete frame bound these receipt fields:

| Field | SHA-256 |
| --- | --- |
| Sounio executable source | `a21248170c321b55423db8cc4afbe0965dc1ddb180fc5afa2a8b339bacbf29fa` |
| Frozen semantics | `1454e6a212f320fbf4194b3cbb220a30abed56fbf5e8041ce076b7dee5cae697` |
| Toolchain record | `2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e` |
| Hardware record | `fd73771f3ac0ac200b6d93641b95744ea673c80a83f6d588adb8a19c5e1cf8f0` |
| Command record | `49aa26a27c7b3defd96d517c77c45978d46590a6a32b923c070a97cabf4cfba1` |
| Sounio result | `a2276391cb7a188727fee27881334eb48c03f7c51075c2a6b9c689e822ad4cac` |

The runtime selftest passed all 33 Sounio-owned cases. The operational runtime
remained the fixture-matched realization of the frozen Loom semantics:

```text
loom_semantics_sha256=16e283166d29d6b18ed690b000e2eb595a7d965e4357553a8380714486429fff
runtime_sha256=208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60
decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
decision_sha256=ad57cad1376867f5cec01323e50d39319c7fb8ba0458a6066086d186a045b8cb
```

A deliberate Python-oracle frame with the same receipt bindings was denied
before an interpreter or requested effect ran:

```text
exit_code=110
decision=SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SOUNIO_EXECUTABLE
decision_sha256=6fb4b46368e5dae161164f82e73ef0803084ae7a5d5cd8ec39588a1b9b44281d
```

This local admission accepts `SEMANTICS_FROZEN`; canonical Concept-ID
registration and evidence acceptance remain with the active Loom owner.
