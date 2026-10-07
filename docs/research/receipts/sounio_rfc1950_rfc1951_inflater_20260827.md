<!-- docs:meta
topic_id: repo.docs.research.receipts.sounio-rfc1950-rfc1951-inflater-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.sounio-rfc1950-rfc1951-inflater-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sounio RFC 1950/RFC 1951 Inflater Receipt

**Date:** 2026-08-27
**Concept-ID:** `SOUNIO-RFC1950-RFC1951-INFLATER`
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

The Garden artifact is commit
`bfc24ea6d36131eede6c2843d6926ff6113e43e6`. This receipt covers the
Sounio executable and its proposed frozen semantics only.
`PARITY_OPEN=false` and `CLAIM_READY=false`.

## Authority Source

| File | SHA-256 |
| --- | --- |
| `docs/internal/garden/seeds/2026-08-27-sounio-rfc1950-rfc1951-inflater.md` | `607d41f557f255c85e18cc3c08b4dc253e8f3f117835e30fa30ddbe798fa20ee` |
| `stdlib/compress/inflate.sio` | `2d788a553d3b0255c67789e1c8c2b2fc997762a76212db9613eaa1d8a5f762e3` |
| `examples/sounio_rfc1950_rfc1951_inflate.sio` | `94e81ef7189b883e9baad056e30ef2fd6cf585880fa3beb46b526db4f89751da` |
| `docs/internal/concepts/sounio-rfc1950-rfc1951-inflater.md` | `eb65a1b0ce0a050a2cc7e3a4ceada118019eef52145849a9020e1c9c8a1ef264` |
| `docs/research/sounio_rfc1950_rfc1951_inflater_semantics.md` | `b22759a6d516c150b43456c34152cbf9d04421fa423b5782a45835f5d4479c5f` |

The executable example hash is bound to Loom as the Sounio source receipt.
The paired semantics document is bound as frozen semantics. The module hash is
separately recorded here so the implementation cannot drift unnoticed.

## Toolchain

```text
engine=lean_single
wrapper_path=bin/souc
wrapper_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
compiler_path=bin/souc-lean-single-x86_64
compiler_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
rebuilt_checker_wrapper=/tmp/pireus-v01-ontology-validation-souc
rebuilt_checker_wrapper_sha256=ac705d2a14710b7034d08bd742c159acd1129b4f3760d69a522ee05fb7933395
```

The exact five-line base toolchain record bound to Loom hashes to
`2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e`.

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

The exact hardware record hashes to
`fd73771f3ac0ac200b6d93641b95744ea673c80a83f6d588adb8a19c5e1cf8f0`.

## Commands

```bash
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check \
  examples/sounio_rfc1950_rfc1951_inflate.sio

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  examples/sounio_rfc1950_rfc1951_inflate.sio

/tmp/pireus-v01-ontology-validation-souc check \
  examples/sounio_rfc1950_rfc1951_inflate.sio

/tmp/pireus-v01-ontology-validation-souc run \
  examples/sounio_rfc1950_rfc1951_inflate.sio
```

The exact direct-run command record, with final `LF`, hashes to
`800e6a385ed36dbf4c469acbb4aa33397b9766d12c2dfc3f32f6d9e8ead93350`.
Two direct runs were byte-identical. The rebuilt check returned `verdict=ok`,
`provenance=rebuilt_direct`, and unanimous driver/fallback evidence. The
rebuilt run's program-output suffix beginning at `SOUNIO_AUTHORITY` was
byte-identical to both direct runs. No raw ELF was invoked.

## Sounio-Produced Result

The normalized logical records are:

```text
SOUNIO_AUTHORITY schema=sounio-rfc1950-rfc1951-inflate.v0 role=SEMANTIC_AUTHORITY
INFLATE_STORED output=ABCD bytes=4 blocks=1 ok=1
INFLATE_FIXED output=hello bytes=5 blocks=1 checksum=1 ok=1
INFLATE_DYNAMIC output=AAAA bytes=4 blocks=1 literals=1 copies=1 max_distance=1 checksum=1 ok=1
INFLATE_DYNAMIC_LITERAL output=Z bytes=1 blocks=1 copies=0 empty_distance_tree=1 ok=1
INFLATE_NEGATIVE reserved=1 stored_length=1 distance=1 truncated=1 zlib_header=1 dictionary=1 adler32=1 output_capacity=1 byte_domain=1 oversubscribed_tree=1 incomplete_tree=1
INFLATE_BOUNDARY byte_stream_only=1 pdf_semantics=0 apple_semantics=0 ffi_zlib=0
INFLATE_SUMMARY failures=0
```

The exact stream includes bootstrap integer-printing line breaks and hashes to:

```text
a49f3da323278c4b20861cf468d8b8efe9515f0799c835b7ef63c6c92188565f
```

## Positive Coverage

The Sounio witness constructs and decodes:

- a final stored block producing `ABCD`;
- a fixed-Huffman zlib stream producing `hello` with matching Adler-32;
- a dynamic-Huffman zlib stream producing `AAAA` through one literal followed
  by an overlapping length-3, distance-1 copy;
- a literal-only dynamic stream producing `Z` with an empty distance alphabet.

The dynamic bitstreams and their code-length alphabets are emitted by Sounio
inside the authority witness. No foreign compressor supplies their expected
result.

## Negative Evidence

All eleven deliberate negatives execute in Sounio and return their exact error
class:

- reserved block type;
- stored `LEN`/`NLEN` mismatch;
- copy distance before produced output;
- truncated stored payload;
- invalid zlib header;
- preset dictionary request;
- Adler-32 mismatch;
- output-limit exhaustion;
- input cell outside the byte domain;
- oversubscribed Huffman tree;
- incomplete one-symbol tree with length two.

A deliberate Python-oracle frame with the same receipt bindings was refused by
Loom before interpreter execution or requested semantic effect.

## Material Note

The checker reports a 12,583,328-byte stack frame in the fixture-heavy `main`.
Both compiler routes execute it successfully on the recorded machine. The
warning belongs to witness storage, where several fixed-capacity input/result
values coexist; it is not evidence of an inflater semantic failure. A future
consumer should reuse bounded buffers instead of reproducing the witness's
parallel fixture layout. This warning is retained as material evidence rather
than suppressed.

## Evidence Boundary

This receipt establishes a bounded pure-Sounio byte inflater and its local
positive and negative witnesses. It does not establish a PDF parser, Apple
feature-table projection, observed hardware capability, instruction ontology,
material cost, cross-ISA equivalence, lowering correctness, or performance
claim. The pinned Apple PDF has not yet been decoded by this bundle.

Lean, Koka, C++, and Haskell parity were not opened. External LLM review was
not invoked: this internal compression substrate changes no mathematical or
clinical claim and is not an external-facing publication.

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
| Sounio executable source | `94e81ef7189b883e9baad056e30ef2fd6cf585880fa3beb46b526db4f89751da` |
| Frozen semantics | `b22759a6d516c150b43456c34152cbf9d04421fa423b5782a45835f5d4479c5f` |
| Toolchain record | `2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e` |
| Hardware record | `fd73771f3ac0ac200b6d93641b95744ea673c80a83f6d588adb8a19c5e1cf8f0` |
| Command record | `800e6a385ed36dbf4c469acbb4aa33397b9766d12c2dfc3f32f6d9e8ead93350` |
| Sounio result | `a49f3da323278c4b20861cf468d8b8efe9515f0799c835b7ef63c6c92188565f` |

The Sounio-owned runtime selftest passed all 33 cases. The operational runtime
remained the fixture-matched realization of frozen Loom semantics:

```text
loom_semantics_sha256=16e283166d29d6b18ed690b000e2eb595a7d965e4357553a8380714486429fff
runtime_sha256=208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60
decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
decision_sha256=ad57cad1376867f5cec01323e50d39319c7fb8ba0458a6066086d186a045b8cb
```

The deliberate Python-oracle frame was denied:

```text
exit_code=110
decision=SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SOUNIO_EXECUTABLE
decision_sha256=6fb4b46368e5dae161164f82e73ef0803084ae7a5d5cd8ec39588a1b9b44281d
```

This local decision admits `SEMANTICS_FROZEN`. Canonical Concept-ID
registration and evidence acceptance remain with the active Loom owner. The
pre-commit `action=10 COMMIT` frame at `SEMANTICS_FROZEN` was also allowed with
the same complete receipt bindings and canonical decision hash.
