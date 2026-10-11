<!-- docs:meta
topic_id: repo.docs.research.sounio-rfc1950-rfc1951-inflater-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sounio-rfc1950-rfc1951-inflater-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sounio RFC 1950/RFC 1951 Inflater Semantics

**Semantic bundle:** `sounio-rfc1950-rfc1951-inflate.v0`
**Date:** 2026-08-27
**Producing language:** Sounio
**Role:** `SEMANTIC_AUTHORITY`

## Normative Sources

The byte grammar is derived from the IETF specifications:

- RFC 1950, *ZLIB Compressed Data Format Specification version 3.3*:
  `https://www.rfc-editor.org/rfc/rfc1950.html`
- RFC 1951, *DEFLATE Compressed Data Format Specification version 1.3*:
  `https://www.rfc-editor.org/rfc/rfc1951.html`

Those documents define the format. The first executable interpretation and
the expected fixture results in this bundle are Sounio artifacts.

## State Machine

The decoder state is the tuple:

```text
(input position, input end, pending bits, pending-bit count,
 output limit, error)
```

Bits are consumed least-significant bit first. Each block reads `BFINAL` and
`BTYPE`, increments a bounded block count, and dispatches to exactly one of:

1. byte-aligned stored bytes after validating `LEN xor NLEN = 65535`;
2. the RFC fixed literal/length and distance alphabets;
3. dynamic alphabets reconstructed from `HLIT`, `HDIST`, `HCLEN`, and the
   code-length repeat symbols 16, 17, and 18.

Canonical Huffman construction counts each bit length, rejects
oversubscription, admits a complete code set or the RFC one-symbol one-bit
exception, and orders symbols by length then symbol value. A dynamic distance
alphabet may be empty only so a literal-only block can execute; an attempted
copy against it fails during symbol decoding.

Literal symbols append one byte. Symbols 257 through 285 derive a length,
decode distance symbols 0 through 29, validate the 32 KiB history boundary,
and copy one byte at a time so overlapping LZ77 references are preserved.
Symbols 286 and 287 and distance symbols 30 and 31 are refused.

## Zlib Wrapper

The wrapper requires compression method 8, `CINFO <= 7`, and an FCHECK value
divisible by 31. `FDICT` is a recognized but unsupported profile feature and
returns `SOUNIO_INFLATE_ERR_DICTIONARY`. After the exact DEFLATE slice is
consumed, Sounio computes Adler-32 over produced bytes and compares it to the
big-endian trailer.

Every admitted input cell is in `0..255`. The API is fixed-capacity and has no
FFI call, allocator dependency, host zlib dependency, filesystem access, or
implicit fallback.

## Sounio-Produced Result

The authority stream is:

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

The bootstrap integer printer places line breaks around printed integer fields.
The exact byte stream, rather than the normalized display above, hashes to:

```text
a49f3da323278c4b20861cf468d8b8efe9515f0799c835b7ef63c6c92188565f
```

Two direct authority runs were byte-identical. The rebuilt validation wrapper
produced an identical program-output suffix.

## Non-Claims

This bundle does not claim:

- support for preset zlib dictionaries, gzip, ZIP, PNG filters, PDF objects,
  object streams, cross-reference streams, or Numbers/IWA containers;
- unbounded input or output;
- formal parity in Lean, effect parity in Koka, or material parity in C++;
- that any compressed Apple document has already been decoded;
- that inflated bytes establish Apple hardware capability, instruction
  equivalence, latency, throughput, or lowering correctness.

The pinned Apple Metal feature-table PDF is a future consumer and does not
define or vote on this byte-stream result.

## Stage Boundary

This bundle is produced by `SOUNIO_EXECUTABLE` and submitted for
`SEMANTICS_FROZEN`. `PARITY_OPEN=false` and `CLAIM_READY=false` remain fixed.
