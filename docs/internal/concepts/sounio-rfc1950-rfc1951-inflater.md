<!-- docs:meta
topic_id: repo.docs.internal.concepts.sounio-rfc1950-rfc1951-inflater
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.sounio-rfc1950-rfc1951-inflater
-->

# Sounio RFC 1950/RFC 1951 Inflater

**Concept-ID:** `SOUNIO-RFC1950-RFC1951-INFLATER`
**Status:** executable candidate; Loom acceptance required
**Owner:** Sounio compression substrate
**Semantic authority:** Sounio

## Boundary

This concept is a bounded byte-stream inflater implemented and witnessed in
Sounio. It exists so later Pireus importers can consume compressed vendor
artifacts without making a foreign decompressor an expected-result oracle.

```text
DEFLATE bytes != PDF objects != Apple feature semantics
successful inflation != trusted vendor claim
```

The implementation recognizes an RFC 1951 stream directly or an RFC 1950 zlib
wrapper. It does not parse PDF, ZIP, gzip, Numbers/IWA, or Apple capability
tables.

## Frozen Profile

The executable profile admits:

- stored, fixed-Huffman, and dynamic-Huffman DEFLATE blocks;
- canonical Huffman trees up to 15 bits;
- literal bytes, length/distance copies, and overlapping LZ77 copies;
- a 32,768-byte maximum back-reference distance;
- dynamic literal-only blocks with an empty distance alphabet;
- zlib CM 8 headers with a window declaration no larger than 32 KiB;
- Adler-32 verification for zlib-wrapped output;
- at most 65,536 input bytes, 262,144 output bytes, and 4,096 blocks.

Preset zlib dictionaries are recognized and refused with a dedicated error.
The caller may set a smaller output limit before execution.

## Fail-Closed Conditions

The executable classifies and refuses:

- negative lengths, capacity violations, or input cells outside `0..255`;
- truncated bitstreams;
- reserved block type 3;
- mismatched stored-block `LEN` and `NLEN`;
- oversubscribed or inadmissibly incomplete Huffman trees;
- invalid literal/length or distance symbols;
- a distance beyond 32 KiB or before the beginning of produced output;
- malformed zlib headers, preset dictionaries, Adler-32 mismatch, trailing
  bytes, or excessive block count.

The empty distance alphabet exception is scoped to dynamic table construction.
If a length symbol attempts a copy through that empty alphabet, symbol decoding
still fails before any copy occurs.

## Authority Witness

`examples/sounio_rfc1950_rfc1951_inflate.sio` constructs its stored and dynamic
fixtures in Sounio. Its dynamic fixtures cover both an overlapping
length/distance copy (`AAAA`) and a literal-only block with an empty distance
alphabet (`Z`). A fixed-Huffman zlib fixture produces `hello` and passes its
Adler-32 trailer.

The witness also executes deliberate negatives for reserved block type, stored
length mismatch, impossible distance, truncation, zlib header, dictionary,
checksum, output capacity, byte domain, oversubscribed tree, and incomplete
tree. No Python, Rust, foreign zlib, or disposable-language oracle produces the
expected results.

## Evidence State

| Stage | State |
| --- | --- |
| `GARDEN` | Established by commit `bfc24ea6d3b1efaf535bca8cffd73e1d9acc90e7`. |
| `SOUNIO_EXECUTABLE` | Implemented by the inflater and executable witness. |
| `SEMANTICS_FROZEN` | Proposed by the paired semantics and receipt; requires Loom acceptance. |
| `PARITY_OPEN` | False. |
| `CLAIM_READY` | False. |

This concept is compression substrate only. Pireus PDF ingestion must begin
with its own Garden artifact after this stage is accepted.
