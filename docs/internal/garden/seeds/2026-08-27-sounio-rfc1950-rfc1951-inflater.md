<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-sounio-rfc1950-rfc1951-inflater
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-sounio-rfc1950-rfc1951-inflater
-->

# Sounio: The Stream That Opens The Charts

> **Status**: Garden | **Date**: 2026-08-27 | **Authority**: founder direction

## Butterfly

Pireus has reached vendor corpora whose useful records are carried inside
compressed streams. The pinned Apple Metal feature tables use PDF
`FlateDecode`; a parser that delegates decompression to an untyped host utility
would let another language choose the bytes from which Sounio later derives
semantics.

The next reusable instrument is therefore a bounded, pure-Sounio inflater for
RFC 1950 zlib streams and RFC 1951 DEFLATE blocks.

```text
transport decompression != semantic projection
host zlib success != Sounio authority
```

The inflater opens bytes. A later corpus-specific grammar decides what those
bytes mean.

## Normative Format Coordinates

The algorithmic contracts are:

```text
RFC 1950=https://www.rfc-editor.org/rfc/rfc1950.html
RFC 1951=https://www.rfc-editor.org/rfc/rfc1951.html
```

RFC 1950 defines the zlib envelope, CMF/FLG checks, optional dictionary flag,
and Adler-32 trailer. RFC 1951 defines stored, fixed-Huffman, and
dynamic-Huffman DEFLATE blocks, literal/length and distance alphabets, and the
32 KiB backward window.

## First Consumer

The first pinned consumer remains Apple's official feature-table PDF:

```text
path=Metal-Feature-Set-Tables.pdf
bytes=3041713
sha256=9f31df15dd6827545702c5a0845f6e36e1889878cd0e534123bd70211e5c00a8
document_date=May 21, 2026
root_page_count=18
content_filter=FlateDecode
```

The PDF page tree contains three child page trees with counts `8`, `8`, and
`2`; the root declares 18 pages. A shallow host `file` probe reports only eight
and is not an authority source.

The Numbers package is not the first executable route. Although its ZIP
members are stored without ZIP compression, table content is fragmented across
private IWA streams without a stable public schema. Pireus will prefer the
documented PDF and DEFLATE grammars before considering an IWA-specific lane.

## Required Sounio Contract

The first executable must implement in Sounio:

1. a bounded least-significant-bit-first reader;
2. stored blocks with byte alignment and `LEN`/`NLEN` validation;
3. fixed Huffman literal/length and distance alphabets;
4. dynamic Huffman code-length decoding in the RFC order;
5. canonical Huffman construction with oversubscribed and incomplete-tree
   rejection;
6. literal bytes, end-of-block, length extra bits, distance extra bits, and
   overlapping LZ77 copies;
7. the 32 KiB backward-distance boundary across block boundaries;
8. zlib CMF/FLG validation with unsupported preset dictionaries rejected;
9. Adler-32 verification over the decompressed bytes;
10. explicit input, output, symbol, block, and recursion-free capacity limits;
11. deterministic error codes and Sounio-produced expected results;
12. negative witnesses for truncation, reserved block type, bad stored length,
    invalid tree, impossible distance, output exhaustion, header failure,
    dictionary refusal, and checksum mismatch.

No C zlib FFI is admitted in this semantic-authority slice. Python and Rust are
prohibited. Node, Ruby, shell, `awk`, `bc`, or a host decompressor may inspect
or transport fixtures but may not produce decoded expected bytes or acceptance
results.

## First Executable Boundary

The first Sounio executable may freeze only the inflater's byte-level behavior
and error taxonomy. It must not yet freeze:

- Apple feature names or family thresholds;
- PDF object selection or text operators;
- font or ToUnicode mappings;
- Metal instruction semantics;
- equivalence to x86, Arm, PTX, SASS, or any lowering;
- a performance claim for the inflater.

The Apple PDF parser becomes a later Sounio executable that consumes this
frozen primitive and independently verifies the pinned PDF hash.

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

This seed establishes only `GARDEN`.
