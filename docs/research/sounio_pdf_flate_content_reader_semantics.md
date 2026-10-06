<!-- docs:meta
topic_id: repo.docs.research.sounio-pdf-flate-content-reader-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sounio-pdf-flate-content-reader-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sounio PDF Flate Content Reader Semantics

**Semantic bundle:** `sounio-pdf-flate-content.v0`
**Date:** 2026-08-27
**Producing language:** Sounio
**Role:** `SEMANTIC_AUTHORITY`

## Authority Input

The sole normative corpus byte stream is the pinned Apple PDF at:

```text
https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
```

The admitted stream is 3,041,713 bytes with SHA-256
`9f31df15dd6827545702c5a0845f6e36e1889878cd0e534123bd70211e5c00a8`.
Sounio uses `file_size`, reads the binary into the compiler-supported dynamic
byte buffer, copies exactly the declared size into an O(1)-indexed global word
buffer, and computes SHA-256 over every byte. A short read becomes zero-tail
digest drift and is refused before PDF parsing.

## Structural Semantics

The executable finds the final `%%EOF`, then the preceding `startxref`, and
requires that the named offset begin a classic `xref` table. It parses every
subsection entry, rejects duplicate identities, requires complete state for
objects `0..Size-1`, rejects entries beyond `/Size`, and refuses `/Prev`.

Indirect object dictionaries are opened only at in-use xref offsets. Balanced
dictionary scanning skips comments, nested dictionaries, literal strings, and
hex strings. Catalog and page objects must be plain dictionary objects ending
in `endobj`.

The page walk is iterative and ordered. A visited-object table rejects cycles;
the `/Parent` backlink must agree with the `/Kids` edge. For every leaf page,
Sounio walks parent links to the root and increments actual descendant counts
for each page-tree node. Every actual count must equal its direct `/Count`.

## Stream Selection

Only direct `/Contents` references from admitted leaf pages are selected.
Their target dictionary must provide a direct positive `/Length` and one or
two direct `/Filter /FlateDecode` entries. Two entries are accepted only when
identical.

The payload begins after the required line ending following `stream` and has
exactly `/Length` bytes. `endstream` must follow. A normal `endobj` is accepted;
the pinned corpus instead omits it, so the alternative is admitted only when
the next non-whitespace offset is exactly another in-use xref object offset.

Selected payloads are passed to `compress::inflate::sounio_inflate_zlib`.
Inflater error `e` maps to PDF error `100 + e`. No host PDF library, zlib,
foreign process, or fallback participates.

## Frozen Inventory

```text
xref_offset=2924864
xref_in_use=5833
xref_declared_size=5834
catalog_object=5525
pages_root_object=5524
page_tree_nodes=4
pages=18
selected_content_streams=18
direct_flate_streams=33
redundant_filter_entries=18
omitted_stream_endobj=18
nonselected_flate_streams=15
oversized_nonselected_flate_streams=5
largest_direct_flate_length=446335
selected_compressed_bytes=198152
selected_decoded_bytes=1095453
inflate_blocks=18
inflate_stored_blocks=0
inflate_fixed_blocks=0
inflate_dynamic_blocks=18
inflate_literals=22723
inflate_copies=80561
max_distance=32505
aggregate_length_prefixed_sha256=22954f014433b91aeded376f74d023ffcaa35ff08f81fd1a0c02e61cdadac526
```

The ordered page objects are:

```text
1 10 18 26 29 45 97 103 106 110 126 130 133 136 139 174 183 227
```

The corresponding content objects are:

```text
4 12 20 28 31 47 99 105 109 112 128 132 135 138 141 176 186 229
```

Their compressed lengths are:

```text
362 4941 11821 11266 12700 13640 16473 15422 15398 8466 11784 11371 9558 12769 14241 10114 14111 3715
```

Their decoded lengths are:

```text
714 22427 65628 66554 70511 66988 98111 93168 93084 36043 64197 69278 56271 75827 72566 46413 82927 14746
```

For the aggregate digest, each decoded stream is prefixed by its eight-byte
big-endian decoded length and then absorbed in page-tree order. This binds both
boundaries and bytes without concatenation ambiguity.

## Exact Authority Stream

The program emits the inventory, all 18 per-content SHA-256 records, eight
negative results, and explicit boundary zeros. Two direct runs and the rebuilt
wrapper's program-output suffix are byte-identical. The exact authority stream
hashes to:

```text
41947f41db6ba3fb1f380a5f6008282144aa19f25384c31842bc7c9f46232232
```

## Non-Claims

This bundle freezes no claim that:

- the reader accepts arbitrary PDF files or newer xref/object-stream profiles;
- non-selected fonts, ICC profiles, CMaps, or metadata were inflated;
- decoded content bytes have been interpreted as text operators;
- a glyph code has a Unicode identity;
- an Apple feature row, family threshold, or processor mapping exists;
- any Apple execution engine was observed;
- a Metal feature is equivalent to x86, Arm, PTX, or SASS;
- any latency, throughput, material cost, lowering correctness, or speedup has
  been established.

## Stage Boundary

This bundle is produced by `SOUNIO_EXECUTABLE` and is proposed for
`SEMANTICS_FROZEN`. `PARITY_OPEN=false` and `CLAIM_READY=false`. Canonical
freeze waits on the owner of the Loom language-authority and docs-registry
transaction; no other language may fill that gap retrospectively.
