<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-sounio-pdf-flate-content-reader
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-sounio-pdf-flate-content-reader
-->

# Sounio: Follow The Page Tree, Not The First Count

> **Status**: Garden | **Date**: 2026-08-27 | **Authority**: founder direction

## Butterfly

The pure-Sounio RFC 1950/RFC 1951 inflater opens compressed bytes. Pireus now
needs the next reusable instrument: a bounded PDF structural reader that can
prove which compressed streams are page contents before decompression.

The pinned Apple feature tables expose why the distinction matters. A shallow
host probe can stop at one child page tree, while the catalog names a root page
tree containing several children. The same file carries large compressed font
programs alongside comparatively small page-content streams.

```text
FlateDecode stream != page content stream
page content bytes != decoded text
decoded text != Apple capability semantics
```

The reader must follow references. It must not infer page membership from
stream order, file offsets, host-tool labels, or the first `/Count` token.

## Pinned First Corpus

```text
url=https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
observed_last_modified=Tue, 09 Jun 2026 00:00:35 GMT
document_date=May 21, 2026
bytes=3041713
sha256=9f31df15dd6827545702c5a0845f6e36e1889878cd0e534123bd70211e5c00a8
declared_pdf_version=1.3
```

The live Apple URL is mutable. The executable must read the complete pinned
file and compute its SHA-256 in Sounio before admitting any structural result.
The payload remains outside Git because the vendor retains its rights.

No page count, object count, stream count, decompressed byte count, content
digest, or operator count is frozen by this seed. Those expected results must
be born in the Sounio executable.

## First Executable Grammar

The first executable must implement in Sounio:

1. O(1) byte access over the runtime's bounded word-buffer `read_file` path,
   avoiding multi-megabyte `string` indexing;
2. streaming SHA-256 over every valid file byte;
3. exact PDF header and terminal marker checks;
4. lexical recognition of indirect object headers with object number,
   generation number, and bounded body extent;
5. duplicate `(object, generation)` refusal;
6. dictionary-name, integer, array, and indirect-reference recognition without
   treating binary stream bytes as tokens;
7. catalog `/Pages` resolution and bounded, cycle-detecting page-tree traversal;
8. distinction between `/Type /Pages` and `/Type /Page`;
9. page-tree `/Kids` and `/Count` consistency checks;
10. exact resolution of each page's `/Contents` reference;
11. direct integer `/Length`, direct `/Filter /FlateDecode`, and exact
    `stream` line-ending validation for selected content objects;
12. byte-exact copying of selected stream payloads into the accepted Sounio
    inflater and propagation of its typed errors;
13. a deterministic digest and structural counters for each admitted decoded
    content stream;
14. negative witnesses created and evaluated in Sounio.

The initial slice may refuse `/Contents` arrays, indirect `/Length`, filter
arrays, object streams, encrypted documents, incremental updates, and selected
content streams exceeding fixed capacity. Any such shape must have a distinct
error; it may not be silently skipped.

## Selection Boundary

Only streams reached through this chain are selected:

```text
Catalog -> Pages root -> Kids* -> Page -> Contents -> stream
```

Compressed ICC profiles, font programs, ToUnicode maps, metadata, and any
unreferenced streams are recorded as non-selected structural observations but
are not inflated by this executable. In particular, an oversized font stream
does not authorize raising the inflater's page-content capacity or truncating
the font.

The output of this first slice is decompressed PDF content syntax, not Unicode
text. Operators such as `BT`, `Tf`, `Tj`, and `TJ`, resource dictionaries,
font encodings, and ToUnicode CMaps require a later Garden-first executable.

## Negative Surface

At minimum, Sounio must deliberately reject:

- wrong corpus digest or byte length;
- malformed indirect object header or duplicate object identity;
- missing or duplicate catalog;
- missing root page reference;
- dangling child or content reference;
- a page-tree cycle, excessive depth, or `/Count` mismatch;
- a `/Page` without the admitted direct `/Contents` shape;
- selected content with unsupported filter or length shape;
- missing stream delimiter, invalid stream line ending, or payload overrun;
- selected compressed input or decompressed output beyond capacity;
- propagated DEFLATE/zlib error;
- missing final PDF marker or trailing non-whitespace after it.

Shell inspection may locate candidate shapes for debugging but cannot create
expected values, decoded bytes, or acceptance. Python and Rust are prohibited;
Node, Ruby, shell, `awk`, `bc`, Poppler, Quartz, or another PDF implementation
cannot be cosmetically substituted as semantic oracle.

## Non-Claims

This Garden seed does not establish:

- the number of pages or page-content streams in the pinned PDF;
- the text printed on any page;
- a Metal feature name, family threshold, or processor mapping;
- a font glyph-to-Unicode mapping;
- an observed Apple execution engine;
- an x86, Arm, PTX, SASS, or Metal instruction equivalence;
- latency, throughput, material cost, lowering correctness, or speedup.

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

This seed establishes only `GARDEN` for
`SOUNIO-PDF-FLATE-CONTENT-READER`.
