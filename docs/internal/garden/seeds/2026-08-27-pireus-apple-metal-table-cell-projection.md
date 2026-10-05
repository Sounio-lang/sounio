<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-pireus-apple-metal-table-cell-projection
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-pireus-apple-metal-table-cell-projection
-->

# Pireus: A Line Is Not Yet A Table Cell

> **Status**: Garden | **Date**: 2026-08-27 | **Authority**: founder direction

## Butterfly

Sounio can now recover the pinned Apple Metal PDF as Unicode-bearing text
objects and deterministic geometric lines. A printed line still does not say
where one table cell ends, whether a fragment continues on the next line, or
whether an empty rectangle means unavailable, inherited, or merely blank.

```text
text object -> geometric line -> table cell -> feature record -> capability fact
```

This Garden opens only the `geometric line -> table cell` boundary.

## Frozen Prerequisites

The normative vendor bytes remain Apple's May 21, 2026 feature tables:

```text
url=https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
bytes=3041713
sha256=9f31df15dd6827545702c5a0845f6e36e1889878cd0e534123bd70211e5c00a8
```

The immediate semantic parent is the Sounio PDF text projection:

```text
concept=SOUNIO-PDF-TEXT-PROJECTION
source_commit=601d150f8858a41af1852074b07d968fa4c07ef5
serialized_bytes=891830
projection_sha256=aebdea5034dc20201edf555bbb257e2971eb86127be602e630be1135564f93f8
```

The parent is locally frozen and has been handed to the canonical Loom owner.
No table executable may begin until that owner records the parent at
`SEMANTICS_FROZEN` under its canonical commit and hash bindings.

The already frozen `SOUNIO-PIREUS-APPLE-METAL-FAMILY-IMPORT` supplies the
vendor `MTLGPUFamily` vocabulary. It is not an authority for the PDF's table
geometry or feature values and cannot fill missing cells retrospectively.

## First Executable Contract

The first Sounio executable must consume the in-memory frozen projection, not
the human-readable report, and must:

1. recompute and require the exact parent projection digest before cell work;
2. preserve page, source object, font, Unicode scalar, raw coordinates, device
   coordinates, line identity, and object boundaries as provenance;
3. recognize only the bounded table geometry actually established from the
   content program, including admitted path/rectangle evidence where needed;
4. derive candidate vertical boundaries and row bands from repeated geometric
   evidence rather than a hand-entered Apple column list;
5. distinguish one-line cells, wrapped cells, merged headers, footnotes,
   captions, and non-table page text through explicit typed states;
6. assign a text object to at most one cell and reject boundary overlap or
   equally valid competing assignments;
7. preserve empty cells as empty geometric cells without assigning a meaning;
8. retain exact scalar sequences and object order inside every cell before any
   whitespace normalization;
9. emit a deterministic page/table/row/column/cell projection and a complete
   Sounio-computed digest;
10. create the first expected table/cell inventory and negative results in
    Sounio.

The executable may begin with a strict subset of pages if that subset is
declared before execution and every excluded page is explicit. It may not scan
until one desired permutation row appears and then present that as the table
grammar.

## Geometry Before Vocabulary

Header strings such as `Apple7`, `Apple10`, `Metal 3 & 4`, or a processor name
remain raw cell text in this layer. Exact string equality may be recorded, but
linking a header to an `MTLGPUFamily` case belongs to a later semantic
projection. A merged header spanning columns is one cell with a span, not a
license to copy its text into every covered column.

Likewise, these shapes remain distinct:

```text
empty cell != Not available != varies != inherited support
checkmark glyph != observed device support
family header != processor observation
feature row label != shader or machine instruction
```

## Required Negative Surface

At minimum, Sounio must deliberately reject:

- vendor bytes or parent projection digest drift;
- a text object whose page, line, scalar extent, or coordinates disagree with
  its frozen parent record;
- missing, crossed, duplicated, non-monotone, or overlapping table boundaries;
- a cell assignment compatible with two columns or two row bands;
- an object split across cells without an admitted geometric boundary;
- a wrapped fragment that can attach to multiple preceding rows;
- a merged header with an unsupported or ambiguous span;
- a footnote marker promoted into cell text without a typed footnote edge;
- an empty cell silently converted to an availability value;
- a source-order or coordinate sabotage that leaves the final digest unchanged;
- capacity exhaustion or any unsupported text-bearing geometry.

Python and Rust are prohibited. Node, Ruby, shell, `awk`, `bc`, Poppler,
Quartz, PDFKit, Numbers, or another disposable parser may transport or inspect
bytes but cannot create table boundaries, cells, expected results, or semantic
links.

## First Result Boundary

The authoritative result of this concept is a raw, provenance-carrying table
cell graph:

```text
Table -> RowBand -> Cell
Cell -> SourceTextObject
Cell -> ExactScalarSequence
Cell -> GeometricBounds
Cell -> OptionalSpan
```

It emits zero Apple capability facts, zero family-support relations, zero
processor observations, zero instruction equivalences, zero lowerings, and
zero cost/performance claims.

## Later Garden

Only after this cell graph is canonically frozen may a new Garden propose:

```text
raw header cell -> MTLGPUFamily identity
raw feature cell -> typed availability value
typed row + column -> Pireus capability record
```

That later layer must state its rules for merged headers, footnotes, `Varies`,
`Not available`, blanks, and any inheritance. It cannot alter cell geometry to
obtain a preferred capability matrix.

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

This seed establishes only `GARDEN` for
`SOUNIO-PIREUS-APPLE-METAL-TABLE-CELL-PROJECTION`.
