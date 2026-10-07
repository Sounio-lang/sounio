<!-- docs:meta
topic_id: repo.docs.research.sounio-pdf-text-projection-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sounio-pdf-text-projection-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sounio PDF Text Projection Semantics

**Semantic bundle:** `sounio-pdf-text-projection.v0`
**Date:** 2026-08-27
**Producing language:** Sounio
**Role:** `SEMANTIC_AUTHORITY`

## Authority Inputs

The normative PDF is Apple's May 21, 2026 Metal feature-table document:

```text
url=https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
bytes=3041713
sha256=9f31df15dd6827545702c5a0845f6e36e1889878cd0e534123bd70211e5c00a8
```

Its selected page-content substrate is the frozen Sounio result with 18
ordered streams, 1,095,453 decoded bytes, and aggregate SHA-256
`22954f014433b91aeded376f74d023ffcaa35ff08f81fd1a0c02e61cdadac526`.

Bytes `0x00..0x7f` follow ASCII. The full `0x80..0xff` table embedded in the
Sounio module comes from Unicode's official Apple Roman mapping at
`https://www.unicode.org/Public/MAPPINGS/VENDORS/APPLE/ROMAN.TXT`, whose pinned
payload hashes to
`18e571645be895e9553ed5c842ea8f65f9c5d3c9ccb43e66e0c33a132ed0d721`.

## Resource And Font Semantics

Each page's effective `/Resources` is resolved through its page/parent chain
with bounded cycle detection. A direct resource dictionary or exactly one
indirect reference is admitted; ambiguous, missing, dangling, wrongly typed,
or cyclic resource shapes are refused.

The pinned result contains 18 indirect resource dictionaries, 99 page-local
font bindings, and 10 unique simple fonts. Seven fonts use MacRoman directly.
Three have direct encoding differences and ToUnicode streams. There are no
descendant fonts. ToUnicode streams are inflated by the Sounio RFC 1950/RFC
1951 implementation and parsed as one-byte CMaps. Duplicate/overlapping source
ranges, codes outside admitted code spaces, invalid Unicode values, malformed
ranges, and output overflow are errors.

The three CMaps contain three code spaces and four `bfrange` mappings. Across
the complete text program, 35,904 scalars follow MacRoman and 74 follow those
CMaps.

## Content Program Semantics

The lexer consumes the exact decoded content extent. It recognizes whitespace,
comments, names, signed integers, fixed-point real tokens, balanced literal
strings, hexadecimal strings, arrays, dictionaries, and operators. Literal
strings implement escaped delimiters, control escapes, up to three octal
digits, and CR/LF continuation.

The corpus census is:

```text
decoded_bytes=1095453
names=16451
numbers=125383
literal_strings=25062
hexadecimal_strings=0
array_pairs=2397
dictionary_pairs=5319
operator_tokens=77986
distinct_operators=27
BT=3193 Tm=3190 Tf=3236 TJ=2397 Tj=839 Tc=1148 ET=3193
```

Text show is valid only inside `BT`/`ET`, after an admitted font selection, and
with the exact operand arity/type. `Tj` decodes one literal. `TJ` decodes every
literal in its array and validates numeric adjustments. The executable records
25,032 shown literal strings, 2,397 shown arrays, 21,822 array adjustments, and
35,978 glyph codes.

## Geometry Semantics

`q` and `Q` maintain a bounded graphics stack. `cm` composes the current
transformation matrix. `Tm` supplies a text object's origin. All values are
stored at scale 1,000,000 and checked before multiplication/addition. The
projected device origin is deterministic integer arithmetic; non-finite host
floating point cannot enter the model.

There are 3,193 text objects. Three are empty and have no `Tm`; no object has
multiple `Tm` operations. The graphics stack reaches depth two.

Nonempty objects are sorted by page ascending, device y descending, device x
ascending, then source object identity. For each page, a maximal anchor band is
formed from the greatest remaining y while `anchor_y - y <= 5,000,000`. This
produces 730 lines. Maximum within-line spread is 5,000,000 and minimum
same-page anchor gap is 6,000,000.

## Digest Semantics

The canonical serialization starts with the 30-byte tagged prefix, followed
by 16 projection/line summary integers, 11 integers per source-order object,
two integers per Unicode scalar, and six integers per line. Every integer is a
signed i64 serialized big-endian. Empty objects carry line identity `-1`.

```text
serialized_bytes=891830
sha256=aebdea5034dc20201edf555bbb257e2971eb86127be602e630be1135564f93f8
```

The SHA-256 state, serialization, expected digest, and comparison are all
Sounio executable semantics. The host's hash of the printed authority stream
is receipt evidence only.

## Frozen Result

```text
pages=18
font_bindings=99
unique_fonts=10
cmap_streams=3
text_objects=3193
nonempty_objects=3190
empty_objects=3
show_operations=3236
unicode_scalars=35978
macroman_scalars=35904
cmap_scalars=74
non_ascii_scalars=152
private_use_scalars=2
lines=730
line_tolerance=5000000
max_line_spread=5000000
min_interline_gap=6000000
ambiguous_assignments=0
projection_sha256=aebdea5034dc20201edf555bbb257e2971eb86127be602e630be1135564f93f8
```

Two direct authority runs are byte-identical. The rebuilt/current-source
checker is unanimous, and its run's program-output suffix is identical to the
direct stream. Seven Sounio negative/sabotage cases report zero failures.

## Non-Claims

This bundle freezes no claim that the projected lines are table rows or cells;
that labels such as `Apple10` imply a capability relation; that feature values
inherit across columns; that a CPU, GPU, or execution engine supports any
feature; or that x86, Arm, Metal, PTX, SASS, or another instruction has an
equivalent lowering. It makes no cost or performance claim and admits no PDF
outside the stated profile.

## Stage Boundary

This is a Sounio-produced proposal for `SEMANTICS_FROZEN`.
`PARITY_OPEN=false` and `CLAIM_READY=false`. Lean, Koka, C++, Haskell, and
external LLMs cannot create or amend the expected result.
