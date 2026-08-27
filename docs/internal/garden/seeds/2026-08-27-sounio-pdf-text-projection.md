<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-sounio-pdf-text-projection
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-sounio-pdf-text-projection
-->

# Sounio: A Glyph Is Not Yet A Character

> **Status**: Garden | **Date**: 2026-08-27 | **Authority**: founder direction

## Butterfly

The frozen Sounio PDF/Flate reader can identify every selected page-content
stream in the pinned Apple corpus and recover its exact decompressed bytes.
Those bytes still describe a graphics program. Pireus needs a second reusable
instrument that follows the document's own text and font references before a
glyph code can become Unicode text.

```text
content bytes -> text operators -> font resource -> character map -> text
text -> table geometry -> Apple capability row
```

Each arrow is a semantic boundary. A byte sequence between `BT` and `ET` is not
Unicode merely because it is printable, and the order of string operands in a
content stream is not necessarily the visual reading order of a table.

## Frozen Prerequisite

The first corpus remains the pinned Apple Metal feature tables:

```text
url=https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
observed_last_modified=Tue, 09 Jun 2026 00:00:35 GMT
document_date=May 21, 2026
bytes=3041713
sha256=9f31df15dd6827545702c5a0845f6e36e1889878cd0e534123bd70211e5c00a8
```

The byte and page-content substrate is the Sounio source commit
`00c2fbe79c6376c56b0fd7fea5fc154f6440424c`. Its frozen ordered content result
is bound to aggregate SHA-256
`22954f014433b91aeded376f74d023ffcaa35ff08f81fd1a0c02e61cdadac526`.

This seed freezes no font count, operator count, glyph count, Unicode scalar,
line, cell, table row, or Apple feature value. Those results must first be
created by the Sounio executable.

## First Executable Grammar

The first executable must implement in Sounio, for the bounded corpus profile:

1. the frozen PDF envelope, xref, page-tree, content-selection, and Flate path;
2. resolution of each page's effective `/Resources` dictionary, including
   admitted inherited resources and explicit refusal of ambiguous inheritance;
3. resolution of each used `/Font` resource name to one in-use indirect font
   object through the parsed xref;
4. bounded recognition of the font dictionary, subtype, encoding references,
   descendant-font references, and `/ToUnicode` reference actually present in
   the corpus;
5. byte-exact inflation of selected character-map streams through the frozen
   Sounio inflater, with a separate capacity and typed error namespace;
6. a bounded CMap interpreter for the admitted `codespacerange`, `bfchar`, and
   `bfrange` forms, including one-to-many Unicode mappings when present;
7. a content-stream lexer for whitespace, comments, names, integers, reals,
   literal strings with escapes and balanced parentheses, hexadecimal strings,
   arrays, and operators without treating operands as operators;
8. explicit `BT`/`ET` text-object state and the text-state operators needed by
   the corpus, including font selection, text matrices, line movement, spacing,
   scaling, rise, rendering mode, and text-showing forms;
9. decoding of string operands only through the selected font's admitted code
   width and character map, never through host locale or byte-printability;
10. deterministic glyph events carrying page order, content order, font
    resource identity, source code bytes, Unicode scalar sequence, text matrix,
    and displacement evidence needed for later layout projection;
11. deterministic line/run projection whose ordering rule is stated by the
    executable and whose complete result is digest-bound;
12. negative witnesses created and evaluated in Sounio through the same lexer,
    resource resolver, CMap interpreter, and text-state transitions.

The initial slice may refuse inline images, Type 3 glyph programs, unsupported
font encodings, malformed surrogate pairs, vertical writing, form XObjects,
marked-content-dependent replacement text, or operators outside the admitted
corpus profile. Every refusal must be typed and fail closed. It may not silently
drop a string, substitute a replacement character, or fall back to a host PDF
implementation.

## Projection Boundary

The authoritative output of this layer is an ordered, provenance-carrying text
projection. It may report geometric runs and candidate line groupings, but it
must not yet name a table column, normalize an Apple family label, interpret a
checkmark, infer feature inheritance, or create a Pireus capability fact.

That later projection must begin from this layer's frozen Sounio result and a
new Garden seed. It cannot retrospectively change glyph decoding or reading
order to obtain a preferred Apple row.

## Negative Surface

At minimum, Sounio must deliberately reject:

- a content aggregate that differs from the frozen prerequisite;
- a missing, dangling, duplicate, or wrongly typed font resource;
- ambiguous or cyclic inherited resources;
- a selected font without an admitted mapping path;
- duplicate or overlapping CMap source ranges;
- a glyph code outside every admitted code-space range;
- an invalid Unicode target, malformed range expansion, or output overflow;
- an unterminated string, hexadecimal string, array, text object, or comment;
- a text-showing operation outside `BT`/`ET` or before font selection;
- an operand stack with the wrong arity or type for the next operator;
- non-finite numeric state, matrix overflow, or excessive nesting;
- unsupported text-bearing form or inline object that would make omission
  semantically visible;
- a deterministic-ordering sabotage that changes the final digest.

Shell inspection may locate candidate byte shapes for debugging but cannot
create expected operators, glyph mappings, text, ordering, or acceptance.
Python and Rust are prohibited; Node, Ruby, shell, `awk`, `bc`, Poppler,
Quartz, PDFKit, or another PDF implementation cannot be substituted as an
oracle.

## Non-Claims

This Garden seed does not establish:

- the text printed on any page of the pinned PDF;
- a font name, CMap shape, glyph count, text run, line, or table cell;
- an Apple feature name, family threshold, availability mark, or processor map;
- equivalence between Apple GPU families and an x86, Arm, PTX, SASS, or Metal
  instruction;
- observed hardware support, material cost, lowering correctness, or speedup;
- complete PDF conformance outside the explicitly admitted corpus profile.

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

This seed establishes only `GARDEN` for `SOUNIO-PDF-TEXT-PROJECTION`.
