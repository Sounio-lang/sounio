<!-- docs:meta
topic_id: repo.docs.internal.concepts.sounio-pdf-text-projection
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.sounio-pdf-text-projection
-->

# Sounio PDF Text Projection

**Concept-ID:** `SOUNIO-PDF-TEXT-PROJECTION`
**Status:** executable candidate; canonical Loom acceptance required
**Owner:** Sounio document substrate
**Semantic authority:** Sounio

## Boundary

This concept turns the frozen page-content byte streams of one pinned PDF into
Unicode-bearing text objects and deterministic geometric line projections.

```text
content bytes -> PDF operators -> font resource -> character map -> scalars
scalars + graphics/text matrices -> text objects -> geometric lines
geometric lines != table cells != Apple capability facts
```

It neither calls a host PDF stack nor treats printable source bytes as text.

## Pinned Inputs

| Input | Frozen value |
| --- | --- |
| Apple Metal PDF | 3,041,713 bytes; SHA-256 `9f31df15dd6827545702c5a0845f6e36e1889878cd0e534123bd70211e5c00a8` |
| Ordered decoded page content | 18 streams; aggregate SHA-256 `22954f014433b91aeded376f74d023ffcaa35ff08f81fd1a0c02e61cdadac526` |
| Apple Roman mapping | Unicode vendor mapping; SHA-256 `18e571645be895e9553ed5c842ea8f65f9c5d3c9ccb43e66e0c33a132ed0d721` |

The vendor PDF and Unicode mapping are not redistributed in Git.

## Admitted Profile

The executable admits the corpus shape it proves:

- 18 indirect page resource dictionaries and no inherited resource edge;
- simple Type1 and TrueType fonts, with no Type0 descendant font;
- MacRoman base encoding and the direct encoding-difference shapes present;
- one-byte ToUnicode CMaps using admitted code spaces and `bfrange` entries;
- content comments, names, integers, fixed-point reals, literal strings,
  hexadecimal strings, arrays, dictionaries, and bounded operators;
- balanced literal parentheses, PDF escapes, octal escapes, and line
  continuation;
- `q`, `Q`, and `cm` graphics state plus `BT`, `ET`, `Tf`, `Tm`, `Tc`, `Tj`,
  and `TJ` text state used by the corpus;
- one-million-unit fixed-point coordinates with checked multiplication;
- at most 4,096 text objects and lines and 65,536 Unicode scalars.

Unsupported or malformed text-bearing shapes fail with the typed `201..214`
error namespace. There is no replacement-character or host-locale fallback.

## Line Semantics

Nonempty text objects are ordered by:

```text
(page ascending, device_y descending, device_x ascending, source object ascending)
```

Within a page, a line begins at the greatest remaining `device_y`. Following
objects join that line while:

```text
anchor_y - object_y <= 5,000,000
```

The threshold is exactly five PDF points in the fixed-point coordinate system.
The next line must have an anchor gap greater than the threshold. The complete
corpus produces 730 lines, maximum line spread 5 points, minimum same-page
anchor gap 6 points, and no ambiguity.

## Canonical Digest

Sounio serializes the complete bounded projection, not its printed rendering:

1. the ASCII tag `SOUNIO_PDF_TEXT_PROJECTION_V1` and NUL;
2. 16 signed 64-bit big-endian summary fields;
3. 11 signed 64-bit fields for each source-order text object;
4. Unicode scalar and font identity as two signed 64-bit fields per scalar;
5. six signed 64-bit fields for each geometric line.

The 891,830-byte serialization hashes in Sounio to:

```text
aebdea5034dc20201edf555bbb257e2971eb86127be602e630be1135564f93f8
```

## Sounio Result

The executable freezes:

- 18 pages, 99 font bindings, 10 unique fonts, and three ToUnicode CMaps;
- 1,095,453 decoded content bytes and 27 distinct operators;
- 3,193 text objects, of which 3,190 are nonempty;
- 3,236 text-show operations and 35,978 Unicode scalars;
- 35,904 MacRoman scalars and 74 CMap-derived scalars;
- 152 non-ASCII scalars and two private-use Apple-logo scalars;
- 730 deterministic geometric lines and the canonical digest above.

The output includes the complete 730-line projection. A line such as
`M5-series|Metal 3 & 4|Apple10` is text and geometry only; this layer does not
declare a table schema or interpret any field.

## Negative Surface

The Sounio negative witness verifies error propagation, page boundaries,
empty-object omission, the inclusive 5-point line threshold, 6-point line
separation, repeated-run determinism, scalar sabotage, and inconsistent line
assignment refusal. Parser, resource, CMap, and text-state error paths remain
fail-closed in the implementation.

## Evidence State

| Stage | State |
| --- | --- |
| `GARDEN` | Canonical commit `a897f96ee446d47496d562e930ea076be0076d7c`; governance-normalized SHA-256 `6aa909e3de1849ee7b08b43353c94927712d2f71be9a0fdcd1bb66019739c385`; source-lane SHA-256 `7177d16b71a391811731df3a8cef2b7cd3656b31830e624194267c3e6c90077e`. |
| `SOUNIO_EXECUTABLE` | Canonical commit `781109c2db1cf7276346aaa51716bfa658f3556e`; complete for the admitted corpus profile; direct and rebuilt checks agree. |
| `SEMANTICS_FROZEN` | Admitted by the paired canonical Sounio semantics, receipt, and Loom acceptance evidence. |
| `PARITY_OPEN` | False. |
| `CLAIM_READY` | False. |

## Non-Claims

This concept does not establish general PDF conformance, OCR, visual table
cells, feature inheritance, Apple-family meaning, processor identity,
instruction equivalence, observed hardware, lowering correctness, cost, or
performance.
