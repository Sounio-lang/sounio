<!-- docs:meta
topic_id: repo.docs.internal.concepts.sounio-pdf-flate-content-reader
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.sounio-pdf-flate-content-reader
-->

# Sounio PDF Flate Content Reader

**Concept-ID:** `SOUNIO-PDF-FLATE-CONTENT-READER`
**Status:** executable candidate; canonical Loom acceptance required
**Owner:** Sounio document substrate
**Semantic authority:** Sounio

## Boundary

This concept identifies page-content streams in one pinned classic-xref PDF
profile and inflates them with the frozen Sounio RFC 1950/RFC 1951 primitive.

```text
FlateDecode stream != page content stream
page content bytes != decoded text
decoded text != Apple feature semantics
```

The reader does not infer pages from physical stream order or the first
`/Count` token. Selection is reachable-reference based:

```text
Trailer /Root -> Catalog /Pages -> Pages /Kids* -> Page /Contents -> stream
```

## Pinned Corpus

| Field | Frozen value |
| --- | --- |
| URL | `https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf` |
| Observed last modified | `Tue, 09 Jun 2026 00:00:35 GMT` |
| Document date | `May 21, 2026` |
| Bytes | 3,041,713 |
| SHA-256 | `9f31df15dd6827545702c5a0845f6e36e1889878cd0e534123bd70211e5c00a8` |
| Header | `%PDF-1.3` |

The complete binary file is read and hashed by Sounio before structural
results are admitted. The vendor payload is not redistributed in Git.

## Admitted Profile

The executable accepts only the observed profile:

- exact PDF 1.3 header and final `%%EOF` marker;
- one classic `xref` table named by the final `startxref`;
- no trailer `/Prev` and no incremental update chain;
- complete bounded xref coverage with at most 6,000 object identities;
- direct-generation indirect objects indexed by xref offsets;
- one catalog and one direct `/Pages` reference;
- bounded, iterative, cycle-detecting page-tree traversal;
- direct `/Type /Pages`, `/Kids`, `/Count`, `/Parent`, `/Type /Page`, and
  `/Contents` shapes;
- direct integer `/Length` and direct name `/Filter /FlateDecode` for selected
  streams;
- exact payload extent from `/Length`, followed by `endstream`;
- selected compressed inputs at most 65,536 bytes and decoded outputs at most
  262,144 bytes.

`/Contents` arrays, indirect lengths, filter arrays, object streams,
encryption, incremental updates, and non-Flate selected streams are distinct
refusals rather than fallback paths.

## Corpus Irregularities

Every selected content dictionary repeats `/Filter /FlateDecode` exactly
twice. The profile admits one or two filter entries only when both values are
the same direct `FlateDecode` name. A divergent second value is refused.

Every selected stream omits `endobj` after `endstream`. That omission is
admitted only when the next non-whitespace byte equals the xref offset of
another in-use object. The executable records both irregularities; neither is
silently normalized.

## Sounio Result

The executable derives and freezes:

- xref offset 2,924,864, 5,833 in-use objects, and declared size 5,834;
- catalog object 5,525 and page-root object 5,524;
- four page-tree nodes and 18 ordered pages;
- 18 selected page-content streams;
- 33 direct Flate streams overall, 15 non-selected, five of those beyond the
  content inflater's input capacity, and largest direct length 446,335;
- 198,152 selected compressed bytes and 1,095,453 decoded bytes;
- 18 dynamic DEFLATE blocks, 22,723 literals, 80,561 copies, and maximum
  observed distance 32,505;
- length-prefixed page-order aggregate content digest
  `22954f014433b91aeded376f74d023ffcaa35ff08f81fd1a0c02e61cdadac526`.

The exact ordered page/content object pairs, compressed and decoded lengths,
per-content SHA-256 digests, and aggregate digest are emitted by the Sounio
witness. The program checks the object pairs, lengths, aggregate digest, and
all totals before returning success.

## Fail-Closed Conditions

The reader refuses file length or digest drift, malformed envelope markers,
invalid or duplicate xref entries, incomplete xref coverage, dangling roots,
malformed dictionaries, duplicate structural keys, page cycles, parent
mismatch, page-count mismatch, unsupported content shape, conflicting filters,
invalid length shape, stream delimiter or xref-boundary failure, capacity
exhaustion, and every propagated inflater error.

Deliberate Sounio negatives exercise digest sabotage, header sabotage,
duplicate xref insertion, repeated page visitation, page-count mismatch,
conflicting duplicate filters, content capacity, and Adler-32 propagation.

## Evidence State

| Stage | State |
| --- | --- |
| `GARDEN` | Locally admitted by Loom; Garden SHA-256 `80bc230e24b4817305249514c71ea7ccbf1f37b6457a585523427106aa4dc42c`; canonical registry commit pending. |
| `SOUNIO_EXECUTABLE` | Implemented and independently checked by both compiler routes. |
| `SEMANTICS_FROZEN` | Proposed by the paired semantics and receipt; canonical Loom acceptance required. |
| `PARITY_OPEN` | False. |
| `CLAIM_READY` | False. |

This concept ends at decompressed content bytes. Text operators, font resource
resolution, ToUnicode CMaps, and Apple feature rows require later Garden-first
concepts.
