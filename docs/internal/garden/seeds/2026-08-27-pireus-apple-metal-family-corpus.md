<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-pireus-apple-metal-family-corpus
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-pireus-apple-metal-family-corpus
-->

# Pireus: The Apple Metal Family Chart

> **Status**: Garden | **Date**: 2026-08-27 | **Authority**: founder direction

## Butterfly

Apple Silicon is a canonical Pireus target, but it is not a third CPU ISA.
Its CPU instruction semantics come from the Arm corpus. Apple contributes a
platform profile, material observations, and a distinct Apple GPU engine whose
capabilities are exposed through Metal.

```text
Arm CPU ISA record != Apple CPU observation
Metal family enum != GPU support observation != shader instruction
```

The first Apple executable will chart the official `MTLGPUFamily` enumeration.
It will not pretend that the Xeon host running Sounio observed an Apple GPU.

## Pinned Structured Corpus

Apple's DocC service publishes the current Metal API record as structured JSON:

```text
url=https://developer.apple.com/tutorials/data/documentation/metal/mtlgpufamily.json
observed_http_status=200
observed_content_type=application/json
observed_content_length=39513
observed_last_modified=Thu, 06 Aug 2026 03:17:23 GMT
observed_etag="9a59-6585853a08c8f"
sha256=f0ed07338d44f0cce19f2ec1aebb2612638f5cab7b9a020fce8957ec21f809ea
```

The API that turns an enum case into a device observation is pinned separately:

```text
url=https://developer.apple.com/tutorials/data/documentation/metal/mtldevice/supportsfamily(_:).json
observed_http_status=200
observed_content_type=application/json
observed_content_length=15314
observed_last_modified=Thu, 06 Aug 2026 03:17:20 GMT
observed_etag="3bd2-658585375cc04"
sha256=d810294a8b2112847e17bd1036e0000d0e09827a91f6a9f2b79d9ed192c79601
```

These live URLs are mutable. The first importer must verify the exact observed
bytes and fail on drift; a later release should reacquire and Garden-pin a new
snapshot instead of silently accepting it.

## Pinned Feature Tables

Apple's Metal resources page describes the feature-set tables as the map of
current Apple GPU families, processors, and older feature sets. The 2026-05-21
rendering is pinned in two vendor forms:

| Surface | Bytes | SHA-256 |
| --- | ---: | --- |
| `Metal-Feature-Set-Tables.pdf` | 3,041,713 | `9f31df15dd6827545702c5a0845f6e36e1889878cd0e534123bd70211e5c00a8` |
| `Metal-Feature-Set-Tables.zip` | 240,046 | `662e4b5df59b61f9a311a00738df61e3c99bf3fab46740b75581aeaf9f1d14ff` |
| inner `Metal-Feature-Set-Tables.numbers` | 401,089 | `27e557837b61c2e0ad4ebe39bc39216edd44f1f5077b9ec4d474e37a72f28a8e` |

```text
pdf_url=https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
pdf_last_modified=Tue, 09 Jun 2026 00:00:35 GMT
zip_url=https://developer.apple.com/metal/Metal-Feature-Set-Tables.zip
zip_last_modified=Tue, 09 Jun 2026 13:46:19 GMT
document_date=May 21, 2026
pages=18
```

The Numbers document is itself a ZIP package containing Apple IWA streams. No
stable public IWA schema is assumed. Neither a shell unzip nor PDF text
extraction may create the semantic projection or expected result.

The table visibly contains candidate permutation rows including quad-scoped
and SIMD-scoped permute operations. Their family thresholds are intentionally
not frozen by this Garden seed. A later Sounio parser must establish those
records from an accepted grammar before Pireus can relate them to a family.

## License And Access Boundary

The DocC JSON carries Apple copyright and links to Apple's site terms. The
feature tables state that all rights are reserved. Pireus records coordinates,
transport metadata, hashes, and Sounio-derived projections; it does not
redistribute the Apple payloads.

Apple's CPU Optimization Guide Version 4 is a separate, agreement-gated source.
It may later supply licensed material-profile evidence, but its access control
must not be bypassed and it cannot replace the open Arm normative ISA corpus.

## First Harbor Slice

The first Sounio executable will consume only the pinned `mtlgpufamily.json`
stream and project:

- the root symbol identity and interface language;
- exact enumeration-case references grouped by Apple, Metal, Common, Mac, and
  Mac Catalyst topic sections;
- deprecation and platform metadata retained as raw vendor fields;
- the distinction between enumeration availability and a runtime
  `supportsFamily` observation;
- an Apple GPU family vocabulary connected to the existing Pireus
  `ExecutionEngine` model without creating a machine observation.

No case count, family partition, deprecation count, or platform result is
frozen here. The Sounio executable must produce those expected results.

## Required Sounio Contract

The importer must:

1. read the complete pinned JSON byte stream;
2. verify byte length and SHA-256 in Sounio;
3. parse JSON structure and reject duplicate object fields;
4. distinguish root metadata, topic sections, references, and symbol records;
5. retain exact vendor identifiers before assigning Pireus roles;
6. reject malformed JSON, schema drift, duplicate selected records, unknown
   selected shapes, missing root identity, empty digest, and capacity overflow;
7. emit the first inventory and expected result in Sounio;
8. emit zero device observations, instruction equivalences, costs, and
   lowerings on this non-Apple host;
9. keep the feature-table permutation rows closed for a later Garden-first
   parser.

Python and Rust are prohibited. Node, Ruby, shell text processing, `awk`, `bc`,
PDF extraction, or another disposable tool may transport or inspect bytes but
may not create the semantic projection or expected result.

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

This seed establishes only `GARDEN`.
