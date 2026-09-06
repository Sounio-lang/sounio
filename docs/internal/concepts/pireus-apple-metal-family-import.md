<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-apple-metal-family-import
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pireus-apple-metal-family-import
-->

# Pireus Apple Metal Family Import

**Concept-ID:** `SOUNIO-PIREUS-APPLE-METAL-FAMILY-IMPORT`
**Status:** executable candidate; Loom acceptance required
**Owner:** Pireus hardware ontology
**Semantic authority:** Sounio

## Boundary

This concept is the first grammar-bounded Apple corpus projection in Pireus.
It imports the pinned `MTLGPUFamily` DocC JSON record and connects that API
vocabulary to the already-declared Apple GPU execution-engine blueprint.

```text
MTLGPUFamily case != supportsFamily observation != Metal shader instruction
Apple GPU blueprint != observed Apple GPU engine
```

The imported enum cases are vendor API catalog records. They do not establish
that this Xeon host, an Apple device, or any other machine supports a family.

## Source Contract

| Field | Frozen value |
| --- | --- |
| URL | `https://developer.apple.com/tutorials/data/documentation/metal/mtlgpufamily.json` |
| Observed last modified | `Thu, 06 Aug 2026 03:17:23 GMT` |
| Bytes | 39,513 |
| SHA-256 | `f0ed07338d44f0cce19f2ec1aebb2612638f5cab7b9a020fce8957ec21f809ea` |

The importer in `stdlib/hardware/pireus/apple_metal_import.sio` reads and
hashes the complete stream in Sounio before accepting any projection.

## Frozen Projection

The executable admits exactly:

- the root `MTLGPUFamily` symbol with Swift interface identity;
- 19 enum-case reference objects whose titles agree with their identifiers;
- 10 Apple, two Metal, three Common, two Mac, and two Mac Catalyst cases;
- 12 active and seven deprecated enum cases;
- five selected topic groups;
- six root platform records with raw `introducedAt`, `beta`, `deprecated`, and
  `unavailable` fields;
- one vocabulary link from the API enumeration to the declared Apple GPU
  execution-engine blueprint;
- zero observed Apple GPU engines and zero device support observations.

The initializer reference `init(rawValue:)` is an explicitly admitted
non-case member of the same DocC `references` object. Any other unknown
`MTLGPUFamily` suffix remains a fail-closed schema change.

## Fail-Closed Conditions

The executable refuses:

- missing input, wrong length, or SHA-256 drift;
- malformed JSON, excessive nesting, duplicate object keys, or capacity
  overflow;
- duplicate enum cases or platform records;
- unknown case or platform identities;
- missing or mismatched root identity, case title, topic partition, platform
  field, boolean field, or introduction version;
- an empty or otherwise non-matching digest.

All deliberate negative witnesses execute in Sounio. Python and Rust are not
used, and disposable languages do not produce expected results.

## Evidence State

| Stage | State |
| --- | --- |
| `GARDEN` | Established by commit `0a9623eed06f191b3aca3f26fcb3ae831dc08a22`. |
| `SOUNIO_EXECUTABLE` | Implemented by the importer and example witness. |
| `SEMANTICS_FROZEN` | Proposed by the paired semantics and receipt; requires Loom acceptance. |
| `PARITY_OPEN` | False. |
| `CLAIM_READY` | False. |

Registration must not promote this concept beyond the accepted evidence stage.
