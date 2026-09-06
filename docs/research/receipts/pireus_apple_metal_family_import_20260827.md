<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-apple-metal-family-import-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-apple-metal-family-import-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus Apple Metal Family Import Receipt

**Date:** 2026-08-27
**Concept-ID:** `SOUNIO-PIREUS-APPLE-METAL-FAMILY-IMPORT`
**Language:** Sounio
**Role:** `SEMANTIC_AUTHORITY`
**Requested stage:** `SEMANTICS_FROZEN`

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

The Garden source pin is commit
`0a9623eed06f191b3aca3f26fcb3ae831dc08a22`. This receipt covers the next
two stages only. `PARITY_OPEN=false` and `CLAIM_READY=false` remain explicit.

## Authority Source

The Sounio authority surfaces are:

| File | SHA-256 |
| --- | --- |
| `stdlib/hardware/pireus/apple_metal_import.sio` | `b43f48c723283d65c3e1df1824f6284303a71967e20deab2c9fe8c7b72f97587` |
| `examples/pireus_apple_metal_family_import.sio` | `0d1ceab383e2be5f4c461275e70e44193a82d0e91c7c353132d7b7b93b255afd` |
| `docs/internal/concepts/pireus-apple-metal-family-import.md` | `c5b00d8dca3c7964a2ce319bb4585a9404861b2ab133259c028f46ce24c81a65` |
| `docs/research/pireus_apple_metal_family_import_semantics.md` | `4d82db39fb636d66f620d8cdb704ff25ef2dcb12507d3fc7105bbff42c0e8411` |

The executable example hash, not this prose receipt, is the source field bound
to Loom frame `9020`. The paired semantics document is the frozen-semantics
field.

## Vendor Corpus

```text
url=https://developer.apple.com/tutorials/data/documentation/metal/mtlgpufamily.json
content_type=application/json
bytes=39513
last_modified=Thu, 06 Aug 2026 03:17:23 GMT
etag="9a59-6585853a08c8f"
sha256=f0ed07338d44f0cce19f2ec1aebb2612638f5cab7b9a020fce8957ec21f809ea
```

The live URL is mutable. The payload is not stored in Git; its exact byte
identity is required by the Sounio executable. A future vendor snapshot must
start a new Garden pin instead of silently changing this result.

The separately pinned `supportsFamily(_:)` JSON, PDF, ZIP, and Numbers feature
tables were not ingested by this executable. They cannot extend this result.

## Toolchain

```text
engine=lean_single
wrapper_path=bin/souc
wrapper_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
compiler_path=bin/souc-lean-single-x86_64
compiler_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
rebuilt_checker_wrapper=/tmp/pireus-v01-ontology-validation-souc
rebuilt_checker_wrapper_sha256=ac705d2a14710b7034d08bd742c159acd1129b4f3760d69a522ee05fb7933395
execution_engine_ontology=stdlib/hardware/pireus/execution_engine.sio
execution_engine_ontology_sha256=8b5063f0e9a39650fb0b60e8b70b315f339723690e06050c2bebacece888e37e
ontology_query_kernel=stdlib/ontology/query.sio
ontology_query_kernel_sha256=e36f9d7bb4e16dd7c68a69dd51ae5f2db96d9bd8209bf61483c9b3ee88ac8cbb
sha256_kernel=stdlib/crypto/sha256.sio
sha256_kernel_sha256=6c5c6895f2d3b094ea114ee3ba894c535cb12e7822e7c902fbd52771aac7537a
```

The exact five-line base toolchain record bound to Loom hashes to
`2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e`.

## Execution Hardware

```text
os=Linux 7.0.2-5-pve
architecture=x86_64
cpu_model=Intel Xeon Gold 6526Y
sockets=2
cores_per_socket=16
threads_per_core=2
logical_cpus=64
```

The exact hardware record hashes to
`fd73771f3ac0ac200b6d93641b95744ea673c80a83f6d588adb8a19c5e1cf8f0`.

## Commands

```bash
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check \
  examples/pireus_apple_metal_family_import.sio

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  examples/pireus_apple_metal_family_import.sio \
  /tmp/pireus-apple-metal-20260521/mtlgpufamily.json

/tmp/pireus-v01-ontology-validation-souc check \
  examples/pireus_apple_metal_family_import.sio

/tmp/pireus-v01-ontology-validation-souc run \
  examples/pireus_apple_metal_family_import.sio -- \
  /tmp/pireus-apple-metal-20260521/mtlgpufamily.json
```

The rebuilt check returned `verdict=ok`, `provenance=rebuilt_direct`, and an
unanimous driver/fallback witness. Its run log includes compiler diagnostics;
the program-output suffix beginning at `SOUNIO_AUTHORITY` was byte-identical to
the `lean_single` authority stream. Two direct authority runs were also
byte-identical. No raw ELF was invoked.

The exact direct-run command record, with final `LF`, hashes to
`9f5680e133709cfc8030eb2996e414acd4c8d053319ac32c394d4d40f7db855f`.

## Sounio-Produced Result

```text
SOUNIO_AUTHORITY schema=pireus-apple-metal-family.v0 role=SEMANTIC_AUTHORITY
PIREUS_APPLE_CORPUS source=mtlgpufamily.json bytes=39513 error=0 sha256=f0ed07338d44f0cce19f2ec1aebb2612638f5cab7b9a020fce8957ec21f809ea digest_match=1
PIREUS_APPLE_ROOT identifier=MTLGPUFamily interface=swift valid=1
PIREUS_APPLE_JSON objects=381 arrays=182 strings=2193 max_depth=10
PIREUS_APPLE_CASES total=19 apple=10 metal=2 common=3 mac=2 mac_catalyst=2
PIREUS_APPLE_LIFECYCLE active=12 deprecated=7 topic_groups=5
PIREUS_APPLE_PLATFORMS total=6 beta_true=0 deprecated_true=0 unavailable_true=0 introduced_13_0=3 introduced_13_1=1 introduced_10_15=1 introduced_1_0=1
PIREUS_APPLE_ENGINE apple_gpu_blueprint_links=1 device_observations=0
PIREUS_APPLE_ONTOLOGY triples=447 cases=19 platforms=6 deprecated_cases=7
PIREUS_APPLE_NEGATIVE duplicate_key=1 selected_shape=1 platform_shape=1 malformed_json=1 duplicate_case=1 capacity=1 digest=1
PIREUS_APPLE_BOUNDARY device_observations=0 metal_permutation_features=0 instruction_equivalences=0 material_costs=0 lowering_claims=0
PIREUS_APPLE_SUMMARY failures=0
```

The exact stream, including bootstrap integer-printing newlines, hashes to:

```text
7a432891473b72b59d22ddcba407718877efe24dc6debf12016a8b51ed2534d1
```

## Negative Evidence

All seven deliberate parser negatives are Sounio functions and return `1`:

- duplicate object key;
- unknown `MTLGPUFamily` case suffix;
- unknown root platform identity;
- malformed or unbalanced JSON;
- duplicate selected case;
- fixed-capacity exhaustion;
- empty SHA-256 digest against the pinned corpus digest.

The first corpus run also exposed `init(rawValue:)` among direct family
references. It was classified explicitly as a non-case member; unknown suffixes
remain refused. A later run exposed that the root `references` object exceeds
32 keys, so the bounded key table was raised to 64 entries per object while its
capacity negative remained active. Both intermediate outputs were discarded.

## Evidence Boundary

This receipt establishes one pinned Apple API enumeration, 19 vendor enum-case
records, six raw platform records, and one vocabulary link to the declared
Apple GPU blueprint. It does not establish a `supportsFamily` observation,
observed Apple hardware, Metal feature-table capability, shader instruction,
cost, cross-ISA equivalence, lowering correctness, or Cayley-Dickson speedup.

Lean, Koka, C++, Haskell, and external LLM parity/review were not invoked.
External LLM offload reviews invoked: none; this internal ontology importer adds
no mathematical, clinical, or external-facing claim.

## Loom Admission

The frozen semantics were submitted to Loom frame `9020` with:

```text
stage=2 SOUNIO_EXECUTABLE
action=3 FREEZE_SEMANTICS
language=1 Sounio
role=1 SEMANTIC_AUTHORITY
policy_state=1 available
semantic_write=1
expected_result_write=1
parity_receipt_valid=0
review_promoted=0
exception_and_waiver_fields=0
guardian_fields=0
parent_semantics_sha256=absent
waiver_sha256=absent
```

The complete frame bound these receipt fields:

| Field | SHA-256 |
| --- | --- |
| Sounio executable source | `0d1ceab383e2be5f4c461275e70e44193a82d0e91c7c353132d7b7b93b255afd` |
| Frozen semantics | `4d82db39fb636d66f620d8cdb704ff25ef2dcb12507d3fc7105bbff42c0e8411` |
| Toolchain record | `2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e` |
| Hardware record | `fd73771f3ac0ac200b6d93641b95744ea673c80a83f6d588adb8a19c5e1cf8f0` |
| Command record | `9f5680e133709cfc8030eb2996e414acd4c8d053319ac32c394d4d40f7db855f` |
| Sounio result | `7a432891473b72b59d22ddcba407718877efe24dc6debf12016a8b51ed2534d1` |

The runtime selftest passed all 33 Sounio-owned cases. The operational runtime
remained the fixture-matched realization of the frozen Loom semantics:

```text
loom_semantics_sha256=16e283166d29d6b18ed690b000e2eb595a7d965e4357553a8380714486429fff
runtime_sha256=208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60
decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
decision_sha256=ad57cad1376867f5cec01323e50d39319c7fb8ba0458a6066086d186a045b8cb
```

A deliberate Python-oracle frame with the same receipt bindings was denied
before an interpreter or requested effect ran:

```text
exit_code=110
decision=SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SOUNIO_EXECUTABLE
decision_sha256=6fb4b46368e5dae161164f82e73ef0803084ae7a5d5cd8ec39588a1b9b44281d
```

This local admission accepts `SEMANTICS_FROZEN`; canonical Concept-ID
registration and evidence acceptance remain with the active Loom owner.
The pre-commit `action=10 COMMIT` frame at stage `SEMANTICS_FROZEN` was also
allowed with the same complete receipt bindings and canonical decision hash.
