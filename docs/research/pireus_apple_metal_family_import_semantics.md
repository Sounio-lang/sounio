<!-- docs:meta
topic_id: repo.docs.research.pireus-apple-metal-family-import-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-apple-metal-family-import-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus Apple Metal Family Import Semantics

**Semantic bundle:** `pireus-apple-metal-family.v0`
**Date:** 2026-08-27
**Producing language:** Sounio
**Role:** `SEMANTIC_AUTHORITY`

## Authority Input

The sole normative byte input is the pinned Apple DocC JSON record:

```text
https://developer.apple.com/tutorials/data/documentation/metal/mtlgpufamily.json
```

The accepted stream is 39,513 bytes with SHA-256
`f0ed07338d44f0cce19f2ec1aebb2612638f5cab7b9a020fce8957ec21f809ea`.
Sounio reads every byte, computes SHA-256, validates the complete JSON grammar,
and admits the selected projection only when length, digest, and shape agree.

The separately pinned `supportsFamily(_:)` JSON and Metal feature tables are
provenance and future-lane inputs only. They do not vote on or extend this
semantic bundle.

## Structural Grammar

The importer is an incremental JSON parser with explicit states for objects,
arrays, strings, escapes, Unicode escapes, and primitives. It:

1. validates the complete document and balanced nesting;
2. rejects duplicate keys inside every object;
3. selects only direct children of the root `references` object whose keys use
   the exact `MTLGPUFamily/` namespace;
4. recognizes 19 enum cases and the one allowed non-case initializer;
5. requires each selected case title to match the key-derived case identity;
6. selects the five root topic groups and six root `metadata.platforms`
   records by structural scope;
7. retains raw platform names, introduction versions, and three boolean flags;
8. rejects unsupported selected shapes instead of dropping or guessing fields.

Shell inspection does not create records, counts, or expected results.

## Sounio-Produced Projection

The logical authority records are:

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

The actual stream contains bootstrap integer-printing newlines within records.
Its exact byte representation hashes to
`7a432891473b72b59d22ddcba407718877efe24dc6debf12016a8b51ed2534d1`.

## Ontology Projection

The projection layers on `pireus_execution_engine_store()` and adds classes
for GPU-family case, API enumeration, vendor corpus, family group, lifecycle,
raw platform record, raw boolean, and raw version. Each enum case links to its
corpus, enumeration, family group, and lifecycle. Each platform record links
to its raw introduction version and boolean fields.

The enumeration has one typed relation to `PIREUS_BLUEPRINT_APPLE_GPU`. That
blueprint was already declared as a GPU engine blueprint with Metal interface.
It is not an observed execution engine and the bundle creates no machine or
device observation. The layered ontology contains 447 triples; SPARQL
witnesses return 19 cases, six platforms, seven deprecated cases, and one
blueprint link.

## Non-Claims

This bundle freezes no assertion that:

- a particular Apple device supports any imported family;
- the current Xeon host observed an Apple CPU or GPU;
- an enum case is a CPU ISA, GPU ISA, or shader instruction;
- a Metal feature-table row is available on any family;
- a family case is equivalent to an x86, Arm, PTX, or SASS instruction;
- any imported record carries latency, throughput, cost, lowering correctness,
  or Cayley-Dickson speedup evidence.

Those relations remain absent. The pinned Metal feature tables require their
own Garden-first Sounio parser before permutation semantics can be proposed.

## Stage Boundary

This bundle is produced by `SOUNIO_EXECUTABLE` and is submitted for
`SEMANTICS_FROZEN`. It does not open `PARITY_OPEN` or `CLAIM_READY` by itself.
