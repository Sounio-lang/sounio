<!-- docs:meta
topic_id: repo.docs.research.pireus-graph-identity-composition-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-graph-identity-composition-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus Graph Identity Composition Semantics v0

Date: `2026-08-27`

Stage: `SEMANTICS_FROZEN`

Language-Producer: `Sounio`

Language-Role: `SEMANTIC_AUTHORITY`

## Mandatory Order

The Garden seed was committed as `9bb946a4ed64` before the identity module or
witness existed. Commit `122a3cc591` then produced the first Sounio result
without an expected count or digest in its source. This document and the
frozen-result predicate were created only after two byte-identical executions.

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
```

Parity and claim promotion remain closed.

## Admitted Producers

The registry is created before any corpus or store is read.

| Graph | Parent | Namespace owner | Frozen source SHA-256 |
| --- | --- | --- | --- |
| model | none | Pireus core | `ee4589ab4dad2a47a136629dcab6e93aa2f215cf114a8e2f7b3f24a89d39ed9d` |
| target profile | model | target profile | `d41726a8a7eba62132e3763cf6a71938de746ec9d58ce8a20caa40709546d6a4` |
| execution engine | target profile | execution engine | `8b5063f0e9a39650fb0b60e8b70b315f339723690e06050c2bebacece888e37e` |
| XED | target profile | XED | `c65d63a490038d874f9d1ae34458ff44793049eb7ec01bee01981df7974cbeb9` |
| AARCHMRS | target profile | AARCHMRS | `ce0693e51f5204f89c67b7917fd129dc1976f069675323ec73d4e2c42913078b` |
| PTX | target profile | PTX | `ca2760d539c4602c85841ac8475a9ffd8a2f760313a8169faf99a32956063bba` |
| Apple Metal | execution engine | Apple Metal | `b43f48c723283d65c3e1df1824f6284303a71967e20deab2c9fe8c7b72f97587` |

The language namespace is a separate eighth owner. Local IRI `1` is its
explicit shared RDF type term.

## Lifted Identity

There are eight bounded, non-overlapping owner allocations. For owner `o` and
local integer `l`, with `0 <= l < 1,000,000`:

```text
allocation_start(o) = o * 2,000,000
lifted_iri(o, l)     = allocation_start(o) + l
lifted_literal(o, l) = allocation_start(o) + 1,000,000 + l
```

The IRI/literal sort is retained independently of the lifted integer. Literal
canonicalization also includes the exact `f64_to_bits` payload.

Each graph processes its exact source triples in order. An already registered
term in the declared ancestor closure becomes an explicit
`FOREIGN_PARENT` reference to the same owner. A new local term becomes owned by
the current producer's declared namespace. More than one ancestor owner for a
local term fails closed. Ownership is never inferred from a numeric range.

## Parent Closures

The seven full stores contain 1,621 source occurrences. For every non-root
graph, Sounio compares the complete declared parent prefix field by field,
including literal sort and exact value bits.

```text
source occurrences       = 1621
inherited occurrences    = 971
producer-local occurrences = 650
canonical lifted triples = 650
canonical triples with multiple occurrences = 290
```

Inherited copies remain provenance. They do not become new canonical query
rows.

## Complete Collision Census

All collision rows have IRI sort `1`, owner `4` for execution engine, and owner
`6` for AARCHMRS.

```text
703000 703001 703002 703003 703004
703100 703101 703102 703103 703104
703200 703210 703211 703220 703221 703230 703231 703240
703300 703301 703302 703303 703304 703310
```

This is 24 distinct local keys and 24 owner pairs. The six `7033xx` keys arise
from store materialization, so the declaration-only observation of 18 values
is rejected as a complete census.

## Composed Queries

| Query | Canonical result | Source occurrences |
| --- | ---: | ---: |
| canonical targets | 3 | 18 |
| machine-to-profile typed join | 5 | n/a |
| Apple graph to execution-engine Apple blueprint | 1 | n/a |

The profile join binds an IRI object and reuses it in subject position. The
query layer rejects an otherwise numerically equal literal binding in that
position.

## Negative Surface

All 26 deliberate negatives pass. They cover absent or duplicate producer and
namespace registrations, overlap, missing/cyclic/reordered/hash-mismatched
parents, changed closure, unregistered IRIs, invalid shared/foreign ownership,
incompatible or double lifted mappings, IRI/literal and query-binding sort
swaps, missing literal owners, changed literal bits, overflow, capacity,
inherited query inflation, occurrence loss, declaration-only census,
provenance loss, and local-query promotion without a composition receipt.

## Frozen Digests

```text
registry=9b56f6f0306d949e2266776ee34f05f3ba1dec4239e0bba9411b3aed9c2b27ce
dependency=4dd37bf1cdd774e4ab840e5444d7b18b8a1d0990063901b8a85743a7ac2abbcc
lifted_graph=0bcf3ef8b9598cb4363864d9ba75d9b050a22df501b80a09eda7290b3e331765
occurrence=57218fbb4a6d640e4651dea0d14a17a54559a2f559e45e3186a46df7d8a05950
collision=3a72cc5158aa0e841b4b13de2a924d1bca516778b651ae3f1fe9be80d26925bb
provenance=1e962677cfb1846a5e5b9dd70c13c25cae5f92ad905f6ad795a8912b4e352f20
```

The pre-freeze Sounio authority stream is 4,813 bytes over 263 lines and has
SHA-256
`5b3efa606d86805aa222ced72a37ed87e7b3dab66b21e58e0547163aa19c83dd`.

## Non-Claims

The frozen semantics emit zero instruction equivalences, processor
observations, capability inheritance, lowering choices, and performance
claims. Apple Silicon and DGX remain canonical targets, not observed devices.
The five observed Darwin CPU profiles remain Xeon.
