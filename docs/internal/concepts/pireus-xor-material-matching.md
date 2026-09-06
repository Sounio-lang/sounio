<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-xor-material-matching
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pireus-xor-material-matching
-->

# Pireus XOR Material Matching

Concept-ID: `SOUNIO-PIREUS-XOR-MATERIAL-MATCHING`

Status: `SEMANTICS_FROZEN`

Semantic-Lane-ID: `pireus-xor-material-20260827`

## Intent

Give the frozen bits=4 Pireus `XOR_PERMUTE` node its first target-shaped
selector plan without selecting an instruction, claiming a lowering, or
recording a hardware observation.

The semantic authority is Sounio:

```text
stdlib/hardware/pireus/xor_material_matching.sio
examples/pireus_xor_material_matching.sio
tests/stdlib/hardware/test_pireus_xor_material_matching.sio
```

## Frozen Shape

The logical `f64x16` value is projected onto two eight-lane chunks. For
displacement `d`, output chunk `c`, and output lane `l`:

```text
i            = 8*c + l
j            = i XOR d
source_chunk = c XOR (d >> 3)
source_lane  = l XOR (d & 7)
```

Sounio enumerates all 256 cells and checks that
`j = 8*source_chunk + source_lane`. It also checks all 32 fixed `(d,c)` groups
and establishes that each group reads one abstract source chunk plus an
eight-lane selector.

This is a finite layout result. It is not an instruction equivalence.

## XED Boundary

The executable reloads the pinned Intel XED AVX-512F corpus, validates its
vendor SHA-256 through the frozen Sounio importer, rebuilds its ontology, and
queries eight relevant forms: four `VPERMPD`, two `VPERMI2PD`, and two
`VPERMT2PD`.

Form presence does not establish selector behavior. The frozen result keeps
selector-semantic receipt, instruction match, immediate sufficiency, and
two-source necessity authorization false.

## Canonical Targets

Darwin Xeon, Apple Silicon, and DGX remain canonical. The two-chunk plan is
attached only to Darwin Xeon as an abstract candidate layout. All targets are
unobserved; Apple Silicon and DGX plans remain unresolved.

## Closed Claims

The result establishes no lowering for `TWIST_APPLY`, `MULTIPLY`, fixed-order
`HORIZONTAL_REDUCE`, or `OUTPUT_LANE`. It records no emitted instruction,
instruction count, cost, latency, throughput, speedup, or target observation.

The external Loom guardian remains mandatory for producer-language and stage
transitions. The local classifier is a semantic role witness, not a guardian
replacement. `PARITY_OPEN=false` and `CLAIM_READY=false`.
