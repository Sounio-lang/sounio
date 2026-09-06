<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-xor-convolution-operation
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pireus-xor-convolution-operation
-->

# Pireus XorConvolution Operation DAG

Concept-ID: `SOUNIO-PIREUS-XOR-CONVOLUTION-OPERATION`

Status: `SEMANTICS_FROZEN`

Semantic-Lane-ID: `pireus-xor-operation-20260827`

## Intent

Bind the frozen bits=4 `XorConvolution` contract to a material-neutral Pireus
operation graph without selecting an ISA instruction, a target lowering, or a
cost model.

The semantic authority is the Sounio pair:

```text
stdlib/hardware/pireus/xor_convolution_operation.sio
examples/pireus_xor_convolution_operation.sio
```

## Frozen Parents

The operation imports and executes the frozen XorConvolution parent. The
Pireus graph-identity parent is bound through its frozen Sounio receipt because
the current bootstrap compiler flattens private imported helpers and cannot
compile both parents in one bundle without name collisions.

This receipt binding does not recreate graph identity. It pins the exact
parent module, semantics, authority stream, registry, dependency, lifted graph,
occurrence, collision, and provenance hashes.

## Operation Graph

The canonical path is:

```text
XOR_PERMUTE
-> TWIST_APPLY
-> MULTIPLY
-> HORIZONTAL_REDUCE
-> OUTPUT_LANE
```

`HORIZONTAL_REDUCE` uses the frozen ascending-`i` order. `TWIST_APPLY`,
`HORIZONTAL_REDUCE`, and `OUTPUT_LANE` are nonassociative barriers. The graph
contains no material instruction identity.

## Canonical Targets

Darwin Xeon, Apple Silicon, and DGX are declared canonical targets. All three
remain unobserved in this semantic result. Canonical declaration is not
material evidence.

## Evidence Boundary

The frozen result establishes the five-node DAG, capability requirements,
parent bindings, exact output bits, negative promotion gates, and six Sounio
digests. It preserves the parent's classification as a nonassociative
XOR-graded algebra with an explicit associator defect.

A Walsh-Hadamard rewrite is not authorized. This records the absence of the
required transform-identity receipt; it does not prove that no structured or
subquadratic transform exists.

The result establishes no instruction coverage, lowering, cost, performance,
hardware observation, Fano-plane interpretation, or cross-language parity.
`PARITY_OPEN=false` and `CLAIM_READY=false`.
