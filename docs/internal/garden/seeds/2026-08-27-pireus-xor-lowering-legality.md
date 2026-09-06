<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-pireus-xor-lowering-legality
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-pireus-xor-lowering-legality
-->

# Garden Seed: Pireus XOR Lowering Legality

Date: `2026-08-27`

Concept-ID: `SOUNIO-PIREUS-XOR-LOWERING-LEGALITY`

Founder direction:

> mas o meu tesao agora e achar a MINHA FORMA de resolver isso

Continue the current Pireus and Loom system. Do not create a second guardian,
compiler, operation semantics, or competing lowering architecture.

Status: `GARDEN`

## Question

Starting from the frozen bits=4 Pireus XorConvolution operation, material
layout, and Intel selector semantics, what complete target-independent lowering
plan can Sounio derive for the five-node operation DAG while preserving every
semantic barrier, especially the fixed ascending-`i` `f64` reduction order?

This Garden admits the first Sounio legality experiment. It does not admit an
expected plan, sign mechanism, instruction sequence, instruction count,
reassociation, target observation, cost, or performance claim.

## Frozen Parents

The child must reject any parent drift from:

```text
operation_source_sha256=bc039d5db9f195b94fbeb08f22f9c96164a174c2cea675739e901a07fdf54db8
operation_semantics_sha256=40fe69829b1feb5843ea8b4720b70516303e8eda37c144989909b52d1b466fb1
operation_receipt_sha256=9e1e416defa4a1cfc477f0630623321e12209a40c9f5f878b85cce40be83d330
material_source_sha256=eadd752fbda1f50f24bed1260c54936d710af10973653982f5687cd8a551a575
material_semantics_sha256=b4791514032859acc0e8888c4d35760f549a6267e02b2cd5f30a96c0b9dee554
material_receipt_sha256=cc157a7d6ba33b945bc9537be1856bf60481573067ade5f166101ca36a98c1df
intel_selector_source_sha256=4f7e007aa432564b873c941e239c353aeed3b11883844b73ec7e31ace4811b20
intel_selector_semantics_sha256=ba25ceb18685ed656ecaf1c577eb95a698ca214fe2a939fce5a8ffd6d106b243
intel_selector_receipt_sha256=fddad1442d0b21201bccf57fce380a2d57a94bb55bee9924636d06473128218f
```

The operation parent owns the mathematical DAG and exact accumulation order.
The material parent owns the complete two-chunk XOR selector layout. The Intel
parent owns only the bounded selector semantics and its finite match verdict.
No parent owns a complete lowering of the operation.

## Operation Boundary

The child must inspect all five frozen nodes in order:

```text
XOR_PERMUTE
TWIST_APPLY
MULTIPLY
HORIZONTAL_REDUCE
OUTPUT_LANE
```

For every node, Sounio must derive a legality record from the frozen parent
capabilities, barriers, topology, and numeric order. A schema may contain:

```text
node_kind
candidate_form_kind
candidate_group_count
source_chunk_role
destination_chunk_role
sign_or_mask_role
exact_order_preserved
reassociation_required
semantic_barrier_crossed
material_receipt_required
lowering_authorized
refusal_reason
```

These are field names, not expected field values. A field that cannot be
derived from the frozen parents must remain absent or unresolved.

## Exact Numeric Boundary

The frozen operation accumulates each output in ascending `i` order. IEEE-754
`f64` addition is not generally associative, so a SIMD tree reduction cannot
be silently substituted for that sequential fold.

In this lane, `exact` means bit-identical to the frozen Sounio parent
evaluation with its declared ascending-`i` order. This Garden does not infer a
rounding mode that the parent did not declare. A target rounding environment
belongs to a later material receipt and cannot amend the semantic parent.

The first executable must distinguish at least:

- an exact candidate that preserves the frozen order;
- a candidate that requires reassociation;
- an unresolved candidate whose numeric behavior lacks evidence.

The existence of a faster reduction shape cannot authorize it. Reassociation
requires a separately admitted Sounio numerical or refinement contract with
its own error and acceptance semantics. This Garden creates no such contract.

## Intel Selector Boundary

The frozen Intel lane establishes only that its accepted vector-control form
can cover the finite `XOR_PERMUTE` selector cells under the exact admitted
profile. It does not lower `TWIST_APPLY`, `MULTIPLY`,
`HORIZONTAL_REDUCE`, or `OUTPUT_LANE`.

The child may consume the frozen selector verdict as coverage for the one
matching DAG node. It must not promote that local fact into:

- a complete operation lowering;
- emitted use on a compiler path;
- a target capability observation;
- a sign-application mechanism;
- a reduction mechanism;
- an instruction or micro-operation count;
- latency, throughput, energy, or speedup.

## First Sounio Executable

After this Garden is committed, the first child executable must:

1. bind all nine frozen parent hashes exactly;
2. live-import and validate the frozen operation, material, and Intel parents;
3. recover the five-node topology and every frozen semantic barrier;
4. derive one canonical legality record for each node;
5. keep target-independent semantic legality separate from material forms;
6. keep exact-order candidates separate from reassociated candidates;
7. refuse any lowering that crosses an unresolved semantic barrier;
8. emit canonical records and a digest before expectations are added;
9. emit the facts that remain unresolved rather than filling them from memory;
10. preserve Darwin Xeon, Apple Silicon, and DGX as canonical targets without
    treating canonical declaration as observation;
11. emit zero cost, performance, hardware-observation, or compiler-emission
    claims;
12. leave `PARITY_OPEN=false` and `CLAIM_READY=false`.

No expected record, record count, digest, lowering authorization, candidate
mechanism, group count, or refusal count may be written before the first
Sounio execution emits it.

## Required Negative Surface

At minimum, the Sounio child must reject:

1. execution before this Garden commit exists;
2. a missing or mismatched operation source, semantics, or receipt;
3. a missing or mismatched material source, semantics, or receipt;
4. a missing or mismatched Intel selector source, semantics, or receipt;
5. a missing, duplicated, reordered, or disconnected DAG node;
6. a changed operation capability or barrier;
7. a changed ascending-`i` reduction order;
8. local selector coverage promoted to whole-operation coverage;
9. a sign mechanism imported from memory or an external implementation;
10. a multiply mechanism without a derived semantic record;
11. a tree reduction promoted to exact lowering;
12. reassociation without a separately frozen Sounio numerical contract;
13. a target material form promoted from ISA-family resemblance;
14. an AVX-512 capability promoted to emitted or executed use;
15. a Darwin, Apple Silicon, or DGX observation without a material receipt;
16. an instruction count, cost, throughput, latency, or speedup claim;
17. a host disassembler, compiler, vendor tool, or hardware probe promoted to
    semantic authority;
18. a parity language or external model promoted to semantic authority;
19. parity or claim promotion before their Loom transitions;
20. Python or Rust as producer, oracle, or guardian.

## Canonical Target Boundary

| Target | Status in this lane | Required later evidence |
| --- | --- | --- |
| Darwin Xeon | canonical, unobserved | frozen plan plus exact compiler and hardware receipt |
| Apple Silicon | canonical, unresolved | frozen plan plus Apple material semantics and receipt |
| DGX | canonical, unresolved | frozen plan plus NVIDIA material semantics and receipt |

All Darwin CPUs in the frozen target profile are Xeon. The present Xeon host
may later provide material parity, but hardware facts cannot create this
lane's semantics. Apple Silicon and DGX remain equally canonical even when
their observation surfaces are elsewhere.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: pireus-xor-lowering-legality-20260827
Owner: codex/session-01a040f3-2b73-76e2-bbf7-
Concept-IDs: SOUNIO-PIREUS-XOR-LOWERING-LEGALITY; SOUNIO-PIREUS-XOR-CONVOLUTION-OPERATION; SOUNIO-PIREUS-XOR-MATERIAL-MATCHING; SOUNIO-PIREUS-INTEL-VPERMPD-SELECTOR-SEMANTICS
Intent-Preserved: Sounio derives the complete lowering-legality boundary before any target implementation measures or confirms it
Transformation: three frozen Pireus parents to a five-node target-independent lowering-legality plan
Types-Changed: none in Garden
Effects-Changed: none in Garden
IR-Changed: none
Claims-Introduced: none in Garden
Claims-Forbidden: expected plan values; mechanisms; instruction counts; reassociation authorization; emitted use; target observation; cost; performance; cross-ISA parity
Assumptions: the three frozen parent triplets remain byte-identical and live-importable
Write-Set: this Garden seed; concept registry; generated documentation governance metadata
Read-Set: frozen operation parent; frozen material parent; frozen Intel selector parent
Positive-Witness: Garden admission before the first child executable
Negative-Witness: deliberate pre-Garden child and prohibited-oracle attempts are denied
Acceptance-Gate: Loom GARDEN admission plus docs registry and semantic coordination gates
Integration-Target: current Pireus operation and material-matching pipeline
Authoritative-Only-If: the first plan is emitted by Sounio after this Garden commit and later frozen by exact hash
```

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

Lean 4, Koka, C++, Haskell, vendor tools, external models, compilers, and
hardware probes cannot create the expected plan. They may compare, prove, or
measure only after the Sounio artifact is frozen and identified by hash.

External LLMs remain `REVIEW_ONLY`. Python and Rust are prohibited. Node may
run only the existing deterministic documentation metadata generator and may
not derive, compute, or confirm lowering semantics.

## What This Is Not

This seed is not:

- an instruction selector implementation;
- a claim that the Intel selector form completes the operation;
- a decision about sign-bit masks or arithmetic negation;
- a decision about scalar, vector, or tree reduction;
- permission to reassociate floating-point addition;
- a `~112` instruction estimate or any replacement estimate;
- a compiler-emission or hardware-performance experiment;
- an Apple Silicon or DGX material result;
- a Walsh-Hadamard, cocycle-diagonalization, or subquadratic claim;
- formal, effect, material, or denotational parity;
- a production, novelty, or claim-ready assertion.

The next executable bridge is one Sounio-derived legality plan for the complete
five-node DAG, with unresolved barriers retained as first-class results.
