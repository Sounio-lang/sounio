<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-pireus-xor-material-matching
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-pireus-xor-material-matching
-->

# Garden Seed: Pireus XorConvolution Material Matching

Date: `2026-08-27`

Concept-ID: `SOUNIO-PIREUS-XOR-MATERIAL-MATCHING`

Founder direction: continue the current Pireus system; do not create a second
guardian, compiler, operation semantics, or competing ontology.

Status: `GARDEN`

## Question

Starting from the frozen bits=4 Pireus XorConvolution operation DAG, what is
the smallest target-shaped selector plan that preserves the exact
`XOR_PERMUTE` node, and which additional vendor-semantic or hardware receipts
are required before any instruction form can be called a material match?

This Garden admits the question and the finite selector-plan experiment. It
does not admit an instruction equivalence, lowering, cost, or speedup.

## Frozen Parents

The child must reject any parent drift from:

```text
operation_semantics_sha256=40fe69829b1feb5843ea8b4720b70516303e8eda37c144989909b52d1b466fb1
operation_receipt_sha256=9e1e416defa4a1cfc477f0630623321e12209a40c9f5f878b85cce40be83d330
xor_convolution_semantics_sha256=da782da938ee5f9e0a49cb1f95dfbb6acac8aa706c9eb6d711565adcb9031502
graph_identity_semantics_sha256=8dc9c6c90d4f21b13c07d8ec3e914839b9f3bfaa1e32f222a25bdcb267c943cb
xed_import_frozen_content_sha256=5d9a56cd05eb141b24dfa80bbab74f41306bb19a01902c25fb0feeda63265612
xed_import_current_envelope_sha256=d96d6d57ba1e296930caec5f4f0aff8e2898b3b1d5df6bfaacb96a19333266f7
xed_import_current_receipt_sha256=2dfc243381acb8d365112b3b4075ccabf944de6ff081b4626f9a4f693f136af6
```

The XED importer froze the semantic content before documentation governance
added a `historical` metadata envelope. Git lineage proves that the content at
commit `2bdf61194c68747f94a9a054824f5bfcff2c22b2` hashes to `5d9a...`; commit
`32f7151419bdccf37115901dcbf76c6574366aec` added only the metadata and status
envelope, yielding the current `d96d...` file. The executable must bind both
objects and the current receipt file, and must reject drift in any of them.

This is a frozen XED evidence snapshot, not an instruction-behavior theorem
and not a claim that the current historically labelled page is a canonical
documentation surface. The frozen content remains useful only for the narrow
form-presence facts already recorded by the Sounio importer receipt.

## Admitted Darwin Candidate Layout

The first target-shaped plan is bounded to:

```text
element_bits=64
logical_dimension=16
chunk_lanes=8
chunk_count=2
```

This is the `f64x16` operation shape projected onto two `f64x8` chunks. It is
not a claim that a compiler selected ZMM registers or that a Darwin machine
executed AVX-512.

For displacement `d` in `[0,15]`, output chunk `c` in `[0,1]`, and output lane
`l` in `[0,7]`, define:

```text
i            = 8*c + l
j            = i XOR d
source_chunk = c XOR (d >> 3)
source_lane  = l XOR (d & 7)
```

The first executable must enumerate all `16 * 2 * 8 = 256` cells and verify:

```text
j = 8*source_chunk + source_lane
```

For fixed `(d,c)`, `source_chunk` is independent of `l`. Therefore the abstract
partner permutation needs one source chunk per output chunk plus an eight-lane
index selector. This is a finite layout fact, not yet an ISA fact.

## XED Evidence Boundary

The frozen XED importer proves only that the pinned vendor corpus contains
eight accepted AVX-512 f64 forms across `VPERMPD`, `VPERMI2PD`, and
`VPERMT2PD`, preserving raw operand and selector-syntax distinctions.

The next executable may report:

```text
candidate_family_present=true|false
selector_semantics_receipt_present=false
instruction_match_authorized=false
```

It must not infer any of the following from form presence alone:

- that `VPERMPD` vector indices select arbitrary f64 lanes;
- that its `IMM8` form realizes every XOR selector;
- that `VPERMI2PD` or `VPERMT2PD` is necessary or unnecessary;
- operand roles not frozen by the importer;
- availability or execution on a Darwin node;
- instruction count, latency, throughput, or lowering legality.

A future material match requires a separately admitted, pinned vendor-semantic
receipt plus a target observation or emitted-code receipt as appropriate.

## Operation Boundary

This lane concerns only `XOR_PERMUTE`. It does not lower:

- `TWIST_APPLY`;
- `MULTIPLY`;
- the fixed ascending-`i` `HORIZONTAL_REDUCE`;
- `OUTPUT_LANE`.

In particular, a tree reduction is not silently equivalent to the frozen
ascending-`i` f64 accumulation. Reassociation requires its own Sounio semantic
contract and exact result evidence.

## Canonical Targets

The executable must retain all three canonical targets:

| Target | Candidate plan in this lane | Observation authority |
| --- | --- | --- |
| Darwin Xeon | two f64x8 chunks | frozen target profile plus later exact run receipt |
| Apple Silicon | unresolved | later Apple vendor semantics and hardware receipt |
| DGX | unresolved | later NVIDIA vendor semantics and hardware receipt |

The frozen target profile records five Darwin CPU machines, all Xeon. The
current control host corresponds to the admitted t560 Xeon Gold 6526Y profile,
but that fact alone does not establish AVX-512 use by this operation.

Apple Silicon and DGX are canonical targets. Canonical declaration is not an
observation and cannot be promoted into one.

## First Sounio Executable

After this Garden is committed, the first executable should:

1. live-import and match the frozen Pireus XorConvolution operation parent;
2. bind the complete frozen operation receipt and XED-import semantics hashes;
3. enumerate the 256 selector cells in Sounio;
4. emit every `(d,c,l,i,j,source_chunk,source_lane)` row in canonical order;
5. prove by complete enumeration that every cell reconstructs `j`;
6. prove by complete enumeration that one source chunk serves each fixed
   `(d,c)` output chunk;
7. record XED family presence separately from selector-semantic authorization;
8. retain Apple Silicon and DGX as unresolved canonical target rows;
9. emit exact Sounio digests and negative witnesses;
10. keep `PARITY_OPEN=false` and `CLAIM_READY=false`.

No expected row, count, digest, selector table, or candidate verdict may be
written before that first Sounio execution.

## Required Negative Surface

At minimum, the Sounio executable must reject:

1. missing operation-parent hash;
2. mismatched operation-parent hash;
3. missing operation receipt;
4. mismatched operation receipt;
5. wrong logical bit width;
6. wrong dimension;
7. wrong element width;
8. wrong chunk width;
9. displacement outside `[0,15]`;
10. output chunk outside `[0,1]`;
11. output lane outside `[0,7]`;
12. a corrupted source-chunk selector;
13. a corrupted source-lane selector;
14. a two-source-necessity claim derived only from layout;
15. an `IMM8` sufficiency claim without vendor-semantic evidence;
16. an XED form-presence to instruction-behavior promotion;
17. an AVX-512 capability to emitted-use promotion;
18. an Apple Silicon or DGX observation without a receipt;
19. a cost or instruction-count claim without a material receipt;
20. parity or claim promotion before their Loom transitions;
21. Python or Rust as producer or oracle.

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

Lean 4, Koka, C++, Haskell, vendor tools, external models, and hardware probes
cannot create the selector plan's expected result. They may act only after the
Sounio artifact is frozen and bound by hash, under their declared roles.

## Closed Claims

Until separately admitted, this Garden does not establish:

- that one-source `VPERMPD` is sufficient on hardware;
- that `VPERMI2PD` or `VPERMT2PD` is required or avoidable;
- any immediate selector encoding;
- any instruction count, including the earlier `~112` estimate;
- any lowering for twist application, multiplication, or fixed-order reduce;
- any observation on Apple Silicon or DGX;
- a Fano-plane explanation;
- Walsh-Hadamard diagonalization or subquadratic complexity;
- Lean, Koka, C++, or Haskell parity;
- performance, novelty, production readiness, or claim-ready status.

External LLMs remain `REVIEW_ONLY`. Python and Rust are prohibited. Node may
run only the existing deterministic documentation metadata generator and may
not compute or confirm selector results.
