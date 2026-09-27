<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-pireus-xor-selector-material-admission
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-pireus-xor-selector-material-admission
-->

# Garden Seed: Pireus XOR Selector Material Admission

Date: `2026-08-27`

Concept-ID: `SOUNIO-PIREUS-XOR-SELECTOR-MATERIAL-ADMISSION`

Founder direction:

> continue, but incorporate this as acceptance criteria

Status: `GARDEN`

## Question

Three canonical target receipts now exist after the frozen bits=4 Pireus XOR
lowering-legality plan:

- one Darwin Xeon receipt for a complete finite operation realization;
- one Apple Silicon receipt for an A64 `TBL` selector realization; and
- one DGX GB10 receipt for a PTX/SASS `SHFL.BFLY` selector realization.

How can Sounio admit these external material observations into one typed
evidence overlay without allowing C++, a compiler, hardware, a vendor operand
encoding, or an external reviewer to revise the frozen operation semantics?

This Garden opens the admission boundary. It does not admit a cost model,
cross-ISA equivalence, an instruction selector, a complete Apple or DGX
lowering, a performance result, or `CLAIM_READY`.

## Why Admission Precedes Cost

The frozen operation has five ordered semantic nodes:

```text
XOR_PERMUTE
-> TWIST_APPLY
-> MULTIPLY
-> HORIZONTAL_REDUCE
-> OUTPUT_LANE
```

The material receipts do not cover this graph uniformly. A cost comparison
before typed coverage admission would compare unlike objects and could turn a
selector observation into a whole-operation price. Pireus must first answer:

```text
which exact frozen parent was compared?
which node was explicitly witnessed?
which node remains unresolved?
which facts are material observations rather than semantic results?
```

Only a later Sounio-first cost Garden may decide how admitted observations are
eligible for comparison.

## Frozen Semantic Parent

The child must live-import and validate the frozen lowering-legality result:

```text
source=stdlib/hardware/pireus/xor_lowering_legality.sio
source_sha256=7087649a5cfdb41a884aa9a2e1b0b64bbe2d25da3ca3cd1d54d5b70429854edb
semantics=docs/research/pireus_xor_lowering_legality_semantics.md
semantics_sha256=9e92f256c25a774979a1b02cb10d5d39a1ae702ccec3a273044f91930f03a970
receipt=docs/research/receipts/pireus_xor_lowering_legality_20260827.md
receipt_sha256=daef832ee6370b656e93ae84c76ba6d17c98aaf5ad1dd86674dee27ba0f84346
```

That parent owns node identity, node order, semantic capabilities, sign masks,
ascending-`i` reduction order, canonical target declarations, and the absence
of cost and whole-operation material authorization.

The child may not modify the parent or write observations retrospectively into
its frozen result.

## Material Evidence Parents

The exact external observations admitted as inputs are:

```text
darwin_receipt=docs/research/receipts/pireus_xor_lowering_darwin_xeon_material_parity_20260827.md
darwin_receipt_sha256=342d8ba8808c2a926bb2bbf0c09488f7b849967239c932687952ec6ae789a906
darwin_evidence=docs/research/evidence/pireus_xor_lowering_darwin_xeon_material_parity_20260827.txt
darwin_evidence_sha256=ee37914bc738eb829f3589249f228e4a8312310fbffa0b00636cd0c9ed9a40d1

apple_receipt=docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md
apple_receipt_sha256=c00a3d4e556688829efadbbf640ea858cfe9520dc04103fa745cf1a8101f7840
apple_evidence=docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt
apple_evidence_sha256=2877bfd463b4d28dc3311b75c69bec2aa1c62b430d08314989187d44b32a781e

dgx_receipt=docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md
dgx_receipt_sha256=3c10882eff43d3b197428839996c7a04c009c8f537d0c1451bdf3e8a13e2f385
dgx_evidence=docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt
dgx_evidence_sha256=2c6b6e448265a5566d17df9a674246ea62c05210e432e48e418d16358496853b
```

These hashes identify observations. They are not semantic-parent hashes and
cannot create a Sounio expected result.

## Ownership Boundary

The admission overlay must preserve both directions of authority:

```text
Sounio owns:
  semantic nodes
  exact finite coordinate requirements
  reduction-order requirement
  receipt schema
  admission predicates
  evidence-strength classification
  promotion gates

Material receipt owns:
  observed target identity
  observed toolchain and hardware
  executed command
  observed emitted artifacts
  observed finite comparison result
  reproducibility qualifications
```

Sounio must not claim to have measured the target. The material producer must
not define the semantic node or expected coordinate map.

## Typed Admission Record

The first Sounio executable may define a record containing:

```text
target_id
producer_language
producer_role
receipt_stage
semantic_parent_sha256
receipt_sha256
evidence_sha256
toolchain_sha256
hardware_sha256
command_sha256
result_sha256
node_coverage_assertions[5]
finite_coordinate_cells
finite_coordinate_matches
finite_coordinate_mismatches
payload_bits_checked
payload_bits_matched
reduction_order_observed
operand_encoding_equal
artifact_reproducibility
receipt_admitted
admission_reason
```

These are field names, not expected output values. The first execution must
emit its canonical records and digest before a matcher contains expected
records or digests.

## Node Coverage Rule

Coverage is per semantic node, never per target name or receipt title. A node
may be admitted only when the receipt names evidence sufficient for that node
under the frozen legality parent.

For example:

- finite XOR coordinates may witness `XOR_PERMUTE`;
- sign-cell and sign-mutation evidence may witness `TWIST_APPLY`;
- term-by-term product evidence may witness `MULTIPLY`;
- ascending-order bit parity with reassociation refused may witness
  `HORIZONTAL_REDUCE`; and
- exact frozen output lanes may witness `OUTPUT_LANE`.

A count such as `unresolved_other_nodes=3` does not identify which nodes were
resolved. The first child must not infer a node identity from that count. It
must preserve `not_named` until an explicit evidence field or a descendant
receipt names the coverage.

Static emitted-instruction sites are material evidence about one compiler and
command. They cannot independently establish semantic node coverage.

## Encoding Boundary

The DGX receipt distinguishes:

```text
frozen Sounio abstract packed c = 15
emitted PTX operand c          = 4127 (0x101f)
```

Finite coordinate and payload parity does not imply operand-encoding equality.
The child must carry these as separate fields and must reject an admission that
requires encoding equality for coordinate coverage.

Likewise, one Apple `tbl.16b` static site and three Darwin `vpermpd` static
sites are observations, not minimum instruction counts or cross-target costs.

## First Sounio Executable

Only after this Garden commit exists, the first child executable must:

1. bind the exact Garden commit and all nine parent hashes above;
2. live-import and validate the frozen lowering-legality parent;
3. construct material receipt inputs with explicit provenance roles;
4. derive the five-node coverage status for each target;
5. keep `observed`, `admitted`, `unresolved`, and `refused` distinct;
6. derive every admitted coordinate expectation from the Sounio parent;
7. preserve target-local parity without asserting cross-ISA equivalence;
8. preserve the DGX coordinate-versus-encoding distinction;
9. refuse unnamed node coverage rather than infer it from a count;
10. emit canonical records and a digest before expectations are added;
11. emit zero latency, throughput, energy, wall-clock, or generic cost fields;
12. leave incomplete Apple and DGX operation coverage visible;
13. leave all transform and subquadratic claims closed;
14. leave `CLAIM_READY=false`.

The child may enter `SEMANTICS_FROZEN` after byte-identical Sounio runs. Its
admission of already sealed material receipts does not retroactively make the
parent lowering plan material or claim-ready.

## Required Negative Surface

At minimum, the Sounio child must reject:

1. execution before this Garden commit exists;
2. a missing or changed lowering source, semantics, or receipt hash;
3. a missing or changed material receipt or evidence hash;
4. a material receipt whose semantic parent does not match its frozen target
   selector or lowering parent;
5. a producer role other than `MATERIAL_PARITY`;
6. a receipt outside `PARITY_OPEN` or without a valid sealed receipt;
7. a zero toolchain, hardware, command, or result binding;
8. a nonzero coordinate mismatch admitted as exact parity;
9. payload parity promoted without the target receipt checking payload bits;
10. an unnamed node inferred from an unresolved-node count;
11. a selector observation promoted to all five nodes;
12. an emitted instruction site promoted to a generic instruction cost;
13. the DGX abstract operand `15` promoted to emitted operand equality;
14. two target-local results promoted to cross-ISA equivalence;
15. a complete Darwin receipt promoted to complete Apple or DGX coverage;
16. a cost, latency, throughput, energy, or speedup field;
17. a Walsh-Hadamard or subquadratic promotion;
18. an LLM review promoted to evidence or authority;
19. `CLAIM_READY` while any required promotion gate is closed; and
20. Python or Rust as producer, oracle, or guardian.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: pireus-xor-selector-material-admission-20260827
Owner: codex/session-01a040f3-2b73-76e2-bbf7-
Concept-IDs: SOUNIO-PIREUS-XOR-SELECTOR-MATERIAL-ADMISSION; SOUNIO-PIREUS-XOR-LOWERING-LEGALITY
Intent-Preserved: Sounio defines how external material observations are admitted without taking ownership of the observations or surrendering semantic authority
Transformation: one frozen five-node legality parent plus three sealed target-local material receipts to a typed Sounio evidence-admission overlay
Types-Changed: none in Garden
Effects-Changed: none in Garden
IR-Changed: none
Claims-Introduced: none in Garden
Claims-Forbidden: inferred unnamed coverage; whole-operation Apple or DGX lowering; cross-ISA equivalence; cost; performance; transform legality; claim readiness
Assumptions: all exact parent files remain byte-identical and the lowering parent remains live-importable
Write-Set: this Garden seed; concept registry; generated documentation governance metadata
Read-Set: frozen lowering legality and the three sealed material receipt/evidence pairs
Positive-Witness: Garden admission before any Sounio child execution
Negative-Witness: pre-Garden child, prohibited producer, unnamed coverage, and promotion attempts are denied
Acceptance-Gate: Loom GARDEN admission plus math review, docs registry, and semantic coordination gates
Integration-Target: Pireus XOR operation evidence overlay before any cost model
Authoritative-Only-If: the first admission records and digest are emitted by Sounio after this Garden commit and later frozen by exact hash
```

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

External LLMs remain `REVIEW_ONLY`. C++ remains `MATERIAL_PARITY`. Lean 4,
Koka, and optional Haskell remain closed until the new Sounio artifact is
frozen. Node may run only the existing deterministic documentation metadata
generator and cannot compute or confirm admission semantics.

## Exit Gate

This Garden reaches its own exit only when:

1. Loom admits the exact Garden frame;
2. the Garden and concept registry commit exists;
3. external review has no unresolved wrong claim;
4. documentation governance is green;
5. no Sounio child has executed before the commit; and
6. no material, cost, equivalence, or claim-ready status is widened.

The next artifact is one first-run Sounio executable that classifies the three
receipt inputs while preserving incomplete target coverage as data.
