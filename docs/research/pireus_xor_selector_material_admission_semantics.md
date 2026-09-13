<!-- docs:meta
topic_id: repo.docs.research.pireus-xor-selector-material-admission-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-xor-selector-material-admission-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Frozen Semantics: Pireus XOR Selector Material Admission

> **Status**: Semantics frozen | **Date**: 2026-08-27
>
> **Producer**: Sounio | **Role**: `SEMANTIC_AUTHORITY`

## Causal Order

```text
GARDEN commit=b53115358687f2d660d3bc5596f07a37aa4929fb
SOUNIO_EXECUTABLE commit=fdd444afc5ba0e7529bfee532640dc0a665bfc3f
SEMANTICS_FROZEN=enclosing Git commit
PARITY_OPEN=false
CLAIM_READY=false
```

The Garden fixed the admissible schema before the child existed. The first
Sounio executable emitted target records and digests before this document or
the exact matcher values were frozen.

## Frozen Inputs

The child owns admission semantics. It does not own the material observations.

| Artifact | SHA-256 |
| --- | --- |
| Garden source | `68b2844934cc1e7544794dd5fdb35d56387a58ad2a536075d81a6378feda34fe` |
| lowering source | `7087649a5cfdb41a884aa9a2e1b0b64bbe2d25da3ca3cd1d54d5b70429854edb` |
| lowering semantics | `9e92f256c25a774979a1b02cb10d5d39a1ae702ccec3a273044f91930f03a970` |
| lowering receipt | `daef832ee6370b656e93ae84c76ba6d17c98aaf5ad1dd86674dee27ba0f84346` |
| Darwin receipt | `342d8ba8808c2a926bb2bbf0c09488f7b849967239c932687952ec6ae789a906` |
| Darwin evidence | `ee37914bc738eb829f3589249f228e4a8312310fbffa0b00636cd0c9ed9a40d1` |
| Apple receipt | `c00a3d4e556688829efadbbf640ea858cfe9520dc04103fa745cf1a8101f7840` |
| Apple evidence | `2877bfd463b4d28dc3311b75c69bec2aa1c62b430d08314989187d44b32a781e` |
| DGX receipt | `3c10882eff43d3b197428839996c7a04c009c8f537d0c1451bdf3e8a13e2f385` |
| DGX evidence | `2c6b6e448265a5566d17df9a674246ea62c05210e432e48e418d16358496853b` |

```text
parent_manifest_sha256=23eeef8d222c99674bc3a3f92ea5cb46772fc5d7a58ed74af36469a9f32ef712
```

The manifest is SHA-256 over the ordered `sha256sum` records above. The Garden
is the admission root; the other nine rows are the required lowering and
material parents. The lowering parent is also re-evaluated and matched live in
Sounio. Each material receipt and evidence file is read and SHA-256 checked by
Sounio; prose hashes are not a substitute for that live path.

## Admission Schema

The material receipt input carries target identity, producer language and
role, receipt stage, sealed-result validity, binding error, five explicit and
sufficient node-evidence flags, unnamed-node count, payload and encoding
status, selector-site observations, and reproducibility qualifications.

The emitted record carries:

```text
target_id
binding_error
receipt_admitted
coordinate_exact
payload_exact
output_exact
node_status[5]
admitted_node_count
unresolved_node_count
refused_node_count
unnamed_resolved_node_count
whole_operation_covered
operand_encoding_status
static_selector_sites
secondary_selector_sites
result_reproducible
binary_reproducible
```

`coordinate_exact` requires both explicit and sufficient evidence for node 0.
For every semantic node:

```text
explicit && receipt_admitted && semantic_authorized && sufficient -> ADMITTED
explicit && otherwise                                             -> REFUSED
not explicit                                                      -> UNRESOLVED
```

An unnamed count never enters this transition.

## Canonical Target Records

### Darwin Xeon

```text
target_id=1
binding_error=0
receipt_admitted=true
coordinate_exact=true
payload_exact=false
output_exact=true
node_status=ADMITTED,ADMITTED,ADMITTED,ADMITTED,ADMITTED
admitted=5
unresolved=0
refused=0
unnamed_resolved=0
whole_operation=true
encoding_status=UNCOMPARED
static_selector_sites=3
secondary_selector_sites=0
result_reproducible=true
binary_reproducible=true
digest_words=3500460042:535259591:2326513808:3224911087:1734937489:1992726449:2327971184:3455846257
```

Whole-operation coverage is a property of this sealed receipt only. The false
payload field remains false and is not upgraded by node coverage.

### Apple Silicon

```text
target_id=2
binding_error=0
receipt_admitted=true
coordinate_exact=true
payload_exact=true
output_exact=false
node_status=ADMITTED,UNRESOLVED,UNRESOLVED,UNRESOLVED,UNRESOLVED
admitted=1
unresolved=4
refused=0
unnamed_resolved=0
whole_operation=false
encoding_status=UNCOMPARED
static_selector_sites=1
secondary_selector_sites=0
result_reproducible=true
binary_reproducible=true
digest_words=4128037373:2968517972:275166301:843980919:2897963936:177801273:2520874503:391658731
```

The material parent records tailnet identity `sounio-language-macbook`, host
`Sounio-Language-MacBook`, model `Mac17,7`, and Apple M5 Max. This lane binds
that parent; it does not perform a new Apple measurement.

### DGX

```text
target_id=3
binding_error=0
receipt_admitted=true
coordinate_exact=true
payload_exact=true
output_exact=false
node_status=ADMITTED,UNRESOLVED,UNRESOLVED,UNRESOLVED,UNRESOLVED
admitted=1
unresolved=4
refused=0
unnamed_resolved=1
whole_operation=false
encoding_status=DIFFERENT
static_selector_sites=32
secondary_selector_sites=32
result_reproducible=true
binary_reproducible=false
digest_words=3400682017:537641554:1229578524:2856974856:2840878714:2939595095:1922878528:3271869735
```

The unnamed count is preserved without assigning it to any of nodes 1 through
4. The abstract Sounio operand `c=15` and emitted PTX/SASS operand `c=4127`
remain different encodings despite finite coordinate parity.

## Aggregate Result

```text
material_files=6
material_file_matches=6
receipts=3
admitted_receipts=3
admitted_nodes=7
unresolved_nodes=8
refused_nodes=0
selector_targets=3
whole_operation_targets=1
incomplete_targets=2
unnamed_node_claims=1
encoding_differences=1
material_observations=3
compiler_emission_observations=3
cost_records=0
generic_instruction_cost=false
cross_isa_equivalence=false
transform_authorized=false
review_promoted=false
parity_open=false
claim_ready=false
admission_digest_words=1981472606:3869995793:2634373272:1299673842:1000193586:119066938:1531096938:209245254
```

The fourteen aggregate integers, in digest order, are:

```text
receipts
material_file_matches
admitted_receipts
admitted_nodes
unresolved_nodes
refused_nodes
selector_targets
whole_operation_targets
incomplete_targets
unnamed_node_claims
encoding_differences
material_observations
compiler_emission_observations
cost_records
```

The seven boundary booleans, in digest order, are:

```text
material_files_valid
generic_instruction_cost
cross_isa_equivalence
transform_authorized
review_promoted
parity_open
claim_ready
```

The aggregate digest binds the exact Garden commit, parent binding, three
target digests, six live-file match bits, these fourteen integers and seven
booleans, and every individual negative witness.

## Negative Surface

All twenty-two Sounio mutation witnesses pass. They cover missing Garden and
lowering bindings, receipt/evidence mismatches, semantic-parent mismatch,
wrong role and stage, invalid receipt, missing command binding, coordinate and
payload promotion, unnamed-node inference, selector-to-whole promotion,
static-site-to-cost promotion, DGX encoding equality, cross-ISA promotion,
target receipt transplant, transform and reviewer promotion, premature parity,
and claim readiness.

A separate live-file negative appends bytes to a copy of the Darwin receipt.
Sounio reports five of six file matches, `binding_error=1`, refuses all five
explicit Darwin nodes, returns `ERR_RECEIPT=2`, and exits nonzero. The canonical
files are never modified.

## Digest Boundary

The per-target digest is a positional SHA-256 commitment to every emitted
record field plus the sealed target parent bindings. The aggregate digest
commits to live file match bits, so even a receipt change that has no semantic
record footprint remains a failed admission.

Digest equality does not confer semantic authority. The expected values were
written only after Sounio emitted them, and the exact matcher is part of the
Sounio source.

## Review Boundary

xAI/Grok 4.5 and Z.AI/GLM-5.2 acted only as hostile reviewers. They found
missing live-file checks, incomplete aggregate binding, a literal claim-boundary
call, and template fields substituted into per-record digests. Sounio was
rerun after each fix and emitted the final records above.

No external model executed Sounio or created or confirmed a record, count,
digest, expected value, Loom decision, or authority claim.

## Closed Claims

This freeze admits target-local evidence coverage. It does not establish a
new material observation, target execution, minimum instruction sequence,
cost model, performance result, cross-ISA equivalence, complete Apple or DGX
lowering, operand-encoding equality, numerical reassociation, transform, or
subquadratic algorithm.

Lean, Koka, C++, and Haskell parity for this overlay remain unopened.
`PARITY_OPEN=false` and `CLAIM_READY=false`.
