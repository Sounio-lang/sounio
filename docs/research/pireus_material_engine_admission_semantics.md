<!-- docs:meta
topic_id: repo.docs.research.pireus-material-engine-admission-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-material-engine-admission-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus Material Engine Admission Semantics

Concept-ID: `SOUNIO-PIREUS-MATERIAL-ENGINE-ADMISSION`

Semantic-Lane-ID: `pireus-material-engine-admission-20260828`

Status: `SEMANTICS_FROZEN`

Semantic authority: `Sounio`

Parity open: `false`

Claim ready: `false`

## Causal Order

| Phase | Commit | Boundary |
| --- | --- | --- |
| Garden | `61e8fa34ed93a083cccabe6e6813f1ce6460e327` | fixed the question and forbidden expected results |
| first Sounio executable | `d0a16b5303a5e455be1b8a1ce90f40d3f89b3ed2` | emitted records and digests before this frozen prose and exact matcher |
| semantics freeze | enclosing Git commit | binds matcher, this semantics, receipt, evidence, registry, and gate |

The Garden is an ancestor of the first executable commit. No new remote target
execution occurred in this lane. The Apple and DGX receipts are immutable
historical parents.

The first executable commit does not contain
`pireus_material_engine_admission_matches_frozen_semantics`. That exact matcher
is a later freeze-candidate addition in the enclosing commit, written only
after the first Sounio stream existed.

## Frozen Sources

| Artifact | SHA-256 |
| --- | --- |
| Garden | `08d6bd3193db9a2ce0cd849db4b1197d389049322be3be2986bd9f955257b21e` |
| Sounio module with exact matcher | `b98d799fec6452f9afaecc7e418578cc0ca72a1d3e7c0a1f84d474e0aa2730ec` |
| authority executable | `075ae3225bd01ac858586186cb43bc404ed5315f9d55a2f1b5725299f46d88a4` |
| dedicated test | `375ea71aad9dd695175fa0fa16e9e9b656a53b4df137fb208ec7afe34eb3a7b2` |
| ordered source manifest | `12a648e39618d3b02302d86da7f7203362f88dbc24f0fb0c352a4c9b5e6adcce` |

The source manifest order is module, authority executable, then dedicated test.

## Frozen Parent Closure

The Sounio executable reads and hashes these files in order:

| Index | Artifact | SHA-256 |
| ---: | --- | --- |
| 0 | Garden | `08d6bd3193db9a2ce0cd849db4b1197d389049322be3be2986bd9f955257b21e` |
| 1 | execution-engine source | `8b5063f0e9a39650fb0b60e8b70b315f339723690e06050c2bebacece888e37e` |
| 2 | execution-engine semantics | `c47668a08ad25f39bebe9d8bef90b66eb2ad7119063c19ab8319fa4fab265233` |
| 3 | execution-engine receipt | `9da8ca53c3cb0e6631c92e55a8e82387aed2bd53863ffa9d646719806eec4ffd` |
| 4 | target-cost source | `7ea2815c112b85476fc6ac4d8bb9388ee032062822c6905485c2084ee416d6bc` |
| 5 | target-cost semantics | `0a899be7cd25375c8c444b9e1f0a71dd102ca8958072a4290073ae21c926a199` |
| 6 | target-cost receipt | `b7577c782a82431eb54312137a52c1811f14316d5e5a5fb9e7aa9581f1c304ca` |
| 7 | target-cost evidence | `06f21108ddc89c8c468097b7cefec6a766bda065918fd5797be194878371577b` |
| 8 | Apple material receipt | `c00a3d4e556688829efadbbf640ea858cfe9520dc04103fa745cf1a8101f7840` |
| 9 | Apple material evidence | `2877bfd463b4d28dc3311b75c69bec2aa1c62b430d08314989187d44b32a781e` |
| 10 | DGX material receipt | `3c10882eff43d3b197428839996c7a04c009c8f537d0c1451bdf3e8a13e2f385` |
| 11 | DGX material evidence | `2c6b6e448265a5566d17df9a674246ea62c05210e432e48e418d16358496853b` |

```text
parent_file_count=12
parent_manifest_sha256=f939a44278411e0954fde1425e3a728afe7069cbe1805e0ce521f26854b097a9
parent_match_mask=4095
```

The execution-engine source is also evaluated and queried. Its returned
cardinalities are a live match against the hash-frozen imported snapshot; they
are not semantic facts created by this child. The target-cost parent is bound
by its four frozen file hashes. Its heavier vendor-dependent evaluator is not
re-executed inside this overlay.

The ordered parent-manifest hash is a custody field recomputed by the dedicated
gate. It is not a value derived by the material-admission matcher.

## Exact Material Coordinates

### Apple CPU

```text
canonical_target=701201
login_locator=demetrios@sounio-language-macbook
tailnet_identity=sounio-language-macbook
hostname=Sounio-Language-MacBook
locator_id=707300
machine_id=707301
machine_model=Mac17,7
engine_id=707302
engine_kind=703210 CPU
engine_name=Apple M5 Max
isa=703221 AArch64
interface=0
material_receipt_id=707303
material_evidence_id=707304
hardware_identity_id=707305
hardware_sha256=49702cf6d0b079bf52bf26f98f377266e41d4ce232fea99eb80c30d6554dbc28
material_subject=A64 TBL XOR selector
```

No Apple GPU or Metal engine is admitted. `Metal` remains an interface term,
not the ISA of this CPU receipt.

### DGX `.24` GPU

```text
canonical_target=701202
login_locator=demetrios@192.168.3.24
hostname=spark-3c59
locator_id=707310
machine_id=707311
engine_id=707312
engine_kind=703211 GPU
engine_name=NVIDIA GB10
isa=703225 NVIDIA SM121
interface=703230 CUDA
material_receipt_id=707313
material_evidence_id=707314
hardware_identity_id=707315
hardware_sha256=8b048f0a20ac0967af5622606935aa4ea4e6caf0baef6a3dcd9b7ff58f2a66d4
material_subject=PTX SHFL.BFLY / SASS SHFL.BFLY XOR selector
```

No DGX CPU engine is admitted. `CUDA` remains an execution interface, not an
ISA.

### DGX `.48` Request

```text
login_locator=demetrios@192.168.3.48
locator_id=707320
identity_request_id=707321
status=UNRESOLVED
machine_id=0
engine_id=0
locator_resolution_count=0
```

The request has no machine or engine identity. A route failure and a shared
canonical target cannot fill either zero.

## Imported Snapshot And Overlay

The first stream produced:

```text
base_cluster_machines=5
base_cpu_engines=5
base_gpu_engines=4
base_darwin_engines=9
base_apple_engines=0
base_dgx_engines=0

overlay_cluster_machines=7
overlay_cpu_engines=6
overlay_gpu_engines=5
overlay_apple_engines=1
overlay_dgx_engines=1
material_admitted_machines=2
material_admitted_engines=2
material_receipts=2
unresolved_identity_requests=1
apple_gpu_engines=0
dgx_cpu_engines=0
```

The evaluator checks the additive equations `7=5+2` cluster machines,
`6=5+1` CPU engines, `5=4+1` GPU engines, `2=1+1` admitted target engines,
and `4=2*2` request-eligibility rows. No base machine or engine is rewritten.

## Ontology Readback And Serialization

Every admission-defining single-valued `(subject, predicate)` pair in the fixed
v1 checklist is read directly from `TripleStore.triples`. A present pair must
contain exactly one non-literal expected IRI and no other object for that same
pair; an expected-absent pair must contain zero edges. The checklist covers locator target and
resolution, machine target and engine, engine kind, ISA, interface, material
receipt, covered subject and evidence role, receipt evidence, hardware and
role, eligibility quantity, and the `.48` request's locator, status and role.
The newly minted `7073xx` resource IDs are pairwise distinct and absent from
the hash-frozen base snapshot. For multi-valued cluster and engine-eligibility
pairs, direct iteration checks each newly named member, the hash-frozen base
constructor fixes the prior members, evaluator counters lock the total
cardinality, and the ordered store digest locks the complete sequence. The
Apple CPU interface and `.48` locator resolution are expected-absent. The DGX
GPU has exactly the CUDA interface edge.

`TripleStore` is append-only. The hash-frozen base constructor plus the fixed
overlay call order define canonical v1 order. The ontology digest absorbs:

```text
tag
triple_count
for each triple in insertion order:
  subject
  predicate
  object_iri
  object_literal
  f64_to_bits(object_value)
  is_object_literal
```

```text
ontology_triples=366
ontology_identity_binding_valid=true
ontology_digest=2058720326:8114423:2692954918:39064:264032718:2437098100:4195323016:150194624
```

The digest commits sequence and fields, not only triple count.

## Request Eligibility Without Cost

Each admitted engine has a dependency-latency and reciprocal-throughput
request-eligibility row:

```text
request_eligible_engines=2
request_eligibility_rows=4
material_cost_values=0
whole_operation_coverage=0
comparisons=0
speedups=0
rankings=0
cross_target_comparable=false
```

Eligibility does not satisfy a request, bind an environment, or produce a
measurement.

## Negative Surface

The fixed order is:

1. missing Garden binding;
2. missing or mismatched control parent;
3. material receipt/evidence/hardware drift bundle;
4. target transplant;
5. locator promoted to machine;
6. blueprint without execution;
7. Apple CPU receipt promoted to GPU;
8. Metal promoted to ISA;
9. Apple receipt transplanted to DGX;
10. DGX `.24` receipt transplanted to `.48`;
11. hypothetical same-target sibling identity;
12. DGX GPU receipt promoted to CPU;
13. CUDA promoted to ISA;
14. DGX receipt transplanted to Apple;
15. selector promoted to whole operation;
16. material parity promoted to semantic authority;
17. static evidence promoted to execution;
18. engine admission promoted to a cost value;
19. engine admission promoted to comparability;
20. external review promoted to authority or external LLM launched as a
    material producer;
21. premature parity or claim readiness;
22. Python and Rust producer pre-launch bundle.

The two named bundles each require all of their component mutations to be
rejected. They are not claimed to be independent atomic faults.

```text
negative_passed=22
negative_total=22
```

## Result Digest

The canonical result order is:

```text
v1 tag
parent mask
Apple candidate fields
DGX .24 candidate fields
DGX .48 unresolved tuple
four ordered request-eligibility pairs
eight ontology-digest limbs
twenty-nine named integer counters
thirteen named boundary booleans
twenty-two negative booleans
negative passed and total
```

All integers use signed `i64` values serialized as eight big-endian bytes.
Booleans serialize as `0` or `1` through the same integer function.

```text
result_digest=1987781134:3449544734:2538874717:3559779810:1021512403:1875438742:1937396250:2579435395
authority_result_lines=131
authority_result_bytes=2258
authority_result_sha256=cdaa653a8ba745aacaf6bb8fae8ac3b34fb16c42ecf6e7247eb7949dce71cec1
test_result_sha256=bbd6018aefba1e0d1bafd48d52d253a41a05dd7ae306788aa4b42fc78175f6f7
```

## Authority And Effects

The executable uses `IO`, `Mut`, `Panic`, `Div`, `Alloc`, and `Epistemic`.
`lean_single` is explicit in the frozen command. The default Madaros frontend
is an advisory path only until it accepts this import graph.

Sounio owns the candidates, binding predicate, ontology readback, request
eligibility, negative mutations, canonical serialization, result, and exact
matcher. The historical C++ receipts remain `MATERIAL_PARITY`. External LLMs
reviewed source logic only and created or confirmed no result.

The causal commits, file hashes, stream line and byte counts, and authority
output SHA in this document are custody records checked by the dedicated gate;
they are not mathematical consequences of the exact matcher.

The child remains:

```text
PARITY_OPEN=false
CLAIM_READY=false
```

No Lean, Koka, C++, Haskell, benchmark, remote execution, or parity run was
opened for this child.
