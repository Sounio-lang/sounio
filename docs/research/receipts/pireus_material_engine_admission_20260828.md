<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-material-engine-admission-20260828
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-material-engine-admission-20260828
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus Material Engine Admission Receipt

Receipt-Schema: `sounio-semantic-authority-receipt.v1`

Date: `2026-08-28`

Concept-ID: `SOUNIO-PIREUS-MATERIAL-ENGINE-ADMISSION`

Semantic-Lane-ID: `pireus-material-engine-admission-20260828`

Producer-Language: `Sounio`

Producer-Role: `SEMANTIC_AUTHORITY`

Stage: `SEMANTICS_FROZEN`

Parity-Open: `false`

Claim-Ready: `false`

## Result

This receipt records a Sounio-owned additive ontology overlay. It admits only
the CPU engine witnessed by the sealed Apple material receipt and the GPU
engine witnessed by the sealed DGX `.24` receipt. It creates a value-free
identity request for `.48`; it records no new remote execution or cost value.

```text
parent_files=12
parent_file_matches=12
base_cluster_machines=5
base_cpu_engines=5
base_gpu_engines=4
base_darwin_engines=9
overlay_cluster_machines=7
overlay_cpu_engines=6
overlay_gpu_engines=5
material_admitted_machines=2
material_admitted_engines=2
material_receipts=2
unresolved_identity_requests=1
identity_requests=1
dgx_48_resolutions=0
apple_gpu_engines=0
dgx_cpu_engines=0
request_eligible_engines=2
request_eligibility_rows=4
material_cost_values=0
whole_operation_coverage=0
negative_witnesses=22/22
comparisons=0
speedups=0
rankings=0
cross_target_comparable=false
parity_open=false
claim_ready=false
```

The module, exact matcher, paired frozen semantics, authority stream, and
dedicated test are the semantic audit surfaces. This receipt records their
custody; it is not an independent result producer.

## Causal Commits

| Phase | Commit | Meaning |
| --- | --- | --- |
| Garden | `61e8fa34ed93a083cccabe6e6813f1ce6460e327` | fixed the identity question without expected child values |
| first Sounio executable | `d0a16b5303a5e455be1b8a1ce90f40d3f89b3ed2` | emitted records and digests before the exact matcher |
| semantics freeze | enclosing Git commit | binds matcher, semantics, receipt, evidence, registry, and gate |

The Garden is an ancestor of the first executable. The first executable commit
does not contain `pireus_material_engine_admission_matches_frozen_semantics`;
the enclosing freeze adds it only after the first Sounio stream existed.
`enclosing Git commit` is intentionally self-locating because embedding that
commit's own hash would be circular. The gate verifies both prior commits and
their ancestry, and Git binds this receipt inside the enclosing freeze commit.

## Frozen Sources

| Artifact | SHA-256 |
| --- | --- |
| Garden | `08d6bd3193db9a2ce0cd849db4b1197d389049322be3be2986bd9f955257b21e` |
| Sounio module with exact matcher | `b98d799fec6452f9afaecc7e418578cc0ca72a1d3e7c0a1f84d474e0aa2730ec` |
| authority executable | `075ae3225bd01ac858586186cb43bc404ed5315f9d55a2f1b5725299f46d88a4` |
| dedicated test | `375ea71aad9dd695175fa0fa16e9e9b656a53b4df137fb208ec7afe34eb3a7b2` |
| ordered source manifest | `12a648e39618d3b02302d86da7f7203362f88dbc24f0fb0c352a4c9b5e6adcce` |
| frozen semantics | `bbbd7bc9c99e1de46a5317d99a7893adebfdf7fb46454cbd7002b3098281b6ee` |

The source manifest order is module, authority executable, then dedicated
test. The Loom source field is the module hash because the module owns the
schema, admission predicate, ontology readback, negative surface, canonical
serialization, result digest, and exact matcher.

## Parent Closure

The ordered closure is the Garden followed by the execution-engine source,
semantics and receipt; the target-cost source, semantics, receipt and evidence;
then the Apple and DGX receipt/evidence pairs.

```text
parent_file_count=12
parent_manifest_sha256=f939a44278411e0954fde1425e3a728afe7069cbe1805e0ce521f26854b097a9
parent_match_mask=4095
```

Every file is read and SHA-256 checked by Sounio. The execution-engine source
is also evaluated and queried as a hash-frozen imported snapshot. The target
cost parent is bound by its four frozen hashes; its heavier vendor-dependent
evaluation is not re-executed in this overlay.

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
engine_kind=CPU
engine_name=Apple M5 Max
isa=AArch64
interface=absent
material_receipt_id=707303
material_evidence_id=707304
hardware_identity_id=707305
hardware_sha256=49702cf6d0b079bf52bf26f98f377266e41d4ce232fea99eb80c30d6554dbc28
material_subject=A64 TBL XOR selector
```

The login locator is a routing coordinate, not hardware identity. No Apple GPU
or Metal engine is admitted. `Metal` remains an interface term and is not the
ISA of this CPU receipt.

### DGX `.24` GPU

```text
canonical_target=701202
login_locator=demetrios@192.168.3.24
hostname=spark-3c59
locator_id=707310
machine_id=707311
engine_id=707312
engine_kind=GPU
engine_name=NVIDIA GB10
isa=NVIDIA SM121
interface=CUDA
material_receipt_id=707313
material_evidence_id=707314
hardware_identity_id=707315
hardware_sha256=8b048f0a20ac0967af5622606935aa4ea4e6caf0baef6a3dcd9b7ff58f2a66d4
material_subject=PTX SHFL.BFLY / SASS SHFL.BFLY XOR selector
```

No DGX CPU engine is admitted. `CUDA` remains an execution interface, not an
ISA.

### DGX `.48`

```text
login_locator=demetrios@192.168.3.48
locator_id=707320
identity_request_id=707321
status=UNRESOLVED
machine_id=0
engine_id=0
locator_resolution_count=0
```

A route failure is not promoted to absence, identity, or equivalence with
`.24`. No sibling machine or engine is inferred.

## Request Eligibility Without Cost

The admitted Apple CPU and DGX `.24` GPU each receive two value-free request
eligibility rows: dependency latency and reciprocal throughput.

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

Eligibility does not satisfy a request, bind a measurement environment, or
create a lowering preference.

## Toolchain And Authority Host

```text
engine=lean_single
wrapper_path=bin/souc
wrapper_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
compiler_path=bin/souc-lean-single-x86_64
compiler_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
toolchain_manifest_sha256=850c094e02d85fee153297ccf8babbe171e3ec47def68ac2976c3473092b36ac

hostname=sounio-workspace-control-0
os=Ubuntu 24.04.4 LTS / Linux 7.0.2-5-pve
architecture=x86_64
cpu_model=INTEL(R) XEON(R) GOLD 6526Y
logical_cpus=64
sockets=2
cores_per_socket=16
threads_per_core=2
hardware_record_sha256=464f1a4530cb0829854ddbafc0786d12cc9fc98cef1afced51f40679ba27517c
```

The hardware hash is the Loom custody identifier used by this freeze. The gate
does not treat that token alone as material evidence: it independently checks
the hostname, OS, kernel, architecture, CPU model, logical CPU count, sockets,
cores per socket, and threads per core against the live authority host.

The authority host transported the Sounio executable. It performed no Apple,
DGX, cost, or whole-operation experiment.

## Commands And Streams

The authority and test commands use explicit `SOUNIO_SOUC_ENGINE=lean_single`
and pass the twelve parent paths in their frozen order.

```text
authority_command_sha256=c2aae8f3b58d3caed7d9f50277f6757e4eea7cffa0e0f714ea96330507fe3051
authority_preexec_frame_sha256=0e868eacdaffef497fe1b9fc5359b60b34b61b69383aeaf91737593e9676ce28
authority_reproducible_runs=3
authority_result_lines=131
authority_result_bytes=2258
authority_result_sha256=cdaa653a8ba745aacaf6bb8fae8ac3b34fb16c42ecf6e7247eb7949dce71cec1
test_command_sha256=d61958b542ad92be7939fc94ee3e43af2a7af9dd6674c6578609bc8d68455206
test_preexec_frame_sha256=5666b5491f13baad06d7805775ac55dc13b7d46d5f3610a1fd7901ed48fba717
test_result_sha256=bbd6018aefba1e0d1bafd48d52d253a41a05dd7ae306788aa4b42fc78175f6f7
test_output=pireus material engine admission test passed
error=0
failures=0
```

The dedicated gate runs the authority executable three times, requires every
stream to have the listed result hash, and compares runs 1, 2 and 3 byte for
byte.

## Ontology And Result Digests

```text
ontology_triples=366
ontology_identity_binding_valid=true
ontology_digest=2058720326:8114423:2692954918:39064:264032718:2437098100:4195323016:150194624
result_digest=1987781134:3449544734:2538874717:3559779810:1021512403:1875438742:1937396250:2579435395
negative_passed=22
negative_total=22
```

The ontology digest includes every append-ordered triple field, including
literal values through `f64_to_bits`. The result digest includes parent mask,
candidates, unresolved request, eligibility rows, ontology digest, 29 counters,
13 boundary booleans, all 22 negative booleans, and the closing negative passed
and total pair. Canonical v1 serializes `negatives.passed` twice: once as
counter 29 and once in that closing pair. This redundant encoding is not an
independent agreement proof and adds no second failure detector. The evaluator
rejects `passed != total`; the matcher checks the aggregate, individual flags,
counts, and digest; and the gate reexecutes all 22 in-process mutations. No
independence or completeness theorem is claimed.

The 29-counter order is: parent matches; base cluster, CPU, GPU, Darwin, Apple,
and DGX counts; overlay cluster, CPU, GPU, Apple, and DGX counts; admitted
machines, admitted engines, receipts, unresolved identity requests, identity
requests, `.48` resolutions, Apple GPUs, DGX CPUs, eligible engines,
eligibility rows, ontology triples, cost values, whole-operation coverage,
comparisons, speedups, rankings, and passed negatives.

The 13-boundary order is: all parents valid; execution-engine parent live;
cost-parent hash bound; Apple admitted; DGX admitted; `.48` unresolved;
ontology identity binding valid; Apple GPU count zero; DGX CPU count zero;
cost and whole-operation counts zero; comparison, speedup, and ranking counts
zero; cross-target comparison, parity-open, and claim-ready all false; and all
negative cases passed.

## Loom Decisions

```text
freeze_frame_sha256=d8afe57e5c61b646173b17957fcf9785cfce27d8087606c37e6b49166d66be45
freeze_decision_sha256=ad57cad1376867f5cec01323e50d39319c7fb8ba0458a6066086d186a045b8cb
freeze_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
authority_seal_frame_sha256=96394b20f8b46b165992c6f306ed89bd9f55fd0cd5a9b4bf81f770536b86da78
test_seal_frame_sha256=03121a38f9385c55a6158b8f4fdb63be5f291ddb54967d9ba14cf57020700e9f
write_frame_sha256=2b5b4d7c1bbf4e9f8773773703f5e3134267f431b1152ece5b4f7da55b75136d
commit_frame_sha256=b77090f12290899c58137d88f57632a91fa2f71ce0bbbf33a19710fd8c3787ae
```

Every listed frame was admitted by the native Sounio Loom language-authority
guardian. No second guardian or policy kernel was introduced.

An earlier pre-commit freeze bound semantics hash
`c41fb8aacc659548d6087d93ad7fddef065dc8820cefd825b6139e3b56a16a47`.
The repository governance synchronizer then normalized the research document's
metadata and lineage status, producing the current hash before any commit,
parity run, or claim promotion. Enforcement failed closed: the earlier freeze
and its dependent seals are superseded, the normalized file was frozen again,
and a second governance sync was required to leave its bytes unchanged.

```text
superseded_freeze_frame_sha256=d15f56f2d36682bc719adfebe02175d95152831c6327ae972d37ea541e731e27
superseded_semantics_sha256=c41fb8aacc659548d6087d93ad7fddef065dc8820cefd825b6139e3b56a16a47
active_semantics_sha256=bbbd7bc9c99e1de46a5317d99a7893adebfdf7fb46454cbd7002b3098281b6ee
governance_resync_byte_identical=true
```

## Negative Enforcement

All 22 in-process Sounio mutations pass. A live test appends a marker to a copy
of parent 11, the DGX material evidence. Sounio observes 11/12 parent matches,
sets `error=1`, reports three failures, and exits 1.

The three failures are not three parent mismatches: the evaluator contributes
one failure for the typed parent error, the executable contributes one because
`valid=false`/`error!=OK`, and the exact frozen matcher contributes one because
the parent mask and result no longer match the frozen semantics.

```text
tampered_parent_index=11
tampered_parent_sha256=67a8641d0793f092364ed5bcb37724c087178ad80bcf19ceb82a16ce50935b63
tampered_command_sha256=dce63dffa83f4ee3505a33c7a93f6b3b0a3490af5c1ef3fd04d0e5c1d650f1c1
tampered_frame_sha256=8e59dc47c3b5d9d8b08610efa1fe9c51d5f2f92f6489209ebb100cba312eff8b
tampered_result_sha256=421c60238e4bdab6487d9849939cfcd0195522f935054bbbf5370bfaa6334448
tampered_parent_match_count=11
tampered_error=1
tampered_failures=3
tampered_exit_code=1
```

The native guardian separately refuses Python before interpreter launch:

```text
python_frame_sha256=a708c9004b3329e6a80274dc701509bc50cac964b3439302c74392f4d85039af
python_decision_sha256=3e2b1112dc7ce41d6c752c48daca33e6ee400b93df1e3fafa795a5709b4aa2a3
python_exit_code=110
python_decision=SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN
interpreter_launch_count=0
```

Rust was not used.

## Review-Only Offload

xAI/Grok 4.5 and Z.AI/GLM 5.2 acted only as orthogonal reviewers. They found
and caused repairs to receipt cross-binding negatives, relation readback,
eligibility multiplicity, digest completeness, unresolved-request naming,
external-LLM producer exclusion, commit custody, and relation quantification.

```text
role=REVIEW_ONLY
raw_garden=/tmp/llm-offload-MJ3MrC/
raw_source_first=/tmp/llm-offload-Mo1DtE/
raw_source_followup=/tmp/llm-offload-SsiCRY/
raw_matcher_first=/tmp/llm-offload-mjfJvQ/
raw_matcher_followup=/tmp/llm-offload-0NiYfq/
raw_focused=/tmp/llm-offload-RxhfKh/
raw_relation_postfix=/tmp/llm-offload-Wf8dq4/
```

`/tmp/llm-offload-uz2VIp/` is explicitly invalid and not counted because an
incorrect full commit ID yielded an empty diff. No provider executed Sounio or
created, derived, or confirmed a result, identity, count, digest, expected
value, measurement, Loom decision, parity verdict, or authority claim.

## Compiler Route

The frozen executable uses explicit `lean_single`. The default Madaros route is
advisory only for this import graph and cannot create a second result. Its live
check returned exit 1 with `science-boundary: mode=advisory verdict=UNKNOWN` and
`run_check_mode: AST closure incomplete nodes=7`, with `unresolved=0` and
`saturated=false`. No fallback path was used.

## Closed Claims

No Apple GPU, Metal engine, DGX CPU, `.48` machine, `.48` engine, cost value,
whole-operation coverage, comparison, speedup, ranking, cross-target
equivalence, lowering preference, parity implementation, or claim-ready result
is established. `PARITY_OPEN=false` and `CLAIM_READY=false`.
