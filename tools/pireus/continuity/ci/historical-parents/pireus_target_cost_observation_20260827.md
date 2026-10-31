<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-target-cost-observation-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-target-cost-observation-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus Target Cost Observation Receipt

Receipt-Schema: `sounio-semantic-authority-receipt.v1`

Date: `2026-08-28`

Concept-ID: `SOUNIO-PIREUS-TARGET-COST-OBSERVATION`

Semantic-Lane-ID: `pireus-target-cost-observation-20260827`

Producer-Language: `Sounio`

Producer-Role: `SEMANTIC_AUTHORITY`

Stage: `SEMANTICS_FROZEN`

Parity-Open: `false`

Claim-Ready: `false`

## Result

This receipt records the first typed Pireus cost vocabulary and value-free
measurement-request ledger. It records no benchmark or material cost value.

```text
parent_files=11
parent_file_matches=11
admission_parent_live=true
operation_parent_live=true
engine_parent_live=true
requests=7
selector_requests=6
whole_operation_requests=1
ontology_triples=413
negative_witnesses=26/26
material_observations=0
derived_summaries=0
comparison_receipts=0
speedups=0
rankings=0
cross_target_comparable=false
parity_open=false
claim_ready=false
```

The Sounio source, exact matcher, authority stream, paired semantics, and flat
evidence file are the audit surfaces. This page records their outcome; it is
not an independent semantic producer.

## Causal Commits

| Phase | Commit | Meaning |
| --- | --- | --- |
| Garden | `de9e1b4d1cb818a0cc1af7cf710e7f31a347211d` | fixed the axes without any expected child result |
| Sounio executable | `ad38229c7c7d21318e6a445fe5c078b5c72f49be` | emitted the ledger and digest before exact matcher or frozen prose |
| semantics freeze | enclosing Git commit | binds matcher, semantics, receipt, evidence, registry, and gate |

The Garden is an ancestor of the first executable commit.

## Sounio Sources

| Artifact | SHA-256 |
| --- | --- |
| Garden | `ecef7f1ff683f1157e89bb5e552e121f51c4191e9662b17a0a5f4a71909223a8` |
| module | `7ea2815c112b85476fc6ac4d8bb9388ee032062822c6905485c2084ee416d6bc` |
| authority executable | `ee237c9419c494879a97ecaaf285ac3d70b0f6ce6055101152a13a39f907c51d` |
| dedicated test | `06518d139e8a70c7734a731019d78f71c6d37dfca89001490c1834ddf62eb73e` |
| source manifest | `cdfa7c1438aa524e884bf3f4a69e19e8ed99582e0f2f2effa16451c582fcb596` |
| frozen semantics | `0a899be7cd25375c8c444b9e1f0a71dd102ca8958072a4290073ae21c926a199` |

The ordered source manifest contains module, executable, and test. The Loom
source field uses the module hash because the module owns the schema, ledger,
negative predicates, digest, and exact matcher.

## Parent Closure

The ordered manifest contains the Cost Garden and ten frozen admission,
execution-engine, and XOR-operation artifacts:

```text
parent_file_count=11
parent_manifest_sha256=9a4f1f28651b8984a0d719ecc4415572b6b301c0ccbee7520d960afeea6bf605
```

Every file is read and hashed by Sounio. The admission and operation parents
are also evaluated against their exact Sounio matchers, while the engine parent
is queried live. The paired semantics lists all eleven hashes in order.

## Engine And Request Coordinates

```text
observed_cpu_engines=5
observed_gpu_engines=4
darwin_observed_engines=9
apple_observed_engines=0
dgx_observed_engines=0
darwin_blueprints=2
apple_blueprints=2
dgx_blueprints=2
```

Darwin requests bind machine `701313` and observed T560 Xeon engine `703303`.
Apple requests bind CPU blueprint `703402`; DGX requests bind GPU blueprint
`703405`. Apple and DGX therefore may receive requests but not material values.

Known transport locators are:

```text
apple_tailnet=demetrios@sounio-language-macbook
dgx_locator_1=demetrios@192.168.3.24
dgx_locator_2=demetrios@192.168.3.48
```

These strings are routing hints only. They are not hardware receipts, observed
engine identities, cost observations, or proof that a key is currently usable.

## Frozen Ledger

| ID | Target | Engine | Subject | Quantity | Unit | Scope |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Darwin Xeon | observed T560 CPU | selector | dependency latency | cycle | dependency chain |
| 2 | Darwin Xeon | observed T560 CPU | selector | reciprocal throughput | cycle | independent stream |
| 3 | Darwin Xeon | observed T560 CPU | complete operation | elapsed duration | nanosecond | complete operation |
| 4 | Apple Silicon | CPU blueprint | selector | dependency latency | cycle | dependency chain |
| 5 | Apple Silicon | CPU blueprint | selector | reciprocal throughput | cycle | independent stream |
| 6 | DGX | GPU blueprint | selector | dependency latency | cycle | dependency chain |
| 7 | DGX | GPU blueprint | selector | reciprocal throughput | cycle | independent stream |

Every request carries:

```text
statistic=MEDIAN
requested_samples=1001
requested_warmups=128
required_environment_mask=2047
bound_environment_mask=0
evidence_role=MEASUREMENT_REQUEST
producer_language=Sounio
producer_role=SEMANTIC_AUTHORITY
value_present=false
```

Sample and warmup counts are requested protocol parameters. No sample was
taken. Reciprocal throughput is cycles per operation; operations per second is
ordinary throughput and is not an admitted reciprocal-throughput unit.

## Toolchain And Hardware

```text
engine=lean_single
wrapper_path=bin/souc
wrapper_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
compiler_path=bin/souc-lean-single-x86_64
compiler_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
toolchain_record_sha256=2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e

hostname=sounio-workspace-control-0
os=Linux 7.0.2-5-pve
architecture=x86_64
cpu_model=INTEL(R) XEON(R) GOLD 6526Y
logical_cpus=64
sockets=2
cores_per_socket=16
threads_per_core=2
hardware_record_sha256=b6326139297e9ba59d82e208a2404b01e3d57445357a5e803c51c845dd388db0
```

This Xeon host transported the semantic authority executable. It did not
perform any requested cost experiment.

## Command And Stream

The exact authority command record is the Sounio `lean_single` invocation of
`examples/pireus_target_cost_observation.sio` followed by the pinned Intel PDF,
XED corpus, six material receipt/evidence inputs, and eleven parent paths,
with output tee'd to `/tmp/pireus-target-cost-observation.authority.txt`.

```text
command_sha256=9e5c49dfefa3278f32d5c2381b49a329f90e59181c6c178a8fa47a38273af885
result_lines=233
result_bytes=3510
result_sha256=99f2e7f0dff71d76c55b5f39f5f514e82128c97a81e31d3a1263a29d2d816d9b
test_command_sha256=497e6a868ee9437034bf4ff72916f4831bce0a3a08108a67267b11fb164860e2
test_result_sha256=04cf5a9aa26fe3405e8c10249f930ef8ad33b040ba49650e3bef4fdbcd6382ab
error=0
failures=0
```

Three authority executions were byte-identical.

## Digest And Rational Boundary

```text
digest_words=166882395:243190428:1767805310:4234095682:63968340:1374547816:344010038:3839699970
digest_sha256_hex=09f26c5b0e7eca9c695e917efc5f2c4203d0145451edef6814812d36e4dd2c02
rational_component_min=0
rational_component_max=1000000000
maximum_cross_product=1000000000000000000
```

The rational limit prevents `i64` overflow in interval comparisons. This
request-only receipt carries no uncertainty value; a later observation must
introduce its interval and evidence explicitly.

## Loom Decisions

```text
freeze_frame_sha256=e22440429d0beec5290d6bca95b4e70d1b6c07c571e87cf9247e501833e88913
freeze_decision_sha256=ad57cad1376867f5cec01323e50d39319c7fb8ba0458a6066086d186a045b8cb
freeze_decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
authority_seal_frame_sha256=93e0867f20a3d6e0a1510f26b6b076a14726c1892c9ba8320c6c600126655306
test_seal_frame_sha256=06a0223480c5ae97d9c7a514ee36a74ef0b0556bb13975770d7fdd57f8568c4c
```

The first executable commit was preauthorized with Loom action `10=COMMIT`:

```text
executable_commit_frame_sha256=c76a4984aa4e803d09347ffbf59a76be45b46eecfa67ae6333493b6e7ffdf90c
```

## Negative Enforcement

All twenty-six in-process Sounio mutations pass. A live test appends a marker
to a copy of parent 10. Sounio reports ten matches, `error=1`, `failures=3`,
and exits 1:

```text
tampered_parent_index=10
tampered_parent_sha256=b6158c1568a55e642e933032698311e73bdf373f853f5471f0338c15327a6270
tampered_command_sha256=8186202c90e3fb344ddd4b911b1ead49ecd9fb054f1d783885d93acdea7f1809
tampered_frame_sha256=bfb0e78a56c730aa28eb9eaf930ebc72b5d962aec4fbcdb7ddd7abe90ab1ad53
tampered_result_sha256=924c5b542f55a8e5bf29e01aa6abe9029a971a9bf07728d67aff47fb8c2a9345
tampered_parent_match_count=10
tampered_error=1
tampered_failures=3
tampered_exit_code=1
```

The native Loom guardian separately refuses Python before interpreter launch:

```text
python_frame_sha256=e4261c050af8f486685bf6dd6da869a9d4c9fefedf67b4a0e5ad090c4a4a4eef
python_decision_sha256=3e2b1112dc7ce41d6c752c48daca33e6ee400b93df1e3fafa795a5709b4aa2a3
python_exit_code=110
python_decision=SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN
interpreter_launch_count=0
```

Rust was not used.

## Review-Only Offload

xAI/Grok 4.5 acted only as an orthogonal reviewer. It found and caused repairs
to partial-ledger validation, rational overflow, vacuous negative witnesses,
and the dimensional distinction between throughput and reciprocal throughput.

```text
role=REVIEW_ONLY
raw_first=/tmp/llm-offload-J6vnXn/
raw_after_fixes=/tmp/llm-offload-8Dzg44/
raw_matcher=/tmp/llm-offload-Uq3xH3/
raw_semantics=/tmp/llm-offload-xBtvvr/
```

No model executed Sounio or created or confirmed a request, count, digest,
value, comparison, Loom decision, or authority result.

## Closed Claims

No material observation, benchmark, Apple/DGX observed engine, cross-target
comparison, speedup, ranking, instruction minimum, lowering choice, parity
implementation, or transform result is established. `PARITY_OPEN=false` and
`CLAIM_READY=false`.
