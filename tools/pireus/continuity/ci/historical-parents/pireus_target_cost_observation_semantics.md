<!-- docs:meta
topic_id: repo.docs.research.pireus-target-cost-observation-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-target-cost-observation-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Frozen Semantics: Pireus Target Cost Observation

> **Status**: Semantics frozen | **Date**: 2026-08-28
>
> **Producer**: Sounio | **Role**: `SEMANTIC_AUTHORITY`

## Causal Order

```text
GARDEN commit=de9e1b4d1cb818a0cc1af7cf710e7f31a347211d
SOUNIO_EXECUTABLE commit=ad38229c7c7d21318e6a445fe5c078b5c72f49be
SEMANTICS_FROZEN=enclosing Git commit
PARITY_OPEN=false
CLAIM_READY=false
```

The Garden defined the axes and prohibited expected child values. The first
Sounio stream then emitted the ledger, ontology cardinalities, negatives, and
digest. Only afterward were the exact matcher and this document written.

## Frozen Parent Files

| Index | Artifact | SHA-256 |
| ---: | --- | --- |
| 0 | Cost Garden | `ecef7f1ff683f1157e89bb5e552e121f51c4191e9662b17a0a5f4a71909223a8` |
| 1 | selector admission source | `b9249fe24f5d08fb012631346164d826b8ee975130b0f298a809ad48f4843a66` |
| 2 | selector admission semantics | `17196cbc2c3fa286c9c2c6e48f042cd3b180d731ee41e0e492077b355ca34ea9` |
| 3 | selector admission receipt | `2615448449a16faf1d826a6d42e0b0212036f485a3a3e815fc064c298070f979` |
| 4 | selector admission evidence | `a59d975337fb4e0d825038e25ba4bf4b11105e28863fdf837d1cba60919ffc7e` |
| 5 | execution-engine source | `8b5063f0e9a39650fb0b60e8b70b315f339723690e06050c2bebacece888e37e` |
| 6 | execution-engine semantics | `c47668a08ad25f39bebe9d8bef90b66eb2ad7119063c19ab8319fa4fab265233` |
| 7 | execution-engine receipt | `9da8ca53c3cb0e6631c92e55a8e82387aed2bd53863ffa9d646719806eec4ffd` |
| 8 | XOR operation source | `bc039d5db9f195b94fbeb08f22f9c96164a174c2cea675739e901a07fdf54db8` |
| 9 | XOR operation semantics | `40fe69829b1feb5843ea8b4720b70516303e8eda37c144989909b52d1b466fb1` |
| 10 | XOR operation receipt | `9e1e416defa4a1cfc477f0630623321e12209a40c9f5f878b85cce40be83d330` |

```text
parent_file_count=11
parent_file_match_count=11
parent_manifest_sha256=9a4f1f28651b8984a0d719ecc4415572b6b301c0ccbee7520d960afeea6bf605
admission_parent_live=true
operation_parent_live=true
engine_parent_live=true
```

The manifest is SHA-256 over the ordered `sha256sum` records in the table.
Sounio reads and hashes every file. A documentary hash does not replace the
live admission and operation matchers or the engine queries.

## Cost Vocabulary

The ontology distinguishes ten quantities:

```text
STATIC_SITE_COUNT
EXECUTED_INSTRUCTION_COUNT
EXECUTED_MICRO_OP_COUNT
DEPENDENCY_LATENCY
RECIPROCAL_THROUGHPUT
ELAPSED_DURATION
TRANSFER_BYTES
ENERGY
FREQUENCY
OCCUPANCY
```

It distinguishes eight units, eight scopes, five statistics, five subject
kinds, and four evidence roles. Structural records, measurement requests,
material observations, derived summaries, and comparison receipts remain
different classes.

`STATIC_SITE_COUNT` is a structural quantity. It cannot satisfy a latency,
dynamic instruction, or micro-op request.

`RECIPROCAL_THROUGHPUT` is admitted in cycles per operation. The vocabulary
term `OPERATION_PER_SECOND` denotes ordinary throughput and has no admitted
quantity pairing in this first freeze; it cannot label reciprocal throughput.

## Frozen Engine Projection

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

Darwin requests bind the observed T560 Xeon engine `703303` on machine
`701313`. Apple requests bind CPU blueprint `703402`. DGX requests bind GPU
blueprint `703405`. A blueprint may carry a value-free request, but cannot
carry a material observation.

The Apple tailnet locator is `demetrios@sounio-language-macbook`. It is a
transport coordinate, not an observed-engine receipt. No DGX IP address is
ingested as material identity in this freeze.

## Frozen Request Ledger

All seven requests use `MEDIAN`, `requested_sample_count=1001`,
`requested_warmup_count=128`, and `required_environment_mask=2047`.
All have `bound_environment_mask=0`, `value_present=false`, and zero rational
value fields.

| ID | Target | Machine | Engine | Blueprint | Subject | Quantity | Unit | Scope |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 701200 | 701313 | 703303 | false | 706281 | 706203 | 706221 | 706241 |
| 2 | 701200 | 701313 | 703303 | false | 706281 | 706204 | 706221 | 706242 |
| 3 | 701200 | 701313 | 703303 | false | 706282 | 706205 | 706222 | 706244 |
| 4 | 701201 | 0 | 703402 | true | 706281 | 706203 | 706221 | 706241 |
| 5 | 701201 | 0 | 703402 | true | 706281 | 706204 | 706221 | 706242 |
| 6 | 701202 | 0 | 703405 | true | 706281 | 706203 | 706221 | 706241 |
| 7 | 701202 | 0 | 703405 | true | 706281 | 706204 | 706221 | 706242 |

```text
requests=7
selector_requests=6
whole_operation_requests=1
static_site_facts=36
secondary_site_facts=32
```

The static-site fields are consumed from the admission parent and remain
structural facts. They are not divided into, substituted for, or otherwise
promoted to any requested material quantity.

## Sampling And Rational Domain

A later material observation must carry positive sample count, complete bound
environment, positive denominators, and ordered rational interval:

```text
0 <= component <= 1000000000
lower <= estimate <= upper
```

The component bound makes every cross-product at most `10^18`, below signed
`i64` capacity. Rational representation makes stored-ratio transport exact;
it does not make hardware, timer, sampling, or capture error vanish.

## Ontology And Boundary

```text
ontology_triples=413
ontology_requests=7
ontology_material_observations=0
ontology_derived_summaries=0
ontology_comparison_receipts=0
material_observations=0
derived_summaries=0
comparison_receipts=0
speedups=0
rankings=0
cross_target_comparable=false
parity_open=false
claim_ready=false
```

The result digest is:

```text
digest_words=166882395:243190428:1767805310:4234095682:63968340:1374547816:344010038:3839699970
digest_sha256_hex=09f26c5b0e7eca9c695e917efc5f2c4203d0145451edef6814812d36e4dd2c02
```

The digest binds the parent mask, every request coordinate and protocol field,
the engine/ontology/boundary counts, parent-live booleans, and negative tally.
The exact Sounio matcher additionally checks every named negative witness.

## Negative Surface

All twenty-six Sounio mutation witnesses pass. They cover parent removal;
structural-to-material promotion; instruction-to-micro-op promotion;
selector-to-operation scope; unnamed DGX node assignment; blueprint promotion;
interface-to-ISA promotion; GPU-to-host-engine transplant; invalid
quantity/unit and latency/scope pairs; zero sample and denominator; inverted
interval bounds; missing environment; valued request; C++ semantic authority;
summary and comparison without evidence; same-unit comparability; reviewer
promotion; premature stage; and Python/Rust producer launch.

The external gate adds live parent-file tamper and native Loom forbidden-Python
tests. The latter must return `E110` before interpreter launch.

## Review Boundary

xAI/Grok 4.5 acted only as `REVIEW_ONLY`. It found a partial-ledger validity
bug, an unstated rational overflow domain, and vacuous negative witnesses.
Those were corrected before this freeze. The final focused pass found no wrong
mask sum, rational ordering, digest packing, request arithmetic, or semantic
compatibility predicate.

No provider executed Sounio or created or confirmed a request, count, digest,
cost value, comparison, Loom decision, or authority result.

## Closed Claims

No material value, benchmark, Apple/DGX engine observation, target comparison,
speedup, ranking, instruction minimum, lowering choice, parity implementation,
or transform result exists here. `PARITY_OPEN=false` and
`CLAIM_READY=false`.
