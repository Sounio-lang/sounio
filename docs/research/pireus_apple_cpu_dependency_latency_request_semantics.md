<!-- docs:meta
topic_id: repo.docs.research.pireus-apple-cpu-dependency-latency-request-semantics
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-apple-cpu-dependency-latency-request-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus Apple CPU Dependency-Latency Request Semantics

Concept-ID: `SOUNIO-PIREUS-APPLE-CPU-DEPENDENCY-LATENCY-REQUEST`

Status: `SEMANTICS_FROZEN`

Canonical source:
`stdlib/hardware/pireus/apple_cpu_dependency_latency_request.sio`

## Scope

This surface freezes one value-free composition:

```text
target-cost request ID 4
+ exact material machine APPLE_MAC17_7
+ exact CPU engine APPLE_M5_MAX_CPU
+ Apple A64 TBL receipt, evidence, and hardware identity
= request state MATERIAL_IDENTITY_BOUND
```

The composition does not mutate request ID 4. `PireusRequestTemplateRef` and
`PireusMaterialEngineBinding` are distinct types. The binding contains no
transport locator and the binding predicate does not read one.

The result establishes a material identity binding only. It does not establish
an executable measurement interface, feasibility of cycle measurement,
environment binding, execution authorization, observation, summary,
comparison, speedup, ranking, parity, or claim readiness.

## Causal Order

The Garden commit is:

```text
b1d80d17f0d2ab915557b34732e9580df269e19a
```

The first executable commit is:

```text
763322b28df3709eb7544d38dbc00ee779071631
```

That commit contains no
`pireus_apple_cpu_dependency_latency_request_matches_frozen_semantics`
symbol. The exact matcher was written only after the first executable emitted
the counts, flags, and digests frozen here.

The first executable source SHA-256 was:

```text
34402df6ab19924a5107d4f386667e03d7f14406c5c5e49194d42e77b5985b3c
```

The matcher-bearing source SHA-256 is:

```text
3a92553aae0c4a9606f4964e1613a31452b7fb4d197b128dbb73cc24b87b550e
```

## Ordered Parent Manifest

SHA-256 covers the raw bytes of each repository file. The fixed order is:

| Index | Role | Path | SHA-256 |
| ---: | --- | --- | --- |
| 0 | committed Garden | `docs/internal/garden/seeds/2026-08-28-pireus-apple-cpu-dependency-latency-request.md` | `b574d7352019576dceceab32675834aedb24961dcd76f97646a5cbed4277aa7f` |
| 1 | target-cost source | `stdlib/hardware/pireus/target_cost_observation.sio` | `7ea2815c112b85476fc6ac4d8bb9388ee032062822c6905485c2084ee416d6bc` |
| 2 | target-cost semantics | `docs/research/pireus_target_cost_observation_semantics.md` | `0a899be7cd25375c8c444b9e1f0a71dd102ca8958072a4290073ae21c926a199` |
| 3 | target-cost receipt | `docs/research/receipts/pireus_target_cost_observation_20260827.md` | `b7577c782a82431eb54312137a52c1811f14316d5e5a5fb9e7aa9581f1c304ca` |
| 4 | target-cost evidence | `docs/research/evidence/pireus_target_cost_observation_20260827.txt` | `06f21108ddc89c8c468097b7cefec6a766bda065918fd5797be194878371577b` |
| 5 | material-admission source | `stdlib/hardware/pireus/material_engine_admission.sio` | `b98d799fec6452f9afaecc7e418578cc0ca72a1d3e7c0a1f84d474e0aa2730ec` |
| 6 | material-admission semantics | `docs/research/pireus_material_engine_admission_semantics.md` | `bbbd7bc9c99e1de46a5317d99a7893adebfdf7fb46454cbd7002b3098281b6ee` |
| 7 | material-admission receipt | `docs/research/receipts/pireus_material_engine_admission_20260828.md` | `352ad87a4d05ef32333acb2f31a48f49e7d520ab99c3e99a8b4a3e6bf3f55aa4` |
| 8 | material-admission evidence | `docs/research/evidence/pireus_material_engine_admission_20260828.txt` | `08a6e0d4b2e5a0bb4e981cfcc8df5192d9b9b77f53eb3862c86f29d9409e425b` |

The ordered `sha256sum` manifest has SHA-256:

```text
aca23c0c43db3fee6d4fd8c7f4ca58ed9bde460c6b0e5978442170d3ab7320af
```

The Sounio evaluator also serializes the nine actual digest values in the same
order and hashes that serialization. The resulting digest words are:

```text
2484893681:3053564117:3251292557:837524480:2326288797:2545336006:1298056031:153129504
```

Both executable source parents are evaluated live and must satisfy their own
exact frozen-semantics matchers. Hash equality without live parent acceptance
is insufficient.

## Frozen Request Template

The immutable projection is ledger index `3`, request ID `4`:

```text
request_id=4
target=APPLE_SILICON
machine=ABSENT
engine=APPLE_CPU_BLUEPRINT
engine_is_blueprint=true
engine_observed=false
subject_kind=LOWERING_NODE
subject_node=XOR_SELECTOR
quantity=DEPENDENCY_LATENCY
unit=CYCLE
scope=DEPENDENCY_CHAIN
statistic=MEDIAN
requested_samples=1001
requested_warmups=128
required_environment_mask=2047
bound_environment_mask=0
evidence_role=REQUEST
producer_language=Sounio
producer_role=SEMANTIC_AUTHORITY
value_present=false
estimate_numerator=0
estimate_denominator=0
lower_numerator=0
lower_denominator=0
upper_numerator=0
upper_denominator=0
subject_covered=true
admitted=true
```

`engine_observed=false` belongs to the request-side blueprint. It is not a
cost observation and cannot be laundered through the material binding.

## Frozen Material Binding

The distinct binding record is:

```text
template_request_id=4
target=APPLE_SILICON
machine=APPLE_MAC17_7
engine=APPLE_M5_MAX_CPU
engine_kind=CPU
isa=AArch64
receipt=APPLE_A64_TBL
evidence=APPLE_A64_TBL_EVIDENCE
hardware=APPLE_MAC17_7_HARDWARE
material_subject=XOR_SELECTOR
material_operation=APPLE_A64_TBL
eligibility=APPLE_DEPENDENCY_LATENCY
state=MATERIAL_IDENTITY_BOUND
execution_authorization_id=ABSENT
measurement_feasibility=UNKNOWN
producer_language=Sounio
producer_role=SEMANTIC_AUTHORITY
```

The Mac locator `demetrios@sounio-language-macbook` remains routing-only. It
is not serialized in the template, binding, ontology projection, or result
digest. No connection to the Mac occurred in this freeze.

## Stage And Boundaries

The request state chain is:

```text
TEMPLATE
< MATERIAL_IDENTITY_BOUND
< EXECUTION_AUTHORIZED
< OBSERVED
< SUMMARY_ELIGIBLE
< COMPARISON_ELIGIBLE
```

This child admits only:

```text
TEMPLATE -> MATERIAL_IDENTITY_BOUND
```

The twenty result-boundary booleans, in serialization order, are:

1. parent files valid;
2. target-cost parent live;
3. material-admission parent live;
4. request template valid;
5. material instance valid;
6. target matches instance;
7. subject matches receipt;
8. quantity eligible;
9. material identity binding valid;
10. measurement interface ready;
11. execution authorized;
12. environment bound;
13. value present;
14. material observation present;
15. derived summary present;
16. comparison present;
17. speedup present;
18. ranking present;
19. parity open;
20. claim ready.

The first nine are `true`. The final eleven are `false`.

The speedup, ranking, parity, claim, and transport-to-identity predicates are
deliberate constant-closed policy boundaries in this pre-launch type. Their
false value is not an empirical observation about the Mac.

## Ontology Projection

The append-only projection contains `26` triples:

```text
4 class declarations
7 request-template triples
15 material-binding triples
```

Nine class or relation queries each return exactly one row for the expected
template, binding, machine, engine, receipt, evidence, eligibility, state, and
feasibility relation. This query-count check is conjoined with the complete
typed binding predicate; query count alone does not prove object identity.

The ontology digest serializes:

```text
tag
triple_count
for each triple in append order:
  subject
  predicate
  object_iri
  object_literal
  f64_to_bits(object_value)
  is_object_literal
```

The frozen ontology digest is:

```text
3934472735:2831266448:3468731955:3829704731:2302707619:4052771782:1038330726:1883965869
```

The `lean_single` lowering crashed when this projection returned a compound
result after several live `SparqlResult` values. The executable therefore uses
an explicit mutable nine-slot output buffer. This is an implementation-path
constraint, not part of the abstract binding meaning.

## Negative Surface

The fixed single-family order is:

1. missing any one of the nine direct parent files, including the Garden;
2. request ID drift;
3. target drift;
4. subject kind or node drift;
5. quantity drift;
6. unit drift;
7. scope drift;
8. statistic, sample-count, or warmup drift;
9. required or bound environment drift;
10. value, interval, or derived artifact inserted;
11. DGX, Xeon, GPU, Metal-interface, or blueprint composition;
12. machine, receipt, evidence, or engine coordinate drift;
13. unresolved DGX `.48` request used as a binding;
14. transport reachability and key state promoted to identity evidence;
15. C++ promoted to semantic authority;
16. external LLM review promoted to authority;
17. Python or Rust promoted to producer, oracle, or guardian;
18. interface, feasibility, or execution authorization inferred;
19. request-side `engine_observed` laundered into a cost observation;
20. premature request stage.

The fixed pair products are:

21. machine plus receipt;
22. engine plus evidence;
23. target plus engine;
24. subject plus eligibility;
25. unit plus feasibility;
26. value plus bound environment;
27. summary plus observation state;
28. execution stage plus execution authorization.

All 28 pass flags are conditioned on the positive composition being valid.
The parent family clears each of the nine parent bits independently. The
unresolved `.48` family includes a positive control for the resolved Apple
coordinate before rejecting the nonzero unresolved request ID. No independence
or Cartesian-product completeness theorem is claimed.

```text
negative_passed=28
negative_total=28
```

## Canonical Result Digest

Strings are length-prefixed and hashed bytewise. Signed `i64` values are eight
big-endian bytes. Booleans are `0` or `1` through the same integer serializer.
Each SHA-256 value is represented as eight big-endian 32-bit words stored in
`i64` lanes.

The result digest order is:

```text
Concept-ID tag
pireus.apple-cpu-dependency-latency-request.v0 tag
direct-parent mask
eight direct-parent-manifest digest words
eight live target-cost result-digest words
eight live material-admission result-digest words
template index and request ID
all 28 request fields
all 17 binding fields
stage from and to tags
twenty result-boundary booleans
ontology triple count
eight ontology-digest words
twenty-eight negative booleans
negative passed and total
```

The result tag remains `v0` because it is the schema born in the first
executable commit. The authority stream becomes `v1 / SEMANTICS_FROZEN` when
it adds exact matcher enforcement; it does not rewrite the result digest.

```text
result_digest=4077763757:3866677980:1417460964:3007782540:1995848965:1363182042:817142050:850659687
authority_result_lines=124
authority_result_bytes=2145
authority_result_sha256=443e49c11a2ad04ac7e3f9b061bebe50fca7959f812aa93039c449c2133b5349
test_result_sha256=88dd937d09afb205e80e77e646300928d3d7dad1f53d98a62a9232b1f6579774
```

## Authority And Claims

Sounio owns the template projection, material binding, predicates, ontology,
negative witnesses, canonical serialization, result, and exact matcher. The
parent Apple hardware receipt remains `C++ / MATERIAL_PARITY`. External LLMs
are `REVIEW_ONLY` and create or confirm no result. Python and Rust remain
forbidden as producer, oracle, or guardian.

The executable uses explicit `lean_single` routing. The default Madaros check
is advisory while its frontend reports an incomplete AST closure for this
import graph.

No Lean, Koka, C++, Haskell, remote Apple command, benchmark, measurement, or
parity run was opened for this child. The frozen result remains:

```text
PARITY_OPEN=false
CLAIM_READY=false
```
