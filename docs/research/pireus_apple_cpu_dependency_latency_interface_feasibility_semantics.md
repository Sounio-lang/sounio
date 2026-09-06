<!-- docs:meta
topic_id: repo.docs.research.pireus-apple-cpu-dependency-latency-interface-feasibility-semantics
authority: historical
audience: researchers
last_validated: 2026-08-28
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.pireus-apple-cpu-dependency-latency-interface-feasibility-semantics
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus Apple CPU Interface-Feasibility Assessment-Schema Semantics

Concept-ID:
`SOUNIO-PIREUS-APPLE-CPU-DEPENDENCY-LATENCY-INTERFACE-FEASIBILITY`

Status: `SEMANTICS_FROZEN`

Empirical verdict: `UNASSESSED`

Canonical source:
`stdlib/hardware/pireus/apple_cpu_dependency_latency_interface_feasibility.sio`

## Scope

This surface freezes a result-free Sounio assessment schema for one exact
material coordinate. Here, result-free means that no candidate, observation,
feasibility verdict, authorization, or expected material value was present when
the schema first executed. It does not mean normatively neutral: the
measurement-integrity requirements below are deliberate stipulations.

```text
target=APPLE_SILICON
machine=APPLE_MAC17_7
engine=APPLE_M5_MAX_CPU
engine_kind=CPU
isa=AARCH64
subject=XOR_SELECTOR
quantity=DEPENDENCY_LATENCY
unit=CYCLE
scope=DEPENDENCY_CHAIN
statistic=MEDIAN
samples=1001
warmups=128
environment_mask=2047
```

The environment mask is inherited from the parent request and is the exact OR
of eleven required fields:

```text
1 hardware                 2 toolchain
4 binary                   8 command
16 workload                32 OS/driver
64 threads                 128 affinity
256 frequency              512 power
1024 timer/counter
```

`samples=1001` gives an odd-sized median request and `warmups=128` is part of
the inherited question. Neither is a power claim. The material harness must
validate the measurand and may refuse this contract; changing either count
requires a new semantics version rather than retrospective adjustment.

It does not freeze a concrete Apple API, a material candidate, a counter
encoding, an authorization, a cost, or an interface-feasibility verdict. Its
authority result is:

```text
stage=ASSESSMENT_REQUESTED
verdict=UNASSESSED
candidate_count=0
terminal_count=0
feasible_count=0
refusal_count=0
cycle_ineligible_count=0
```

The `Mac17,7 / Apple M5 Max` coordinate was already present in the exact parent
request and its pinned material-inventory ancestry. That identity is not
evidence that any particular interface exists. The Mac transport locator is
absent from the request, result, ontology, and digests. The freeze receipt
attests that the authority commands ran only on the recorded Xeon authority
host and that no target-transport command was issued.

## Causal Order

The Garden commit is:

```text
30237723bc53bbee48a93893be4da5b5f2118053
```

The first executable commit is:

```text
c924d0014c88af8873eeaa3ca5d2c11cf468a167
```

That commit contains no
`pireus_apple_cpu_dependency_latency_interface_feasibility_matches_frozen_semantics`
symbol and no hardcoded child result.

The first executable source SHA-256 was:

```text
0893a32298d30cd1978039fa5b69c637e446aa5da112812fcf776cd52fbc4767
```

The matcher-bearing source SHA-256 is:

```text
d8c7e6f9410c36f6858fb2379efa010a5adbaa32c615d89edc3e764a0606a6be
```

The Garden commit is an ancestor of the first executable commit, which is an
ancestor of the freeze commit. The dedicated gate extracts the first executable
tree, reruns that matcher-free source, and requires this exact stdout witness:

```text
sha256=8d1fd281f079f0287e4ceddfea31a3c51594e8cfb4b0196e2c9fa1b68b236c06
lines=141
bytes=2499
```

The matcher-bearing output adds only the frozen-stage witness and has SHA-256
`488b92632a0fdaa985618a67d03f84b81f69f0d7b33e2af243360f84215e81f5`.
Thus the committed matcher-free source independently regenerates the same
request, classifier result, ontology words, and result words. This is Git
ancestry plus deterministic replay, not an external timestamp or transparency
log claim.

## Ordered Parent Manifest

Raw SHA-256 covers each repository file in this fixed order:

| Index | Role | Path | SHA-256 |
| ---: | --- | --- | --- |
| 0 | committed Garden | `docs/internal/garden/seeds/2026-08-28-pireus-apple-cpu-dependency-latency-interface-feasibility.md` | `19482cbceb1bf7f3f7236446ebeff8b7d46c7b99249ba5910ece145fad641dd7` |
| 1 | request source | `stdlib/hardware/pireus/apple_cpu_dependency_latency_request.sio` | `3a92553aae0c4a9606f4964e1613a31452b7fb4d197b128dbb73cc24b87b550e` |
| 2 | request semantics | `docs/research/pireus_apple_cpu_dependency_latency_request_semantics.md` | `9bd767db814e47bfc087e07c0f9ff33b65faea5b885ae0f8ed3a6e646c015e6d` |
| 3 | request receipt | `docs/research/receipts/pireus_apple_cpu_dependency_latency_request_20260828.md` | `0ee12f3502efb26056bdbcf850360c0a5df727627a3c67499d363744f7c73272` |
| 4 | request evidence | `docs/research/evidence/pireus_apple_cpu_dependency_latency_request_20260828.txt` | `cf4455690426038cc7477b673bcf763e9755e8147f1ff55e882086826626482b` |

The ordered raw-file manifest SHA-256 is:

```text
bb0c19a4f03dea06ed496b3a9f7d8f29b3122962a8d08f0cc03f848cb0b91607
```

It is computed under `LC_ALL=C` as the SHA-256 of the newline-terminated,
ordered `sha256sum` records, including each relative path:

```text
sha256sum <the five paths above, in order> | sha256sum
```

The evaluator separately serializes the five actual digest values in the same
order. Its parent-manifest digest words are:

```text
3519613688:585074947:3796294859:195503079:3757313902:1414521947:791258454:3618061202
```

The request source is also executed live and must pass its exact frozen matcher.
Hash equality without live parent acceptance is insufficient.

## Requirement Vocabulary

The exact claim-admissibility predicate requires one complete candidate record
to carry all of these properties simultaneously:

```text
family in {
  CORE_CYCLE_COUNTER,
  PROCESS_PMU_CYCLE_EVENT,
  SYSTEM_TRACE_CYCLE_EVENT
}
native_unit=CYCLE
counter_domain=EXECUTING_CPU_CORE
event_semantics_evidence in {
  PRIMARY_AND_ON_DEVICE,
  ON_DEVICE_CONFIGURATION
}
availability=AVAILABLE
privilege in {
  USER_EVIDENCED,
  ELEVATED_EVIDENCED,
  ENTITLEMENT_EVIDENCED
}
read_boundary=SERIALIZED_BEFORE_AND_AFTER
core_migration=DETECTED_REJECTED
migration_evidence=DETECTED_REJECTED
multiplexing=DISALLOWED
0 < counter_width_bits <= 64
wrap=MODULO_WIDTH
read_overhead=BASELINES_REQUIRED
binary_identity=HASH_BOUND
command_identity=HASH_BOUND
producer_language=C++
producer_role=MATERIAL_PARITY
evidence_origin_count=1
transport_locator_ref=ABSENT
refusal=NONE
```

The interface-physics conjuncts are family, unit, counter domain, event
identity, serialization, migration, multiplexing, width, wrap, and overhead.
The hash, producer, role, origin-count, locator, and refusal conjuncts are
evidence-admissibility policy. Requiring C++/`MATERIAL_PARITY` does not assert
that another language is physically incapable of reading cycles; it prevents a
parity producer from becoming semantic authority under the founder contract.

The complete family vocabulary has six exclusive values:

1. `CORE_CYCLE_COUNTER`;
2. `PROCESS_PMU_CYCLE_EVENT`;
3. `SYSTEM_TRACE_CYCLE_EVENT`;
4. `ARCHITECTURAL_TIMER_TICK`;
5. `OS_MONOTONIC_TIME`;
6. `FREQUENCY_DERIVED_ESTIMATE`.

The final three are negative-control families. A frequency-derived candidate
can never satisfy the frozen cycle-native predicate. Timer ticks, nanoseconds,
elapsed time, nominal-frequency cycles, average-frequency cycles, and scaled
sampling estimates are not silently retyped as native cycles.

This version has six typed observation slots. That is an explicit executable
capacity, not a theorem that macOS has only six interfaces. A future manifest
with more candidates requires a new semantic version and cannot be squeezed
into this frozen type.

## Evidence And Refusal Algebra

Event-semantics evidence is one of:

```text
PRIMARY_AND_ON_DEVICE
ON_DEVICE_CONFIGURATION
NAME_OR_HEADER_ONLY
DERIVED_OR_INFERRED
ABSENT
CONFLICTING
```

Migration evidence is one of:

```text
DETECTED_REJECTED
INCAPABLE
UNKNOWN
CONFLICTING
ABSENT
```

Privilege evidence is one of:

```text
USER_EVIDENCED
ELEVATED_EVIDENCED
ENTITLEMENT_EVIDENCED
UNKNOWN
DENIED_EVIDENCED
ABSENT
```

`UNKNOWN`, `CONFLICTING`, and `ABSENT` are distinct. They cannot be promoted
to an evidenced positive or a named terminal refusal.

The terminal classes are pairwise disjoint:

```text
SATISFIES
TERMINAL_REFUSAL
CYCLE_INELIGIBLE
```

An unavailable privilege candidate is terminal only when its exact material
receipt evidences `PRIVILEGE_DENIED`, its privilege field is
`DENIED_EVIDENCED`, and all read-result fields irrelevant to that refusal are
explicitly absent. A migration refusal requires an otherwise cycle-native,
available candidate with `migration_evidence=INCAPABLE`. A cycle-ineligible
terminal requires material evidence for non-absent native unit and counter
domain plus the named failed predicate.

One refusal does not close a candidate set. A falsified verdict additionally
requires:

```text
transport_failure=false
manifest_closed=true
closure_evidence_ref!=0
declared_candidate_count=observed_candidate_count>0
all candidate IDs pairwise distinct
all observations bind the same candidate_manifest_ref
every observed candidate terminal
no feasible candidate
```

`closure_evidence_ref` must identify the future hash-bound material receipt
that states the enumeration method, exact host and engine, candidate-manifest
identity, declared count, observed count, and family coverage. Closure is only
for that declared candidate manifest. It is never a claim that the six-family
vocabulary exhausts every current or future macOS interface.

The classifier precedence is total:

1. any complete satisfying candidate yields `FEASIBLE_CYCLE_NATIVE`;
2. otherwise a complete closed terminal set yields
   `FALSIFIED_FOR_FROZEN_CANDIDATE_SET`;
3. otherwise a classified set yields `INDETERMINATE`.

Before classification, `ASSESSMENT_REQUESTED` and
`MATERIAL_EVIDENCE_RECEIVED` both carry `UNASSESSED`.

## Synthetic Baselines

The negative harness does not claim material Apple feasibility. It validates
the classifier against four internally constructed records:

```text
complete cycle-native candidate -> FEASIBLE_CYCLE_NATIVE
closed exact privilege denial -> FALSIFIED_FOR_FROZEN_CANDIDATE_SET
closed migration-incapable candidate -> FALSIFIED_FOR_FROZEN_CANDIDATE_SET
closed cycle-ineligible candidate -> FALSIFIED_FOR_FROZEN_CANDIDATE_SET
```

If any baseline fails, the 32-negative aggregate is forced to zero. These are
classifier examples, not observations from `Mac17,7`. Their nonzero receipt and
evidence references are opaque synthetic fixtures: they exercise record shape,
precedence, and partition arithmetic, not receipt authenticity. Authenticity is
owned by Loom and the later material-parity gate. Named negatives 21 and 22
also require the explicit `INDETERMINATE` branch for an open refusal and for
conflicting evidence.

## Frozen Authority Result

The frozen result contains:

```text
parents=5/5
request_parent_live=true
requirement_valid=true
families=6
family_manifest_closed=false
material_candidate_count=0
evidence_received=false
classification_requested=false
candidate_manifest_ref=ABSENT
declared_candidate_count=0
manifest_closed=false
stage=ASSESSMENT_REQUESTED
verdict=UNASSESSED
ontology_triples=25
negatives=32/32
remote_execution=false
environment_bound=false
harness_frozen=false
measurand_validated=false
execution_authorized=false
cost_present=false
parity_open=false
claim_ready=false
```

The ontology digest words are:

```text
19739672:3332345842:2129057504:398490585:1926487274:3200654896:4045422321:2496634919
```

The result digest words are:

```text
175669911:1759239310:3072997883:3130912607:2026014757:2084589116:1191266368:4096324446
```

## Canonical Serialization

Each `i64` is serialized by its 64-bit two's-complement bit pattern as eight
big-endian bytes; all frozen semantic values are non-negative. Booleans use the
same encoding with `0` or `1`. Strings are serialized as an eight-byte byte
length followed by their unnormalized source bytes; this schema uses ASCII
domain tags and paths. SHA-256 outputs are exposed as eight non-negative
big-endian 32-bit limbs stored in `i64`. Parent files must have
`0 < size < 1048576` bytes or their digest is the all-zero refusal sentinel.

The result digest serializes, in order:

1. domain and schema tags;
2. five-parent match mask, parent-manifest digest, and live parent result digest;
3. request and requirement fields;
4. ordered family manifest and assessment request;
5. observation-set state and all six complete observation slots;
6. decision fields;
7. execution-authorization and cost records, including absent values;
8. 15 boundary booleans without collapsing parity and claim readiness;
9. ontology count and digest;
10. 32 named negatives plus passed and total.

The fifteen boundary booleans are, in order:

```text
parent_files_valid
request_parent_live
requirement_valid
manifest_valid
assessment_valid
observations_valid
decision_valid
remote_execution
environment_bound
harness_frozen
measurand_validated
execution_authorized
cost_present
parity_open
claim_ready
```

The first seven are true. The final eight are false.

Here, `manifest_valid`, `observations_valid`, and `decision_valid` mean that the
empty structures obey the frozen type, cardinality, reference, and precedence
rules. They do not mean closed, observed, or classified. Those distinct facts
remain `manifest_closed=false`, `evidence_received=false`, and
`classification_requested=false`.

## Ontology Projection

The append-only projection contains exactly these 25 triples. Each tuple is
`(subject, predicate, object)` and `rdf:type=1`:

```text
(709700..709705, 1, 700000)                         [6]
(709720, 1, 709700) (709721, 1, 709701)             [2]
(709722, 1, 709702) (709723, 1, 709703)             [2]
(709723, 709730, 709720) (709723, 709731, 709721)   [2]
(709723, 709732, 709722)                            [1]
(709723, 709734, 707301) (709723, 709735, 707302)   [2]
(709721, 709736, 706221) (709721, 709737, 709100)   [2]
(709723, 709738, 709500) (709723, 709739, 709510)   [2]
(709722, 709733, 709000..709005)                    [6]
```

The six queries are `?s rdf:type 709700`, `?s rdf:type 709701`,
`?s rdf:type 709703`, `709722 709733 ?family`, `709723 709738 ?stage`, and
`709723 709739 ?verdict`. Their cardinalities are `1,1,1,6,1,1`.
Cardinality is conjoined with the typed predicates and exact result matcher; it
is not used as an identity proof by itself.

## Negative Witnesses

The 32 named negatives refuse:

1. direct-parent mask drift;
2. non-live parent semantics;
3. request-ID drift;
4. material-machine drift;
5. foreign target or engine;
6. widened subject;
7. request-contract drift;
8. timer ticks as cycles;
9. nanoseconds as cycles;
10. frequency estimates as native cycles;
11. name-only event semantics;
12. absent counter source or domain;
13. unknown privilege as availability;
14. multiplexed events as native;
15. absent width or wrap evidence;
16. unserialized reads;
17. unknown migration as safe;
18. advisory affinity as enforced binding;
19. merged evidence and duplicate candidate IDs;
20. manifest closure without evidence and count binding;
21. one refusal as universal falsification;
22. silenced conflicting evidence;
23. transport locator as interface evidence;
24. transport failure as falsification;
25. inferred environment binding;
26. inferred execution authorization;
27. inserted value or summary;
28. C++ as semantic authority;
29. external review as material evidence;
30. Python or Rust as producer;
31. premature parity or claim promotion;
32. sibling-machine inference.

The first negative is an in-module mask-level mutation witness. It is not a
claim that a raw file was physically altered during the authority run. The
dedicated freeze gate supplies the separate raw-file tamper test.

## Authority Boundary

Sounio is the only semantic producer. The frozen result allows no child
language execution by itself. After the semantics hash is sealed, Loom may
open only the next parity stage:

```text
C++ -> MATERIAL_PARITY
Lean 4 -> FORMAL_PARITY
Koka -> EFFECT_PARITY
Haskell -> OPTIONAL_DENOTATIONAL_BASELINE
external LLM -> REVIEW_ONLY
Python -> PROHIBITED
Rust -> PROHIBITED
```

C++ may later return candidate and refusal evidence for classification. It may
not change the candidate families, requirement predicates, precedence rule,
expected Sounio result, or semantic hash. SSH or tailnet transport may move a
hash-bound probe and return bytes; it cannot act as an oracle.

## Deliberately Open

- which concrete Apple interfaces belong in the first closed candidate manifest;
- whether any interface exposes native executing-core cycles;
- privilege and entitlement outcomes on the exact Mac;
- read serialization and migration behavior;
- the complete typed environment supplement;
- the frozen measurement harness and validated measurand;
- execution authorization;
- any cost observation, comparison, lowering choice, parity, or claim.

`SEMANTICS_FROZEN` closes the predicate AST, enum values, precedence,
serialization, six-slot capacity, ontology projection, and the exact
`UNASSESSED` authority result. The concrete candidate manifest, harness,
measurand, and empirical verdict remain open. Changing any closed item requires
a new schema/domain version and new Garden lineage. The answer is not known.
