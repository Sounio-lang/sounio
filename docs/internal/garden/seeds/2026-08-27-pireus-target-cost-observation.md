<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-pireus-target-cost-observation
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-pireus-target-cost-observation
-->

# Garden Seed: Pireus Target Cost Observation

> **Status**: Garden | **Date**: 2026-08-27 | **Authority**: founder

Concept-ID: `SOUNIO-PIREUS-TARGET-COST-OBSERVATION`

Semantic-Lane-ID: `pireus-target-cost-observation-20260827`

## Butterfly

> "O cluster Darwin reune varias arquiteturas... e muito solo fertil."

Pireus can now name machines, execution engines, ISAs, interfaces, target-local
selector realizations, and incomplete operation coverage. The next loss of
information happens when all of that becomes one untyped number called
"cost".

A static instruction site is not a retired instruction. A retired instruction
is not a micro-operation. A dependency-chain latency is not reciprocal
throughput. A selector-node price is not the price of the five-node operation.
A cycle count without frequency state is not a duration. A duration without
workload, affinity, toolchain, binary, and statistic is not a comparable
observation.

The Pireus form is therefore:

```text
CostObservation<
    target,
    machine,
    engine,
    subject,
    quantity,
    unit,
    scope,
    statistic,
    environment,
    evidence
>
```

The parameters are part of the meaning. Dropping one is not simplification; it
is a change of claim.

## Question

How can Pireus prepare material cost experiments across Darwin Xeon, Darwin
NVIDIA engines, Apple Silicon, and DGX without letting a structural count,
vendor table, compiler listing, shell harness, parity language, or external
reviewer invent a cost value or make unlike measurements comparable?

## Existing Frozen Parents

The first executable child must consume, not revise, these Sounio authority
surfaces:

| Parent | Role | SHA-256 |
| --- | --- | --- |
| selector material admission source | admission schema and exact target records | `b9249fe24f5d08fb012631346164d826b8ee975130b0f298a809ad48f4843a66` |
| selector material admission semantics | frozen coverage boundary | `17196cbc2c3fa286c9c2c6e48f042cd3b180d731ee41e0e492077b355ca34ea9` |
| selector material admission receipt | frozen Sounio authority record | `2615448449a16faf1d826a6d42e0b0212036f485a3a3e815fc064c298070f979` |
| selector material admission evidence | flat frozen record | `a59d975337fb4e0d825038e25ba4bf4b11105e28863fdf837d1cba60919ffc7e` |
| execution-engine source | machine to engine to ISA model | `8b5063f0e9a39650fb0b60e8b70b315f339723690e06050c2bebacece888e37e` |
| execution-engine semantics | frozen Darwin multi-engine meaning | `c47668a08ad25f39bebe9d8bef90b66eb2ad7119063c19ab8319fa4fab265233` |
| execution-engine receipt | frozen Sounio authority record | `9da8ca53c3cb0e6631c92e55a8e82387aed2bd53863ffa9d646719806eec4ffd` |
| XOR operation source | five-node subject graph | `bc039d5db9f195b94fbeb08f22f9c96164a174c2cea675739e901a07fdf54db8` |
| XOR operation semantics | frozen node and barrier meaning | `40fe69829b1feb5843ea8b4720b70516303e8eda37c144989909b52d1b466fb1` |
| XOR operation receipt | frozen Sounio authority record | `9e1e416defa4a1cfc477f0630623321e12209a40c9f5f878b85cce40be83d330` |

The child must live-check the executable parents where the current Sounio path
permits it and hash-check every frozen documentary parent. Parent mismatch is a
typed refusal, never a warning followed by fallback.

## Semantic Axes

### Target, Machine, And Engine

A cost observation belongs to one material engine. Target, machine, and engine
are separate fields:

```text
canonical target != observed machine != observed execution engine
```

Darwin CPU observations attach to a named Xeon engine. Darwin GPU observations
attach to a named NVIDIA engine on the same or another machine. An Apple or DGX
blueprint cannot receive a value until a material engine receipt exists.

### Subject

The measured subject is one of:

```text
instruction form
lowering node
complete operation
data movement edge
synchronization barrier
```

The initial executable may issue requests only for subjects that the frozen
admission parent actually covers. Selector coverage cannot silently authorize
the other four nodes or the complete operation.

### Quantity

The first vocabulary distinguishes at least:

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

`STATIC_SITE_COUNT` is structural. It is intentionally not a material cost
quantity and cannot satisfy a cost request.

### Unit

Units are explicit ontology terms, never implied by a field name:

```text
COUNT
CYCLE
NANOSECOND
BYTE
JOULE
HERTZ
OPERATION_PER_SECOND
RATIO
```

Quantity and unit must form an admitted pair. For example, latency may use
cycles or nanoseconds, while transfer bytes may use bytes. A mismatched pair is
refused before comparison.

### Scope

Scope distinguishes the experiment being performed:

```text
STATIC_BINARY
DEPENDENCY_CHAIN
INDEPENDENT_STREAM
SINGLE_NODE
COMPLETE_OPERATION
HOST_TO_DEVICE
DEVICE_TO_HOST
DEVICE_LOCAL
```

An independent stream cannot answer a dependency-chain question. A single
node cannot answer a whole-operation question.

### Statistic And Sampling

Every observed value carries a sampling contract:

```text
sample_count
warmup_count
statistic
quantile_numerator
quantile_denominator
estimate_numerator
estimate_denominator
lower_numerator
lower_denominator
upper_numerator
upper_denominator
```

The initial statistics vocabulary distinguishes raw sample, minimum, median,
mean, and quantile. Values are rational records rather than ambient `f64`
literals, so transport of the stored ratio is exact. Device, timer, sampling,
and any prior capture error remain explicit obligations of the interval and
evidence record; rational transport does not erase them. The interval is
evidence, not decoration.

### Environment

A material observation is incomplete unless it binds:

```text
hardware receipt
toolchain receipt
binary hash
command hash
input/workload hash
OS or driver record
thread count
affinity policy
frequency policy or observed frequency state
power policy when relevant
timer or performance-counter source
```

An unknown field remains explicitly unknown. It does not inherit a convenient
default from another target.

### Evidence Role And Stage

The cost ontology distinguishes:

```text
MEASUREMENT_REQUEST
MATERIAL_OBSERVATION
DERIVED_SUMMARY
COMPARISON_RECEIPT
```

Sounio is `SEMANTIC_AUTHORITY`: it defines admissibility, request meaning, and
the first executable ledger. C++ may later produce `MATERIAL_PARITY` receipts.
Lean, Koka, and Haskell retain their established parity roles. External LLMs
remain `REVIEW_ONLY`.

A request has no value. A material observation cannot become a comparison
receipt merely because two values share a unit.

## Comparability

Two observations are comparable only if a Sounio predicate explicitly accepts
their quantity, unit, scope, subject meaning, workload identity, statistic,
sampling contract, and required environment fields.

```text
same unit != same experiment
same operation spelling != same subject realization
same result bits != same cost
same static site count != same dynamic work
```

Cross-target comparison remains false until a separate Sounio comparison
contract consumes complete target-local observations. This Garden does not
define a winner, speedup, ranking, or equivalence class.

## First Sounio Executable

The first child should be:

```text
stdlib/hardware/pireus/target_cost_observation.sio
examples/pireus_target_cost_observation.sio
tests/stdlib/hardware/test_pireus_target_cost_observation.sio
```

It must:

1. bind the ten frozen parent artifacts above;
2. construct the cost vocabulary as Pireus ontology triples;
3. keep structural facts, requests, observations, summaries, and comparisons
   as different classes;
4. derive measurement requests only from explicitly admitted target subjects;
5. retain incomplete Apple and DGX whole-operation coverage;
6. represent Darwin CPU and GPU engines without attaching one engine's ISA or
   costs to another;
7. emit complete request records with no material value;
8. emit zero material observations, zero derived summaries, zero comparison
   receipts, zero speedups, and zero rankings;
9. reject invalid quantity/unit, subject/scope, target/engine, evidence-role,
   environment, and promotion combinations;
10. commit every emitted record and negative witness to a deterministic digest;
11. keep `PARITY_OPEN=false` and `CLAIM_READY=false`.

No expected request count, result digest, target digest, or negative count is
defined here. Those values may first appear only in the Sounio executable
stream, after this Garden is committed.

## Negative Surface

The executable must include mutation witnesses for at least:

1. missing Garden binding;
2. missing or mismatched parent binding;
3. static site count promoted to latency;
4. static site count promoted to dynamic instruction count;
5. instruction count promoted to micro-op count;
6. selector request promoted to complete-operation scope;
7. unnamed DGX coverage assigned to a named node;
8. Apple or DGX blueprint promoted to an observed engine;
9. CUDA or Metal promoted to an ISA;
10. a Darwin GPU cost attached to its host Xeon engine;
11. quantity/unit mismatch;
12. dependency latency labeled as independent-stream throughput;
13. zero sample count;
14. zero denominator;
15. lower bound greater than estimate;
16. estimate greater than upper bound;
17. missing toolchain, hardware, binary, command, or workload binding;
18. missing affinity or frequency state where required;
19. request carrying a material value;
20. material observation carrying `SEMANTIC_AUTHORITY` from C++;
21. derived summary without admitted observations;
22. comparison without a comparison receipt;
23. same-unit values promoted to comparable without semantic compatibility;
24. external review promoted to authority;
25. premature parity or claim readiness;
26. Python or Rust producer/oracle request reaching interpreter launch.

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

This Garden must be committed before any expected child result is written. No
Lean, Koka, C++, Haskell, benchmark, or remote target execution may run for the
child before the Sounio result and semantics are frozen by hash.

## Enforcement

The existing Loom language-authority guardian remains the only stage and role
enforcer. The child gate must:

- authorize Sounio execution before launch;
- fail closed on missing policy, error, timeout, hash drift, or parent drift;
- log every `ALLOW` or `DENY` decision and reason;
- verify the exact source, semantics, parents, toolchain, hardware, command,
  and result hashes;
- run a live parent-file tamper negative;
- request a Python producer deliberately and prove `E110` with zero interpreter
  launches;
- contain no Python or Rust guardian or semantic oracle.

Shell may transport, hash, and compare frozen records. It may not calculate
expected cost semantics or expected Sounio results.

## Evidence State

| Layer | Status |
| --- | --- |
| `Garden` | Cost meaning, admissibility axes, and first executable boundary fixed here. |
| `Hypothesis` | Typed observations prevent cross-engine and cross-experiment cost promotion. |
| `Executable` | Pending the first post-Garden Sounio child. |
| `Claim-ready` | No. No material value, comparison, speedup, or ranking exists. |

## What This Is Not

- It is not a benchmark result.
- It is not an instruction-cost table.
- It is not a latency or throughput claim for any Xeon, NVIDIA, Apple, or DGX
  engine.
- It is not permission to divide an elapsed time by static sites and call the
  quotient an instruction cost.
- It is not cross-ISA equivalence or a performance ranking.
- It is not a claim that one metric is sufficient for lowering selection.
- It is not a new Guardian or a parity implementation.
- It is not a claim that external review confirmed a value.

## Next Material Bridge

After the Sounio child is frozen, a separate Garden may choose one complete
request and realize it on named Darwin engines under Loom. Apple and DGX may
join only through their own material engine receipts. The first comparison
must remain target-local until Sounio explicitly proves the two request
contracts comparable.

## Semantic Lane Declaration

Concept-IDs: `SOUNIO-PIREUS-TARGET-COST-OBSERVATION`;
`SOUNIO-PIREUS-XOR-SELECTOR-MATERIAL-ADMISSION`;
`SOUNIO-PIREUS-EXECUTION-ENGINE`;
`SOUNIO-PIREUS-XOR-CONVOLUTION-OPERATION`

Semantic-Boundary: typed target-local cost requests and observations without
material values, cross-target comparison, speedup, ranking, or claim promotion

Transformation: frozen target, engine, operation, and admission parents to a
Sounio-owned cost vocabulary and measurement-request ledger

Proof-Obligation: each request is complete, value-free, target-local,
engine-correct, subject-covered, unit-compatible, digest-bound, and promotion
closed

Write-Set: Garden, first Sounio module/executable/test, concept contract,
semantics, receipt, evidence, registry, governance metadata, CI gate, and
review log

Effects-Changed: ontology construction and query use `Mut` and `Epistemic`;
receipt hashing and executable output additionally use `IO`, `Alloc`, `Panic`,
and `Div`

Public-API: `hardware::pireus::target_cost_observation`

Migration-Impact: additive; every frozen parent remains immutable

Integration-Target: current Pireus operation, execution-engine, and selector
material-admission pipeline

Verification-Gate: `scripts/ci/pireus_target_cost_observation.sh`

Acceptance-Evidence: post-Garden Sounio stream, dedicated Sounio test, frozen
receipt/evidence hashes, parent tamper refusal, Loom decisions, and forbidden
Python pre-launch refusal

Open-Questions: first material request selection; counter sources by engine;
frequency-state normalization; target-local uncertainty policy; later
cross-target comparability proof
