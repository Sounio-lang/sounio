<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-28-pireus-apple-cpu-dependency-latency-interface-feasibility
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-28-pireus-apple-cpu-dependency-latency-interface-feasibility
-->

# Garden Seed: Pireus Apple CPU Dependency-Latency Interface Feasibility

> **Status**: Garden | **Date**: 2026-08-28 | **Authority**: founder

Concept-ID:
`SOUNIO-PIREUS-APPLE-CPU-DEPENDENCY-LATENCY-INTERFACE-FEASIBILITY`

Semantic-Lane-ID:
`pireus-apple-cpu-interface-feasibility-20260828`

The Concept-ID is the durable semantic identity. The lane ID names this
implementation episode. Neither a transport locator nor a later material
receipt may replace the Concept-ID in an authority record.

## Butterfly

> "mac is sounio-language-macbook na tailnet"

Pireus now owns a frozen, value-free request for dependency latency in `CYCLE`
and a distinct material binding to the exact `Mac17,7 / Apple M5 Max CPU`.
That composition intentionally stopped before inventing a measurement
interface.

The next question is not how many cycles the selector costs. The next question
is whether the exact material CPU exposes any interface that can satisfy the
already-frozen unit and scope without converting a timer into cycles, assuming
a frequency, or hiding an inaccessible privilege boundary.

## Question

For the frozen request ID `4` bound to `APPLE_MAC17_7 / APPLE_M5_MAX_CPU`, what
evidence is sufficient for Sounio to classify a concrete measurement interface
as cycle-native, falsified for a declared closed candidate set, or still
indeterminate, while keeping execution authorization and every cost value
absent?

This Garden assigns no feasibility verdict, interface availability, candidate
count, result hash, digest, measurement, or expected child output.

## Existing Frozen State

The direct semantic parent is the committed Apple CPU request freeze:

```text
parent_commit=91035d55266959bc554c81f9e09d3ae3dfc83bea
request_id=4
target=APPLE_SILICON
machine=APPLE_MAC17_7
engine=APPLE_M5_MAX_CPU
engine_kind=CPU
isa=AArch64
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
request_state=MATERIAL_IDENTITY_BOUND
measurement_interface_ready=false
measurement_feasibility=UNKNOWN
execution_authorization_id=ABSENT
value_present=false
parity_open=false
claim_ready=false
```

This display is orientation only. The first executable must obtain every field
by live Sounio evaluation of the hash-pinned parent and its exact matcher. It
must never parse this Garden to manufacture a parent result.

`XOR_SELECTOR` remains the lowering node that selects logical source coordinate
`i XOR d` for lane `i` at fixed displacement `d`. It excludes sign application,
multiplication, reduction, accumulation, and the whole Cayley-Dickson
operation. This lane may define how to bracket that node in a dependency chain;
it may not broaden or rename the subject.

## Identity And Transport Remain Distinct

```text
material machine identity != tailnet identity
tailnet identity != SSH host-key identity
SSH reachability != interface availability
interface availability != cycle-native semantics
cycle-native semantics != execution authorization
execution authorization != observation
observation != summary or comparison
```

The operational locator is:

```text
demetrios@sounio-language-macbook
```

It is routing metadata only. It cannot enter the assessment request, interface
candidate identity, feasibility classifier, ontology digest, or expected
result. Only a later Loom-authorized transport may consume it, and that
transport must separately bind the remote host key and the inherited material
machine identity.

No remote command may run in this Garden or in its first Sounio executable
phase.

## Distinct Typed Objects

The first executable must keep at least these objects representation-distinct:

```text
PireusAppleCpuLatencyRequestRef
PireusMeasurementInterfaceRequirement
PireusMeasurementInterfaceCandidate
PireusInterfaceAssessmentRequest
PireusMaterialInterfaceObservation
PireusInterfaceFeasibilityDecision
PireusExecutionAuthorizationRef
PireusCostObservation
```

The assessment request contains requirements and an ordered candidate-family
manifest. A material interface observation is later parity evidence. A
feasibility decision is Sounio classification of such evidence. None of these
is a cost observation.

The types may share IDs only through explicit references. They may not alias
storage or use a common record whose meaning changes by stage.

## Requirement Schema

The frozen requirement must carry the inherited request coordinate plus these
interface obligations:

```text
required_native_unit=CYCLE
required_counter_domain=EXECUTING_CPU_CORE
required_subject=XOR_SELECTOR
required_scope=DEPENDENCY_CHAIN
required_read_boundary=SERIALIZED_BEFORE_AND_AFTER
required_multiplexing=DISALLOWED
required_core_migration=DETECTED_AND_REJECTED
required_counter_width=EVIDENCED
required_wrap_rule=EVIDENCED
required_privilege=EVIDENCED
required_event_semantics=PRIMARY_OR_ON_DEVICE_EVIDENCE
required_binary_identity=HASH_BOUND
required_command_identity=HASH_BOUND
required_environment=ALL_PARENT_BITS_PLUS_TYPED_SUPPLEMENT
```

`CYCLE` means the cycle domain named by a materially evidenced CPU-core cycle
counter. It does not mean architectural timer ticks, nanoseconds, elapsed time,
nominal-frequency cycles, average-frequency cycles, or a scaled sampling
estimate.

The first executable must define total predicates for every obligation. No
free `interface_ready` boolean may be stored beside contradictory fields.

## Candidate-Family Vocabulary

Sounio must define semantic candidate families before a material probe names
any concrete API:

```text
CORE_CYCLE_COUNTER
PROCESS_PMU_CYCLE_EVENT
SYSTEM_TRACE_CYCLE_EVENT
ARCHITECTURAL_TIMER_TICK
OS_MONOTONIC_TIME
FREQUENCY_DERIVED_ESTIMATE
```

These are classifications, not availability claims.

Each candidate has exactly one family, chosen from its evidenced observation
semantics, counter domain, and enablement scope rather than from an API name.
If several access paths expose the same underlying counter source, they remain
distinct candidate IDs that explicitly share `counter_source`; their fields
cannot be merged and their evidence cannot be counted as independent sources.

The later material probe may interrogate concrete surfaces such as direct
AArch64 counter reads, Apple performance-counter facilities, system tracing or
Instruments facilities, the architectural timer, and OS monotonic clocks. A
concrete surface enters a family only through an evidence-bearing adapter.
Names, headers, symbols, or successful linking alone do not establish its
counter domain or native unit.

Event semantics are admissible only when the evidence binds the concrete
interface and event identity to the executing-core cycle domain. The first
executable must distinguish at least:

```text
PRIMARY_SPEC_AND_ON_DEVICE_BINDING
ON_DEVICE_ENUMERATION_AND_CONFIGURATION
NAME_OR_HEADER_ONLY
DERIVED_OR_INFERRED
ABSENT
CONFLICTING
```

An admissible positive must bind the interface or event encoding, enablement
scope, configuration or open outcome, native unit, counter domain, read
semantics, and anti-multiplexing state. Primary documentation may define those
semantics only when a hash-bound on-device record establishes that the exact
documented interface and event are the ones present. On-device enumeration and
configuration may establish them only when the returned metadata names the
native unit and executing-core domain without a frequency conversion. The
remaining four evidence classes cannot establish a cycle-native positive.

The final three families above are explicit negative controls for the frozen
`CYCLE` requirement unless later primary evidence proves that a supposedly
timer-shaped surface actually exposes the required CPU-core cycle event. A
frequency-derived estimate is never cycle-native in this contract.

## Candidate Record

Every candidate must carry, without defaults inferred from its name:

```text
candidate_id
family
concrete_interface_id
access_path
counter_source
native_unit
counter_domain
event_identity
event_semantics_evidence
read_privilege
availability_state
read_serialization_contract
core_binding_contract
migration_observable
multiplexing_state
counter_width_bits
wrap_rule
read_overhead_contract
toolchain_identity
binary_identity
command_identity
material_receipt_ref
material_evidence_ref
producer_language
producer_role
```

An absent or unknown field stays typed as absent or unknown. It is not filled
from platform convention, an API name, a header comment, an external LLM, or
another architecture.

The first executable must close every enum, refusal code, optional-field tag,
and serialization tag used by this record. Every match over those vocabularies
must be exhaustive; later parity producers may populate only those frozen
values.

## Branching Decision, Not A Total Stage Chain

The assessment stage and feasibility verdict are distinct:

```text
assessment stage:
  ASSESSMENT_REQUESTED
  MATERIAL_EVIDENCE_RECEIVED
  CLASSIFIED

feasibility verdict:
  UNASSESSED
  FEASIBLE_CYCLE_NATIVE
  FALSIFIED_FOR_FROZEN_CANDIDATE_SET
  INDETERMINATE
```

The first executable must derive its emitted stage and verdict from its own
input state. This Garden does not assign which verdict it emits.

The classifier must be total and exclusive. `ASSESSMENT_REQUESTED` and an
unclassified `MATERIAL_EVIDENCE_RECEIVED` emit `UNASSESSED`. At `CLASSIFIED`,
it applies this precedence:

1. emit `FEASIBLE_CYCLE_NATIVE` when at least one complete candidate satisfies
   every requirement predicate;
2. otherwise emit `FALSIFIED_FOR_FROZEN_CANDIDATE_SET` only when its full rule
   below holds;
3. otherwise emit `INDETERMINATE`.

Thus partial or conflicting evidence about another candidate cannot override an
already complete feasible witness. In the absence of that witness, rule three
captures every state that lacks the complete closed-manifest refusal required
by rule two.

`FEASIBLE_CYCLE_NATIVE` requires at least one complete material candidate that
satisfies every requirement predicate simultaneously. Several incomplete
candidates may not be combined into one fictional interface.

`FALSIFIED_FOR_FROZEN_CANDIDATE_SET` requires all of the following:

- the candidate-family manifest is explicitly closed for this assessment;
- every declared candidate has a complete terminal material classification;
- each candidate has a named, evidence-backed refusal reason indexed by
  candidate ID and the failed requirement predicate, or a named cycle-ineligible
  classification indexed by the native-unit and counter-domain predicates;
- no timer, duration, frequency conversion, or scaled event is promoted to a
  cycle-native candidate;
- the verdict is scoped to the frozen manifest and exact material engine, not
  to every possible future macOS or hardware interface.

Evidence completeness is relative to a candidate's terminal state. A readable
candidate needs the full positive record. A terminally unavailable candidate
may be complete without a successful counter read only when a frozen refusal
predicate, such as a hash-bound privilege or entitlement denial for that exact
interface, is evidenced and all fields irrelevant to that terminal state stay
explicitly absent. `UNAVAILABLE` is not the same as `UNKNOWN`, and neither one
candidate nor one privilege denial closes the manifest.

`INDETERMINATE` is mandatory when evidence is partial, candidate closure is not
established, event semantics are ambiguous, access depends on an unbound
privilege or entitlement, sources conflict, migration capability remains
unknown or conflicting, or a candidate is visible but cannot be safely
classified.

An interface evidenced as incapable of detecting and rejecting migration has a
terminal refusal for `required_core_migration`; unknown or conflicting evidence
about that capability is not a refusal and remains indeterminate.

A refused privilege may make one candidate unavailable. It does not by itself
prove universal infeasibility.

## Environment And Protocol Supplement

The parent environment mask retains its eleven required categories: hardware,
toolchain, binary, command, workload, OS or driver, threads, affinity,
frequency, power, and timer or counter.

The interface assessment additionally requires typed evidence for:

```text
exact machine and CPU engine
OS build and kernel
core class
logical CPU at each read boundary
thread migration status
affinity mechanism and whether it is advisory or enforced
counter configuration and event identity
counter enablement scope
counter multiplexing state
counter width and wrap behavior
read ordering or serialization mechanism
privilege, entitlement, and ownership requirements
frequency, power, and thermal state visibility
compiler, assembler, linker, and binary hashes
probe source and command hashes
host-key and transport receipt
```

An interface can be semantically cycle-native while the later experiment still
lacks a bound environment. Therefore:

```text
interface_feasible != environment_bound
environment_bound != execution_authorized
execution_authorized != value_present
interface_feasible != harness_frozen
harness_frozen != measurand_validated
```

## Dependency-Chain Harness Requirements

The later probe contract must make the measured subject unambiguous:

- each iteration depends on the preceding selector result;
- the selector source coordinate remains `i XOR d` for the fixed logical
  displacement under test;
- compiler elimination, hoisting, constant folding, and vector reassociation
  are prevented and evidenced without changing the operation being tested;
- bracketing reads are ordered according to the admitted interface contract;
- empty-chain and read-only controls provide typed baselines for later estimates
  or bounds on interface cost;
- counter wrap, interrupt, migration, and multiplexing failures are typed;
- raw samples remain observations and are not silently replaced by a median;
- warmups, samples, outlier policy, dispersion, and interval construction are
  introduced by a later observation Garden, not invented here.

This lane judges only whether an admissible interface and harness contract can
exist. It does not assume additive or content-independent read overhead, derive
subject cost by subtracting a control, or produce the 1001 requested samples.
It does not establish an instruction or IR realization of `XOR_SELECTOR`, a
microarchitectural latency interpretation, or the scientific validity of a
later measurand. Those require a separately frozen harness before observation.

## Sounio Classifier Boundary

Sounio owns:

- the requirement and candidate schemas;
- candidate-family meaning;
- manifest closure rules;
- feasibility predicates and refusal reasons;
- canonical ontology projection and serialization;
- decision digest;
- negative witnesses;
- the exact matcher after the first executable result exists.

C++ may later populate a `PireusMaterialInterfaceObservation` as
`MATERIAL_PARITY`, but it may not choose the feasibility verdict or change a
requirement. A later Sounio executable must ingest the hash-bound material
receipt and apply the frozen classifier.

Lean 4 may prove `FORMAL_PARITY` of the classifier. Koka may check
`EFFECT_PARITY` of access, privilege, and failure effects. Optional Haskell may
serve as a denotational baseline. None may execute before the Sounio semantics
are frozen by hash.

External LLMs remain `REVIEW_ONLY` and cannot confirm availability, counter
semantics, feasibility, or a result. Python and Rust remain prohibited as
producer, oracle, or guardian.

## Material Probe Boundary

Only after `SEMANTICS_FROZEN` may Loom separately authorize a material probe on
the Mac. The parity receipt must bind at least:

```text
Sounio source hash
Sounio semantics hash
candidate manifest hash
producer_language=C++
producer_role=MATERIAL_PARITY
Apple toolchain identity
exact hardware identity
remote host-key identity
probe source hash
probe binary hash
command hash
environment record
raw result hash
```

Shell or SSH may transport bytes and start an already-authorized command. They
may not parse probe output into expected values or decide feasibility. No
Python, Rust, Node, Ruby, awk, `bc`, or similar disposable program may act as a
semantic or material oracle.

If the Mac is unreachable, the result is a transport failure. It is not
`FALSIFIED_FOR_FROZEN_CANDIDATE_SET` and does not change material identity.

## Mandatory Stage Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

This Garden must be committed before any first executable source is created.
The first executable commit must precede the exact frozen-semantics matcher.
The material probe cannot run until the matcher, semantics, receipt, evidence,
and gate are committed and a fresh Loom `PARITY_EXECUTE` decision permits the
exact command.

`PARITY_OPEN` for a material probe does not imply `CLAIM_READY`. A feasibility
decision still contains no latency value.

## Frozen Parent

The first executable must bind these direct parent artifacts in this exact
order after the committed Garden occupies index zero:

| Index | Artifact | Path | SHA-256 |
| ---: | --- | --- | --- |
| 1 | frozen Apple request source | `stdlib/hardware/pireus/apple_cpu_dependency_latency_request.sio` | `3a92553aae0c4a9606f4964e1613a31452b7fb4d197b128dbb73cc24b87b550e` |
| 2 | frozen Apple request semantics | `docs/research/pireus_apple_cpu_dependency_latency_request_semantics.md` | `9bd767db814e47bfc087e07c0f9ff33b65faea5b885ae0f8ed3a6e646c015e6d` |
| 3 | frozen Apple request receipt | `docs/research/receipts/pireus_apple_cpu_dependency_latency_request_20260828.md` | `0ee12f3502efb26056bdbcf850360c0a5df727627a3c67499d363744f7c73272` |
| 4 | frozen Apple request evidence | `docs/research/evidence/pireus_apple_cpu_dependency_latency_request_20260828.txt` | `cf4455690426038cc7477b673bcf763e9755e8147f1ff55e882086826626482b` |

The protocol is two-phase. This Garden is committed first; its raw-byte SHA-256
is derived only afterward. The first executable then binds the ordered
five-entry parent manifest and live-evaluates the parent Sounio source through
`pireus_apple_cpu_dependency_latency_request_evaluate` plus its exact matcher.
Hash equality without live parent acceptance is insufficient.

The parent evaluation still receives its own 33 transitive inputs in their
frozen order. The child may not replace that evaluation with receipt parsing.

## First Sounio Executable

Only after the Garden commit may these paths be created:

```text
stdlib/hardware/pireus/apple_cpu_dependency_latency_interface_feasibility.sio
examples/pireus_apple_cpu_dependency_latency_interface_feasibility.sio
tests/stdlib/hardware/test_pireus_apple_cpu_dependency_latency_interface_feasibility.sio
```

The first executable must provide:

- distinct typed objects named above;
- a value-free interface-assessment request;
- ordered candidate-family vocabulary;
- total requirement and feasibility predicates;
- named typed refusal reasons;
- a closed-manifest predicate distinct from candidate availability;
- append-ordered ontology projection;
- canonical serialization and digest;
- a finite, named negative table;
- an emitted authority stream;
- no exact frozen-semantics matcher yet.

It must derive its own records, states, verdict, counts, ontology, negative
results, and digests. This Garden supplies none of their expected values.

## Required Negative Surface

The first executable must reject at least these semantic mutations:

1. missing or drifted Garden or direct parent;
2. parent matcher not live or parent result invalid;
3. request other than ID `4`;
4. machine or engine other than the exact Apple material binding;
5. GPU, Metal shader, DGX, Xeon, blueprint, or sibling Mac substitution;
6. subject widened beyond `XOR_SELECTOR`;
7. quantity, unit, scope, statistic, sample count, or warmup changed;
8. timer ticks labelled as CPU cycles;
9. nanoseconds labelled as CPU cycles;
10. elapsed time multiplied by nominal, advertised, sampled, or average
    frequency and labelled as native cycles;
11. event name, event number, symbol, or header treated as event semantics;
12. counter source or domain absent;
13. privilege or entitlement absent but access marked available;
14. multiplexed or scaled counter marked cycle-native;
15. counter width or wrap rule absent;
16. before or after read lacks the admitted ordering contract;
17. migration unobservable or observed but candidate accepted;
18. advisory affinity treated as enforced affinity;
19. fields from several incomplete candidates combined into one candidate;
20. candidate set marked closed without closure evidence;
21. one inaccessible candidate promoted to universal falsification;
22. conflicting evidence resolved silently instead of `INDETERMINATE`;
23. SSH reachability, host label, model string, or locator used as interface
    evidence;
24. transport failure used as a feasibility verdict;
25. environment binding inferred from interface feasibility;
26. execution authorization inferred from environment or feasibility;
27. value, sample, summary, interval, comparison, speedup, ranking, or lowering
    preference inserted;
28. C++ promoted to semantic authority or verdict producer;
29. external review promoted to authority or material evidence;
30. Python or Rust used as producer, oracle, or guardian;
31. parity or claim-ready stage opened before the required receipts;
32. timer or counter interface availability inferred from another Apple or
    AArch64 machine.

Named bundles may contain several related mutations, but a passing bundle
requires every component mutation to be refused. The finite table is not an
independence or completeness theorem. Each named negative must identify the
requirement predicate or stage invariant it is expected to falsify and must be
evaluated as a mutation of a valid positive baseline, so a globally invalid
fixture cannot make the table pass vacuously.

## Acceptance Boundary

The first executable phase is acceptable only when Sounio can represent the
assessment request, candidate families, branching verdict algebra, refusal
reasons, and all closed claim boundaries without contacting the Mac or
producing a cost value.

The later material phase is acceptable only when the frozen classifier receives
a separately authorized, hash-bound material observation and produces a
scoped verdict without changing the request.

This Garden establishes no interface availability, PMU access, entitlement,
counter event, counter width, cycle semantics, affinity guarantee, migration
control, measurement feasibility, environment binding, execution
authorization, latency, throughput, duration, sample, summary, interval,
comparison, speedup, ranking, lowering preference, parity verdict, or
claim-ready result.
