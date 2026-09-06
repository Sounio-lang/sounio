<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-28-pireus-apple-cpu-dependency-latency-request
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-28-pireus-apple-cpu-dependency-latency-request
-->

# Garden Seed: Pireus Apple CPU Dependency-Latency Request

> **Status**: Garden | **Date**: 2026-08-28 | **Authority**: founder

Concept-ID: `SOUNIO-PIREUS-APPLE-CPU-DEPENDENCY-LATENCY-REQUEST`

Semantic-Lane-ID: `pireus-apple-cpu-dependency-latency-request-20260828`

The Concept-ID is the durable semantic identity. The lane ID names this
implementation episode, and the `topic_id` in metadata is its docs-governance
projection; neither may replace the Concept-ID in a receipt.

## Butterfly

> "mac is sounio-language-macbook na tailnet"

Pireus now has two different pieces of Sounio authority that were intentionally
kept apart:

- a value-free Apple selector request bound to a CPU blueprint;
- a material admission for the observed `Mac17,7` machine and its Apple M5 Max
  CPU execution engine.

The next move is not to benchmark the Mac. It is to state, in Sounio, whether
that already-existing abstract request may be composed with that
already-admitted material engine without changing either frozen parent.

## Question

How can Pireus bind the frozen Apple selector dependency-latency request to an
admitted material engine by composition, rather than mutation, while refusing
every accidental change of target, subject, quantity, unit, scope, statistic,
sampling contract, environment obligation, authority role, or stage?

## The Four Distinct Objects

The first child must keep these objects distinct:

```text
request template
material-identity-bound request
execution authorization
material observation
```

A material-identity-bound request is still not an execution authorization. An
execution authorization is still not an observation. An observation is still
not a comparison or a lowering preference.

In particular:

```text
material engine admitted != remote launch authorized
remote locator known != key availability observed
request environment required != environment bound
request exists != sample exists
```

## Frozen Parent Request

The target-cost parent owns request ID `4`. Its semantic coordinates are:

```text
target=APPLE_SILICON
machine=UNBOUND
engine=APPLE_CPU_BLUEPRINT
engine_is_blueprint=true
engine_observed=false
subject_kind=LOWERING_NODE
subject_node=XOR_SELECTOR
quantity=DEPENDENCY_LATENCY
unit=CYCLE
scope=DEPENDENCY_CHAIN
statistic=MEDIAN
requested_sample_count=1001
requested_warmup_count=128
required_environment_mask=2047
bound_environment_mask=0
evidence_role=MEASUREMENT_REQUEST
producer_language=Sounio
producer_role=SEMANTIC_AUTHORITY
value_present=false
```

This display is an orientation, not a second source of truth. The first child
must obtain every field by live Sounio evaluation of the hash-pinned parent and
must never parse this document to construct a request. If the display and the
parent ever disagree, the parent wins and governance drift is reported. This
Garden assigns no child result.

## Frozen Material Admission

The material-engine admission parent owns the Apple material coordinate:

```text
transport_locator=demetrios@sounio-language-macbook
tailnet_identity=sounio-language-macbook
hostname=Sounio-Language-MacBook
machine_model=Mac17,7
machine=APPLE_MAC17_7
engine=APPLE_M5_MAX_CPU
engine_kind=CPU
isa=AArch64
interface=ABSENT
material_subject=XOR_SELECTOR
```

This display is likewise non-normative; the hash-pinned admission parent and
its live exact matcher own the coordinate. The locator is a routing coordinate.
It does not prove that an SSH key is present, that the host is reachable now,
or that the request environment is bound. The binding predicate is
definitionally forbidden to read the locator.

## The Only Admitted Composition

The first Sounio child must not rewrite request ID `4`. It constructs a separate
binding record with, at minimum:

```text
template_request_id=4
material_machine=PIREUS_MACHINE_APPLE_MAC17_7
material_engine=PIREUS_ENGINE_CPU_APPLE_M5_MAX
material_receipt=PIREUS_RECEIPT_APPLE_A64_TBL
material_evidence=PIREUS_EVIDENCE_APPLE_A64_TBL
eligibility=PIREUS_ELIGIBILITY_APPLE_LATENCY
state=MATERIAL_IDENTITY_BOUND
execution_authorization_id=ABSENT
```

The request remains an immutable parent. A derived view may expose the material
machine and engine, but every experimental field is projected from request ID
`4`, never copied into independently mutable storage.

The target-cost parent's `engine_observed` field describes whether the engine
inside that older request record was already an observed engine rather than a
blueprint. It remains a parent fact and is not reused as a cost-observation
flag. The new binding uses the distinct predicate
`engine_identity_materially_admitted`; `material_cost_observed` remains false.

The composition is admitted only when named total predicates over the two live
parent results prove all of these at once. The first executable must provide
these signatures or stricter equivalents using distinct, non-aliasing types:

```text
request_template_exact(PireusRequestTemplateRef) -> bool
material_instance_exact(PireusMaterialEngineBinding) -> bool
request_target_matches_instance(template, binding) -> bool
request_subject_matches_receipt(template, binding) -> bool
request_quantity_is_eligible(template, binding) -> bool
material_identity_binding_valid(template, binding) -> bool
stage_allows_identity_binding(PireusRequestStage) -> bool
```

Together they require:

```text
target matches the request target
machine belongs to that target
machine contains the engine
engine kind is CPU
engine implements AArch64
engine material subject is XOR selector
engine carries the Apple material receipt
engine has dependency-latency eligibility
material parent is exact and valid
```

No label, hostname, vendor family, architecture name, interface, or nearby
resource may substitute for this complete coordinate.

`PIREUS_MACHINE_APPLE_MAC17_7` is the instance IRI minted by the
material-admission parent for the one observed host bound to its hardware
identity and receipt. It is not the class of every machine whose model string
is `Mac17,7`. The composition requires that exact instance IRI plus its hardware
binding; neither the model string nor the locator may select a sibling machine.

The child must obtain these IRIs from
`pireus_material_engine_apple_candidate`, validate the candidate with
`pireus_material_engine_candidate_valid`, and validate the quantity through
`pireus_material_engine_request_eligible`. It may not reimplement those parent
predicates from labels or constants copied out of this Garden.

Dependency-latency eligibility means only that this request shape may be bound
to this engine. It does not assert that a counter, timer, harness, or measurement
interface is ready. The frozen Apple coordinate has no admitted interface, so
`measurement_interface_ready=false` and `execution_authorization_id=ABSENT` are
mandatory in this child. `measurement_interface_ready` is a projection derived
from the absence of an admitted interface receipt and is not stored as a free
field; the absent execution-authorization ID is part of the binding record. A
later Garden must name and admit the actual measurement interface before any
launch.

The frozen request's `CYCLE` unit may prove infeasible on this target. This lane
therefore preserves `measurement_feasibility=UNKNOWN`. The later interface lane
may admit a cycle-native observation contract or falsify the request as
unmeasurable. It may not silently substitute timer ticks, nanoseconds, estimated
cycles, or a frequency conversion.

## Request State

The child must define these exact closed state tags; they may not be renamed or
collapsed in the first executable:

```text
TEMPLATE
-> MATERIAL_IDENTITY_BOUND
-> EXECUTION_AUTHORIZED
-> OBSERVED
-> SUMMARY_ELIGIBLE
-> COMPARISON_ELIGIBLE
```

This Garden opens only `TEMPLATE -> MATERIAL_IDENTITY_BOUND`. Every boundary
flag below is a pure projection of the state and binding record, never an
independently mutable boolean:

The following remain false:

```text
execution_authorized
environment_bound
value_present
material_observation_present
derived_summary_present
comparison_present
speedup_present
ranking_present
parity_open
claim_ready
```

The required environment vocabulary and its eleven bits remain owned by the
hash-pinned target-cost parent: hardware, toolchain, binary, command, workload,
OS or driver, threads, affinity, frequency, power, and timer or counter. This
child preserves both masks exactly, including `bound_environment_mask=0`, and
refuses any attempt to infer environment binding. `MEDIAN`, `1001`, `128`, and
`CYCLE` are inherited request parameters, not claims of invariance across Apple
core classes, DVFS, thermal state, or migration; those confounders remain later
environment obligations.
The parent does not yet specify sample scheduling, chain construction, core
class, or dispersion reporting. Missing protocol fields remain unknown and may
force a later typed refusal; they are not filled from convention.

## Producer And Authority Roles

Sounio remains `SEMANTIC_AUTHORITY`. The first executable request and its
expected behavior must be born in Sounio.

C++ may later act as `MATERIAL_PARITY`, but only after a frozen Sounio request
exists and Loom separately authorizes the execution stage. C++ cannot change
the request meaning or create its expected value.

Lean 4, Koka, and optional Haskell remain parity surfaces. External LLMs remain
`REVIEW_ONLY`. Python and Rust are forbidden as producer, oracle, or guardian
at every stage.

## Mandatory Stage Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

This Garden must be committed before the first child source is created. The
first executable commit must precede its exact frozen-semantics matcher. No
remote Apple command may run in this lane before `SEMANTICS_FROZEN` and a new
Loom execution authorization.

## Frozen Parents

The first child must consume these Sounio authority artifacts in a declared,
ordered parent manifest:

| Parent | Path | Role | SHA-256 |
| --- | --- | --- | --- |
| target-cost source | `stdlib/hardware/pireus/target_cost_observation.sio` | owns request ID 4 and cost vocabulary | `7ea2815c112b85476fc6ac4d8bb9388ee032062822c6905485c2084ee416d6bc` |
| target-cost semantics | `docs/research/pireus_target_cost_observation_semantics.md` | freezes request meaning and value-free boundary | `0a899be7cd25375c8c444b9e1f0a71dd102ca8958072a4290073ae21c926a199` |
| target-cost receipt | `docs/research/receipts/pireus_target_cost_observation_20260827.md` | freezes the seven-request ledger | `b7577c782a82431eb54312137a52c1811f14316d5e5a5fb9e7aa9581f1c304ca` |
| target-cost evidence | `docs/research/evidence/pireus_target_cost_observation_20260827.txt` | flat Sounio authority record | `06f21108ddc89c8c468097b7cefec6a766bda065918fd5797be194878371577b` |
| material-admission source | `stdlib/hardware/pireus/material_engine_admission.sio` | owns exact material candidates and eligibility | `b98d799fec6452f9afaecc7e418578cc0ca72a1d3e7c0a1f84d474e0aa2730ec` |
| material-admission semantics | `docs/research/pireus_material_engine_admission_semantics.md` | freezes identity and request-eligibility meaning | `bbbd7bc9c99e1de46a5317d99a7893adebfdf7fb46454cbd7002b3098281b6ee` |
| material-admission receipt | `docs/research/receipts/pireus_material_engine_admission_20260828.md` | freezes the observed Apple CPU coordinate | `352ad87a4d05ef32333acb2f31a48f49e7d520ab99c3e99a8b4a3e6bf3f55aa4` |
| material-admission evidence | `docs/research/evidence/pireus_material_engine_admission_20260828.txt` | flat Sounio authority record | `08a6e0d4b2e5a0bb4e981cfcc8df5192d9b9b77f53eb3862c86f29d9409e425b` |

The protocol is explicitly two-phase. This Garden is committed first; that
commit determines its raw-byte SHA-256. Only then may the first child hard-code
the ordered nine-entry manifest: committed Garden at index zero, followed by
the eight rows above in table order. The child refuses an uncommitted or
unhashed Garden. SHA-256 covers the raw bytes of each repository file. The child
must live-evaluate both Sounio source parents through
`pireus_target_cost_observation_evaluate` plus its exact matcher and
`pireus_material_engine_admission_evaluate` plus its exact matcher, in addition
to checking all nine file hashes. Its receipt must also pin the wrapper and
compiler bytes that performed that evaluation. Parent drift is a typed refusal,
never a warning followed by a fallback.

## First Sounio Executable

The first child must use the existing Pireus ontology machinery and the
Concept-ID-derived stem in:

```text
stdlib/hardware/pireus/apple_cpu_dependency_latency_request.sio
examples/pireus_apple_cpu_dependency_latency_request.sio
tests/stdlib/hardware/test_pireus_apple_cpu_dependency_latency_request.sio
```

It should define:

- distinct immutable `PireusRequestTemplateRef` and
  `PireusMaterialEngineBinding` types with no representation alias;
- an exact parent-path record;
- named total predicates for request projection, material coordinate,
  eligibility, binding, and stage;
- an absent-by-default Loom execution-authorization slot;
- a value-free evaluator;
- an append-ordered ontology projection;
- canonical integer, boolean, string, and digest serialization;
- named negative witnesses;
- typed errors for parent, request, material coordinate, authority, and stage;
- an executable stream suitable for later semantic freezing.

The negative witnesses must be a finite, named table with stable order. Its
basis is every single mutation in the list below plus these exact pair products:
machine+receipt, engine+evidence, target+engine, subject+eligibility,
unit+feasibility, value+bound-environment, summary+observation, and
stage+execution-authorization. No unbounded Cartesian-product claim is made.

The first executable must derive its own counts, masks, digests, and output. No
expected result, hash, count, or digest is assigned in this Garden.

## Required Negative Surface

The child must reject at least these mutation classes:

1. missing Garden or drifted parent;
2. request other than frozen ID `4`;
3. target changed from Apple Silicon;
4. subject changed from XOR selector;
5. latency changed to throughput or another quantity;
6. cycle changed to another unit;
7. dependency chain changed to another scope;
8. median, sample count, or warmup count changed;
9. required or bound environment mask changed;
10. value, interval, sample, summary, comparison, speedup, or ranking inserted;
11. Apple request composed with DGX, Xeon, GPU, Metal, or a blueprint;
12. correct engine paired with a different machine, receipt, or evidence;
13. `.48` unresolved identity used as a material coordinate;
14. locator reachability or key availability inferred without evidence;
15. C++ promoted to semantic authority;
16. external review promoted to producer or authority;
17. Python or Rust used as producer, oracle, or guardian;
18. measurement interface, feasibility, or execution authorization inferred
    from eligibility;
19. parent `engine_observed` laundered into a material cost observation;
20. execution, parity, or claim stage opened prematurely.

Named bundles may contain several related field mutations, but a passing bundle
requires every component mutation to be refused.

## Acceptance Boundary

The first child is acceptable only when its finite Sounio evaluator accepts the
exact parent composition and keeps the request value-free,
measurement-interface-unready, and execution-closed. The exact obligations and
negative table become machine-checkable only in that post-Garden executable;
this Garden claims no independence or completeness theorem. Later stages may
compare a material observation only after that observation carries the complete
required environment and its own frozen receipt.

This Garden establishes no cycle count, latency, throughput, duration, sample,
interval, binary, command, toolchain, timer, frequency state, power state,
remote reachability, cross-target comparison, lowering preference, speedup,
ranking, parity verdict, or claim-ready result.
