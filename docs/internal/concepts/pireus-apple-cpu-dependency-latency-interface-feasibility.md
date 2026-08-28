<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-apple-cpu-dependency-latency-interface-feasibility
authority: repo_only
audience: users
last_validated: 2026-08-28
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pireus-apple-cpu-dependency-latency-interface-feasibility
-->

# Pireus Apple CPU Interface-Feasibility Assessment Schema

Concept-ID:
`SOUNIO-PIREUS-APPLE-CPU-DEPENDENCY-LATENCY-INTERFACE-FEASIBILITY`

Status: `SEMANTICS_FROZEN`

Canonical surface:
`stdlib/hardware/pireus/apple_cpu_dependency_latency_interface_feasibility.sio`

Frozen semantics SHA-256:

```text
6819916ac4240923a149dd95ee9dcbeaba8d3826b7452dd819e177ff62ce8c7f
```

## Meaning

This concept is the Sounio-owned question and total classifier for whether the
exact `Mac17,7 / Apple M5 Max` material CPU has an admissible interface for the
already frozen request ID `4`: XOR-selector dependency latency in native
executing-core cycles.

It preserves these distinctions:

```text
cycle request != cycle interface
candidate family != concrete API
API name != event semantics
timer tick != CPU cycle
available != unknown != unavailable
partial evidence != terminal refusal
one refusal != closed-manifest falsification
transport failure != material falsification
semantic authority != material parity
interface feasibility != cost observation
semantics frozen != parity open
```

## Frozen Result

The first Sounio result is deliberately value-free:

```text
stage=ASSESSMENT_REQUESTED
verdict=UNASSESSED
families=6
candidates=0
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

The exact matcher was added only after commit `c924d0014c88`, which contains
the first executable without any child expected result.

## Candidate Algebra

The positive families are core-cycle counter, process PMU cycle event, and
system-trace cycle event. Architectural timer ticks and OS monotonic time are
negative controls. A frequency-derived estimate is never native-cycle evidence
in this contract.

The classifier is exclusive and ordered:

1. a complete satisfying candidate yields `FEASIBLE_CYCLE_NATIVE`;
2. otherwise a complete, evidenced, closed terminal set yields
   `FALSIFIED_FOR_FROZEN_CANDIDATE_SET`;
3. otherwise a classified set yields `INDETERMINATE`.

Unavailable privilege, migration incapacity, and cycle-ineligible semantics
have distinct terminal predicates. Unknown, conflicting, or absent evidence is
not silently promoted to any terminal state.

## Closed Manifest

The `v0` executable has six observation slots. A closed material set requires:

```text
one nonzero candidate-manifest reference
one nonzero closure-evidence reference
pairwise-distinct candidate IDs
declared_candidate_count=observed_candidate_count>0
every observed candidate terminal
```

Six is a versioned capacity, not a completeness theorem about macOS. A larger
candidate set requires a new semantic version.

## Authority Boundary

The locator `demetrios@sounio-language-macbook` is routing-only and is absent
from the semantic records and digests. No Mac access occurred in the Garden,
first executable, or freeze stages.

After the exact semantics hash is admitted by Loom, C++ may act only as
`MATERIAL_PARITY`. It may report hash-bound candidate evidence but cannot
change the Sounio requirement, classifier, expected result, or semantic hash.

Lean 4 is `FORMAL_PARITY`, Koka is `EFFECT_PARITY`, Haskell is an optional
denotational baseline, external LLMs are review-only, and Python and Rust are
prohibited producers and oracles.

## Canonical Evidence

The exact parent order, vocabulary, predicates, classifier, ontology, negative
witnesses, serialization schedule, digest words, and open boundaries are in:

`docs/research/pireus_apple_cpu_dependency_latency_interface_feasibility_semantics.md`

The dedicated gate is:

```bash
bash scripts/ci/pireus_apple_cpu_dependency_latency_interface_feasibility.sh
```

## Deliberately Open

- the concrete interface manifest for the exact Mac;
- whether a native cycle candidate exists;
- exact privilege and entitlement outcomes;
- migration, serialization, width, wrap, and anti-multiplexing evidence;
- environment and harness freeze;
- execution authorization;
- every cost, comparison, ranking, lowering choice, parity, and claim.

## Semantic Outcome

Semantic-Outcome: value-free Apple CPU interface-feasibility request and total classifier

Concept-Status-Before: `garden`

Concept-Status-After: `executable`

Distinctions-Added: candidate family versus concrete interface; availability
versus terminal refusal; migration evidence state; manifest closure versus one
candidate refusal; semantic result versus later material observation

Distinctions-Preserved: exact request, target, machine, engine, subject,
quantity, native unit, counter domain, scope, statistic, environment
obligation, producer role, and claim stage

Distinctions-Erased: none

Evidence-Run: first Sounio authority stream, exact post-stream matcher,
dedicated Sounio test, live parent evaluation, 25-triple ontology projection,
32 named negatives, canonical digests, and raw-file tamper gate

Fallback-Path: none; authority execution uses explicit `lean_single` routing

Legacy-Kept: the frozen Apple CPU request and all its transitive parents remain
unchanged and live-evaluated

Conflicting-Lanes: none observed at phase start

Next-Semantic-Interface: hash-bound C++ candidate-manifest and interface probe,
followed by Sounio classification of returned material evidence
