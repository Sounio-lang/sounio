<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-apple-cpu-dependency-latency-request
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pireus-apple-cpu-dependency-latency-request
-->

# Pireus Apple CPU Dependency-Latency Request

Concept-ID: `SOUNIO-PIREUS-APPLE-CPU-DEPENDENCY-LATENCY-REQUEST`

Status: `executable`

Canonical surface:
`stdlib/hardware/pireus/apple_cpu_dependency_latency_request.sio`

## Meaning

This concept is the Sounio-owned composition of the immutable Apple dependency
latency request ID `4` with the exact material Apple CPU identity admitted for
`Mac17,7 / Apple M5 Max`.

It preserves these distinctions:

```text
request template != material binding
blueprint engine != material engine
material machine != transport locator
material identity bound != interface ready
request eligible != measurement feasible
cycle requested != cycle measurement admitted
execution authorized != observation present
material parity != semantic authority
review != authority
```

The request remains value-free and retains its inherited subject, quantity,
unit, scope, statistic, sampling counts, and environment obligation.

## Exact Composition

The binding requires all of the following at once:

1. the committed Garden plus eight direct parents match their raw SHA-256;
2. both executable Sounio parents pass their exact live matchers;
3. ledger index `3` is request ID `4` for Apple XOR-selector dependency
   latency in cycles over a dependency chain;
4. the material machine is exactly `APPLE_MAC17_7`;
5. the material engine is exactly `APPLE_M5_MAX_CPU`, kind CPU, ISA AArch64;
6. the receipt, evidence, hardware, material subject, material operation, and
   dependency-latency eligibility are the exact admitted Apple values;
7. the only accepted transition is `TEMPLATE -> MATERIAL_IDENTITY_BOUND`;
8. interface, feasibility admission, environment binding, authorization,
   value, observation, comparison, parity, and claim readiness remain closed.

The transport locator is not a binding field and does not appear in the result
digest. `demetrios@sounio-language-macbook` is routing metadata only.

## Frozen Semantics

The exact parent order, records, stage predicates, 26 ontology triples, 28
negative witnesses, 20 boundary flags, digest schedules, result words, and
authority boundary are frozen in:

`docs/research/pireus_apple_cpu_dependency_latency_request_semantics.md`

The dedicated gate is:

```bash
bash scripts/ci/pireus_apple_cpu_dependency_latency_request.sh
```

## Deliberately Open

- whether an admitted Apple interface can measure dependency latency in
  `CYCLE` without an invalid conversion;
- how the eleven required environment bits will be bound and evidenced;
- the later Loom execution-authorization receipt;
- the eventual observed interval and its uncertainty;
- any target-local comparison or lowering choice.

A later interface lane may set measurement feasibility to `FALSIFIED`. The
frozen request is not silently converted to nanoseconds or another quantity.

## Claims Forbidden

- that the Mac was contacted or measured in this lane;
- that a measurement interface is ready;
- that `CYCLE` is already feasible on Apple Silicon;
- that the environment is bound;
- any cost, latency, throughput, comparison, ranking, or speedup;
- any cross-target equivalence or lowering preference;
- parity or claim-ready promotion;
- external LLM confirmation;
- Python or Rust authority, oracle, or guardian use.

## Semantic Outcome

Semantic-Outcome: immutable request-template plus receipt-bound material CPU identity

Concept-Status-Before: `garden`

Concept-Status-After: `executable`

Distinctions-Added: template reference versus material binding; material
identity versus measurement interface; identity-bound state versus execution
authorization

Distinctions-Preserved: target, machine, engine, blueprint, engine kind, ISA,
receipt, evidence, hardware, subject, operation, eligibility, quantity, unit,
scope, statistic, environment, producer role, and claim stage

Distinctions-Erased: none

Evidence-Run: first Sounio authority stream, exact post-stream matcher,
dedicated Sounio test, live parent evaluation, ordered ontology projection,
canonical digests, and finite negative witnesses

Fallback-Path: none; the frozen executable uses explicit `lean_single` routing

Legacy-Kept: target-cost request ID `4`, the material-admission overlay, and
their exact matchers remain unchanged frozen parents

Conflicting-Lanes: none observed at phase start

Next-Semantic-Interface: Apple CPU dependency-latency measurement-interface
feasibility and environment-binding contract, followed by a separately
authorized material execution
