<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-target-cost-observation
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pireus-target-cost-observation
-->

# Pireus Target Cost Observation

Concept-ID: `SOUNIO-PIREUS-TARGET-COST-OBSERVATION`

Status: `SEMANTICS_FROZEN`

Semantic-Lane-ID: `pireus-target-cost-observation-20260827`

## Intent

Represent cost as a target-local, engine-bound observation contract rather
than an untyped number. Static instruction sites, executed instructions,
micro-operations, latency, throughput, duration, transfer, energy, frequency,
and occupancy remain different quantities.

The semantic producer is Sounio:

```text
stdlib/hardware/pireus/target_cost_observation.sio
examples/pireus_target_cost_observation.sio
tests/stdlib/hardware/test_pireus_target_cost_observation.sio
```

## Causal Order

```text
GARDEN commit=de9e1b4d1cb818a0cc1af7cf710e7f31a347211d
SOUNIO_EXECUTABLE commit=ad38229c7c7d21318e6a445fe5c078b5c72f49be
SEMANTICS_FROZEN=enclosing Git commit
PARITY_OPEN=false
CLAIM_READY=false
```

The first executable commit contains no expected result digest or frozen
matcher. Those were written only after the post-Garden Sounio stream existed.

## Observation Key

The meaning of a record is the complete key:

```text
CostObservation<
  target, machine, engine, subject, quantity, unit,
  scope, statistic, environment, evidence
>
```

Dropping an axis changes the claim. In particular:

```text
canonical target != observed machine != observed execution engine
static site count != executed instruction count != executed micro-op count
dependency latency != reciprocal throughput != elapsed duration
same unit != comparable experiment
```

## First Ledger

The frozen child emits seven measurement requests and no material values:

| Target | Engine coordinate | Subject | Quantity | Unit | Scope |
| --- | --- | --- | --- | --- | --- |
| Darwin Xeon | observed T560 CPU engine | selector node | dependency latency | cycle | dependency chain |
| Darwin Xeon | observed T560 CPU engine | selector node | reciprocal throughput | cycle | independent stream |
| Darwin Xeon | observed T560 CPU engine | complete operation | elapsed duration | nanosecond | complete operation |
| Apple Silicon | CPU blueprint | selector node | dependency latency | cycle | dependency chain |
| Apple Silicon | CPU blueprint | selector node | reciprocal throughput | cycle | independent stream |
| DGX | GPU blueprint | selector node | dependency latency | cycle | dependency chain |
| DGX | GPU blueprint | selector node | reciprocal throughput | cycle | independent stream |

Every request asks for median over `1001` samples after `128` warmups and
requires all eleven environment bindings. These are protocol requirements,
not observations: every request has `bound_environment=0` and
`value_present=false`.

Apple and DGX remain canonical blueprints because this lane has no material
engine receipt for either. The Mac transport locator is
`demetrios@sounio-language-macbook` on the tailnet, but a locator does not
promote a blueprint to an observed engine. Likewise, DGX addresses do not
become material identity without a sealed receipt.

## Rational Record Boundary

Observed values, when a later material lane exists, use non-negative rational
components bounded by `10^9`. Therefore every cross-product used for interval
ordering is at most `10^18`, below signed `i64` capacity. Positive denominators
and the order `lower <= estimate <= upper` are mandatory.

Exact rational transport does not remove timer, device, sampling, frequency,
or prior-capture error. Those remain evidence obligations.

## Parent Closure

The child hashes eleven files: the Cost Garden plus ten frozen admission,
engine, and operation artifacts. It also live-evaluates the selector admission
and XOR operation matchers and queries the execution-engine ontology.

The frozen live engine projection contains five Darwin Xeon CPU engines, four
Darwin NVIDIA GPU engines, nine Darwin observed engines total, and zero Apple
or DGX observed engines. Each canonical target retains two blueprints.

## Closed Claims

This freeze establishes no:

- material cost observation or benchmark result;
- measured sample, latency, throughput, duration, energy, or occupancy;
- Apple or DGX observed execution engine;
- cross-target comparability, speedup, ranking, or winner;
- compiler lowering choice or minimum instruction sequence;
- Lean, Koka, C++, or Haskell parity for the cost ontology;
- semantic authority for an external LLM;
- `PARITY_OPEN` or `CLAIM_READY` state.

The external Loom guardian remains the only stage and producer-language
authority.
