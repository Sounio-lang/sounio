# GARDEN: LOOM Routing Authority V1

Status: preregistered
Action: `9032`
Semantic authority: Sounio

## Question

Can LOOM turn a frozen routing configuration and positively observed runtime
facts into one authorized provider execution without allowing the UI, OCaml
realization, provider CLI, or an external LLM to invent the selected route or
its expected result?

## Authority Order

This action obeys the mandatory progression:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY`

Sounio creates the first executable decision table and expected results. OCaml
may realize those frozen decisions operationally. Lean 4, Koka, C++, and
Haskell may only compare, prove, or measure after the Sounio semantics hash is
frozen. External LLMs are review-only. Python and Rust are prohibited.

## Route Law

The configured route is an ordered, bounded candidate chain. For each
candidate, action `9032` receives positive observations for:

- exact task, configuration, candidate, and predecessor-chain bindings;
- ownership availability;
- quota state and quota-pool health;
- adapter health and resolved implementation language;
- model availability;
- observation authority and freshness.

`PLAN` allows only the first admissible candidate. Candidate zero has an empty
predecessor chain. A later candidate additionally requires a bound receipt that
every earlier candidate was denied by action `9032`. Silence, missing data,
timeout, stale observation, or an unbound predecessor is not exhaustion.

Quota state is exactly one of `exact`, `estimated`, or `unknown`. Unknown quota
never authorizes execution. Estimated quota authorizes only when the frozen
policy explicitly permits it. Pool health and adapter health remain distinct.

## Dispatch Law

`DISPATCH` reevaluates the selected candidate and additionally requires:

- a bound, unexpired action-`9032` PLAN decision;
- an exact provider-native argv plan digest;
- an allowed frozen execution-grant parent;
- a result-receipt destination bound before launch;
- OCaml classified only as `OPERATIONAL_REALIZATION`.

The provider process must be launched through the existing Provider ABI and
Guardian custody. No shell evaluates the provider argv. The dispatch decision
does not promote provider output or an LLM opinion to semantic authority.

## Receipt Law

Every operational receipt exposes at least:

`taskId, policy, poolId, adapterId, model, effort, reason, fallbackChain, status`

and binds the Sounio source hash, frozen semantics hash, producing language and
role, toolchain, hardware, command, and result. A `running` receipt proves only
authorized launch and custody. A terminal receipt additionally requires the
existing execution-outcome authority; process exit alone cannot manufacture
semantic success.

## Stable Decisions

`0` is ALLOW. Denials are stable and fail closed:

| Code | Reason |
| ---: | --- |
| 601 | policy missing |
| 602 | policy timeout |
| 603 | policy error |
| 604 | wrong stage or operation |
| 605 | malformed semantic field |
| 606 | authority binding missing |
| 607 | ownership block |
| 608 | quota unknown |
| 609 | quota estimate forbidden |
| 610 | quota pool degraded |
| 611 | quota exhausted |
| 612 | quota authentication required |
| 613 | adapter broken |
| 614 | adapter missing |
| 615 | adapter authentication required |
| 616 | model unavailable |
| 617 | forbidden implementation language or role |
| 618 | predecessor chain not closed |
| 619 | plan decision missing, stale, or mismatched |
| 620 | provider plan not bound |
| 621 | execution grant missing |
| 622 | receipt destination not bound |
| 623 | review or parity promoted to authority |

## Required Negative Tests

The executable gate must deliberately test policy absence, timeout, error,
malformed fields, ownership conflict, unknown quota, forbidden estimated quota,
every pool and adapter failure state, model absence, fallback-chain bypass,
unbound or stale PLAN decisions, missing provider plan, missing ExecGrant,
missing receipt binding, LLM promotion, and resolved Python and Rust adapters.
A causal sabotage build must remove one decisive rule and demonstrate that the
unchanged negative frame becomes admitted.

## Nonclaims

Preregistration does not freeze semantics, open parity, launch a provider,
prove billing quota, establish model entitlement, or make the current UI an
operational router. A mock RouteReceipt remains non-authoritative.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: loom-routing-authority-v1
Owner: codex-1
Concept-IDs: SOUNIO-LOOM-ROUTING-AUTHORITY
Intent-Preserved: Sounio creates route semantics and expected results before operational realization
Transformation: frozen configuration plus positive observations to PLAN or DISPATCH decision
Types-Changed: none
Effects-Changed: provider execution becomes conditional on frozen action 9032
IR-Changed: none
Claims-Introduced: a RouteReceipt can prove Sounio-authorized provider launch
Claims-Forbidden: UI authority, OCaml semantic authority, quota inference, LLM result confirmation
Assumptions: existing Provider ABI and execution-grant authority remain valid parents
Write-Set: action 9032 source, adapter, freeze, gates, operational realization, Apple projection
Read-Set: language authority, Provider ABI, execution grant, execution outcome, routing UI contract
Positive-Witness: exact-quota healthy pool, healthy adapter, available model, ownership clear
Negative-Witness: deliberate Python-oracle dispatch attempt
Acceptance-Gate: scripts/ci/sounio_loom_routing_authority_selftest.sh
Integration-Target: LOOM message/runtime bridge and SwiftUI RouteReceipt projection
Authoritative-Only-If: action 9032 semantics are frozen and every operational receipt binds that hash
```
