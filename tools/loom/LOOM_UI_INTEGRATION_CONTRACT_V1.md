# Loom UI Integration Contract V1

## Status

This document is the product-client contract for Loom routing visibility and
configuration. It governs native clients first; the web cockpit may implement
the same read model for remote, infrequent operation.

It does not define routing semantics. The routing backend remains the only
component that arbitrates a task, selects a provider, spends quota, or creates
a receipt.

## Ownership boundary

| Component | Owns | Must not own |
| --- | --- | --- |
| Routing backend | policy evaluation, provider/model selection, fallback, quota accounting, `RouteDecision`, `RouteReceipt` | presentation state |
| Client UI | visualization, filtering, explicit configuration requests, user-readable explanation | arbitration, quota mutation, synthetic receipts |
| `CliAdapter` | provider-specific invocation and capability reporting | account identity, shared quota policy |
| `ProviderAccount` | provider account identity and authentication binding | adapter health or quota aggregation |
| `QuotaPool` | a separately identified quota allocation and its measurement confidence | CLI invocation or account authentication |

`ProviderAccount`, `QuotaPool`, and `CliAdapter` are separate entities. A UI
must never merge them into one provider-health badge: an account can require
authentication while its adapter is healthy, and a healthy adapter can target
an exhausted pool.

## Route data model

The backend owns the following causal sequence:

```text
Task -> RouteDecision -> RouteReceipt
```

`Task` is the requested unit of work. `RouteDecision` is the backend's current
or final arbitration result. `RouteReceipt` is the immutable record for a
decision that was attempted or completed. A client may render all three, but
may not create a receipt or infer a decision from local UI state.

Every `RouteReceipt` contains exactly these fields:

| Field | Meaning |
| --- | --- |
| `taskId` | stable task identity |
| `policy` | routing policy evaluated by the backend |
| `poolId` | selected quota pool identity |
| `adapterId` | selected CLI adapter identity |
| `model` | selected model identifier |
| `effort` | backend-selected effort class |
| `reason` | bounded, user-readable routing explanation |
| `fallbackChain` | ordered alternatives considered or attempted |
| `status` | receipt lifecycle/outcome status |

## Independent state vocabularies

Quota measurement is exactly one of:

```text
exact | estimated | unknown
```

Pool health is exactly one of:

```text
healthy | degraded | exhausted | auth_required
```

Adapter health is exactly one of:

```text
healthy | broken | missing | auth_required
```

These values are intentionally not interchangeable. For example, `unknown`
quota is not `exhausted`; `adapter.auth_required` is not
`pool.auth_required`; and an adapter marked `broken` must not be represented as
a model-capacity or quota failure.

## Required UI states

The UI must render distinct, explainable states for:

1. quota exhausted;
2. CLI adapter broken;
3. expired or otherwise required authentication;
4. cooldown in effect;
5. unknown quota measurement;
6. model unavailable;
7. ownership block.

An ownership block shows the current owner and the protected surface where that
fact is authorized. It must not offer a competing write action. A cooldown
shows its scope and expiration when supplied by the backend, rather than
claiming a retry will succeed. Model unavailability and an exhausted quota pool
must remain visually distinct even when both lead to a fallback.

## Client behavior

Clients may:

- display backend facts and receipt history;
- let a user request an allowed configuration change;
- show the fallback chain, its reason, and authoritative owner;
- retain a stale last-known view with its observation timestamp.

Clients must:

- fail closed when route data or authority is absent;
- label stale or unknown facts as such;
- keep configuration requests separate from applied routing decisions;
- require the backend to echo the resulting `RouteDecision` and
  `RouteReceipt` before presenting a change as applied.

Clients must not:

- select a provider/model as an authoritative outcome;
- decrement, estimate, or repair quota locally;
- collapse account, pool, and adapter state;
- mint a `RouteReceipt` or promote a local configuration request into one.

## Mock-state acceptance matrix

Every native and web client implementation must include deterministic mocks for
each required UI state, plus the crossed states below:

| Scenario | Pool health | Quota measurement | Adapter health | Expected presentation |
| --- | --- | --- | --- | --- |
| Capacity available | `healthy` | `exact` | `healthy` | selected route with receipt details |
| Quota exhausted | `exhausted` | `exact` | `healthy` | quota-specific fallback or block |
| Quota uncertain | `degraded` | `unknown` | `healthy` | uncertainty warning; no local estimate |
| CLI broken | `healthy` | `exact` | `broken` | adapter-specific fallback or block |
| CLI absent | `healthy` | `estimated` | `missing` | installation/capability absence, not auth failure |
| Authentication expired | `auth_required` | `unknown` | `auth_required` | account/pool and adapter causes shown separately |
| Cooldown | `degraded` | `estimated` | `healthy` | retry time and scope from backend |
| Model unavailable | `healthy` | `exact` | `healthy` | model-specific fallback chain |
| Ownership block | `healthy` | `exact` | `healthy` | owner, protected surface, no competing control |

The mock suite is a UI acceptance gate, not an alternate routing engine.

## Platform direction

The heavy-use client is SwiftUI for macOS and iOS. The macOS client is the
primary operational surface; iOS is for monitoring, notifications, and bounded
approvals. The web cockpit is a remote and infrequent-operation client. All
three consume the same backend-owned route and receipt model.

## Experience acceptance bar

Loom is an operational instrument, not an admin form. Visual quality and
ergonomics are acceptance criteria alongside correct state rendering.

- **macOS:** give heavy users a calm, dense, spatial workspace: a route graph,
  a receipt/detail inspector, and a compact action surface must remain
  simultaneously legible without forcing serial modal navigation.
- **iOS:** preserve the decision, its reason, authority, and safe action in one
  focused flow. It is a companion for monitoring and bounded approval, not a
  shrunken desktop cockpit.
- **Visual semantics:** pool health, adapter health, quota confidence,
  cooldown, and ownership are separate visual channels. Color supports the
  label; it must never be the only distinction.
- **Explanatory graphics:** route graph and fallback-chain visualizations must
  show why a route was selected or refused, not merely decorate a dashboard.
- **Operational calm:** maintain stable layout while data refreshes; avoid
  alert noise, decorative status motion, ambiguous disabled controls, and
  interactions that hide the authoritative reason.
- **Accessibility:** support dynamic type, keyboard operation on macOS, Voice
  Over labels, sufficient contrast, and reduced-motion behavior.

Before implementation, the Figma source must provide desktop and iPhone flows
for every mock-state scenario. Before release, compare implemented SwiftUI
screens against those flows at desktop and mobile viewport/device sizes; visual
review supplements, but never replaces, the route and receipt acceptance gate.
