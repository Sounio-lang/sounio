# Loom Spatial Observatory

Status: implementation slice v1, native Apple client validated on macOS 27

Figma source: <https://www.figma.com/design/wVSqacgCCyTRGtltZChNXk>

- Page `06 - Spatial Glass Observatory`: macOS command center
- Page `07 - Material + Motion Spec`: material, motion, and accessibility

## Product role

Loom Spatial is the heavy-use interface for macOS 27 and the review companion
for iOS 27. The web cockpit remains a useful remote read surface. Neither UI is
the owner of session custody, routing outcomes, or semantic truth.

The primary interaction is a spatial operational fabric:

1. inspect quota pools, CLI adapters, models, lanes, and authority;
2. follow one selected route through the fabric;
3. inspect the backend-produced receipt;
4. keep the agent conversation beside the evidence;
5. configure policy inputs without computing the decision in the UI.

## Integration contract

The UI preserves these independent entities:

- `ProviderAccount`
- `QuotaPool`
- `CliAdapter`

The routed flow is:

```text
Task -> RouteDecision -> RouteReceipt
```

Quota state is exactly one of:

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

`RouteReceipt` contains exactly the integration fields:

```text
taskId policy poolId adapterId model effort reason fallbackChain status
```

The native client decodes the existing read-only Loom projections. Backend
code arbitrates. A visual filament, animation, shader, LLM opinion, or parity
artifact cannot create or promote a receipt.

## State gallery

The in-app scenario selector covers:

- nominal exact quota
- estimated quota
- quota exhausted
- CLI broken
- adapter missing
- auth expired
- cooldown
- unknown quota
- model unavailable
- ownership block

Collectively the mocks exercise every wire value of `QuotaState`, `PoolHealth`,
and `AdapterHealth`.

## Material system

The ambient layer is a full-window procedural Metal field. It combines warped
noise, caustic interference, a spectral horizon, and a restrained lattice.
This layer is decoration only and accepts presentation state, never domain
authority.

Operational surfaces use thin native material over the field with:

- 8 pt maximum corner radius
- hairline white edge for physical separation
- cyan for selected authority
- green for healthy/live
- amber for degraded/estimated/cooldown
- red for refused/broken/exhausted/auth/ownership block
- magenta as a secondary spectral signal

The topology is an unframed three-stratum stage for quota pools, CLI adapters,
and models. Its plates use shallow perspective and stable dimensions. The
route filament is a Canvas overlay whose color follows receipt state.

## Motion and accessibility

Motion is informative but non-authoritative:

- ambient field: slow continuous drift
- selected route: directional dash phase
- selected node: low-amplitude pulse
- receipt changes: color/state transition

With Reduce Motion, time is frozen for the Metal field and route pulse. With
Reduce Transparency, glass resolves to an opaque graphite surface. Increased
Contrast strengthens panel boundaries. No status depends on color alone.

## Runtime boundary

Current read path:

```text
Loom Spatial
  -> GET /api/fleet
  -> GET /api/events
  -> GET /api/snapshot
  -> verified/read-only projection
  -> Loom OCaml kernel + Sounio authority
```

Current command path:

```text
Loom Spatial
  -> POST /v1/messages + bearer capability
  -> loopback-only OCaml message bridge
  -> fixed sender identity + bounded destination/request payload
  -> sounio-coord durable message bus
  -> receipt after SENT
  -> GET /v1/threads/<request-id> + bearer capability
  -> request, wake, response, ACK, timeout, durable-only event projection
  -> POST /v1/messages/<response-id>/ack + durable ACK event
```

The command bridge is a separate listener from the fleet read projections. It
never posts directly to a provider CLI or injects a tmux pane. Its authenticated
thread projection derives records only through the coordination runtime's
`outbox`, `inbox`, `message-status`, and `ack` commands; it does not read or
invent bus state itself. Missing or invalid
capabilities, insecure token files, malformed fields, remote binds without an
explicit override, runtime refusal, and runtime timeout all fail closed. Audit
records contain ALLOW/DENY, sender identity, reason, and a receipt digest, but
never the capability or message body.

## Evidence boundary

The validated macOS slice displays real read-only fleet state and selects a
live lane with an active endpoint before falling back to a durable-only target.
The endpoint state and delivery expectation remain visible before submission.
The Conversation tab renders the authenticated correlated thread projection:
request, wake, response, ACK, timeout, and durable-only are distinct durable
events. The composer becomes active only when its separate bridge is configured
and reports durable acceptance independently. Route receipts remain scenario
data and are visibly marked as non-authoritative. Native build, test, runtime,
source-hash parity, screenshot, command-plane, and negative-test evidence are recorded in
`evidence/2026-09-03-spatial-v2-native-gate.txt` and
`evidence/2026-09-03-message-bridge-gate.txt`. The live delivery-aware
round-trip is recorded in
`evidence/2026-09-03-delivery-aware-roundtrip-gate.txt`; thread-truth coverage
is recorded in `evidence/2026-09-03-thread-truth-gate.txt`.
