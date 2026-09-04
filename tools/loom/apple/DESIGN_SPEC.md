# Loom Native Workbench

Status: Beta refinement in progress. The macOS 27 baseline was validated on Apple
Silicon; the current conversation-first refinement awaits a fresh native gate.

Figma source: <https://www.figma.com/design/kXOVfoBejagrX2GdHACg6G>

- Frame `Conversation Space / Native Workbench` (`19:2`): approved primary surface
- Frame `Conversation Space / Native Workbench Beta Review` (`33:137`): protected
  refinement with lane search/scope, backend authority, and linear receipt context

## Product role

Loom Native Workbench is the heavy-use interface for macOS 27 and the review companion
for iOS 27. The web cockpit remains a useful remote read surface. Neither UI is
the owner of session custody, routing outcomes, or semantic truth.

The primary interaction is the Conversation Space. The spatial operational fabric
is a supporting mode, never a prerequisite for speaking naturally with an agent:

1. choose a real, published lane through search and `All / Ready / Watch` scope;
2. continue a natural conversation with a durable transcript and per-lane draft;
3. keep the live delivery state and backend-produced receipt close but quiet;
4. move to spatial evidence or policy configuration only when that changes the
   next decision;
5. configure policy inputs without computing the decision in the UI.

## Conversation contract

Conversation is the primary human interaction, not a routing form. The user
writes naturally to the selected agent with a multiline composer and sees one
continuous chronological transcript for that lane. Each outbound turn still
creates a durable, auditable Loom request, but request IDs, wake state, ACKs,
timeouts, routing choices, and receipts stay progressively disclosed as
operational evidence instead of interrupting the dialogue.

Changing lanes changes conversations without discarding drafts or durable
history. The UI may aggregate correlated request/response threads into the
human transcript, but it must never merge, synthesize, or invent their backend
records. Delivery metadata remains inspectable in the Evidence surface.

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

`RouteReceipt` always contains these routing-identity fields:

```text
taskId policy poolId adapterId model effort reason fallbackChain status
```

It may carry additive, backend-produced provenance fields such as semantic
source hashes, toolchain, hardware, quota observations, and language roles.
Those fields are evidence only: their presence does not give the UI, a parity
language, or an LLM authority to create or promote a routing outcome.

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

Current configuration path:

```text
Loom Spatial Configure tab
  -> GET/PUT /v1/routing/config + bearer capability
  -> loopback-only OCaml message bridge
  -> atomic declarative config revision and digest receipt
  -> private shared Git metadata or deployment-owned private state directory
  -> future backend routing arbiter consumes inputs
```

The configuration document contains only policy, model, effort, pool order,
and adapter order. It is not a `RouteDecision` or `RouteReceipt`; the UI does
not execute a provider, choose a pool, or manufacture evidence. A malformed,
duplicate, unauthorized, or unavailable update is rejected without replacing
the last accepted configuration.

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
The Beta lane rail never replaces an empty live fleet with sample agents; it
shows the explicit empty or non-matching state instead.
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
is recorded in `evidence/2026-09-03-thread-truth-gate.txt`. Declarative
routing configuration persistence and refusal coverage are recorded in
`evidence/2026-09-03-routing-config-gate.txt`.
The visual Beta review is recorded in
`evidence/2026-09-04-native-workbench-beta-refinement-gate.txt`; its native
revalidation remains intentionally unclaimed until the Apple endpoint is
reachable.
