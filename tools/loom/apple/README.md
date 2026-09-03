# Loom Spatial for Apple platforms

This package is the native macOS 27 and iOS 27 client for Loom. It is the
heavy-use visualization and configuration surface over the existing Loom
kernel. It does not own PTYs, routing authority, receipts, or session custody.

The first slice connects read-only to `bin/loom serve`:

```sh
bin/loom serve --bind 127.0.0.1 --port 8787
```

Open `Package.swift` in Xcode 27 and run the `LoomSpatial` scheme. The default
endpoint is `http://127.0.0.1:8787`. Without a reachable kernel the UI remains
fully inspectable through its deterministic scenario gallery.

The conversation composer uses a separate, authenticated command plane. Start
the source-built bridge with a private bearer capability and a fixed sender:

```sh
bin/sounio-loom message-serve \
  --cwd "$PWD" --bind 127.0.0.1 --port 8789 \
  --token-file /private/path/loom-message.cap \
  --agent founder-ui --lane loom-apple
```

Then launch the native client with the capability file, which must remain
outside the repository and mode `0600`:

```sh
tools/loom/apple/launch-macos.sh \
  --message-url http://127.0.0.1:8789 \
  --message-token-file /private/path/loom-message.cap
```

For a native development launch from Terminal, build an application bundle and
open it through LaunchServices:

```sh
tools/loom/apple/launch-macos.sh
```

Visual QA can request a snapshot produced by the Loom process itself. This
captures only its own content view and does not require global Screen Recording
access:

```sh
tools/loom/apple/launch-macos.sh --snapshot /tmp/loom-spatial.png
```

On an Apple runner, the package gate is:

```sh
tools/loom/apple/validate-apple.sh
```

## Boundaries

- `LoomDomain` contains the routing visualization contract and mock gallery.
- `LoomSpatial` contains SwiftUI, the Metal ambient field, topology, and agent
  conversation dock.
- `/api/fleet`, `/api/events`, and `/api/snapshot` remain read-only projections.
- `POST /v1/messages`, `GET /v1/threads`, `GET /v1/threads/<request-id>`, and
  `POST /v1/messages/<response-id>/ack` exist only on the separate loopback
  message bridge.
- `GET /v1/routing/config` and `PUT /v1/routing/config` use that same
  authenticated loopback bridge. They store a bounded declarative policy,
  model, effort, and pool/adapter order with an atomic revision and digest
  receipt. The default state directory is private shared Git metadata;
  `--routing-state-dir /absolute/private/path` selects an explicit
  deployment-owned location.
- The bridge fixes sender identity at startup; the UI supplies only destination,
  message text, and the bounded `request` kind.
- A send is displayed as successful only after a durable bus receipt is decoded.
- The Conversation tab is an authenticated, correlated projection of durable
  bus records. It renders request, wake, response, acknowledgement, timeout,
  and durable-only events rather than a locally invented transcript. The UI
  issues an ACK only through the bridge and then reloads the resulting durable
  ACK event.
- Initial selection prefers a live lane with an active delivery endpoint. A
  stale or missing endpoint remains addressable through the durable bus, but is
  labeled `DURABLE ONLY` before submission.
- `ProviderAccount`, `QuotaPool`, and `CliAdapter` are deliberately independent.
- The UI renders `Task -> RouteDecision -> RouteReceipt`; the backend arbitrates.
- The Configure tab only edits `loom-routing-config-v1`. Saving it cannot run a
  provider, choose an adapter, create a route, or promote a receipt.

The Metal field is ornamental. It receives only presentation state and cannot
produce, mutate, or promote a routing or semantic receipt.

## Validated slice

The macOS 27 v1 slice has been compiled, tested, launched, and visually
inspected on Apple Silicon. The durable gate receipt is:

```text
evidence/2026-09-03-spatial-v2-native-gate.txt
evidence/2026-09-03-message-bridge-gate.txt
evidence/2026-09-03-delivery-aware-roundtrip-gate.txt
evidence/2026-09-03-routing-config-gate.txt
```

Live fleet and journal data are labeled `LIVE / READ ONLY`. The configured
composer can publish a durable coordination request through the authenticated
bridge, and the Conversation tab follows its correlated durable lifecycle.
Route receipts remain explicitly labeled scenarios because the UI neither
arbitrates nor promotes them. The authenticated OCaml message bridge is present
in the promoted shared runtime, and the delivery-aware SwiftUI slice has a live
request/reply/ack receipt.
