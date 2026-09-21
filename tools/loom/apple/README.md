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
tools/loom/_build/default/src/loom.exe message-serve \
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
- `POST /v1/messages` exists only on the separate loopback message bridge.
- The bridge fixes sender identity at startup; the UI supplies only destination,
  message text, and the bounded `request` kind.
- A send is displayed as successful only after a durable bus receipt is decoded.
- `ProviderAccount`, `QuotaPool`, and `CliAdapter` are deliberately independent.
- The UI renders `Task -> RouteDecision -> RouteReceipt`; the backend arbitrates.

The Metal field is ornamental. It receives only presentation state and cannot
produce, mutate, or promote a routing or semantic receipt.

## Validated slice

The macOS 27 v1 slice has been compiled, tested, launched, and visually
inspected on Apple Silicon. The durable gate receipt is:

```text
evidence/2026-09-03-spatial-v2-native-gate.txt
evidence/2026-09-03-message-bridge-gate.txt
```

Live fleet and journal data are labeled `LIVE / READ ONLY`. The configured
composer can publish a durable coordination request through the authenticated
bridge, while route receipts and the displayed conversation history remain
explicitly labeled scenarios. The message bridge is currently validated from
the source-built OCaml runtime; shared-runtime promotion remains a separate
release step.
