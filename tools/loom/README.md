# Sounio Loom

Loom is Sounio's native OCaml session kernel and terminal multiplexer. A
recoverable guardian owns the PTY generation, child process, durable output, and
custody journal. The kernel owns semantic input authority and disposable client
connections. Clients or the entire kernel can disappear and recover without
changing the guardian, child, or generation identity.
Persisted descriptors are revalidated against process birth identity: an absent
kernel with a live Guardian is shown as `recoverable`, while loss of both is
shown as `lost` rather than being laundered as an active lane.

## Build

The local build requires OCaml, Dune, findlib, Cryptokit, and OpenSSL. On Debian or
Ubuntu the corresponding packages are:

```sh
sudo apt-get install ocaml-nox ocaml-dune ocaml-findlib libcryptokit-ocaml-dev openssl
scripts/dev/build_sounio_loom.sh
```

The stable `bin/sounio-loom` launcher selects the content-addressed shared
runtime when one is active. Set `SOUNIO_COORD_RUNTIME_MODE=local` to force the
source-worktree build for diagnosis.

## Operate

```sh
bin/sounio-loom start --agent codex --lane experiment -- COMMAND ARG...
bin/sounio-loom recover --agent codex --lane experiment
bin/sounio-loom guardian-status --agent codex --lane experiment
bin/sounio-loom fleet-enroll --slot codex-1 --kind codex --home "$HOME" --cwd "$PWD"
bin/sounio-loom fleet-reconcile
bin/sounio-loom fleet-reconcile --apply
bin/sounio-loom list
bin/sounio-loom tui
bin/sounio-loom serve --bind 127.0.0.1 --port 8787
bin/sounio-loom beagle-serve --bind 127.0.0.1 --port 4372
bin/sounio-loom verify-continuity-receipt \
  --receipt PATH --public-key PUBLIC.pem --adapter PATH
```

`serve` is read-only and binds to loopback by default. A non-loopback bind is
refused unless `--allow-remote` is explicit. The session directory and token are
local capabilities and must remain private to the owning user.

`fleet-enroll` stores desired lane intent under the repository's persistent Git
common directory. `fleet-reconcile` is a no-mutation plan by default; `--apply`
starts only enabled absent slots and verifies that each becomes active. Repeated
application is idempotent, and `fleet-disable` prevents an intentional stop from
being relaunched. The current launcher boundary delegates process creation to
the compatibility `agentd` adapter while desired-state parsing and reconciliation
policy live in the OCaml runtime.

`scripts/dev/install_sounio_loom_kubernetes_hook.sh` installs the one-shot
reconciler in the workspace StatefulSet. It refuses any update strategy other
than `OnDelete`, runs reconciliation as the lane-owning user instead of the
container's root user, and verifies that applying the template did not replace
the current Pod. Use `--dry-run` to inspect the strategic merge patch.

`beagle-serve` is a loopback compatibility backend for the existing Beagle
Workspace Agent. It implements `beagle-pty-supervisor-v1` without changing the
external `beagle-terminal-v1` client protocol. Beagle retains agent routing,
sessions, blocks, redaction, and memory policy; Loom owns PTY custody, process
identity, input authority, replay, and recovery. The bridge exposes verified
local journal heads, recovery counts, and a hash-chained generation-lineage
receipt as additive response fields. If kernel, Guardian, and harness all die,
Loom truthfully creates a new physical generation linked to the verified
predecessor instead of presenting a replacement process as the old one. See
`docs/internal/concepts/loom-multiplexer.contract` for the authority matrix and
canary rules.

Set `SOUNIO_LOOM_REQUIRE_SIGNED_RECEIPTS=1`,
`SOUNIO_LOOM_SIGNING_KEY`, and `SOUNIO_LOOM_VERIFY_KEY` to activate the
fail-closed Ed25519 receipt protocol. The private key signs a canonical payload
containing the native Sounio verdict, adapter digest, continuity facts, and
predecessor receipt token. `verify-continuity-receipt` needs only the public key
and adapter: it verifies the signature, canonical encoding, adapter identity,
and a replay of the Sounio policy. A signed successor refuses an unsigned
predecessor, a different-generation receipt signed by the same key, or an
incomplete keypair. With signing unset, Loom retains receipt v1 compatibility;
that mode carries no authenticity claim.

## Evidence Boundary

Loom-1.4 tolerates observer, interactive-client, GUI, and kernel loss on one Unix
host. `recover` reconciles bytes fsynced by the guardian while no kernel existed
and semantically revokes input leases whose sockets died with the old kernel.
It cannot re-adopt the same PTY after Guardian or host loss. It can detect that
loss and reconcile a Beagle pane into a new generation whose append-only
lineage receipt binds both verified predecessor journal heads. The Kubernetes
startup hook activates that one-shot reconciliation after Pod loss. The local
coordination gate and a source-pinned Kubernetes canary now exercise four
complete generations: an unacknowledged wake reaches generations one through
three, same-generation retries deduplicate, and an ACK recorded after the
independent third-generation depth control suppresses generation four. This is
at-least-once wake delivery, not exactly-once execution of the work named by the
message.
Existing fleet generations therefore retain the Python `agentd` launch adapter
during migration.
The Beagle bridge passed its source gate, an isolated second-process canary,
and a source-derived four-Pod canary with a dedicated retained PVC. The native
Sounio adapter keeps initial generation, clean respawn, and Pod resurrection
promotion states distinct. The Ed25519 gate additionally refuses missing keys,
payload and signature mutation, the wrong public key, and a validly signed
receipt spliced from another generation. These controls establish bounded
receipt authenticity under the mounted key and adapter, not signer hardware
attestation or protection against compromise of the signing authority. The
current real-Pod witness remains single-node. Cross-node PVC reattachment,
deployed Cockpit, canonical-memory, Warp, Madaros parity, and exactly-once
external effects remain separate gates; see the 2026-08-24 receipts under
`tools/loom/evidence/`.
See `docs/internal/concepts/loom-multiplexer.contract` for the full semantic and
falsification contract.
