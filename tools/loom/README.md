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

The local build requires OCaml, Dune, findlib, and Cryptokit. On Debian or
Ubuntu the corresponding packages are:

```sh
sudo apt-get install ocaml-nox ocaml-dune ocaml-findlib libcryptokit-ocaml-dev
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

## Evidence Boundary

Loom-1.4 tolerates observer, interactive-client, GUI, and kernel loss on one Unix
host. `recover` reconciles bytes fsynced by the guardian while no kernel existed
and semantically revokes input leases whose sockets died with the old kernel.
It cannot re-adopt the same PTY after Guardian or host loss. It can detect that
loss and reconcile a Beagle pane into a new generation whose append-only
lineage receipt binds both verified predecessor journal heads. The
Kubernetes startup hook activates that one-shot reconciliation after Pod loss,
but pending-inbox replay across the new generation still needs its own gate.
Existing fleet generations therefore retain the Python `agentd` launch adapter
during migration.
The Beagle bridge passed its source gate, an isolated second-process canary
against the live Workspace Agent image, and a source-derived separate-Pod
canary with a dedicated PVC. The replacement Pod retained two Beagle blocks,
created a new physical instance, and exposed a verified link to the unclean
predecessor. The source gate also refuses a mutated lineage before spawn. Loom
has not replaced production authority or passed canonical-memory, Cockpit,
Warp, cross-node, or pending-inbox gates; see the receipts under
`tools/loom/evidence/beagle-workspace-agent-*-20260824.txt`.
See `docs/internal/concepts/loom-multiplexer.contract` for the full semantic and
falsification contract.
