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

## Evidence Boundary

Loom-1 tolerates observer, interactive-client, GUI, and kernel loss on one Unix
host. `recover` reconciles bytes fsynced by the guardian while no kernel existed
and semantically revokes input leases whose sockets died with the old kernel.
It cannot re-adopt the same PTY after Guardian or host loss. It can detect that
loss and reconcile an enrolled lane into a new verified generation. The
Kubernetes startup hook activates that one-shot reconciliation after Pod loss,
but pending-inbox replay across the new generation still needs its own gate.
Existing fleet generations therefore retain the Python `agentd` launch adapter
during migration.
See `docs/internal/concepts/loom-multiplexer.contract` for the full semantic and
falsification contract.
