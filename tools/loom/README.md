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
bin/sounio-loom list
bin/sounio-loom tui
bin/sounio-loom serve --bind 127.0.0.1 --port 8787
```

`serve` is read-only and binds to loopback by default. A non-loopback bind is
refused unless `--allow-remote` is explicit. The session directory and token are
local capabilities and must remain private to the owning user.

## Evidence Boundary

Loom-1 tolerates observer, interactive-client, GUI, and kernel loss on one Unix
host. `recover` reconciles bytes fsynced by the guardian while no kernel existed
and semantically revokes input leases whose sockets died with the old kernel.
It does not yet recover after guardian or host loss. Existing fleet generations
therefore remain on the Python `agentd` compatibility path until a separate
live-migration gate preserves generation identity and pending delivery.
See `docs/internal/concepts/loom-multiplexer.contract` for the full semantic and
falsification contract.
