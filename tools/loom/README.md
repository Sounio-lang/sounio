# Sounio Loom

Loom is Sounio's native OCaml session kernel and terminal multiplexer. The
kernel, not a TUI or browser, owns the PTY generation. Clients can disappear and
reattach at a durable output cursor without changing the child process identity.

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
bin/sounio-loom list
bin/sounio-loom tui
bin/sounio-loom serve --listen 127.0.0.1 --port 8787
bin/sounio-loom verify --agent codex --lane experiment
```

`serve` is read-only and binds to loopback by default. A non-loopback bind is
refused unless `--allow-remote` is explicit. The session directory and token are
local capabilities and must remain private to the owning user.

## Evidence Boundary

Loom-0 tolerates observer, interactive-client, and GUI loss on one Unix host.
It does not yet re-adopt a PTY after kernel-daemon or host loss. Existing fleet
generations therefore remain on the Python `agentd` compatibility path until a
separate live-migration gate preserves generation identity and pending delivery.
See `docs/internal/concepts/loom-multiplexer.contract` for the full semantic and
falsification contract.
