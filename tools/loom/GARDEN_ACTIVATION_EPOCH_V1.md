# Garden: LOOM Activation Epoch v1

Status: GARDEN
Owner: codex-1 / loom-activation-epoch-v1-20260904
Parent: frozen Sounio action 9048

## Seed

A generation pin must remain immutable for the lifetime of its process, while
the compatibility head read by that pinned runtime must be allowed to advance.
Each advance is a new activation epoch, not a rewrite of history.

## Required transition

`ADVANCE(previous_head, next_runtime, unchanged_pin_set)` is allowed only when:

- the exact previous head is archived under its SHA-256 before replacement;
- the previous head names the runtime currently selected by `current`;
- the next runtime is an installed, immutable, native runtime with a verified
  manifest and executable hashes;
- the complete pin inventory and every pin byte remain unchanged;
- state and installation locks are held;
- the replacement head, epoch receipt, predecessor hash, and audit decision
  are ready before the selector moves;
- Python, Rust, and disposable oracle execution are absent;
- every error and timeout fails closed.

The stable compatibility head remains `generation-runtime-pins/activation.v1`
because already-running immutable runtimes know that path. Its exact prior
bytes are never lost: they are content-addressed in `activation-epochs/heads/`.
The append-only epoch receipt in `activation-epochs/epochs/` binds predecessor,
successor, pin inventory, Sounio authority, toolchain, command, and result.

## Evidence order

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY`

OCaml may project and enforce this decision only after the Sounio executable is
frozen. It cannot decide a transition by itself.
