# GARDEN: LOOM Resident Membrane V5 Action 9031

Status: `PREREGISTERED`

## Question

Can the existing single persistent Sounio resident expose frozen action `9031`
without creating a second authority process, changing any expected decision, or
opening product execution?

This Garden describes transport attachment only. The semantics and expected
results already belong exclusively to frozen Sounio action `9031`:

- manifest: `tools/loom/kernel_peer_activation_capsule_authority.freeze.v1`;
- manifest SHA-256:
  `f2da55138bcfe5a8a2c65ebd79c1e534f152b33af5c6cc3d1f2b4eb3b4af6e7e`;
- action: `9031`;
- role: `SEMANTIC_AUTHORITY`.

## Single-Resident Rule

Resident v5 is a strict extension of resident v4. One Sounio process must serve
all six frozen actions during one generation:

- route `1`: action `9024`;
- route `2`: action `9023`;
- route `3`: action `9025`;
- route `4`: action `9029`;
- route `5`: action `9030`;
- route `6`: action `9031`;
- route `0`: stop.

No companion daemon, per-action broker, fallback interpreter, or post-hoc
decision adapter is permitted. OCaml may supervise this process later, but it
may not translate, repair, or replace a Sounio decision.

## Exact-Parity Boundary

The v5 build mechanically renames each frozen single-shot Sounio entrypoint and
assembles it with the resident dispatcher. For every test frame, the resident
response must be byte-for-byte identical to the response from the corresponding
source-fresh standalone Sounio executable.

The selftest must cover all four legal action-`9031` transitions and its current
material refusal. It must also preserve representative decisions from routes
`1` through `5`, including their Python-oracle refusals where defined. Expected
codes are read from frozen Sounio receipts or compared with standalone Sounio;
shell and OCaml do not author them.

## Process Identity

All round trips must occur through the same PID and the same Linux process birth
identity. A restart, EOF, timeout, malformed route, extra output, source drift,
parent-manifest drift, or compiler drift fails closed. Transport recovery may
start a new generation, but it cannot pretend that it is the old resident.

## Parent Chain

Resident v5 binds:

1. the frozen manifests for actions `9023`, `9024`, `9025`, `9029`, `9030`, and
   `9031`;
2. the frozen resident-v4 manifest;
3. action `9031`'s exact bindings to V13 action `9025` and action `9030`;
4. the source-fresh `lean_single` wrapper and compiler hashes used to rebuild
   the resident twice deterministically.

Any mismatch refuses the build or runtime load before a frame is accepted.

## Authority Separation

- Sounio remains `SEMANTIC_AUTHORITY` and produces all decisions.
- OCaml is the future `OPERATIONAL_REALIZATION` and may only supervise and apply
  an already-authorized transition.
- C++ remains the transitory material/kernel observation layer.
- Lean, Koka, and optional Haskell remain parity roles after semantic freeze.
- External LLMs remain `REVIEW_ONLY`.
- Python and Rust are prohibited.

The dispatcher itself defines no expected code. Route selection is not semantic
authority.

## Acceptance Boundary

This phase may establish:

- `resident_v5_frozen=true`;
- `single_resident_pid=true`;
- `action_9031_route=6`;
- exact standalone/resident Sounio parity;
- deterministic source-fresh rebuilds.

It must retain:

- `ocaml_capsule_started=false`;
- `capsule_material=false`;
- `production_activation=false`;
- `launch_open=false`;
- `recycle_open=false`;
- `exec_attached=false`;
- `commit_attached=false`;
- `ci_attached=false`;
- `parity_open=false`;
- `claim_ready=false`.
