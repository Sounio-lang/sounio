# Garden: LOOM Exec Operation Catalog V1

Status: `GARDEN_PREREGISTERED`

## Preserved phrase

> General execution is a typed operation space, not an arbitrary shell string.

## Question

The provider lifecycle now reaches a fresh DynamicUser ExecCell and returns the
frozen calibration result. Can Sounio generalize that exact witness into more
than one useful operation without turning the provider's shell text into
execution authority?

## Authority order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> OCAML_CATALOG_PROJECTION
-> HOST_PAYLOAD_SELECTION
-> PROVIDER_LIFECYCLE_ATTACHMENT
-> PARITY_OPEN
-> CLAIM_READY
```

Sounio is `SEMANTIC_AUTHORITY`. OCaml may validate and project a request onto a
frozen entry. C++20, Linux, and systemd may bind that entry to an executable and
a fresh principal. No parity or material layer may add an operation, weaken its
argument grammar, or invent its result schema.

## V1 entries

The first catalog contains two entries:

1. `calibration`: the exact action-9030 command and action-9033 result already
   measured on the host.
2. `sounio-check`: a read-only compiler check over one worktree-relative `.sio`
   path. Traversal, absolute paths, shell metacharacters, writes, and network
   effects are outside the operation.

`sounio-check` is only a semantic catalog entry at this stage. It is not
materially executable until a later host payload-selection receipt binds its
frozen schemas to a concrete source-built toolchain.

## Falsifier

The load-bearing rule is operation-specific command-template equality. An
unchanged `sounio-check` witness with only that binding absent must be refused
with `DENY567`. A source mutant deleting only this comparison must admit the
same witness. If another rule still refuses it, the claimed rule is not causal.

## Nonclaims

- `ocaml_catalog_projection_attached=false`
- `host_payload_selection_attached=false`
- `provider_lifecycle_attached=false`
- `arbitrary_shell=false`
- `general_exec_attached=false`
- `production_activation=false`
- `parity_open=false`
- `claim_ready=false`
