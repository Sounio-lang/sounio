# Garden: LOOM Exec Result Record V1

Status: `GARDEN_PREREGISTERED`

## Preserved phrase

> A dynamic result is evidence about an execution, never execution authority.

## Question

The frozen action-9033 handle proves one exact calibration receipt. How can a
typed action-9035 operation return a result whose values are known only after a
fresh ExecCell runs, without letting OCaml, C++, the provider, or an LLM invent
the meaning of that result?

## Authority order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> OCAML_RECORD_PROJECTION
-> DYNAMIC_USER_HOST_ATTACHMENT
-> PROVIDER_RESULT_RETURN
-> PARITY_OPEN
-> CLAIM_READY
```

Sounio is `SEMANTIC_AUTHORITY`. Action 9036 realizes the opaque
`sounio-check` result-schema identity from action 9035 as a canonical dynamic
record. OCaml may populate and hash that record. C++20, Linux, and systemd may
measure the ExecCell. None may turn the record handle into a bearer capability
or execution authority.

## Canonical record

The record binds operation, semantic event, command template, generation,
source, compiler, argument vector, artifact, artifact size, stdout, stderr,
diagnostics, sandbox profile, measured principal identity, inherited-descriptor
binding, consumed-grant receipt, and exit code. Its handle is derived from the
event, generation, and canonical record hash. The artifact is measured but is
not executed by this protocol.

Expected artifact bytes or hashes are not encoded in the material layer. A
successful result means the frozen compiler exited successfully and the
canonical record contains the observed measurements.

## Falsifier

The load-bearing rule is artifact-to-record binding. An unchanged otherwise
valid witness with only `artifact_bound_to_record=false` must be refused with
`DENY577`. A Sounio mutant deleting only that comparison must admit the same
witness.

## Nonclaims

- `ocaml_record_projection_attached=false`
- `dynamic_user_host_attached=false`
- `provider_result_returned=false`
- `handle_is_bearer=false`
- `handle_is_execution_authority=false`
- `production_activation=false`
- `parity_open=false`
- `claim_ready=false`
