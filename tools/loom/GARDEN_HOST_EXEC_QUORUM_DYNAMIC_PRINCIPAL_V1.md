# Garden: host ExecQuorum dynamic principal v1

Status: preregistered before host implementation and measurement.

This experiment derives from the frozen host ExecQuorum v1 Garden whose
SHA-256 is
`67aecd9785a1aa6e95f80cac41f7344bbf7a1fc0eb6c27e07ec378986fc5a7a0`.
It narrows host materialization without changing the frozen Sounio semantics,
fixtures, controller lifecycle, local result, or language roles.

## Research question

Can the already-frozen three-object ExecQuorum materialize one non-serializable
grant inside a real kernel-distinct `DynamicUser` PrincipalCell, while the
deployed decision-only broker remains unchanged and rollback remains armed?

## Preregistered host materialization

The host experiment uses the existing broker and the frozen descriptor barrier.
It may add a host-only PrincipalCell entrypoint, but it must not add a second
broker, controller, resident, or semantic decision path.

The broker launches the PrincipalCell as a transient `systemd` service with
`DynamicUser=yes`. `systemd-run --pipe` mechanically transfers two anonymous
pipe descriptors through `StartTransientUnit`: service standard input is the
barrier read end and service standard output is the observation write end. The
service receives no capability pathname, listening socket, token, command, or
user payload. Environment fields identify the experiment but are not authority;
copying them without the inherited descriptor cannot open the barrier.

The PrincipalCell must affirm all of the following before reading the barrier:

- PID 1 is its parent and the unit cgroup matches the preregistered transient
  unit name;
- real, effective, and saved UID/GID are equal and non-root;
- `NoNewPrivileges` is set, dumpability is disabled, and effective and ambient
  capabilities are zero;
- standard input and output are distinct anonymous pipes;
- the broker observes the same PID, start tick, executable, cgroup, UID/GID,
  and live pidfd immediately before release.

Because creation of a `DynamicUser` service is slower than the frozen 75 ms
release window, the inherited input stream has two phases. The broker first
writes the exact non-authoritative record `ARM\n` after PrincipalWitness
authentication. The PrincipalCell consumes exactly those four bytes without
reading ahead, then enters the unchanged frozen descriptor-barrier reader. Only
the subsequent exact generation record can materialize a grant. Missing,
truncated, duplicated, reordered, or extended arm records refuse before the
barrier. `ARM` is synchronization, never a receipt or capability by itself.

## Preregistered causal matrix

The host gate must exercise treatment, positive quorum, same-UID refusal,
semantic refusal, replay, generation mismatch, controller death, resident
death, copied-text refusal, and isolated exact-write sabotage. The positive and
sabotage rows must each observe exactly one `BARRIER_OPENED`; every other row
must observe zero. A second release attempt must refuse locally and cannot emit
a second sentinel.

Passing may set `material_grant=true`, `non_bearer_exec_quorum=true`, and
`descriptor_barrier_causal=true`. It must retain `material_execution=false`,
`launch_open=false`, `recycle_open=false`, `exec_attached=false`,
`commit_attached=false`, `ci_attached=false`, `parity_open=false`, and
`claim_ready=false`.

## Release and rollback

The first host run is installed as a content-addressed experimental release
beside the deployed decision-only release. It must not repoint the production
`current` or broker symlink. The prior release identity and symlink targets are
captured before installation and rechecked after the gate, making rollback the
identity operation: remove the experimental release only after proving the
deployed targets never moved. Host transport and packaging remain mechanical
and cannot promote any semantic or execution authority.
