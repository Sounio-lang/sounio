# LOOM Product DynamicUser Exec Counterexample V1

Status: `COUNTEREXAMPLE_EXECUTABLE`

## Question

Does the product path already compose its descriptor-bound native hook, host
material grant, `ProcessWitness`, and material peer isolation into a
kernel-distinct command execution?

## Observed answer

No. The parents exist, but the product composition does not.

The current native hook observes `ExecIngress` before issuing an `ExecGrant`,
but descriptor absence is allowed in dark mode. The resulting replacement
command re-enters the same OCaml executable as `exec-capability`. That executable
connects to the kernel with `SOUNIO_LOOM_TOKEN_FILE`, consumes the generation,
then calls `Unix.fork` and `Unix.execve` without changing UID, GID, cgroup, or
systemd unit. The material child therefore inherits the broker's effective
principal.

This does not invalidate the frozen host experiments. It proves that their
properties are not yet product properties:

- the host quorum has a distinct UID and material grant, but
  `material_execution=false` and `exec_attached=false`;
- `ProcessWitness` has a distinct UID, two-phase close, and affirmative
  extinction, but `material_execution=false` and `exec_attached=false`;
- V13 records material peer isolation and material execution inside its bounded
  experiment, but `production_activation=false` and `exec_attached=false`;
- the shared product ExecIngress is real and active in dark mode, but
  `required_mode_default=false` and `distinct_uid_product_broker=false`.

## Falsifier

This counterexample stops passing only when the product command path no longer
uses a bearer token file, no longer re-enters the same executable to materialize
the command, and no longer forks the command under the broker's effective UID.
The replacement must be a host-bound `LaneCell -> HostGuardian -> ExecCell`
transition whose acceptance is still authored by frozen Sounio semantics.

## Boundary

This artifact is a structural counterexample, not a semantic oracle. Bash
checks recorded boundaries and implementation structure. It may not promote an
ALLOW, encode a replacement Sounio result, or claim production execution.
