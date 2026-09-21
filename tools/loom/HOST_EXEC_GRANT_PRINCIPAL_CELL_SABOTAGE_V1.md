# LOOM Host PrincipalCell Same-Principal Sabotage V1

Status: `PREREGISTERED_CONTROL`

## Causal Question

Did the treatment refuse cross-cell signal operations because the cells had
kernel-distinct principals, or would the surrounding systemd hardening,
`PR_SET_DUMPABLE=0`, cgroup separation, and pidfd handling have refused them
without distinct UID/GID identity?

## Intervention

Repeat the two-cell experiment with the same source-built C++20 binary and the
same service hardening, while changing exactly the principal allocation:

- treatment: two simultaneous `DynamicUser` service identities with distinct
  UID/GID values;
- control: two simultaneous services name the same `DynamicUser`, forcing one
  shared non-root UID/GID while preserving distinct PIDs and cgroups.

The control's attacker inherits a root-opened pidfd for the other process, then
drops irreversibly to the shared UID/GID. It invokes only permission probes;
signal number zero must not change process state.

## Prediction

The control is causal only if all of these hold:

1. both processes retain `NoNewPrivileges=yes`, zero effective and ambient
   capabilities, `PR_SET_DUMPABLE=0`, stable start ticks, and distinct cgroups;
2. their UID/GID vectors are identical and non-root;
3. `kill(peer_pid, 0)` succeeds;
4. `pidfd_send_signal(copied_peer_pidfd, 0)` succeeds;
5. the same two operations were `EPERM` in the distinct-principal treatment;
6. both processes remain alive and identity-stable after the probes.

`ptrace`, `/proc/<pid>/mem`, `process_vm_readv`, `/proc/<pid>/fd`, and
`pidfd_getfd` may remain refused because dumpability, Yama, or ptrace access
checks are independent controls. Their continued refusal does not falsify the
signal-permission intervention.

If the shared-principal probes remain refused, the result is inconclusive and
the kernel-distinct-principal claim must not be promoted as causal. If the
shared-principal cells fail to start or differ in another required posture, the
control is invalid rather than negative.

## Boundaries

This control defines no Sounio decision and issues no grant. Its only expected
material contrast is:

`DISTINCT_UID: signal=EPERM -> SHARED_UID: signal=ALLOWED`

The frozen action `9030` remains semantic authority. C++20 is
`MATERIAL_PARITY`; systemd is the host mechanism; Bash is mechanical transport.
All product attachment, launch, material-grant, and grant-extinction flags stay
false regardless of the control result.
