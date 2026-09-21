# LOOM Host ExecGrant PrincipalCell V1

Status: `MATERIAL_PREREGISTERED`

## Purpose

This contract realizes one prerequisite of frozen Sounio action `9030`: two
simultaneous hostile lanes must occupy kernel-distinct principals before an
`ExecGrantCell` may become material. It does not open execution and does not
change any Sounio decision.

The authority order remains:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> MATERIAL_MEASUREMENT`

The frozen semantic authority is
`tools/loom/kernel_exec_grant_cell_authority.freeze.v1`. C++20 and systemd are
transitory material mechanisms. Bash transports and invokes the measurement.
None of them may define an expected Sounio result, grant authority by
possession, or promote a material claim.

## PrincipalCell Candidate

Each candidate cell is a transient systemd service with:

- `DynamicUser=yes`, yielding a simultaneous host UID/GID distinct from every
  other live candidate cell;
- an independent cgroup and stable process start tick;
- `NoNewPrivileges=yes`, an empty effective and ambient capability set, and
  `PR_SET_DUMPABLE=0`;
- a live pidfd held by the root measurement boundary;
- strict filesystem, device, kernel, control-group, realtime, personality, and
  address-family restrictions;
- no authority-bearing file, token, environment variable, or command result.

Dynamic UID allocation is a mechanism, not a semantic decision. A future
resident host broker must bind the observed UID, GID, cgroup, pidfd, start tick,
boot identity, executable, invocation, command, and grant generation to the
same frozen action `9030` receipt before a barrier can open.

## Hostile Controls

While two cells are alive simultaneously, the native measurement deliberately
forks one attacker under each cell's exact UID/GID. Each attacker inherits a
root-opened pidfd for the opposite cell. The following operations must still be
refused by the kernel in both directions:

1. `kill(pid, 0)`;
2. opening `/proc/<pid>/mem`;
3. `PTRACE_ATTACH`;
4. `process_vm_readv`;
5. opening `/proc/<pid>/fd/1`;
6. `pidfd_send_signal(pidfd, 0)` with the copied pidfd;
7. `pidfd_getfd(pidfd, 1)` with the copied pidfd.

The two pidfds must remain live and the start ticks unchanged throughout the
attacks. A copied pidfd that authorizes either signal or descriptor extraction
falsifies this candidate boundary.

## Fail-Closed Rules

The gate refuses when root or systemd is absent, `DynamicUser` is unavailable,
the services alias a UID/GID or cgroup, capabilities are nonzero, a process
changes identity, pidfd support is absent, an attack succeeds, cleanup fails,
or transport hashes drift. Timeout and malformed output are refusals.

The gate records the observed dynamic identities but makes these conservative
claims:

- `kernel_distinct_principal_candidate=true` only after all hostile controls;
- `same_uid_peer_isolation=false`, because the mechanism avoids same-UID peers
  rather than making same-UID processes mutually hostile;
- `material_grant=false` and `grant_extinction=false`, because no action `9030`
  grant is issued or extinguished by this experiment;
- `exec_attached=false`, `commit_attached=false`, `ci_attached=false`, and
  `launch_open=false`.

## Acceptance

The material prerequisite is measured, not promoted, when:

- two source-identical C++20 builds are byte-identical;
- the binary has no Python or Rust runtime dependency and contains no frozen
  Sounio `ALLOW` or `DENY` result table;
- two live DynamicUser cells have distinct UID/GID and cgroup identities;
- pidfds and start ticks are stable across all reciprocal hostile controls;
- every ordinary and copied-pidfd attack is refused;
- both transient units are stopped and reset by the host gate;
- a receipt binds source, binary, host gate, transport, hardware, kernel, and
  observed results.

Only a later action-9030-bound broker integration may change the material grant
or product attachment flags.
