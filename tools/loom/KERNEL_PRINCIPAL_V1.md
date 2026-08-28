# LOOM Kernel Principal V1

Status: `SEMANTICS_FROZEN_MATERIAL_REFUSED`

LOOM kernel-principal extrusion gives each mutually hostile lane a kernel
identity distinct from the outer account and from every sibling lane. It is the
planned boundary below the in-memory `ExecGrant` protocol. A lane name, handle,
socket, ancestry chain, PID namespace, or cgroup path is never accepted as a
principal by itself.

## Authority

Sounio action `9026` is frozen by
`kernel_principal_authority.freeze.v1`. It owns the expected decisions and is a
child of frozen effect-closure action `9025`. The C++20 probe is
`MATERIAL_PARITY`: it observes Linux state and emits an action frame, but cannot
change a DENY into an ALLOW.

The semantic order remains:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY`

Python and Rust are absent. Shell scripts build and carry records only; they do
not compute the expected decision.

## Material Probe

Build and run the source-fresh gate with:

```sh
bash scripts/ci/sounio_loom_kernel_principal_material_selftest.sh
```

The probe records, independently:

- configured subordinate UID/GID ranges;
- setuid-root presence of `newuidmap` and `newgidmap`;
- a basic user-namespace map and its effective outer UID;
- attempted automatic subordinate mapping and the actual map receipt;
- PID namespace, mount namespace, and cgroup-v2 availability;
- whether the outer account can noninteractively regain root;
- a SHA-256 receipt and the mechanically derived action `9026` frame.

Every spawned mechanism has a five-second deadline and a one-megabyte output
limit. Timeout, output overflow, malformed output, or missing receipts fail
closed.

## Current Result

The recorded Linux x86_64 pod has ranges `100000:65536`, setuid-root uidmap
helpers, user/PID/mount namespaces, and cgroup v2. Two independent blockers
remain:

1. `newuidmap` is invoked with the documented automatic subordinate mapping but
   the kernel returns `EPERM`, so neither exact UID nor GID map exists;
2. the outer UID `1000` can execute `sudo -n`, so a hostile peer can regain a
   capability domain that defeats lane isolation.

Sounio returns `DENY455 subordinate-allocation-incomplete`. A basic
`--map-current-user` succeeds but maps back to host UID `1000`; the probe records
`current_principal_distinct=0` and does not promote it.

The material sabotage replaces the mapping attempt with a command that exits
zero. Because no kernel map receipt appears, `mapping_materialized` remains zero
and Sounio still returns `DENY455`. Helper exit status is therefore not an
authority signal.

## Admission Path

Material admission requires a host or service boundary that provides all of:

- working disjoint subordinate UID/GID maps;
- a non-passwordless, non-regainable outer account;
- exact `SO_PEERCRED`, pidfd/start-tick, executable, ancestry, worktree, and
  cgroup binding;
- non-dumpable/no-new-privileges execution with no effective, permitted,
  bounding, or ambient `CAP_SYS_PTRACE`;
- outer, sibling, and right-principal/wrong-ancestry attack processes that fail
  before grant lookup or burn;
- atomic one-shot grants and crash revocation;
- five material sabotage receipts bound to the frozen Sounio source and
  semantics.

The likely production realization is a narrow host-owned LOOM principal broker
that allocates subordinate IDs before launching a lane, then permanently drops
privilege. The interactive harness never receives the broker's privilege and
never stores a bearer credential capable of manufacturing a principal.

## Nonclaims

- The current pod is not hostile-peer isolated.
- Installing `uidmap` did not establish a mapping.
- A successful user namespace, PID namespace, or helper exit is not isolation.
- Passwordless sudo makes a same-UID hostility claim false.
- The C++ probe is not semantic authority.
- General Exec/Bash, commit, and CI attachment remain refused.
