# LOOM Product DynamicUser LaneCell Canary V1

Status: `MATERIAL_CANARY_PREREGISTERED`

## Purpose

This experiment attaches the already-native product `agent-hook` to one real
host `LaneCell`. It is the first product-path test in which the hook principal
has a kernel identity distinct from the root-owned `HostGuardian` that creates
its inherited `ExecIngress` descriptor.

It extends the current LOOM implementation. It does not create a second
guardian, a second hook, or a competing execution architecture.

## Authority Chain

The experiment preserves this order:

```text
GARDEN_PRODUCT_DYNAMIC_USER_EXEC_ATTACHMENT_V1
-> Sounio action 9031 executable
-> product_exec_ingress_dark.runtime.v1 frozen
-> OCaml native hook attachment
-> C++20/Linux/systemd material canary
-> host receipt
```

Sounio is `SEMANTIC_AUTHORITY`. OCaml is the operational product attachment.
The C++20 broker, Linux identity observations, and systemd service manager are
transitory `MATERIAL_PARITY`. They may establish that a principal, descriptor,
pidfd, cgroup, and refusal physically existed. They may not author an ALLOW,
replace an action-9031 result, or promote a material observation to semantic
authority.

Python and Rust are prohibited from the authority path. Their command names
appear only as hostile inputs whose execution sentinels must remain absent.

## Material Topology

The capsule installs an immutable, root-owned authority root and native
runtimes. The host broker then starts the product hook in a transient service
with:

- `DynamicUser=yes` for the treatment principal;
- `PrivateUsers=no`, so the hook and guardian compare UIDs in the same host
  user namespace rather than through a namespace-relative fake root;
- a dedicated cgroup and runtime directory;
- `NoNewPrivileges=yes`, an empty capability set, private devices, private
  network, private temporary storage, and a read-only system view;
- the actual native `agent-hook --agent codex` entrypoint;
- the frozen Sounio language-authority and resident action-9031 runtimes;
- an exact-path Git `safe.directory` projection for this root-owned, read-only
  capsule, with no global or wildcard trust;
- no bearer-token file;
- one connected Unix-stream descriptor created during the root execution setup
  that PID 1 authorizes through systemd `OpenFile=`, then inherited as
  descriptor 3 across the credential drop and `execve` into the hook.

`OpenFile=` is transport only. On the measured systemd 255 path, the connection
is created by the root execution child of PID 1 before that same process becomes
the service `MainPID`. Linux freezes its root UID/GID in `SO_PEERCRED` at
connect time. The broker therefore requires the connector to have parent PID 1,
root peer credentials, the transient unit cgroup, and then the same PID, process
start tick, and cgroup as the live `DynamicUser` hook after `execve`. This is an
authenticated root-to-`DynamicUser` identity transition, not a blanket rule
that any root peer may authorize.

The hook binds the event and command SHA-256 values, requires EOF after the
exact response, and invokes the resident Sounio action 9031. The host broker
also authenticates the live hook with pidfd, executable identity, UID, GID, and
service hardening properties before responding.

The descriptor is non-bearer in the relevant sense: a pathname, copied JSON,
file number, repository read, or process running as the old workspace UID does
not recreate its open file description.

## Causal Matrix

Treatment:

- start the genuine hook as a distinct `DynamicUser`;
- inherit the descriptor created by the PID-1-owned service transition;
- bind the exact event and command hashes;
- evaluate the frozen current-material action 9031 projection;
- require the Sounio DENY receipt, hook continuation in `PROBE_ONLY`, and an
  absent command sentinel.

Sabotage controls:

1. Select the frozen action-9031 `seal` label, which returns ALLOW. The dark
   hook rule itself must refuse this unexpected ALLOW before execution.
2. Echo a different command hash. The hook must refuse the response binding.
3. Close the guardian endpoint without a response. The hook must fail closed.
4. Run the hook as root, matching the connector's root credential. The hook
   must refuse the non-distinct peer.
5. Omit the inherited descriptor. Required mode must refuse before any Sounio
   grant lookup or command materialization. This fast-exit structural control
   is not relabeled as a Sounio decision.
6. Submit Python and Rust oracle commands. The resident Sounio denial must be
   observed and both execution sentinels must remain absent.

The experiment fails if a command sentinel exists, a sabotage returns through
the treatment path, a material component authors a semantic decision, the
connector is not a PID-1 child with root connect-time credentials, does not
become the exact lane `MainPID`, changes start tick or cgroup across the
transition, the treatment UID equals the connector credential, or any mutable
or bearer-token authority is present.

## Receipt Boundary

Every ingress decision records both sides of the kernel identity comparison:
the hook PID, parent PID, real/effective UID and GID, plus the inherited
socket's `SO_PEERCRED` PID, UID, and GID. A distinct-principal claim is valid
only when those values are captured by the same hook decision record; an
out-of-band `systemctl` observation alone is insufficient.

The host driver types whether each case must emit a bound request or refuse
before any request. EOF before a request is admissible evidence only for an
explicit pre-request refusal control, such as the same-UID principal. The same
EOF in the treatment or a binding case fails the experiment.

A passing host receipt may set only:

```text
product_lane_cell_canary=true
distinct_uid_product_broker_canary=true
inherited_descriptor=true
command_executed=false
```

It must keep all of these false:

```text
fleet_lane_cell_attached=false
exec_cell_attached=false
material_execution=false
production_activation=false
launch_open=false
recycle_open=false
exec_attached=false
commit_attached=false
ci_attached=false
parity_open=false
claim_ready=false
```

The next authority-changing experiment is the separate-identity `ExecCell`.
Until its command execution and affirmative extinction are measured together,
this canary is evidence that product admission reached a real kernel boundary,
not evidence that LOOM command execution is production-ready.
