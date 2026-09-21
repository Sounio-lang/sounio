# Garden: Product DynamicUser Exec Attachment V1

Status: `GARDEN_PREREGISTERED`

## Preserved phrase

> Every lane is an operating-system principal; every command is a separate
> cell.

## Seed

LOOM should not merely coordinate provider CLIs that happen to share one Unix
user. It should make the identity boundaries of the collaboration material.
The provider harness for a lane should live in a long-lived `LaneCell` with a
kernel-distinct principal. Each authorized command should run in a fresh,
short-lived `ExecCell` with another kernel-distinct principal. A resident host
guardian should be the only process that can join the two cells through a
one-shot, descriptor-bound protocol.

The intended product is therefore more than a terminal multiplexer. It is an
epistemic operating surface in which attention, authority, execution, and
evidence have different lifetimes and different kernel identities.

## Authority order

This seed preserves the mandatory order:

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> MATERIAL_ATTACHMENT
-> PRODUCT_CANARY
-> PARITY_OPEN
-> CLAIM_READY
```

Sounio remains `SEMANTIC_AUTHORITY`. The frozen Sounio actions `9025`, `9030`,
and `9031` define effect closure, `ExecGrant`, and product activation. OCaml is
the product attachment and supervision language. The existing C++20/Linux/
systemd broker and principal cells are transitory `MATERIAL_PARITY`; they may
measure and realize kernel facts but may not invent an expected Sounio result.
Python and Rust are forbidden from the authority and execution path.

## Existing parents to attach

This is not permission for a second implementation. The attachment must reuse:

- `kernel_peer_material_judgment_v13.freeze.v1`, whose material experiment
  records the action-9025 peer-isolation parent;
- `host_exec_quorum_host.runtime.v1`, whose host experiment records a linear,
  non-bearer action-9030 material grant;
- `process_witness_host.runtime.v1`, whose host experiment records the
  two-phase execution and affirmative extinction core;
- `product_exec_ingress_dark.runtime.v1`, whose native hook attachment observes
  a one-shot inherited descriptor before `ExecGrant` issuance;
- the existing native OCaml LOOM kernel, provider custody, durable outcomes,
  recovery, and shared content-addressed runtime.

The new work is the composition and product attachment of these parents.

## Candidate architecture

### HostGuardian

A resident root-owned host service owns no scientific semantics. It verifies
the frozen Sounio capsule, launches cells, holds pidfds and cgroup identities,
passes inherited descriptors, consumes grants, and records kernel facts. Its
public input is a bounded typed request, not shell text. It fails closed on
missing policy, malformed input, timeout, guardian restart, or identity drift.

### LaneCell

Each provider CLI session runs as a systemd `DynamicUser` with a dedicated
cgroup, pidfd, private runtime directory, isolated temporary storage, no
ambient capabilities, and explicit read-only/read-write mounts for its claimed
worktree surface. Provider credentials enter through scoped systemd credentials
and never become LOOM authority. A lane cannot inspect another lane's process,
descriptors, environment, credentials, PTY, or execution cells.

The lane PTY is durable state owned by the guardian, not by tmux. A pod or GUI
may disappear and reconnect without becoming the custody authority.

### ExecCell

Every approved command runs in a fresh `DynamicUser` cell distinct from the
requesting `LaneCell` and every concurrent `ExecCell`. The cell receives only:

- the exact argv and cwd frozen in the action-9030 grant;
- the declared environment projection;
- the exact claimed filesystem mounts;
- one input descriptor and bounded output descriptors;
- the one-shot close phase required by `ProcessWitness`.

The cell has no provider credential, coordination token, host socket pathname,
or reusable capability file. Completion is not inferred from disappearance:
the guardian records exit status, pidfd extinction, empty descendant set,
unpopulated cgroup, inactive unit, generation extinction, and authority
extinction.

### Descriptor chain

The guardian creates the descriptor before the provider turn begins. The
`LaneCell` inherits one endpoint; no pathname or file can recreate it. At
`PreToolUse`, the native hook binds the event hash and command hash. The
guardian validates the live lane principal and the frozen Sounio capsule, then
either refuses or issues one action-9030 generation to an `ExecCell`. The
descriptor is consumed once and closed at both ends.

The provider CLI does not execute the command after approval. It receives a
stable LOOM result handle whose output is streamed from the `ExecCell`. This is
the architectural switch that removes same-UID material execution from the
product path.

## First differentiating experiment

Run two hostile `LaneCell` probes and one `ExecCell` on the host at the same
time. Give the attacker the command JSON, event bytes, every repository file,
the provider CLI executable, the harness ancestry shape, and the public runtime
manifest. Do not give it an inherited descriptor or a kernel principal.

Treatment:

- the intended lane binds one request, receives one Sounio-authorized grant,
  and the separate `ExecCell` produces a committed outcome.

Sabotages:

- attacker replays the JSON without the descriptor;
- attacker copies a descriptor number but not the open file description;
- attacker races before and after binding;
- lane, guardian, or exec cell dies at every protocol phase;
- an `ExecCell` aliases the lane UID, another exec UID, cgroup, pid, start tick,
  executable, grant generation, argv, cwd, mount set, or environment;
- the Sounio action is replaced with an operational-language allow;
- Python or Rust is requested as an oracle or compatibility bridge.

The hypothesis is falsified if any sabotage executes a byte, if the treatment
executes under the lane UID, if a stale generation commits an outcome, or if a
non-Sounio receipt is promotable to semantic authority.

## Acceptance gates

1. Freeze the current product counterexample: the native hook is attached, but
   its absent-descriptor default continues into an action-9030 broker that
   authenticates with a bearer token file and forks the command under the
   broker's effective UID.
2. Attach a host `LaneCell` canary with a distinct UID and inherited descriptor,
   while keeping action 9031 dark and execution closed.
3. Attach the existing host quorum and `ProcessWitness` to one test-only
   `ExecCell`; require a Sounio-authored treatment and sabotage control.
4. Replace the same-UID OCaml `fork/exec` material path with the host cell path.
5. Make descriptor absence fail closed for product execution tools.
6. Prove crash recovery at every transition without replaying a grant or
   reporting an extinct PTY as live.
7. Package the guardian, Sounio capsule, OCaml client, material broker, source
   hashes, hardware facts, command, and result in one immutable runtime receipt.
8. Roll out one canary lane, then the fleet, with atomic rollback that cannot
   reopen an old generation.

## Explicit nonclaims

At Garden preregistration:

- `lane_cell_attached=false`;
- `distinct_uid_product_broker=false`;
- `exec_cell_attached=false`;
- `material_execution=false`;
- `production_activation=false`;
- `launch_open=false`;
- `recycle_open=false`;
- `exec_attached=false`;
- `commit_attached=false`;
- `ci_attached=false`;
- `parity_open=false`;
- `claim_ready=false`.

The existing dark ExecIngress rollout remains valid and active while this
Garden seed is tested.
