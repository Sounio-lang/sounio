<!-- docs:meta
topic_id: repo.docs.internal.concepts.loom-sovereign-execution-kernel
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.loom-sovereign-execution-kernel
-->

# LOOM Sovereign Execution Kernel

Concept-ID: `SOUNIO-LOOM-SOVEREIGN-EXECUTION-KERNEL`

## Authority

Sounio is the semantic authority for action 9042. The action composes the
frozen action-9025 peer judgment, action-9030 ExecGrant cell, and action-9031
peer activation capsule. Material and operational layers may measure this
contract, but they cannot weaken it or manufacture an expected result.

## Contract

The sovereign kernel admits an execution only when all of the following are
affirmatively observed:

- the ExecGrant exists only in Loom-kernel memory, is non-bearer, single-use,
  and is consumed by one atomic state transition after the HostGuardian has
  registered the material worker;
- no token, handle, or transport descriptor is execution authority;
- the requesting process is bound by `SO_PEERCRED`, pidfd, start tick, harness
  ancestry, executable identity, and operation identity;
- GUI, TUI, CLI, Pod, and tmux processes have zero release authority;
- loss of transport, GUI, and coordinator leaves the material witness running;
- a hostile process with the same UID is refused before material execution;
- death of the true HostGuardian revokes the grant and extinguishes the
  not-yet-released material operation;
- production activation is impossible until the joined material observation
  establishes `same_uid_peer_isolation=true`.

The decisions are intentionally separate:

- `EXEC_ADMIT` proves an exact material witness can run under a live Guardian;
- `GUARDIAN_REVOKE` proves fail-closed extinction after Guardian death;
- `PRODUCTION_GATE_READY` proves the production prerequisite is satisfied, but
  does not itself activate production.

## Semantic Boundary

Action 9042 owns the join between the frozen 9025, 9030, and 9031 authorities.
It does not redefine their peer, grant, or capsule semantics. It also does not
claim that transport survival preserves the same PTY after HostGuardian loss.

The first material realization is a transitory C++20/Linux canary. It must bind
to frozen Sounio output and is not a production release mechanism. The target
implementation remains a Sounio-compiled HostGuardian.
