<!-- docs:meta
topic_id: repo.docs.internal.concepts.loom-kernel-exec-grant-cell
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.loom-kernel-exec-grant-cell
-->

# LOOM Kernel ExecGrantCell

Concept-ID: `SOUNIO-LOOM-KERNEL-EXEC-GRANT-CELL`

Status: executable and semantics-frozen in Sounio action `9030`. Operational
convergence, material grant custody, hostile-peer isolation, and product
attachment remain closed.

Canonical artifacts:
`tools/loom/GARDEN_KERNEL_EXEC_GRANT_CELL_V1.md`,
`stdlib/coordination/loom_kernel_exec_grant_cell_authority.sio`, and
`tools/loom/kernel_exec_grant_cell_authority.freeze.v1`.

## Meaning

An `ExecGrantCell` joins three already-frozen Sounio lineages:

- action `9029` identifies the exact principal-and-effect `InvocationCell`;
- action `9021` decides the exact command before issue and consume;
- action `9022` decides whether the observed outcome is complete.

The resulting handle is a lookup coordinate, not a bearer capability. A
material operation still requires an independently authenticated kernel peer,
the exact principal and Guardian ancestry, a legal pre-validated state write,
an unexpired in-memory grant in the same generation, and the same Sounio parent
decisions.

## Preserved Distinctions

```text
Opaque handle             != Execution authority
Grant table miss          != Proven extinction
Dead process              != Extinct authority
Post-write validation     != Safe ingestion
SO_PEERCRED plus ancestry != Hostile same-UID isolation
Atomic table removal      != Complete outcome
Restarted broker          != Recovered grant
InvocationCell ALLOW      != ExecGrantCell ALLOW
Parity proof              != Semantic authority
Semantic fixture          != Material attachment
```

## Write-Time Shape

Action `9030` validates the proposed transition before any mutation. The shape
is closed over operation, state, next state, generations, hashes, deadline,
budget, principal, peer, command, old-state receipt, proposed receipt, and
legal edge. Unknown or incomplete data refuses before grant-table, barrier,
process, obligation, or journal mutation.

This adopts the useful SHACL boundary without making RDF or a second schema
engine authoritative. Sounio owns both the rule and its decision.

## Affirmative Extinction

Extinction is represented as positive evidence about a negative condition. It
requires one terminal receipt binding:

1. observed absence of the exact handle after an atomic terminal transition;
2. retirement of broker, lease, custody, invocation, and grant generations;
3. revoked barrier, descriptor, and grant authority at the exact journal head.

A missing file, missing heartbeat, closed socket, dead pane, empty process
listing, timeout, failed observer, or restart is only `UNKNOWN`. It cannot
satisfy the triple. Uncertainty poisons and quarantines the generation.

## Supported Claim

The frozen Sounio executable defines `ISSUE`, `CONSUME`, `CLOSE`, and `REVOKE`,
all expected decisions, named refusals `491` through `501`, malformed refusal
`424`, and eleven causal single-rule sabotages. The deliberate Python-oracle
witness returns `DENY499`, and its executable sentinel is not reached.

The deterministic freeze binds the Garden commit, executable commit, parent
actions `9029`, `9021`, and `9022`, source, adapter, fixtures, toolchain,
hardware, command, result, and canonical decisions. It reports
`material_grant=false`, `same_uid_peer_isolation=false`, `parity_open=false`,
and every attachment flag false.

## Falsifier

The semantic claim is falsified if a copied handle becomes sufficient
authority, peer authentication occurs after lookup, an illegal write mutates
state before refusal, a restart reconstructs an outstanding grant, silence
satisfies extinction, an incomplete outcome closes, a non-Sounio producer
supplies an accepted expected result, or any unchanged unsafe witness remains
refused after removing its one intended rule.

## Pending Interface

The existing OCaml `EXEC_ISSUE`, `EXEC_CONSUME`, and `EXEC_OUTCOME` state machine
must be extended rather than replaced. It must consume the exact frozen action
`9030`, validate before mutation, preserve wrong-peer non-burning refusal,
record an explicit consuming generation, materialize terminal extinction, and
bind the root broker's `PrincipalCapsule` and `InvocationCell` receipts.

The transitory C++ broker may retain privileged pidfds, barriers, and material
observations but cannot encode expected results. The first material gate must
run on a host with kernel-distinct lane principals and measured anti-injection
posture. Until then, general Exec/Bash, child execution, commit, and CI remain
closed.

## Forbidden Claims

- Sounio values are statically affine or linear.
- Same-UID ancestry alone isolates actively hostile principals.
- A semantic or OCaml parity fixture proves kernel material behavior.
- Action `9030` opens a barrier, executes a command, or attaches a hook.
- A handle, token, capsule, PID, pidfd number, digest, or receipt grants
  authority by possession.
- This project-local construction establishes external novelty or priority.
