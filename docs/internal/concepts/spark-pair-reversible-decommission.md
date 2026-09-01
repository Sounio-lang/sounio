<!-- docs:meta
topic_id: repo.docs.internal.concepts.spark-pair-reversible-decommission
authority: repo_only
audience: users
last_validated: 2026-09-01
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.spark-pair-reversible-decommission
-->

# Spark Pair Reversible Decommission

Status: executable (`SEMANTICS_FROZEN`; bounded parity proven; `CLAIM_READY=false`)
Concept-ID: `SOUNIO-REVERSIBLE-COMPUTE-CUSTODY`
Authority: founder direction, bounded by executable Sounio and material receipts

## Definition

Decommission is an explicit transfer of compute custody, not deletion of the
controller that made custody visible. At every observable point the Spark pair
has exactly one of four custody modes:

- `FENCED`: no compute owner may open the protected devices;
- `SLURM`: only the exact GPU-bound Slurm pair may admit compute;
- `K8S`: only the exact epoch-bound Kubernetes pair may admit compute;
- `LEGACY_HOST`: only the content-addressed legacy host inventory may admit
  compute.

No missing, empty, expired, or deleted control object implies
`LEGACY_HOST`. Absence of coordination is `FENCED` until a Sounio receipt says
otherwise.

## Frozen Plan Path

```text
SLURM_OWNED
  -> DECOMMISSION_DRAINING
  -> DECOMMISSION_FENCED
  -> SCHEDULERS_WITHDRAWN
  -> LEGACY_RESTORING
  -> LEGACY_HOST_OWNED
```

Any incomplete step enters `DECOMMISSION_RECOVERY_REQUIRED`. Recovery may
return only to `DECOMMISSION_FENCED` until the pair-wide facts are complete.
It must not guess whether Slurm, Kubernetes, or a legacy service won a partial
race.

Recommission is the explicit reverse transfer. It begins by fencing and
quiescing the exact legacy inventory, reinstalls the admission and scheduler
surfaces from frozen identities, and reaches `SLURM_OWNED` only after the
existing bootstrap gates pass again. `LEGACY_HOST_OWNED` is therefore
reversible, not an untracked terminal escape.

## Required Ordering

1. Capture a two-node restore snapshot before first installation. It must bind
   service enablement, restart policy, container identity, NodeSet identity,
   device-plugin identity, taints, labels, boot identity, and protected paths.
2. Admit decommission only from a live `SLURM_OWNED` Lease with zero jobs,
   steps, allocations, reservations, Kubernetes workloads, and unexpected GPU
   consumers.
3. Drain both Slurm nodes, then establish the pair-wide `FENCED` grant.
4. Remove both GPU-bound `slurmd` Pods and both Spark device-plugin surfaces;
   prove Slurm remains drained and Kubernetes GPU capacity is zero.
5. Prepare the exact legacy restore on both hosts. A partial prepare grants
   nothing.
6. Commit one `LEGACY_HOST` grant for the pair, restore the snapshotted legacy
   services, and prove their exact consumer and restart-policy identities.
7. Action `41` may authorize removal of host boot gates, the root barrier,
   admission objects, and the active DaemonSet only after the schedulers are
   withdrawn and the pair-wide legacy receipt is durable. It remains in
   `LEGACY_RESTORING`; admission is removed last.
8. Lease-independent action `49` confirms `LEGACY_HOST_OWNED` only after the
   exact ordered absence of those arbiter surfaces is bound to the tombstone.
9. Retain a content-addressed decommission tombstone containing the frozen
   Sounio source hash, restore-snapshot hash, final owner, toolchains, hardware,
   commands, and both host receipts.

## Falsifiers

The executable contract is false or incomplete if any of these occur:

- a legacy service can start before both scheduler surfaces are withdrawn;
- one Spark reaches `LEGACY_HOST` while the other remains scheduler-owned;
- loss of the Lease or network implicitly restores a service;
- a process with an already-open GPU device survives a required quiescence
  gate and remains undeclared;
- decommission cannot be interrupted at every material step and recovered to
  `FENCED`;
- recommission requires an unreceipted manual repair;
- the restore snapshot cannot reproduce the measured pre-install state.

## What This Is Not

- It is not permission to install or remove the current fence.
- It is not evidence that the current host inventory is complete.
- It is not a material decommission dispatcher. Frame `9026` emits
  `effect=NONE`, and the frame `9025` controller cannot dispatch it.
- It is not a claim that denying new device access evicts an already-running
  CUDA process.
- It is not authority for Bash, Kubernetes, Slurm, C++, or an LLM to choose the
  final owner.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: pireus-decommission-garden-20260901
Owner: codex
Concept-IDs: SOUNIO-REVERSIBLE-COMPUTE-CUSTODY (proposed)
Intent-Preserved: decommission preserves exclusive and explicit compute custody instead of erasing the arbiter
Transformation: define reversible transfer from scheduler custody to exact legacy-host custody
Types-Changed: SparkPairDecommissionFacts and decommission plan states 11..19 in a frame 9026 extension; frame 9025 is byte-identical
Effects-Changed: none; every frame 9026 decision emits effect=NONE and no material dispatcher exists
IR-Changed: none
Claims-Introduced: frozen Sounio frame 9026 defines the only admissible reversible custody plan and refusal reasons
Claims-Forbidden: material dispatch; current decommission safety; implicit owner from deletion; eviction of already-open CUDA consumers; restore snapshot completeness; CLAIM_READY
Assumptions: the measured pre-install host state can be captured exactly and replay-verified without restarting protected runtimes
Write-Set: Sounio frame 9026 extension, adapter, vectors, executable receipt, freeze, parity-open receipt, this contract, Garden seed, and concept registry row
Read-Set: current Sounio authority, vectors, material policy, host fence, installer, backend, and live read-only host inventory
Positive-Witness: both hosts reach one exact LEGACY_HOST receipt and can recommission through the same frozen Sounio authority
Negative-Witness: partial restore, live job, live workload, stale Lease, snapshot drift, one-node commit, surviving undeclared GPU consumer, or failed recommission is refused
Acceptance-Gate: GARDEN(completed) -> SOUNIO_EXECUTABLE(completed) -> SEMANTICS_FROZEN(completed) -> PARITY_OPEN(completed) -> CLAIM_READY(blocked)
Integration-Target: Spark Pair Arbiter phase 2
Authoritative-Only-If: tools/cluster/spark_pair_decommission.first.v1 precedes and is hash-bound by tools/cluster/spark_pair_decommission.freeze.v1; no material decommission command exists
```

## Executable Evidence

Evidence-Pass: tests/fixtures/spark_pair_arbiter/spark_pair_decommission_vectors.sio
Evidence-Refuse: tests/fixtures/spark_pair_arbiter/spark_pair_decommission_vectors.sio

- authority: `stdlib/coordination/spark_pair_decommission.sio`;
- frame: `9026`, actions `33..49`, states `1` and `11..19`;
- expected decisions: 69 Sounio vectors;
- first executable receipt: `tools/cluster/spark_pair_decommission.first.v1`;
- semantic freeze: `tools/cluster/spark_pair_decommission.freeze.v1`;
- parity opening: `tools/cluster/spark_pair_decommission.parity-open.v1`;
- Lean structural parity: action 41 remains non-terminal, only action 49 enters
  legacy custody from a non-owned state, recovery exits only through `FENCED`,
  and every admitted structural plan has effect `NONE`;
- Koka effect parity: the complete action namespace `33..49` is inferred pure
  and maps only to `effect=NONE`;
- C++ material parity: the consumer accepts only the exact frame `9026`
  `effect=NONE` surface and exposes no process, filesystem, network, BPF,
  Kubernetes, or Slurm dispatch API;
- parity gate: `scripts/ci/spark_pair_decommission_parity_selftest.sh`;
- intentionally unproved: full cross-language guard/reason equivalence;
- intentionally absent: a material decommission implementation and any live
  cluster execution;
- current material effect: `NONE`;
- current blocking evidence: exact two-node live restore snapshot and replay are
  both `NOT_PRESENT`.

## Restore Capsule Interface

Frame `9027` is the content-addressed prerequisite surface for the two frozen
frame `9026` restore facts. It is a separate Sounio authority with states
`20..23`, actions `50..53`, 77 executable vectors and a unique output prefix.
Every decision has `effect=NONE` and no material consumer.

The frame transports four lowercase 64-hex SHA-256 content identities: ordered
pair manifest, node-0 manifest, node-1 manifest and offline replay witness. It
also transports the exact parent `9026` freeze identity and the predecessor
decision-receipt identity. Sounio converts each identity to eight canonical
32-bit limbs, checks the parent digest against the frozen constant, rejects
zero or reused identities where evidence is required, and admits only explicit
coverage masks.

Snapshot coverage includes pre-install provenance, system and user service
enablement, restart and linger relationships, Docker recreate identity,
NodeSet and device-plugin manifests, complete taints and labels, capture boot
identity, protected-path content receipts, toolchain/hardware/commands,
read-only capture and explicit absence for unknown fields.

Replay coverage requires an isolated filesystem root with no network, cluster
credentials, source-host mounts or privileged effects. It binds exactly two
replay nodes, the parent frame `9026` freeze, the service/container/scheduler
surfaces and a byte-identical repeated witness.

This is schema authority, not replay evidence. The current receipts say
`offline_replay_evidence=SCHEMA_ONLY_NOT_RUN`. No frame `9027` receipt promotes
`restore_snapshot_pair_exact` or `restore_snapshot_replay_verified` into frame
`9026`; a future bridge needs its own Sounio-first contract and material gate.

Executable surfaces:

- `stdlib/coordination/spark_pair_restore_capsule.sio`;
- `tests/fixtures/spark_pair_arbiter/spark_pair_restore_capsule_vectors.sio`;
- `tools/cluster/spark_pair_restore_capsule.first.v1`;
- `tools/cluster/spark_pair_restore_capsule.freeze.v1`;
- `tools/cluster/spark_pair_restore_capsule.parity-open.v1`;
- `scripts/ci/spark_pair_restore_capsule_selftest.sh`.
