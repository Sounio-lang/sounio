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

## Read-Only Capture Interface

Read-Only Capture v1 is a strict producer profile for frame `9027`; it does
not add a frame or relax a restore fact. Sounio is the first executable
authority and freezes four canonical material schemas: the 38-field node
manifest, 17-field post-install restorable candidate, 29-field boot-scoped
observation and 16-field ordered pair manifest. It also freezes the 22-domain
prefix-free framing used by the native SHA-256 collector.

The complete current observation projects to action `51`, state `20`,
authority mask `473`, snapshot mask `491455` and replay mask `0`. Frame `9027`
is invoked and refuses the candidate with `PREINSTALL_PROVENANCE/315`. The
current manifests deliberately say:

```text
capture_temporality=CURRENT_POSTINSTALL_OBSERVATION
historical_preinstall_receipt=NOT_PRESENT
protected_content_receipt=NOT_OBSERVED
restorable=false
snapshot_binding_receipt=NOT_ISSUED
state_transition=false
```

The C++20 material observer was compiled natively and produced byte-identical
fixtures on both ARM64 Sparks. Its source SHA-256 is pinned and checked before
any privileged transport. Its fixed domain enum and SHA-256 implementation
produce the observation digests; the shell gate only transports fixed queries
and bytes. The Multus bridge must report the same pinned runtime `imageID`
digest on both nodes, not merely the expected mutable image tag. Kubernetes API
GETs remain the material source for NodeSet,
device-plugin, taint, label and node identity. Slurm observation uses only the
existing login Pod and read-only `scontrol`, `squeue` and `sinfo` queries.

The accepted live evidence is:

- profile freeze:
  `3edfa1e7394b8e82ce8d5e4c81e0450b88dc5b72e1eb71c6acf33f6e2c705223`;
- ordered pair receipt:
  `deb5285f1cf1a2e46b8cdf49d4040419b9a6fe57eb3385ddd1197b56a105b6eb`;
- material parity receipt:
  `235b85efcd7be87db6e073773bf51eb09caf8e2e8869cade4e80f0d0a623781d`;
- both node manifests and observations bind `scheduler_mutation=NONE`,
  `host_configuration_mutation=NONE` and equal pre/post managed-state
  sentinels;
- the `/dev/shm` source and binary materializations were removed and verified
  absent on both hosts before the receipts were published.

An earlier pair receipt,
`c0f6235dc7b93aca8d674ba66c28d66ea34d00b0bcd5b36904cef8b8891120a9`,
is rejected and explicitly superseded. It domain-separated the pre and post
sentinels, making equality impossible, while its aggregate receipt still said
`NONE`. The corrected gate uses one frozen sentinel domain and refuses to
publish unless each node manifest, each observation and the aggregate decision
agree on the exact mutation state.

`pods/exec` creates processes and Kubernetes audit events, so the producer
effect is `READ_ONLY_OBSERVATION`, not `NONE`. Native compilation makes a
bounded ephemeral `/dev/shm` write. Neither effect changes scheduler or host
configuration and neither is a decommission or restore dispatch.

Executable surfaces:

- `stdlib/coordination/spark_pair_read_only_capture_profile.sio`;
- `tools/cluster/spark_pair_read_only_capture_profile_main.sio`;
- `tools/cluster/spark_pair_read_only_capture_profile.freeze.v1`;
- `tools/cluster/spark_pair_read_only_capture.cpp`;
- `scripts/dev/spark_pair_read_only_capture_arm64_gate.sh`;
- `scripts/ci/spark_pair_read_only_capture_profile_selftest.sh`;
- `scripts/ci/spark_pair_read_only_capture_material_selftest.sh`;
- `tools/cluster/spark_pair_read_only_capture.material-parity.v1`.

This closes material parity only for current read-only observation. It does not
open offline replay, supply protected content, create historical pre-install
provenance, promote a fact into frame `9026`, dispatch decommission, or make
the concept `CLAIM_READY`.

## Historical Provenance Authority

Frame `9028` closes the authority gap behind the single
`preinstall_provenance_exact` bit in frame `9027`. It does not modify frame
`9027`. Instead, it defines a separate effect-free Sounio decision whose only
admitted terminal state is `HISTORICAL_PROVENANCE_ADMITTED`.

The frame binds seven distinct content identities: source bundle, two ordered
node sources, pair manifest, first-install anchor, parent frame `9027` freeze
and predecessor receipt. Admission also requires complete facts for
cryptographic ordering, evidence custody and restore coverage. Custody here is
the custody chain of the historical evidence, not ownership of live GPU
compute; compute exclusivity remains under frames `9025` and `9026`.

Source classes `1..4` may be admitted: a frozen Sounio pre-install receipt, an
immutable two-node machine snapshot, a canonical pre-install export bundle or
a conflict-free composite closure. Class `5`, partial pre-install backup, may
be bound as a fragment but cannot be admitted alone. Current/post-install,
mutable or clock-only, and review-only sources are refused as classes `6..8`.
An external LLM can review a receipt but cannot produce its authority facts.

The exact frozen surface is:

- authority: `stdlib/coordination/spark_pair_historical_provenance.sio`;
- frame: `9028`, actions `54..56`, states `30..32`;
- executable vectors: 95, including every ordering, custody and completeness
  bit, predecessor-transcript substitution, digest aliasing and schema crossing;
- first executable receipt:
  `tools/cluster/spark_pair_historical_provenance.first.v1`;
- semantic freeze:
  `tools/cluster/spark_pair_historical_provenance.freeze.v1`, SHA-256
  `79b4ea331f78ac3abc8bee9e295b0411d1dad1ab18d9264ddebe4a8f3c7d43ea`;
- source assessment:
  `tools/cluster/spark_pair_historical_provenance.source-assessment.v1`,
  SHA-256
  `a84e418c33784d47393e72ed32228fb066216cda9d8440c81927e4713a0b66a7`;
- gate: `scripts/ci/spark_pair_historical_provenance_selftest.sh`.

The read-only assessment found no admissible class `1..4` source. The retained
etcd and Slurm database backups are class `5` component evidence without the
two node leaves, host services, protected payloads or ordered pair closure.
The existing pair capture is class `6` and is refused with
`POSTINSTALL_SOURCE/354`. Slurm live accounting, ext4 timestamps and live
Kubernetes objects are class `7` clock or mutable evidence.

There is also a concrete temporal falsifier for reconstructing the complete
current surface as if it were pre-mutation history. The first observed Slurm
DRAIN is `2026-08-30T17:58:11Z`; the current content identity of the required
`vxlan-cluster.service` on `spark-3c59` reflects a Claude recovery edit at
`2026-08-31T08:38:44Z`, and protected paths on `spark-8e54` were created after
that first DRAIN. The edit timestamp is not an original installation time.
This evidence does not identify the exact first Pireus install or action; it
proves only that the observed fragments cannot form a complete pre-mutation
image of the current required surfaces.

The persistent Pireus fence has not been installed. If that narrower event is
chosen as the boundary, the current capture may be proposed as pre-fence, but
frame `9028` still refuses it with `FIRST_INSTALL_ANCHOR/349`; it also lacks
protected payloads and coverage closure. Against the earlier cluster boundary,
the producer's `CURRENT_POSTINSTALL_OBSERVATION` classification remains exact.
Boundary ambiguity cannot be resolved by choosing whichever label admits the
same incomplete bytes.

Therefore frame `9028` remains in `HISTORICAL_SOURCE_EMPTY`, no strict
projector into frame `9027` exists, `preinstall_provenance_exact` remains false,
and frame `9027` continues to return `PREINSTALL_PROVENANCE/315`.
`OFFLINE_REPLAY` is closed by construction.
