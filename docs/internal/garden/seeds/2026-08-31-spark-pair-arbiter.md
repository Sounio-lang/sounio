<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-31-spark-pair-arbiter
authority: repo_only
audience: users
last_validated: 2026-09-01
validated_by: Codex
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-31-spark-pair-arbiter
-->

# Garden Seed: Spark Pair Arbiter

Status: SEMANTICS_FROZEN; PARITY_OPEN offline gates and ARM64 child-cgroup
device-barrier canary green; root install and live mutual-exclusion gate pending
Date: 2026-08-31
Owner: Sounio founder direction
Concept-ID: SOUNIO-SPARK-PAIR-ARBITER

## First phrase

Two schedulers must never be allowed to believe they own the same GB10.

## Problem

The Kubernetes NodeSet `slurm-pilot-worker-spark` manages two DGX Spark
workers:

- Kubernetes node `spark-3c59`, Slurm node
  `gpuorangefs-multi-spark-3c59`;
- Kubernetes node `spark-8e54`, Slurm node
  `gpuorangefs-multi-spark-8e54`.

Each node exposes one `nvidia.com/gpu`. The live `slurmd` pods advertise one
Slurm GRES `gpu:gb10:1`, are privileged, and can reach the NVIDIA devices, but
currently request zero Kubernetes GPUs. Kubernetes consequently reports both
GPUs free while Slurm also reports both workers idle and usable.

A Kubernetes Lease by itself is not a material fence. It serializes arbiters,
but it does not remove device access from `slurmd`, stop Slurm scheduling, or
prevent an old holder from acting after losing authority.

## Boundary

This seed covers only the Spark Pair Arbiter:

- one Kubernetes Lease with a monotone fencing epoch;
- pair-wide Slurm drain and resume;
- material detachment and restoration of both `slurmd` pods;
- one persistent host watchdog and one boot-bound host receipt per Spark;
- immutable content-addressed host-fence and reservation-probe programs;
- one content-addressed C++20 raw-BPF device barrier, compiled on each ARM64
  Spark as a transient material mechanism rather than semantic authority;
- one Kubernetes GPU reservation per Spark, with NVML retained as telemetry;
- heartbeat, rollback, recovery, decision receipts, and negative tests.

It does not:

- download or execute Inkling;
- alter LiteLLM;
- exercise the Multus fabric or NCCL;
- change Pireus operator semantics;
- claim that the current cluster has already passed the mutual-exclusion gate.

On 2026-09-01 the frozen C++ helper was compiled natively with GCC 13.3.0 on
both canonical ARM64 Sparks running Linux 6.17.0-1021-nvidia. A transient canary
rejected the cgroup root as a target, canonically resolved each Pod's strict
child cgroup, and attached the helper through an FD-scoped BPF link. It denied
`mknod` for compute majors 498 and 501 while leaving shared NVIDIA major 195 and
DRM major 226 allowed, proved exact baseline restoration after an injected
post-deny failure, then repeated the successful path. Both
nodes produced binary SHA-256
`f7f087cf2015004d90f49e35d76b1c6473ea84413cf167370a4b98de17e870fd` and BPF
tag `bfdb0f3533dc586c`. No root-cgroup attachment was attempted.

The child canary originally denied the complete inventory. Read-only host
evidence then showed majors `195` and `226` are both held by Xorg/GNOME on both
Sparks and that `247` is not part of the compute fence. The production profile
therefore keeps exact inventory `195,226,247,498,501` while denying only NVIDIA
compute majors `498,501`; a future root fence must leave `195` and `226`
available. This canary proves denial of new access, not eviction of an existing
process with an already-open device, so consumer quiescence remains a separate
host-fence gate.

## Authority

Sounio is the executable authority for transition admission. Host tooling may
observe Kubernetes, Slurm, Slinky, NVML, processes, hashes, and timeouts, then
submit a complete numeric fact frame to the frozen Sounio policy. The host
bridge may materialize an allowed transition, but it may not invent a state,
reason code, expected result, or fallback admission.

The required order is:

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

## Material invariant

At every observable boundary the pair has exactly one material disposition:

```text
SLURM_OWNED | K8S_OWNED | RECOVERY_REQUIRED
```

Transient states authorize no user workload. An unknown, partial, timed-out,
stale, or contradictory observation is a refusal and moves toward
`RECOVERY_REQUIRED`; it is never interpreted as availability.

`SLURM_OWNED` requires both `slurmd` pods to request and limit exactly one
`nvidia.com/gpu`, one on each canonical Spark. A NodeSet in DaemonSet mode does
not scale to zero through `spec.replicas`; the arbiter instead uses a dedicated
node-selector fence to remove or restore both pods.

`K8S_OWNED` requires:

- both canonical Slurm nodes drained;
- zero active Slurm jobs, steps, allocations, prologs, and epilogs;
- both `slurmd` pods absent;
- both boot-bound host receipts linked to the current Lease;
- both local watchdogs bound to the same source, freeze, epoch and owner;
- the root-cgroup device barrier attached exactly in `FENCED` and detached
  exactly under a valid `SLURM` or `K8S` grant;
- one durable Lease intent binding both non-authorizing prepare receipts before
  either host grant becomes usable;
- exact known Docker/systemd consumers fenced and managed cgroups empty;
- protected CPU, network, database and checkpoint resources unchanged;
- exactly two reservation pods, one on each canonical Kubernetes node;
- both reservations bound to the current epoch;
- exactly one GPU allocated per reservation;
- one successful Lease compare-and-swap committing the pair.

Every `SANDBOX_READY` Slurm or Pireus Pod must have a canonical CRI identity
and exactly one cgroup-v2 slice. The fence uses only atomic `cgroup.kill`; a
missing or ambiguous slice is evidence failure, not permission to enumerate and
kill PIDs.

NVML UUID, driver, process, MPS, utilisation and unified-memory observations
remain in receipts, but do not decide ownership on GB10.

TP1 is never a fallback for a failed TP2 acquisition.

## State machine

```text
UNINITIALIZED
  -> SLURM_OWNED
  -> DRAINING_SLURM
  -> SLURM_QUIESCENT
  -> DETACHING_SLURMD
  -> K8S_RESERVING
  -> K8S_OWNED
  -> K8S_RELEASING
  -> VERIFYING_GPU_CLEAN
  -> SLURM_RESTORING
  -> SLURM_OWNED

any ambiguous or failed transition -> RECOVERY_REQUIRED
```

Lease expiry never resumes Slurm automatically. A recovery holder must obtain
a new epoch and prove the release facts again.

## Fencing

The epoch is an explicit monotone integer stored with the Lease. Kubernetes
`resourceVersion` is only a compare-and-swap token and is not the epoch.
`leaseTransitions` is audit metadata and is not the sole epoch source.

Every reservation, workload, decision receipt, drain reason, node label, and
NVML receipt carries the same epoch. An old holder can remain alive, but its
Lease CAS must fail and its epoch must not authorize new work.

The material scheduler fence has four sides:

1. under Slurm ownership, the two `slurmd` pods consume the two Kubernetes GPU
   resources;
2. under Kubernetes ownership, the Slurm nodes remain drained and the
   `slurmd` selector fence removes both pods before reservations begin.
3. on each host, a persistent systemd watchdog admits only the local grant
   matching boot, source, freeze, epoch and owner; K8s grants expire against
   monotonic time and fence locally.
4. a raw `BPF_CGROUP_DEVICE` program on the root cgroup blocks new GPU-device
   opens before cleanup and at boot, including host processes outside
   `kubepods.slice`; existing descriptors still require termination and census.

The Spark nodes also require a dedicated `NoSchedule` taint tolerated only by
the Slurm workers, arbiter probes, and admitted Spark-pair workloads.
Creation and update of Pods carrying that toleration or targeting either node
directly is guarded by a fail-closed `ValidatingAdmissionPolicy`. Phase 1
admits no user workload role; the future Inkling role remains denied.
The control policy protects both Node objects and the privileged host-fence
`pods/exec` bridge as well: non-authority updates
may change unrelated metadata, but must preserve the Pireus taint, host-fence
labels and Slurm selector label exactly.

## Refusal surface

The Sounio policy must refuse at least:

- policy missing, policy error, or observation timeout;
- malformed frame, unknown state, or unsupported action;
- foreign holder, stale/regressed epoch, or dead Lease;
- pair identity or NodeSet generation drift;
- missing or non-exclusive GPU resource configuration;
- partial drain, active job, active step, or nonzero allocation;
- only one `slurmd` removed or restored;
- only one reservation, two reservations on one node, or wrong epoch;
- wrong host boot/receipt, epoch, owner, watchdog, inventory, cgroup or memory;
- unexpected Docker, systemd, Kubernetes, Slurm or root GPU consumer;
- live or terminating workload during release;
- heartbeat loss or incomplete resume;
- a missing, foreign, stale, or wrongly attached root-cgroup device barrier.

## Open questions

- The canonical GB10s report `[N/A]` for framebuffer memory because they use
  unified memory. Phase 1 therefore treats NVML as supplementary and proves
  host ownership from exact process, service, container and cgroup identity.
- The boot unit is required by `basic.target`, Docker, containerd, and kubelet.
  It attaches the root-cgroup device barrier before invalidating grants. The
  continuous watchdog advances its public heartbeat only after a complete
  successful cycle.
- `/dev/watchdog` exists on both Sparks, but phase 1 intentionally leaves the
  hardware watchdog disarmed. Loss of coordination fences GPU access; it does
  not force a physical reboot of WiFi nodes outside the UPS.
- A deliberate host administrator can replace the fence and remains outside
  the boundary. Undeclared root GPU processes are inside it and must be denied.
- Enabling MIG or device-plugin time sharing invalidates the one-GPU-one-owner
  proof and must fail the preflight gate.
- Phase 1 still has no material decommission dispatcher. The separate Sounio
  frame `9026` now defines an effect-free plan; deleting the DaemonSet must not
  silently re-enable legacy GPU services.

## Reversible decommission seed

The phrase to preserve is: **removing the DaemonSet is not decommission**.
Decommission is a custody transfer from the active scheduler arbiter to one
explicit terminal owner. The executable concept
`SOUNIO-REVERSIBLE-COMPUTE-CUSTODY` defines that owner as
`LEGACY_HOST_OWNED`, with Slurm drained and its GPU workers absent, Kubernetes
GPU capacity withdrawn, and only the exact snapshotted legacy services restored.

The frozen plan path is:

```text
SLURM_OWNED
  -> DECOMMISSION_DRAINING
  -> DECOMMISSION_FENCED
  -> SCHEDULERS_WITHDRAWN
  -> LEGACY_RESTORING
  -> LEGACY_HOST_OWNED
```

A failure enters decommission-specific recovery and can return only to a fenced
pair. It cannot restore either scheduler or a legacy service from absence,
timeout, Lease expiry, or a one-node observation. Recommission is the reverse
Sounio-authorized transfer, not manual reconstruction after deletion.

The Garden phase is complete. Frame `9026` now has 69 executable Sounio vectors,
a first-executable receipt, a semantic freeze, and bounded post-freeze parity.
Lean proves the structural custody invariants, Koka proves the planning action
namespace is pure, and C++ proves that the only current material interpretation
is a no-op consumer. This is not full guard equivalence and it is not an
uninstall script: every result says `effect=NONE`, and the unchanged frame
`9025` controller rejects its schema, action namespace, and output prefix.

`CLAIM_READY` remains blocked until the Mac-side inventory supplies the exact
two-node pre-install service and restart-policy snapshot and replay evidence
exist. A material decommission dispatcher remains deliberately absent.

The full draft semantic contract is
[`spark-pair-reversible-decommission.md`](../../concepts/spark-pair-reversible-decommission.md).

## Pireus Restore Capsule v1

The next prerequisite now has its own effect-free Sounio surface. Frame `9027`
defines an immutable binding between an ordered two-node manifest, the two
canonical node manifests and a deterministic offline replay witness. Its first
phrase is: **decommission is reversible only when the state we intend to
restore has an identity before anything is removed**.

The evidence order is:

```text
GARDEN
-> SOUNIO_CAPSULE_EXECUTABLE
-> CAPSULE_SCHEMA_FROZEN
-> READ_ONLY_CAPTURE
-> OFFLINE_REPLAY_PROVEN
-> LIVE_REPLAY_GATE
```

The current lane reaches `CAPSULE_SCHEMA_FROZEN` with 77 executable Sounio
vectors and opens parity. It does not claim capture or replay. The adapter
accepts only complete lowercase 64-hex SHA-256 identities, converts them to
eight 32-bit limbs and emits
`effect=NONE`. It neither reads hosts nor computes a snapshot.

The schema freezes UTF-8, LF, fixed field order, normalized decimal and path
representations, unique keys and four distinct digest domains: ordered pair,
node 0, node 1 and replay witness. A lowercase hash is necessary but not
sufficient; explicit facts also bind the algorithm, canonical bytes and
domain separation.

Current host observations cannot satisfy historical pre-install provenance.
Boot IDs, PIDs, cgroups, Kubernetes UIDs, monotonic timestamps and BPF link IDs
are boot-scoped observations, not restore targets. The current protected-path
baseline is also insufficient because a new boot may replace it with the
current inode identity and canonize drift after power loss.

Live snapshot, historical pre-install receipt, material capture/replay,
live restore/recommission, a bridge into frame `9026`, cluster credentials,
external network, full material parity and `CLAIM_READY` remain `NOT_PRESENT`.
The frame transports the exact parent `9026` freeze digest and a predecessor
receipt digest; neither identity is represented by a bare boolean alone.
