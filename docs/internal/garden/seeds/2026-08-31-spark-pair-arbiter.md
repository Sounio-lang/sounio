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
`mknod` for majors 195, 226, 247, 498, and 501, proved exact baseline restoration
after an injected post-deny failure, then repeated the successful path. Both
nodes produced binary SHA-256
`427ae944d4bb2922930ffb23baf65aa65a128df355050b6079da0945c725acf3` and BPF
tag `539451426e7078af`. No root-cgroup attachment was attempted.

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
- Phase 1 has no Sounio-authorized decommission action yet. Deleting the
  DaemonSet must not silently re-enable legacy GPU services; reversible restore
  remains an explicit gate before `CLAIM_READY`.
