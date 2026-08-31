<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-31-spark-pair-arbiter
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-31-spark-pair-arbiter
-->

<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-31-spark-pair-arbiter
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-31-spark-pair-arbiter
-->

# Garden Seed: Spark Pair Arbiter

Status: GARDEN
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
- one Kubernetes GPU reservation and one NVML receipt per Spark;
- heartbeat, rollback, recovery, decision receipts, and negative tests.

It does not:

- download or execute Inkling;
- alter LiteLLM;
- exercise the Multus fabric or NCCL;
- change Pireus operator semantics;
- claim that the current cluster has already passed the mutual-exclusion gate.

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
-> MATERIAL_BRIDGE_OPEN
-> MUTUAL_EXCLUSION_GATE
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
- exactly two reservation pods, one on each canonical Kubernetes node;
- both reservations bound to the current epoch;
- exactly one GPU allocated per reservation;
- two clean NVML receipts from the same epoch;
- one successful Lease compare-and-swap committing the pair.

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

The material scheduler fence has two sides:

1. under Slurm ownership, the two `slurmd` pods consume the two Kubernetes GPU
   resources;
2. under Kubernetes ownership, the Slurm nodes remain drained and the
   `slurmd` selector fence removes both pods before reservations begin.

The Spark nodes also require a dedicated `NoSchedule` taint tolerated only by
the Slurm workers, arbiter probes, and admitted Spark-pair workloads.

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
- NVML unavailable, wrong node/UUID, or active GPU process;
- live or terminating workload during release;
- heartbeat loss or incomplete resume.

## Open questions

- The current Spark nodes have no DCGM exporter and the current `slurmd` image
  lacks a usable `nvidia-smi`. The material bridge therefore needs an exclusive
  per-node reservation probe that obtains the GPU and executes the host NVIDIA
  toolchain before `K8S_OWNED` can be admitted.
- Root processes launched outside both Kubernetes and Slurm are outside the
  initial threat boundary. A later host fence agent would be required for that
  stronger claim.
- Enabling MIG or device-plugin time sharing invalidates the one-GPU-one-owner
  proof and must fail the preflight gate.
