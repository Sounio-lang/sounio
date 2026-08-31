<!-- docs:meta
topic_id: repo.docs.ops.spark-pair-arbiter
authority: repo_only
audience: users
last_validated: 2026-08-31
validated_by: Codex
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ops.spark-pair-arbiter
-->

# Spark Pair Arbiter

Status: semantics frozen; offline gates required; live installation not yet
claimed.

The arbiter gives the two DGX Spark nodes to exactly one scheduler at a time.
Sounio is the executable transition authority. Bash transports observations and
effects, Kubernetes admission blocks unauthorised Pods, and the Slurm NodeSet
owns one Kubernetes GPU per `slurmd` while Slurm is active.

## Canonical pair

| Kubernetes | Slurm | GPU |
|---|---|---|
| `spark-3c59` | `gpuorangefs-multi-spark-3c59` | 1 GB10 |
| `spark-8e54` | `gpuorangefs-multi-spark-8e54` | 1 GB10 |

The exact node, NodeSet, device-plugin, source, policy, backend, admission and
native-executable hashes are frozen in
`tools/cluster/spark_pair_arbiter.freeze.v1`.

## Gate order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> OFFLINE_NEGATIVES_GREEN
-> INSTALL_REVIEWED
-> MUTUAL_EXCLUSION_LIVE_GATE
```

Run the non-mutating checks first:

```bash
bash scripts/ci/spark_pair_arbiter_selftest.sh
bash scripts/dev/install_spark_pair_arbiter.sh --check
bash scripts/ci/spark_pair_arbiter_live_gate.sh --check
```

Installation and the live gate are separate operator decisions. The live gate
will not install an absent arbiter:

```bash
bash scripts/dev/install_spark_pair_arbiter.sh --apply
bash scripts/ci/spark_pair_arbiter_live_gate.sh --apply
```

Do not run either `--apply` while the frozen hashes, admission server dry-run,
offline negatives, or an adversarial review are red.

## Bootstrap

The initial bootstrap is deliberately Lease-first. Action 28 creates the
`UNINITIALIZED` Lease before writing the bootstrap journal, so a process crash
always leaves a Kubernetes-visible exclusion anchor. If the crash happens
between those effects, action 27 takes over only after Lease expiry and
reconstructs the missing journal before completing the fenced recovery.

The canonical controller refuses all runtime path overrides and refuses
`fixture-v1`. The self-test executes fixture mode only through a copied binary
named `spark-pair-arbiter-fixture`. A principal that can copy or replace local
controller files already has host-administrator capability and is outside this
cluster arbitration boundary.

Sounio action 28 first authorizes creation of the `UNINITIALIZED` Lease and then
the bootstrap journal from a pair-exact, queue-empty prebootstrap frame.
Actions 24, 23, 25, and 26 then authorize, in order:

1. install fail-closed Pod and manual-binding admission, prove that a generic
   GPU Pod is denied, patch device-plugin tolerations, and census existing Pods;
2. drain both Slurm nodes only after admission and the pair taint are proved;
3. make the NodeSet request and limit one GPU and bind one `slurmd` per node;
4. resume Slurm only after both GPU owners are proved.

Action 1 finally commits `SLURM_OWNED`. Every material command requires the
current holder, epoch, Lease state and an action-bound Sounio receipt. A failed
bootstrap stays fenced and leaves a durable ConfigMap journal; it does not
attempt an unreceipted rollback.

The Lease and journal both pin the Sounio source hash and the complete semantics
freeze hash. A controller built from a different freeze refuses observation,
takeover, and material mutation; changing those bindings requires an explicit
migration rather than retrospective reinterpretation.

Resume an expired `UNINITIALIZED` bootstrap with:

```bash
SOUNIO_SPARK_PAIR_HOLDER="bootstrap-recovery-$(hostname)" \
  bash scripts/dev/spark_pair_arbiter.sh bootstrap-recover
```

## Acquisition and release

Acquisition drains Slurm, proves zero jobs and allocations, removes both
`slurmd` Pods, creates one exact GPU reservation on each node, and records GPU
UUID, driver, process, MPS, utilisation and memory evidence before committing
`K8S_OWNED`.

The two canonical GB10s expose unified memory and return `[N/A]` for
`nvidia-smi memory.used`. The frozen policy records this as
`UNAVAILABLE_UNIFIED` only for the exact two UUIDs, `NVIDIA_GB10` product and
driver `580.159.03`. It never substitutes for the process, `pmon`, MPS or
utilisation probes.

Release stops pair workloads, reprobes both GPUs, deletes the reservations,
restores both GPU-bound `slurmd` Pods, resumes Slurm, and commits
`SLURM_OWNED`. Lease expiry never resumes Slurm.

Manual recovery is:

```bash
SOUNIO_SPARK_PAIR_HOLDER="recovery-$(hostname)" \
  bash scripts/dev/spark_pair_arbiter.sh recover
```

Sounio decides recovery on the pre-takeover frame. Only its receipt authorizes
the Lease CAS, epoch increment and node annotation update.

## Threat boundary

Phase 1 denies ordinary Pods using `nodeName`, the dedicated toleration,
generic GPU requests during bootstrap, manual bindings, stale reservation
epochs, spoofed scheduler roles, or the future workload role.
Exact infrastructure service accounts remain admitted so CNI, CSI, metrics and
device-plugin DaemonSets can operate. A cluster administrator can alter or
delete admission itself and is outside this phase's threat model. Root GPU
processes created outside Kubernetes and Slurm are detected at handoff, but a
persistent host fence agent is required to prevent them proactively.

This phase does not download Inkling, change LiteLLM, run TP2, or change Pireus
operator semantics.
