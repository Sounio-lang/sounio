<!-- docs:meta
topic_id: repo.docs.ops.spark-pair-arbiter
authority: repo_only
audience: users
last_validated: 2026-09-01
validated_by: Codex
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ops.spark-pair-arbiter
-->

# Spark Pair Arbiter

Status: Sounio frame 9025 executable; semantics frozen; offline gates and the
ARM64 child-cgroup device-barrier canary are green; root installation and live
mutual exclusion are not yet claimed.

The arbiter gives the two DGX Spark nodes to exactly one scheduler at a time.
Sounio is the executable transition authority. Bash transports observations and
effects, Kubernetes admission blocks unauthorised Pods, and the Slurm NodeSet
owns one Kubernetes GPU per `slurmd` while Slurm is active.

## Canonical pair

| Kubernetes | Slurm | GPU |
|---|---|---|
| `spark-3c59` | `gpuorangefs-multi-spark-3c59` | 1 GB10 |
| `spark-8e54` | `gpuorangefs-multi-spark-8e54` | 1 GB10 |

The exact node, NodeSet, device-plugin, source, policy, backend, admission,
host-fence manifest and native-executable hashes are frozen in
`tools/cluster/spark_pair_arbiter.freeze.v1`.

The executable host fence, transient C++ device-barrier source, and reservation
probe are stored in immutable, content-addressed ConfigMaps. Their names carry
the first 12 hexadecimal digits of the exact source SHA-256, and the material
backend recomputes each binding before accepting the objects.

## Gate order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

`PARITY_OPEN` contains the offline negatives, install review and live
mutual-exclusion gate. `CLAIM_READY` remains false until all three are green.

Run the non-mutating checks first:

```bash
bash scripts/ci/spark_pair_arbiter_selftest.sh
bash scripts/dev/spark_pair_device_barrier_arm64_gate.sh --check
bash scripts/dev/install_spark_pair_arbiter.sh --check
bash scripts/ci/spark_pair_arbiter_live_gate.sh --check
```

The ARM64 gate is a separate, transient material-parity probe. Its `--apply`
mode compiles the frozen C++20 helper natively on both Sparks, resolves and
canonically proves a target strictly below `/sys/fs/cgroup`, and attaches an
FD-scoped BPF link only there. It proves `mknod` denial for the frozen NVIDIA/DRM
majors, injects a post-deny failure and proves link-close restoration, repeats
the successful path, and deletes its content-addressed ConfigMap and Pods. The
orchestration image is pinned by digest; the compiler and resulting helper
execute from the host root:

```bash
bash scripts/dev/spark_pair_device_barrier_arm64_gate.sh --apply
```

This evidence does not authorize installation on `/sys/fs/cgroup` and does not
replace the live pair-wide exclusion gate.

Root installation and the live gate are separate operator decisions. The live
gate will not install an absent arbiter:

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
Actions 24, 29, 23, 30, 31, 25, and 26 then authorize, in order:

1. install fail-closed Pod and manual-binding admission, prove that a generic
   GPU Pod is denied, patch device-plugin tolerations, and census existing Pods;
2. install the persistent host watchdog in non-mutating `ARMED` mode and bind
   one cryptographic host receipt per boot to the Lease;
3. drain both Slurm nodes only after admission and the pair taint are proved;
4. activate the pair-wide host fence only after zero jobs and allocations are
   observed on the drained pair;
5. grant host access to Slurm, make the NodeSet request and limit one GPU, and
   bind one `slurmd` per node;
6. resume Slurm only after both GPU owners are proved.

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

Acquisition drains Slurm, proves zero jobs and allocations, activates the local
fence on both hosts, removes both `slurmd` Pods, grants a 30-second monotonic
Kubernetes epoch on both hosts, then creates one exact GPU reservation on each
node before committing `K8S_OWNED`. During `K8S_RESERVING`, the controller
preserves the `BEGIN_RESERVE` receipt and refreshes the paired host grant with
new Sounio action-32 receipts. Loss of that refresh aborts acquisition. A
partial two-host grant revokes both sides and enters recovery.

Each pair grant is a two-phase transaction. Both hosts first emit
non-authorizing prepare receipts. The controller commits the transaction ID,
Lease UID, base resource version and both receipts to the Lease before either
host can activate a grant. Each host recomputes the pair digest and binds its
grant to the durable intent resource version. A killed controller cannot
produce scheduler effects from a partial commit; a final CAS conflict requires
both hosts to report `FENCED` before the operation returns.

The two canonical GB10s expose unified memory and return `[N/A]` for
`nvidia-smi memory.used`. The frozen policy records this as
`UNAVAILABLE_UNIFIED` only for the exact two UUIDs, `NVIDIA_GB10` product and
driver `580.159.03`. NVML is supplementary telemetry and is not an ownership
gate. The material gate uses exact Docker configuration, systemd units, Pod
UID/cgroup ownership, process sets, memory floor and protected-resource state.
Every managed `SANDBOX_READY` Pod must have a canonical CRI UID and map to
exactly one cgroup-v2 slice. Fencing requires the atomic `cgroup.kill` file;
missing or ambiguous mappings fail closed and there is no PID-by-PID fallback.
The host fence also attaches a raw `BPF_CGROUP_DEVICE` program to the root
cgroup before changing the local record to `FENCED`. It denies new opens for
the exact frozen GPU-related majors, including processes launched outside
Kubernetes through `systemd-run`. BPF does not revoke descriptors already
open, so service shutdown, `cgroup.kill`, and the `/proc` descriptor census
remain independent mandatory gates.

Release stops pair workloads, fences both hosts, deletes the reservations,
grants Slurm on both hosts, restores both GPU-bound `slurmd` Pods, resumes
Slurm, and commits `SLURM_OWNED`. Lease expiry never resumes Slurm. K8s grant
expiry fences locally even when the control-plane path is unavailable.

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
epochs, spoofed scheduler roles, or the future workload role. A persistent
systemd boot unit attaches the root-cgroup device barrier before `basic.target`;
Docker, containerd, and kubelet each require that unit. Only then does it
invalidate any old grant. The continuous watchdog starts after those runtimes are
observable, refreshes the protected-resource baseline for the current boot,
and publishes a heartbeat only after a complete successful cycle. It gates the
known Ollama, Beagle embedding and TEI consumers without restarting those
runtimes. It never stops `vxlan-cluster`, Docker, containerd, kubelet, the
Beagle Postgres data path, or the Sounio checkpoint/toolchain paths. The
hardware watchdog is deliberately not armed in phase 1.

Exact infrastructure service accounts remain admitted so CNI, CSI, metrics and
device-plugin DaemonSets can operate. Their admitted Pod specs are exact and
GPU-free except for the two canonical device plugins. A separate fail-closed
control policy protects the admission objects, Lease, NodeSet, infrastructure
controllers and the pair Nodes. Non-authority Node updates may change unrelated
metadata but must preserve the Pireus taint and host/Slurm selector labels.
The same policy rejects `pods/exec` into the privileged host-fence Pods before
execution unless the caller is the canonical material controller identity. A
deliberate host administrator can replace the watchdog binary or kernel state
and remains outside this phase's threat model; ordinary root GPU consumers are
inside the root-cgroup barrier.

This phase does not download Inkling, change LiteLLM, run TP2, or change Pireus
operator semantics.

Removing the DaemonSet is not a decommission procedure. Phase 1 intentionally
keeps legacy GPU services stopped while either scheduler owns the pair, and no
unreceipted command may re-enable them. A reversible, Sounio-authorized
decommission action remains required before `CLAIM_READY` can become true.
