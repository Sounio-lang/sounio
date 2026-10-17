# Inkling on the Spark pair

Model and OCI identities are in runtime-lock.json. The derived ARM64 SIF SHA256
is 3dbfccad3355b27d8a09bd4c0c5895d02a43e1b64203960905b810bbb6bccbe3.
Large binaries and weights stay in persistent scratch/cache outside Git.

## Qualified runtime

The real launcher passed two-rank collectives under Slurm job 11854, using the
same SIF on both Sparks, PyTorch 2.11.0+cu130, CUDA 13.0, NCCL 2.30.7,
SM121 GPUs and NET/IB. Logs explicitly report GDR 0: RDMA transport with host
staging, not proven GPUDirect or NVLink. Jobs terminate with srun and leave no
resident model daemon. Model serving is a separate pending acceptance gate.

Run inside remote tmux from the workspace:

```sh
python3 tools/pireus/continuity/runtime/launch_pair.py qualify --minutes 10
python3 tools/pireus/continuity/runtime/launch_pair.py qualify-model --minutes 30
python3 tools/pireus/continuity/runtime/launch_pair.py serve --minutes 120
```

The launcher checks the Spark pair lease, resolves current worker pod identities
and IPs, assigns rank 0 to 3c59 and rank 1 to 8e54, checks the SIF hash in each
allocation, and uses exclusive srun with kill-on-bad-exit. Serving rechecks all
model bytes before launching SGLang; a stale qualification receipt cannot bypass
that check. Resource/time limits belong to the job. Text, context 16384,
concurrency 1, max output 4096 in the generation client, no speculative decoding.

## Installation prerequisites

The isolated non-setuid Apptainer 1.5.3 prefix and copied library hashes are in
runtime-lock.json. The narrowly scoped AppArmor profile is installed on both
hosts; its source is apparmor.pireus-apptainer-runtime.

Apptainer clears helper library environment. worker_prerequisites.py preserves
the original squashfuse_ll ELF and installs a wrapper invoking the ARM64 loader
with the isolated library directory. It also raises only each worker slurmd's
MEMLOCK limit from 8 MiB to unlimited. This live process change must be reapplied
after a worker pod restart; the launcher does so before allocation. Jobs use
--propagate=NONE to avoid reintroducing the submission shell's small limit.
It does not globally disable AppArmor or restart the nodes.

## Snapshot custody

stage_model.py resumes public Hub downloads at the pinned revision, retains
partial bytes, refuses corrupted completed files, and verifies SHA256 for LFS
and Git blob IDs for other files. The initial Kubernetes exec transport is
retained as distribute_model.py. For faster internal transport, serve_snapshot.py
exposes only completed manifest files and implements exact byte ranges;
fetch_snapshot.py resumes worker partial files. A completed transfer is labeled
UNQUALIFIED until qualify_model.py validates all pinned hashes.

The internal HTTP server is temporary and must be terminated after both
transfers finish. Its root is only this public snapshot, never the workspace.
No model/cache contents or credentials are committed. Known transport failures
remain in the session logs; successful qualification receipts are separate.

Validation: unittest discovery covers early EOF/range resumption, corruption
preservation and LFS/Git qualification refusal. Shell syntax and Python parsing
pass. Actual two-rank execution is in the committed validation logs.

## Bounded recovery detachment

recovery_detach.sio is a distinct, explicit native Sounio recovery authority.
It does not forge or reuse an ALLOW from the predecessor Spark Pair Arbiter.
Its only effect is removing the predecessor policy's Slurm worker selector
from the two UID-pinned Spark nodes. No GPU grant, lease transition, barrier
change, memory-floor change, service restart or resume belongs to this action.
After detachment, the unchanged frozen arbiter must prove recovery itself.

The predecessor recovery drains, fences, then detaches workers. In the
2026-09-06 incident the hosts were already FENCED, but kubelet recreated
worker processes before the empty-cgroup proof. Two unchanged recovery
attempts failed. This extension stops that recreation while preserving
the GPU barrier. It has a separate source/executable/predecessor freeze lock.

The input frame has 19 unsigned decimal fields:
schema 1; exact node UIDs; exact NodeSet UID; predecessor verification;
RECOVERY_REQUIRED; owned lease; lease with >60 seconds remaining;
matching frame/host/lease epoch and host owner; matching host lease UID;
both FENCED with invalid grants; exact source/freeze bindings;
fresh watchdogs; bound device barriers; protected resources unchanged;
legacy inventory/services/restarts/Docker claims quiesced; predecessor
Slurm observation mask; exact GPU consumer sets; two fresh host memory
observations in MiB. All booleans must be one. Slurm mask must include
controller-ready, both-drained, zero-jobs and zero-allocations (mask 30).
Both memory observations must be >=32768 MiB. Unknown/malformed inputs refuse.

The adapter captures raw facts and invokes the hash-pinned native decision.
Only an ALLOW can reach --apply. Each node patch tests UID, resourceVersion
and the original selector value; the lease resourceVersion is checked again
before each patch. Pair-wide atomicity is not claimed: a failure after the
first node leaves partial detachment, with both GPUs still fenced, and
retains the exact effect record. Existing allocation and GPU admission remain
under the predecessor authority. No test or review implies live recovery.

Control gate: test_recovery_detach.py NATIVE_EXECUTABLE.
Use remote tmux, --arbiter-root, --engine, --holder and a new --evidence
directory; omit --apply for a decision-only run. An expired lease refuses.

## Versioned recovery observer migration

The stale-memory recovery fix changes only material_backend_sha256 in the
Spark Pair Arbiter freeze. Both fresh host memory predicates remain required;
Slurm FreeMem no longer vetoes recovery while detached workers cannot refresh
it. The native arbiter, its compiler/executable, host-fence source, device
barrier, 32768 MiB floor and protected baseline remain pinned.

recovery_migrate.sio admits REBIND_RECOVERY_OBSERVER, independently of the
unchanged arbiter. recovery_migrate.py is its observation/CAS transport.
recovery-migration-lock.json pins both freezes and the new admission engine.
The 19-field frame is defined by the ordered checks in recovery_migrate.py:
schema; unchanged native authority; lease revision binding; exact pair/NodeSet
identity; owned expired RECOVERY_REQUIRED lease; selectors absent; workers
absent; both FENCED; host revision/boot binding; fresh watchdogs; device barriers;
protected resources; quiesced legacy/consumer/cgroup facts; no pair workloads;
epoch/owner/lease identity; Slurm mask with bits 2,4,8,16; bound journal; both
host memory observations. Missing/unknown or sub-floor facts refuse.

Run inside remote tmux with --old-root, --new-root, --engine, --holder and a new
--evidence directory. Without --apply this only captures native admission.
Apply repeats the complete observation and native decision, persists intent,
CAS-updates the journal then only the lease freeze annotation, and verifies
unchanged scheduling authority plus fenced hosts. Partial journal migration
can be replayed only against the same lease UID/resource version and exact
revision pair. Recovery is then run through the new root's canonical arbiter.
This is a revision-specific recovery operation, not a general lease editor.

Controls: test_recovery_migrate.py NATIVE_EXECUTABLE;
bash runtime/test_memory_observation.sh from the continuity directory;
scripts/ci/spark_pair_arbiter_selftest.sh from the repository root.
Raw independent reviews and live migration evidence are committed separately
from subsequent canonical recovery and hardware acceptance.

## Host grant transaction serialization

A second recovery revision serializes host grant mutations, watchdog cycles
and compound reports through the same host-local flock. Waiting is capped at
10 seconds; the operation has a separate 45-second timeout plus a 2-second kill grace.
flock --close keeps the lock descriptor out of child processes, and the flock
parent stays outside timeout's process group. Emergency acquisition can wait
60 seconds for a bounded external owner; systemd uses TimeoutStopSec=120 and
KillMode=control-group. Only the already-managed legacy GPU containers use a bounded 5-second stop
grace; protected services remain outside that stop set. Contention, timeout or any failed cycle
does not refresh the watchdog heartbeat. Installation remains the canonical
arbiter's responsibility; the lock does not admit a GPU grant.

This prevents a watchdog that sampled FENCED from imposing that stale state
after a concurrent Slurm commit. A temporary-root test with real competing
processes reproduces the old interleaving and verifies that serialization
preserves the later commit, rejects lock contention, kills a wedged operation,
releases the lock, and refuses a failed daemon cycle's heartbeat.

recovery_serialize.py reuses the frozen 19-field Sounio observer-rebind
authority under recovery-serialization-lock.json. It admits only the exact
new host-fence manifest, its content-addressed ConfigMap name in material
policy, and the three occurrences of that name in admission rules. Every
other policy field and every other byte of admission rules must match the
old revision. Native decision predicates, memory floor and device barrier are unchanged.
The canonical backend stages the old bridge under its old admission rules,
installs the new exact rules, proves generic GPU admission still refuses,
and only then creates the revised host Pods. Journal/lease updates are sequential CAS operations,
not pair-wide atomicity. Installation and Slurm restoration are separate.

Gate: python3 tools/pireus/continuity/runtime/test_host_transition_lock.py
EXTRACTED_HOST_FENCE_SCRIPT. The fixture uses a temporary host root and never
executes GPU operations. Use recovery_serialize.py with the same arguments
as recovery_migrate.py and the preserved observer revision as --old-root.
