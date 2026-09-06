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
