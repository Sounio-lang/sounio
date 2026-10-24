# External process-bound memory observer

Implements the three next steps from MEMORY_COUNTER_INVENTORY.md: process
identity, actual cgroup resolution, and a CPU-only allocation/release control.
Live-control results and source custody are archived below.

The observer runs as a separate OS process. It imports no model, torch, CUDA
or NVML; it never signals the observed target or changes cgroup limits. The
existing guardian and frozen Inkling runtime are unchanged.

## Identity and scope

A controller supplies expected Slurm job, PIREUS rank, worker UID, boot ID,
PID and process starttime ticks. Binding verifies:
- Boot ID and worker UID in the observer's kubelet mount provenance.
- Target PID/starttime and live state from /proc/PID/stat.
- Only the target SLURM_JOB_ID and PIREUS_RANK environment values.
  Other environment entries are neither recorded nor emitted.
- The target /proc/PID/cgroup membership against the observer's visible
  cgroup2 mount root/mountpoint. Exactly one mapping must be available.
- Membership in the declared job_<id> cgroup, directory device/inode and
  target pid/mnt/cgroup namespace identities.

The controller must also verify worker pod UID and scheduler allocation;
environment labels alone are not scheduler authority. PIREUS rank is the
declared runtime rank (host mapping), not an assumed Slurm task ordering.
The CPU control uses real Slurm job identity, not fabricated model receipts.

Binding reads identity twice. Each sample revalidates before and after its
metric reads; a changed/missing identity invalidates the target and discards
the in-flight sample. This detects observed PID reuse, cgroup migration,
directory replacement or namespace changes. It is not an atomic kernel
snapshot and does not prove absence of an undetected transient migration
that returns between checks. No fallback substitutes a worker/root cgroup.

The current resolver refuses ambiguous mounts, invisible membership, traversal
and deleted membership. It does not claim support for arbitrary namespace
layouts. Target membership is read from the observer's /proc view and mapped
through that observer's cgroup2 mount.

## Observations

Host meminfo, vmstat and PSI; target process status and smaps_rollup; target
cgroup memory.current/peak/stat/events/events.local/pressure. Individual reads
carry timestamps, durations, format and explicit error/null values. kB fields
are converted to bytes; raw mixed-unit counters retain their kernel text.
A missing metric does not become zero. Identity-file failure stops attribution;
ordinary metric failure remains explicit missing data.

Each record carries job/rank, observer PID, target PID and binding digest.
Journal creation is exclusive. Sampling is bounded to0.1–5s intervals and
a3600s loop deadline; actual gaps include read cost and are recorded. The
deadline is checked between reads and is not a hard realtime bound on a blocked
kernel read; an outer Slurm time limit bounds the CPU qualification. Completion by
duration is distinct from TARGET_INVALIDATED. An invalidated target yields
exit3 and metrics:null, with no subsequent sample attributed to that target.

## CPU-only qualification contract

external_observer_cpu_control.py starts a separate target under Slurm. It
holds a baseline, touches64MiB of anonymous mmap pages, unmaps them, then
exits. An independent observer samples at0.2s. The test requires at least
three settled samples per phase, observed PSS rise/fall of at least48MiB and
target-cgroup memory.current rise/fall of at least32MiB. It then requires
TARGET_INVALIDATED/exit3 after the target exits. These are intentionally
broad lower bounds, not exact accounting identities.

The control allocates no GPU memory and loads no Inkling checkpoint. It
qualifies CPU-control identity, visibility and lifecycle only. It does not
qualify CUDA accounting, inference overhead, a new GPU profile or pilot
acceptance. Process and cgroup views are not additive.

Local tests cover valid binding and units; wrong job/rank/worker/boot/start;
PID reuse/disappearance; cgroup migration; refusal of worker substitution;
missing/malformed metrics; identity change during sampling; invalidation
journaling and exclusive creation; mount-root mapping and ambiguity.

Reference semantics:
- https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html — starttime field22.
- https://man7.org/linux/man-pages/man5/proc_pid_mountinfo.5.html — mount root/mountpoint.
- https://docs.kernel.org/admin-guide/cgroup-v2.html — membership and memory interfaces.

## Verified CPU pair — job11973

Source638c35fb992c1a65cb51ded5d2d12dc3824e6d1c; immutable source copies and hashes
are in validation/external-observer-cpu-20260909. Fresh preflight passed.
Job11973 completed0:0 on both Sparks from2026-09-09T13:56:10 to13:56:17
(raw scheduler timestamps).35 samples were recorded per rank.

| Measurement | rank0 /3c59 | rank1 /8e54 |
|---|---:|---:|
| PSS increase/decrease |67,108,864 /67,108,864 bytes|67,108,864 /67,108,864 bytes|
| cgroup current increase/decrease |67,866,624 /67,338,240 bytes|67,391,488 /67,108,864 bytes|
| max measured sample gap |205.592ms|205.546ms|
| max sample duration |4.969ms|4.903ms|
| target exit detection after exiting marker |148.001ms|147.874ms|

Both observers exited3 with TARGET_INVALIDATED and metrics:null after their
targets exited. Both had distinct observer and target PIDs. Runtime rank0
mapped to Slurm task_1 and rank1 to task_0; the binding correctly preserves
actual rank identity rather than assuming task-index equality.

Actual membership was the worker subtree /system.slice/slurmstepd.scope/
job_11973/step_0/user/task_N. The observed group was the task's real cgroup,
not the worker parent. Source copies, expected identity, cgroup device/inode,
namespace IDs, journals, phase events, worker UID before/after and scheduler
accounting are archived. No target reattachment or fallback occurred.

Twelve local tests pass, including directory replacement, namespace change and
permission denial. Three offline packet controls pass: exact reproduction,
modified journal refusal and modified qualification refusal.
Reproduce with:
python3 tools/pireus/continuity/ops/verify_external_observer_cpu.py

The raw Slurm stdout interleaves the two large JSON summaries. It is archived
without edits; acceptance uses separate worker control files and journals.
Slurm also reported an unavailable inherited /workspace/.tmp and fell back to
/tmp. The control completed; no rerun was performed for that warning.
Future launchers should set TMPDIR=/tmp explicitly.

Exclusive node allocation reserved both GPUs (AllocTRES gres/gpu=2) even though
the requested workload and step used CPU only. This is resource reservation,
not CUDA execution: no model, CUDA library or GPU kernels were invoked.

The1-2-3 implementation and CPU qualification are complete. They do not
establish memory accounting for a loaded Inkling process, inference overhead,
root cause of11972, or acceptance of a new GPU execution profile. The next
separate phase is integration/qualification of this external observer around
the actual inference process, retaining guardian33GiB/floor32GiB and immutable
prior attempts. No Inkling rerun is part of this CPU-control packet.
