# Prospective CPU/container qualification for external observer v3

State: FROZEN_NOT_EXECUTED. No job has been allocated.

Source commit: 6e470eab01d8ba3645765e82fc604d75975cdb7f
Protocol SHA256: 659d32dca714a98deed3853a1020d652e1374e96650a241d18e308ab6d50e81a
Manifest SHA256: 33ae4e7ee290e0a7b11daa4649575053b2d1c56e9cb18235d7fd96fa74ed419d

The packet contains 11 pinned source/build files, including the v3 observer,
unchanged CPU target, guardian and container launcher, handoff supervisor,
v2 memory oracle and v3 extension. It is a CPU-control packet, not an Inkling
runtime acceptance or a replacement for closed attempts 11975/11977.

## Predeclared acceptance

The unchanged v2 numerical controls require a 64 MiB allocation/release to be
visible in PSS and cgroup anon (48..80 MiB differences), with three settled
samples per phase, unchanged task limits and no OOM counter increments.
Cache/reclaim remain observations; total cgroup-current growth is not a proxy
for anonymous allocation.

The v3 extension additionally requires:
- exact v3 profile, journal schema and actual-start deadline scheduling;
- actual gaps no larger than 500 ms, including start and target-exit boundaries;
- reported gaps consistent with monotonic timestamps and nonoverlapping samples;
- observer resource identity separate from target PID;
- available RSS/high-water values in bytes, with nondecreasing high-water RSS;
- cumulative observer CPU counters nondecreasing and advancing during the run;
- observer status-read times contained within their parent sample;
- complete target-exit coverage.

The oracle checks readability/accounting and cadence. It records observer RSS
and CPU consumption without claiming that their overhead meets a loaded-model
budget. No absolute observer-overhead threshold is inferred after seeing data.
Success still requires separate source, scheduler, worker UID/boot, runtime,
handoff, output and guardian custody. Running with python -O is rejected because
the inherited frozen v2 oracle uses assertions.

## Execution boundary

One exclusive two-node CPU/container attempt; 512M requested memory and two
CPUs per node, five-minute job limit, same 33 GiB host guardian and 32 GiB floor.
The target allocates 64 MiB during baseline/allocated/released phases lasting
2/3/2 seconds. No model loading or inference, no automatic retry.

Before any allocation:
1. Required checks must pass on the exact frozen source: CI Decision,
   transport-and-archive, Archived script custody (no runtime replay).
2. Verify this packet and the locally tested cpu_v3_attempt.py launcher/collector.
   Hardware custody remains unverified until the terminal collection qualifies.
3. Fresh pair preflight must establish empty queue, exclusive ownership, exact
   worker pod UIDs/boot IDs, SIF hash and source bytes.
4. Bind those identities and the sole launch command to an exclusive attempt
   receipt and run the owned controller in remote tmux.
5. After terminal accounting, collect all available receipts once, including
   failed/incomplete evidence, and apply custody before promoting any result.

Historical v2 CPU rows augmented with v3 metadata/resource values are synthetic
unit fixtures only. They do not make 11975 a v3-qualified attempt.

## Local validation

12 v3 oracle tests pass. They cover accepted synthetic data and refusal of old
profiles, wrong scheduling, target/observer identity confusion, missing/wrong
units, CPU regression, decreasing high-water RSS, misplaced timestamps, forged
gaps, delayed exit, missing OOM counters and optimized Python.

The freeze verifier checks every source byte and protocol against pinned hashes,
then verifies protocol/oracle agreement. Changing the observer source is rejected.

    python3 tools/pireus/continuity/validation/external-container-cpu-v3-20260909/verify_freeze.py
    python3 -m unittest discover -s tools/pireus/continuity/ops -p 'test_cpu_memory_control_v3.py'

Source CI is a separate live gate. A run being pending does not authorize
allocation; no source-CI success or hardware qualification is recorded here.

## Launch and terminal custody implementation

cpu_v3_attempt.py provides launch, collect and qualify modes. The launcher requires
remote tmux, exact-source CI, fresh pair preflight and an empty queue before staging.
It uses an exclusive protocol-addressed directory on each worker, stages read-only
source bytes, verifies worker UID/boot and all bytes again before entry inside the
allocation, verifies the SIF hash, and runs only the CPU target through the unchanged
container launcher and guardian. Existing attempt directories refuse replay.

The start receipt binds the actual command, source protocol, workers and hashes of
the launcher, source-readiness helper, preflight and freeze verifier. Qualification
rechecks that command and those orchestration hashes. The terminal collector checks
unique job accounting and worker identity before/after copying, preserves missing
files explicitly, and hashes every collected artifact. It never submits a job.

Qualification requires successful terminal accounting, complete collection, source
checks, the job-bound pre-entry barriers, runtime bytes, target/observer handoff,
expected and observed binding identities, journal bindings, supervisor completion
and both guardian exits before running the frozen v3 oracle.

14 synthetic launch/custody tests pass, including a full CPU-only custody fixture,
failed partial collection, refusal outside tmux, duplicate attempt, pending CI,
wrong worker, altered guard command, observed-binding mismatch and failed CI. These
do not constitute a live Slurm/container test. The 12 v3 oracle tests also pass.

Launch once from an owned remote tmux session:

    python3 tools/pireus/continuity/ops/cpu_v3_attempt.py launch --frozen /workspace/.cache/pireus-continuity/external-container-cpu-v3-freeze-20260909 --stage /workspace/.cache/pireus-continuity/external-container-cpu-v3-driver-20260909/attempt

After the job and launcher are terminal, use collect with the exact job ID, then
qualify with the SHA256 of collection.json. A failed collection is retained and
cannot qualify. The unchanged frozen manifest is the sole CPU runtime source;
new orchestration code is separately hashed in start.json.
