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

## CI wiring

The continuity workflow now runs all 91 observer/handoff/CPU-oracle/attempt
controls, verifies this frozen packet and executes the small Linux resource-reader
control. The reader receipt is included with the existing custody artifact.
All those commands passed locally. No YAML linter was available locally; actual
execution of the new workflow steps remains a separate CI gate after publication.

This workflow edit does not change the frozen runtime or the waiting driver's
orchestration hashes. It is queued behind the exact-source run 34415603204 rather
than replacing that validation with a new head.

### Source CI interruption and bounded recovery

CI run 34415603204 attempt 1 ended in failure on 2026-09-10 at 00:44 UTC.
The Madaros fixed-point step logged a runner shutdown signal and exit 143,
after Merged IR: 13269 functions. This is not a successful self-compile;
the shutdown's underlying cause is not established. The original driver
stopped at STOP_SOURCE_CI without allocating a hardware attempt.

source-ci-attempt-1/manifest.json pins the raw failed-job log, run/job
metadata and original terminal driver log. Only failed CI jobs were explicitly
rerun on the same source revision. ci-recovery.py watches run attempt 2
in remote tmux and invokes the unchanged original driver only after that
attempt succeeds. It refuses a changed run identity or an existing hardware
attempt. The original driver still performs its full source, hash and pair
preflight checks. A CI rerun is not a hardware retry; the experimental limit
remains one attempt, zero retries. No CPU/container or loaded-model
qualification follows from this recovery setup.

Recovery controls: six isolated subprocess controls passed, including existing-attempt, changed-source, changed-run-attempt, failed-CI and changed-driver refusals. Archived failure hashes also passed. These controls do not execute hardware. Replay with python3 tools/pireus/continuity/validation/external-container-cpu-v3-20260909/ci-recovery-controls.py.

### CI recovery terminal refusal

Attempt 2 ended in failure at the imported-capacity boundary gate. The wrapper
timed out the compiler after 300 seconds (compiler status 124, gate status 1)
on the 16383-function witness. Attempt 1 passed this gate on the same checkout
merge ba65bbce7b2d19d051a507e6ec96eac29b0995f0. Semantic failure is not established.
The recovery watcher stopped without invoking the original driver and no
hardware attempt exists. source-ci-attempt-2/manifest.json pins this refusal.
Next: measure the archived compiler and boundary workload independently before
changing a timeout or interpreting the failure. No third CI rerun is scheduled.

### Separate capacity diagnostic: accounting correction

The separate CPU timing diagnostic allocated jobs 11978, 11979 and 11980.
All three were cancelled almost immediately with only an extern step recorded.
No program-start receipt, compiler log, output ELF or result JSON exists on
the staged worker. The configured client eventually reported that the node
was still not ready. The step-start cause is not established.

Earlier accounting queries ran as root without all-users selection and
incorrectly returned no jobs for the sounio user. The previous no-diagnostic-
allocation statement is superseded by capacity-diagnostic-transport/decision.json
and its all-users accounting receipt. Original routing records are preserved
even where that mistaken inference appears; they are not acceptance evidence.

This was separate raw-compiler troubleshooting. No Spark v3 CPU/container
attempt occurred and no CI timeout, model-memory guard or frozen workload
was changed. The compiler ZIP remains pinned in the persistent private cache.
Diagnose step-manager initialization before submitting another capacity workload.

Follow-up observation: the worker log path is a FIFO with three old tail/grep
readers (two at least six days old, one at least two days old). The absence of
step-start errors from pod stdout cannot establish their absence from the daemon.
No existing reader or daemon was modified. The underlying step-start cause is
still open. On published head 7617d6ff62, CI run 34424138904 job 102705970176
completed the initial lowering gates successfully, including the imported-
capacity gate, and reached the self-compile step. This does not supply a full
CI verdict or CPU/container qualification. No timeout change is justified by
these observations alone. See capacity-diagnostic-followup.json.
