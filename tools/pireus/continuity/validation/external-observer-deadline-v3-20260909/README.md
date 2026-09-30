# External observer deadline v3: implementation and local controls

Status: LOCAL_CONTROLS_PASS. No Spark or loaded-model qualification.

Job 11977's rank-0 cadence failure combined a 410.2 ms sample (408.3 ms
smaps_rollup read) with approximately 200 ms post-sample wait. This version
accounts for collection and journal-emission time in the configured interval.

The next start deadline is anchored to the actual preceding sample start.
If work finishes before that deadline, sleep only for the remaining interval.
If work overruns, take one next sample immediately and anchor its subsequent
deadline to its actual start. Missed slots are never replayed as catch-up bursts.
Actual sample timestamps and gaps remain recorded; a read longer than 500 ms
can still violate the acceptance ceiling. The run limit prevents starting new
samples after expiry but cannot interrupt a blocking kernel read.

Every sample also records observer-process CPU nanoseconds and /proc/self/status
RSS/high-water RSS in bytes under observer_resources. These values are separate
from the target's metrics. Missing or malformed observer RSS remains null/error.
CPU is cumulative process CPU, not wall time; deltas may be computed between
samples. RSS is the kernel status view, not PSS, GPU memory or a total-system
allocation measure. The added self-accounting itself has overhead.

Identity is explicit:
- observer profile: external-observer-deadline-v3
- journal schema: pireus-external-memory-observation-v2
- integration identity: feedback-lifecycle-external-observer-deadline-v3
- binding retains its identity schema and binds the new helper hash
- guardian and inference entry hashes remain unchanged

## Validation

65 existing and new local tests pass: 49 external-observer/freezer/custody/
launcher tests, 8 handoff tests and 8 CPU-oracle tests. New controls exercise:
410 ms reads, 700 ms overruns, scheduler oversleep, reads crossing the run limit,
fast-read cadence, separate observer/target RSS and missing observer metrics.

With a simulated 410 ms first read, starts are 0, 410, 610 and 810 ms. The old
post-read sleep would instead delay the second start to 610 ms. A 700 ms read
remains visible as a 700 ms gap; no test changes the 500 ms acceptance ceiling.

A small separate Linux workspace reader control allocated 8 MiB and recorded
an 8,396,800-byte observer RSS rise and increasing process CPU time. This checks
the resource reader only; it does not exercise the Slurm/container handoff or
qualify Spark overhead. Its source and raw result are included.

The integration builder generated the seven-file launch subset with the new
observer hash while preserving guardian/container/entry identities. This subset
is not a full runtime freeze. Its manifest explicitly sets source CI, container
control and loaded-model qualification to false.

The pinned 11976/11977 negative-evidence replay remains byte-identical. Original
runtime archives, frozen v2 criteria, guard 33 GiB and floor 32 GiB were not changed.
No existing frozen attempt is restarted by these changes.

## Reproduction

From the repository root:

    python3 -m unittest discover -s tools/pireus/continuity/ops -p 'test*external*.py'
    python3 -m unittest discover -s tools/pireus/continuity/ops -p 'test_observer_handoff.py'
    python3 -m unittest discover -s tools/pireus/continuity/ops -p 'test_cpu_memory_control_v2.py'
    python3 tools/pireus/continuity/validation/external-observer-deadline-v3-20260909/resource_reader_control.py

Timing controls use a deterministic fake clock. The reader control uses real
Linux procfs; its future PID, timings and exact RSS values will differ.

Next: freeze a separate prospective CPU/container qualification that requires
the v3 profile, observer RSS/CPU availability and monotonic counters, complete
target-identity/attachment custody, and measured cadence. Obtain source CI and
exclusive-pair preflight before that control. Only after its qualification may
a new loaded-model protocol be considered; 11977 remains a failed closed screen.
