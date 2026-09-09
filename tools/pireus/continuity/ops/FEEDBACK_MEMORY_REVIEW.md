# Feedback memory-envelope review — 2026-09-09

State: archived evidence reviewed; root cause unresolved; no new runtime profile qualified.
Source: immutable jobs11969/11970, attempt closure SHA256
928d6ac9f19c7a2252464366cefd7044c3d48bbc3ef7aa4719be5ff1f7a1cb57.
Reproduction: python3 tools/pireus/continuity/ops/review_feedback_memory.py --output NEW_PATH.
The output uses exclusive creation and verifies every artifact named by the pinned closure.
Detailed observations and source log line numbers are in
validation/feedback-memory-review-20260909/review.json.

## Observed result

| Arm / rank | Minimum available GiB | Margin above33GiB, MiB |
| --- | ---: | ---: |
| without-feedback /0 |33.020908|21.410156|
| without-feedback /1 |33.278564|285.250000|
| with-feedback /0 |33.304333|311.636719|
| with-feedback /1 |32.980450|-20.019531|

11969 completed8/8 requests;11970 stopped during request index5 after5/8 saved.
The slowest accepted margin was only21.41MiB. The failed measurement exceeded
the early-stop boundary by20.02MiB; it remained above the protected32GiB floor.
The successful arm does not establish robust headroom for another run.

At matched request indices0..5, both ranks in the feedback arm have exactly
75,497,472 bytes (72MiB) more CUDA reserved memory and12,800 bytes more CUDA
allocated memory at OFFLINE_EXTEND_END. After index0, allocated and reserved
values are each constant within every arm/rank at the sampled boundaries.
For indices1..5, feedback process RSS is about79.24–86.86MiB higher, with
Pss_File about69.81–71.63MiB higher.

These observations do not identify a72MiB leak. The allocator differences are
between separate jobs, and are stable at these request boundaries. They also
do not rule out transient decode allocations or growth in another subsystem.
RSS/PSS, allocator accounting and host MemAvailable are different views,
especially on the shared-memory Spark platform; do not add their deltas as an
accounting identity or subtract them to invent an attributed residual.

The minimum rank flips between arms: rank0 is tighter in11969 and rank1 in11970.
That asymmetry, sequential arm order, different prompt lengths344/808 and
unobserved host/cache changes prevent causal attribution to feedback semantics.
The five completed feedback responses are a censored subset, not a balanced
eight-request comparison.

## What the current trace cannot discriminate

The source logs memory before/after extend, and proposal-save events before
runner.cleanup(batch). It has no paired immediate before/after cleanup
observations, nor synchronized host/process/CUDA accounting at the guardian
minimum during decode. Guardian samples carry MemAvailable but do not assign
its variation to the model, other processes, page cache, slab or kernel/driver.
A flat extend-end allocator series therefore cannot distinguish a decode peak
from host pressure outside that allocator.

## Next bounded diagnostic proposal (not launched or qualified)

Declare an observational diagnostic identity before running anything. Keep
the same model/checkpoint/image, cache6144, context16384, output4096, TP2,
seeds, complete request bytes, arm order, guard33GiB and floor32GiB. Preserve
11969/11970 without editing their runtime or resubmitting their job identity.

Add read-only, timestamped, request-indexed observations at decode entry,
proposal-save, immediately before/after cleanup and the next extend entry.
Record monotonic time, rank, job, request/token index, host MemAvailable plus
available meminfo components (Cached, SReclaimable, Shmem, Slab, Unevictable),
process PSS/RSS, owned-child RSS, and CUDA allocated/reserved/peak counters.
Host/cgroup or device counters unavailable on a node must be marked missing,
not fabricated or treated as zero. Keep these views separate.

The guardian must retain its existing50ms decision path and33GiB threshold.
Do not put expensive smaps reads or CUDA calls into that critical loop.
Use a separate bounded observer for host accounting, and tie its observations
to runtime events by monotonic timestamps. Instrumentation overhead and
sample gaps become part of the diagnostic receipt; a diagnostic pass does
not qualify the uninstrumented profile.

Falsifiers to evaluate before proposing a memory optimization:
- Persistent allocation: increasing post-cleanup CUDA allocated/reserved or
  process-anonymous memory across fixed requests supports investigation of
  retained objects; flat values weaken that mechanism at those boundaries.
- Transient working set: peak growth during decode followed by recovery at
  cleanup supports a transient-pressure mechanism; snapshots alone do not.
- Host accounting: falling MemAvailable with stable process/allocator views
  calls for host/cache/kernel attribution; a coincident change is not causation.

No cache reduction, empty_cache/malloc_trim insertion, prompt truncation,
output reduction or larger guard tolerance follows from this review. Select
one intervention only after discriminating evidence, give it a new profile
identity, and qualify separately before returning to a paired experiment.
Any stopped diagnostic remains negative evidence with no automatic retry.

Original pilot stays1/9 cells,32/288,0 gain-qualified;11956 is unchanged.
M5/M6, V13/V14, native gain and HTTP/general16K acceptance are not promoted.

## Diagnostic implementation receipt

ops/build_lifecycle_diagnostic.py now generates a separate offline_generate.py
from the exact fa4fb62f... source, embedding ops/lifecycle_observer.py.
Artifact and identity: validation/lifecycle-diagnostic-20260909/manifest.json.
The qualified runtime and all11969/11970 input files remain unchanged.

Hooks cover extend entry, decode entry, every16 decode steps, decode exit,
proposal save, cleanup before/after, and references released. Each hook records
monotonic time, request/token index, process PSS/RSS, child RSS and CUDA
allocated/reserved/lifetime peak counters. No counters are reset.
A separate daemon observer samples host meminfo every1second, records the last
runtime context, sample gaps and observation duration. It does not run in or
change the50ms guardian. The journal is exclusively created per job/rank at
/scratch/pireus/receipts/lifecycle-JOB-RANK.jsonl.
Unavailable cgroup/device accounting is explicitly null and named unavailable.
A stopped process may leave no OBSERVER_END; missing tail samples must remain
missing. Hook durations do not measure all scheduling/GIL overhead, and GPU
counters are not synchronized here; hardware impact is still unmeasured.

Five local controls PASS: missing data is null; live observer lifecycle and
counter capture; refusal of existing journal and invalid interval; modified
base refusal; generated AST matches the original after removing observation
hooks. These establish local construction/control behavior, not GPU acceptance.

Next: freeze a complete separately identified runtime and the unchanged paired
inputs, qualify its source and observer artifacts, then obtain fresh host
preflight before any hardware diagnostic. Neither instrumentation performance
nor a new serving/inference profile is qualified by this implementation.
