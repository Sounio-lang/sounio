# Spark memory-counter inventory — 2026-09-09

State: read-only worker inventory complete; no model execution or new inference
profile qualified. Source receipts: validation/memory-counter-inventory-20260909.
Both workers retained their recorded pod UIDs and boot identities during the
inventory. The GPU partition was idle at the initial scheduler observation.

## Findings and scope

Both GB10 devices return N/A for memory.total, memory.used and memory.free
through the queried nvidia-smi interface. The command exits0, so successful
command execution must not be interpreted as a numeric memory measurement.
The NVIDIA documentation defines N/A as unsupported information:
https://docs.nvidia.com/deploy/nvidia-smi/index.html

Both workers expose /proc/meminfo, /proc/vmstat, /proc/pressure/memory and
/proc/self/smaps_rollup. Raw fields and units are archived. PSI measures
resource-stall time, not allocated bytes; its total counter is in microseconds:
https://docs.kernel.org/accounting/psi.html

The initial root-cgroup inventory lacked memory.current, memory.peak,
memory.events and memory.max. Resolving /proc/self/cgroup against the observed
cgroup2 mount instead finds all these interfaces in BOTH worker-container
cgroups, along with memory.stat, memory.events.local and memory.pressure.
This matches the kernel specification: memory.current exists on non-root
cgroups and accounts for the cgroup and descendants:
https://docs.kernel.org/admin-guide/cgroup-v2.html

The initial observations remain intact, with their original probe source.
Resolved observations have separate files, probe hashes and UID custody.
Do not retroactively change the unavailable fields in11971/11972 journals.

The resolved cgroup is the current probe/worker container, NOT a model rank.
Its observed memory.current is48,324,608bytes on3c59 and46,673,920bytes on8e54.
Its configured memory.max is429,496,729,600bytes (400GiB), which is neither
physical capacity nor the32GiB protected host floor. No limit was changed.
All recorded worker memory.events counters are zero at this idle observation;
this does not reconstruct events in a previous job's cgroup.

## Idle read cost

Each file was opened/read ten times; median and maximum elapsed monotonic
time are recorded. These are small, sequential, warmed reads with no model,
not p95/p99 estimates or a loaded-inference observer-overhead qualification.

| Source |3c59 median|8e54 median|Scope |
|---|---:|---:|---|
|/proc/meminfo, initial probe|9.160us|21.544us|host view exposed in worker namespace|
|/proc/vmstat, initial probe|19.216us|37.664us|raw kernel counters, per-field units|
|/proc/pressure/memory, initial probe|8.496us|19.840us|host pressure stall information|
|/proc/self/smaps_rollup, initial probe|34.104us|90.337us|small probe process only|
|resolved memory.current|9.432us|27.152us|current worker/probe cgroup|
|resolved memory.stat|12.112us|34.801us|current worker/probe cgroup|

The initial nvidia-smi commands took approximately23–27ms each and supplied
no numeric GPU-memory totals. They should not be placed in a50ms guardian
decision loop; this inventory does not establish a safe repeated rate.

## Next implementation boundary

Build a separate external observer with explicit identity and a bounded
lifetime. It should bind to a declared job/rank/worker UID/boot ID, resolve
the actual process PID plus /proc/PID/stat starttime, then record and verify
that process's cgroup membership and cgroup2 mount mapping. Do not silently
substitute the worker, root or another job's cgroup when resolution fails.

Candidate observations: host meminfo and vmstat; PSI totals; target process
status and smaps_rollup; actual target cgroup memory.current/stat/events/
pressure. Record independent monotonic timestamps, read durations, actual
sample gaps and explicit missing/error values. A disappeared/reused PID or
changed membership must terminate or invalidate target attribution.

Validate on a bounded CPU-only allocation/release control before any Inkling
run. That control should prove identity, scope, missing-data handling and
observable changes; it cannot establish CUDA-accounting completeness or
loaded-model overhead. Keep the observer separate from the guardian and
separate from the model's Python/GIL. Resolve Slurm rank identity in an
explicitly declared qualification before reentering a frozen GPU experiment.

No new GPU job, retry, cache change, allocator flush, cgroup-limit change,
guardian change, service restart or driver modification occurred.
The11971/11972 attempt remains closed, paired inference incomplete, root
cause unresolved, and original pilot acceptance unchanged.
