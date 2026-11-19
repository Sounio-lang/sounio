# Passive source-CI resource diagnostic

The c267 source qualification ended with runner shutdown / rc 143. Its archived annotations do not establish timeout or OOM. The previous successful and interrupted compiler sources share the compared compiler/stdlib/gate Git objects, but compiler binary identity and runner resource conditions remain unproven.

The existing command observer now emits GATE_RESOURCE records before execution, at its existing sample cadence, and after ordinary command completion. Sources are Linux meminfo (MemTotal, MemAvailable, SwapFree), vmstat (host oom_kill counter), loadavg, memory PSI, and memory.events at the visible cgroup mount root. Missing readable counters are explicitly unavailable. The cgroup mount-root scope is not asserted to equal the command's cgroup. Counters are cumulative and require baseline/delta interpretation; even a delta does not attribute an OOM to Madaros. A final sample is not guaranteed after runner termination.

Only the observer's output changes. The command argv, command log bytes, return status, timeout and retry policy remain unchanged. Reading counters has nonzero overhead; the resulting source qualification is distinct from previous runs and is not a performance benchmark. The Spark runtime and its frozen profiles are unaffected.

Five controls execute the real shell observer: exact stdout/stderr log preservation, exit 7 propagation, successful command despite a telemetry failure, periodic sampling without command replay, and invalid-interval refusal before command execution. These controls are wired into Pireus Continuity Custody, including observer-file path triggers.

Next qualification must inspect these resource records together with runner annotations and elapsed stages. This change improves diagnostics; it does not resolve the shutdown or qualify the old source-342 frozen Inkling attempt. Model execution remains stopped until a separately source-bound packet satisfies all required gates.
