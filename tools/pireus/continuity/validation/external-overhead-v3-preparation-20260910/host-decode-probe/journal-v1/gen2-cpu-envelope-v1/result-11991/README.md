# Job 11991: completed independent gen2 CPU measurement

Slurm accounting confirms COMPLETED / 0:0 for job, extern, and step 0, ending
2026-09-10T12:59:14. The compiler returned zero without timeout and produced a
34,132,424-byte gen2 binary. Independent packet audit checked 2,481 source files
against the pinned archive, input/runner/artifact hashes, job, boot and allocation.
Worker UID and boot were unchanged across collection.

The compilation took 5,131.146 seconds (85 min 31 sec). Child resource accounting
recorded 5,080.044 user CPU seconds, 39.806 system CPU seconds, maximum child RSS
27,294,756 KiB, and zero major faults. Slurm step MaxRSS is separately recorded in
accounting.txt; these counters have different collection scopes.

The binary SHA-256 is 8f7aacf30db89edc07c342c56a90b41ed30e981a0872aaf206ac999e24167370.
Its bytes and size exactly match fixed-point/madaros.gen2 from historical CI run
34439721975, artifact 10138844087, whose ZIP hash is in summary.json. This is gen2
identity, not a gen3 fixed-point proof.

The first direct banner launch did not execute: the emitted file has mode 0644.
Its PermissionError is preserved in banner-transport.json. A separate byte-identical
copy outside the frozen packet was given mode 0755; --version returned zero within
the 15-second limit and identified Madaros v0.80.0. Original bytes/mode were preserved.

Raw samples (1,026 rows), original binaries and full worker observations remain in
the private collection named in summary.json, with hashes in collection-manifest.json.
Public receipts preserve exact compiler logs, runner result, audit and accounting.

This qualifies this frozen gen2 CPU measurement on DL380 (4 CPUs / 36 GiB / 90 min).
It does not replace source CI, qualify Inkling, resume the stopped 32-request pilot,
or demonstrate memory caused historical CI failures. CPU hardware, timing envelope,
and runtime conditions differ from those CI attempts. No model job or retry ran.
