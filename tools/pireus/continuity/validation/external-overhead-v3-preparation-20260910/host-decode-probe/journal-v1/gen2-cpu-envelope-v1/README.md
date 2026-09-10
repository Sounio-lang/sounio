# Frozen gen2 CPU resource envelope v1

This is a separately identified DL380 measurement, not CI replacement, a same-host memory intervention, or Inkling qualification. The compiler is pinned to the byte-identical executable from the successful and timed-out CI artifacts. The source archive contains the entire self-hosted and stdlib trees plus the canonical invocation helper from source 0655747507.

The declared command compiles gen2 only, preserving Madaros build argv through souc_compile and the helper's 1.5-GiB stack request. The worker must verify the packet hashes before extraction/execution, bind hostname and boot identity, set the stdlib path to the frozen tree, and record granted limits. Reserve 4 CPUs / 36 GiB through Slurm, with a 5400-second command deadline and 92-minute allocation. No automatic retry.

Before launch, record fresh node/queue state and worker pod identity. Capture monotonic wall time, child process CPU usage, peak RSS, minor/major faults, host memory/swap/PSI and process identity. Preserve partial artifacts on timeout as unqualified evidence; never execute a partial output. A successful compile may subsequently undergo a bounded banner check, recorded separately. The runner must refuse an existing attempt marker.

Job 11991 entered the verified worker at 2026-09-10T11:33:42 UTC. The packet passed hash verification on both ends of transport; fresh queue and host preflight passed. Execution runs under tmux pireus-gen2-cpu-v1. Twenty samples and compiler diagnostics confirmed execution in progress; no terminal result or completed compilation is asserted. See execution-state.json for the bounded observation.
