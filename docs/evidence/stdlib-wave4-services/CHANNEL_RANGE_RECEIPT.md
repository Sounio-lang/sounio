# NativeChannelRangeProducer qualification and custody

Implementation commit: `f22ed9d6fdc80fe8ba52652bb0b61ca4b3442300`.
The qualified source was staged with `git archive` from that exact commit.
Uncommitted WorkerFn and normal-pipeline changes were excluded.

| Slurm job | Purpose | Result |
| --- | --- | --- |
| 18907 | Initial finite-deadline controls | COMPLETED 0:0, 27 s |
| 18927 | Preserved INT64_MAX failure on initial runtime | Collector 0:0, child 1, 10 s |
| 18984 | Absolute CLOCK_MONOTONIC correction and full focused gate | COMPLETED 0:0, 35 s |
| 19003 | Same gate on exact implementation commit | COMPLETED 0:0, 35 s |
| 19009 | Source/receipt verification, durable archive and readback | COMPLETED 0:0, 9 s |

Job 18927 retains the actual UBSan report: signed integer overflow in
GCC 13's chrono addition, reached from the original relative condition-variable
wait at an INT64_MAX deadline. The same miniprogram passes after replacing the
relative wait with `pthread_cond_clockwait(..., CLOCK_MONOTONIC, absolute)`.
A positive wait counter verifies that the maximum-deadline control reaches
the real worker's backpressure path before cancellation.

The exact committed-tree gate reports `WAVE4_CHANNEL_RANGE_GATE_OK`.
C++23 normal and ASan/UBSan controls, the preserved maximum-deadline
miniprogram, and the public Sounio check/compile/native programs all pass.
The migrated original channel test keeps its data and assertions. This receipt
does not qualify the uncommitted WorkerFn migration or generic Sounio IO callbacks.

Raw Sounio compiler:
`/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/named-ref-v2/results/madaros`,
SHA256 `21b3b011c0d621041d12ba1661f756b55315eb78d0ce071ec1cad2f5e5ab26d2`.
C++ toolchain: Ubuntu GCC 13.3, driver
`/usr/bin/x86_64-linux-gnu-g++-13`; compiler, frontend, linker,
dependencies, sources, executables and nonempty inputs are sealed in the runs.

Scratch runs are under
`/scratch/sounio-stdlib-wave3-20261027/services/`:
`channel-range01`, `channel-deadline-max-before01`,
`channel-range02`, `channel-range-committed01`, and
`channel-range-custody01`.

Durable archive:
`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/channel-range-export01/channel-range.tar.gz`.

SHA256:
`a88d9dc47ebd505879cd8c61f60c85190e9cd0e450a80751baeee02d4b00e585`.

Job 19009 rechecks all existing nonempty source, tool, executable, input,
dynamic-library and artifact manifests for the three Range runs, plus source
and executable manifests for the preserved failure. It lists the archive,
syncs it to storage and verifies the archive hash by readback. Its marker is
`NATIVE_CHANNEL_RANGE_DURABLE_OK`. Receipt metadata for all four validation
jobs is retained in `channel-range-custody01/SLURM_RECEIPTS`.
