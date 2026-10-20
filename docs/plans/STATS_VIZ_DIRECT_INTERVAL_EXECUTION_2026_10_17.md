<!-- docs:meta
topic_id: repo.docs.plans.stats-viz-direct-interval-execution-2026-10-17
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stats-viz-direct-interval-execution-2026-10-17
-->

# Direct interval construction — 2026-10-17

## Implemented and accepted

The explicit forest-interval constructor in stdlib/viz/ir.sio now fills its
validated destination slot directly. It removes the rows and no_variance
temporary arrays and the call through viz_scene_add_series. Slot initialization,
all five persistent arrays, interval semantics, capacities, and the public
signature remain. Rejection checks still precede any scene mutation.

Foundry run: stats-viz-memory-20261017-direct.
Slurm job: 14757, COMPLETED, exit 0:0, elapsed 14 seconds.
JUnit: 4 tests, 0 failures, 0 skipped, 0 errors.
Tests: raw MAD, Student/F/Spearman tails, chart builders, interval forest.

This is also the first runtime acceptance of the smaller package built by
scripts/dev/package_stats_viz_runtime.sh. Snapshot size: 8.5M (ls -lh).
Snapshot SHA256:
1b98bb8953a51b0384fab438e54d8cc67b8d081fd7f306737a6bfd1755c23501

ir.sio SHA256:
e373ac46d90e07293c37563583451c3a9ecada5752c0da254f4a1332b1df5990

## Resource comparison

A separate Slurm job 14759 compares the prior accepted snapshot
stats-viz-tails-20261016-final with stats-viz-memory-20261017-direct.
Both use the same committed Madaros ELF:
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f

The profile checks ir.sio, verifies identical fixture source, performs three
interleaved compilations and executions per version, and compares complete
stdout byte for byte. GNU time records peak RSS, elapsed seconds and exit code
separately for compilation and execution. Extraction/materialization are outside
the measured intervals. The profile removes only its own freshly created
temporary directory on exit.

Profile root:
/var/lib/cephfs-slurm/training/sounio-foundry/runs/stats-viz-memory-20261017-profile

C++23 profile_summary.cpp computes sample medians and ranges from metrics.tsv.
The native workload is a short correctness fixture, not a throughput benchmark.
Three repetitions cannot establish broad statistical performance superiority.
Zero-duration or zero-RSS samples must not be interpreted as zero resource use.

Job 14759 completed with exit 0:0, elapsed 2 minutes, RESULT=VIZ_MEMORY_AB_PASS.
The two input snapshots and compiler hashes are recorded in the profile root.
Both library checks succeeded. All six complete stdout files are byte-identical:
45ca4f64396199179ef8007813ea877033250ed41a2fe8919ad8e4f75a677f89

C++23 summary of the three samples per cell:
variant/phase | RSS median KiB | elapsed median seconds
before/compile | 3036844 | 19.10
after/compile | 3035776 | 13.92
before/execute | 1596 | 0.00
after/execute | 1604 | 0.00

Generated ELF sizes: before 909768 bytes, after 893384 bytes.
C++23 subtraction confirms 16384 bytes saved.
The temporary ELFs were executed and size-recorded, then removed with the
profile's scratch directory. Output ELF hashes were not captured by this run.

No clear process-RSS improvement is demonstrated: runtime median is slightly
higher, within the observed sample variation, and compiler memory remains large.
Elapsed compilation medians are observations on a shared worker, not a guaranteed
speedup. The short runtime fixture is below timer resolution in most samples.
Do not claim zero execution cost or a statistically established speedup.

The major measured memory cost is compilation, not this renderer execution.
For repeated use, compile once and reuse the resulting ELF; repeated souc run
includes compilation again. A compilation cache is not implemented by this patch.

Scheduling record: the original ten-minute request was shortened to two minutes
to fit backfill. A subsequent user-level increase was denied; no elevated
override was used. The original job completed successfully. A contingency job
14763 was cancelled by uid1000 before allocation (elapsed 0); it supplied no
samples. The previous 8192 MiB reservation is not measured application RSS.

## Source / delivery boundary

Isolated worktree: /workspace/sounio-sprint-stats-viz, based on f5b2c585.
Remote main was rechecked at 6861acf. No rebase, commit, push or merge.
The successful Slurm execution is focused runtime evidence, not full corpus
or current-source compiler validation.
