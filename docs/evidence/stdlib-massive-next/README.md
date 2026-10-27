# Stdlib follow-through: 2026-10-27

Source baseline: main 77ac662f9be1c477c796b946a0be21fd34f783cb.
Consolidated prior sprint candidate: aa83d85bddd8c6a295ec9957389ec2fb86c7484f.
The four existing draft PRs remain separate; this wave adds channels and dynamic containers.

## Compiler provenance and privacy repair

Slurm 17912 built the host bootstrap and modular compiler from the pinned main source.
Modular raw SHA256: 0b46f28d2d3a8e358f5fd3d3fb4cb7bc43781625de2092c98148b73b33fd5b73.
The platform/container validation below uses that exact compiler through the canonical wrappers.

Separate bootstrap repair: https://github.com/Sounio-lang/sounio/pull/3245.
Slurm 17914 reproduces five accepted private-field violations before repair and E259 rejections
after repair, for x86_64-linux and aarch64-macos. Legal ownership/public-field controls compile;
x86-64 execution passes. Three compiler generations are byte-identical; stage2/stage3 fixed point passes.
ARM64 execution and full CI are separate, pending gates.

## Bounded native channels

Queue ownership is linear; copyable tokens do not own storage. Destroy invalidates tokens;
IDs are never reused. Close rejects new sends while preserving buffered data for draining.
Receive distinguishes zero payload, temporary empty, deadline expiry, closed-and-drained, and stale tokens.
The budget charges capacity * sizeof(i64), not allocator metadata or process RSS.

The C++23 primitive uses a mutex, predicate condition-variable waits and monotonic deadlines.
Algorithms and worker orchestration stay in Sounio. Worker loader contexts cannot release the last
library reference with live resources; the root context retains the existing resource-close rule.
The ABI version stays 1; the loader tolerates ENOSYS from context acquisition in older ABI-1 libraries.
The added effectful thread entrypoint declares the worker callback effect row explicitly and preserves
the existing pure-callback entrypoint.

Slurm 17916: C++ platform and channel references PASS; Sounio check, compile and native execution
PASS for channel, platform, bytes, dynamic collections and time.
The channel reference checks four producers and two consumers, every payload exactly once,
zero and negative payloads, queue saturation, deadlines, close/drain, budget and stale handles.
Source patch SHA256: 1bff7ad2881b503416a7127ebd8996bc4b59953022d68a802c5ba95b8c9c683b.

## Dynamic heap and set

Heap reuses IntegerBuffer, with checked construction and failure-preserving push.
Set reuses DynamicMap allocation, hashing and tombstones. Neither introduces a second allocator.
Slurm 17918: check, compile and native execution PASS, with >1000 elements, empty operations,
duplicates, tombstone deletion, integer extremes and budget failure preservation.
Source patch SHA256: 64e73419ff63adc7b6e59b9978cba4cc91e056019912dda33a4face995e177f6.

## Receipts and remaining gates

Durable packet roots:
- /orangefs/training/sounio-foundry/runs/stdlib-next-20261027-02/results
- /orangefs/training/sounio-foundry/runs/stdlib-next-20261027-05/results
- /orangefs/training/sounio-foundry/runs/stdlib-next-20261027-06/results
- /orangefs/training/sounio-foundry/runs/stdlib-next-20261027-08/results

These are focused receipts, not a full-corpus or final-tip acceptance.
Main/candidate full slow-enabled comparison is job 17917, packet 07, using the same raw compiler.
The baseline layers the candidate Node metadata harness onto main; main stdlib/test sources are unchanged.
No failures are suppressed and no numerical tolerance is relaxed.
Executor, nested JSON, file I/O and domain-reader expansion remain in development.
No merge, deployment or auto-merge is authorized by these receipts.
