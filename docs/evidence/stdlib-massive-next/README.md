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
Subsequent focused receipts are recorded below. VCF header/schema expansion and final corpus acceptance remain pending.
No merge, deployment or auto-merge is authorized by these receipts.

## Subsequent focused receipts

- PR 3246: typed function-pointer result layout repair. Slurm 17940 proves the original native failure and corrected callback/alias/cast behavior. The polling executor passes check/compile/native. Packet 24 compares older controls before and after; three retain existing failures and two pass on both. Packet 16 is therefore not an all-green combined gate.
- Packets 21, 22 and 23: JSON decoding/serialization, dynamic FASTA and GFF3 check/compile/native pass. JSON keeps exact numeric lexemes and owns its source copy. FASTA preserves nucleotide ambiguity symbols and supports multiple records beyond fixed sequence caps. GFF3 checks rows, coordinates, phase and lexical attributes; ontology and ID/Parent graph validation are caller responsibilities.
- Packet 25 / Slurm 17952: CSV overflow regression fails before and passes after; independent C++23 numeric and function-pointer-layout references pass.
- Packet 27 / Slurm 17955: checked file -> owned existing CSV reader passes with quoted newlines, long fields and explicit budgets. C++ file/text ownership and earlier platform/byte controls pass.
- Packet 28 / Slurm 17956: nonblocking socket and cooperative scheduling controls pass, including sticky asynchronous connection errors in the C++23 reference. A pending read yields to another completed task, then resumes on readiness.
- Packet 31 / Slurm 17964 on dl380-dl380-proxmox: producer/checker compile and native workflow gate pass. The existing QR, SVD, quantile, SVG and PNG APIs process the checked CSV. JSON replays exactly; independent C++23 arithmetic verifies coefficients, rank-related identities and quantiles, and checks SVG primitives and PNG CRC/Adler/pixels. The original PNG source refused at undeclared append_file in packet 29; its checked I/O repair is exercised in packet 31.

Durable roots follow /orangefs/training/sounio-foundry/runs/stdlib-next-20261027-XX/results. Compiler raw for these modular receipts: e1ee21074398f771d5eeec32a5281a5ff57a8bdf27f09c8a0a2b39a0dd000b41. These remain focused receipts until the final integrated gates complete.

## Interrupted comparison and recovery

Slurm 17917 was interrupted when cpuops-t560-proxmox was removed from the cluster and its partition disappeared. The scheduler requeued it to an invalid partition; the obsolete requeue was cancelled to preserve the incomplete packet. It did not complete main or candidate full acceptance. The captured partial results and scheduler receipt are preserved at /orangefs/training/sounio-foundry/runs/stdlib-next-20261027-07/interrupted-20261027T0228/; partial-results.tar.gz SHA256 b40ceae20b7d679a660a6eb4d6f24c182bd4fff3534718fa5c94d861eb89ffb1. Source archives and raw compiler hashes were verified while migrating validation to the DL380 CPU worker.
