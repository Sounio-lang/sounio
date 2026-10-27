<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-massive-next.readme
authority: repo_only
audience: users
last_validated: 2026-10-27
validated_by: devpass
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-massive-next.readme
-->

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

## Implemented second-wave services and composition

- Native checked file owners and owned text for the existing CSV parser; checked integer/scaled parsing refuses range overflow.
- Owned nested JSON with UTF-8/escape checks, exact number lexemes, depth and payload budgets, decoded access and serialization replay.
- Dynamic FASTA and GFF3 records; VCF 4.3 owned header/sample context with declared INFO/FORMAT type/cardinality validation. Breakend/reference/file-wide semantics remain explicit limits.
- Bounded MPMC i64 channels, typed socket readiness and nonblocking connect, and a Sounio polling executor with cancellation and monotonic deadlines.
- Existing dynamic QR, direct Jacobi SVD and quantiles feed a structured JSON report and existing SVG/PNG renderers. Checked native PNG output replaces the undeclared append_file route; actual caller effect rows are propagated.

## Qualification packets

| Packet / Slurm | Verified result |
| --- | --- |
| 25 / 17952 | Original CSV overflow witnesses fail before and pass after; independent C++23 numeric and callback-layout references pass. |
| 27 / 17955 | Checked file/owned CSV Sounio and C++23 file/text reference pass. |
| 28 / 17956 | Nonblocking connect/readiness and actual yielding socket executor pass, with independent native socket controls. |
| 31 / 17964 | CSV-to-QR/SVD/quantile-to-JSON/SVG/PNG workflow passes; JSON owned replay and independent C++23 numerical/chunk/pixel checks pass. |
| 37 / 17971 | Fresh modular build using the privacy-fixed source-built seed passes. Its SHA256 e1ee21074398f771d5eeec32a5281a5ff57a8bdf27f09c8a0a2b39a0dd000b41 is byte-identical to the raw compiler already consumed by the matrix. |
| 38 / 17973 | Expanded integrated native gate, metadata controls, ownership/privacy negatives, scientific artifacts, and exp-helper gates on both engines pass. |
| 39 / 17975 | Initial VCF header context and preserved original per-record fixtures pass check/compile/native. |
| 40 / 17976 | Expanded gate including hardened VCF/independent C++ reference passes. Subsequent deep replay was stopped after its sparse worker lacked /usr/bin/time; native 127 rows are dependency failures, not successful execution. |
| 41 / 17979 | Installed canonical seed self-reproduces; repeated compiles agree; live receipt source/seed hashes and wrong-source mutant control pass. |
| 42 / 17981 | Final integrated/ABI, registry, known-failure and deep replay launched with explicit sealed GNU time; see durable receipt for completion. |
| 32,33 / 17965,17966 | Same-compiler main/candidate full slow matrix in progress; per-case receipts are durable. No complete full-corpus acceptance is claimed. |

All packet roots are /orangefs/training/sounio-foundry/runs/stdlib-next-20261027-NN.
Compiler implementations are still independently reviewable in #3245 and #3246.
The integrated tree consumes their source changes; this is not evidence that either PR was merged.
The full matrix candidate snapshot precedes VCF/header and final harness changes; final-tip replay remains required.

## Evidence recovery and interrupted runs

The original packet05 worker created directory05 but copied into directory03.
This overlaid common result files in the durable packet03 directory. Scratch originals were intact.
Exact original packet03 results are restored under packet03-recovered; exact packet05 results,
worker and source repair archive are restored under packet05. The mixed historical packet03 directory
is preserved for audit and must not be used as authoritative acceptance evidence.
Seed hash checks blocked packets34/35 before any compilation when the first copy was empty.
Packet36 then reached a missing-g++ preflight; packet38 reran successfully on the equipped DL380.
Packet37's completed compiler and packet36's dependency receipts were copied from local DL380p
scratch to the shared durable roots after their jobs completed.

The old cpu-ops node/partition was removed externally during packet07. That full run was interrupted;
its partial archive and original scratch results are preserved. Replacement full matrices32/33 use
private local worker copies and write each case directly to shared CephFS storage.

## Frozen source qualification and remaining acceptance

Implementation snapshot 1616f882f09a942331c2e571e7a30e9f1f34a428 uses the qualified
bootstrap 9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a.
Slurm 18013 / packet61 rebuilt the full frozen compiler closure, sealed self-hosted
and stdlib source catalogs before and after, and reproduced raw compiler
81047d46f377e0b01dd3f93bc8152ab950583ab3156284812bd0fbc3ad98a196 byte for byte.
Its build, source-integrity and binary-comparison statuses are all zero.

Slurm 18011 / packet60 passed the expanded integrated gate and original complex
fixtures on the bootstrap, including the independent C++23 complex reference.
The subsequent deep gate exited one: its focused child gates still called
/usr/bin/time directly, absent on that worker. Packet65 repeats the unchanged
frozen source after installing that dependency; completion is a separate receipt.

Matrices51/52 were interrupted by NODE_FAILURE at 2026-10-27T05:09:28Z.
Their per-case receipts are retained, but neither has a completed full result.
The trap-written WORKER_STATUS=0 is not evidence of completion.
Matrix32 completed with 584 selected cases and no selection errors; failures
remain. Matrix33 observed 647 cases but missed nb2_slot_cache/profile_main.sio;
its two selection errors prevent full acceptance. Packet68 repeats main on a
stable DL380 using raw810. A final candidate replay remains necessary.

BASELINE61.json classifies every historical failure using identified individual
case receipts and focused before/after evidence. Seven collection cases, the
original IntResult case and original complex case now pass with their focused
repairs. Three other cases are observed passing in the candidate. The remaining
49 have failures independently observed in main and candidate (30 checker/API
contracts, 13 native failures, six declared-budget overruns). Those are
classifications, not root-cause repairs or a whole-stdlib green claim.

Hosted CI run40451567363 for head1616 passed source-bootstrap, Linux and macOS
native self-host, current-source f64, witness and contract jobs. Full Test Suite
failed three original graphics smokes after PNG acquired a native-loader
dependency unavailable to the bootstrap. The separate deep job failed only
genomics-count-identities: its old gate expected the bootstrap to accept private
field reads, while the qualified new bootstrap correctly emits E259.
Both regressions are being repaired without skips or tolerance changes.

Slurm18023 / packet69 passed the full genomics count identity gate with the
stronger privacy expectations: all numerical, scale, mutant, canonical privacy,
bootstrap E259 and both-engine linear controls passed; source, compiler, caller
and executable hashes verified after execution. No numerical case, budget or
tolerance changed. This repairs the exact sole failure in the hosted deep report.
