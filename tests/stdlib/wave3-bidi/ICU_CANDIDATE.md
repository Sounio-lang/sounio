This is a supplemental ICU79.1 diagnostic candidate, not a replacement of the pristine pinned oracle and not an acceptance waiver.

The minimal patch is `tests/reference/wave3_bidi_icu79_isolate_override.patch`:
- patch SHA256 a49e4522d9e1526d069d436a543c523c9750083f815b1eeecbafdfca11f47907
- pristine source/common/ubidi.cpp SHA256 d6bfdaedb6cc9b3fafcb3c0fb0c13c58075b62eb9ff278dfb979bae245f32ebd
- derived ubidi.cpp SHA256 7ea44ea43be9fd2f420b4d692d202512998c2c03a464be9f70547eb8abae2e67

Exactly two assignments change: isolate initiators LRI/RLI and PDI retain `embeddingLevel` including its current override flag. The derived file is bound to exact unique source anchors, and reversing those substitutions must reproduce the pristine bytes. The patch changes no test, expected value, Unicode data, input or tolerance.

`scripts/stdlib/wave3_bidi_icu_candidate.sh FRESH_ABSOLUTE_OUTPUT` is a Slurm-only reproduction helper requiring ICU_PRISTINE_ROOT and the C++/Node tools. It extracts the pinned official archive into a fresh private build, applies only those two exact substitutions, emits the reviewable unified diff, and rebuilds with C++23. It hashes and rechecks the entire pristine installation/source/archive, the derived source tree, candidate installation and proof executables. It never writes into or selects the pristine prefix. The main BIDI acceptance gate refuses a prefix identified by this candidate's SCOPE metadata.

Slurm18354 builds and installs the candidate SDK in:
`/scratch/sounio-stdlib-wave3-20261027/services/icu79-isolate-candidate-02/results/candidate/install`.
The build itself succeeds and all source/pristine/candidate guards pass. Its first harness stopped before the comparisons because an overstrict linkage check demanded unused libicui18n, correctly omitted by --as-needed for BIDI. The reproduction helper now requires the actually used libicuuc and libicudata bindings.

Slurm18358 performs the read-only comparison follow-up against that immutable SDK:
`/scratch/sounio-stdlib-wave3-20261027/services/icu79-isolate-proof-03/results`.
The independent three-witness C++ proof passes. The exact official line61 diagnostic changes from disagreement to agreement, in explicit and auto paragraph modes, through both ICU level APIs and the visual map. However, the FULL strict official-corpus comparison still FAILS at BidiCharacterTest line67, mode2, scalar6:
`202A 05D0 0028 05D1 202C 202D 0029;2;1;x 3 3 3 x x 2;3 2 1 6`.

Therefore this patch is not full ICU conformance and is not accepted as the locked oracle. The pristine comparison remains failed, the candidate comparison remains failed, and every official Sounio level/order/base expectation remains mandatory. No exception list or fallback was introduced. The first attempt18351 stopped before building because the worker lacked an optional patch utility; the helper instead uses its already verified exact source substitutions and unified-diff artifact.

The installed source-level mechanism and the bounded independent proof are documented in the frozen line61 diagnostic receipts. They establish the three named witness results, not a general independent UAX9 implementation. Further patch scope or oracle-contract decisions belong to root review.

All closed BIDI/ICU receipts were recovered after the DL380p worker eviction without executing code on the host. The durable copy is `/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services-closed-export-20261027`. Its relative `HOST.before.sha256` and `HOST.after.sha256` are byte-identical with digest `1213214aec24e6ae71a340a98b678dbb4402972dcfd41e54a2c7f688c8b6ab2d`; `DURABLE.verify` rechecks every copied file. This custody proof preserves failed attempts and diagnostic-only statuses; it does not convert a failed conformance result into acceptance.

The official generator and wrong-level mutation control passed Slurm18337. Slurm18372 regenerated the complete corpus byte-for-byte and proved the acceptance gate rejects this diagnostic candidate prefix with status2 before creating an acceptance output. The canonical small `data/current.bin` is a 64-case original-data bundle for the ordinary fixture runner; it does not replace the full 861948-case gate.
