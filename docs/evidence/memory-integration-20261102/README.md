<!-- docs:meta
topic_id: repo.docs.evidence.memory-integration-20261102.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.memory-integration-20261102.readme
-->

# Integration review

VERDICT: LGTM. Integration validation passed on Slurm21231, COMPLETED0:0 in7m22s.

Base: origin/main 26152257e04e. Candidate: f50f698d8fc44d7132e4287c018ad3ab4ec16899, PR https://github.com/Sounio-lang/sounio/pull/3413.

The reviewed file avx512.sio is byte-identical to the previously executed candidate. The original idx_id is undefined. Production register-register callers already supply the source register extension bit; no-index memory callers supply0. Consuming the supplied x_bit preserves both contracts. For mod10 memory, low-three base bits100 require SIB;0x24 denotes no index and preserves the base extension in EVEX.B for R12. The inserted byte precedes the full displacement. Existing no-displacement addressing and other base classes are unchanged.

Golden cases cover complete bytes for both load/store, RSP/R12, positive, negative and unaligned displacements. The expanded independent36-case matrix also covers RDX/RBX/RBP/R13 and executes the generated instructions with full destination-buffer guards. The source-to-native scalar fixture verifies ten signed/unsigned-byte/word values. Both semantic mutations have discriminating failures.

Review fixes included before publication:

- Encoding checker rejects malformed numeric fields, out-of-range bytes and metadata not matching the expected case order before executing machine code.
- Historical Slurm scripts explicitly documented as receipts with fixed paths, not portable CI entrypoints.
- AVX512 suite wrapper comments updated to73 cases.
- Changelog fragment added and docs metadata generated only for the two new README files using repository helpers; registry and snapshot files not changed.

Validation run on integration snapshot: Slurm21231. Canonical build_modular_madaros.sh, public bin/souc checks and compilation with explicit fresh Madaros, C++23 scalar oracle and word-read mutation, emitter unit with pinned bootstrap seed, hardened checker plus native36-case execution, malformed input rejection and no-SIB mutation. Final completion must be read from the matching Slurm receipt, not inferred from submission.

Auxiliary review: darwin-llm qwen38-27b failed HTTP500 and is not counted as completed. Fallback coder-next completed and found no concrete encoding regression under the declared caller contract. Actual backend behind aliases not attested. Primary review inspected the actual diff/callers and remains responsible for the verdict.

Local nonnumerical checks: git diff --check, docs registry check and changelog fragment gate passed. Entire Sounio suite was not run manually; GitHub CI state is tracked separately. No claims about full AVX512 ISA support, every MIR path, compiler distribution, or unrelated backends.

The integration branch is based on current main and contains only the reviewed changes, not the pbpk branch ancestry. Shared dirty worktree preserved. No force push or history rewrite.

## Integration receipt

Fresh current-main Madaros SHA256: af567d40828af64764758229feae30fbb3e0c69fe5b4b76ec601175b44e67599. Public checks of avx512.sio and bytewidth_active_backend.sio passed. Scalar oracle accepted the witness and rejected the widened-read program. Emitter73/73, independent36/36 LLVM cases and36/36 native loads/stores passed. Malformed-case input returned6. No-SIB mutation returned1 and failed T73. Controller log ended INTEGRATION_VALIDATION_PASS.

Slurm21238 is a failed audit assertion, not a compiler failure: the trace correctly showed bin/souc delegating through env MADAROS_RAW_BIN to bin/madaros, but the grep incorrectly expected direct execution. Slurm21241 also executed a successful compiler check but its audit grep omitted the timeout wrapper. Slurm21242 then verified both existing execution traces with the actual env/timeout delegation and completed0:0: FRESH_COMPILER_EXECUTION_IDENTITY_CONFIRMED. Both earlier audit jobs failed their overly narrow grep patterns, not the compiler check. No compiler or tested-source change was made.

The generated prebuilt may be materialized by the launcher for provenance comparison even with a valid override. That message alone does not establish which compiler ran; the selected override and execution trace delimit this claim.

GitHub CI is tracked in PR3413 separately. This receipt does not claim completion of every GitHub job or the full manual test suite.

Main branch rules were checked live: required CI Decision, merge queue, no required approving-review count, and resolved review threads. Integration follows the normal queue without an administrative bypass.

## Final checker review correction

The final primary review found that checking eof() after integer extraction could accept an overflowing extra integer at the end of a case line. The reader now consumes whitespace before deciding that a line is finished and treats every failed integer extraction as malformed input. Slurm21247 COMPLETED0:0 demonstrates the old checker falsely accepting the terminal-overflow fixture, the corrected checker returning6, the ordinary malformed fixture returning6, and all36 independent-byte/native cases still passing. No compiler source or compiled compiler artifact changed. The integration script retains this negative control. Auto-merge was temporarily disabled while this fix was prepared and is re-enabled for the corrected head.
