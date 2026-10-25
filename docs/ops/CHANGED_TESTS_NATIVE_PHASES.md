<!-- docs:meta
topic_id: repo.docs.ops.changed-tests-native-phases
authority: repo_only
audience: users
last_validated: 2026-10-25
validated_by: devpass-SIO-281-operational
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ops.changed-tests-native-phases
-->

# ChangedTests compile/native phase protocol

The ChangedTests gate explicitly selects `changed-tests-v1` for its identified
raw modular compiler on Linux x86_64. A run-pass test compiles once within a
fixed 30-second compilation budget, then executes that retained ELF within the
existing native budget. The native default remains 30 seconds; the existing
`//@ timeout:` parser remains authoritative. This harness does not read
`.timeout` sidecars. Other callers retain the existing wrapper-run path.

The hosted job [121309744095](https://github.com/Sounio-lang/sounio/actions/runs/40239955915/job/121309744095)
at source `3c7c8073411a7e7254d0d9424f59f8e49eb49180` passed 10 of 12 selected
ChangedTests; the genomics batch pipeline outlier and prior tests exceeded the
combined compile/run 30-second envelope. The earlier f64 and SCO phases passed.
The tested raw compiler SHA-256 was
`a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3`.

Serial diagnostic Slurm 17588 completed successfully with that compiler and
those sources. The original combined envelopes took 21.54 and 22.27 seconds;
separate compiles took 13.57 and 13.50 seconds and executions of the retained
ELFs took 7.96 and 8.78 seconds. This did not reproduce the hosted timeout or
establish concurrency, f64 lowering, or numerical wrong-return causality.
The protocol provides bounded phase attribution; these observations are not a
performance improvement measurement.

## Explicit compiler binding

The gate supplies these variables together:

- `SOUNIO_TEST_NATIVE_PHASES=changed-tests-v1`
- `SOUNIO_SOUC_RAW_MODE=modular`
- `SOUNIO_TEST_SOUC_BIN` and `SOUNIO_TEST_PHASE_COMPILER`: the same selected raw ELF
- `SOUNIO_TEST_PHASE_COMPILER_SHA256`: the gate's SHA-256 of that ELF
- `SOUNIO_TEST_NATIVE_PHASE_OUT`: an absolute existing output directory

The harness requires an explicit compiler selection and identical canonical
paths for the recorded raw compiler, selected engine, and routed raw compiler.
It checks the expected digest and ELF magic. Unsupported hosts, protocol names,
compiler modes, identities, or output paths refuse before compilation. There
is no alternate compiler discovery in this protocol.

The helper uses the existing raw modular argv `raw source -o fresh_elf` directly,
in the same working directory and exported environment/stdlib context.
The modular wrapper currently ignores a raw nonzero status if a nonempty output
exists. Direct invocation in this restricted protocol preserves the actual
compiler status; it does not repair the wrapper for other callers. A compiler
error diagnostic with status zero is also refused.

## Execution and evidence

Every test receives a new unique directory. Compilation must produce a regular,
nonempty, non-symlink x86_64 ELF. The helper makes it executable, records its
SHA-256 and device/inode/size/mode, and runs that same path once. Compiler and
entry-source hashes are checked before native execution and afterwards; ELF
hash and identity are checked after execution. Imported sources are covered by
the caller's frozen source catalog, not by the entry-source hash alone.

Each directory retains command/context records, compile/native logs and statuses,
phase timestamps, compiler/source/ELF hashes, ELF header, and integrity receipts.
`RESULT.phase` and `RESULT.status` identify the terminal phase and status.
The compiler budget is a fixed 30 seconds and the native timeout is passed from
the existing harness parser. GNU timeout receives a five-second kill grace for
each phase; this grace does not expand the productive phase budget.

Native stdout alone satisfies run-pass stdout markers. Compiler logs remain
available in diagnostics and the existing known-failure matching path.
Existing exit/xfail evaluation remains in the harness. A failed compile, missing
ELF, or integrity failure cannot execute the program as a successful run-pass.

When `SOUNIO_MADAROS_F64_LOWERING_GATE_DIR` is set, the gate creates
`changed-tests-native-phases/` below it so the existing caller artifact upload
retains receipts. The directory must not already exist. Without that persistent
caller directory, receipts follow the gate's existing temporary cleanup policy.

## Operational validation and limits

Run `bash scripts/ci/madaros_native_phase_selftest.sh`. Its C++23 fixture is an
operational fakecompiler that copies an ELF; it compiles no scientific Sounio.
It checks legacy routing and the existing wrapper status-mask behavior, one
compile per test, distinct phase timeouts, invalid/missing ELF refusal,
raw failure even with an emitted ELF, error-with-status-zero refusal,
native-only markers, source/ELF mutation, xfail/exit handling, unique concurrent
outputs, and compiler identity/mode refusals.

Operational controls are separate from execution of the original scientific
fixtures and from hosted CI acceptance. The source checkpoint contains no claim
that the new path has reproduced or cured the hosted timeout. A focused replay
also does not qualify the entire ChangedTests corpus or the separate SIO-185
Stdlib Reliability E2E caller.
