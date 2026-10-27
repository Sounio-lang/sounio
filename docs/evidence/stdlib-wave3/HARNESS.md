# Phase-isolated stdlib runner

The runner discovers the complete test corpus and retains existing annotations.
Check and compile each have a separate 1200-second default deadline. Native
execution uses the original test deadline (including slow-test annotations).
Each phase has a 16 MiB output bound. Exceeding it terminates the process group,
records exit 122 and output-limit, and preserves the captured prefix plus an
explicit failure marker. A prefix containing the expected marker cannot pass.

Compilation uses the supported wrapper compile command and requires a nonempty
artifact before native execution. Compile refusal, compilation timeout, native
failure, native timeout, and output limit remain separate result reasons.
Captures seal check, compile, and native outputs separately. Worker copies bind
every stdlib file, including generated binary assets.

Slurm 18111 qualified the bounded subprocess helper with successful execution,
nonzero exit preservation, missing executable, a TERM-resistant timeout, and a
runaway writer. Slurm 18112 qualified the complete runner using isolated mock
compiler controls: one pass, compile refusal, native output overflow despite an
expected marker, and native timeout. All eleven phase receipts finalized.
These are harness checks; they do not establish Sounio compiler or corpus passes.

Raw receipts are under the wave3 scratch packet, harness-qa-3 and
harness-e2e-qa. Earlier setup attempts 18107 and 18110 are retained separately:
the first used an absent Node path; the second started before its copy completed.

The modular native wrapper now preserves the raw compiler exit status, requires
the explicit compilation-success record and a nonempty artifact, and rejects
NATIVE_REFUSAL even when a partial ELF and success record coexist. Slurm 18194
preserved the previous false acceptance and rejected refusal, exit 9, signal 132,
empty output, and absent success record. Both the fresh modular compiler and
the canonical seed compiled and executed an unchanged real control successfully.
The C++23 test double is tests/reference/stdlib_fake_compiler.cpp.
