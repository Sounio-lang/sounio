# Literal argv sidecars and complete stdlib input guards

The canonical runner now executes optional TEST.sio.args as literal file-path arguments, one LF-delimited line per argument. Blank lines are ignored and an unterminated final line is accepted, matching the existing Pireus argv gate. Spaces, quotes, dollar signs and other shell metacharacters are ordinary bytes; no eval, shell expansion or word splitting is used. Relative paths resolve against the test repository root. Absolute external paths remain literal and are guarded too.

The shared Node helper validates UTF-8 and refuses NUL, unreadable, non-file, missing referenced paths and dangling sidecars. A missing sidecar continues to mean empty argv. Check-only, ignore, compile-fail, native refusal, slow selection and deadline rules are unchanged. Native execution remains inside the same bounded-command deadline and output cap; the helper starts the ELF with an argv array in the same process group.

Diagnostics bind sidecar presence/content, literal order, resolved file identity and dependency hashes before execution, and persist the observed after-manifest even when a mutation prevents completion. The full runner records ARGV_INPUTS and ARGV_WORKERS before/after manifests, plus SOURCE_INPUTS and SOURCE_WORKERS manifests. Its root and copied-source guards include existing tests, stdlib, scripts, self-hosted, docs, tools and runtime directories. Missing optional directories alone are not an error. File and directory partition selections retain the same test paths and counts. These are before/after identity guards, not a universal execve attestation.

## Frozen material-engine inputs

With the original runner, the original test checked and compiled but returned 2 because no arguments were passed. After argv support, it returned 1 with nine structural failures: five current Markdown documents no longer matched the hashes already frozen in pmea_expected_parent_hash. This new failure was retained rather than changing any expected hash or assertion.

The five exact preimages were recovered from Git commit8e2f11899cd5b556f6abbf3a01d8c9ce431f3915. FROZEN_PARENTS.json records the full twelve-parent order, current original hashes, historical commit/blob and expected SHA-256. The sidecar changes only those five paths to immutable historical fixtures. The other seven paths and all existing assertions/constants remain unchanged. Current docs are untouched. Historical fixture prose does not become a new validation or hardware receipt.

Slurm19603 first qualified the unchanged original test with the candidate sidecar: check/compile/native0 in loop-region modes0 and1. The same ELF refused no arguments with2 and the current drifted documents with1. Only after this positive was the production sidecar changed and the obsolete known-failure comment removed. Slurm19606 replayed the final source in both modes, all seven harness suites and the same negative controls.

## Preserved executions

- 19597 FAILED1 after seven harness suites passed: staging could not overwrite an old diagnostics file created with root ownership. No real material test phase ran. It is retained as an orchestration failure.
- 19599 FAILED1: twenty argv controls and seven harness suites passed; the original material case changed from no-argv2 to current-docs1, printing failed9. Final input/raw guards passed.
- 19603 COMPLETED0/54s: candidate twelve frozen inputs verified; original material case passed both modes; no-argv2 and current-docs1 controls passed; final guards passed.
- 19606 COMPLETED0/95s: final production test/sidecar/fixtures passed the same two original imported runs, seven harness suites and both negative controls; final guards passed.
- 19608 COMPLETED0/46s: all seven suites passed after expanding full-suite source guards to docs/tools/runtime. Controls cover worker mutations in all three directories and original-root mutation in directory partition. The twenty literal-argv controls passed again. Only run_full_suite.mjs and its contract fixture differ from the material-qualified19606 bundle; standalone runner, helper, compiler, program and all data are byte-identical.

All Sounio execution used raw8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932. Compile/native bounds remained1200/30seconds for the material test. Source test count is unchanged. No whole-stdlib zero-failure claim is made. Recipes accompany this record; full source, captures, failure receipts, executables and raw compiler are being sealed into a separate custody supplement.
