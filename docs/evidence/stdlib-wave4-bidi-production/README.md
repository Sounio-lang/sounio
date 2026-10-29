# Production BIDI runtime-fixture artifact integration

Status: production scripts implemented and operationally qualified. The canonical
whole-runner replay over the official BIDI corpus remains a separate root gate.
This lane did not bootstrap a compiler or repeat complete native BIDI conformance.

## Source scope and custody

The initial root WIP was copied into an isolated worktree, without editing root:

| File | Copied root WIP SHA-256 |
| --- | --- |
| runtime_fixtures.mjs | 1ee48da47ba94445e90f72e0af9506bbd9d1c12837ef85da1b35fd8fcac3c1e9 |
| run_stdlib_e2e.sh | 67b6bb8add2f4dafdb0aeaea33bea842d2568309b2e39a956381d6ab85d3263d |
| stdlib_e2e_diagnostics.mjs | bc5131b46a3fa4d4f3033ac8e46b5c646eb8efc77ba12ce67c7344af1cdc9a89 |

The copied runner WIP has three narrow contract corrections: reject dangling
.args symlinks, always execute an explicit runtime-fixture even when the generic
run-pass fallback is disabled, and retain/announce fixture output when no
--artifacts-dir is provided. Other changes are in the runtime helper,
diagnostics, new operational controls and these receipts. The root's
test annotation, full Unicode runner, original C++23 generator/header, corpus,
ICU policy and expected answers are untouched.

Golden is owned by the science lane, independently qualified by job19637:
`tests/fixtures/wave4_bidi_runtime/official-corpus.json`, SHA
`b428aba82a8ed4da0b3e21f93a5228761c945d18769900acd153d6e961408eb5`,
immutable commit `3fe291f58e3a969dea3bde35b111722ac62707cb`.
Its 214 file identities were checked byte for byte against the qualified
previous corpus. This lane consumed that exact immutable JSON in private test
snapshots; it does not modify or duplicate the golden in its commit.

## Retained ELF and temporal input

Native start validates the exact bounded helper command and registered test,
the compile phase's successful command/output binding, the helper/golden initial
caller pins and the compiled output's nonempty ELF header/size/SHA. It records
`runtime_fixture` in `native/start.json` before launching the helper.

The helper copies that output to `runtime-fixture/program.elf`, checks equality,
and uses that retained path for every official chunk and both controls:
211 positive inputs plus absent-input and wrong-level controls. Receipt rows
name `program.elf` and its SHA. The generator is a separately identified ELF.
The original compiled temporary file is observed in INPUTS/INPUTS.final and the
ELF inventory before helper completion; it can then be deleted by the runner.
Terminal diagnostics verify the retained copy and the sealed temporal identity,
without trying to reopen the removed temporary file.

Per-chunk `bytes` is the fixture size. Native stdout/stderr bytes actually stored are separately
`output_bytes`, for both chunk and control records. `observed_output_bytes`
counts received data for limiting; discarded overflow buffers do not inflate
the stored-byte receipt. Review found the original
spread `{...row,...native}` had overwritten fixture bytes with output bytes.
The new actual-helper partial-run control uses unequal fixture/log sizes, so
that defect cannot be hidden by matching test-double values.

These are file/command binding receipts, not a universal execve or runtime
dependency attestation. Existing phase deadlines, output limits, signal groups
and drain behavior are preserved. Child processes are not detached.

## Golden and artifact manifests

The golden reader requires the agreed v1 schema, exact counts, 214 ordered
file/byte/SHA entries, four exact producer/source pins, 46 ordered upstream pins
and qualified provenance. Corpus validation checks directory/file regularity,
complete chunk census/header bounds, every byte identity including CHUNKS,
COUNTS and SMOKE, and the original producer's literal counts.

All corpus bytes are checked against the golden before execution and after
execution. Producer/source inputs and upstream pins are checked before/after;
DERIVED, FIXTURES, ELFS and the complete result/control inventories are verified.

`ARTIFACTS.json` uses `sounio.stdlib.runtime-fixture.artifacts.v1`:
fixture, test_path, integer status0/1, executedElf, self, files and directories.
The files list is the exact sorted census of all regular files below the
runtime-fixture root, with relative path, bytes and SHA; directories include
empty directories. Symlinks, escapes, duplicates, nonregular entries and unlisted
files/directories refuse. ARTIFACTS.json declares itself as `self` and is excluded
from its own hash list; its complete size/SHA is instead sealed in
`native/result.json.runtime_fixture_artifacts.manifest`. There is no recursive
self-hash claim.

Recoverable producer/native failures still emit status1, INPUTS.final,
FAILURE and the exact artifact manifest. Missing/incomplete manifests after an
external kill are metadata refusal65, never a completed scientific capture.
Changed source/input identities likewise cannot be promoted to a clean footer.

## Exact diagnostics admission

The only newly admitted subtree is exactly
`tests/stdlib/wave3-bidi/conformance.sio/runtime-fixture` under the capture root.
It must have the corresponding completed native phase with correct helper,
compile output, original ELF, retained ELF path and output bindings.
The verifier then checks all manifest entries/hashes and the manifest's own
native-result seal. A directory named runtime-fixture elsewhere is refused.
There is no broad artifact-file whitelist.

Existing phase census/status/source/caller/argv checks remain active. The new
subtree cannot bypass unfinished/missing native receipts, mutations in earlier
phase output, symlink checks or final source/caller census checks.

## Qualification

Final source gate runs in a private copy of root-bidi-inputs03. No root snapshot
or foreign worktree is mutated. Operational test inputs are explicitly labeled
test doubles; their markers do not constitute Sounio conformance.

-19638: syntax and all29 existing diagnostics controls pass.
-19639: existing29 plus22 artifact controls pass.
-19640: same controls with stricter inventory/upstream consistency pass.
-19642: fixed byte-envelope logic, existing29 plus23 artifact controls pass.
-19644: durable TMPDIR and explicit fixture/output byte receipts pass.
-19645: observed/stored output separation and actual partial overflow control
  pass, existing29 plus24 artifact controls.
-19646: production runner assertions pass until a test incorrectly looked for
  the dangling-args reason in stdout; the reason is in schema-v1 JSON. This
  failed test is preserved; only the assertion was corrected.
-19647: final source gate, existing29 plus25 artifact/runner controls; final
  results are recorded alongside this document.

The25 new controls cover recoverable failure after unlinking the temporary ELF,
retained ELF substitution, exact census, missing/extra files/directories, symlink
files/directories, escaping/duplicate/bad-size manifests, helper/output/compile
bindings, status mismatch and changed input provenance. An actual-helper test
uses CXX=/bin/false and verifies a sealed producer failure with zero native chunks.
Another uses a clearly labeled C++23 metadata ELF that reads only the protocol
header, accepts one chunk and deliberately refuses the second. It validates
actual RESULTS serialization with unequal fixture/output sizes and a retained
ELF after unlinking the temporal original. It does not run the BIDI algorithm. A second invocation deliberately exceeds
1MiB on the second chunk, preserving a valid status1 artifact with native
status122 and distinct observed/stored counts. A runner-only control combines
an explicitly labeled compile-envelope double with the actual helper's producer
failure: the explicit fixture reaches compilation with RUN_ALL_RUN_PASS=0,
retains/announces program.elf without --artifacts-dir, and a dangling .args link
is refused before either compiler phase.

The science lane's independent22 corpus corruption/pin controls are a separate
qualification:19643 passed22/22 on the preceding f4f459 runtime; the final8b35
runtime changes only native output-byte accounting. Its exact-source replay19648 passed22/22 in16seconds on blob
8b35a59e4b63c2b0df0797ec5c57aea7f01c24e6067a23b1334a31ec2e1b4627,
with clean source/corpus/module/golden readbacks. The independent receipts are
under /scratch/sounio-stdlib-wave3-20261027/services/bidi-protocol50/results
(protocol.after8b35.stdout, AFTER8B35_STATUS and AFTER8B35_INPUTS.final.verify). The canonical root replay
must still execute all211 original native chunks and both real controls using
the retained Sounio program ELF and final golden/scripts, then produce a valid
diagnostics footer.
