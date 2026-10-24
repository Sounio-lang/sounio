<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-inference-scale-2026-10-24.compiler-gen2-ci.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-inference-scale-2026-10-24.compiler-gen2-ci.readme
-->

# Actual PR3080 gen2 CI: expected run rung reached

This is read-only custody of GitHub run40115178059, job120934877273, at exact
source92c9e22b71997c413b5c03cfa0ceeab89f148f7b. The single-job API now reports
COMPLETED/SUCCESS, started2026-10-24T17:14:01Z and completed17:55:52Z. Both job
metadata and checkout/git-log output bind the tested source. No cancellation,
retry, rung edit, compiler edit or new native execution was performed here.

[GitHub job](https://github.com/Sounio-lang/sounio/actions/runs/40115178059/job/120934877273)

## Actual rung, including its remaining wall

The complete job log records:

- Gen1 checks self-hosted/compiler/main.sio: rc0/errors0.
- Gen1 compiles gen2: rc0, merged_ir_functions14931, into_acc_done129 against
  the unchanged minimum122.
- Gen2 identifies itself: banner=madaros.
- Gen2 attempts to compile gen3: rc181. The preserved gen3.log explicitly says
  `arena full (allocation of 2080 bytes)` and describes the generated runtime's
  fixed bump arena. This is not evidence of a host OOM kill.
- `reached run`, `expected run`, and the gate's MADAROS_FIXED_POINT_OK message
  explicitly names gen3 as the next wall.

The successful result is **the recorded run rung**, not gen3 compilation or
bit-identical gen2/gen3 fixed point. The workflow did not override EXPECT; the
preserved exact-source gate defaults SOUNIO_MADAROS_FP_EXPECT to run. The workflow
and gate source are retained as text evidence, without executing them here.
The current-source f64 lowering and scalar PTX steps also completed successfully.

## Downloaded artifact and actual native identities

Artifact12570551304 is named
`madaros-gen2-proof-92c9e22b71997c413b5c03cfa0ceeab89f148f7b`, size26358353bytes,
and was non-expired at retrieval. The downloaded ZIP SHA256
`8ee874acb16b565ae17010a135b3a321bd6de2943037ed881f6c185ec035a72a`
matched the upload digest in the job log. The ZIP and executables remain in
scratch/temp custody, not Git. ELF bytes were streamed from the ZIP into SHA256;
no ELF was invoked or copied into this folder.

- Gen1 `madaros`: d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6,
  identical to the separately guarded compiler-effectbridge build.
- Gen2 `fixed-point/madaros.gen2`: 9a847402bbc4afc14600c7af416de76c85cd009f40a37a54bcb421d0ca0bb10a,
  uncompressed109945288bytes. Its execution evidence is the CI rung log above.
- Record witness ELF: 0c707d37e9f3f126a4becae590ea6dcf52b7639fd134aacc7426ebf4f524fb33.
- Tuple witness ELF: 9a19d5c319928c3d8e679e30d9e1ff05e2701bfc1452b75df348aa3871b063cc.

The artifact's nested-record-owner RESULTS records compile/native/stdout-marker0
for both witnesses and expected1/actual1 rejection of incompatible owners. Its
original compiler/input hashes and verification receipts are retained. Other
capacity-witness ELF identities are in ARTIFACT_ELFS_STREAM.sha256; these are
artifact identity observations, not newly executed witnesses.

## Custody boundaries

Original JSON metadata, job log, selected artifact text logs/receipts and archive
listing are preserved. Large text is losslessly gzipped. ARTIFACT_TEXT_ORIGINAL
and API_ORIGINAL hash the original uncompressed bytes; SOURCE_MAP links copied
artifact members to their artifact ID. FILES.sha256 is the authoritative final
copied-file manifest. No ELF, ZIP, cache or full source archive is included.
The source texts `.sh.txt` and `.yml.txt` came from GitHub's contents endpoint at
exact92c, not from a later moving branch. This folder does not represent the
separate parent Linux/macOS/Lean CI or change its existing retry policy.
