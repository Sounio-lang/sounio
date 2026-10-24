<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-inference-scale-2026-10-24.materializer-exit-status.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-inference-scale-2026-10-24.materializer-exit-status.readme
-->

# Launcher materialization exit status: isolated before/after regression

The narrow change in bin/souc and bin/madaros replaces a negated materialization
condition with a positive condition and an else branch. Previously `$?` in the
failure branch was the status of `! bash ...`, namely0; without a higher-priority
override the launcher therefore exited0 instead of the materializer's78. The
new branch retains the actual78. Materialization, strict mode, provenance and
resolution order are otherwise unchanged. The source commit supplied by the
coordinator is6ffc1a5bd8eaebeef2d15ccced830bfbc832a648.

Final source identities:

- bin/souc72ce4b8158c1ec33647040803c06d745c05433feb2290992813a734b6155a23a
- bin/madarose308d270bba082fee62230627196fc19fdb28b869a2edbd8f3e712dc6e1f0471
- tests/operational/materializer_exit_status.sh2bd175ed07d32fc5020b8e249f0ddeb6140585b0d90aa00bcb4c5d91a037c4dc

## Actual Slurm observations

**17018** exited127 before executing the cases because the worker lacked the
optional `file` utility used to verify ELF type. That operational failure, its
script, toolchain output and status remain preserved. It is not failing-before
semantic evidence. No utility was installed; the test used the existing `od`
ELF-magic check on retry.

**17019** completed STATUS0. Its actual RESULTS.tsv contains216 case rows,
including both before and after launchers. Each process invoked the real `check`
verb on an isolated readable source path, not --version/info. C++23 programs were
compiled inside the Slurm allocation as actual ELF controls: one prints a marker
and returns17; a different ELF returns23. Their producer `.comment` strings report
GCC Ubuntu13.3.0; source, compile commands/logs, ELF magic and SHA256 are retained.

These are **launcher exit-status controls**, not Sounio compilation, typechecking,
numerical correctness or scientific compiler acceptance. The controlled ELF
intentionally does not parse the source path passed to check.

For empty/corrupt gzip, missing gzip, empty/missing SHA, malformed SHA and wrong
SHA, without an override the actual before launcher returned0 and the after
launcher returned78, with no control-program stdout. Valid raw, alias and local
artifact overrides still executed17 in ordinary mode, while strict mode rejected
uncommitted identity with78. A valid committed payload and a byte-identical
explicit override executed17 under strict mode. A different valid override
executed23 normally and was rejected78 under strict mode. Input and control-ELF
hashes were verified unchanged after the cases.

Every invocation used env-i with fixture-local HOME/TMP and an isolated resolution
root. No inherited compiler override, host artifact fallback, shell compiler stub,
active checkout mutation or lazy materialization bypass was involved. Both
packed files absent is the materializer's pre-existing sparse-tree/no-op policy;
this test does not change that policy or claim it must return78.

## Compact custody and replay

Before/after wrapper bytes are retained as `.txt`, along with each job/testdriver,
C++ source, RESULTS/READY, input and ELF identity/verification receipts, and
scheduler metadata. No ELF or repeated synthetic gzip payload is copied here.
The complete scratch case trees remain under
`/scratch/sounio-stdlib-inference-scale-20261024/materializer_exit_status01` and
`materializer_exit_status02`.

CASE_RECEIPTS.tar.gz contains ONLY the216 cases' stdout, stderr, status and GNU
resource text files. The sorted NUL file list and original-byte hash manifest are
retained. Archive uid/gid and mtime are normalized, pax atime/ctime are removed,
pax header paths are fixed, and gzip carries no timestamp. A second independent
pack produced the same SHA256; final extraction/readback checked every case file.
The deterministic archive SHA256 is
4109dcda869711a1cd3eb84f30abb54c398ed848da3cb1de14a2e4c7b3875bf2.
No source/fixture trees, executables or gzip payload duplicates are inside it.

SOURCE_MAP.tsv/ORIGINAL_FILES.sha256 bind original scratch receipts to copied
bytes. Large text is losslessly compressed. Original per-run manifests may refer
to omitted scratch artifacts and are historical receipts. FILES.sha256 is the
final copied-file manifest. Custody preparation reran no control program or
compiler, and makes no performance or final-corpus acceptance claim.
