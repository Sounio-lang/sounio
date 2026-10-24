<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-inference-scale-2026-10-24.input-catalog-preparation.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-inference-scale-2026-10-24.input-catalog-preparation.readme
-->

# Input-catalog preparation and refused qualification

This custody separates a passing **operational metadata control** from a
functional gate and from qualification acceptance. It contains no compiler ELF,
full source archive, new campaign output or rebound functional receipt.

## Genuine gate17009, refused readiness

`genuine-17009-readiness-rejected65.tar.gz` preserves the original seven-field
receipt from `combined_final_cache02`, its historical gate results/status0 and
compiler identity, the original readiness stdout/stderr and catalog comparison.
The receipt is genuine for that historical gate. **It was not accepted for
qualification**, because its initial catalog omitted two files subsequently
materialized by the wrapper. Verifying only the old listed files was insufficient
to establish full catalog equality. The original receipt is copied byte-for-byte;
it is neither edited nor rebound to a newer catalog.

The original readiness operation did not leave a separate exit-status file. A
new actual metadata-only recheck ran on the worker as UID1000 with clean C locale,
using the same frozen source, guard and genuine receipt. Its real exit65,
stdout/stderr, UID, timestamp and before/after input hashes are included as
`readiness-recheck.*`. No synthetic or self-reported exit file substitutes for
that capture. No compiler ran, allocation ledger row was added or campaign was
submitted by the recheck.

## Operational control17016

`catalog-control-17016.tar.gz` records Slurm17016 STATUS0. The initial manifest
versus the existing warm source differs by exactly two added derived files and
no removed or changed earlier file:

- `bin/madaros-linux-x86_64`;
- `bin/.madaros-linux-x86_64.verified`.

The streamed decompressed archive SHA, tracked `.sha256` value and materialized
ELF SHA agree. The stamp matches that ELF's hash, size, inode, mtime and ctime.
Two normal-mode materializer calls leave the complete warm catalog byte-identical.
The selected guarded compiler d74 remains separately identified; the materialized
tracked prebuilt125c is a derived input, not a substitution for that compiler.
No compiler was executed by this control.

The resulting preparation contract is to materialize in the final source tree
before taking the initial complete catalog, then require complete catalog equality
at the end. Later copies or chmods of the ELF can invalidate the filesystem stamp;
the exact final tree must therefore be prepared before its provenance snapshot.
The metadata control does not establish numerical correctness or qualify a run.

## Partial17017 superseded before acceptance

`superseded-17017-partial.tar.gz` preserves sourceHEAD830e, initial compiler/input
identities, partial results, preparation logs and cancellation/supersession records.
Slurm stderr records cancellation at2026-10-24T18:24:04 by signal Terminated. The
older `SUPERSEDED_17017.txt` RUNNING snapshot is explicitly a pre-cancellation
observation. No final artifacts/STATUS file or success receipt existed at custody
collection. Partial rows are not a completed gate.

The run was stopped before acceptance so that the separately discovered wrapper
error-status capture defect could receive a narrow repair and regression. This
bundle does not claim that repair passed, that numerical/compiler sources changed,
or that later full04/qualification execution succeeded. Prepared old qualification
packets and the historical refusal remain separate; a new final-source gate and
genuine receipt are required before any authorized campaign submission.

## Verification

Run `sha256sum -c MANIFEST.sha256` and `sha256sum -c ARCHIVES.sha256` here. Extract
each archive into its own temporary directory and verify its inner
`MANIFEST.sha256`. Original worker artifacts remain under
`/scratch/sounio-stdlib-inference-scale-20261024/`; no originals were deleted.
Governance metadata records no document-validation event; the operational outcomes
above are supported by their raw receipts and have their explicitly limited scope.
