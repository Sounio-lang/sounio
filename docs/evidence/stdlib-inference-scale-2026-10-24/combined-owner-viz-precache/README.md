<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-inference-scale-2026-10-24.combined-owner-viz-precache.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-inference-scale-2026-10-24.combined-owner-viz-precache.readme
-->

# Combined owner and visualization gate before the rank-cache change

Slurm **16933**, run `combined_owner_viz02`, finished with `STATUS=0` on source
`8d4da7665c0b4164bea2ad1c9b9e8e25dcb389b0`. This records the combined scientific
native/C++ gate **before the `82b` performance-cache change**. It must not be
used as evidence that later cache changes, subsequent compiler-source repairs,
or the final integrated tip passed this same gate.

The selected compiler identity is
`92d2108ef595c5382d582aa363df6e95dab907a554f039d5f8f008073e535c21`.
`RESULTS.tsv` records the individual checks, compilations, executions, stdout
markers, independent C++ comparisons, negative controls and final input/compiler
hash checks. The exact protocol is retained in `PROTOCOL_GATE.sh` and
`PROTOCOL_JOB.sh`; these are historical copies, not new entrypoints or updates
to the shared scripts. Resource files describe their individual programs and
are not a general performance benchmark.

`receipts.tar.gz` contains the original `receipts/` tree with timestamp-free gzip copies of raw stdout/stderr, check and
compile logs, resource measurements, artifact identities and input manifests.
`logs/` preserves Slurm output. `SOURCE_HEAD.txt`, `SOURCE_COMMIT_PREFIX`,
`COMPILER.sha256` and `SUBMISSION.txt` bind the run to its recorded identities.
No ELF, generated executable, full source archive, raw campaign asset or cache
is duplicated here. Originals remain under
`/scratch/sounio-stdlib-inference-scale-20261024/combined_owner_viz02`.

The campaign report in this gate describes the frozen original B199 batch0
observations. Rendering and numerical concordance do not establish statistical
calibration, inference eligibility, campaign completion or human acceptance.
The later compiler self-check scopes are kept in the separate
`compiler-selfcheck-scopes` sibling directory.

Verify this custody with `sha256sum -c MANIFEST.sha256` from this directory.
The README and its generated governance metadata are included in that manifest.

## Compact receipt archive

The loose receipt copies were packed after byte-for-byte extraction and full
historical-manifest verification. Original worker artifacts and the prior Git
commit remain intact. No numerical result, source or acceptance criterion changed.
The archive uses sorted paths, fixed zero timestamps, numeric owner/group0 and
timestamp-free gzip. A second independent packing produced identical bytes.

Archive SHA256: `910b72edd900f59598be9e344ff2510e145bb88ffc08ad5005f220963dd3a2fb`.

First verify this compact custody with `sha256sum -c MANIFEST.sha256`. To replay
the original receipt manifest, copy this directory to a temporary directory,
extract `receipts.tar.gz` there, then run `sha256sum -c RECEIPTS_MANIFEST.sha256`
from that copy. The historical manifest intentionally addresses the extracted
`receipts/` paths. `PACKING_VERIFY.txt` records the completed byte-preservation
checks; no executable or full-source archive is introduced.
