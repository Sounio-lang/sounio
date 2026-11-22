<!-- docs:meta
topic_id: repo.docs.evidence.logarithmic-2026-10-21.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.logarithmic-2026-10-21.readme
-->

# Logarithmic primitives native evidence, 2026-10-21

Initial Slurm job 15823 rebuilt the source-tracking seed and modular compiler from source 5708a48a22684f242b3fa5d8d376d6389c456b68. Final job 15825 used source 27ea4ec6c2b97892e9e92cf982b0af90e2241023 and reused the exact attested compiler after proving unchanged bin/scripts/self-hosted/stdlib trees. The final source changes only the visible seven-sample caption. Both jobs completed 0:0.

Each job ran seven explicit typechecks and compiled/executed both real visualization examples. No tests or independent numerical oracle were added or run. The generic transport label check_corpus is not a full-corpus acceptance claim.

The two top-level example directories contain final exact stdout, compressed native ELF, resource receipts and derived artifacts. figure.html and figure.svg were extracted from stdout; the SVG interior is unchanged with one trailing newline. preview.png is a full-size 1380x620 sips rendering. Both final SVGs passed XML parsing and both full images were visually inspected. Canvas execution is in memory; no pixel comparison is claimed.

Raw logs and input manifests are losslessly compressed; original hashes are retained. Source snapshots and large compiler binaries remain in remote custody. FILES.sha256 hashes all files here except itself using relative paths. GNU time records one whole-program observation per final example, not isolated kernel or comparative performance. A displayed 0.00 seconds is centisecond resolution, not zero elapsed time.

Numerical concordance, global ULP accuracy, boundary runtime coverage, calibration, scaling comparisons, full corpus, CI and merge remain open. Shared quota remains 3 TiB, with no new deletion.
