<!-- docs:meta
topic_id: repo.docs.evidence.sampled-uncertainty-2026-10-20.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.sampled-uncertainty-2026-10-20.readme
-->

# Sampled uncertainty: retained execution evidence

Frozen executable source: `52ba99b87d49da04e97524fa0d8911b1d1d4d46b`.
Remote main incorporated: `085da1da15f3c05946b66e4ddf7b60a0e29d5d40`.

- `v1`: job15771, failed on missing caller Panic after six successful checks.
- `v2`: job15772, compiler build and fourteen checks succeeded; demo returned1.
- `v3`: job15775, attested compiler reuse and eighteen checks succeeded; demo returned1.
- `v4`: job15776, eighteen checks, native compile and figure execution succeeded.
- `figure`: final native HTML, standalone extracted SVG, compressed Linux native
  ELF, source/program identities, compilation log and exit receipt.
- `ORIGIN.txt`, `COMPILER.sha256`, `worker-source.sha256`: compiler custody identity.
- Dispatchers/submission scripts: package-only Forge transport definitions for
  the attempts; no replacement of the worktree's canonical test harness.
- `SLURM_ACCOUNTING.txt`, `STORAGE.txt`: dated whole-job/storage observations.
- `FILES.sha256`: relative checksums for this retained evidence set.

The source snapshot itself and the large compiler ELF stay in remote custody,
with identities recorded here. Git contains the source and the small program/output
artifacts. v4/MANIFEST.sha256 refers to the original source-packet paths, not paths
inside this evidence directory. Use FILES.sha256 to check the retained files here.

The SVG is extracted from the actual native HTML, with interior bytes unchanged
and a trailing newline. XML parsing and visual inspection concern this artifact.
They do not establish independent scientific numerical agreement, Canvas execution,
pixel parity, confidence coverage, a corpus run, full CI or merge.

No tests were added or run for this increment. The historical114-entry acceptance
remains separate. See the [execution receipt](../../plans/SAMPLED_UNCERTAINTY_EXECUTION_2026_10_20.md).
