<!-- docs:meta
topic_id: repo.docs.evidence.compiler-path-attribution-2026-10-22.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.compiler-path-attribution-2026-10-22.readme
-->

# Compiler path attribution — 2026-10-22

Compact public evidence for analysis of the retained Slurm15940 trace. This is
an actual C++23 analysis of existing records on t560, not a new compiler build,
a test suite, a cache activation or numeric concordance.

`PUBLIC_SUMMARY.txt` contains counters and exact-exec-path role hints. Role table
columns are role, syscall records, records with `-1 errno`, decoded requested
pathname operands. Absolute source candidate counts are unique literal-prefix
path spellings per role; they omit relative paths and are not dependency counts.
`DECOMPRESSION.sha256` lists readback hashes in the same table order as
`DERIVED_RAW.sha256`; both are preserved in the private execution recipe too.

The original trace remains in prior private custody. All final detailed tables
and deterministic gzip copies remain private on t560 under
`/home/devsounio/sounio-artifact-archive/compiler-path-attribution-20261022/final`.
This is host-local custody, not an independent physical backup. No private table,
raw trace, executable binary or source snapshot is included in this directory.
Source/trace/binary and derived-table identities are retained for review.

Policy: candidate_only=true; complete_input_closure=false; cache_authority=false.
No source deletion, historical cleanup, quota increase or cache guard change.
