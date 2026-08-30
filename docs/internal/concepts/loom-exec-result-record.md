<!-- docs:meta
topic_id: repo.docs.internal.concepts.loom-exec-result-record
authority: repo_only
audience: users
last_validated: 2026-08-30
validated_by: codex-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.loom-exec-result-record
-->

# SOUNIO-LOOM-EXEC-RESULT-RECORD

Concept-ID: `SOUNIO-LOOM-EXEC-RESULT-RECORD`

Kind: `executable`

Owner: `founder`

Semantic-Lane-ID: `loom-hostd-exec-cell-attachment-20260830`

Write-Set: Garden, Sounio action 9036, freeze gates, OCaml projection, host
attachment, provider return, and evidence.

Read-Set: frozen actions 9030 through 9035 and the source-built compiler
measurement.

Integration-Target: a fresh DynamicUser ExecCell emits a canonical action-9036
record and the provider receives a read-only, non-bearer result handle.

Authoritative-Only-If: frozen Sounio action 9036 refuses the unchanged
artifact-binding witness with `DENY577`, while a mutant deleting only that
comparison admits it.

Semantic-Boundary: runtime values are observations, not expected results.
Sounio defines the canonical fields and handle recipe. Material layers may
measure those fields but may not omit bindings, promote the handle, execute the
artifact, or synthesize semantic authority after execution.
