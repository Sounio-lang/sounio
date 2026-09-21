<!-- docs:meta
topic_id: repo.docs.internal.concepts.loom-exec-operation-catalog
authority: repo_only
audience: users
last_validated: 2026-08-30
validated_by: codex-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.loom-exec-operation-catalog
-->

# SOUNIO-LOOM-EXEC-OPERATION-CATALOG

Concept-ID: `SOUNIO-LOOM-EXEC-OPERATION-CATALOG`

Kind: `executable`

Owner: `founder`

Semantic-Lane-ID: `loom-hostd-exec-cell-attachment-20260830`

Write-Set: Garden, Sounio action 9035, catalog executable, freeze gates, and
evidence.

Read-Set: frozen actions 9030, 9031, 9033, and 9034; provider lifecycle host
receipt; source-built Sounio compiler resolution.

Integration-Target: OCaml catalog projection followed by root-owned host
payload selection and a fresh DynamicUser ExecCell.

Authoritative-Only-If: frozen Sounio action 9035 refuses the unchanged
template-mismatch witness with `DENY567`, while a mutant deleting only the
operation-template comparison admits it.

Semantic-Boundary: a provider command can request a catalog entry but cannot
create one. Catalog entry, argument grammar, result schema, sandbox profile,
and semantic event identity all originate in Sounio.
