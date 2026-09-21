<!-- docs:meta
topic_id: repo.docs.internal.concepts.loom-exec-intent-envelope
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.loom-exec-intent-envelope
-->

# SOUNIO-LOOM-EXEC-INTENT-ENVELOPE

Concept-ID: `SOUNIO-LOOM-EXEC-INTENT-ENVELOPE`

Kind: `executable`

Owner: `founder`

Semantic-Lane-ID: `loom-hostd-exec-cell-attachment-20260830`

Write-Set: Garden, Sounio action 9034, packed executable adapter, freeze gates,
and evidence.

Read-Set: frozen actions 9031 and 9033, native provider-hook event shape, and
the product DynamicUser ExecCell Garden.

Integration-Target: native OCaml intent projection followed by hostd provider
lifecycle attachment.

Authoritative-Only-If: frozen Sounio action 9034 refuses the exact
wrong-command witness and a mutant deleting only the command-binding rule
admits that unchanged witness.

Semantic-Boundary: raw provider JSON remains evidence; only the Sounio-defined
intent projection is the event identity consumed by action 9033.
