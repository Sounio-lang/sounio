<!-- docs:meta
topic_id: repo.docs.implementation.selfhost-release-train
authority: historical
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.implementation.selfhost-release-train
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Selfhost Release Train

## Purpose

This file is the exact promotion checklist for refreshing the checked-in self-hosted artifact baseline.

Use this flow when promoting:

- `artifacts/self-hosted/souc-self-hosted-x86_64`
- `artifacts/self-hosted/souc-self-hosted-x86_64.provenance.json`

## Operator entrypoint

Canonical promotion command:

```bash
bash scripts/selfhost/update_selfhost_artifact.sh
```

This script:

1. validates the bootstrap artifact with the authority gate
2. rebuilds the current source into a promoted artifact
3. validates the promoted artifact again
4. refreshes provenance for the promoted artifact

## Promotion preconditions

- start from a fresh isolated worktree
- keep the primary dirty worktree untouched
- do not promote from a dirty tree
- do not promote while fixed-point is broken
- do not promote while current head has unresolved parity regressions versus the accepted artifact baseline

## Canonical promotion sequence

1. Create an isolated worktree from the intended integration branch.
2. Run the authority gate:

   ```bash
   bash scripts/selfhost/selfhost_authority_gate.sh
   ```

3. Run the focused ABI/parity gate:

   ```bash
   bash scripts/selfhost/selfhost_ci_abi_parity_gate.sh
   ```

4. Refresh the artifact and provenance:

   ```bash
   bash scripts/selfhost/update_selfhost_artifact.sh
   ```

5. Inspect the provenance file:

   - [`artifacts/self-hosted/souc-self-hosted-x86_64.provenance.json`](../../artifacts/self-hosted/souc-self-hosted-x86_64.provenance.json)

6. Open a PR with both artifact and provenance updates.
7. Wait for the required CI checks listed in [`scripts/selfhost/selfhost_required_checks.v1.json`](../../scripts/selfhost/selfhost_required_checks.v1.json).
8. Merge only after required checks pass and the required review policy is satisfied.

## Artifact provenance policy

The checked-in selfhost artifact may be refreshed only by the promotion entrypoint above.

Mandatory provenance fields:

- artifact sha256
- artifact size
- source path and source sha256
- source-producing git commit
- fixed-point `gen2` and `gen3` md5 values
- gate set executed during promotion
- promotion timestamp
- bootstrap artifact sha256 before update

Repository-local provenance verification is the active policy path for this repo. CI verifies provenance self-consistency; it does not auto-promote the artifact.

## Merge preconditions for artifact refresh PRs

- `Selfhost Authority` passes
- `Selfhost ABI/Parity` passes
- no unresolved parity regressions relative to the accepted artifact baseline
- provenance was refreshed in the same PR as the artifact update
- unsupported target behavior remains fenced or expected-fail, not silently successful
