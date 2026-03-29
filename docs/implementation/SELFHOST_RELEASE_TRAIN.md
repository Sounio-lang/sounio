<!-- docs:meta
topic_id: repo.docs.implementation.selfhost-release-train
authority: repo_only
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.implementation.selfhost-release-train
-->

# Selfhost Release Train

## Purpose

This file is the exact promotion checklist for refreshing the checked-in self-hosted artifact baseline.

Use this flow when promoting:

- `artifacts/self-hosted/souc-self-hosted-x86_64`
- `artifacts/self-hosted/souc-self-hosted-x86_64.provenance.json`
- `scripts/selfhost/selfhost_promotion_policy.v1.json`

## Operator entrypoint

Canonical promotion command:

```bash
bash scripts/selfhost/update_selfhost_artifact.sh
```

Canonical release-candidate command:

```bash
bash scripts/selfhost/selfhost_release_candidate_gate.sh
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
2. Run the release drift gate:

   ```bash
   bash scripts/selfhost/selfhost_release_drift_gate.sh
   ```

3. Run the promotion policy gate:

   ```bash
   bash scripts/selfhost/selfhost_promotion_policy_gate.sh
   ```

4. Run the authority gate:

   ```bash
   bash scripts/selfhost/selfhost_authority_gate.sh
   ```

5. Run the focused ABI/parity gate:

   ```bash
   bash scripts/selfhost/selfhost_ci_abi_parity_gate.sh
   ```

6. Refresh the artifact and provenance:

   ```bash
   bash scripts/selfhost/update_selfhost_artifact.sh
   ```

7. Inspect the provenance file:

   - [`artifacts/self-hosted/souc-self-hosted-x86_64.provenance.json`](../../artifacts/self-hosted/souc-self-hosted-x86_64.provenance.json)

8. Run the provenance verification gate:

   ```bash
   bash scripts/selfhost/selfhost_artifact_provenance_gate.sh
   ```

9. Open a PR with both artifact and provenance updates.
10. Wait for the required CI checks listed in [`scripts/selfhost/selfhost_required_checks.v1.json`](../../scripts/selfhost/selfhost_required_checks.v1.json).
11. Merge only after required checks pass and the required review policy is satisfied.

Required checks for merge:

- `Contracts`
- `Selfhost Authority`
- `Selfhost ABI/Parity`
- `Sounio Lint`
- `Website`

## Artifact provenance policy

The checked-in selfhost artifact may be refreshed only by the promotion entrypoint above.

Mandatory provenance fields:

- artifact sha256
- artifact size
- source path and source sha256
- source-producing git commit
- promotion policy manifest and hash
- required-check manifest and hash
- fixed-point `gen2` and `gen3` md5 values
- gate set executed during promotion
- promotion timestamp
- bootstrap artifact sha256 before update

Repository-local provenance verification is the active policy path for this repo. CI verifies provenance self-consistency; it does not auto-promote the artifact.

## Release candidate path

Use the release-candidate gate before opening a promotion PR:

```bash
bash scripts/selfhost/selfhost_release_candidate_gate.sh
```

This gate is expected to fail if:

- required-check drift exists
- promotion policy is incomplete
- authority/parity gates regress
- checked artifact provenance is stale relative to the current source tree

## Rollback and recovery

If promotion fails:

1. stop in the isolated worktree
2. do not force-fix the checked artifact manually
3. discard the worktree or revert the promotion branch locally
4. rerun `scripts/selfhost/selfhost_authority_gate.sh` on the last accepted baseline
5. if an inconsistent artifact/provenance pair was committed, revert both files together in one commit

## Merge preconditions for artifact refresh PRs

- `Selfhost Authority` passes
- `Selfhost ABI/Parity` passes
- no unresolved parity regressions relative to the accepted artifact baseline
- provenance was refreshed in the same PR as the artifact update
- unsupported target behavior remains fenced or expected-fail, not silently successful
