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

Canonical AArch64 runtime validation command:

```bash
bash scripts/selfhost/selfhost_aarch64_runtime_gate.sh
```

Canonical dual trust command:

```bash
bash scripts/selfhost/selfhost_dual_trust_gate.sh
```

Canonical reproducible bootstrap command:

```bash
bash scripts/selfhost/selfhost_reproducible_bootstrap_gate.sh
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

6. Use the canonical AArch64 runtime gate directly when investigating or promoting AArch64 support:

   ```bash
   bash scripts/selfhost/selfhost_aarch64_runtime_gate.sh
   ```

   The authority gate and focused ABI/parity gate already run this surface as part of their blocking validation.

7. Refresh the artifact and provenance:

   ```bash
   bash scripts/selfhost/update_selfhost_artifact.sh
   ```

8. Inspect the provenance file:

   - [`artifacts/self-hosted/souc-self-hosted-x86_64.provenance.json`](../../artifacts/self-hosted/souc-self-hosted-x86_64.provenance.json)

9. Run the provenance verification gate:

   ```bash
   bash scripts/selfhost/selfhost_artifact_provenance_gate.sh
   ```

10. Run the dual trust gate:

   ```bash
   bash scripts/selfhost/selfhost_dual_trust_gate.sh
   ```

11. Open a PR with both artifact and provenance updates.
12. Wait for the required CI checks listed in [`scripts/selfhost/selfhost_required_checks.v1.json`](../../scripts/selfhost/selfhost_required_checks.v1.json).
13. Merge only after required checks pass and the required review policy is satisfied.

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
- AArch64 runtime validation status for any runtime-supported target claims
- runtime validation evidence for any promoted AArch64 closure-arity expansion
- promotion timestamp
- bootstrap artifact sha256 before update

Repository-local provenance verification is the active policy path for this repo. CI verifies provenance self-consistency; it does not auto-promote the artifact.

Supplemental trust plane:

- repo-local reproducible bootstrap must show `artifact == gen1 == gen2 == gen3`
- dual trust passes only when both provenance and reproducible bootstrap pass on the refreshed artifact

Hosted artifact attestation remains optional and unwired here; the release train must stay truthful on repos where hosted attestation support is unavailable or inappropriate.

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
- the refreshed artifact does not satisfy the dual repo-local trust plane

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

## Release closure criteria

Declare the self-hosted compiler baseline institutionally closed only when all are true:

- required checks are stable and enforced on `main`
- source↔artifact parity is green against the accepted baseline
- fixed-point holds on the promoted artifact
- provenance is refreshed in the same PR as the artifact
- dual trust passes on the checked artifact
- remaining debt is explicitly fenced, named, and non-threatening to supported paths
