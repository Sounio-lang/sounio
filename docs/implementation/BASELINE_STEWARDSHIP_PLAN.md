<!-- docs:meta
topic_id: repo.docs.implementation.baseline-stewardship-plan
authority: repo_only
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.implementation.baseline-stewardship-plan
-->

# Baseline Stewardship Plan

## Purpose

This file defines the post-program operating model for the self-hosted compiler now that the authority program is institutionally closed.

Use it to answer:

- how the accepted baseline is operated week to week
- which work belongs to baseline stewardship versus technical campaigns
- what quarterly milestones matter after the authority program
- which acceptance criteria keep the baseline mature

## Tracks

### 1) Baseline Stewardship Track

Scope:

- checked artifact refresh
- provenance refresh
- release-candidate execution
- release drift checks
- required-check health
- rollback if a promotion or refresh is inconsistent

Cadence:

- weekly: run authority, ABI/parity, drift, and provenance checks
- per promotion: run the full release-candidate path before opening the PR
- monthly: review debt register, target taxonomy, and required-check stability

Canonical entrypoints:

- [`scripts/selfhost/selfhost_authority_gate.sh`](../../scripts/selfhost/selfhost_authority_gate.sh)
- [`scripts/selfhost/selfhost_ci_abi_parity_gate.sh`](../../scripts/selfhost/selfhost_ci_abi_parity_gate.sh)
- [`scripts/selfhost/selfhost_release_candidate_gate.sh`](../../scripts/selfhost/selfhost_release_candidate_gate.sh)
- [`scripts/selfhost/selfhost_release_drift_gate.sh`](../../scripts/selfhost/selfhost_release_drift_gate.sh)
- [`scripts/selfhost/selfhost_artifact_provenance_gate.sh`](../../scripts/selfhost/selfhost_artifact_provenance_gate.sh)
- [`scripts/selfhost/selfhost_dual_trust_gate.sh`](../../scripts/selfhost/selfhost_dual_trust_gate.sh)
- [`scripts/selfhost/update_selfhost_artifact.sh`](../../scripts/selfhost/update_selfhost_artifact.sh)

Acceptance criteria:

- fixed-point stays green
- source↔artifact parity stays green
- provenance matches artifact bytes and source commit
- dual trust passes
- required check names remain stable and unambiguous
- no promotion happens outside the documented release train

### 2) Targeted Technical Campaigns

Scope:

- AArch64 runtime-surface expansion
- elimination or proof of reachability for fenced missing-SRET sites
- cleanup of inherited baseline noise when payoff is clear
- focused runtime or ABI debt retirement that does not weaken the authority model

Campaign rules:

- keep each campaign mergeable on its own
- promote support classes only with runtime evidence
- keep unsupported behavior fenced or expected-fail until proven
- update manifests, docs, and tests in the same change that updates support claims

Acceptance criteria:

- the unsupported or fenced surface is measurably reduced, or more sharply bounded
- Wave 1-12 authority, parity, provenance, and release gates remain green
- no new unresolved failures are introduced relative to the accepted artifact baseline

## Quarterly Milestones

### Q1 — Stewardship Stability

- zero recurring release-drift surprises
- zero manual artifact promotions outside the release train
- release-candidate path used as routine pre-PR validation

### Q2 — Runtime Debt Reduction

- close at least one residual missing-SRET debt item or prove it unreachable in supported paths
- expand `aarch64-runtime-supported` where runtime evidence is straightforward

### Q3 — Cross-Target Consolidation

- reclassify compile-proof-only surfaces only where runtime evidence now exists
- keep unsupported or partial target claims explicit and machine-validated

### Q4 — Annual Baseline Review

- review artifact, provenance, trust planes, debt register, and taxonomy as one package
- confirm the protected-branch model and required checks still match repo reality

## Maintainer Checklist

When operating the baseline:

1. start from an isolated worktree
2. run the release-candidate gate
3. refresh artifact and provenance only through the canonical promotion entrypoint
4. verify required checks on the latest PR SHA
5. merge only when authority, parity, provenance, and dual trust all remain green

When running a technical campaign:

1. define the exact runtime or ABI surface being changed
2. add or update focused regression coverage first or together with the fix
3. run fixed-point plus all affected selfhost gates
4. update taxonomy and docs only after evidence is green

## Mature Baseline Criteria

The self-hosted compiler baseline remains institutionally mature only while all are true:

- the accepted artifact is promoted under explicit policy
- the checked artifact is not a parallel truth separate from source
- AArch64 runtime-supported claims are backed by the runtime gate
- deferred debt is explicit, bounded, and non-systemic
- rollback remains short, documented, and reversible

## Related Docs

- PR-ready summary: [BASELINE_STEWARDSHIP_PR_NOTE.md](BASELINE_STEWARDSHIP_PR_NOTE.md)
- maintainer handoff: [BASELINE_STEWARDSHIP_MAINTAINER_MEMO.md](BASELINE_STEWARDSHIP_MAINTAINER_MEMO.md)
- authority overview: [SELFHOST_AUTHORITY_MODEL.md](SELFHOST_AUTHORITY_MODEL.md)
- release train: [SELFHOST_RELEASE_TRAIN.md](SELFHOST_RELEASE_TRAIN.md)
- deferred debt register: [SELFHOST_DEBT_REGISTER.md](SELFHOST_DEBT_REGISTER.md)
