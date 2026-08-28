<!-- docs:meta
topic_id: repo.docs.internal.concepts.loom-kernel-invocation-cell
authority: repo_only
audience: users
last_validated: 2026-08-28
validated_by: founder-direction
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.loom-kernel-invocation-cell
-->

# LOOM Kernel InvocationCell

Concept-ID: `SOUNIO-LOOM-KERNEL-INVOCATION-CELL`

Status: preregistered hypothesis for Sounio action `9029`.

Canonical preregistration:
`tools/loom/GARDEN_KERNEL_INVOCATION_CELL_V1.md`

## Meaning

An `InvocationCell` is a non-bearer observation that joins one frozen
`PrincipalCapsule` identity lineage with one frozen subprocess effect-closure
lineage. Neither parent is substitutable for the other. The cell binds exactly
one principal generation, membrane generation, event sequence, operation,
command, worktree, claim scope, deadline, and outcome lineage.

The cell digest carries no execution authority. The root broker retains the
pidfd, barrier, and single-use grant. Material one-shot behavior is therefore a
reference-monitor obligation, not a static linearity property of Sounio values.

## Preserved Distinctions

```text
PrincipalCapsule       != ExecGrant
Principal identity     != Effect closure
Cell digest            != Bearer authority
Observed one-shot      != Static affine typing
Root exit              != Tree quiescence
Timeout termination    != Successful outcome
Review receipt         != Semantic authority
Path string            != Kernel object identity
Parent ALLOW           != Joined invocation ALLOW
Frozen synthetic ALLOW != Material host attachment
```

## Supported Claim

After action `9029` is executable and frozen, Sounio may define and test the
expected decision for the complete join, including named refusals and causal
single-rule sabotages. That supports only the frozen semantic boundary.

## Forbidden Claims

- A cell, capsule, token, PID, pidfd number, ancestry, or cgroup grants authority
  by possession.
- Current Sounio values enforce affine or linear consumption.
- A semantic fixture proves Linux mediation, hostile peer isolation, crash
  revocation, or attachment to Exec/Bash, commit, or CI.
- An OCaml, C++, shell, hook, or LLM result may create expected decisions.
- This project-local construction establishes external novelty or priority.

## Falsifier

The semantic claim is falsified if any unchanged unsafe witness remains refused
after removing its single intended rule, if either parent can be omitted without
refusal, if a copied digest authorizes a material action, or if a non-Sounio
producer supplies an expected result accepted by the gate.

## Pending Interface

`broker-custodied-one-shot-material-realization` remains pending. It must prove
pre-effect stopping, exact peer and ancestry, race-resistant object identity,
atomic grant consumption, kill-tree timeout, crash poisoning, and complete
receipt closure on a host with kernel-distinct lane principals.
