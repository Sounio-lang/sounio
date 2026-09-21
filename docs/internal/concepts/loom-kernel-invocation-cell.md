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

Status: executable and semantics-frozen in Sounio action `9029`; resident Sounio
v3 transport and the OCaml operational kernel are frozen. Material invocation
and product attachment remain closed.

Canonical artifacts:
`tools/loom/GARDEN_KERNEL_INVOCATION_CELL_V1.md`,
`stdlib/coordination/loom_kernel_invocation_cell_authority.sio`, and
`tools/loom/kernel_invocation_cell_authority.freeze.v1`.

Operational artifacts:
`tools/loom/resident_membrane.runtime.v3`,
`tools/loom/src/loom_invocation_cell.ml`, and
`tools/loom/kernel_invocation_cell.runtime.v1`.

Material-parity admission artifacts:
`tools/loom/INVOCATION_CELL_MATERIAL_ADMISSION_V1.md`,
`tools/loom/src/loom_kernel_principal_broker.cpp`, and
`tools/loom/kernel_invocation_cell.material.v1`.

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

Action `9029` defines and tests the expected decision for the complete join,
including four positive operations, named refusals, and ten causal single-rule
sabotages. That supports only the frozen semantic boundary.

The OCaml operational kernel validates those exact frozen artifacts before it
spawns one resident Sounio v3 process. It then enforces the lifecycle
`UNPREPARED -> PREPARED -> EFFECT_STOPPED -> CLOSED | POISONED`, monotonic
resident correlation, deadlines, receipt binding, and terminal invalidation.
Replay, an operation/state mismatch, timeout, EOF, or typed abort makes the
generation permanently unusable. It contains no semantic expected-result table:
the freeze gate rejects Sounio decision strings or the named `481`/`488` results
inside the OCaml module.

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

The frozen C++ material-admission adapter now proves that one broker binary
transports both positive and current-material frames to the exact Sounio action
`9029` authority without encoding `DENY481`. The immutable host bundle now pins
that manifest and executable and exposes a root-controller-only, decision-only
`ADMIT` request. Offline and live paths emit the same non-authorizing receipt;
`LAUNCH` remains closed and the journal is not mutated.

`broker-custodied-one-shot-material-realization` therefore remains pending
after admission. It must prove pre-effect stopping, exact peer and ancestry,
race-resistant object identity, atomic grant consumption, kill-tree timeout,
crash poisoning, and complete receipt closure on a host with kernel-distinct
lane principals.
