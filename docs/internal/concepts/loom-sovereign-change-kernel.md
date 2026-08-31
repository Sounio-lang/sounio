<!-- docs:meta
topic_id: repo.docs.internal.concepts.loom-sovereign-change-kernel
authority: repo_only
audience: users
last_validated: 2026-08-31
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.loom-sovereign-change-kernel
-->

# LOOM Sovereign Change Kernel

Concept-ID: `SOUNIO-LOOM-SOVEREIGN-CHANGE-KERNEL`

Status: `executable`

Semantic-Lane-ID: `loom-sovereign-change-kernel-20260831`

Owner: `codex-1`

Concept-IDs: `SOUNIO-LOOM-SOVEREIGN-CHANGE-KERNEL`

Intent-Preserved: one exact structured mutation is authorized by Sounio
before execution, held as non-bearer kernel memory, consumed once against the
exact post-image, and admitted to Git and CI without policy reinterpretation.

Transformation: founder-authorized mutation authority becomes executable
action 9043 with ordered prepare, consume, commit, CI, and production decisions.

Types-Changed: none

Effects-Changed: none

IR-Changed: none

Claims-Introduced: action 9043 is an executable semantic authority for the
bounded ChangeIntent decision protocol and its enumerated refusal controls.

Claims-Forbidden: production mutation authority, arbitrary shell-write
authority, commit admission, CI admission, or `claim_ready=true` before the
frozen action-9043 executable and installed adversarial product gate exist.

Assumptions: action 9042 remains frozen and production-active; material peer,
filesystem, Git, and CI observations remain facts rather than semantic oracles.

Read-Set: action-9042 freeze and product manifests, structured hook event,
authenticated peer state, Git `HEAD` and index, canonical worktree pre-image,
post-image, commit, and receipt.

Positive-Witness: `scripts/ci/sounio_loom_sovereign_change_kernel_selftest.sh`

Negative-Witness: action-9043 refusals `DENY613` through `DENY628`, including
the isolated CI no-reinterpretation sabotage control in the positive witness.

Acceptance-Gate: `scripts/ci/sounio_loom_sovereign_change_kernel_selftest.sh`

Integration-Target: frozen action-9043 authority followed by the existing
OCaml Loom kernel, structured mutation hooks, Git hook, and CI admission job.

## Authority

Sounio action 9043 is the semantic authority for mutation, commit, and CI
admission. It extends the frozen action-9042 peer and execution boundary; it
does not redefine that boundary. OCaml may realize the state machine and Linux
and Git may supply material facts, but neither may manufacture an expected
decision.

## Contract

A ChangeIntent binds the exact structured mutation, patch bytes, canonical file
set, authenticated peer, worktree identity, `HEAD`, index, and pre-change
worktree state. Its ChangeGrant exists only in Loom-kernel memory, is never
exported, and is consumed atomically at most once after the exact post-image is
observed.

A commit is admissible only when its index and prospective tree equal the
consumed post-image and its parent and message are bound into the receipt. CI
verifies that receipt against the exact commit and frozen action-9043 manifest.
CI cannot rerun policy over raw facts, promote a parity result, or reinterpret a
Sounio refusal.

## Ordered Decisions

- `CHANGE_PREPARED`
- `CHANGE_CONSUMED`
- `COMMIT_ADMIT`
- `CI_ADMIT`
- `PRODUCTION_GATE_READY`

Each decision proves only its own completed prefix. Later facts in an earlier
mode are a refusal, not evidence of extra readiness.

## Semantic Boundary

Action 9043 owns the join from structured mutation through Git and CI. It does
not grant arbitrary shell-write authority, authorize paths outside the claimed
worktree, or make a GUI, TUI, CLI, provider, transport, Git hook, or CI runner a
semantic authority.

Write-Set: Garden, Sounio action 9043, frozen executable, OCaml ChangeGrant
kernel, structured provider hooks, Git admission hook, CI verifier, manifests,
and receipts.

Authoritative-Only-If: the Sounio executable precedes operational attachment,
the exact frozen hashes remain bound end-to-end, and the isolated
CI-reinterpretation mutant admits the unchanged refused witness.
