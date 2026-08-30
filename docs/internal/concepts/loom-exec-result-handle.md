<!-- docs:meta
topic_id: repo.docs.internal.concepts.loom-exec-result-handle
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.loom-exec-result-handle
-->

# Loom Exec Result Handle

Status: executable in Sounio action `9033`; semantic freeze pending.

Concept-ID: `SOUNIO-LOOM-EXEC-RESULT-HANDLE`

## Founder Intent

The provider CLI must not execute an approved command after the native hook
returns. A fresh host `ExecCell` executes it, while the provider receives a
stable result handle whose meaning survives transport, lane, and supervisor
loss. The handle names evidence. It never becomes execution authority.

The preregistered Garden is
`tools/loom/GARDEN_PRODUCT_DYNAMIC_USER_EXEC_ATTACHMENT_V1.md`. Action `9033`
is the first executable definition of the result-handle boundary required by
that Garden. It composes the already-frozen action-`9030` grant close and
action-`9031` capsule extinction decisions; it does not replace either parent.

## Semantic Lane

```text
Semantic-Lane-ID: loom-hostd-exec-cell-attachment-20260830
Owner: codex-1
Concept-IDs: SOUNIO-LOOM-EXEC-RESULT-HANDLE, SOUNIO-LOOM-KERNEL-EXEC-GRANT-CELL, SOUNIO-LOOM-KERNEL-PEER-ACTIVATION-CAPSULE
Intent-Preserved: provider execution is replaced by a fresh host ExecCell and only a Sounio-bound result may cross back
Transformation: define publish and resolve judgments for an immutable event-bound ExecCell result handle
Types-Changed: adds LoomExecResultHandle semantic records only
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a handle may name one terminal result after affirmative execution and authority extinction
Claims-Forbidden: handle grants execution; handle proves material execution by itself; handle permits replay; same-UID local execution is attached; production activation is open
Assumptions: frozen actions 9030 and 9031 remain byte-identical parents; the first fixture is the preregistered product calibration command
Write-Set: this contract, registry row, action 9033 source, executable adapter, build and freeze gates, evidence
Read-Set: product DynamicUser Garden, frozen actions 9030 and 9031, frozen product ExecCell fixture
Positive-Witness: exact fixture publishes and resolves one canonical handle
Negative-Witness: the same frame with only the command digest changed is DENY534
Acceptance-Gate: scripts/ci/sounio_loom_exec_result_handle_freeze_selftest.sh
Integration-Target: native OCaml result store and real hostd ExecCell attachment
Authoritative-Only-If: frozen Sounio action 9033 refuses the command-digest sabotage and the isolated rule-deletion mutant admits that unchanged witness
```

## Operations

Action `9033` accepts exactly two operations:

1. `PUBLISH`: validate before mutation that the exact event, command, grant
   generation, and result receipt are bound to frozen parents; require a
   terminal outcome, affirmative ExecCell and authority extinction, an
   immutable content-addressed receipt, and an absent destination ready for
   atomic commit.
2. `RESOLVE`: validate the same lineage against an already committed record and
   expose it through a read-only lookup. Resolution cannot issue, consume,
   execute, or replay anything.

For V1 the canonical handle is:

```text
loom-result-v1:113b0f7d2a1f7adc5b68a92ef37be3689accfe4bbb00cb4e8c15fc3ab7b70013:1:5805e6579b6420ba0dd693d385715943955d0e69e657f44e94e23d20a20d27d1
```

Its components are the product event SHA-256, action-`9030` generation, and
terminal result-receipt SHA-256. The string is a public lookup identity, not a
secret or bearer token.

## Absence And Failure

Publication is legal only after affirmative absence is joined from all of:

- the ExecCell pid and descendants are extinct;
- its cgroup is empty and transient unit inactive;
- the action-`9030` grant and action-`9031` capsule are extinct.

Silence, timeout, policy absence, missing hashes, partial lineage, mutable
storage, a pre-existing publish target, or an authority-bearing reader all fail
closed. Python, Rust, review-only LLM output, and parity receipts cannot produce
or confirm the expected result.

## Nonclaims

At action-`9033` freeze, `material_execution=false`, `exec_cell_attached=false`,
`result_store_attached=false`, `provider_hook_switched=false`,
`production_activation=false`, `parity_open=false`, and `claim_ready=false`.
