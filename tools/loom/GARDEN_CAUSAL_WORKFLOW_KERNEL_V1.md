# Garden: LOOM Causal Workflow Kernel V1

Status: `GARDEN_PREREGISTERED`

## Preserved phrase

> The workflow survives because authority is a causal receipt chain, not a
> terminal session.

## Question

Can LOOM execute the typed sequence `COMPILE -> RUN_EXACT -> ATTEST` through
fresh principals, lose the controller or its Pod after compilation, and resume
without recompiling or launching `RUN_EXACT` twice?

## Authority order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> OCAML_DURABLE_JOURNAL
-> HOST_DYNAMIC_USER_ATTACHMENT
-> CRASH_RECOVERY_MEASURED
-> PARITY_OPEN
-> CLAIM_READY
```

Sounio is `SEMANTIC_AUTHORITY`. Action 9037 defines the exact three-node graph,
the state transition relation, the expected result of the canonical source,
the receipt-chain grammar, and the conditions under which recovery may
continue. OCaml may replay a frozen journal. C++20, Linux, and systemd may hold
the one material run-ticket record and measure fresh DynamicUser cells. The
ticket is a non-bearer idempotency identity bound one-to-one to the action-9030
grant identity and generation; action 9030 remains the only launch authority.
No operational or material layer may add a node, replace a handle with a
pathname, invent an expected result, or issue a second run-ticket record or
execution grant.

## Canonical experiment

The source is `tests/verify-ir/call_b.sio`. Its frozen workflow is:

1. `COMPILE` consumes the source identity and produces one immutable,
   non-bearer artifact handle.
2. The controller or its Pod is killed after the compile receipt is durable.
3. A new controller verifies the receipt chain and asks the surviving
   HostGuardian about the existing workflow generation.
4. `RUN_EXACT` resolves the artifact handle as identity under HostGuardian
   custody and consumes one distinct HostGuardian-custodied run-ticket record.
   The record is bound one-to-one to the action-9030 execution grant that alone
   authorizes launch. It expects exit code zero and empty stdout and stderr.
5. `ATTEST` binds source, frozen semantics, compiler, artifact, observation,
   sandbox, principals, toolchain, and hardware into one immutable result
   handle.

Each material stage uses a fresh DynamicUser. `COMPILE` has one bounded write
effect: it creates exactly one immutable, Guardian-custodied artifact record.
`RUN_EXACT` is the second bounded material effect. A handle identifies a sealed
record in HostGuardian custody; it is not a bearer capability and cannot be
converted into a caller-selected path.

## Exactly-once boundary

Exactly-once is defined relative to a live HostGuardian generation. The
HostGuardian durably commits the run-ticket record before launch, binds it to
one action-9030 grant identity and generation, owns the ExecCell pidfd, and
seals the result before exposing completion. A restarted controller may observe
or continue the same ticket and grant but cannot mint either one again.

Loss of the HostGuardian, host, or durable receipt store is fail-closed and is
not recoverable in V1. This prevents the design from claiming an impossible
exactly-once guarantee across loss of the authority that knows whether the
effect occurred.

## Falsifier

The load-bearing rule is transition-specific receipt continuity. An unchanged
recovery witness that keeps every positive fact but replaces the predecessor
receipt with a different digest must be refused with `DENY589`. A Sounio mutant
deleting only that equality must admit the same witness. If another rule still
refuses it, receipt continuity was not shown to be causal.

## Nonclaims

- `ocaml_durable_journal_attached=false`
- `dynamic_user_workflow_attached=false`
- `controller_loss_recovery_measured=false`
- `pod_loss_recovery_measured=false`
- `hostguardian_loss_recovery=false`
- `exactly_once_materially_measured=false`
- `arbitrary_shell=false`
- `production_activation=false`
- `parity_open=false`
- `claim_ready=false`
