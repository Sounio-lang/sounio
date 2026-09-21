# Garden: Loom Host Boot Reconciler v1

Status: `GARDEN`

## Question

Can Loom become a host service that survives presentation-Pod loss and starts
at boot without silently turning process loss into a claim that an old PTY was
recovered?

The reconciler is not a new terminal runtime. It observes the existing Loom
Guardian/kernel state and asks a frozen Sounio program for one of five
decisions:

1. `NOOP_ACTIVE`: the verified Guardian, harness, kernel, and journals are live;
2. `RECOVER_SAME_PHYSICAL`: the verified Guardian and harness remain, but the
   disposable kernel is absent;
3. `HOLD_LINEAGE_REQUIRED`: the Guardian is absent, so the old PTY is gone;
4. `HOLD_DISABLED`: the host service is not explicitly enabled;
5. `HOLD_UNENROLLED`: no desired lane exists.

The first version never opens a successor generation automatically. It makes
boot reconciliation durable while keeping lineage resurrection closed until a
separate proof-carrying transition is materialized.

## Language Authority

- Sounio: `SEMANTIC_AUTHORITY`, action `9041`.
- OCaml: `EFFECT_PARITY`, host observation, and reconciliation loop.
- systemd and shell: installation and transport only.
- External LLMs: `REVIEW_ONLY`.
- Python and Rust: prohibited from the authority and operational critical path.

The mandatory order is:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY`

## Fail-Closed Boundary

Every decision requires a present policy, frozen semantics and runtime hashes,
a bound desired-state catalog, current boot observation, a bound state root,
material observation, and the preregistered sabotage count. Missing policy,
timeout, malformed output, or executable hash drift is a refusal.

`RECOVER_SAME_PHYSICAL` additionally requires the Guardian PID, start tick,
instance, harness PID, harness start tick, command, boot, journals, and output
prefix to match the durable descriptor. A PID without its start tick is not an
identity.

When the Guardian is absent, a live kernel is contradictory and must be
refused. Otherwise the only v1 result is `HOLD_LINEAGE_REQUIRED`, with an
affirmative assertion that no same-PTY claim is being made.

## Preregistered Sabotage

The load-bearing rule is `guardian_start_verified == 1`. The positive recovery
frame is copied exactly except that this field is zero. The shipped Sounio
program must return `DENY545`. An isolated Sounio mutant that changes only that
literal must return `RECOVER_SAME_PHYSICAL` for the unchanged frame.

## First Operational Canary

The OCaml alpha will run against a temporary desired-state catalog:

1. launch the existing durable-lane canary under Loom;
2. observe `NOOP_ACTIVE`;
3. kill only the kernel and observe `RECOVER_SAME_PHYSICAL`;
4. apply the decision and verify the same Guardian/harness identities;
5. submit the start-tick sabotage and observe `DENY545` with no recovery;
6. stop the Guardian and observe `HOLD_LINEAGE_REQUIRED` with no new process;
7. restart the reconciler and verify its receipt chain continues.

The systemd installer is staged and disabled by default. Enabling it is an
explicit operator action, never a side effect of installation.
