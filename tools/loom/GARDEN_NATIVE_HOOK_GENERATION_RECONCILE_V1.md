# Garden: Causal Hook-Generation Reconciliation

Concept-ID: `SOUNIO-LOOM-NATIVE-HOOK-CUTOVER`

Status: `GARDEN`

## Butterfly

> A stale heartbeat is not a dead process, and a missing process is not yet a
> safe filesystem deletion.

Action 9046 refuses the native-hook cutover while any provider generation is
unknown or unresponsive. That refusal is necessary, but it exposes a second
semantic object: a durable presence record whose process generation has ended.
Ordinary expiry cannot distinguish a slow live provider from a dead generation,
and deleting an inconvenient record would manufacture the zero required by the
cutover gate.

The missing operation is **causal generation reconciliation**. It moves an old
presence generation and its exact related artifacts into an append-only
quarantine only after the kernel observation proves that the recorded process
generation cannot be the current process generation.

## Hypothesis

A provider presence is eligible for quarantine only when all three affirmative
claims hold:

1. **record identity**: the regular, readable record name, agent, lane, session,
   generation, PID, boot identifier, PID namespace, and process start tick are
   bound to one digest;
2. **kernel identity**: the observer has fresh boot, PID-namespace, PID, and
   process-start observations under the coordination state lock; and
3. **causal absence**: at least one exclusive cause proves that the recorded
   generation ended: boot changed, PID namespace changed, PID is absent, or the
   PID now has a different start tick.

This is the **causal absence triple**. A heartbeat timeout with the recorded PID
still present and the same start tick is not causal absence. An unreadable or
malformed record is not causal absence. Both remain blocking evidence.

## Candidate State Machine

```text
OBSERVED
-> IDENTITY_BOUND
-> KERNEL_BOUND
-> CAUSAL_ABSENCE_PROVEN
-> QUARANTINE_PREPARED
-> ARTIFACTS_MOVED
-> SIGNED_COMMIT
```

A crash after `QUARANTINE_PREPARED` leaves a recoverable write-ahead record.
Recovery may complete only the already authorized moves, with the same source
digests and destination. It may not recalculate eligibility under a weaker
observation.

## First Executable

Sounio action 9047 consumes a bounded observation containing:

- parent action 9046 freeze binding;
- plan or apply mode;
- readable regular-record and filename/identity bindings;
- inventory digest and fresh state-lock observation;
- current boot and PID-namespace bindings;
- recorded PID and start-tick bindings;
- the four mutually exclusive causal-absence observations;
- exact related-artifact digest coverage;
- new quarantine destination and write-ahead receipt readiness;
- fail-closed and forbidden-oracle facts; and
- causal sabotage coverage.

It emits `KEEP` for an identity-bound generation that is still live. It emits
`QUARANTINE_READY` only in apply mode when the causal absence triple and the
transactional receipt conditions hold. Missing or contradictory evidence
refuses.

## Sabotage Controls

1. Remove only the causal-absence fact from an otherwise eligible record.
2. Change the PID start tick after the plan but before the state lock commits.
3. Replace the presence record while preserving its filename.
4. Expire the heartbeat while retaining the same live PID generation.
5. Make the record unreadable or malformed.
6. Pre-create or redirect the quarantine destination.
7. Omit one related artifact from the prepared receipt.
8. Crash after preparation and attempt recovery with a different source hash.

Each sabotage must refuse because of action 9047 or its exact transaction
binding. Python, Rust, shell arithmetic, an LLM opinion, and a timeout alone
cannot establish absence.

## Evidence Boundary

This seed does not authorize killing a live provider, deleting unreadable
state, weakening action 9046, or activating the candidate runtime. It defines
the only permitted transition from kernel-proven ended generations into
quarantine so that action 9046 can later measure affirmative absence without a
fabricated zero.
