# LOOM Resident Sounio Authority V1

> Status: Garden seed | Date: 2026-08-28 | Parent: frozen subprocess
> membrane action `9023`

## Butterfly

> Semantic authority should be a resident witness, not a process-launch tax.

## Pressure

The diagnostic subprocess membrane invokes the hash-pinned Sounio `9023`
runtime once per effect. This preserves semantic authority but adds variable
`fork`/`exec`, dynamic-loader, and process-reaping latency to every stopped
syscall. Under the namespace backstop, a shell-to-descendant route can spend
seconds waiting for decisions while the tracees remain stopped.

Moving the decision into OCaml or C would reduce latency by destroying the
language-authority contract. A resident Sounio process must instead retain the
frozen `9023` decision function while giving OCaml a bounded, correlated
transport.

## Protocol

One resident generation has exactly one owning OCaml adapter and one frozen
Sounio child process. The child receives newline-delimited requests and emits
exactly one newline-delimited response for each accepted request.

```text
START(generation, parent_9023_hash, deadline)
REQUEST(generation, sequence, previous_sequence, request_digest, deadline)
RESPONSE(generation, sequence, request_digest, result_digest)
STOP(generation, final_sequence)
```

The first request sequence is `1`. Every later sequence must be exactly one
greater than the last completed sequence. A sequence advances only after a
valid correlated response. A timeout, EOF, malformed response, wrong sequence,
wrong request digest, resident exit, or adapter loss poisons the complete
generation. Poisoned generations cannot be resumed or replaced under the same
identity.

The request payload is an exact `9023` frame. The response payload is the exact
`SOUNIO_SUBPROCESS_MEMBRANE_ALLOW` or `...DENY` line produced from that frame.
Action `9024` governs transport admission; action `9023` remains the decision
authority for the effect itself.

## Authority Invariants

1. The parent `9023` manifest, source, semantics, and runtime hashes are frozen before `9024` exists.
2. A resident generation is random, nonzero, single-owner, and never reused.
3. Request sequence is strictly monotonic and has no gaps or replay.
4. The request digest is computed before write and bound to the correlated response.
5. The response sequence and request digest must equal the outstanding request.
6. Only one request may be outstanding per resident generation.
7. Every request has a finite monotonic deadline.
8. Timeout, EOF, partial line, oversized line, malformed decision, or child exit poisons the generation.
9. A poisoned generation refuses all later requests and cannot be laundered as a transient retry.
10. The child inherits only its protocol pipes and the frozen Sounio executable.
11. Python, Rust, external LLMs, OCaml, and C cannot produce the expected decision.
12. OCaml may supervise, correlate, hash, journal, and revoke; it may not synthesize `ALLOW`.
13. Startup and shutdown are receipt-bearing state transitions, not implicit process observations.
14. Resident latency is evidence metadata, never a reason to weaken a decision or deadline.

## Action 9024

The Sounio transport authority consumes these facts:

- stage;
- event kind (`START`, `REQUEST`, `RESPONSE`, `STOP`);
- parent frozen;
- generation bound;
- current sequence and previous sequence;
- request present and request digest nonzero;
- response present and result digest nonzero;
- correlation valid;
- deadline bound;
- resident healthy;
- generation poisoned.

Expected refusals:

| Code | Meaning |
| --- | --- |
| `440` | parent `9023` is not frozen or hash-bound |
| `441` | resident generation is missing or invalid |
| `442` | sequence is replayed, skipped, or not correlated |
| `443` | request or response binding is incomplete |
| `444` | deadline, resident health, or transport state failed |
| `445` | a poisoned generation was reused |

## Causal Controls

The executable Sounio selftest must include two single-rule sabotages:

1. Remove only strict sequence progression. The unchanged replay request becomes `ALLOW`.
2. Remove only the frozen-parent binding. The unchanged request with an unfrozen `9023` parent becomes `ALLOW`.

These controls prove that resident speed did not replace causal authority with
pipe success or parser coincidence.

## Stages

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> OCAML_RESIDENT_REALIZATION
-> PERFORMANCE_GATE
-> MEMBRANE_INTEGRATION
```

`OCAML_RESIDENT_REALIZATION` cannot begin until the `9024` manifest is frozen.
`MEMBRANE_INTEGRATION` does not imply general Bash/Exec attachment; the broader
native coverage and Unix-socket gaps remain independent blockers.

## Positive Witness

One Sounio resident accepts `START`, then three exact `9023` frames with
sequences `1`, `2`, and `3`, returning the same decisions as three isolated
invocations, then accepts `STOP`. The resident process identity remains stable
throughout.

## Negative Witnesses

- repeated sequence;
- skipped sequence;
- mismatched response correlation;
- zero request digest;
- missing deadline;
- resident EOF with a request outstanding;
- resident timeout;
- malformed or oversized response;
- reuse after poison;
- parent `9023` hash mismatch.

## Nonclaims

- This is not a distributed authority or consensus protocol.
- It does not make OCaml a semantic oracle.
- It does not permit multiple concurrent outstanding decisions.
- It does not make a resident process immortal or recoverable after crash.
- It does not close filesystem Unix sockets, device effects, or all syscall families.
- It does not attach Bash/Exec, commit, or CI by itself.
