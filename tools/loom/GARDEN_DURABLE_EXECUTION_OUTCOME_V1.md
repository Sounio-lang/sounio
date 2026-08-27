# Loom Durable Execution Outcome

> **Status**: Garden seed | **Last validated**: 2026-08-27 | **Source**: execution-custody V2 attachment refusal

## Butterfly

> A capability that disappears into `execve` can prove that execution began,
> but it cannot prove how execution ended.

Loom already refuses an unmeasured command, re-runs the frozen Sounio decision
at consume time, and keeps each one-use grant in kernel memory. The remaining
gap is temporal: the broker replaces itself with the child command, so no
trusted process remains to bind exit, signal, duration, and an outcome digest
to the consumed grant.

## Core Idea

Execution is a two-phase custody protocol, not a single permission bit.

```text
EXEC_ISSUE
-> first Sounio decision
-> EXEC_CONSUME
-> second Sounio decision
-> child observed
-> outcome record durably written
-> Sounio outcome decision
-> EXEC_OUTCOME committed by the same broker generation
```

`EXEC_CONSUME` creates a pending outcome obligation. It does not create a
successful result. The broker remains alive, runs the measured executable as a
child with inherited standard streams, waits for exit or signal, writes an
immutable outcome record, asks Sounio whether that record is complete, and
commits only its digest to the Loom kernel journal.

## Authority Invariants

1. Sounio defines outcome completeness before OCaml implements it.
2. The outcome binds the source, frozen semantics, toolchain, hardware,
   command, consumed grant, kernel generation, and both pre-execution Sounio
   decisions.
3. Exit and signal are disjoint positive observations. Silence is neither.
4. Elapsed time is an observation and must be finite and non-negative.
5. The result digest is nonzero and binds the complete observational record;
   it is not an expected scientific result and grants no semantic authority.
6. The same authenticated broker process that consumed the grant must report
   its outcome. Same-UID possession of a visible handle is insufficient.
7. The kernel journals the outcome before retiring its pending obligation.
8. A duplicate, replayed, wrong-generation, or malformed outcome is refused.
9. Kernel death with a consumed but unclosed grant materializes an explicit
   `EXEC_OUTCOME_INCOMPLETE` event during recovery.
10. A missing policy, policy timeout, policy error, missing decision, missing
    receipt, or failed durable write fails closed.
11. Child stdout and stderr remain inherited. Loom does not reinterpret
    terminal bytes as the semantic result.
12. Python, Rust, shell, OCaml, and external LLMs cannot manufacture outcome
    authority. OCaml is only the operational realization of the frozen Sounio
    decision.

## Outcome Algebra

| Kind | Required observation | Forbidden observation |
| --- | --- | --- |
| `EXITED` | exit code in `0..255`; signal is zero | missing exit code or nonzero signal |
| `SIGNALED` | signal in `1..255`; exit code is zero | missing signal or nonzero exit code |
| `INCOMPLETE` | positive recovery or supervision evidence | promotion to a completed receipt |

An `EXITED` receipt with a nonzero exit code is still a complete observation.
It reports command failure faithfully; it does not turn failure into success.

## Receipt Boundary

The durable record contains:

- Sounio source and frozen-semantics hashes;
- toolchain, hardware, command, environment, and executable hashes;
- grant digest and kernel generation;
- first and second Sounio decision hashes;
- outcome kind, exit code or signal, and elapsed microseconds;
- result digest and Sounio outcome-decision hash.

The kernel journal stores only the receipt digest and grant identity digest.
The full record remains an append-only evidence artifact outside the command's
stdout/stderr stream.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: loom-durable-execution-outcome-20260827
Owner: codex-1/loom-transactional-custody-transfer-20260827
Concept-IDs: SOUNIO-LOOM-MULTIPLEXER
Intent-Preserved: no execution becomes claim-ready without a durable, grant-bound observed ending
Transformation: extend one-use execution custody from issue/consume to issue/consume/outcome
Types-Changed: add outcome kind, pending outcome obligation, and outcome receipt
Effects-Changed: child execution becomes supervised and outcome commit becomes Sounio-gated
IR-Changed: none
Claims-Introduced: a completed receipt proves a bounded observed process ending for one consumed grant
Claims-Forbidden: consume proves success; stdout is semantic truth; process silence proves exit; OCaml creates expected results
Assumptions: inherited terminal streams preserve user interaction; kernel journal fsync remains durable on the current storage boundary
Write-Set: Sounio outcome authority, frozen manifest, OCaml broker/kernel protocol, gates, evidence
Read-Set: execution authority V2, in-memory grant custody, semantic and Guardian journals
Positive-Witness: exit-zero, exit-nonzero, and signaled children each close exactly one pending grant
Negative-Witness: missing, replayed, wrong-generation, foreign-peer, and recovery-incomplete outcomes refuse
Sabotage-Control: removing only the nonzero result-digest rule admits an otherwise unchanged zero-result frame
Acceptance-Gate: Sounio expected cases, frozen semantics, causal sabotage, OCaml outcome parity, crash recovery, replay, and foreign-peer tests
Integration-Target: Loom Exec/Bash pre-execution attachment
Authoritative-Only-If: the Sounio outcome executable is frozen by hash before OCaml realization and every completed receipt carries all required bindings
```

## Evidence State

| Layer | Status |
| --- | --- |
| `GARDEN` | Captured by this seed. |
| `SOUNIO_EXECUTABLE` | Not yet. |
| `SEMANTICS_FROZEN` | Not yet. |
| `PARITY_OPEN` | No. |
| `CLAIM_READY` | No. |

## What This Is Not

- Not global Exec/Bash attachment.
- Not a claim that command output is scientifically correct.
- Not hostile same-UID isolation.
- Not a substitute for per-lane UID, namespace, LSM, or cgroup isolation.
- Not a semantic role for OCaml, shell, Python, Rust, or an external LLM.
- Not permission to mutate the existing execution-authority V2 freeze.

## Next Executable Bridge

Implement the outcome algebra and its expected results in a Sounio executable.
Commit it as the immediate child of this Garden commit. Freeze its source,
entrypoint, toolchain, command, and result hashes before any OCaml outcome
protocol is written.
