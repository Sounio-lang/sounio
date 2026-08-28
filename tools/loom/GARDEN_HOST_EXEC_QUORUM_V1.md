# Garden: host ExecQuorum v1

Status: preregistered before implementation and integrated measurement.

## Research question

Can LOOM materialize a single-use execution grant as a non-serializable kernel
fact, requiring semantic authority, operational lifecycle, and host principal
identity in one transaction, without exposing a bearer token or an execution
command?

## Authority and language roles

The authority order is fixed:

1. Frozen Sounio action 9030 is `SEMANTIC_AUTHORITY`. It alone decides whether
   a transition is semantically admissible.
2. OCaml is `EFFECT_PARITY`. One `ExecGrantController` owns the lifecycle and
   exactly one resident Sounio v4 process.
3. C++20 plus Linux and systemd are transitory `MATERIAL_PARITY`. They supervise
   identities, descriptors, deadlines, and fail-closed teardown.
4. External LLMs are `REVIEW_ONLY` and cannot confirm any result.
5. Python and Rust are prohibited as producer, oracle, launcher, fixture
   generator, or policy bridge.

The controller must not contain a table of expected Sounio decisions. It may
recognize the closed shape of a receipt and refuse malformed output, but it may
not reinterpret `ALLOW` or manufacture one after a refusal.

## Single-resident architecture

There must be one semantic process path, not two competing implementations:

`root broker -> OCaml ExecGrantController -> one hash-pinned Sounio resident v4`

The existing broker-to-Sounio resident path remains the deployed decision-only
baseline until parity is demonstrated. The integrated treatment replaces that
path atomically for the experiment; it does not run a second resident beside
the OCaml controller and compare whichever answer is convenient.

The controller is internal to the broker generation. It has no listening
socket, user command, shell, path, token, or payload surface. Its transport is a
bounded inherited descriptor protocol. Broker or controller death kills the
resident and closes every pending barrier by EOF.

## The three-object quorum

A grant threshold may open only when these objects refer to the same random
generation and request digest:

1. `SemanticPermit`: resident Sounio action 9030 returns an exact frozen
   semantic `ALLOW` receipt.
2. `LinearConsumption`: OCaml moves the cell from `ISSUED` to
   `OUTCOME_PENDING` exactly once; replay, mismatch, timeout, EOF, or revoke
   poisons or closes it.
3. `PrincipalWitness`: the broker revalidates the target PrincipalCell PID,
   start tick, executable, cgroup, distinct dynamic UID/GID, and live pidfd.

No object is serializable as authority. Digests and receipts are audit data,
not capabilities. The material grant is the inherited write end of a private
descriptor barrier. It never receives a pathname and is never exported through
the broker protocol or filesystem.

After all three objects match, the broker writes the exact generation record
once and irreversibly closes its write descriptor. The PrincipalCell may report
only `BARRIER_OPENED`; v1 still has no user execution surface. Every failure
path closes the descriptor without writing, producing `BARRIER_CLOSED`.

## Absence as an affirmative fact

The absence of permission is not inferred from a missing token. It is the
affirmative triple:

`state != OUTCOME_PENDING OR generation != request_generation OR authority != Sounio-9030`

This triple is validated before the one possible descriptor write. Missing,
unknown, stale, duplicated, partially written, or unverifiable state is a DENY.
There is no default allow and no recovery by reconstructing a textual handle.

## Preregistered treatment

The current frozen material request produces Sounio `DENY491`. The controller
must preserve the cell state permitted by the frozen semantics, the broker must
close without writing, and the PrincipalCell must observe exactly:

`BARRIER_CLOSED reason=eof`

Expected product flags remain:

- `semantic_decision=DENY491`
- `linear_consumption=false`
- `principal_witness=measured-but-unused`
- `barrier_release=false`
- `material_grant=false`
- `material_execution=false`

## Preregistered positive experiment

The positive experiment uses only an ALLOW transition already born in and
frozen by Sounio action 9030. It must not be generated retrospectively by
OCaml, C++, shell, or a test oracle. With the matching issue and consume
transitions, a fresh PrincipalCell, and one generation:

1. Sounio returns exact ALLOW receipts.
2. OCaml reaches `OUTCOME_PENDING` once.
3. The host PrincipalWitness remains live and distinct.
4. The broker performs one descriptor write.
5. The PrincipalCell reports one `BARRIER_OPENED` sentinel.
6. A second consume or second write refuses and cannot emit another sentinel.

Passing this experiment may establish `material_grant=true`. It does not
establish material execution because the PrincipalCell still executes no user
payload.

## Causal sabotage matrix

Each row changes one predicate while holding the remaining transaction fixed:

1. Replace Sounio ALLOW with frozen `DENY491` -> barrier closed.
2. Replace the matching consume with replay -> OCaml poisoned, barrier closed.
3. Replace request generation with another valid generation -> barrier closed.
4. Kill or stop the Sounio resident before consume -> barrier closed.
5. Kill the OCaml controller after semantic receipt but before release -> EOF,
   barrier closed.
6. Drift PrincipalCell PID/start tick/executable/cgroup/pidfd -> barrier closed.
7. Use the same UID for the supposed distinct principal -> barrier closed.
8. Copy the textual generation without the inherited descriptor -> barrier
   closed.
9. Duplicate, truncate, or extend the descriptor record -> at most zero open
   sentinels.
10. Present an LLM, Python, Rust, shell, or file receipt as authority -> refuse
    before execution.
11. Isolated laboratory sabotage: bypass only the quorum predicate and perform
    the exact descriptor write -> one open sentinel. This proves that the
    quorum predicate, rather than an unrelated guard, is what withholds release.

The sabotage path is compile-time test-only, unavailable to the installed
broker, and must share the same barrier implementation as treatment.

## Crash and recovery contract

- A grant generation cannot survive broker, controller, resident, or
  PrincipalCell death.
- systemd may start a new broker generation, but cannot resume an issued grant.
- No journal record can recreate the descriptor capability.
- Audit receipts persist; capabilities do not.
- Timeout or uncertain teardown fails closed and requires a fresh semantic
  transaction.

## Public protocol boundary

The installed broker continues to refuse `EXEC`, `LAUNCH`, `RECYCLE`, direct
`BARRIER_RELEASE`, textual grant handles, paths, commands, and payloads. The
integrated v1 experiment is reachable only through a source-fresh selftest and
later a root-only host gate. Promotion requires a new immutable release and
rollback remains armed until the host causal matrix passes.

## Acceptance boundary

An integrated local gate must prove the treatment, positive experiment, all
negative rows, and the isolated exact-write sabotage. A separate host gate must
repeat the quorum with a measured distinct PrincipalCell and verify persistence
after the transport pod exits.

The experiment may establish:

- `single_resident_controller=true`
- `non_bearer_exec_quorum=true`
- `descriptor_barrier_causal=true`
- `linear_grant_consumption=true`
- `material_grant=true`

It must retain until a later execution experiment:

- `material_execution=false`
- `launch_open=false`
- `recycle_open=false`
- `exec_attached=false`
- `commit_attached=false`
- `ci_attached=false`
- `parity_open=false`
- `claim_ready=false`
