# Loom Durable Execution Outcome V1

Status: `SEMANTICS_FROZEN_RUNTIME_BUNDLED_ATTACHMENT_REFUSED`

Loom Durable Execution Outcome V1 closes the semantic gap between consuming an
execution grant and proving what happened to the measured child. Grant
consumption is not success. Silence is not success. A receipt file by itself is
not success. The kernel records success only after frozen Sounio action `9022`
admits a complete outcome and the exact consuming broker closes its pending
obligation.

## Authority order

The implementation follows:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY`

The Garden seed is
`tools/loom/GARDEN_DURABLE_EXECUTION_OUTCOME_V1.md`. Sounio is
`SEMANTIC_AUTHORITY`; OCaml is the operational kernel and broker realization.
The frozen identities are:

- action: `9022`;
- manifest SHA-256:
  `f5e63a2fd6a946cea1a4cb57013ae0cfa1772c42c3cc52e42d300dfb7b45e16e`;
- semantics SHA-256:
  `c98c13d30d66ba2fb3d0fb34d75bd21b14b353bc88fd80acf7dbb385cb9fa914`;
- executable SHA-256:
  `be1fc973e7a86f43a556fde66b6ad54f188b598cfd22a919829492677cec3fd5`.

Python and Rust are forbidden as semantic authorities and Guardian
implementations. The Sounio executable was frozen before the OCaml realization
opened. The shared runtime bundle advertises
`loom-durable-execution-outcome-v1` only when its 9022 executable, semantics,
manifest, and transactional-custody dependency all verify.

## State transition

`EXEC_CONSUME` atomically burns the grant and opens one in-memory outcome
obligation bound to:

- kernel instance and generation;
- random grant handle digest;
- exact broker PID and Linux process-start tick;
- exact issued cwd;
- capability body digest and consumed timestamp.

The broker remains alive, forks the measured leaf, inherits standard streams,
and waits for one kernel-reported outcome. OCaml normalizes portable signal
identities to stable Linux signal numbers before building the Sounio frame.

The permitted observed forms are:

- `EXITED`: exit code `0..255`, signal `0`;
- `SIGNALED`: exit code `0`, signal `1..255`;
- `INCOMPLETE`: never an allowed success outcome.

After observation, the broker fsyncs an observation record, invokes action
9022, fsyncs the final receipt, and submits `EXEC_OUTCOME`. The kernel
reauthenticates the peer and requires the same broker identity, cwd, generation,
handle, receipt hash, and receipt body before journaling
`EXEC_OUTCOME_RECORDED`. It removes the obligation only after that journal
append succeeds. A duplicate submit is a refused replay.

## Receipt binding

Each self-contained receipt binds:

- Sounio source, entrypoint, semantics, manifest, and executable hashes;
- parent execution-authority manifest;
- toolchain and hardware;
- command, environment, and executable;
- grant and kernel generation;
- issue and consume Sounio decisions;
- outcome kind, exit code or signal, and elapsed microseconds;
- observation result digest, 9022 frame digest, Sounio decision digest, and
  whole-record digest.

stdout and stderr remain inherited user-visible streams. They are not semantic
truth and are not silently promoted into the result digest.

## Crash semantics

The kernel journal is the closure authority. On orderly shutdown, or on replay
after kernel loss, every consumed handle without a committed outcome is
materialized as `EXEC_OUTCOME_INCOMPLETE`. Recovery also rotates the kernel
generation, so the old handle cannot be closed retrospectively. This is a
durable negative fact, not a guessed child result.

While the kernel remains live, it also revalidates the consuming broker's PID
and Linux process-start tick. A broker that exits after consume but before
commit causes immediate `EXEC_OUTCOME_INCOMPLETE` materialization. Missing or
invalid outcome policy therefore refuses before child execution and cannot
leave an indefinitely pending success candidate.

Guardian, host, or storage loss remains outside this same-process proof. A
retained journal can prove that an outcome was incomplete; it cannot recreate a
lost process or claim that unretained external side effects were exactly once.

## Gates

Run:

```sh
bash scripts/ci/sounio_loom_execution_outcome_selftest.sh
bash scripts/ci/sounio_loom_execution_outcome_freeze_selftest.sh
bash scripts/ci/sounio_loom_execution_custody_selftest.sh
bash scripts/ci/sounio_coord_runtime_selftest.sh
```

The tests cover 28 Sounio-owned semantic cases, the load-bearing missing-result
rule, normal and nonzero exit, SIGTERM, same-UID outsider refusal, cwd drift,
single use, replay, expiry, kernel recovery, receipt-body hashing, and the crash
window between receipt creation and kernel commit.

Global Bash/Exec attachment remains refused. The audited leaf protocol is
complete; arbitrary shell closure is not. `parity_open=false` and
`claim_ready=false` remain truthful until the remaining language-parity and
attachment gates are separately satisfied.
