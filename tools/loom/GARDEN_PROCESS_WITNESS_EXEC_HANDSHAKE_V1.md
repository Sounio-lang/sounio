# GARDEN: LOOM ProcessWitness Exec Handshake V1

Status: `PREREGISTERED`

## Question

Can a Sounio payload remain alive long enough for LOOM to observe the exact
principal-cell-to-payload `execveat` transition on the same pidfd, without
making timing, polling luck, a shell, or a textual acknowledgement authoritative?

## Frozen Parents

This is a derived material experiment under the existing ProcessWitnessCell. It
does not add a semantic state or decision.

- Sounio action `9030` remains `SEMANTIC_AUTHORITY`.
- `tools/loom/host_exec_quorum_host.runtime.v1` remains the frozen distinct-UID
  host grant with `material_grant=true` and `material_execution=false`.
- `tools/loom/process_witness_payload.freeze.v1` remains the frozen single-shot
  Sounio calibration result.
- `tools/loom/GARDEN_PROCESS_WITNESS_CELL_V1.md` remains the parent hypothesis.

If implementation requires semantics not already present in action `9030`, it
must stop and return to `GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN` before
any C++ change. Python and Rust remain forbidden. No second broker is permitted.

## Hypothesis

The observation race disappears if the Sounio payload itself implements a
bounded, descriptor-custodied two-phase protocol:

```text
PrincipalCell --exact release--> execveat(frozen payload fd)
Sounio payload --READY---------> broker observes same pidfd and new /proc/exe
broker --------CLOSE-----------> Sounio payload emits DONE and exits
broker --------closure---------> outcome plus extinction receipt
```

`READY` and `DONE` are observations, not authority. The authority is the
conjunction of the consumed action-`9030` grant, descriptor custody, exact
payload fd, and unchanged kernel identity. A forged line without that
conjunction cannot advance state.

## Exact Protocol

The broker creates one anonymous full-duplex custody channel before launching
the DynamicUser unit. The material bootstrap may temporarily use two anonymous
pipes if systemd cannot transport one socketpair without weakening identity;
the receipt must declare the topology.

The pre-exec principal cell consumes, without read-ahead:

```text
ARM\n
<64 lowercase hex generation>\n
```

It does not wait for EOF. Extra bytes already present, an early EOF, timeout,
wrong generation, duplicate generation, or malformed frame refuses before
`execveat`. The writer remains open and becomes the sole post-exec close
authority.

After validating the release, the cell:

1. opens the root-owned, non-writable, non-symlink Sounio payload;
2. hashes the open descriptor and compares it with the frozen payload manifest;
3. validates regular-file metadata and executable mode;
4. retains only stdin, stdout, stderr, and the executable descriptor;
5. calls `execveat(payload_fd, "", argv, empty_env, AT_EMPTY_PATH)` directly.

The payload receives no semantic value through argv or environment. Its argv is
one frozen display name and its environment is empty.

The Sounio payload emits one exact bounded `READY` line, then reads an exact
byte-framed `CLOSE\n` record. It emits the frozen `DONE` line and returns zero
only for that record. EOF, timeout, extra bytes, partial input, or any other
record emits a Sounio-defined refusal and returns nonzero.

## Same-PID Observation

After `READY` and before `CLOSE`, the broker must observe all of these for the
same live pidfd captured before release:

- PID and start tick unchanged;
- four-value UID and GID vectors unchanged and distinct from the broker;
- cgroup and namespace unchanged;
- `/proc/<pid>/exe` changed exactly from the frozen principal-cell binary to the
  frozen Sounio payload;
- post-exec executable bytes match the frozen payload hash;
- no child, sibling, shell, interpreter, or cgroup escape exists;
- controller generation remains live and already consumed.

Only then may the broker send `CLOSE`. Polling may discover the transition, but
poll timing cannot decide it: failure to observe every fact before the bounded
deadline is a refusal.

## Closure

After `DONE`, the broker closes the writer, waits on the same pidfd, reaps the
unit, observes the cgroup unpopulated, and binds stdout, stderr, wait status,
process identity, executable transition, effect set, and the state/generation/
authority extinction triple into one candidate receipt. Action `9022`, action
`9029`, and action `9030` must close that exact observation before any success
promotion.

## Causal Controls

The host gate must preserve identical payload bytes and independently prove:

1. no release means no `execveat` and no `READY`;
2. bypassing only the release check produces `READY`, proving causality;
3. wrong generation means no `execveat`;
4. payload hash substitution means no `execveat`;
5. forged `READY` without the executable transition does not cause `CLOSE`;
6. wrong close bytes produce a Sounio refusal and nonzero status;
7. controller death before release means no `execveat`;
8. broker death after `READY` causes PDEATHSIG/unit cleanup and no success;
9. replay of the terminal receipt starts no process;
10. same-UID execution remains closed;
11. omission of any extinction member prevents close;
12. Python or Rust oracle attempts are refused before their process exists.

The direct-release bypass is sabotage only. It must remain unreachable from the
product protocol.

## Acceptance Boundary

This Garden commit must predate the two-phase Sounio payload and every
exec-capable C++ byte. A local payload gate and freeze come first. The first host
probe remains a side-by-side experimental release with production current and
production broker unchanged.

`material_execution=true` is allowed only after the host proves one treatment,
one positive exact execution, the causal bypass, all negative controls, the
same-pid executable transition, complete effects, and affirmative extinction.
Until then it remains false.

Even after a successful calibrated host execution:

- `launch_open=false`;
- `recycle_open=false`;
- `exec_attached=false`;
- `commit_attached=false`;
- `ci_attached=false`;
- `parity_open=false`;
- `claim_ready=false`.

## Stop Rule

Stop on timing-based identity, a path reopened after hashing, shell mediation,
environment authority, premature EOF as success, same-UID admission, missing
pidfd continuity, unobserved descendants, incomplete effects, inferred
extinction, production activation, or any expected result first defined outside
Sounio.
