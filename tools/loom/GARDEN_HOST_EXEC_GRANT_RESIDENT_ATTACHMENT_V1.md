# Garden: host ExecGrant resident attachment v1

Status: preregistered before implementation and measurement.

## Question

Can the existing root-owned, socket-activated LOOM broker bind frozen Sounio
action 9030 to one persistent Sounio resident generation without opening a
material execution route?

## Authority boundary

- Sounio action 9030 remains the only semantic authority for grant decisions.
- The frozen action 9030 manifest SHA-256 is
  `8687d889e08f69190daaf3cdbee02741cde3ce62f136ba63df1fa9c2ccb0d051`.
- The frozen resident v4 manifest SHA-256 is
  `f61c93a3aefdbab792ed757faddf778017d34e0fa6bed97c565b56fe3147d473`.
- OCaml is operational policy, lifecycle, and effect parity. It does not author
  expected semantic decisions.
- C++20 is a transitory host material realization. It may verify hashes,
  supervise processes, move bytes, enforce deadlines, and refuse. It may not
  reinterpret a Sounio decision.
- Python and Rust are prohibited as producer, oracle, launcher, or fixture
  generator.

## Treatment

The broker starts exactly one hash-pinned resident v4 process for its lifetime.
It records PID, `/proc/<pid>/stat` start tick, executable identity, a random
generation digest, and a monotonic request sequence. Every action 9030 request
is enclosed by the frozen resident action 9024 sequence:

`START -> REQUEST -> route 5/action 9030 -> RESPONSE -> ... -> STOP`.

The public broker command is `GRANT_ADMIT <single-line action-9030-frame>`.
`ADMIT` remains the action 9029 route. `LAUNCH`, `RECYCLE`, `EXEC`, and every
unknown command remain closed.

## Fail-closed rules

The resident generation is permanently poisoned, with no in-process restart,
on timeout, EOF, malformed output, extra output, PID/start-tick/executable
drift, sequence drift, or resident action 9024 envelope refusal. A semantic
DENY from action 9030 is a valid decision and does not poison or release any
host capability.

The resident child must die when its broker parent dies. A later service start
creates a new broker generation and a new resident generation; it cannot resume
an old request or grant.

## Preregistered controls

1. Two valid requests in one broker lifetime must report the same resident PID,
   start tick, executable digest, and generation digest with strictly increasing
   sequence numbers.
2. The current material frame must return the frozen `DENY491` decision.
3. The frozen Python-laundering frame must return `DENY499`; its sentinel must
   remain absent.
4. Manifest, runtime, and broker argument tamper must refuse before resident
   execution.
5. Timeout, malformed output, resident death, and PID identity drift must poison
   the generation and make the next request refuse without a restart.
6. Replaying a completed sequence or a response from another generation must
   refuse.
7. A semantic `DENY491` must not write or release the PrincipalCell barrier.

## Acceptance boundary

This experiment may establish `resident_action_9030_attached=true` and
`decision_transport_material=true`. It must retain:

- `material_grant=false`
- `material_execution=false`
- `launch_open=false`
- `recycle_open=false`
- `exec_attached=false`
- `parity_open=false`
- `claim_ready=false`

Opening execution requires a separate, preregistered host quorum experiment.

