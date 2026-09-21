# Host ExecGrant resident attachment v1

Status: packaging ready; host activation not yet measured.

## Frozen lineage

This attachment implements the experiment preregistered by
`GARDEN_HOST_EXEC_GRANT_RESIDENT_ATTACHMENT_V1.md`.

- Semantic authority: Sounio action 9030
- Action 9030 manifest SHA-256:
  `8687d889e08f69190daaf3cdbee02741cde3ce62f136ba63df1fa9c2ccb0d051`
- Resident v4 manifest SHA-256:
  `f61c93a3aefdbab792ed757faddf778017d34e0fa6bed97c565b56fe3147d473`
- Resident v4 runtime SHA-256:
  `58d2bff59bd3ad43080d1e9dc670268735d458431ac524ee4138f17f55cbbcca`
- Broker resident implementation commit: `6d8603197a`
- Local persistent-resident gate commit: `8d4514f6bb`

The host broker is a transitory C++20 material realization. The resident process
is compiled Sounio. C++ may enforce identity, framing, sequence, deadline, and
failure closure, but it does not author an expected semantic result.

## Host protocol

The root-owned socket protocol adds one decision route:

`GRANT_ADMIT <single-line action-9030-frame>`

The route passes through one broker-lifetime Sounio process and the frozen
action 9024 envelope:

`START -> REQUEST -> route 5/action 9030 -> RESPONSE -> ... -> STOP`.

The material receipt binds the action and resident manifests, resident runtime,
PID, start tick, random generation digest, monotonic sequence, input frame,
Sounio decision, decision code, and latency. A semantic DENY is returned as a
valid receipt with `barrier_release=false`.

## Failure closure

Timeout, EOF, malformed/extra output, process death, executable or start-tick
drift, and envelope refusal poison the resident generation. The broker performs
no in-process restart. systemd may restart the failed broker service, which
creates a new broker generation and a new resident generation. The child uses
`PR_SET_PDEATHSIG=SIGKILL`, so broker death cannot leave its Sounio process
resident.

## Local evidence

The source-fresh local gate observed two decisions in one generation:

- sequence 1: current material -> `DENY491`
- sequence 2: Python authority laundering -> `DENY499`

PID, start tick, executable and generation remained stable. Separate destructive
generations proved that death, timeout, and malformed output poison the
generation and that replay after poison refuses. Action manifest, resident
manifest, and resident runtime tamper all refused before resident admission.

## Promotion boundary

The immutable release lineage changes from `9029-*` to `9030-*` and includes the
action 9030 manifest, resident v4 manifest, source-fresh resident runtime, broker,
prior authorities, unit files, and this contract.

Until a root/systemd host gate records the live resident generation:

- `host_activation=unmeasured`
- `resident_action_9030_attached=local-gate`
- `decision_transport_material=true`
- `material_grant=false`
- `material_execution=false`
- `barrier_release=false`
- `launch_open=false`
- `recycle_open=false`
- `exec_attached=false`
- `parity_open=false`
- `claim_ready=false`

