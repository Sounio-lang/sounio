# Garden: Sovereign Execution Kernel V1

Action: 9042

Concept-ID: `SOUNIO-LOOM-SOVEREIGN-EXECUTION-KERNEL`

## Question

Can LOOM execute one material operation without exporting release authority,
continue that operation through interface and coordinator death, reject a
hostile same-UID peer before execution, and revoke fail-closed when the true
HostGuardian dies?

## Ordered Experiment

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> MATERIAL_JOIN -> PARITY_OPEN`

The Sounio executable must exist before the material canary. The material layer
receives frozen Sounio decisions and the frozen hashes of actions 9025, 9030,
and 9031. It cannot encode alternative expected decisions.

## Treatment

1. HostGuardian issues an in-memory grant bound to one exact client identity.
2. The client connects over Unix domain transport with no token or grant
   handle and requests consumption.
3. HostGuardian authenticates `SO_PEERCRED`, pidfd, start tick, harness
   ancestry, executable, and operation before an atomic `ISSUED -> CONSUMED`
   transition.
4. The transport, GUI surrogate, and coordinator surrogate die.
5. The material witness completes exactly once under HostGuardian custody.

## Causal Controls

- Same-UID hostile peer attempts `CONSUME` and `RELEASE`; both must be refused
  before any material process starts.
- A copied textual request contains no authority and must not consume a grant.
- Killing the true HostGuardian before release must revoke the grant and cause
  affirmative process extinction with no material completion marker.
- Flipping only the Sounio same-UID production rule must promote the unchanged
  negative witness, proving the rule is load-bearing.
- Python and Rust oracle attempts must be refused before execution.

## Exit Claim

The experiment may set `same_uid_peer_isolation=true` only for the joined
action-9025 material boundary plus the action-9042 spoof path. It must keep
`production_activation=false`, `exec_attached=false`, and `claim_ready=false`
until the public `loom exec` path is switched and independently remeasured.
