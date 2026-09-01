# LOOM Product DynamicUser Garden Remaining V1

Status: `CENSUS_EXECUTABLE`

This is not a Garden seed and not a host freeze. It reads already-recorded
parents and reports which acceptance gates of
`GARDEN_PRODUCT_DYNAMIC_USER_EXEC_ATTACHMENT_V1.md` are still open.

## Question

After the host LaneCell and ExecCell canaries, did the product path replace
same-UID `fork/exec`, fail closed on missing descriptors, prove crash
recovery, seal an immutable runtime receipt, and roll out a fleet lane?

## Observed answer

No. Historical canaries exist. The product composition gap remains.

| Gate | Recorded fact | Product status |
|---|---|---|
| 1 Freeze the counterexample | named-let probe still PASSes | measured; gap live |
| 2 LaneCell canary | `MATERIAL_CANARY_FROZEN`, `fleet_lane_cell_attached=false` | recorded; freeze not live |
| 3 test-only ExecCell | `exec_cell_attached=true` with `test_only=true` | recorded; freeze not live |
| 4 Replace same-UID `fork/exec` | `broker_command_kernel` + `supervise_child` still re-enter and fork | **open** |
| 5 Descriptor absence fail-closed | runtime field and OCaml `required_mode` unset/0 both `false` | **open** |
| 6 Crash recovery | `recycle_open=false` | **open** |
| 7 Immutable runtime receipt | no product receipt; canary `test_only=true` | **open** |
| 8 Fleet canary then fleet | `fleet_lane_cell_attached=false` | **open** |

A historical canary with `exec_cell_attached=true` is not product attachment.
`parity_open` and `claim_ready` remain false.

## Falsifier

The census fails if it cannot run the named-let counterexample, the
payload-identity gate, or the OCaml `required_mode` default ratchet; if
`required_mode_default` becomes true while gate 4 is still open; or if
`parity_open` or `claim_ready` is raised. A red counterexample is not
licence to mark gate 4 closed.

## Boundary

```text
Semantic-Lane-ID: loom-dynamic-user-garden-remaining-20260831
Owner: grok-cli1
Concept-IDs: none
Transformation: none
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: gates 4-8 of the DynamicUser exec Garden remain open on this tip
Claims-Forbidden: parity_open, claim_ready, required_mode_default=true, product exec_attached, production_activation, gate 4 closed
Write-Set: this contract, the census selftest, and the evidence receipt
```

Bash compares recorded fields and invokes existing observational gates.
Python and Rust are forbidden.
