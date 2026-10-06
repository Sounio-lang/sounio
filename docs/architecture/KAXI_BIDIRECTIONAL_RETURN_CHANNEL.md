<!-- docs:meta
topic_id: repo.docs.architecture.kaxi-bidirectional-return-channel
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.kaxi-bidirectional-return-channel
-->

# K-AXI Bidirectional Return Channel (Sprint 1 Design)

## Scope

This document defines the bidirectional extension for K-AXI while preserving Sprint 1 hard gates:

- Keep `L1/L2/L3 + QIR telemetry` unchanged.
- Do not introduce new release gates for this feature in Sprint 1.
- Keep fallback-safe behavior (no regression of current unidirectional path).

## Goals

1. Add a return path from FPGA epistemic fabric to host/runtime.
2. Carry hardware-computed epistemic signals (counter deltas, status, faults, optional replay digest).
3. Keep transport deterministic and reproducible.
4. Preserve compatibility with current K-AXI publish ring and strict gate scripts.

## Non-Goals (Sprint 1)

1. No mandatory runtime activation in release gates.
2. No PCIe/NVLink vendor-specific binding in this phase.
3. No change to the current `OMEGA_REQUIRE_*` hard gate matrix.

## Channel Model

Current channel (already implemented):

- `TX`: host/runtime -> `k_axi_master` -> `k_axi_slave_epistemic` -> epistemic fabric

New channel (design in this doc):

- `RX`: epistemic fabric -> `k_axi_return_mux` -> host/runtime return queue

Both channels remain logically independent and share only configuration/control.

## Return Packet Schema (KAXI-RX v1)

All fields are fixed-width and deterministic:

- `kind` (8 bits): packet class
- `status` (8 bits): `ok`, `degraded`, `fallback`, `error`
- `op_kind` (8 bits): mirrors TX op class (`add/mul/div/fma/prop_var`)
- `flags` (8 bits): reserved policy bits
- `tx_seq` (32 bits): sequence id from transmit side
- `fidelity_inc` (32 bits): delta for hardware fidelity counter
- `prov_depth_inc` (32 bits): delta for provenance depth
- `formal_cov_inc` (32 bits): delta for formal coverage lane
- `epi_log_q32_32` (64 bits): accumulator snapshot or delta
- `digest` (256 bits): replay digest (Merkle-lane compatible)
- `timestamp_cycles` (64 bits): optional monotonic cycle stamp

Recommended packed payload size: 544 bits (align to 576 bits if bus tooling requires).

## State Machine (Reference)

`k_axi_return_mux` reference states:

1. `IDLE`: wait for `fabric_valid`.
2. `CAPTURE`: latch payload and metadata atomically.
3. `QUEUE`: write into return FIFO/ring slot.
4. `EMIT`: assert return valid toward host-side consumer.
5. `ACK_WAIT`: optional backpressure/ack handling.
6. `DROP_SAFE`: if overflow policy is `drop_oldest` or `drop_newest`, increment overflow counter and continue.

Determinism rule:

- For equal input stream and equal backpressure schedule, emitted RX sequence must be identical.

## Flow Control and Reliability

1. Backpressure:
- RX channel must support ready/valid handshaking.
- TX must never stall solely because RX is congested in Sprint 1 compatibility mode.

2. Overflow policy:
- Default: `drop_newest` with explicit counter increment.
- Optional mode: `drop_oldest` for bounded-latency telemetry.

3. Error signaling:
- `status=error` packets are best-effort and must not deadlock the data plane.

4. Reproducibility:
- All drops and overflow events are counted and exported in report artifacts.

## Integration Points

Planned modules (non-gating in Sprint 1):

- `hardware/fpga/k_axi_return_mux.v` (new)
- `hardware/fpga/k_axi_return_fifo.v` (new)
- `hardware/fpga/tb_k_axi_bidirectional.v` (new)

Existing modules extended by wiring only:

- `hardware/fpga/k_axi_slave_epistemic.v` (emit RX deltas)
- `hardware/fpga/epistemic_power_accumulator.v` (snapshot lane, optional)

Scripts (optional, non-hard-gate additions):

- `scripts/research/run_fpga_epistemic_seed.sh`: run bidirectional TB when files exist.
- `artifacts/fpga/fpga_seed_report.json`: append `k_axi_return_*` fields when available.

## Compatibility Contract

1. If RX modules are absent:
- Build and gate behavior remains unchanged.
- Report marks RX status as `not_present`.

2. If RX modules are present but fail:
- In Sprint 1, report as warning unless explicit strict flag is enabled.

3. Suggested future strict flag:
- `OMEGA_REQUIRE_K_AXI_BIDIR=1` (Sprint 2+).

## Security and Safety Notes

1. Digest lane is advisory in Sprint 1 and must not be treated as cryptographic proof.
2. Return packets must not leak raw host pointers or unsafe addresses.
3. Invalid op/status combinations are normalized to `status=error`.

## Accumulator Approximation Bounds (Formalized)

Hardware accumulator computes:

- `L_hw = floor(F / 8) + floor(P / 16) + floor(Q / 32)`

where:

- `F`: fidelity accumulator (integer, non-negative)
- `P`: provenance accumulator (integer, non-negative)
- `Q`: quantum accumulator (integer, non-negative)

Real-valued weighted reference:

- `L_ref = F/8 + P/16 + Q/32`

Define floor errors:

- `e_F = F/8 - floor(F/8)`, `e_P = P/16 - floor(P/16)`, `e_Q = Q/32 - floor(Q/32)`
- each `e_*` is in `[0, 1)`.

Then:

- `L_ref - L_hw = e_F + e_P + e_Q`
- `0 <= L_ref - L_hw < 3`

Therefore absolute approximation bound:

- `|L_ref - L_hw| < 3`

Interpretation:

1. Hardware score is a conservative under-approximation.
2. Worst-case gap is strictly less than `3` counter units.
3. Relative error decays as counters grow, so long-running workloads converge toward stable ranking consistency.

## Verification Plan (Deferred to Sprint 2 gate)

1. Exhaustive small-domain check:
- Enumerate `F,P,Q` in `[0, 255]` and verify bound.

2. Random large-domain check:
- Sample across realistic counter scales from simulation logs.

3. Monotonicity check:
- Verify that increasing any one of `F,P,Q` does not decrease `L_hw`.

4. Ranking sanity:
- Compare ordering induced by `L_hw` vs `L_ref` over corpus traces.

## Decision Log

1. Keep bidirectional channel as design/integration-ready for Sprint 1, not hard gate.
2. Keep current unidirectional gate path untouched.
3. Adopt explicit approximation bound to support future formal proof and gate hardening.
