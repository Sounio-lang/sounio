<!-- docs:meta
topic_id: repo.docs.architecture.kaxi-bidirectional.v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.kaxi-bidirectional.v1
-->

# K-AXI Bidirectional v1

## Purpose

`K-AXI` v1 defines the return channel from FPGA epistemic fabric to host/runtime with deterministic packet replay and strict gate visibility.

## Packet Header (32-bit packed lane)

- byte0 `kind`
- byte1 `status`
- byte2 `op_kind`
- byte3 `flags`

Packed form:

- `header = kind | (status << 8) | (op_kind << 16) | (flags << 24)`

## Return Payload Core

- `tx_seq` (u32)
- `fidelity_inc` (u32)
- `prov_depth_inc` (u32)
- `formal_cov_inc` (u32)
- `epi_log_q32_32` (i64)
- `digest` (u256)
- `timestamp_cycles` (u64)

## Replay Semantics

- `kind=0xA1` marks epistemic return packets.
- `op_kind in [1..5]` marks epistemic operations.
- `status` values:
  - `0x01 ok`
  - `0x02 degraded`
  - `0x03 fallback`
  - `0x04 error`

## Gate Contract

Strict gate must confirm:

1. `k_axi_return_sim_status=pass`
2. `k_axi_return_synth_status=pass`
3. replay adapter exists at `hardware/rtl/kaxi/bidirectional_return_adapter.sio`
4. waveform snapshot exists at `artifacts/fpga/waveforms/tb_k_axi_bidirectional.vcd`

## Notes

- This v1 spec is additive and compatible with `docs/architecture/KAXI_BIDIRECTIONAL_RETURN_CHANNEL.md`.
- It is intentionally hardware-first and self-hosted oriented.
