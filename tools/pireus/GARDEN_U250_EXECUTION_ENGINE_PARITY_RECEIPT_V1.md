# Pireus: U250 Engine Parity Must Have Its Own Receipt

> **Status**: Garden | **Date**: 2026-08-28 | **Authority**: internal review

## Discovery

The `v0` freeze correctly emitted `execution_engine_parity_open=false`, but its
request classifier contained a latent promotion path: a caller could set
`semantics_frozen=true` and `parity_open_requested=true`, then receive an open
engine-parity result without presenting an engine-parity receipt.

No gate or recorded result exercised that path. The frozen `v0` evidence is
therefore an honest diagnostic artifact, not the current acceptance contract.

## Correction

The U250 execution-engine projection in this phase has no parity-receipt
ingestor. It must consequently refuse every `parity_open_requested=true`
request, both before and after its semantics freeze.

```text
semantics_frozen=false + parity requested -> REFUSED
semantics_frozen=true  + parity requested -> REFUSED
```

The result remains:

```text
parent_material_parity_open=true
execution_engine_parity_open=false
claim_ready=false
```

The parent flag records lineage only. A future transition to engine parity
requires a new Sounio-first contract and an ingestor that validates a distinct,
hash-bound parity receipt. It may not reuse the parent inventory receipt as
proof of the new ontological projection.

## Required Negative

Add a negative test that sets both `semantics_frozen=true` and
`parity_open_requested=true`. The request must be invalid, must report
`PARITY_REFUSED`, and must keep `execution_engine_parity_open=false`.

## Order

```text
GARDEN_V1_CORRECTION
-> SOUNIO_EXECUTABLE_V1
-> SEMANTICS_FROZEN_V1
```

No material probe is repeated and no FPGA kernel is launched during this
correction.
