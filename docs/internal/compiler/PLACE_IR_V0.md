# Place IR v0

Status: experimental differential shadow
Semantic-Lane-ID: `SOUNIO-PLACE-IR-V0`
Integration: disabled
Legacy path: retained as the differential oracle

## Decision

A place is not merely a register plus an opcode-specific integer. It is the
identity of a location and the ordered path used to reach it. Place IR v0 makes
that identity explicit without replacing the current IR, lowerer, borrow
checker, SOIR format, or any backend.

The v0 module is intentionally absent from `self-hosted/ir/mod.sio` and from the
default compiler pipeline. `place_v0_verify_for_codegen` rejects every legacy
shadow place with `PLACE_V0_ERR_DIFFERENTIAL_ONLY`. Passing the v0 probe means
only that the representation and adapter truth table are executable.

## Current Legacy Contract

Repository inspection at base `17b0858f6e7d75c9cfc9e545b1b9f0805fa9d5d6`
found these location forms:

| Source place | Legacy operation | Hidden contract |
|---|---|---|
| local read/write | `IrCopy` | local identity is a vreg binding |
| global read/write | `IrLoadGlobal` / `IrStoreGlobal` | `imm_i64` is a BSS byte offset |
| managed field | `IrFieldGet` / `IrFieldSet` | `label_id = 0` |
| reference field | `IrFieldGet` / `IrFieldSet` | `label_id = 1`, implicit autoderef |
| managed index | `IrIndexGet` / `IrIndexSet` | `label_id = 0` |
| reference index | `IrIndexGet` / `IrIndexSet` | `label_id = 1`, implicit autoderef |
| raw word index | `IrIndexGet` / `IrIndexSet` | `label_id = 2`, scale 8 |
| raw byte index | `IrIndexGet` / `IrIndexSet` | `label_id = 3`, scale 1 |
| raw dereference | `IrUnaryOp(OpDeref)` / `IrStorePtr` | pointer address space is implicit |

`label_id` also carries unrelated meanings for control flow, SIMD masks,
certificates, and other opcodes. `field_idx` similarly serves unrelated payload
roles. `IrIndexSet` stores its value register in `imm_i64`. These are historical
encodings, not semantic laws.

The checker already has a separate borrow-analysis `Place` with a base variable
and `Field`, `Index`, and `Deref` projections. It does not reach lowering or
codegen and does not carry address space, mutability, value category, or
type/layout provenance. Place IR v0 does not replace that structure.

## Representation

`PlaceV0` preserves six independent facts:

1. **Root identity**: local, parameter, global, or temporary plus a stable ID.
2. **Address space**: stack, global, managed handle, raw reference, raw word,
   or raw byte.
3. **Ordered projections**: field ordinal, dynamic index vreg, and dereference.
4. **Mutability**: shared, mutable, or frozen.
5. **Value category**: location, managed handle, or reference.
6. **Type/layout provenance**: type ID, layout ID, defining module, source, and
   an explicit completeness bit.

Semantic identities remain separate from physical layout. A field projection
stores its field ordinal and the layout identity that can resolve it; it does
not pretend the ordinal is already a byte offset. An index projection stores
its index-vreg identity and explicit byte scale.

Reference field/index autoderef is represented as an actual `Deref` projection
followed by `Field` or `Index`. The order therefore survives rather than being
reconstructed by a backend.

## Legacy Shadow Adapter

Internally, `LegacyPlaceInstrV0` is an opcode-independent view of the exact
legacy fields. A future pipeline bridge must explicitly match `IrOpcode` into
`LegacyPlaceOpV0`; enum ordinal values are never accepted as wire semantics.
The Place aggregate, projections, enums, and aggregate verifier result are
private to the module. Public adapter and codegen-verifier boundaries accept
stable raw `i64` tags and return an exact scalar status only, avoiding
dependence on both the legacy aggregate-return ABI and cross-module enum/global
transport.

The adapter:

- maps every known location encoding to an explicit `PlaceV0`;
- rejects unknown operation or label tags;
- rejects writes unless the place is explicitly mutable;
- rejects missing type/layout provenance;
- marks every result `differential_only = true` and `requires_oracle = true`.

Structural verification permits comparison with the legacy oracle. Codegen
verification rejects the same value until the differential contract below is
discharged. The only public adapter/verifier slices in v0 are deliberately
fixed receipts: `place_v0_legacy_field_label7_status` proves that raw
`label_id = 7` maps to scalar `PLACE_V0_ERR_LEGACY_LABEL`, while
`place_v0_legacy_field_valid_codegen_status` proves that a valid shadow field
is rejected with `PLACE_V0_ERR_DIFFERENTIAL_ONLY`. Parameterized entrypoints,
including index, deref, global, local, access, mutability, root, and type/layout
context, remain internal. The current imported native path erased even one
scalar argument in the executable probe; widening the public API would
therefore create an unsupported contract.

## Differential Contract

The legacy pipeline remains authoritative. A future shadow runner must compare
both paths for the same checked program and record, per access:

- root identity;
- ordered projection sequence;
- address-space class;
- read versus write;
- field ordinal or index identity and scale;
- type/layout provenance;
- produced value or mutation;
- backend target and ABI.

The minimum parity matrix is:

```text
local read/write
global read/write
managed field read/write
shared-reference field read
mutable-reference field write
managed index read/write
reference index read/write
raw word index read/write
raw byte index read/write
raw dereference read/write
nested deref.field.index read/write
```

Negative witnesses must include unknown address tags, missing provenance,
shared writes, projection overflow, invalid field/index identities, and layout
disagreement.

## Authority Boundary

Place IR v0 is authoritative only if all of the following become true:

1. a pipeline bridge exists without deleting the legacy path;
2. every row in the parity matrix passes on each claimed backend;
3. unknown and contradictory states fail closed;
4. SOIR preserves the complete place representation across round-trip;
5. borrow and alias analysis consume the same place identity;
6. the founder explicitly opens the integration decision.

Until then, no result from this module supports a claim that Place IR is
integrated, backend-complete, ABI-stable, or a replacement for the legacy IR.

## Explicitly Not Done

- Default pipeline integration: **NOT DONE**.
- Differential pipeline integration and program-level parity runner: **NOT DONE**.
- SOIR serialization of `PlaceV0`: **NOT DONE**.
- Backend consumption of `PlaceV0`: **NOT DONE**.
- Replacement or unification of the borrow checker's legacy `Place`: **NOT DONE**.
- Parameterized public adapter ABI: **NOT DONE**. The executable witness proves
  only fixed zero-argument scalar receipts; aggregate returns, imported enums,
  imported globals, and even a scalar input were observed losing payload on the
  current legacy import path.

## Semantic Lane Receipt

```text
Semantic-Lane-ID: SOUNIO-PLACE-IR-V0
Owner: Codex place_ir_v0 lane
Concept-IDs: SOUNIO-PLACE-IR-V0 (draft, not registered)
Intent-Preserved: location identity and ordered access path remain observable
Transformation: legacy location fields -> explicit differential PlaceV0
Types-Changed: none in the default language or pipeline
Effects-Changed: none
IR-Changed: standalone shadow module only
Claims-Introduced: bounded representation and adapter truth table are executable
Claims-Forbidden: integration, backend parity, ABI parity, borrow-checker replacement
Assumptions: legacy behavior remains the oracle during shadowing
Write-Set: this spec, place_v0.sio, place_v0_probe.sio, place_ir_v0_gate.sh
Positive-Witness: PLACE_IR_V0_PASS from the bounded truth table
Negative-Witness: unknown label, shared write, incomplete provenance, codegen rejection
Acceptance-Gate: bash scripts/dev/place_ir_v0_gate.sh
Integration-Target: none in v0
Authoritative-Only-If: full differential parity contract and founder decision
```
