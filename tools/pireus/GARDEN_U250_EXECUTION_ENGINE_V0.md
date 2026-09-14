# Pireus: U250 as an Execution Engine

> **Status**: Garden | **Date**: 2026-08-28 | **Authority**: founder direction

## Butterfly

The admitted AMD Alveo U250 should not remain a detached inventory fact. It is
a material execution engine of Pireus: an FPGA fabric attached to the DL380,
reached through XRT/XDMA, with a declared two-slot fleet and one currently
admitted physical member.

This does not create a second FPGA implementation. The existing HLS and XRT
surfaces remain the implementation line. Pireus supplies the semantic graph
that says what engine exists, what evidence supports it, and what conclusions
do not yet follow.

## Parent Witness

The parent U250 admission semantics and the first material receipt are already
frozen. Sounio classified the sealed C++ facts as:

```text
target_family=AMD_ALVEO_U250
declared_card_count=2
discovered_card_count=1
admitted_card_count=1
missing_card_count=1
status=INVENTORY_PARTIAL
parity_open=true
claim_ready=false
```

The new object may project that frozen result into the existing
`SOUNIO-PIREUS-EXECUTION-ENGINE` graph. It may not reinterpret the material
receipt or manufacture facts for the unresolved second slot.

## Ontological Shape

The U250 specialization must represent:

1. `AMD_ALVEO_U250` as a canonical target;
2. `FPGA` as an engine kind;
3. `XCU250` as a fabric, not an ISA;
4. `XRT_XDMA` as an execution interface, not an ISA or operation;
5. a target blueprint declaring two engine slots;
6. one observed engine attached to the canonical DL380 machine;
7. one unresolved engine slot, which is not an observed engine;
8. the admitted engine's four-bank, 64 GiB external DDR profile;
9. the sealed material receipt as evidence for presence only;
10. zero admitted operation capabilities.

The historical U250 material-machine identifier is bridged to the canonical
DL380 identifier already used by the execution-engine ontology. The bridge is
explicit; Pireus must not silently fork the machine into two entities.

## Refusals

Engine admission is not operation admission. In particular:

```text
XRT present != instruction set
shell ready != kernel correct
DDR present != lowering legal
one observed card != dual fleet complete
material receipt != semantic authority
engine admitted != cost known
engine admitted != performance claim
```

The first executable must therefore keep
`operation_capability_count=0`, `lowering_authorized=false`,
`cost_present=false`, `kernel_correctness_present=false`, and
`claim_ready=false`.

Requests that smuggle any of those conclusions into the projection are
refused. Python and Rust remain prohibited as oracles. C++ remains
`MATERIAL_PARITY`; external LLMs remain `REVIEW_ONLY`.

## Expected First Result

The first Sounio executable consumes the already sealed parent receipts and
must emit:

```text
status=ENGINE_INVENTORY_PARTIAL
canonical_target=true
engine_kind=FPGA
fabric=XCU250
interface=XRT_XDMA
declared_engine_count=2
observed_engine_count=1
unresolved_engine_count=1
memory_profile_count=1
operation_capability_count=0
lowering_authorized=false
parent_material_parity_open=true
execution_engine_parity_open=false
claim_ready=false
```

`parent_material_parity_open=true` records lineage. It does not skip the new
object's own order of evidence.

## Required Negative Tests

The executable and native gate must refuse or contain:

- a CPU or GPU kind transplanted onto the U250 blueprint;
- XRT represented as an ISA;
- XCU250 represented as an operation;
- shell readiness promoted to kernel correctness;
- DDR capacity promoted to lowering authorization;
- one physical identity promoted to a complete two-card fleet;
- a C++ or external-LLM result promoted to semantic authority;
- a Python or Rust oracle before process launch;
- a non-zero operation capability count without a frozen operation receipt;
- `PARITY_OPEN` or `CLAIM_READY` promotion before this semantics freeze.

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

This seed establishes `GARDEN`. The next commit must contain the first Sounio
executable and its expected result. A later commit may freeze that semantics by
hash. No FPGA kernel is launched in either transition.

## Connections

- [`2026-08-28-pireus-u250-dual-card-admission.md`](../../docs/internal/garden/seeds/2026-08-28-pireus-u250-dual-card-admission.md)
- [`execution_engine.sio`](../../stdlib/hardware/pireus/execution_engine.sio)
- [`u250_material_ingestion.sio`](../../stdlib/hardware/pireus/u250_material_ingestion.sio)
- [`hardware/fpga/u250_catastrophe_scan/`](../../hardware/fpga/u250_catastrophe_scan/)
