# Pireus: U250 Xclbin Material Receipt v0

> **Status**: Garden | **Date**: 2026-08-28 | **Authority**: Sounio-first

## Question

What is the smallest receipt that can bind the frozen `krnl_san_scan`
blueprint to recovered U250 bits without promoting the bits to an ISA,
operation, execution result, correctness result, or performance claim?

## Object

```text
XclbinMaterialReceipt(artifact, digest, metadata, abi, engine)
```

The parent `FpgaKernelArtifact(bits, abi, engine)` semantics are frozen as:

```text
parent_semantics_sha256=e7d4a83e81c054a1d15808292d49fbcda6ea43a06dbf31469e7c4c81d51d3fe5
parent_freeze_sha256=89278d99fab89bc2b582958a27d2806775b0b13a1e8f258550924fc20e3dc05e
```

The recovered candidate is observational inventory until Sounio admits a
sealed material receipt:

```text
artifact_name=krnl_san_scan.hw.xclbin
artifact_size_bytes=41112056
artifact_sha256=d30078c7b2e8690aef892b4b6cf96af0f490b70e2b367e5e3679be04fcd4bdbf
xclbin_uuid=c50267ec-ae68-48a1-1559-3473f046689c
kernel=krnl_san_scan
platform=xilinx_u250_gen3x16_xdma_4_1_202210_1
vitis_version=2025.1
requested_frequency_mhz=250
achieved_frequency_tenths_mhz=1352
```

The artifact remains outside Git. The receipt binds its bytes by SHA-256 and
records the read-only recovery path as provenance; it does not import the
41 MB bitstream into the source tree.

## Material ABI

The receipt compares two distinct relations. An AXI interface is not a DDR
bank assignment.

| Pos | Argument | Interface | Material bank |
|---:|---|---|---|
| 0 | `samples` | `M_AXI_GMEM0` | `bank1` |
| 1 | `lut` | `M_AXI_GMEM1` | `bank1` |
| 2 | `q_delta` | `S_AXI_CONTROL` | n/a |
| 3 | `n_points` | `S_AXI_CONTROL` | n/a |
| 4 | `n_samples` | `S_AXI_CONTROL` | n/a |
| 5 | `hist_out` | `M_AXI_GMEM1` | `bank1` |
| 6 | `catastrophe_out` | `M_AXI_GMEM1` | `bank1` |
| 7 | `flops_out` | `M_AXI_GMEM1` | `bank1` |

This is consistent with the parent ABI: `samples` remains `gmem0` even though
the linker connects that port to physical `bank1`.

## Authority Boundary

Sounio defines the expected identity, comparison, refusal states, and result.
After this executable is frozen, C++ may run as `MATERIAL_PARITY` to hash the
binary and extract facts with native XRT tooling. Its output has no semantic
verdict. Python and Rust are forbidden; external LLMs remain `REVIEW_ONLY`.

The Sounio classifier must fail closed for:

```text
missing policy or policy timeout
material execution before frozen Sounio semantics
missing bits or missing digest
artifact size or digest drift
kernel, platform, UUID, toolchain, or clock drift
argument count, order, interface, or bank drift
receipt not bound to the parent semantics and freeze
C++ or external LLM promoted to SEMANTIC_AUTHORITY
Python or Rust used as an oracle
receipt promoted to ISA, operation, lowering, correctness, or performance
CLAIM_READY requested from material parity alone
```

## Expected Boundary

```text
bitstream_present=true
bitstream_digest_present=true
metadata_present=true
kernel_symbol_matches=true
platform_matches=true
abi_matches=true
material_bank_mapping_matches=true
artifact_parity_open=true
kernel_execution_observed=false
kernel_correctness_present=false
operation_capability_count=0
lowering_authorized=false
performance_present=false
claim_ready=false
```

## Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

This lane may end at `PARITY_OPEN`. It performs no HLS build, FPGA programming,
kernel launch, scientific validation, or speed measurement.
