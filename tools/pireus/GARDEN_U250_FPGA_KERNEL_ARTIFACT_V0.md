# Pireus: U250 FPGA Kernel Artifact v0

> **Status**: Garden | **Date**: 2026-08-28 | **Authority**: Sounio-first

## Question

What is the smallest executable Pireus object that can name a real U250
kernel surface without claiming that a bitstream exists, that the kernel ran,
or that an operation is correct?

## Object

```text
FpgaKernelArtifact(bits, abi, engine)
```

- `bits` is the identity of a sealed `.xclbin`. It is absent in v0 because no
  repository `.xclbin` or `.xo` is available for admission.
- `abi` is a Sounio-authored declaration of the ordered XRT kernel arguments.
- `engine` is the observed U250 engine slot admitted by the frozen parent
  execution-engine semantics.

The first artifact blueprint is `krnl_san_scan`, the only implementation in
`hardware/fpga/u250_catastrophe_scan/` whose checked-in source presents a full
kernel entry point and host call surface. The neighboring `krnl_census` source
is an outline and is refused as a material artifact candidate.

## ABI v0

The Sounio declaration fixes eight positions:

| Position | Name | Direction | Width | Interface |
|---:|---|---|---:|---|
| 0 | `samples` | input buffer | 512 | `m_axi:gmem0` |
| 1 | `lut` | input buffer | 64 | `m_axi:gmem1` |
| 2 | `q_delta` | scalar input | 15 | `s_axilite` |
| 3 | `n_points` | scalar input | 32 | `s_axilite` |
| 4 | `n_samples` | scalar input | 32 | `s_axilite` |
| 5 | `hist_out` | output buffer | 32 | `m_axi:gmem1` |
| 6 | `catastrophe_out` | output buffer | 32 | `m_axi:gmem1` |
| 7 | `flops_out` | output buffer | 64 | `m_axi:gmem1` |

The table describes the checked-in boundary. It does not prove HLS synthesis,
host/kernel layout parity, kernel behavior, or scientific meaning.

## Authority Boundary

The checked-in HLS comment names a Python model as a golden model. Under the
current language-authority contract that statement is historical metadata
only. Python cannot produce or confirm Pireus semantics. C++/HLS and XRT may
later provide `MATERIAL_PARITY` receipts only after this Sounio artifact is
frozen by hash.

The v0 executable must refuse all of these promotions:

```text
source present -> bitstream present
build script present -> bitstream present
ABI declared -> ABI parity
ABI declared -> kernel correctness
XRT interface -> ISA
kernel artifact -> operation capability
kernel artifact -> legal lowering
kernel artifact -> performance or cost
parent engine parity -> artifact parity
semantics frozen without receipt -> artifact parity
Python/Rust/C++/LLM -> semantic authority
```

## Initial Result

```text
kernel_blueprint_count=1
abi_argument_count=8
input_buffer_count=2
scalar_input_count=3
output_buffer_count=3
m_axi_argument_count=5
s_axilite_argument_count=3
bitstream_present=false
abi_parity_open=false
kernel_execution_observed=false
kernel_correctness_present=false
operation_capability_count=0
lowering_authorized=false
performance_present=false
artifact_parity_open=false
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

This lane ends at `SEMANTICS_FROZEN`. A later material ingestor may hash an
actual `.xclbin`, inspect its kernel metadata, compare the ABI, and open parity.
No build, FPGA programming, or kernel launch belongs to v0.
