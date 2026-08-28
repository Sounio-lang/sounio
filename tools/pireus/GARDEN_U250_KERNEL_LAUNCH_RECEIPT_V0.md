# Garden: U250 Kernel Launch Receipt v0

Status: `GARDEN`

This Garden defines the next Pireus admission boundary after
`XclbinMaterialReceipt`: observe one deliberately small launch of the recovered
`krnl_san_scan` bitstream on the canonical DL380 U250 without treating returned
numbers as a correctness oracle.

## Authority order

The only admissible order is:

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY`

For this receipt, `CLAIM_READY` remains closed. Sounio owns the first executable
representation of the invocation and the acceptance result. C++ may execute the
frozen invocation through XRT and emit material facts only.

## Parent admission

The parent `XclbinMaterialReceipt` must already have `artifact_parity_open=true`
for the exact recovered material:

- artifact: `krnl_san_scan.hw.xclbin`
- SHA-256: `d30078c7b2e8690aef892b4b6cf96af0f490b70e2b367e5e3679be04fcd4bdbf`
- XCLBIN UUID: `c50267ec-ae68-48a1-1559-3473f046689c`
- kernel symbol: `krnl_san_scan`
- platform: `xilinx_u250_gen3x16_xdma_4_1_202210_1`

A launch observation for any other identity is refused.

## Frozen invocation

Sounio defines this invocation before a C++ launch probe may exist:

| Field | Value |
| --- | ---: |
| device BDF | `0000:d8:00.1` |
| card serial | `22000321B01F` |
| `n_samples` | `4` |
| `n_points` | `2` |
| `q_delta` | `16384` |
| packed sample beats | `1` |
| sample beat bytes | `64` |
| sample payload | all zero bits |
| LUT entries | `8192, 16384, 0, 0, 0, 0, 0, 0` |
| histogram output bytes | `32` |
| catastrophe output bytes | `4` |
| FLOP output bytes | `8` |

The values returned in the three output buffers are observations, not expected
results. This receipt neither compares nor interprets them.

## Required material observations

The C++ parity probe must emit all of these facts after the Sounio semantics are
frozen:

1. exact artifact digest observed;
2. exact device BDF and card serial observed;
3. XRT version observed;
4. device programmed with the exact XCLBIN UUID;
5. kernel symbol opened;
6. five buffers allocated using the kernel argument group IDs;
7. input and LUT buffers synchronized to the device;
8. run submitted and completed without an XRT exception;
9. all three output buffers synchronized back to the host;
10. returned output widths and values recorded without a semantic verdict.

Sounio may open `execution_parity_open` only when every required observation and
every frozen invocation field matches.

## Refusals

The receipt fails closed when any of these is true:

- the parent material receipt is invalid or its freeze does not match;
- execution is attempted before the Sounio freeze;
- producer language or role is not the one allowed for the current phase;
- artifact, device, card, kernel, invocation, buffer width, or lifecycle fact
  differs from the frozen contract;
- the child emits a semantic verdict or expected numerical output;
- kernel correctness, operation capability, lowering, performance, speedup, ISA,
  or `CLAIM_READY` is requested;
- Guardian policy is absent, errors, or times out;
- Python or Rust is proposed as an oracle;
- an external LLM or a parity language is promoted to semantic authority.

## Receipt boundary

An accepted receipt proves only:

> The exact recovered bitstream was programmed on the identified U250, the
> frozen kernel invocation was submitted, it completed, and its output buffers
> were returned to the host.

It does not prove that the returned values are correct, that the kernel is fast,
that a Sounio lowering produced the bitstream, or that the U250 supports any
general operation class. Those require later, separately frozen receipts.
