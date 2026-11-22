<!-- docs:meta
topic_id: repo.docs.decisions.adr-022-fpga-backend-mvp
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-022-fpga-backend-mvp
-->

# ADR-022: FPGA backend MVP — profile subset, kernel IR, first kernel

**Status**: experimental
**Date**: 2026-10-26
**Related**: ADR-021 (oracle stack for this backend), ADR-009,
ADR-008, `docs/research/san_imagenet_fpga_dl380_spec_2026-08-02.md`

---

## Context

ADR-021 fixed who may judge a generated FPGA kernel before any
lowering exists. This ADR fixes what the first backend accepts,
what it lowers through, and which kernel proves the path end to
end. The U250 kernel in `hardware/fpga/u250_catastrophe_scan/` is
hand-written HLS; nothing in `souc` targets an FPGA today.

A prototype already exercises the path in this branch:
`examples/fpga/san_scan_kernel.sio` (valid Sounio, self-checking
under `souc run`) is translated by `tools/fpga/sio_fpga_emit.py`
to HLS C++, compiled natively (no Xilinx toolchain on this node),
and matched bit-exactly against the Futhark reference on a
200-sample random cohort by `scripts/ci/fpga_prototype_gate.sh`.

## Decision

### 1. FPGA profile subset (v0)

A kernel is one function in the following subset. Anything outside
it is a compile-time rejection, never a silent fallback.

- Types: `i64`, `bool`; static arrays `[i64; N]` and
  `[[i64; K]; N]`. (`u32`/`u64` and narrower widths arrive with the
  real width system; v0 kernels speak `i64` and the ABI packs.)
- Control: bounded `for x in A..B`, `if`/`else`, comparisons.
- State: `var`/`let` scalars, array element reads and writes.
- No heap, no recursion, no function calls, no strings, no floats.
- One trailing bare expression is the return value.
- Geometry (array sizes, loop bounds) is static in the source.

### 2. Kernel IR

The backend lowers through a small kernel IR in the spirit of
`self-hosted/gpu/kernel_ir.sio`, extended with an `Fpga` target
kind. The IR must represent, explicitly: kernel arguments with
their ABI class, static array geometry, the pipeline loop and its
target II, and on-chip vs off-chip storage for every array. If a
fact is not in the IR, the emitter may not invent it.

### 3. ABI rules (v0)

- Array parameters: `m_axi`. Scalar parameters: `s_axilite`.
  Return value: `s_axilite port=return`.
- Output arrays are passed as parameters (e.g. `hist`), never
  returned. Packing into bus beats and DDR bank assignment are
  ABI concerns, judged by the `sounio_closed_form_twin` of
  ADR-021 §3, not by the Futhark reference.
- The first loop over samples carries `#pragma HLS PIPELINE II=1`.

### 4. First kernel: the SAN scan

The first generated kernel is the catastrophe scan itself
(`examples/fpga/san_scan_kernel.sio`), because its semantics
already have a promoted `verified_foreign_reference`
(`scripts/research/san_scan_reference.fut`, ADR-021 steps 2–4) and
a hand-written HLS counterpart to diverge from honestly.

### 5. Prototype and its limits

`tools/fpga/sio_fpga_emit.py` is prototype tooling (ADR-009: it
judges nothing). It translates the subset above line by line and
aborts on any unrecognised construct. Known prototype limits,
accepted for the MVP and recorded here so they are not mistaken
for the design:

- `hist` is read-modify-written through `m_axi` per sample. The
  hand-written kernel keeps per-lane private histograms on chip
  and writes totals in an epilogue; the real backend must do the
  same (the IR storage fact in §2 exists for this).
- Validation on this node is a native g++ build of the generated
  code (a csim-equivalent), not Vitis csim/cosim, and never a
  bitstream. On-target validation stays with the DL380 flow.
- One kernel, one geometry family (7 intermediate points,
  256-sample static bound) — the shape of the SAN scan, not a
  general compiler yet.

Promotion path: the emitter's translation rules move into
`self-hosted/` behind a `--target fpga-u250` mode of `souc` only
after a second, differently-shaped kernel survives the same gate
without emitter changes. Until then the prototype lives in
`tools/fpga/` and the gates say so.

## Acceptance

- `examples/fpga/san_scan_kernel.sio` compiles and self-checks
  under `souc run` (`SAN_SCAN_KERNEL_SIO_PASS`).
- `scripts/ci/fpga_prototype_gate.sh` prints `FPGA_PROTOTYPE_PASS`:
  generated kernel == Futhark reference, bit-exact, 200-sample
  cohort (seed 220307). Measured in this branch:
  `hist=[80, 53, 27, 17, 12, 4, 1, 6]`, `macs=47400`.
- A construct outside §1 (e.g. a `while` loop or a call) makes the
  emitter fail with `SIO_FPGA_EMIT_FAIL`, not with generated code.

## Consequences

- The subset is deliberately smaller than Sounio. Kernels that
  need more must wait for a subset revision in a new ADR, not for
  emitter special cases.
- Index and registry entries for this ADR are added when the
  stacked PRs (#3232–#3234) land, to avoid index churn across the
  stack; the file number 022 follows ADR-021 directly.

## Grounded in

- ADR-021 (oracle stack, ABI twin, gate names `fpga_kernel_parity`
  — this prototype gate is its first, native-csim form).
- `hardware/fpga/u250_catastrophe_scan/krnl_san_scan.cpp` — the
  hand-written target shape (II=1, per-lane histograms, epilogue).
- `self-hosted/gpu/kernel_ir.sio` — IR precedent and target enum.
- Prototype measurements recorded in Acceptance above.
