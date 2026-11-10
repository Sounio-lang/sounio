<!-- docs:meta
topic_id: repo.docs.decisions.adr-021-fpga-backend-oracle-stack
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-021-fpga-backend-oracle-stack
-->

# ADR-021: Oracle stack for the Sounio-to-FPGA backend (U250)

**Status**: experimental
**Date**: 2026-10-26
**Related**: ADR-008 (single semantic clock), ADR-009 (verified foreign
reference oracles), ADR-006 (fixed-point seed scope),
`docs/decisions/claim_oracle_inventory.schema.md`,
`hardware/fpga/u250_catastrophe_scan/README.md`

---

## Context

The repo already proves one hand-written path to the AMD Alveo U250:
`hardware/fpga/u250_catastrophe_scan/krnl_san_scan.cpp` is a
synthesis-ready Vitis HLS kernel (250 MHz, `xcu250-figd2104-2L-e`,
II=1 pipeline, `m_axi`/`s_axilite` ABI) with an XRT host, reported
bit-exact on the DL380/U250 (`SAN_DL380_T3_VERDICT T3_GREEN`).

That proof does not generalise. No `.sio` source lowers to a kernel
today:

- The only hardware emitter in the compiler,
  `self-hosted/compiler/codegen/hardware/kaxi_emitter.sio`, extracts
  fixed templates from `stdlib/hardware/kaxi.sio`; it does not lower
  arbitrary Sounio code.
- The kernel IR that exists (`self-hosted/gpu/kernel_ir.sio`) targets
  PTX, Metal and SPIR-V only. There is no FPGA target in `souc`.
- The Pireus U250 layer (`stdlib/hardware/pireus/`) is an admission
  ontology for externally produced artifacts. Its freeze record states
  `BLUEPRINT_ONLY` with `lowering_authorized=false`,
  `bitstream_present=false`, `kernel_execution_observed=false`. It
  certifies descriptions of kernels; it neither generates nor
  executes them.

A real backend (`souc --target fpga-u250`) therefore needs its oracle
stack decided *before* the lowering exists, or every generated kernel
will inherit whichever judge happens to be nearby. The nearby judges
are currently wrong for that role: the SAN scan contract
(`scripts/research/san_imagenet_fpga_dl380.py`, class
`U250SanScanModel`) is Python, and ADR-009 excludes Python from claim
authority without exception.

## Decision

### 1. Oracle stack for FPGA kernel claims

Claims are separated into three kinds. Each kind has exactly one
authority class; mixing them is the drift ADR-008 forbids.

| Claim kind | Authority (`oracle_class`) | Notes |
|---|---|---|
| Kernel semantics (outputs bit-exact for given inputs) | `verified_foreign_reference` — **Futhark first** (see §2) | Twin remains the fallback where Futhark cannot express the claim (§3) |
| Provable properties (accumulator bounds, no-overflow, packing injectivity) | `formal_only` today (Lean axis); F* admissible later under ADR-009 §1 | A proof is not a numeric expected-value source (ADR-008) |
| Performance and resources (throughput, LUT/FF/BRAM, Fmax, power) | none — measurement only | Never a semantic clock; reported, never hard-fail on semantics |

### 2. Futhark-first for data-parallel kernel semantics

For kernels in the FPGA profile (fixed-width integers, static arrays,
bounded loops, no heap), the primary semantic reference is a
**Futhark** implementation, admitted per-kernel as a
`verified_foreign_reference` under ADR-009 §1 only when all four
criteria hold:

1. **Independently authored** from the kernel specification (the
   `.sio` kernel contract and its packing rules), not transliterated
   from the generated HLS C++.
2. **Pure and statically typed** — Futhark satisfies ADR-009 §2 for
   the GPU/data-parallel numeric domain; the same purity argument
   covers the FPGA data-parallel subset.
3. **Pinned toolchain** — Futhark version recorded in the gate script
   and in `artifacts/audit/claim_oracle_inventory.tsv`. First pin:
   Futhark 0.25.27 (git `824e46bfd9fbf240389e626981caf39a51f5e980`),
   the version installed in the workspace. The Vitis/XRT versions
   used for synthesis are pinned alongside it in the same gate.
4. **Insufficiency rationale** — recorded per kernel. The standing
   rationale for this backend: a `sounio_closed_form_twin` pair would
   share the same packing, quantisation (e.g. Q0.15) and
   fixed-point misunderstanding, because both paths would be written
   against the same Sounio-side contract. Self-agreement is not
   independence for exactly the decisions an FPGA kernel makes
   silently (field packing, saturation, exit priority).

Until a per-kernel Futhark reference meets all four criteria, that
kernel has **no** foreign claim clock: its gate stays
`research_harness` or `sounio_closed_form_twin`, and no measurement
on the card may be promoted to a correctness claim.

Futhark-first is a statement about authoring order for the MVP, not
a waiver of ADR-009 §4: the reference is written from the spec
first, and the generated kernel is judged against it.

### 3. Where Futhark does not reach

- **ABI and packing** (which argument becomes `m_axi` vs `s_axilite`,
  field offsets in a 512-bit beat, DDR bank assignment) are not
  data-parallel semantics. They are checked by a
  `sounio_closed_form_twin`: a Sounio-side packer and an independent
  Sounio-side unpacker/verifier must agree, plus boundary sentinels
  in Sounio (`sounio_native_expected`).
- **Epistemic sideband behaviour** (K-AXI provenance/fidelity
  counters) stays Sounio-owned: `stdlib/hardware/kaxi.sio` is the
  source of truth, so its claims remain `sounio_native_expected`.
  Icarus/Yosys simulation and synthesis of the emitted templates are
  measurement and corroboration, not a second semantic authority.
- **HLS C++ testbenches** (e.g. `tb_san_scan.cpp`) are part of the
  Vitis harness. C++23 is admitted by ADR-009 only for the bootstrap
  oracle role; that admission does not extend to kernel semantics,
  so testbenches corroborate and debug, they do not judge claims.

### 4. Classification of existing gates

Provisional classes follow `artifacts/audit/claim_oracle_inventory.tsv`
(2026-08-31 scan) as amended by direct inspection for this ADR.

| Gate / artifact | Class today | Disposition |
|---|---|---|
| `scripts/ci/san_imagenet_fpga_dl380_gate.sh` (Python model is the bit-accuracy judge, clause I6) | `research_harness` in inventory; **would be `forbidden_as_claim_oracle` if cited as a backend claim clock** | **Migrate first** (§5). Python demoted to `external_corroboration_only` |
| `scripts/ci/fpga_catastrophe_scan_gate.sh` (C census model) | `research_harness` | Keep as research. Census HLS is outline-only; no claim clock until a kernel exists |
| `scripts/ci/fpga_receipt_staleness_gate.sh` | `research_harness` (meta-guard) | Keep unchanged |
| Pireus U250 artifact executable + freeze (`u250_fpga_kernel_artifact`) | `sounio_native_expected` for the *blueprint* claim only | Keep. Must never be cited as execution, correctness or lowering evidence; its own freeze says so |
| K-AXI gates (`kretikos_kaxi_*`, `semantic_orc_*kaxi*`) | mixed `sounio_native_expected` / `research_harness` | Keep. FPGA fabric claims inherit §3, second bullet |
| `scripts/research/san_dl380_t3_acceptance.py` (`T3_GREEN`) | measurement receipt | Keep as on-target measurement record. It is evidence a card run happened, not a reusable semantic oracle |

### 5. Migration plan (SAN scan first)

1. **Demote.** Reclassify the Python SAN model in the inventory via
   `claim_oracle_inventory.overrides.tsv` to
   `external_corroboration_only`, `migration=promote_verified_foreign`.
   The gate keeps running it as corroboration; it stops being the
   bit-accuracy judge.
2. **Author the reference.** Write the SAN scan in Futhark from
   `docs/research/san_imagenet_fpga_dl380_spec_2026-08-02.md`
   (first-exit priority over Q0.15 confidences, per-lane histograms,
   prefix-MAC LUT), independently of `krnl_san_scan.cpp`.
3. **Pin.** Record Futhark 0.25.27 and the Vitis/XRT versions in
   the gate script and the inventory row.
4. **Promote.** Add the override row: `verified_foreign_reference`,
   `foreign_hard_fail=yes`, `foreign_runtimes=futhark`. From this
   point the Futhark reference — not Python, not the card — decides
   bit-accuracy claims for the scan.
5. **Extend before generalising.** The first generated (non-manual)
   kernel reuses this exact pattern: spec, Futhark reference, pin,
   promote. A kernel without step 2 does not ship with a correctness
   claim, only with measurements.

### 6. New gates the backend MVP must add

| Gate | Class | Judges |
|---|---|---|
| `fpga_kernel_parity` (generated kernel csim/cosim vs Futhark reference) | `verified_foreign_reference` | Kernel semantics |
| `fpga_abi_packing_twin` (Sounio packer vs independent Sounio verifier) | `sounio_closed_form_twin` | ABI, packing, bank assignment |
| `fpga_hw_measurement` (bitstream run on U250: throughput, resources, power) | `research_harness` | Nothing — report only |

## Consequences

- The SAN gate's current Python judge is a known ADR-009 tension,
  not a precedent. §5 step 1 is the cheapest correctness fix in this
  ADR and should land before any generated kernel exists.
- Futhark becomes a pinned CI toolchain for the FPGA backend only.
  This does not change its GPU-domain status under ADR-009 and does
  not admit any other foreign runtime; Python and Rust remain
  excluded as claim oracles everywhere in this backend.
- `sounio_closed_form_twin` is not displaced where it is the honest
  tool (ABI/packing, epistemic sideband). Futhark-first applies to
  data-parallel kernel semantics, the domain ADR-009 maps to it.
- Performance numbers from the U250 (including the reported
  ~24 Msamples/s scan rate) remain measurements. They may motivate
  work; they may not close a correctness claim.
- Inventory scanner support for auto-detecting Futhark references is
  out of scope; promotion stays manual/override-driven per the
  schema note in `claim_oracle_inventory.schema.md`.

## Grounded in

- ADR-008 authority paragraph and oracle-class table.
- ADR-009 §1 (four admission criteria), §2 (Futhark for
  data-parallel numeric kernels), §4 (new-work rule), and the hard
  exclusion of Python and Rust.
- `hardware/fpga/u250_catastrophe_scan/README.md` — synthesis-ready
  kernel, on-target `T3_GREEN`, and the Python contract it cites.
- `tools/pireus/u250_fpga_kernel_artifact.freeze.v0` —
  `BLUEPRINT_ONLY`, `lowering_authorized=false`.
- `self-hosted/compiler/codegen/hardware/kaxi_emitter.sio` (10
  lines, template extraction only) and
  `self-hosted/gpu/kernel_ir.sio` (PTX/Metal/SPIR-V targets only) —
  the absence this ADR prepares for.
- Futhark 0.25.27 present in the workspace
  (`futhark --version`, git `824e46bfd9fbf240389e626981caf39a51f5e980`).
