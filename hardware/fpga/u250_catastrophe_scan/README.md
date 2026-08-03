# U250 catastrophe-scan accelerator — reference implementation outline

**Status:** outline only (2026-07-26), pre-hardware. Not synthesized, not simulated with RTL tools.
**Spec:** `docs/research/u250_catastrophe_scan_fpga_spec_2026-07-26.md`
**Executable contract (CI-gated):** `scripts/research/fpga_census_kernel_model.c` via `scripts/ci/fpga_catastrophe_scan_gate.sh`

## Files

- `krnl_census.cpp` — Vitis HLS kernel outline (Phase 1 census engine, 16 PEs, II=1 target).
- `host.cpp` — XRT/OpenCL host outline (sign-table build + pack, DMA, verify vs law and CPU model).

## Intended build flow (when the U250s arrive)

```bash
# C-simulation against the gated model semantics, then C/RTL co-sim:
vitis_hls -f run_hls.tcl          # csynth + cosim (to be written at first synthesis)
v++ -c -k krnl_census --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 krnl_census.cpp
v++ -l -o krnl_census.xclbin krnl_census.xo --platform xilinx_u250_gen3x16_xdma_4_1_202210_1
g++ -O2 host.cpp -o host -lxrt_coreutil
./host krnl_census.xclbin
```

Acceptance: kernel census must equal the CI-gated C model at levels b = 4..9
(growth law `Z(b)`, nullity histograms, fiber counts), and measured
throughput must be reported back into the spec's benchmark table.
