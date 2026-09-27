# U250 catastrophe-scan accelerator — reference implementation outline

**Status:** SAN scan kernel **synthesis-ready v2** (2026-08-02): complete
SIMD-4 HLS kernel + testbench + HLS/v++ flow + XRT host, packing design
validated by a stdint smoke test (`SMOKE_PASS`, 100003-sample cohort,
bit-exact vs independent golden). Census kernel remains outline-only.
**DL380 on-target T3 acceptance: `SAN_DL380_T3_VERDICT T3_GREEN`** (spec §13).
**Spec:** `docs/research/u250_catastrophe_scan_fpga_spec_2026-07-26.md` (census),
`docs/research/san_imagenet_fpga_dl380_spec_2026-08-02.md` (SAN scan)
**Executable contract (CI-gated):** `scripts/research/fpga_census_kernel_model.c` via `scripts/ci/fpga_catastrophe_scan_gate.sh`

## Files

- `krnl_census.cpp` — Vitis HLS kernel outline (Phase 1 census engine, 16 PEs, II=1 target).
- `host.cpp` — XRT/OpenCL host outline (sign-table build + pack, DMA, verify vs law and CPU model).
- `krnl_san_scan.cpp` — **synthesis-ready v2** SAN catastrophe-scan + FLOP-metering kernel: per-sample first-exit priority encode over Q0.15 confidences packed 7×15-bit per 128-bit record, 4 samples per 512-bit beat (II=1), per-lane private histograms, exact prefix MAC accumulation from a host-loaded stage-cost LUT; one bitstream serves SAN-ResNet-50 and SAN-ViT-large (LUT + scalars reload).
- `tb_san_scan.cpp` — csim/cosim testbench: independent golden, boundary cases (`conf == q_delta` settles, all-below → catastrophe, tail beats), random cohorts at both trunk geometries.
- `run_hls_san_scan.tcl` — `vitis_hls` flow (csim → csynth → cosim), part `xcu250-figd2104-2L-e`, 250 MHz.
- `build_san_scan_xclbin.sh` — v++ compile+link for `xilinx_u250_gen3x16_xdma_4_1_202210_1` (targets hw / hw_emu / sw_emu).
- `host_san_scan.cpp` — complete XRT-native host: loads the exported T3 artifacts (`artifacts/san_dl380_t3/`), packs cohorts, runs the card, verifies hist/catastrophes/MACs **bit-exactly** against the control VM (spec T3), reports measured throughput.
- `host_san_scan_e2e.cpp` — stdin-to-card host for the Python-orchestrated end-to-end loop. Reads a packed cohort from stdin (binary header + 512-bit beats), runs `krnl_san_scan`, and reports decomposed timing: setup, DMA H2D, kernel, DMA D2H, total.
- `scripts/research/san_fpga_endtoend.py` — end-to-end orchestration: train/load SAN-ResNet-18 on ImageNette2-160, run PyTorch forward, quantize confidences to Q0.15, pack into the same 512-bit beats as `host_san_scan`, stream to `host_san_scan_e2e`, and validate the card output bit-exactly against a Python golden scan. Supports `--mock-host` for CI/offline validation.
- Gated golden model: `scripts/research/san_imagenet_fpga_dl380.py` via `scripts/ci/san_imagenet_fpga_dl380_gate.sh` (clause I6); on-target acceptance: `scripts/research/san_dl380_t3_export.py` + `san_dl380_t3_acceptance.py`.

## Intended build flow (on the DL380, Vitis 2022.1+ / XRT sourced)

```bash
cd hardware/fpga/u250_catastrophe_scan
vitis_hls -f run_hls_san_scan.tcl          # csim + csynth + cosim; expect TB_SAN_SCAN_PASS
bash build_san_scan_xclbin.sh hw_emu       # quick emulation smoke (optional)
bash build_san_scan_xclbin.sh hw           # real bitstream (~hours)
g++ -O2 -std=c++17 host_san_scan.cpp -o host_san_scan \
    -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil -lpthread
for ds in val_resnet val_vit stress_1p2M; do
  ./host_san_scan krnl_san_scan.hw.xclbin /path/to/artifacts/san_dl380_t3 $ds
done
# acceptance: HOST_SAN_SCAN_PASS for all three datasets (bit-exact vs control VM)
```

## End-to-end SAN-ImageNette flow

The Python+C++ loop connects a real SAN-ResNet-18 trunk to the U250 scan kernel:

```bash
# 1. Build the stdin host (on DL380, XRT sourced)
cd hardware/fpga/u250_catastrophe_scan
g++ -O2 -std=c++17 host_san_scan_e2e.cpp -o host_san_scan_e2e \
    -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil -lpthread

# 2. Ensure ImageNette2-160 is at datasets/imagenette2-160

# 3. Run end-to-end (CPU forward on DL380, then card scan)
cd ../..
.venv/bin/python scripts/research/san_fpga_endtoend.py \
    --xclbin hardware/fpga/u250_catastrophe_scan/krnl_san_scan.hw.xclbin
# expect: SAN_FPGA_ENDTOEND_PASS bit_exact=True

# 4. Offline / CI validation without FPGA
.venv/bin/python scripts/research/san_fpga_endtoend.py --mock-host
```

Measured on the DL380/U250 (2026-08-04, single cohort of 3 925 real ImageNette
images, CPU forward): forward ≈ 18.4 s, pack ≈ 40 ms, xclbin setup ≈ 135 ms,
DMA H2D ≈ 0.12 ms, kernel ≈ 0.66 ms, DMA D2H ≈ 0.15 ms, total ≈ 136 ms. The
xclbin setup is paid once per process; the kernel itself sustains the same
~24 Msamples/s single-shot rate reported in Table 1 for small cohorts.

Census kernel acceptance (unchanged): kernel census must equal the CI-gated
C model at levels b = 4..9 (growth law `Z(b)`, nullity histograms, fiber
counts), with measured throughput reported back into the census spec.

