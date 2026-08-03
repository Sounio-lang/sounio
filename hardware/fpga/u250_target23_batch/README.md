# Target-23 U250 batch propagator

This directory contains the hardware half of the target-23 hybrid scout:

- `kernel.cpp`: Q24.40 fixed-point RK4 propagation of 331 center orbits,
  including two oriented `w=0` event localizations.
- `host.cpp`: XRT launcher, bit-exact comparison against an independently
  generated CPU transcript, determinant reconstruction, and measurement.
- `testbench.cpp` / `run_hls_csim.tcl`: full 331-orbit C simulation compared
  word-for-word with the independent transcript before synthesis.
- `build.sh`: Vitis 2025.1 U250 compile/link flow for VM100.
- `target23_batch.cfg`: DDR placement and 250 MHz target.

The FPGA output is a pointwise falsification and ranking surface. It is not an
interval integrator and does not replace the retained Arb certificate.
