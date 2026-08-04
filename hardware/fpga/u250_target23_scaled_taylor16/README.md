# Target-23 scaled Taylor-16 HLS kernel

This kernel reconstructs a frozen S1.I31.F96 step-scaled Taylor transcript for
target-23 leaf 331. It emits center coefficients through degree 15, a rigorous
componentwise Lagrange remainder enclosure from degree 16 over the certified
Picard box, the polynomial enclosure, and the enclosed next state.

`testbench.cpp` compares all 459 raw words with an independently verified exact
rational oracle. CSim matched all words and synthesis generated RTL. The linked
xclbin was then executed on the physical U250 at `0000:d8:00.1`; `host.cpp`
observed the same `459/459` words with zero mismatches.

`build_xclbin.sh` targets the U250 XDMA 4.1 platform and fixes the two AXI
masters to DDR banks 0 and 1. `host.cpp` uses the native XRT C++ API and compares
all physical-card outputs byte for byte with the frozen oracle.

The link requested 200 MHz, but Vitis automatically scaled the kernel clock to
102.9 MHz. The routed image met its adjusted constraints with WNS `+0.020 ns`.
Physical execution therefore establishes the bounded one-step artifact at the
effective 102.9 MHz clock. It does not establish 200 MHz closure, a full orbit,
a whole leaf, global H-PG, novelty priority, or an open-problem solution.
