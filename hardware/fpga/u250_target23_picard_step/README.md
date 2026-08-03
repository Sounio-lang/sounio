# Target-23 Picard-step HLS kernel

This kernel recomputes one frozen S1.I31.F96 Picard self-map transcript for
target-23 leaf 331 and refuses reversed, out-of-domain, non-strict, or
non-contracting candidates. `testbench.cpp` compares every raw output word with
vectors independently generated from exact Python rational arithmetic.

Passing CSim is not physical U250 execution and does not certify a full orbit,
a whole leaf, global H-PG, novelty, or an open problem.

`run_hls_csynth.tcl` generates RTL and a resource estimate for the U250 part;
successful synthesis still is not physical-card execution.
