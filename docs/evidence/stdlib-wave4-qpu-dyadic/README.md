# Exact dyadic scale in QPU wide counts

wide_counts now multiplies the converted unsigned high32 bits by the exact binary64 value2^-32, instead of dividing by2^32. The LCG multiplier/addend, full64-bit state, seed-zero rule, thresholds, comparisons, field projections, draw count and accumulation order are unchanged. The patch changes exactly two lines in qpu/chsh.sio.

Slurm19594 completed0. An independent C++23 executable compared binary64 bit patterns for all4294967296 unsigned32-bit inputs, using runtime volatile denominator/reciprocal and no fast-math; all matched. A deliberately adjacent wrong reciprocal was refused. This is exhaustive finite numerical concordance under ADR009, not a Lean theorem.

The native discriminator derives both kernels from the unchanged source body, with an exact two-replacement check. Fourteen cases include zero and negative-zero probabilities, zero seed, exact quarter/half/three-quarter thresholds reached by inverse-LCG seeds, nextafter boundaries, maximum high32 bits, NaNs and three10million-draw streams. Both loop-region modes agree exactly with the C++23 original-division oracle. A separately compiled one-ULP reciprocal mutant fails the above50 boundary case; its nonzero oracle status is required. No tolerances were introduced.

All three paired timings are retained. Mode0 literal/dyadic ratios:1.11635922,1.10879394,1.10541868; aggregate1.11019536. Mode1:1.0857127,1.08342837,1.0811397; aggregate1.08342726. Each pair uses the same node, raw, probabilities, seed and draw count; order alternates for the middle stream. These are isolated loop measurements, not end-to-end throughput or successful completion of the61billion-draw original.

Slurm19596 completed0 with the actual modified library and three unchanged original imported programs: test_qpu_chsh, test_born_shot_law and test_qasm_tape. All six combinations of program and loop-region mode checked0, compiled0 and executed0 with ALL_PASS, retaining original1200s compiler and30s native deadlines. Source/raw/executable manifests verified after execution. Raw identity remains8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932.

The historical3000-second slow timeouts remain unchanged. This improvement does not close the slow original, loop-checkpoint optimization, full-stdlib acceptance or a hardware-QPU claim. Complete artifact custody is recorded separately when its readback finishes.
