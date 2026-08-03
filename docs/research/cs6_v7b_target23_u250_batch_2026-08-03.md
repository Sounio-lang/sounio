# CS6 V7-B target-23 U250 batch checkpoint

**Status:** retained hardware-execution checkpoint over the 331 frozen target-23
center coordinates. This checkpoint does not establish a leaf-wide interval
certificate, a full H-PG bridge, V7-B eligibility, novelty, priority, or a
solution to an open problem.

## Question tested

Can the frozen target-23 pointwise replay be expressed as a deterministic
fixed-point accelerator, built on VM100 for the U250 shell, executed on the
physical U250 in the DL380, and bound back to the retained numerical evidence
without silently changing the scientific predicate?

The pre-execution contract fixed signed Q24.40 arithmetic, classical RK4 with
step `2^-10`, 24 event bisections, 331 inputs, and 2,648 output words. Acceptance
required bit-exact agreement with the independent Python integer model, two
section events for every orbit, negative pointwise determinants, and containment
in both retained CAPD intervals.

## Physical execution

- Builder: VM100 `vitis-u250-builder`, Vitis/Vivado 2025.1.
- Executor: AMD Alveo U250 physically installed in `dl380-proxmox`.
- Shell: `xilinx_u250_gen3x16_xdma_shell_4_1`, PCIe Gen3x16.
- XCLBIN SHA-256: `682165fa43a2ce609f0d1f3f81a91b13b101db022ab57f081cb8cb999e5b349c`.
- Executed XCLBIN UUID: `5909daaf-bf72-a01e-ba62-a15e29b6274b`.
- Prior baseline UUID: `13259b30-d0d2-d4db-deba-bfc0153a26d2`; therefore this
  was not a replay of the pre-existing image.

The routed kernel used 14,264 LUTs, 16,482 registers, 12 BRAMs, and 496 DSPs.
The implementation met timing at the platform-selected 300 MHz with routed
WNS `+0.057 ns`, TNS `0`, WHS `+0.010 ns`, and THS `0`. The contract requested
250 MHz; the linked shell selected and reported 300 MHz, so the receipt records
the implemented frequency rather than rewriting the pre-execution request.

An earlier synthesis shape allowed HLS to replicate the 24 event-bisection
iterations and estimated 8,752 DSPs, which could not fit one U250 SLR. Adding an
explicit serial-unroll constraint reduced the final routed kernel to 496 DSPs.
This is a machine-level engineering result: bounded event refinement must be
made structurally serial when the compiler's default parallel expansion exceeds
the physical resource envelope.

## Results

The host executed 20 timed repetitions. Mean kernel time was `0.449751859 s`
per 331-orbit batch, or `735.961383` orbits/s. All 2,648 returned words matched
the Python Q24.40 model bit for bit. All 331 orbits produced the required two
events, all 331 pointwise determinants were negative, and all 331 results were
inside both retained CAPD intervals. The largest absolute difference from the
decimal replay was `1.9909275908247997e-18`; the smallest retained CAPD margin
was `5.0702387543190643e-15`.

Thirty-eight electrical samples were captured while the kernel marker was
active. Their maximum was `29.570838 W`; the reallocated pre-run baseline was
`32.334828 W`. These asynchronous board-level sensor readings are retained for
audit only. They do not support an energy-efficiency or causal power-reduction
claim.

## Independent interval rechecks

Slurm job `8559` recomputed the three frozen minimum-margin center cases with
Python-FLINT/Arb at 256-bit precision. Each Arb interval was strictly negative,
independently certifying the sign for that validated center orbit.

The U250 determinant is not inside these extremely narrow Arb intervals. The
separations are approximately `1.64e-18`, `1.58e-18`, and `1.53e-18`, consistent
with the fixed-point/discretization scale and far smaller than the retained CAPD
margins. The receipt records this non-containment as `false`. Arb and U250 thus
provide complementary evidence; they are not combined into a rigorous FPGA
interval certificate.

Jobs `8556` and `8558` failed before mathematical execution due, respectively,
to truncated expected hashes and an invalid wheel filename. Job `8559` used the
full hashes and a valid platform wheel name and completed successfully in 53 s.

## Audit surface

The durable receipt is
`scripts/research/receipts/cs6_v7b_target23_u250_batch_v1/`. It includes the
bit-exact raw output, per-leaf tables, power/thermal state, baseline device state,
VM100 build reports, Arb transcripts, and a SHA-256 manifest. Run:

```bash
bash scripts/research/cs6_v7b_target23_u250_batch_gate.sh
```

The gate regenerates the integer reference, compares it to the retained table,
checks the complete hardware receipt, and requires rejection of six critical
mutations. The retained run passes all stages.

## Evidence boundary and next step

This checkpoint shows that the frozen pointwise workload can be executed
reproducibly on the U250 without loss relative to its specified Q24.40 model. It
does not propagate intervals through the hardware integrator, certify whole
leaves, or close the full H-PG bridge.

The next research step is an outward-rounded interval or affine-arithmetic
kernel whose hardware outputs are themselves enclosures. Only then can FPGA
execution contribute directly to leaf-wide certification instead of serving as
a fast, bit-exact pointwise replay engine.
