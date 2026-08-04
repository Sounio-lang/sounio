# Target-23 proof-carrying scaled Taylor-16 step

Date: 2026-08-03

## Result

This checkpoint turns the previously certified target-23 leaf-331 Picard box
into one bounded Taylor step that can be evaluated by an AMD U250 kernel. The
Taylor coefficients are step-scaled as `b_k = a_k h^k`, with `h = 2^-8`, so
orders through 16 remain representable in signed S1.I31.F96 fixed point.

For the accepted frozen center, an independent exact-rational verifier proves:

- maximum absolute order-16 Lagrange remainder endpoint:
  `50104134 * 2^-96`, approximately `6.3240308004e-22`;
- maximum next-state interval width:
  `100184611 * 2^-96`, approximately `1.2645075668e-21`;
- one accepted case and two fail-closed cases, with statuses `[1, -4, -1]`;
- exact reconstruction of all 459 signed 128-bit output words;
- rejection of 19 receipt mutations spanning inputs, coefficients, remainder,
  next state, CSim, CSynth, routed-link evidence, and physical execution.

For the widest component, the raw next-state width decomposes exactly as
`100184611 = 100184593 + 18`: order-16 remainder-enclosure width plus the
directed-rounding width accumulated while evaluating the degree-15 polynomial.

## Why the remainder is rigorous

The enclosure does not treat the order-16 coefficient as the first term of an
infinite Taylor-series tail. Taylor's componentwise Lagrange theorem gives

`x_i(h) = sum(k=0..15, h^k*x_i^(k)(0)/k!) + h^16*x_i^(16)(xi_i)/16!`

for some `xi_i` in `[0,h]`. The predecessor strict Picard self-map keeps every
trajectory state `x(xi_i)` in the certified box. Evaluating the normalized
order-16 autonomous-flow derivative over the whole box therefore encloses each
component's Lagrange remainder. XAI/Grok 4.3 and Z.AI/GLM-5.2 independently
accepted this corrected argument and the scaled Cauchy-product recurrences.

## Machine evidence

Vitis HLS 2025.1 CSim on VM100 matched all `459/459` output words against the
exact-rational oracle. CSynth for `xcu250-figd2104-2L-e` generated RTL with:

| Measure | Result |
|---|---:|
| Target clock | 4.00 ns |
| Estimated clock | 2.920 ns |
| Estimated Fmax | 342.47 MHz |
| Maximum latency, 3 cases | 42271 cycles / 169.084 us |
| BRAM_18K | 19 |
| DSP | 1350 |
| FF | 152148 |
| LUT | 88020 |
| URAM | 0 |

The remaining DSP count is the direct cost of exact 128-by-128-bit interval
endpoint products with directed rounding. Sharing the center/box recurrence
instance reduced DSP use from 2150 to 1350 while preserving all 459 words; this
kernel still favors auditable arithmetic over throughput.

The full hardware link requested 200 MHz and generated a 56,974,000-byte
xclbin (`sha256:15e73b5...24c75`). The request did not close at 200 MHz: Vitis
automatically reduced the kernel clock to 102.9 MHz. The routed design at that
effective frequency has WNS `+0.020 ns`, TNS `0`, and meets its adjusted timing
constraints. The earlier accidental 500 MHz attempt failed with post-placement
WNS `-2.486 ns` and congestion; that negative result is retained.

The physical U250 on `dl380-proxmox`, at `0000:d8:00.1`, runs the
`xilinx_u250_gen3x16_xdma_shell_4_1` shell and was ready under XRT 2.23.0. The
native XRT host loaded the xclbin and executed all three cases. All `459/459`
signed 128-bit output words matched the exact-rational oracle, with zero
mismatches. Thus physical execution is established for the auto-scaled
102.9 MHz image; 200 MHz closure is not established.

## Evidence boundary

This result certifies one fixed center step under the frozen predecessor Picard
precondition. It does not certify a full orbit, all points of leaf 331, all 331
adaptive leaves, global H-PG, V7-B eligibility, novelty priority, promotion, or
an open-problem solution.

The next mathematical bottleneck is compositional: carry the narrow successor
box forward while controlling interval dependency and event isolation over many
steps. The next machine bottleneck is to pipeline or further share the wide
multipliers so that 200 MHz closes without changing a single directed-rounding
bit. The retained physical result gives that optimization a bit-exact oracle.
