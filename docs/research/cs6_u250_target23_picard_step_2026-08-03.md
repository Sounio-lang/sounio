# CS6 U250 target-23 one-step Picard certificate

## Question

Can a small proof-carrying HLS kernel recompute a rigorous interval self-map
and strict contraction obligation for one real step of the target-23 vector
field, rather than merely reproduce a point trajectory?

## Frozen experiment

The input is the exact-decimal center of target-23 leaf 331,
`U08-0000000223_S09-0000000325`, rounded outward to signed S1.I31.F96
endpoints. The frozen scalar parameter is the exact decimal rational
`zs = 22.3274637391`, also rounded outward. The time interval is `[0, 2^-8]`.
The four-dimensional field is

```text
x'   = 2*y*y - x*y
y'   = x*y - y*(w + zs)/2
w'   = x*y - w - zs
ell' = x - y - (w + zs)/2 - 1
```

An exact Python `Fraction` constructor monotonically iterates
`hull(B, X0 + [0,h] F(B))` to a fixed point and then inflates every endpoint by
`2^64` raw units. The candidate is accepted only if the recomputed Picard image
is strictly inside `B` on all four axes and the sufficient contraction
condition `h ||J(B)||_infinity < 1` holds under an outward-rounded row-sum
bound.

## Result

The independent exact verifier and Vitis HLS 2025.1 CSim agree on all 88 output
words for four cases. Each 22-word output contains eight vector-field endpoint
words, eight Picard-image endpoint words, four Jacobian row-sum upper bounds,
one contraction upper bound, and one status word. The valid target-23 candidate
has:

- minimum strict self-map margin: `13362935843645108892` raw F96 units;
- contraction upper bound: `12543560845867825682829769920` raw F96 units;
- contraction upper bound as a real number: approximately `0.15832199621160117`;
- three fail-closed controls: non-strict box, reversed interval, and endpoint
  outside the frozen domain.

The HLS target is `xcu250-figd2104-2L-e`. CSim produced zero mismatches. RTL
synthesis also completed, with an estimated 2.920 ns clock period (342.47 MHz),
267-cycle latency, and estimates of 19 BRAM_18K, 50 DSP, 22,508 FF, 29,269 LUT,
and zero URAM. The requested loop constraints were not all met: the report gives
a 268-cycle top-level interval, so these are feasibility estimates rather than
a final throughput claim. Eight mutations must be rejected by the final gate.

## Exact claim boundary

This is a bounded one-step Picard self-map and contraction certificate for one
exact initial-state enclosure, independently replayed from an integer
transcript. It is evidence that the proposed proof-carrying arithmetic can
express a real validated-ODE obligation in HLS semantics and synthesize to RTL
for the U250 target.

It is not physical U250 execution. It does not include a Taylor remainder,
advance the state, isolate a Poincare event, certify a full orbit, cover the
leaf, establish global H-PG or V7-B, prove novelty or priority, or solve an open
problem. The next falsifiable step is physical-card execution, then composition
with a rigorous remainder-bearing state advance.

## Reproduction

```bash
bash scripts/research/cs6_u250_target23_picard_step_gate.sh
```

The retained receipt is
`scripts/research/receipts/cs6_u250_target23_picard_step_v1/`.
