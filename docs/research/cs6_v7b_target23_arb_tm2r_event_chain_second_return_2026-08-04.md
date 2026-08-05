# CS6 leaf 331: rigorous event-chain transport to the second section passage

Date: 2026-08-04 to 2026-08-05

## Result

The full source leaf `U08-0000000223_S09-0000000325` is covered by four
algebraically reconstructed source tiles (`XLEL`, `XLEH`, `XHEL`, `XHEH`).
Each tile independently validates this event chain with Arb at 256 bits:

1. first upward return and interval-Newton projection to exact `w = 0`;
2. first downward passage and parametric TM2 interval-Newton projection to
   exact `w = 0` with negative transversality;
3. transport from that projected carrier to the next upward passage;
4. parametric TM2 interval-Newton projection of the second passage to exact
   `w = 0` with positive transversality.

The aggregate verifier reports:

- downward event-time hull after the first projection:
  `[0.572127559783535, 0.572179547840906]`;
- second upward event-time hull after the first projection:
  `[4.17753161276524, 4.17768761676851]`;
- second-return derivative hull:
  `[53.4949993092108, 54.9793408558446]`;
- second-return projected normal hull:
  `[54.3440334052774, 54.3721663698261]`;
- five projected second-event leaves, from one validated split of `XHEL`;
- zero unresolved leaves.

This certifies the full leaf through the second section passage. It does not
certify a return-map determinant, a covering relation, a global HPG result, or
chaos.

## Carrier change

The failed carrier stored wide Arb intervals inside polynomial coefficients.
After repeated reconditioning, even the constant coefficient of the section
derivative enclosed both signs. Spatial subdivision could not shrink that
constant interval: the search exhausted its 16,383-node budget while allowing
derivative-directed refinement down to depth 18, and produced no sign-resolved
leaf. The node count is a search budget, not the size of a complete depth-18
binary tree.

The successful carrier moves every coefficient radius into explicit residual
generators before QR compression. Polynomial coefficients retain point
midpoints, while the added generators cover the discarded coefficient
uncertainty on the normalized source domain `|xi| <= 1`, `|eta| <= 1`: every
retained source monomial (of degree at most two here) therefore satisfies
`|z^alpha| <= 1`. Every generator is checked by rational `Q * Q^-1`
reconstruction before the step is accepted.

After each downward event projection, the carrier is stabilized by rebuilding
every Arb coefficient from its exact rational lower and upper endpoints. Each
rebuilt ball must contain its input. This preserves the TM2 polynomial and
residual variables; it is not interval-box flattening. Each isolated tile
receipt records 39 successful stabilization checks.

## Parametric event Newton

For a carrier state `X(z)` near a section event, the worker constructs a TM2
event-time predictor

`tau_0(z) = -w(z) / mid(d w / dt)`.

It evaluates the order-12 flow polynomial at the variable time `tau_0(z)`,
bounds the order-13 temporal remainder over a contracting signed Picard slab,
and computes an interval correction

`C = -range(w(Phi_tau_0(X))) / range(d w / dt)`.

The event is accepted only when `tau_0 + C` lies strictly inside the Picard
slab and the derivative has the requested strict sign throughout that slab.
The projected carrier retains the source and residual symbols, while its `w`
component is set exactly to zero. At the second event, `XHEL` required one
binary carrier split; both reconstructed children passed the same test.

## Reproduction

Fast receipt verification and negative mutations:

```bash
bash scripts/research/cs6_v7b_target23_arb_tm2r_event_chain_second_return_gate.sh
```

Full regeneration is intentionally expensive and uses the isolated per-tile
path:

```bash
CS6_REGENERATE=1 \
  CS6_PYTHONPATH=/tmp/sounio-cs6-arb-full-leaf-deps \
  bash scripts/research/cs6_v7b_target23_arb_tm2r_event_chain_second_return_gate.sh
```

The next mathematical target is a candidate covering relation on the five
second-event carriers. It requires rigorous section coordinates, face images,
entry/exit inequalities, and a return-map derivative or degree certificate;
none of those claims is made here.
