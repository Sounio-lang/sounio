# CS6 target-23: first local h-set covering candidate

Date: 2026-08-05

## Result

The five rigorous TM2R carriers at the second `w=0` section were converted
into a bounded local h-set candidate for the complete source leaf
`U08-0000000223_S09-0000000325`. The source h-set is the existing normalized
`(xi,eta) in [-1,1]^2` leaf. The target uses the frozen rational affine rows

```text
U_raw = -0.74317640761419856 x + 0.66909552917775095 y
S_raw = -0.66909552917775095 x - 0.74317640761419856 y.
```

The target center and radii are constructed with exact rational arithmetic
from the two exit-face center enclosures and the complete support image. The
target linear coordinate determinant is rigorously positive, so these rows
define an invertible section chart.

Three obligations have different outcomes:

```text
HSET_COORDINATES_CERTIFICATE=true
ENTRY_BOUNDARY_AVOIDANCE_CERTIFICATE=true
RETURN_MAP_DETERMINANT_CERTIFICATE=true
EXIT_FACE_INEQUALITIES_CERTIFICATE=false
COVERING_DEGREE_CERTIFICATE=false
LOCAL_HSET_COVERING_RELATION_CERTIFICATE=false
```

Thus this is the first rigorous local h-set candidate on the new carriers, but
it is not a certified covering relation.

## Support and faces

The complete support remains covered by the four exact source tiles
`XLEL`, `XLEH`, `XHEL`, and `XHEH`. Their second event has five projected
carriers because `XHEL` needs one residual split along `RHO0`.

The source exit faces `xi=-1` and `xi=+1` were then transported independently.
Using the whole support carriers as face enclosures was rejected because their
uncertainty includes source interior that is absent from a face. The face
worker fixes `xi` before the first integration and keeps `eta` symbolic.
The difficult outer `eta` halves were split once before integration:

```text
left face:  XLEL/L, XLEL/H, XLEH/root
right face: XHEL/root, XHEH/L, XHEH/H
```

All six face carriers validate both section passages, exact final `w=0`, and
positive final transversality. No face carrier needs an event-time split.

## Certified obligations

The normalized complete support stable image is

```text
[-0.5000000009313226, 0.5000000009313226]
```

so it avoids the target entry boundary with margin about
`0.4999999990686774`.

Liouville transport through the retained `ell` component, positive initial
and final normal velocities, the source chart determinant, and the positive
target chart determinant give

```text
physical source-scaled determinant:
  [-2.1684882127625672e-16, -2.1647419966754964e-16]

normalized source-to-target determinant:
  [-6.868740758431309e-7, -6.856874472647469e-7]
```

Both intervals are strictly negative. This certifies the determinant sign and
local nonsingularity on every retained support carrier. It does not determine
the one-dimensional covering degree without the exit inequalities.

## Exit-face falsifier

The rational target direction was selected by a bounded directional scout and
then re-evaluated with Arb over the independently transported face carriers.
The face centers have candidate orientation `+1`, but their rigorous images
overlap:

```text
left normalized unstable image:
  [-38.02234876155853, 34.02234876155853]

right normalized unstable image:
  [-34.01244896650314, 38.01244896650314]

signed face gap:  -9.95754656931221e-6
face overlap:       9.95754656931221e-6
exit margin:      -35.02234876155853
```

The center separation in the unnormalized target coordinate is about
`5.854384472583735e-7`, while the combined directional radii are about
`1.054298502367601e-5`. The next carrier must therefore reduce this directional
uncertainty by roughly a factor of 18, or prove the same separation through a
validated monotonicity argument.

Two diagnostics are deliberately not counted as certificates. Unsplitted
outer face carriers spent about 30 minutes in final-event subdivision before
being interrupted. Two degree-three face transports also reached the same
expensive event-projection regime and were interrupted without output. The
retained evidence uses only the completed degree-two support and refined-face
receipts.

## Reproduction

Fast verification, deterministic reanalysis, and negative mutations:

```bash
bash scripts/research/cs6_v7b_target23_arb_tm2r_hset_covering_gate.sh
```

Full carrier regeneration is intentionally expensive:

```bash
CS6_REGENERATE=1 \
  CS6_PYTHONPATH=/tmp/sounio-cs6-arb-full-leaf-deps \
  bash scripts/research/cs6_v7b_target23_arb_tm2r_hset_covering_gate.sh
```

The retained worker path is Python/Arb at 256 bits. CAPD, a point fallback,
and FPGA execution are not used.

## Independent review

xAI/Grok 4.3 independently accepted the strict-negative determinant
enclosures, the stable entry-boundary avoidance, the positive exit-face
overlap, and the consequent refusal to claim a covering degree. It identified
validated monotonicity of `U_raw o P^2` as the natural way to strengthen the
next attempt.

The mandatory Z.AI leg reached its five-hour usage limit. Independent retries
through Qwen, DeepSeek, and Groq also failed at their provider boundaries
(credit, balance, and key errors respectively), before producing reviews. The
review disposition is therefore `PASS_SINGLE_PROVIDER_DEGRADED`, not a dual
pass. The provider failures are retained beside the xAI review, and a second
independent review remains required when provider availability is restored.

## Next attack

Seed the rigorous QR remainder basis with the frozen target unstable covector
and transport that directional error as a privileged doubleton/tripleton
coordinate on the six face branches. The falsifier is exact: the sum of the
left and right directional radii must fall below the certified center
separation `5.854384472583735e-7`. A complementary C1 route may instead prove a
strict signed derivative of `U_raw o P^2` across the source and combine it with
one anchored face value.

No recurrent covering graph, Fibonacci relation, global HPG certificate,
chaos, chaotic attractor, novelty, priority, or open-problem solution is
claimed.
