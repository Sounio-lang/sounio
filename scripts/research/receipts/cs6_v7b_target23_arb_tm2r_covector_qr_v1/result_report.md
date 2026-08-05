# CS6 target-23 covector QR falsifier and anchored C2 covering

Date: 2026-08-05

Leaf: `U08-0000000223_S09-0000000325`

Map: `P^2` on the section `w=0`

## Result

The proposed dynamic QR tripleton did not reduce the frozen-covector directional radius. It completed the second section, but its declared L1 generator-radius metric was `3.5153602960538319` times the retained baseline. The achieved improvement factor was `0.28446586289392589`, so this parameterization worsened the radius and falsified the requested `18x` reduction.

The alternative route succeeded. A CAPD 5.3 / FILIB C2 enclosure proves strict monotonicity of `U_raw o P^2` in the unstable leaf coordinate over the entire leaf. Combined with the retained Arb stable-support and determinant receipts, it certifies one local h-set covering relation with unstable degree `+1`.

This is not a recurrent covering graph and is not a proof of chaos.

## QR experiment

The worker transports one or two privileged residual directions and completes the remaining rational basis by a QR-selected doubleton/tripleton reconditioning. Every generator reconstruction is checked by interval containment.

The dynamic tripleton completed with:

- 2,570 reconditionings;
- 178,564 generator reconstructions;
- 2,569 transports of the frozen unstable direction;
- one certified second-section carrier;
- no point fallback.

Its directional total radius divided by the original face-carrier radius is exactly the rational recorded as `TRIPLETON_TO_BASELINE_RADIUS_RATIO_Q` in `aggregate.txt`, approximately `3.51536 > 1`. The doubleton reached the first-event projection and was interrupted before a second-section receipt. The fixed-covector mode has no completed receipt. Consequently only the completed tripleton run is a quantitative falsifier; the other two modes remain uncertified rather than mathematically rejected.

## C2 monotonicity

The direct C1 enclosure of the frozen covector applied to the first derivative crosses zero:

```text
c . DP_col0 in [-4.626603776969373e-5, 1.955445093317306e-4].
```

It cannot prove monotonicity. The retained C2 result supplies the center derivative and a full-leaf Hessian. The executed worker's convention is:

```text
actual diagonal second derivative = 2 * IHessian diagonal coefficient
actual mixed second derivative    = 1 * IHessian mixed coefficient
```

Using `delta0=xi/256` and `delta1=eta/512`, exact rational interval arithmetic gives

```text
d(U_raw o P^2)/dxi in
[2.8645704390911962e-7, 2.9666749271875577e-7].
```

The derivative is therefore strictly positive on the closed source h-set.

## Anchored exit faces

The left face is evaluated by a second-order Taylor enclosure at `delta0=-1/256`, with the full `delta1` interval retained. For every fixed eta, integration of the whole-leaf derivative gives

```text
U_right(eta) >= U_left(eta) + 2 * inf(dU/dxi).
```

Using the left lower endpoint to anchor the right lower bound and then subtracting the left upper endpoint leaves a rigorous gap of approximately

```text
5.6159854759851525e-7.
```

A separate direct C2 evaluation of the right face gives a stronger lower bound and serves as a cross-check. The target unstable center is the midpoint of the certified left upper and anchored right lower bounds; its radius is one quarter of their gap. In this chart the left upper maps exactly to `-2`, the right lower maps exactly to `+2`, and the target interval is `[-1,1]`, giving exit margin exactly `1`.

The retained stable support maps into

```text
[-536870913/1073741824, 536870913/1073741824],
```

with entry margin `536870911/1073741824`. The new target chart determinant is strictly positive, while the retained physical and rescaled return determinants are strictly negative. Uniform opposite exit signs and the strictly positive unstable derivative give Brouwer degree `+1` in unstable dimension one.

Therefore:

```text
HSET_COORDINATES_CERTIFICATE=true
ENTRY_BOUNDARY_AVOIDANCE_CERTIFICATE=true
ANCHORED_EXIT_FACE_INEQUALITIES_CERTIFICATE=true
COVERING_DEGREE_CERTIFICATE=true
RETURN_MAP_DETERMINANT_CERTIFICATE=true
LOCAL_HSET_COVERING_RELATION_CERTIFICATE=true
```

## Evidence and review

The gate reopens the committed CAPD archive, checks the unique attempt-199 member, binds `results.tsv`, source and binary hashes, checks the exact normalized source radii, reconstructs all intervals from outward hexadecimal endpoints, and compares the result byte-for-byte with the aggregate. The independent verifier then checks the algebraic identities and 12 adversarial mutations.

Dual math review was completed with xAI Grok 4.3 and Z.AI GLM-5.2. Z.AI independently rederived the certificate. Grok's initial convention objections were dispositioned against the executed worker and whole-leaf enclosure; a focused Grok re-review then marked every disputed step correct. The local review inputs, outputs, raw JSON, and disposition are retained beside the aggregate.

Run:

```bash
bash scripts/research/cs6_v7b_target23_arb_tm2r_covector_qr_gate.sh
```

## Boundary

The certificate supplies one edge candidate, not a closed symbolic-dynamics construction. The following remain false:

```text
RECURRENT_COVERING_GRAPH_CERTIFICATE=false
FIBONACCI_COVERING_CERTIFICATE=false
GLOBAL_HPG_CERTIFICATE=false
CHAOS_PROVED=false
CHAOTIC_ATTRACTOR_PROVED=false
OPEN_PROBLEM_SOLVED=false
NOVELTY_OR_PRIORITY_CLAIMED=false
```

The next falsifier is no longer directional remainder reduction. It is composability: construct a target h-set whose image returns to the certified source or to a second certified node, then prove the corresponding edge with the same entry, exit, degree, and determinant discipline.
