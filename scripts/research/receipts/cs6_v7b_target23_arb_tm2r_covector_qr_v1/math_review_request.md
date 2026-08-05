# Math review request: anchored C2 local covering certificate

Audit this as a hostile interval-dynamics review. Classify every issue as BLOCKER, MAJOR, MINOR, or NONE. The claim is deliberately local: one h-set covering relation for one second-return leaf, not a recurrent graph and not chaos.

## Raw evidence and coordinate convention

- Leaf: `U08-0000000223_S09-0000000325`, source `N0`, map `P^2`, section `w=0`, MinusPlus crossings.
- CAPD 5.3 / FILIB C2 receipt: outward binary64 hex intervals, order 8, return count 2.
- Global normalized source perturbations satisfy `delta0 in [-1/256,1/256]` and `delta1 in [-1/512,1/512]`.
- Local h-set coordinates satisfy `delta0 = xi/256`, `delta1 = eta/512`, with `(xi,eta) in [-1,1]^2`.
- Frozen target covector is `c=(ux,uy)` with
  `ux=-4644852547588741/6250000000000000`,
  `uy=13381910583555019/20000000000000000`.
- The executed worker declares CAPD `IHessian` entries as normalized Taylor coefficients: actual diagonal second derivative is `2*D2P_i00` or `2*D2P_i11`; the off-diagonal actual derivative is `D2P_i01`.

## Monotonicity derivation

The direct full-set C1 enclosure crosses zero:

`c . DP_col0 in [-4.626603776969373e-5, 1.955445093317306e-4]`.

It is therefore retained as a failed direct test, not used as a certificate.

Using center C2 `DP` and full-leaf C2 Hessian, define

```
g0  = c . DP_center_col0
h00 = 2*(ux*D2P000 + uy*D2P100)
h01 =    ux*D2P001 + uy*D2P101
dU/ddelta0 = g0 + h00*delta0 + h01*delta1
```

Exact rational interval evaluation gives

```
dU/ddelta0 in
[6926106331844325260675748061435313 / 94447329657392904273920000000000000000,
 35864899178803885713340791869911947 / 472236648286964521369600000000000000000]
```

and therefore, because `delta0=xi/256`,

```
dU/dxi in
[6926106331844325260675748061435313 / 24178516392292583494123520000000000000000,
 35864899178803885713340791869911947 / 120892581961462917470617600000000000000000]
```

whose lower endpoint is approximately `2.864570439091196e-7 > 0`.

## Anchored faces

At each fixed face `delta0 = +/-1/256`, evaluate the second-order Taylor enclosure

```
U = U(center) + g0*d0 + g1*d1
    + (1/2)*h00*d0^2 + h01*d0*d1 + (1/2)*h11*d1^2,
```

using `d1 in [-1/512,1/512]`, the center C2 value/gradient, and full-leaf Hessian. Squares are dependency-aware nonnegative interval squares.

The left-face enclosure is approximately

`U_left in [-3.972967495842092, -3.972967484526552]`.

Strict monotonicity yields, uniformly in eta,

`min_eta U_right(eta) >= left.lower + 2*(dU/dxi).lower`.

Subtracting `left.upper` leaves exact positive gap

`86903166009589810455173320337792871 / 154742504910672534362390528000000000000000`

or approximately `5.615985475985153e-7`. A separate direct right-face C2 enclosure has lower endpoint approximately `-3.972966912717556`, stronger than the anchored lower bound approximately `-3.972966922928005`.

Choose target unstable center as the midpoint of `left.upper` and the anchored right lower bound, and radius as one quarter of their gap. This maps the certified left upper bound to `-2` and right lower bound to `+2`, so both exit faces avoid `[-1,1]` with normalized margin exactly `1`.

The retained Arb support stable image is
`[-536870913/1073741824, 536870913/1073741824]`, giving positive entry margin `536870911/1073741824`. The target affine chart determinant is positive; the retained physical and rescaled return determinants are strictly negative.

For unstable dimension one, uniform opposite exit signs plus strict positive derivative are used to assign Brouwer degree `+1`. These facts are asserted to certify one local h-set covering relation.

## QR falsifier, separate from the positive route

The dynamic tripleton completed the second section with 2,570 reconditionings. Under the declared L1 max-absolute generator-radius metric, its directional total radius divided by the baseline is approximately `3.51536 > 1`; it therefore fails both improvement and the target `18x` reduction. The doubleton was interrupted after first-event projection and has no second-section certificate. The fixed covector mode has no completed receipt.

## Questions

1. Is the C2 mean-value derivative enclosure mathematically sound with the stated CAPD Hessian coefficient convention and source scaling?
2. Is the anchored inequality valid despite using `left.lower` for the right lower bound and `left.upper` for separation?
3. Does the face Taylor enclosure use the correct `1/2` factors and dependency-aware squares?
4. Do entry avoidance, opposite exit inequalities, chart invertibility, and unstable degree `+1` suffice for this local covering relation under the standard h-set definition, or is a homotopy/degree condition still missing?
5. Is combining the CAPD C2 exit/degree evidence with the independently retained Arb stable-support and determinant evidence legitimate when all are bound to the same leaf, map, section, and frozen coordinates?
6. Identify any overclaim. Recurrent graph, Fibonacci covering, global HPG, chaos, attractor, open-problem solution, novelty, and priority are all explicitly false.
