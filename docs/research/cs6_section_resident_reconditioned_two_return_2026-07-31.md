# CS6 Correlation-Preserving Two-Return Carrier

Date: 2026-07-31
Evidence level: bounded local CAPD 5.3.0/FILIB CPU execution
Promotion status: not eligible

## Question

Does preserving the first-return source correlation, then reconditioning the
tangent basis, recover the orientation information lost by the flattened
section carrier?

This experiment answers that question for the frozen N0 tile
`(u,s)=(20000,15000)/(40000,30000)` across two MinusPlus returns. It retains
the preceding flattened experiment as an exact control and compares three
tangent gauges in the same fixed source frame.

## Invariant

`INV-20260731-cs6-section-resident-reconditioned-two-return`

Let the source tile be represented by the normalized variable `delta`:

```text
X(delta) = xc + Q0 * delta
delta    = ((u-uc)/ru, (s-sc)/rs, 0)
```

The first local return supplies an interval Jacobian `J1` in the fixed `Q0`
coordinates. A separate singleton-C0 return supplies `Pc`, an enclosure of the
image of the source midpoint. Define:

```text
c   = mid(Pc)
M   = mid(J1)
rho = (Pc-c) + (J1-M) * delta
```

The mean-value theorem gives the correlated event carrier:

```text
P(X(delta)) subset c + M * delta + I * rho
```

The leading term keeps the common source variable in CAPD's `r0`; only the
point-integration and Jacobian-radius errors are boxed in `r`. C0 factor
reorganization is disabled for the continuation so CAPD does not fold that
residual back into `r0`.

## Tangent Gauges

All three lanes use the same mean-value C0 carrier. They differ only in the C1
seed `R`:

```text
IDENTITY:     R = Pi,             Y = Pi
MIDPOINT_M:   R = M,              Y contains M^-1
ORIENTED_QR:  R = Q(mid(J1)[:,0]), Y contains Q^-1
```

Here `Pi=diag(1,1,0)` is the tangent projector. `Y` is a certified inverse on
the two-dimensional tangent subspace, not an inverse of the singular embedded
3x3 matrix. For every lane:

```text
T        = Y * J1
J1       subset R * T
L2       contains DP(P(X)) * R
J2_fixed = L2 * T
```

Only `J2_fixed`, reconstructed in the original `Q0` coordinates, is used for
cross-gauge width and determinant comparisons. Factor widths are not evidence:
a compensated rescaling of `R`, `T`, and `L2` leaves the physical matrix
unchanged.

## Fail-Closed Verification

The verifier uses exact Python `Fraction` endpoint arithmetic after parsing the
one-ULP outward binary64 serialization. It independently:

1. Reconstructs every raw C0 and C1 doubleton component.
2. Re-derives the frozen normalized tile variable `delta`.
3. Recomputes `c`, `M`, the center error, linearization error, and `rho`.
4. Requires the event and all three continuations to share the canonical raw
   mean-value C0 representation.
5. Certifies both inverse products against `Pi` and `R*T` against `J1`.
6. Recomputes each Poincare derivative from the serialized flow tangent.
7. Recomputes every `L2*T` product and determinant in fixed `Q0` coordinates.
8. Rechecks event ordering, duration, post-section time, and Plus-side witness
   independently for every gauge.
9. Recomputes the total Liouville determinant, including an independent
   rational enclosure of `exp(ell)`.
10. Recomputes the flattened subchain digest and binds it to the preceding
    receipt and physical digest.

Here the Liouville route reconstructs the oriented Poincare determinant
`exp(ell2) * nu0/nu2 * det_xy(Q0)`, not the always-positive flow determinant
`exp(ell2)` alone. The negative sign is carried by the oriented source minor
and the normal-velocity ratio; `det_xy(Q0)` is the nonzero 2x2 tangent minor,
not the determinant of the singular 3x3 dummy-normal embedding.

The adversarial gate retains the preceding mutations and adds coordinated
mean-value and gauge attacks. In particular, it rejects transformations that
preserve a hull while changing the canonical `M*delta` identity, replacements
of the certified inverse by a transpose, compensated gauge scalings used to
claim a fake width gain, wrong product order, fabricated fixed-frame matrices,
and cross-challenge evidence mixing.

## Result

The bounded certificate passes. The mean-value C0 carrier materially reduces
second-return state widths relative to the flattened carrier:

```text
x width ratio = 0.493701462
y width ratio = 0.517196929
```

All three gauges remain jointly consistent with the direct two-return and
Liouville enclosures through nonempty joint intersections. Their fixed-frame
determinant intervals are:

```text
FLATTENED    [-1.6267567264e-08, +1.6210061326e-08]  width ratio 1.000000000
IDENTITY     [-1.6124781634e-08, +1.6067371623e-08]  width ratio 0.991210093
MIDPOINT_M   [-1.6012195201e-08, +1.5954934058e-08]  width ratio 0.984281508
ORIENTED_QR  [-2.0457660634e-08, +2.0400230084e-08]  width ratio 1.258031836
LIOUVILLE    [-2.8655230413e-11, -2.8655104567e-11]
```

The state result is gauge-independent, as it must be. Identity and midpoint-M
produce modest physical determinant-width gains; the well-conditioned QR
factorization makes interval cancellation worse after reconstruction. Every
matrix determinant still crosses zero. The scalar Liouville witness remains
strictly negative and roughly eight orders of magnitude narrower.

The scientific classification is therefore:

```text
CORRELATION_PRESERVED=true
STATE_ENCLOSURE_IMPROVED=true
ANY_GAUGE_SIGN_DEFINITE=false
ORIENTATION_FROM_MATRIX_ENCLOSURE=false
LIOUVILLE_DETERMINANT_NEGATIVE=true
PROMOTION_ELIGIBLE=false
```

## What We Learned

The preceding diagnosis was only partly about C0 box flattening. Preserving the
source variable recovers about half of the state precision, but only a small
fraction of the determinant precision. The dominant loss is now localized to
the interval C1 carrier and to cancellation in the reconstructed matrix.

This is a stronger negative result than trying more return counts: it separates
state wrapping from tangent wrapping on the same orbit segment. It also shows
that, in this bounded three-gauge comparison, choosing the best-conditioned
basis alone does not select the narrowest reconstructed interval matrix.

## Next Machine Experiment

The next differentiating carrier should preserve tangent dependence rather
than add another return immediately. Two compatible routes are now justified:

1. Use a C2 enclosure to construct an affine-in-`delta` Jacobian model
   `J(delta)=Jbar+A*delta+R`, then propagate that dependency through a custom
   C1 doubleton/tripleton layer. CAPD 5.3 has no built-in C1 tripleton, so this
   requires a small explicit carrier rather than another wrapper around the
   current hull.
2. Carry orientation and cone information in exterior/projective coordinates:
   a scalar log-determinant or Liouville channel plus an interval slope/Riccati
   channel, with the full matrix retained only as a cross-check.

The second route is especially suitable for a later U250 population engine:
the FPGA can evaluate many independent tiles and projective channels, while a
CPU verifier checks a small retained certificate. Hardware parallelism does
not repair dependency loss, so FPGA work remains downstream of this carrier
choice.

## Evidence Boundary

The canonical worker is compiled without optimization to keep this larger CAPD
template instantiation inside a bounded local compile envelope. Correctness
comes from CAPD/FILIB interval arithmetic, effective `-frounding-math`, exact
receipt verification, dependency hashes, and fresh replay, not from compiler
optimization.

This run did not use Foundry, Slurm, remote attestation, or either AMD U250. It
does not prove:

- a full-source carrier or adaptive cover;
- matrix orientation or nonsingularity at return two;
- a cone field, hyperbolicity, or a chaotic attractor;
- a correlation-preserving C1 carrier;
- validity for nonautonomous fields;
- three- or six-return closure;
- remote or FPGA execution.

The preceding flattened receipt, provenance, verifier, and physical digest are
retained unchanged as the control witness for this experiment.
