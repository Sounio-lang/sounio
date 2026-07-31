# CS6 Section-Resident Two-Return Composition

Date: 2026-07-31
Evidence level: bounded local CAPD/FILIB CPU execution
Promotion status: not eligible

## Question

Can the section-resident carrier from the first CS6 return be used as the
input to a second local return, while retaining a rigorous cumulative tangent
map from the original source coordinates?

This experiment answers that question for one frozen N0 tile across two MinusPlus returns.
It does not cover the full source, six returns, or an
adaptive cover.

## Invariant

`INV-20260731-cs6-section-resident-two-return`

For the autonomous CS6 vector field, let `P` be the return map on
`Sigma={w=0}`. Let `Q0` be the frozen global source tangent basis, `B1` the flattened
first-event box, and `Qsigma=diag(1,1,0)`. The checked construction is:

```text
J1       = DP(source) Q0
J2_local = DP(B1) Qsigma
J2x1     = J2_local * J1
```

The first event carrier stores `J1`. Its continuation carrier stores the same
raw C0 box and clock, but resets C1 to `Qsigma`; `J1` remains explicit
metadata. The second event carrier stores the resulting local derivative
`J2_local`. The second continuation carrier again resets C1 to `Qsigma` and
stores both `J2_local` and `J2x1` as metadata.

The product order is part of the invariant. The receipt also serializes
`J1*J2_local` and requires at least one disjoint matrix entry, so the tile
distinguishes the correct order from its reversal.

## Why The Reset Is Sound Here

The CS6 field used by this probe is autonomous. Starting the second carrier at
the interval event time therefore does not add an omitted
`partial(P)/partial(t) * Dt1` term. This construction is not a proof for a
nonautonomous vector field; such a system would need an augmented time state
and its variational equation.

CAPD 5.3 leaves a set initially on the section before seeking the next
crossing. The local adapter preserves the protected fast path:

```text
integrateUntilSectionCrossing(before, after, 1)
crossSectionInOneStep(before, after, local_time, image)
computeOneStepSectionEnclosure(...)
computeDP(image, flow_tangent, absolute_event_time)
```

The public cumulative baseline uses a fresh solver, map, and source set with
`return_count=2`. It is not obtained by reusing a set already mutated by a
one-return call.

## Independent Checks

The fail-closed verifier performs these calculations with exact Python
`Fraction` endpoint arithmetic after undoing the one-ULP serialization layer:

1. Reconstruct every raw C0 and C1 carrier hull.
2. Recompute every Poincare projection from the serialized flow tangent.
3. Recompute `J2_local*J1` and the reversed product.
4. Recompute every reported 2x2 determinant.
5. Recompute `exp(ELL)` independently with a rational Taylor enclosure and a
   geometric tail bound.
6. Recompute the two-return Liouville determinant
   `exp(ell2) * nu0/nu2 * det_xy(Q0)`.
7. Require joint, rather than merely pairwise, intersections across the local,
   public, and Liouville routes.

The fifth check closes a known limitation of the one-return predecessor:
`EXP_ELL_RECOMPUTED=true` in this receipt.

Here `det_xy(Q0)` is the oriented 2x2 minor of the two nonzero tangent
columns. It is not the zero 3x3 determinant of the dummy-normal matrix, and it
is not the physical area of the tiny C0 tile. `Q0` is a deliberately fixed
global comparison basis over every point in that tile.

## Result

The canonical bounded run passed. Representative decimal views of the
retained binary64 intervals are:

```text
t1 local       = [2.4068690012, 2.4068690627]
t2 local       = [6.5845413697, 6.5845414503]
t2 public      = [6.5845413815, 6.5845414384]
post-t2 local  = [6.5854549541, 6.5854550156]
```

The exact verifier established:

```text
P1_STATE_JOINT_OVERLAP=true
P1_DP_JOINT_OVERLAP=true
P2_STATE_JOINT_OVERLAP=true
P2_DP_JOINT_OVERLAP=true
P2_DETERMINANT_JOINT_OVERLAP=true
COMPOSITION_EXACT_RECOMPUTED=true
REVERSED_ORDER_EXACT_RECOMPUTED=true
EXP_ELL_RECOMPUTED=true
```

The cumulative composed map and the fresh public `P^2` enclosure intersect in
every entry. Their determinant enclosures also have a joint intersection with
the independently reconstructed Liouville determinant.

## The Important Negative Signal

The construction is sound on this tile, but its wrapping is already severe.
Both cumulative determinant intervals cross zero at return two, while the
Liouville enclosure is a much narrower strictly negative interval. Therefore
this box-flattening carrier does not preserve determinant sign strongly enough
to support a two-return cone or hyperbolicity claim.

This is not a contradiction: the wider intervals still contain the same true
quantity. It is a machine-level diagnosis that naive section-box flattening is
the next bottleneck.

## Next Machine Experiment

The next differentiating experiment should preserve event-one correlations
instead of increasing the return count immediately:

1. Keep the section-resident C0 doubleton/tripleton representation rather than
   replacing it by its interval hull.
2. Recenter and recondition the two tangent coordinates at the event, retaining
   an explicit transition matrix back to the source basis.
3. Compose the transition matrices exactly and repeat this same two-return
   direct/Liouville gate.
4. Promote to return three only if the cumulative determinant becomes
   sign-definite without losing the existing inclusions.

The later AMD U250 path can accelerate a population of independently checked
tiles, but it does not repair wrapping by itself.

## Evidence Boundary

This run used CAPD 5.3.0 with FILIB and effective `-frounding-math` on bounded
local CPU. It did not use Foundry, Slurm, remote attestation, or either AMD
U250. Context7 did not resolve the CAPD library; the exact local CAPD 5.3
headers and executable behavior are the documentation authority for this
artifact.

The result does not prove:

- a full-source carrier;
- determinant sign at two returns;
- a cone field, hyperbolicity, or a chaotic attractor;
- validity for nonautonomous fields;
- six-return closure;
- remote or FPGA execution.

Legacy one-return artifacts remain unchanged and are the parent witness for
this follow-up experiment.
