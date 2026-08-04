# CS6 target-23 Arb TM2R: event projection and second-phase boundary

**Date:** 2026-08-04

**Status:** the complete critical leaf 331 has a rigorously projected carrier on
the first-return section. The projection preserves explicit dependence on the
two leaf variables and the same degree-2 reconditioned Taylor model advances
707 further validated steps. Its 708th post-event step is refused because the
raw Taylor endpoint is not contained in the Picard tube. There is no certified
second return, determinant, covering relation, chaos, attractor, novelty, or
open-problem result.

## Event projection

The first return remains bracketed by step 617. Its complete event tube has
strictly positive `w'=x*y-w-zs`, so every source point has one unique event in
that step. At the positive endpoint the worker encloses its backward event-time
correction by

```text
delta = -w_endpoint / w'(event tube)
delta subset [-2^-8, 0].
```

The midpoint of this interval is propagated backward with a signed Picard tube
and the order-12 autonomous Taylor flow. The remaining symmetric time radius is
enclosed with the vector field over the complete event tube. A second signed
Picard enclosure for the whole position slab `delta in [Delta_lower,0]` is
explicitly required to be contained in that validated event tube. Only after event
existence and uniqueness are established is `w` set exactly to zero. Two
QR-derived reconditionings then produce the carrier for the next phase.

This construction keeps 15 nonconstant pure `xi,eta` monomials explicit in
`x,y,ell`. The residual event-time uncertainty is intervalized, so the artifact
claims preservation of source dependence only up to interval enclosure of the
event-time/source correlation, not an exact symbolic event-time function or its
Jacobian.

## Retained result

With Arb 0.8.0 at 256 bits, source degree 2, time order 12, and step `2^-8`:

```text
Newton correction              [-728214639/274877906944,
                                -179021577/68719476736]
fixed backward shift            -1444300947/549755813888
residual shift radius            12128331/549755813888
projection Picard iterations     2
full-slab Picard iterations      2
full-slab inside event tube      true
projected carrier max width      539659285/1099511627776
pure source monomials retained   15
second-phase completed steps     707
second-phase attempted steps     708
validated post-event time        707/256
failure                          ENDPOINT_ESCAPES_PICARD
```

The failing step still closes a Picard tube, but its raw TM endpoint enclosure
escapes that tube. The implementation therefore rejects the endpoint and does
not recondition or use it. This is a fail-closed numerical boundary, not
evidence that the physical return does not exist.

Increasing only the second-phase temporal order to 24, and separately allowing
source degree 3, did not move this step boundary in exploratory runs. Those
variants are diagnostic observations, not retained certificates in this gate.

## Mathematical boundary

```text
FULL_LEAF_FIRST_RETURN_CERTIFICATE=true
INTERVAL_NEWTON_EVENT_PROJECTION_CERTIFICATE=true
FULL_LEAF_SECOND_RETURN_CERTIFICATE=false
RETURN_MAP_DETERMINANT_CERTIFICATE=false
GLOBAL_HPG_CERTIFICATE=false
V7_B_ELIGIBILITY=false
CHAOS_PROVED=false
CHAOTIC_ATTRACTOR_PROVED=false
OPEN_PROBLEM_SOLVED=false
```

The next falsifier must reduce the long-time endpoint over-enclosure after the
event. Since neither higher temporal order nor one extra source degree changed
the boundary, the evidence points to a stronger residual representation or a
validated subdivision of the carrier rather than another local-order increase.
