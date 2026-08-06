# Math review request: event-local TM2R discriminator

Audit the rigor and causal interpretation of this diagnostic before it is run.
The state is a degree-2 Taylor model with two source variables and four
doubleton residual variables.  It is replayed along an already observed
12-split branch, without performing the later return transport.

For the same final branch state `X` and downward crossing tube `B`, compare:

1. Production symmetric slab: for radii `r = 2^-18, ..., 2^-7`, prove a
   signed Picard tube around `X`, form
   `D = B_x * B_y - B_w - z_s`, require `sup(D) < 0`, form the parametric
   predictor `p = -X_w / mid(D)`, then require the predictor and its interval
   Newton correction to lie strictly inside `[-r,r]`.
2. Chart test: apply the existing point-coefficient QR reconditioner to `X`
   and repeat exactly test 1.
3. Event-criterion test: reuse the already validated crossing-step tube `B`.
   Require `X subset B`, `sup(D) < 0`, and
   `delta = -range(X_w)/D subset (-1/256,0)`.  Prove a signed Picard slab from
   `X` over the full lower endpoint of `delta`, require that slab to be
   contained in `B`, then use fixed midpoint time plus interval residual time
   to project exactly onto `w=0` and require the section normal to remain
   strictly negative.

Implementation controls require the logged dominant split variable at every
depth, child interval containment in its parent, endpoint containment in the
stored crossing tube, equality of the stored and recomputed crossing
derivatives, and equality between the direct TM derivative and the first flow
coefficient.

The classification is deliberately fail-closed:

- any failed control -> `IMPLEMENTATION_INCONSISTENCY`;
- raw refuses, reconditioned accepts, anchored refuses -> `CHART_DRIFT`;
- raw and reconditioned refuse, anchored accepts -> `EVENT_CRITERION`;
- both alternatives accept -> `MIXED_CHART_AND_EVENT_CRITERION`;
- all three refuse with controls green -> `UNRESOLVED_ENCLOSURE`.

Questions:

1. Does containment of the full anchored Picard slab in the original crossing
   tube justify reuse of its strictly signed derivative for every branch
   trajectory and every `delta` in the Newton interval?
2. Does the chart/event classification overclaim causality, especially because
   the anchored projection uses fixed midpoint time plus an interval residual?
3. Which additional invariant is required to distinguish an implementation
   defect from a merely over-conservative interval representation?
4. Identify any sign error, containment-direction error, or missing strictness.

This diagnostic does not claim a covering relation, chaos, or a solution of
the open problem.
