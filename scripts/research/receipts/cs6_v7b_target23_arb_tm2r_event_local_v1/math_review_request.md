# Math review request: event-local TM2R discriminator

Audit the rigor and causal interpretation of this diagnostic before it is run.
The state is a degree-2 Taylor model with two source variables and four
doubleton residual variables.  It is replayed along an already observed
12-split branch, without performing the later return transport.

For the same final branch domain, instrument the unchanged production
`integrate_downward_return` call to retain both its delayed endpoint `X_d` and
the last accepted step endpoint `X_c` whose refined temporal tube `B_c`
actually contains `w=0` with strict downward derivative.  The wrapper records
each successful exact step and requires their sum to equal the production
reference time.  Compare:

1. Production symmetric slab: for radii `r = 2^-18, ..., 2^-7`, prove a
   signed Picard tube around `X`, form
   `D = B_x * B_y - B_w - z_s`, require `sup(D) < 0`, form the parametric
   predictor `p = -X_w / mid(D)`, then require the predictor and its interval
   Newton correction to lie strictly inside `[-r,r]`.
2. Chart test: apply the existing point-coefficient QR reconditioner to `X_d`
   and repeat exactly test 1.
3. Event-criterion test: apply the identical production symmetric-slab test to
   `X_c`, after applying the same exact 12 affine branch substitutions used for
   `X_d`.  This changes only which already validated endpoint is used to center
   the event chart.
4. Joint test: QR-recondition the reanchored `X_c` and repeat test 3.  Acceptance
   here is deliberately classified as mixed, not attributed to either cause.

The earlier fixed-midpoint anchored Newton test is retained only as an
observation.  It does not drive the version-2 classification.

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
   tube justify identifying `X_c` as an event-local chart anchor?
2. Does applying the same exact branch substitutions to `X_c` and `X_d`
   preserve the intended source subdomain strongly enough for the comparison?
3. Which additional invariant is required to distinguish an implementation
   defect from a merely over-conservative interval representation?
4. Identify any sign error, containment-direction error, or missing strictness.

This diagnostic does not claim a covering relation, chaos, or a solution of
the open problem.
