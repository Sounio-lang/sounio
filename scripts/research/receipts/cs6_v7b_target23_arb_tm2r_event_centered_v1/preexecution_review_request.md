# Predictor-centered event chart pre-execution review

Review the worker, verifier, mutations, runner, Slurm job, and contract named
`cs6_v7b_target23_arb_tm2r_event_centered_*`. Focus on mathematical soundness
and fail-closed behavior.

The frozen raw event-time predictor interval straddles the lower boundary of
the zero-centered `1/128` slab. Its exact midpoint is retained as a rational
fixed-time shift. The new worker:

1. replays the hash-bound XLEL branch and requires the predictor midpoint to
   equal the frozen exact rational value;
2. validates the fixed-time flow to that midpoint by signed Picard
   self-containment, strict contraction, Taylor order 12, and an Arb remainder;
3. builds symmetric residual slabs around the centered state;
4. requires signed Picard closure in both time directions and a strictly
   negative event derivative on their union;
5. forms a TM2 parametric predictor and an exact rational residual domain `R`
   with `predictor(P)+R` strictly inside the Picard slab, evaluates the flow at
   the predictor, and requires the interval-Newton image
   `-w(P,predictor(P))/D(P,predictor(P)+R)` to lie strictly inside `R`;
6. evaluates the parametric event state, projects `w` exactly to zero, and
   requires a strictly negative section normal;
7. requires positive aggregate coefficient weights for all six normalized
   variables `xi`, `eta`, `rho0`, `rho1`, `rho2`, `rho3` in the critical,
   centered, and accepted projected carriers.

The worker does not flatten to a box, use a point fallback, or perform the next
complete transport. Acceptance is only an event-local enclosure result. It
does not imply full support, a covering relation, a recurrent graph, chaos, or
an open-problem solution.

The strict Newton inclusion is the existence gate for every parameter; the
strictly negative whole-tube derivative is also the uniqueness gate. Please
identify any remaining missing hypothesis, invalid use of the signed Picard
tube, symbolic-dependence loss, or verifier path that could promote a malformed
receipt.
