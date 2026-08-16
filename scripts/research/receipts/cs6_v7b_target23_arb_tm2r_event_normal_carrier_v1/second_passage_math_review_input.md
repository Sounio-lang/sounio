# Focused review: continuation to the second section passage

Review this bounded extension and return PASS or a concrete missing obligation.

After a rigorously accepted upward section projection of every retained source
leaf, the worker keeps the actual projected TM2R carrier objects in memory. For
each one it calls the existing hash-bound event-chain routines:

1. `integrate_downward_return(carrier)` requires the initial projected carrier
   to depart with a strictly positive whole-tube derivative;
2. it advances the same TM2R with the active event-normal reconditioner, tracks
   a strictly positive phase, and accepts a later section-containing tube only
   when `upper(x*y-w-ZS) < 0` on the whole tube;
3. it returns only after that strictly downward tube is followed by an endpoint
   with `w < 0`;
4. `project_downward_event` constructs a two-sided signed Picard slab, requires
   the derivative upper endpoint to remain negative on the whole slab, requires
   the parametric interval-Newton event-time image strictly inside the slab,
   evaluates the TM2 flow, and reconditions on the exact `w=0` section;
5. it requires the projected normal `x*y-ZS` to have strictly negative upper
   endpoint and retains pure source monomials.

The new wrapper runs those routines for every accepted first-section leaf. It
serializes each second-section carrier, derivative, normal, event time, primary
weights, accepted substeps, bisections, and downward-section-tube count. It
classifies the second passage as accepted only if all first-section leaves
produce a branch and all six original weights (`xi,eta,rho0..rho3`) have
positive upper endpoints.

The independent verifier additionally requires exact serialized `w=0`, strict
negative derivative and normal, positive first-six weights, positive substep
and downward-tube counts, exact source-leaf/branch cardinality, and consistent
per-branch and aggregate flags.

Question: if these checks pass, is this a rigorous continuation of the same
retained TM2R leaf through the next downward section passage? No covering
relation, recurrence, chaos, or open-problem claim is made.
