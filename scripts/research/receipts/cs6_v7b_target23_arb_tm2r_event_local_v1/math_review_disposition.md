# Disposition of event-local diagnostic review

The xAI code review produced one useful reproducibility check: the derivative
stored by the crossing integrator must equal the derivative recomputed from the
retained crossing tube.  That exact interval identity is now an implementation
control.  The worker also enforces Python 3.10 or newer and records the active
reconditioner by qualified name.

The mathematical review's sign objection is rejected.  The code requires
`upper(X_w) < 0` and `upper(D) < 0`; consequently `-X_w` is strictly positive
and interval division by the strictly negative `D` is strictly negative.  It
also explicitly requires `upper(delta) < 0` and `lower(delta) > -1/256`.

The slab objection is also rejected.  For a negative step, `signed_picard_box`
uses the complete time interval from that step to zero.  Calling it with
`lower(delta)` therefore encloses every intermediate time for every correction
in `delta`.  The diagnostic then requires this entire slab to be contained in
the already validated crossing tube before reusing the tube derivative.  The
fixed-midpoint plus residual-time enclosure uses `field_interval(B)` over that
same contained slab; the projected normal is evaluated only after the residual
has been included in the Taylor-model remainder.

The replay does not assume unchanged remainders silently.  Every split is the
exact rational affine substitution implemented by `split_state`; that function
checks inverse-child coefficient and remainder reconstruction, and this worker
additionally requires the chosen child's interval range to lie in its parent.

`MIXED_CHART_AND_EVENT_CRITERION` is intentionally non-causal: it records that
both alternatives accept and refuses to choose one explanation.  Likewise,
`UNRESOLVED_ENCLOSURE` is not evidence that the implementation is correct in an
absolute sense; it means only that the enumerated implementation controls did
not identify a discrepancy.

Z.AI and all available fallback providers failed operationally as recorded in
`provider_failures.md`.  This review is retained as single-provider degraded,
not as a dual-provider pass.
