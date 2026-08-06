# Pre-execution review disposition

## Adopted mathematical finding

The initial xAI math review correctly rejected the first draft's use of a
Newton candidate without a fixed a-priori residual domain. The worker now
constructs an exact rational interval `R`, requires `predictor(P)+R` to lie
strictly inside the residual Picard slab, evaluates the event derivative on
that entire slab, and accepts only when the interval-Newton image is strictly
inside `R`. This is the existence gate. The same whole-tube derivative is
strictly negative, which gives uniqueness.

The focused xAI re-review marked the predictor center, Picard argument, strict
Newton inclusion, derivative uniqueness, section projection, six-variable
weights, and claim boundaries all `OK`.

## Code-review findings

The later hostile review used the corrected packet but retained four objections
that do not match the executable implementation:

1. The reconditioner name is correct. Runtime inspection prints
   `cs6_v7b_target23_arb_tm2r_subdivided_second_return_worker` for both
   `adaptive.__name__` and `point_coefficient_recondition.__module__`, exactly
   matching the verifier's frozen string. A frozen independent-verifier value
   is preferable to importing the worker's expectation.
2. A monomial depends on a normalized variable exactly when that variable's
   exponent is positive. Changing the test to exponent `>= 0` would count every
   monomial, including constants, and would make the preservation gate unsound.
3. The derivative used by interval Newton is evaluated on the union of the
   signed Picard tubes for the complete symmetric residual slab. The worker now
   also records and requires explicitly that `predictor(P)+R` is strictly
   inside that slab, so the derivative enclosure covers the Newton domain.
4. A failed implementation check can only produce
   `IMPLEMENTATION_INCONSISTENCY`; the verifier independently recomputes and
   requires that classification. Such a negative diagnostic receipt is valid
   evidence of a failed implementation control but cannot be promoted to event
   acceptance.
5. The pure-source-monomial count intentionally uses rows `x`, `y`, and `ell`
   because the section projection sets row `w` identically to zero. Preservation
   of all six normalized variables, including residual variables, is a separate
   positive-weight gate on those three retained rows.

The useful part of finding 3 was nevertheless adopted as a redundant worker
check and an independently recomputed verifier check.

## Focused code-disposition review

A final hostile pass produced four additional findings. They are dispositioned
as follows:

1. The path to `cs6_v7b_target23_arb_tm2r_event_local_v1` is intentional: this
   new experiment is cryptographically bound to the preceding event-local
   receipt. Pointing it at its own event-centered output directory would create
   a circular and initially missing dependency.
2. The review packet was not a distribution bundle. The committed worker hashes
   every imported research source, and the Slurm job archives the repository
   snapshot plus the complete `cs6_v7b_target23_arb_tm2r_*.py` source-hash
   manifest. Reproducibility is supplied by that staged snapshot and provenance,
   not by embedding transitive modules into the review prompt.
3. The verifier does recompute the critical certificate predicates from exact
   rational endpoints: Newton image strictly inside the a-priori domain,
   predictor plus domain strictly inside the Picard slab, derivative and normal
   strict signs, combined-time addition, and all six positive weights. Source
   hashes bind the interval generation. Re-running Arb would duplicate the
   worker rather than independently verify its serialized certificate.
4. Full states intentionally use rows `x,y,w,ell`; projected section carriers
   use `x,y,ell` because `w` is exactly zero. The worker now names these masks
   `STATE_ROWS` and `SECTION_ROWS` and documents the distinction.
