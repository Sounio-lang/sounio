1. [BLOCKER] Reconditioner module path in receipt payload will never match verifier expectation
   <location: scripts/research/cs6_v7b_target23_arb_tm2r_event_centered_worker.py:281 and verify.py:92>
   <why it matters> The worker installs `adaptive.point_coefficient_recondition` (imported from the local-diagnostic prior) while the verifier hard-requires the string `"cs6_v7b_target23_arb_tm2r_subdivided_second_return_worker.point_coefficient_recondition"`, so every run produces an immediate `require` failure on a field the contract claims is frozen.
   <minimal fix> Make the worker import the exact subdivided module that the verifier string names, or change the verifier expectation to match the actual module used.

2. [BLOCKER] `variable_weights` and `model_variable_weights` only sum coefficients whose monomials have a positive exponent in the chosen variable, silently dropping constant and cross terms that affect preservation claims
   <location: worker.py:58-74 and 76-86>
   <why it matters> The six SYMBOLIC_GATE checks and the verifier's `positive_weights` calls rely on these aggregates; an omitted monomial can make a zero-weight variable appear positive or allow a lost dependence to pass the `retained == 0` test.
   <minimal fix> Accumulate `base.upper_abs(coefficient)` for every monomial that contains the variable at all (exponent >= 0), not only when exponent > 0.

3. [MAJOR] Newton-image test uses the wrong derivative for the interval-Newton inclusion
   <location: worker.py:172>
   <why it matters> `newton_image = -predicted_state[2].range() / derivative` divides by the centered tube derivative rather than the derivative evaluated over the full residual domain `newton_domain`; the strict-inclusion claim required by the contract therefore does not certify existence inside R.
   <minimal fix> Replace `derivative` by an interval enclosure of the derivative over `tube` widened by the Newton domain, or recompute the Jacobian interval over the enlarged set.

4. [MAJOR] Verifier accepts a receipt when `implementation_checks_passed` is False provided the chart is also refused, yet the contract and schema both require all six-variable preservation checks to have succeeded
   <location: verify.py:140-142 and worker.py:312-314>
   <why it matters> This creates a path where a malformed carrier that lost a symbolic variable can still be classified `PREDICTOR_CENTERED_EVENT_REFUSED` without triggering the expected `IMPLEMENTATION_INCONSISTENCY` gate.
   <minimal fix> Require `checks_passed` unconditionally for any non-`IMPLEMENTATION_INCONSISTENCY` classification.

5. [MINOR] `retained_source_monomials` counts monomials only from rows 0,1,3 yet the SYMBOLIC_GATE contract lists six variables including rho2/rho3
   <location: worker.py:88-95>
   <why it matters> The final acceptance record can claim "pure source monomials retained" while rho2/rho3 dependence may already have been eliminated by the projection step.
   <minimal fix> Extend the row tuple to (0,1,2,3) or document why rho2/rho3 are exempt from the retained-monomial count.
