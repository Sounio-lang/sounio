# Review Issues

1. [BLOCKER] The Jacobian row for the third equation is incorrect: the actual row is [y, x, -1, 0] but the code's mu_infinity computation uses |y|+|x| for the off-diagonal terms, which is correct, yet the diagonal entry -1 combined with the row sum gives -1+|y|+|x|, matching your statement — however, the second row's diagonal term x-(w+zs)/2 is not bounded by the Picard box in the code's mu_infinity evaluation, since w and z are state variables whose box is only known after the current Picard iteration completes.
   - location: §Jacobian/mu_infinity evaluation, Picard loop
   - why it matters: The contraction condition h*L_infinity(B)<1 requires L_infinity(B) computed on the entire box B, but if the code evaluates mu_infinity at a point or with partial box information, the bound is invalid and the enclosure proof collapses.
   - minimal fix: Compute mu_infinity(B) using interval arithmetic on all state variables (x, y, w, z) simultaneously, taking the supremum over B before checking h*L_infinity(B)<1.

2. [BLOCKER] The claim that "final-normal lower>0 proves transversality through that box" is insufficient: transversality requires the normal component of the vector field to be nonzero at every point of the event surface intersection with the box, but your lower bound only proves positivity of the propagated scalar, not that the vector field's normal component is uniformly bounded away from zero across the entire box.
   - location: §event lower state propagation, transversality claim
   - why it matters: If the vector field's normal component changes sign inside the box, the event surface could be tangent or crossed multiple times, invalidating the single-crossing guarantee needed for the one-center enclosure.
   - minimal fix: Propagate an interval for the normal component of the vector field evaluated on the event surface inside the box, and require its lower bound to be strictly positive.

3. [MAJOR] The polynomial map a_41(z)=flow derivative 41/41! is evaluated over the entire Picard box with Arb outward rounding, but the Taylor remainder after term 41 is not bounded by any explicit a priori estimate; the code only adds Arb output radii to the propagated radius, which assumes the remainder is captured by the interval arithmetic of the polynomial evaluation itself.
   - location: §Arb polynomial evaluation, remainder handling
   - why it matters: Arb's polynomial evaluation with outward rounding bounds the polynomial value, not the analytic remainder of the truncated Taylor series. Without a rigorous remainder bound (e.g., via Cauchy estimates or a validated integrator), the enclosure is not rigorous.
   - minimal fix: Add an explicit remainder bound R_41(B) computed from the supremum of the 42nd derivative on B, and include it in the radius addition.

4. [MAJOR] The requirement T(B)=X0+[0,h]F(B) subset B is checked, but the code's implementation of F(B) uses the actual Jacobian rows listed, yet the mu_infinity row for the first equation is -y+|4y-x|, which is correct, but the code must also verify that the interval evaluation of F(B) is done with outward rounding on the entire box B, not just at vertices; if the code evaluates F at a finite set of points, the subset check is invalid.
   - location: §Picard box subset check
   - why it matters: A finite-point check cannot certify T(B) subset B for nonlinear maps; interval arithmetic on the full box is required.
   - minimal fix: Use interval extensions of each component of F over B with outward rounding, then verify the resulting interval box is contained in B.

5. [MAJOR] The event lower state propagation "independently from the fixed step start" is not reproducible from the supplied text: no step size, no integration method for the lower state, and no explicit statement that the same Picard contractive enclosure is applied to the lower state trajectory.
   - location: §event lower state, reproducibility
   - why it matters: Without the same rigorous enclosure, the lower state could be a non-validated numerical approximation, undermining the transversality proof.
   - minimal fix: Specify the exact algorithm (same Picard iteration, same h, same box) for the lower state and include its contraction check.

6. [MINOR] The divergence correction: partial_x(2y^2-xy) = -y is correct, but the prior divergence calculation is not shown in the artifact, so the reader cannot verify whether the corrected value is actually used in the flow derivative computation; if the divergence appears in a_41, an uncorrected version would invalidate the polynomial coefficients.
   - location: §divergence correction, a_41 computation
   - why it matters: A single wrong coefficient in the 41st derivative term could produce a spurious radius that is not an upper bound.
   - minimal fix: Display the full corrected divergence expression and the resulting a_41 polynomial coefficients for audit.

7. [MINOR] The condition h*L_infinity(B)<1 is stated as required "for every Picard call," but the code does not show how h is chosen adaptively if the condition fails; a fixed h that works for one box may fail for a larger box, and the code must either reduce h or enlarge the box iteratively.
   - location: §Picard call, contraction condition
   - why it matters: Without an adaptive strategy, the proof may silently fail for boxes that grow during the integration.
   - minimal fix: Document the h-refinement loop and prove termination (e.g., h->0 ensures the condition eventually holds for compact B).

8. [NIT] The statement "All decimal constants are exact base-10 rationals enclosed by Arb" is true but does not address that intermediate arithmetic operations (e.g., division by 2 in the Jacobian) must also be performed with outward rounding; if the code uses floating-point for intermediate constants, the enclosure is broken.
   - location: §Arb enclosure, decimal constants
   - why it matters: A single unrounded division can produce a non-enclosing interval.
   - minimal fix: State explicitly that all arithmetic, including constants and divisions, is performed via Arb interval operations with outward rounding.

9. [NIT] The one-center enclosure claim requires that the Picard box B contains the true solution for all t in [0,h], but the proof only shows T(B) subset B for the map T; this is sufficient by Banach fixed-point, but the code must also verify that the initial box X0 is a valid starting enclosure (e.g., contains the initial condition), which is not stated.
   - location: §Picard box initialization
   - why it matters: If X0 does not contain the initial condition, the fixed-point argument is vacuous.
   - minimal fix: Add an explicit assertion that X0 contains the initial state vector.
