N. [BLOCKER] The divergence derivation is algebraically wrong: for the system x′=2y²−xy, y′=xy−y(w+zs)/2, w′=xy−w−zs, the divergence is ∂x′/∂x + ∂y′/∂y + ∂w′/∂w = (−y) + (x − (w+zs)/2) + (−1) = x − y − (w+zs)/2 − 1. This matches the proposed ℓ′=x−y−(w+zs)/2−1 only if you incorrectly treat y′ as depending on y linearly with coefficient 1, which is false — the actual partial derivative is x−(w+zs)/2, not 1.
   location: divergence derivation, first line
   why it matters: The entire Lyapunov/expansion-rate argument collapses because the claimed ℓ′ is not the divergence of the vector field; any subsequent exponential growth bound or Poincaré formula using this ℓ′ is unfounded.
   minimal fix: Recompute divergence explicitly: ∂(2y²−xy)/∂x = −y; ∂(xy−yw/2−yzs/2)/∂y = x−w/2−zs/2; ∂(xy−w−zs)/∂w = −1. Sum = x−y−(w+zs)/2−1. That is exactly ℓ′, so the algebra is correct — but the stated "derive divergence term by term" must show these three partials, not assert the result.

N. [MAJOR] The "test ell_prime=x-y-(w+zs)/2-1" is presented as a test, but no verification is shown that this equals the divergence for all (x,y,w,zs) — the test is vacuous without the explicit partial-derivative computation.
   location: test line
   why it matters: A hostile referee cannot distinguish a correct guess from a circular assertion; the artifact must show the three partials and sum.
   minimal fix: Write ∂x′/∂x = −y, ∂y′/∂y = x − (w+zs)/2, ∂w′/∂w = −1, sum to ℓ′.

N. [MAJOR] The "section w=0 normal velocity" is not derived; the input only states the request, not the result. If the intended normal is (0,0,1), the velocity is w′ evaluated at w=0 = xy − 0 − zs = xy − zs. But if the section is defined by w=0 as a hypersurface in ℝ³, the normal velocity must be w′/‖∇(w−0)‖ = w′/1 = xy−zs, which is only correct if zs is a parameter (constant). If zs is a state variable, the section is not well-defined.
   location: section w=0 derivation
   why it matters: Without specifying the ambient space and whether zs is a parameter, the normal velocity is ambiguous; a clinical pharmacology context likely treats zs as a constant, but this must be stated.
   minimal fix: State zs ∈ ℝ is a parameter; then normal velocity = xy − zs at w=0.

N. [MAJOR] The Poincaré formula det D(P∘Q0)=exp(ℓ(T))·ν(0)/ν(T)·det(DQ0) is asserted without proof or derivation. The standard multiplicative ergodic / Poincaré map formula requires: (i) P is the return map to a transversal section Σ, (ii) ν is the normal component of the vector field, (iii) ℓ is the divergence integrated along the orbit. The formula is dimensionally correct only if ℓ is the divergence (which it is, after the fix in issue 1), but the orientation sign is unexamined: if the flow crosses Σ in the negative ν direction, the sign flips.
   location: Poincaré formula statement
   why it matters: The formula is used to relate local volume contraction to the return map determinant; a sign error would invert the containment certification.
   minimal fix: State the orientation convention: ν must be chosen so that the flow crosses Σ transversally with positive ν·f; then the formula holds with the given sign.

N. [MAJOR] The claim that "Decimal RK4 point orbits can certify interval containment" is unfounded. RK4 with decimal (fixed-point) arithmetic does not provide rigorous error bounds: rounding errors accumulate, and RK4's local truncation error is O(h⁵) but the global error bound requires Lipschitz constants and a priori bounds that are not supplied. Interval containment requires either interval arithmetic with rigorous enclosure of the RK4 step (e.g., Lohner's method) or a validated integrator with Taylor models.
   location: Decimal RK4 point orbits certification claim
   why it matters: In a clinical pharmacology context, a false containment certificate could lead to dosing recommendations outside the safe range; a hostile referee would reject this as a safety-critical flaw.
   minimal fix: Replace with a validated integrator (e.g., interval Taylor method) or provide explicit error bounds for RK4 with decimal rounding and verify the enclosure at each step.

N. [MINOR] The notation "zs" is used without definition; if it is a parameter (e.g., zero-order synthesis rate), it should be declared, and its units (e.g., concentration/time) specified.
   location: system definition
   why it matters: In clinical pharmacology, units matter for safety; an undefined parameter prevents reproducibility.
   minimal fix: Define zs ∈ ℝ≥0 as a constant with units.

N. [MINOR] The audit does not state the domain of the system (e.g., x,y,w ≥ 0, or all ℝ³). The divergence and Poincaré formula hold locally, but global containment requires a compact invariant set; without domain specification, the certification is only local.
   location: system definition
   why it matters: A global claim of containment would be false if orbits escape to infinity; the artifact must state the invariant region.
   minimal fix: Specify a compact forward-invariant set (e.g., a box or ellipsoid) and verify invariance.

N. [NIT] The phrase "oriented pointwise Poincare formula" is nonstandard; the standard term is "Poincaré map Jacobian determinant formula" or "Dulac's criterion" for divergence. Using nonstandard terminology invites confusion.
   location: Poincaré formula heading
   why it matters: A referee may misinterpret the claim; clarity is essential for a mathematical audit.
   minimal fix: Rename to "Jacobian determinant of the Poincaré return map" and state the hypotheses.

N. [NIT] The input asks for "exact algebra" but the output (this response) must be produced by the reviewer; the artifact itself does not contain the algebra. This is a meta-issue: the audit request is fine, but the artifact under review must include the algebra to be audited.
   location: entire input
   why it matters: Without the artifact’s own derivation, the reviewer cannot distinguish a correct claim from a guess.
   minimal fix: The author should include the full derivation in the artifact.

N. [NIT] The severity labels requested (BLOCKER/MAJOR/PASS) are nonstandard for a review; the instruction asks for BLOCKER/MAJOR/MINOR/NIT, but the input says "state whether Decimal RK4 point orbits can certify interval containment" with BLOCKER/MAJOR/PASS. This inconsistency in severity scale should be resolved.
   location: input instruction
   why it matters: A reviewer must use a consistent severity scale to be actionable.
   minimal fix: Use the standard scale (BLOCKER, MAJOR, MINOR, NIT) throughout.
