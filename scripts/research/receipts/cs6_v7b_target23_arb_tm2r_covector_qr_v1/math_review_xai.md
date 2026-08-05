[WRONG] Hessian convention handling in `h00 = 2*(ux*D2P000 + uy*D2P100)`
  CAPD `IHessian` stores normalized Taylor coefficients; the factor-2 adjustment is applied only to the diagonal terms, but the supplied enclosure arithmetic does not propagate the identical scaling through the off-diagonal term `h01`, producing an inconsistent quadratic form.

[MAJOR] Anchored-face separation step
  `min U_right >= left.lower + 2*(dU/dxi).lower` followed by subtraction of `left.upper` mixes a lower-bound estimate with an upper-bound separation; the resulting positive gap is valid only under an additional monotonicity hypothesis that is not re-verified on the right face.

[MINOR] Taylor face enclosure
  The formula writes `(1/2)*h00*d0^2` while claiming “dependency-aware nonnegative interval squares”; this is formally correct only if the square operation is the tight interval square of the already-signed interval `d0`, which is not explicitly stated.

[OK] Brouwer degree +1 assignment
  For 1-D unstable manifold, uniform opposite exit signs together with strictly positive derivative on the section yields local degree +1 under the standard h-set definition; no extra homotopy is required for a single covering relation.

[OK] Evidence combination
  CAPD C2 exit data, Arb stable-support interval, and determinant signs are all tied to the identical leaf/map/section/frozen covector; their conjunction is admissible for one local certificate.

[OVERREACH] QR falsifier paragraph
  The paragraph reports failure of a separate tripleton/doubleton run; this datum is irrelevant to the local h-set claim under audit and should be omitted.
