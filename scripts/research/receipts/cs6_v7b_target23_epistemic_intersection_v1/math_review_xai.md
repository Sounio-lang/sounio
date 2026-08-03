[OK] nonempty joint intersection of the six determinant enclosures (C1/C2/affine/resident-reconstructed/homogeneous/Liouville) certifies common sign when one enclosure (Liouville) is strict-negative
  max-of-lowers ≤ min-of-uppers on the six intervals is exactly the nonempty-intersection predicate; combined with `same_strict_sign` against Liouville this forces the intersection interval itself to lie in (−∞,0).

[OK] scaling/orientation consistency among the six enclosures
  `homogeneous_determinant`, `reconstructed_determinant`, `affine.determinant` and `liouville.determinant` are all derived from the same `frame_det·radius_u·radius_s` seed and the same two-return chain; the explicit `total_scale0/1` and `event*_reconstructed_rays` factors close the loop.

[TIGHTENABLE] “200 retained attempts” statement
  Report correctly limits every count to the 200-run cohort; no universal claim is made, but adding the explicit qualifier “in the 200-run sample” would remove any possible misreading.

[OK] exact binary64-endpoint analyzer
  Treating IEEE-754 endpoints as exact dyadic rationals yields a sound (if pessimistic) interval enclosure for the algebraic expressions shown; no rounding-mode error is introduced inside the reported predicates.
