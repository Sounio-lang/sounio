[OK] `event_covector` gradient definition
  Comment states grad(D)= (y,x,-1,0) for D=x*y-w-z_s; matches the four-component state layout used everywhere.

[OK] `fixed_kernel_candidates` pivot logic
  Assumes covector[2]==-1 exactly; code enforces this invariant before constructing the three kernel vectors.

[WRONG] `matrix_rank` Gaussian elimination
  Implementation divides by `scale=matrix[rank][column]` without checking that the chosen pivot is nonzero after prior elimination steps; can silently produce incorrect rank on ill-conditioned rational input.
  Minimal correction: add an explicit nonzero test (or tolerance) immediately after the `scale=` line and fall back to the next candidate column.

[OVERREACH] `kernel_projection` exactness claim
  Asserts that `dot(covector,projected)==0` after a single subtraction; this holds only because the supplied covector is exactly rational and the arithmetic is exact Fraction, not because of any interval enclosure.

[TIGHTENABLE] `carrier_normal_form` monomial test
  Current predicate rejects any monomial whose carrier part is nonzero yet primary part sums to >0; the surrounding reconditioning logic never produces such monomials, so the test can be strengthened to an assertion rather than a runtime filter.
