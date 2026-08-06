[OVERREACH] containment of full anchored Picard slab in original crossing tube justifies `X_c` as event-local chart anchor
  containment alone does not entail locality or uniqueness of the zero-crossing chart; a monotonicity or isolation argument on the tube is required

[OK] exact affine 12-split substitutions applied identically to `X_c` and `X_d`
  substitutions are linear and therefore preserve the source subdomain exactly when performed on the same branch

[TIGHTENABLE] logged controls (split variable, containment, derivative equality, flow coefficient match)
  these detect implementation defects but do not separate them from genuine over-approximation; an additional invariant on the width of the Taylor remainder term after reconditioning is needed

[WRONG] `p = -X_w / mid(D)` with `sup(D)<0` and subsequent interval Newton step
  sign of the predictor is inverted relative to a standard downward crossing (`D` should appear with opposite sign or the formula should be `X_w / mid(-D)`); minimal correction: replace by `p = X_w / mid(-D)` while retaining the strict `sup(D)<0` test

[OK] fail-closed classification table (raw/refuse patterns -> CHART_DRIFT, EVENT_CRITERION, MIXED, ...)
  exhaustive case analysis on the three tests matches the stated decision logic with no omitted combinations
