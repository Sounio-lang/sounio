[OK] DIVERGENCE = x - y - w + zs/2 - 1
  Matches trace(J) of the supplied (x,y,w) vector field; ell-component is exactly the divergence.

[OK] determinant = exp(∫div) * (initial_normal/final_normal) * q0_area
  Standard Liouville area factor for the oriented 2-form transported by the flow; normal velocities are the same bilinear form evaluated at the section points.

[OK] dyadic center construction
  Midpoint of the dyadic rectangle in (u,s) coordinates, linearly mapped by the unstable/stable frame; arithmetic is exact in Decimal.

[OK] RK4 + negative-to-nonnegative bisection (48 steps)
  Classical; localization error bounded by 2^-48 * step, negligible at the stated precisions.

[OK] coarse/fine self-consistency test (Δ ≤ 1e-16, both det < 0, normals > 0)
  Correct numerical sanity filter; does not constitute an enclosure.

[TIGHTENABLE] "independent pointwise falsification of retained CAPD Liouville enclosures"
  The experiment is a high-precision floating-point scout that can falsify but cannot certify containment; the contract and summary correctly label it "POINTWISE_FALSIFICATION_ONLY=true" and "RIGOROUS_INTERVAL_CERTIFICATE=false". No claim exceeds this scope.

[PASS] verifier soundness & mutation gate
  All 14 mutations (including determinant sign flip, containment flag, provenance tampering) are rejected by the verifier; no escape paths.
