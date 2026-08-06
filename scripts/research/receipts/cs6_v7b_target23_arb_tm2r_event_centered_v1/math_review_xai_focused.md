[OK] predictor midpoint retained as exact rational from frozen 1/128 slab
  Midpoint extraction via `Fraction(exact_fraction(mid()))` is exact; no rounding.

[OK] signed Picard self-containment + strict contraction at Taylor order 12 + Arb remainder
  Standard rigorous enclosure for non-autonomous flow; contraction supplies uniqueness of the fixed-time map.

[OK] interval-Newton image strictly inside a-priori residual domain R
  Classic existence proof for a zero of the event function inside every parameter; the factor-1/2 margin guarantees strict inclusion.

[OK] whole-tube derivative strictly negative ⇒ uniqueness gate
  Sign condition on the Lie derivative of the event function over the symmetric tube precludes multiple roots.

[OK] exact w=0 projection + strictly negative normal
  Direct verification that the section is transverse and the projected carrier lies on the surface.

[OK] positive aggregate coefficient weights for all six variables
  Sufficient (if coarse) syntactic certificate that no symbolic dependence was lost under reconditioning or projection.

[TIGHTENABLE] reconditioner string in verifier
  Hard-coded path `cs6_v7b_target23_arb_tm2r_subdivided_second_return_worker.point_coefficient_recondition` should be derived from `adaptive.__module__` rather than frozen; harmless for this receipt but brittle.

All other checks are defensive, fail-closed, and correctly implement the stated contract. No downstream mathematical claim is invalidated.
