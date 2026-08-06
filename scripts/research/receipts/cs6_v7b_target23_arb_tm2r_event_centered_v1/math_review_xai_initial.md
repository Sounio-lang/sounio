[OVERREACH] parametric Newton correction in `centered_event_chart`
  `correction = -predicted_state[2].range() / derivative` followed by `with_remainder` only produces a candidate enclosure; no Krawczyk operator, no verified Newton contraction mapping, and no explicit 0 ∉ N'(X) test is performed to guarantee a unique zero inside the slab.

[OK] signed Picard usage
  `signed_picard_box` called symmetrically on ±radius with explicit contraction and derivative.upper()<0 checks; consistent with the contract.

[OK] symbolic-dependence preservation
  `variable_weights`, `retained_source_monomials`, and the six positive-weight gates are enforced on critical/centered/projected carriers; matches the stated requirement.

[TIGHTENABLE] existence/uniqueness for the event-time root
  The residual-slab test after correction is necessary but not sufficient for a rigorous existence proof; add an interval-Newton or Krawczyk acceptance predicate before declaring ACCEPTED.
