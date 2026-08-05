# Math review disposition

Both xAI/Grok 4.3 and Z.AI GLM-5.2 accepted the source-chart inverse and the
rigor of trying a larger event slab under unchanged strict Picard, derivative,
predictor, and interval-Newton predicates.

Both reviewers flagged the same major auditability gap: residual-variable
splits were not represented in the serialized source domain, so a downstream
consumer could not independently establish that the terminal children cover
their parent.

Disposition: fixed.  The worker now tracks exact rational bounds and split
lineage for all six variables (`XI`, `ETA`, `RHO0` through `RHO3`).  Every split
checks exact parent/left/right endpoint identities.  Before emitting a complete
tile receipt, the worker recursively verifies that terminal lineages form a
binary cover of the original tile.  The atomic runner rejects receipts without
that certificate.

The report was also tightened to call the ten partial objects provisional and
to state explicitly that only five branches of an incomplete family finished.
The `1/128` slab is described as the next alternative radius, not an additive
margin.  Provenance now includes the exact chart determinant, all source chart
parameters, dependency hashes, section, step, deterministic/no-RNG status,
acceptance predicate, and complete symbolic domain.

The focused recheck accepted the binary-domain induction.  Z.AI then made the
acceptance conditional on the TM2R parameter map being identity or affine and
requested serialized split midpoints.  The implementation is affine exactly:
`adaptive.split_state` substitutes old parameter
`-1/2 + new/2` for the left child and `1/2 + new/2` for the right child, then
verifies inverse-child reconstruction against the parent coefficients and
remainder.  The receipt now serializes the parent interval, cut, child interval,
and affine substitution for every lineage edge.

Z.AI also requested injectivity of the event projection because the dominant
split variable is measured on a projected carrier.  Disposition: disagreement,
not a soundness requirement.  The measurement produces only an integer
heuristic `variable`; regardless of how that integer was selected, both the
TM2R state and its symbolic parameter domain are split on that same pre-event
branch and variable.  Exhaustiveness follows from the affine binary partition,
not from injectivity of the heuristic's input map.
