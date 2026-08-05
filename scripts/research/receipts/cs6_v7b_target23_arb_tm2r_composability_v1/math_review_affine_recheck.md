# Final math review: affine TM2R split bridge

The remaining review condition was whether the symbolic split matches the
actual TM2R parameter reparameterization.

The dependency implementation is:

```python
left = reparameterize_component(parent, variable, -1/2, 1/2)
right = reparameterize_component(parent, variable, 1/2, 1/2)
inverse_left = reparameterize_component(left, variable, 1, 2)
inverse_right = reparameterize_component(right, variable, -1, 2)
```

Here `reparameterize_component(p,v,c,r)` substitutes `old_v = c + r*new_v`
in every degree-2 polynomial monomial using the binomial formula, leaving the
interval remainder unchanged.  It verifies that each inverse child's
coefficients and remainder enclose the parent.

The new symbolic-domain code applies the corresponding global rational split:
left `[C-R,C]`, right `[C,C+R]`, leaving all orthogonal variables unchanged.
Every carrier serializes a trace entry containing parent bounds, cut, child
bounds, and the local TM2R substitution center `side/2` and radius `1/2`.

The terminal-lineage verifier accepts a node only if it is a leaf without
descendants or has exactly `variable+L` and `variable+H` children and both
subtrees pass.  Therefore the terminal parameter domains cover the root tile.

Variable choice may be measured from a post-projection carrier.  This value is
only an integer heuristic.  The actual `split_state` and `split_domain_pair`
both operate on the same pre-event branch and the selected integer variable.
Projection injectivity is not used in the partition proof: any selected axis
would yield an exhaustive left/right parameter partition.

Does this close the prior MAJOR auditability issue?  Report any remaining
soundness flaw in the affine bridge, inverse reconstruction, serialized trace,
or terminal cover induction.  The interrupted run still emits no complete tile
receipt and makes no claim of a covering relation, recurrence, or chaos.
