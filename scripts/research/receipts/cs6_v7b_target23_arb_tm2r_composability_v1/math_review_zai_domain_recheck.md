```
[OK] Split endpoints: left=[c-r,c], right=[c,c+r] cover parent [c-r,c+r] with measure-zero overlap at c.
  Union equals parent; rational arithmetic preserves endpoint equality.

[OK] Orthogonal-coordinate invariance: bounds on the five non-split variables are unchanged by a single-axis split.
  This is the standard kd-tree invariant; design assertion, not derived.

[OK] Recursive cover check is sound for axis-aligned binary partitions of a 6-D box.
  Induction on depth: root covers the box; if children are exactly {v+L,v+H} of one parent interval, their union recovers the parent along v and matches on the other 5 axes; both subtrees covering their halves ⇒ parent covered. Measure-zero point overlaps do not break covering.

[OK] `[left]` correctly classified as DOMAIN_COVER_INCOMPLETE.
  Missing the (c,c+r] half; failure is forced by the "exactly {v+L,v+H}" guard.

[OVERREACH] "The same variable index is used for adaptive.split_state(parent,v) and for the symbolic-domain split."
  Variable-index coincidence is necessary but not sufficient. If TM2R reparameterizes variable VALUES (e.g., log/logit/non-affine map), splitting the symbolic interval at its rational midpoint c does NOT split the adaptive state interval into halves meeting at the preimage of c. Exact rational endpoint equality in symbolic coords ≠ endpoint equality in adaptive coords. This silently breaks the "no hole" guarantee precisely in the case the artifact asks the reviewer to hunt for.
  Correction: require TM2R restricted to each variable to be identity, or an affine map with rational slope/intercept; otherwise the symbolic partition must be the preimage of the adaptive partition (and the recursive check must verify that preimage, not the symbolic midpoint).

[OVERREACH] "A variable may be selected by measuring a post-projection carrier ... the selection does not alter the partition identity."
  Holds iff the slab-to-carrier projection is injective on the current branch. If the projection collapses distinct pre-states to one carrier value, the selection variable becomes ill-defined as a function of pre-state and the partition identity claim has no referent.
  Correction: state and check injectivity of the projection on the pre-event branch, or restrict variable selection to coordinates already present in the pre-event state.

[OK] Conservation of interrupted run: "emitted no complete tile receipt and claims no covering relation, recurrence, or chaos."
  Procedural/conservatism claim; no mathematical content to falsify. Acceptable as stated.

[TIGHTENABLE] "Each carrier serializes exact bounds for all variables and its split lineage."
  For the cover certificate to be reproducible, the serialized midpoint used in each split must also be present (not just L/H tokens and end-bounds). Otherwise an auditor cannot recompute c from [c-r,c]∪[c,c+r] and confirm the rational midpoint agrees with adaptive.split_state's chosen split point.
  Suggested: serialize (center,radius) per node, or serialize the split midpoint explicitly alongside L/H.
```

Bottom line on the original MAJOR: the residual-variable auditability gap is closed *for axis-aligned, identity-coordinate partitions*. It is **not** closed if TM2R reparameterizes variable values, or if the carrier projection is non-injective on the branch. The two OVERREACH items above are the residual holes; both are checkable in code with a one-line guard each, but neither guard is present in the artifact as written.
