# Math review recheck: symbolic split coverage

The first xAI+ZAI review accepted the chart inverse and strict event-slab logic,
but correctly rejected the auditability of residual-variable splits because the
receipt tracked only `xi,eta`.

The worker now represents a symbolic domain by exact rational center/radius for
all six variables `XI, ETA, RHO0, RHO1, RHO2, RHO3`, plus an immutable binary
split lineage.  For a split of parent interval `[c-r,c+r]` in variable `v`, it
constructs

```
left  = [c-r,c]
right = [c,c+r]
```

and asserts exact rational endpoint equality.  All orthogonal variable bounds
must remain exactly equal to the parent bounds.  The same variable index is
used for the existing algebraically checked `adaptive.split_state(parent,v)`
and for the symbolic-domain split.  A variable may be selected by measuring a
post-projection carrier, but the actual state and domain partition are both
performed on the same pre-event branch and the selection does not alter the
partition identity.

Before emitting JSON, `certify_terminal_domain_cover` examines every terminal
lineage relative to the source-tile root.  Recursively, a node is accepted only
if it is terminal with no descendants, or if its next lineage tokens are
exactly `{variable+L, variable+H}` for one variable and both subtrees pass.  A
test confirms that `[left,right]` covers a root while `[left]` raises
`DOMAIN_COVER_INCOMPLETE`.

Each carrier serializes exact bounds for all variables and its split lineage.
The payload sets `terminal_domain_cover_certified=true` only after this check;
the atomic runner rejects a receipt without it.

Please determine whether this closes the earlier MAJOR issue.  Look especially
for a mismatch between the TM2R reparameterization and symbolic-domain split,
or a lineage pattern that the recursive check could incorrectly accept while
leaving a hole or double-counting an uncovered parent region.  Also verify that
the report remains conservative: the interrupted local run emitted no complete
tile receipt and claims no covering relation, recurrence, or chaos.
