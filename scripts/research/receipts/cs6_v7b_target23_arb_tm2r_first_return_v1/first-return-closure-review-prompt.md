# Follow-up audit: first-return closure

The initial review accepted TM arithmetic, zonotope reconditioning, repeated
integration, and existence/orientation of the target crossing, but requested a
no-earlier-crossing argument before using the phrase "first return".

The strengthened run partitions every time step from `t=0` through the target
step by its validated Picard tube. For each tube it rigorously evaluates the
complete `w` interval and `w'=x*y-w-zs` interval. The retained exhaustive
classification is:

```text
initial departure tubes       1
prior strictly downward tubes 1
prior tubes excluding w=0     614
target upward tube             1
total                           617
```

The initial tube starts at the initial section `w=0`, ends with every endpoint
strictly positive, and has `w'>0` throughout. Every one of the 614 section-free
tubes has `0 notin w(B)`. The single remaining prior section-touching tube has
`w'<0` throughout, so any zero in it is a unique downward crossing. The target
tube starts with the complete endpoint set strictly at `w<0`, ends strictly at
`w>0`, and has `w'>0` throughout. The tubes cover consecutive steps without a
gap. All interval inequalities are strict.

Questions:

1. Does this exclude every positive-oriented section crossing after the
   initial departure and before the target tube?
2. Does `w'>0` throughout the target tube plus strict endpoint signs prove one
   and only one zero per represented trajectory in that tube?
3. Is "first positive return after the initial departure" now justified for
   every trajectory in the complete leaf?

Return BLOCKER with a concrete missing case if not. No second-return or chaos
claim is made.
