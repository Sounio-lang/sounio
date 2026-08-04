# Independent math review: Arb TM2R event projection

Review the mathematical soundness and claim boundary of the implementation in
`scripts/research/cs6_v7b_target23_arb_tm2r_second_return_worker.py` together
with its retained receipt and research note. Be adversarial. Classify findings
as BLOCKER, MAJOR, MINOR, or PASS.

## Validated premises

- The complete two-source, four-residual degree-2 carrier has a rigorous first
  negative-to-positive return bracket in one step of width `h=2^-8`.
- Its pre-step `w` interval is strictly negative and its post-step `w` interval
  is strictly positive.
- On the complete Picard event tube,
  `w'=x*y-w-zs` is strictly positive. Thus each source point has exactly one
  event in that step.
- Every autonomous step is accepted only after a Picard self-map, strict
  contraction, and containment of the raw Taylor endpoint in the Picard tube.

## Projection construction to review

At the positive endpoint, with full-leaf endpoint enclosure `W_end` and the
complete event-tube derivative enclosure `D`, the code forms

```text
Delta = -W_end / D subset [-h,0].
```

For each actual source point, the mean value theorem and strict monotonicity are
intended to imply that its exact backward event time lies in `Delta`.

The code then splits `Delta = delta_mid + delta_res`, where `delta_mid` is an
exact Arb midpoint and `delta_res` is a symmetric interval. It propagates the
complete endpoint TM backward by `delta_mid` using a signed Picard tube and an
order-12 autonomous Taylor enclosure. For `x,y,ell`, it adds
`F_i(complete event tube) * delta_res` as an interval remainder. Since all
times between the endpoint and every member of `Delta` lie inside the original
event tube, this is intended as a mean-value enclosure of the residual flow.
After event existence and uniqueness have already been proved, it replaces the
`w` component by exact zero and reconditions twice with the existing rigorous
QR-derived zonotope hull.

The projected carrier retains 15 nonzero pure source monomials involving only
`xi,eta` in `x,y,ell`; all residual event-time dependence is intervalized. The
claim is therefore only that explicit source dependence survives, not that the
exact symbolic event-time graph survives.

## Retained outcome

- `Delta = [-728214639/274877906944, -179021577/68719476736]`.
- Projection Picard contraction upper bound is strictly below one.
- Projected `w` is exactly zero; max carrier width is
  `539659285/1099511627776`; 15 pure source monomials survive.
- The same degree-2 carrier validates 707 further steps.
- The 708th post-event step closes its Picard tube but the raw TM endpoint
  escapes that tube, so the step is refused and never reconditioned or used.
- `FULL_LEAF_SECOND_RETURN_CERTIFICATE=false`; no determinant, covering,
  chaos, attractor, novelty, priority, or open-problem claim is made.

Please answer these exact questions:

1. Does strict positivity of `w'` on the complete event tube plus the endpoint
   sign bracket justify existence and uniqueness for every source point and the
   quotient enclosure `-W_end/D`?
2. Is the fixed midpoint backward flow plus `F(event tube)*delta_res` a valid
   enclosure of every source-dependent event point, assuming the full time slab
   remains in the original validated event tube?
3. Is setting `w=0` exact after those obligations sound, or does it discard an
   uncertainty needed in the other coordinates?
4. Is “preserves explicit xi,eta dependence” appropriately narrow given that
   event-time/source correlation is placed in an interval remainder?
5. Does the fail-closed 708th-step refusal support only a bounded-method result,
   with no negative claim about the physical return?

Do not infer a second return or chaos. Identify any missing containment check,
dependency mistake, sign error, or overclaim that must be fixed before commit.
