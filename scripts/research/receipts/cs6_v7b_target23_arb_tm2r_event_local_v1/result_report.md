# Target-23 TM2R event-local diagnostic

## Objective

Discriminate, before another complete transport, among three hypotheses for the
persistent `PREDICTOR_ESCAPED` refusal: chart drift, a mismatch in the event
criterion, or an implementation defect. The diagnostic instruments the
unchanged production first-return integration and preserves the same exact
affine substitutions in `xi`, `eta`, and `rho0` through `rho3`.

## Foundry execution

The final diagnostic ran as Slurm job 8727 on
`gpuorangefs-multi-r740-proxmox` from source snapshot `4d89ba3f82`. It completed
with exit code `0:0` in `00:39:45`; the receipt SHA-256 is
`0a47c711f8442bfe4bb3ce844cb247c74aaf9922f5dcf224411f58acd2bb3146`.
The independent verifier accepted the receipt, and all 11 negative mutations
were rejected.

The first completed run, job 8723, is retained as `initial_*`. Its
`IMPLEMENTATION_INCONSISTENCY` verdict was a diagnostic-control false positive:
it required the returned root endpoint to be contained in the refined crossing
tube and required bitwise identity between two equivalent Arb expressions.
Production `advance_with_endpoint_intersection` only requires interval overlap,
and overlap, rather than expression identity, is the rigorous invariant. Version
2 replaces these controls by capturing successful production steps, recomputing
the stored crossing derivative, and comparing the production endpoint with the
last strictly downward tube containing `w=0`.

## Result

The final classification is `UNRESOLVED_ENCLOSURE`.

- All 44 implementation controls pass.
- The production endpoint and captured crossing event have the same exact
  reference time, `389/256`; their delay is exactly zero.
- Raw, point-coefficient QR-reconditioned, and captured-crossing diagnostics all
  refuse with `PREDICTOR_ESCAPED`.
- At radius `2^-7 = 1/128`, their predictor intervals are exactly identical as
  rational endpoints.
- Delayed anchored interval Newton and joint reanchoring plus QR also refuse.
- No point fallback, box flattening, or complete transport was performed.

The common predictor at radius `1/128` is approximately

```text
[-0.00783640191552156098975152664338,
 -0.00764670838956778967609913044129].
```

It straddles the lower slab boundary `-1/128 = -0.0078125` by
`0.0000239019155215609897515266433794`. Its width is
`0.000189693525953771313652396202087`. If its center is held fixed, containment
therefore requires dividing the current predictor width by a factor strictly
greater than

```text
1.33690840672555515729782656608.
```

Equivalently, the new width must be strictly less than
`0.7479943988453678...` times the current width. Equality in either formulation
only touches the boundary and is not sufficient for strict containment.

These decimal values are displays of exact `Fraction` calculations retained in
`analysis.txt`; the acceptance decision does not use decimal rounding.

## Discrimination

This run rules out the tested chart-drift mechanism: the active
point-coefficient QR reconditioner does not change the critical predictor
interval. It also rules out endpoint-time drift between the production return
and the captured crossing event for this branch. The enumerated implementation
defects are absent under 44 fail-closed checks.

It does **not** prove that the event criterion itself is the cause. The surviving
hypothesis is narrower: dependency or placement in the zero-centered event-time
slab, or enclosure geometry not altered by the current QR carrier. The exact
classification remains `UNRESOLVED_ENCLOSURE`.

## Next falsifier

Build a predictor-centered event-time chart at the exact rational predictor
center, retaining `xi`, `eta`, and `rho0` through `rho3`. Require signed Picard
self-containment, a strict whole-tube event derivative, and parametric interval
Newton containment. The quantitative target is to divide the critical predictor
width by more than `1.336908406725555...` (equivalently, multiply it by less
than `0.7479943988453678...`), or obtain an equivalent recentering containment.
Do not launch a new complete transport until this event-local gate passes.

## Claim boundary

This receipt certifies no complete support, h-set target, covering relation,
recurrent graph, degree, determinant edge, chaos, novelty, priority, or solution
of an open problem. Existing production and legacy transport paths are retained
unchanged.
