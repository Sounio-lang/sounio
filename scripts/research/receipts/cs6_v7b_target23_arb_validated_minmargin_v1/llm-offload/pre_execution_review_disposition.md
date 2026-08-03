# Pre-execution review disposition

## Applied findings

- Every decimal parameter is now parsed as an exact base-10 `Fraction` and
  enclosed through Arb integer division.
- Every Picard self-map check now also requires the explicit Banach condition
  `h * L_infinity(B) < 1`; the maximum contraction factor is emitted exactly.
- Initial-state containment, Picard contraction, and event transversality are
  explicit fail-closed output obligations.
- The predeclared accounting now fixes 1791 `advance` calls, 1793 Picard calls,
  and one ambiguity-limited event bisection.

## Mathematical resolution

Grok returned PASS on all seven requested obligations. Z.AI returned an empty
artifact, so DeepSeek was used as the independent fallback.

DeepSeek's remaining BLOCKER labels are documented disagreements:

- The first Jacobian row is `[-y, 4*y-x, 0, 0]`, so its infinity logarithmic
  norm row is exactly `-y + abs(4*y-x)`. DeepSeek's alternative differentiated
  `2*y*y-x*y` with respect to the wrong variable.
- `field(box)` is evaluated by Arb over all four interval components, not at a
  point or at sampled vertices. The Picard image is checked for containment.
- The polynomial is truncated after order 40. Therefore Taylor's componentwise
  remainder uses derivative 41 divided by `41!`, exactly the interval extension
  `a_41(B) * h^41` computed by the recursion. A derivative-42 term would apply
  only after retaining the order-41 term.
- The low event state is a validated propagation from the fixed step start.
  Its Picard box over `[low, high]` contains the whole trajectory segment and
  therefore every possible event state. `x*y-zs` is evaluated over that entire
  box; its positive lower bound supplies uniform transversality and uniqueness.
- Arb radii from every Taylor-polynomial operation are added to the propagated
  radius. Arb supplies outward rounding for all interval operations.
- `lower()` and `upper()` return exact binary endpoints; `fmpq()` serializes
  those exact endpoints rather than rounding an inexact ball.
- The chart factor is an oriented two-dimensional determinant, not a Euclidean
  norm area, so it requires no orthogonality assumption.
- CAPD containment is an additional compatibility gate. The center-orbit
  enclosure is first established by the worker without reading CAPD.

## Scope after review

The positive result, if execution passes, certifies only the exact frozen center
trajectory for one leaf under the frozen decimal parameters. It is not a
leaf-wide enclosure, a full independent leaf engine, global H-PG, V7-B,
novelty, promotion, FPGA evidence, or a solution of an open problem.
