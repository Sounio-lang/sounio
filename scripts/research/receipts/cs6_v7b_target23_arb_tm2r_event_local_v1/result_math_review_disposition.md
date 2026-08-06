# Result math-review disposition

## Initial xAI review

The first result review accepted the interval-equality inference, conservative
classification, and explicit nonclaims. It correctly identified that the phrase
"width-reduction factor" was ambiguous: `1.336908...` is the divisor of the old
width, while its reciprocal `0.747994...` is the multiplier applied to the old
width.

The analyzer and report now name and retain both exact quantities:

- `MINIMUM_WIDTH_DIVISOR = (width/2)/(center-boundary)`;
- `MAXIMUM_WIDTH_MULTIPLIER = 1/MINIMUM_WIDTH_DIVISOR`.

Strict containment requires division by a value strictly greater than the first,
equivalently multiplication by a value strictly less than the second. Equality
only touches the boundary.

## Focused xAI re-review

The focused re-review marked every arithmetic and scope item `OK`: exact width,
boundary deficit, center and clearance; the divisor and reciprocal multiplier;
the center-fixed containment argument; equality of the three predictor
intervals; the non-causal unresolved classification; and all explicit
nonclaims.

## Provider status

Z.AI returned usage-limit error 1308 on the initial result review and empty
artifacts on the immediate retry. It supplied no mathematical opinion. The
mandatory xAI math review is substantive and green after the focused correction.
