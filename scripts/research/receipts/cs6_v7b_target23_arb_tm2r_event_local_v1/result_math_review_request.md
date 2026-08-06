# Focused result review request

Review only the exact arithmetic and inference scope below. Report any blocking
mathematical error or overclaim.

The hash-bound receipt has classification `UNRESOLVED_ENCLOSURE`, 44/44 passing
implementation checks, exact endpoint delay zero, and no complete transport.
At radius `r=1/128`, the raw, point-coefficient QR-reconditioned, and captured
crossing-event predictor intervals have exactly identical rational endpoints.
Their decimal display is

```text
L = -0.00783640191552156098975152664338
U = -0.00764670838956778967609913044129
```

and the lower slab boundary is `b=-1/128=-0.0078125`. Exact `Fraction`
arithmetic gives

```text
width = U-L = 0.000189693525953771313652396202087
boundary deficit = b-L = 0.0000239019155215609897515266433794
center = (L+U)/2 = -0.00774155515254467533292532854234
center clearance = center-b > 0
(width/2)/(center-b) = 1.33690840672555515729782656608
2(center-b)/width = 0.7479943988453678...
```

Please check:

1. Conditional on holding the center fixed, is it correct that the current
   width must be divided by a factor strictly greater than the first ratio,
   equivalently multiplied by a factor strictly less than the second ratio, for
   the interval to lie strictly above `b`?
2. Does exact equality of the three critical predictor intervals support the
   narrow conclusion that neither the tested point-coefficient QR transform nor
   production-vs-captured crossing endpoint timing changes this refusal?
3. Is it appropriately conservative to leave the cause as unresolved enclosure
   geometry or zero-centered slab placement, rather than declaring the event
   criterion causal?
4. Are the explicit nonclaims sufficient: no full support, covering relation,
   recurrent graph, degree/determinant edge, chaos, or open-problem solution?
