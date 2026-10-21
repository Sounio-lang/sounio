# Conditional NB2 residuals with supplied parameters

This synthetic executable compares Pearson NB2 residuals with signed saturated
NB2 deviance residuals for counts0,1,2,3,4,6,8,12,16, supplied mean4 and alpha0.25.
No model or dispersion is fitted. The horizontal coordinate is observed count;
the vertical coordinate is a dimensionless signed residual. Both series share a
viewport and an explicit zero reference. Lines join nine discrete counts and do
not assert a continuous accuracy envelope.

The program checks every required diagnostic/scene status before consuming its
payload. Supplemental parameter scenarios are printed without expected-result
assertions: alpha0 Poisson, a very large mean, local very large alpha and a
subnormal-product regime. Their availability states are descriptive, not an
independent numerical reference or a boundary-coverage claim. Default decimal
printing does not resolve every tiny residual.

Two1024-element plotting buffers and one64-byte label buffer are reused. Scene
builders copy active prefixes into owned storage. SVG-only emission avoids Canvas
allocation; it does not imply that the whole program is allocation-free.

The new scalar API has independent availability for the residual and its square.
A successful residual must be checked separately from deviance_status before
consuming the squared deviance. A mathematical result may be representable even
when a conservative intermediate guard reports arithmetic unavailable.

[Module contract](../stdlib/stats/regression/NB2_DEVIANCE.md).
[Execution receipt](../docs/plans/NB2_DEVIANCE_EXECUTION_2026_10_22.md).
No p-values, goodness-of-fit acceptance, confidence intervals, global accuracy,
comparative performance, numerical concordance or inference calibration follow
from this visualization.
