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
independent numerical reference or a boundary-coverage claim. Residual and deviance payloads use the existing io::sci_print::println_f64_sci
formatter with 15 significant decimal digits. This scientific notation keeps
tiny nonzero printed magnitudes visible; it is not a binary64 roundtrip
representation, an accuracy certificate, or independent numerical verification.
The formatter normalizes signed zeros to positive-zero text. Statuses are
printed before payloads: unavailable residuals or squared deviances print the
word unavailable rather than an error payload zero. Both statuses and, when
available, both scientific payloads are reported for every supplemental
scenario, including count8, mean1e-308 and alpha1e-15. Availability is observed
from the kernel; no expected status or value is asserted.

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

## Scientific report formatting increment

This increment changes report formatting and status-gated supplemental output;
the plotted observations, graph geometry, domains, scene nodes and labels are
unchanged. It adds no tests or numerical oracle. Source preparation alone does
not establish typecheck, execution, correctness of the formatter, or acceptance
of a new arithmetic regime. Existing execution receipts describe their frozen
sources and do not automatically validate this revised report.

## Balanced continuation native checkpoint

Slurm15874 rebuilt the source-tracking seed/modular compiler from frozen eeeab3025c,
passed five explicit typechecks including the formatter, and compiled/executed the
real example with exit0. Scientific output preserves a visible nonzero local
large-alpha residual. The far subnormal-product scenario now reports both statuses0
and scientific payloads; the preceding frozen report recorded status5. These are
observed descriptive outputs, not independent numerical agreement. The current SVG
is byte-identical to the preceding figure; XML parsing and full1040x620 image
inspection completed. No tests or numerical oracle were run.
[Balanced continuation receipt](../docs/plans/NB2_BALANCED_EXECUTION_2026_10_22.md).
