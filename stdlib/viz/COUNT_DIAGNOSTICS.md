# CountGLM computation-to-figure diagnostics

`viz::count_diagnostics` consumes a checked `DynamicCountFit`, the original
integer counts and that fit's mean buffer. It does not refit, estimate alpha,
assess assumptions, calculate inference or substitute normalized values for raw
count observations. The caller is responsible for keeping the fit record and
buffers from the same computation; this API validates availability, shapes and
payloads rather than certifying record provenance.

`count_diagnostic_prepare(fit,counts,means,observed,residual,observed_mask,residual_mask)`
fills caller-owned buffers with observed counts as f64 and signed saturated
NB2/Poisson deviance residuals, using the fit's supplied alpha. Negative counts,
counts above2^53, nonpositive/nonfinite means and invalid dimensions reject before
ANY output mutation. The2^53 bound makes the plotted integer conversion exact.
A finite signed residual is available independently of whether its square
(deviance) is representable. Arithmetic-unavailable residuals receive zero
placeholders with mask0, explicit unavailable count and first status; they are
never interpreted as actual zeros. Observation masks are1 for all valid rows.

A reusable private deviance slot is constructed once per preparation, reused in
two arithmetic passes. The first pass resolves availability/envelopes before the
second commits outputs. There is no per-observation rich snapshot. Workspace
payload is explicit32*n bytes: two f64 outputs and two i64 masks, excluding
caller input owners and constant-sized metadata/slots. The same budget is
available BEFORE allocation through
`count_diagnostic_workspace(n)` (bytes, or-1 for invalid/unrepresentable shape).
Preparation returns
`CountDiagnosticData` status0 available,1 unavailable/invalid fit,2 owner/shape/
budget,3 source payload. Unavailable result metadata is not permission to render.

`count_diagnostic_plan(data,width,height)` requests a complete document with
observed count versus fitted count (identity reference), and signed deviance
residual versus fitted count (zero reference). The mean X domain is common.
The count domain includes0; the residual domain is symmetric. All-zero residuals
receive a declared[-1,1] display window. If a symmetric span is not representable
or labels do not fit, the plan is unavailable rather than silently cropped.

`count_diagnostic_svg` and `count_diagnostic_canvas` accept fit,data,plan,text,
label bytes and the prepared owners. They verify shape/alpha/mask-count consistency
and then use the scientific document's GLOBAL preflight. Status30 means that
the fit/prepared-data/document linkage is invalid. No output is emitted in that
case. Both masks and all active late rows are checked before any frame output.
Titles and footer must honestly state variables, units, supplied alpha and the
descriptive purpose; units for residuals are dimensionless. The SVG also carries
supplied-alpha, unavailable-residual, assumptions-assessed=false and
inference-available=false metadata. Alpha uses scientific15 output, which is
readable finite formatting, not a round-trip/accuracy certificate.

These are point-fit diagnostics. No CI, goodness-of-fit test, overdispersion
assessment or finite-sample calibration follows from the figure. Reducing a
diagnostic for display would not authorize reducing input used for inference.

The prepared native integration fits1028 raw rows with two independent indicator
columns, zero offsets, fixed alpha.25 and raw counts repeating0,2,4,6. It consumes
the actual fitted means in BOTH backends. An independent C++23 reference uses
the closed-form group means2 and4 and direct NB2 deviance arithmetic; geometry
uncertainty is propagated from an explicit2e-8 relative-plus-absolute numeric
fixture budget before integer truncation (no blanket pixel tolerance).
Global late-error zero-output/unchanged-Canvas witnesses are included.
All numerical execution and actual-output concordance remain pending root's
fresh-compiler Slurm gate. No runtime, inference or performance result is claimed
from unexecuted fixtures.
