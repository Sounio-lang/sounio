# Poisson means and NB2 moment dispersion: descriptive example

Source: `viz_nb_dispersion_diagnostic.sio`.

## Implementation status

This example passed typecheck with a current-source Madaros build in Slurm
15747, exit 0:0. See the [typecheck receipt](../docs/plans/NB_DISPERSION_TYPECHECK_2026_10_20.md).
It has not been compiled to a program ELF, executed, numerically verified or
visually inspected as part of this change. It adds no test and does not extend
a native runtime or scientific acceptance claim.

## Data and model

Eight synthetic observations belong to two groups, A (rows 1–4) and B
(rows 5–8). Counts are [0, 1, 2, 20, 1, 3, 4, 28].
Known exposure durations in seconds are [1, 2, 4, 8] in each group.
All counts are small exactly representable binary64 integers; the general
kernel's count conversion may round integers above 2^53.

The design explicitly contains an intercept and a group-B indicator.
The example calls `poisson_fit_offset` with the logarithms of exposure as
offsets. It uses the returned count means, including exposure, directly in
`nb2_moment_dispersion`; it declares the two fitted mean parameters.
It never substitutes rates for count means in the dispersion calculation.

The moment API estimates the signed NB2 moment coefficient for
Var(Y) = mu + alpha * mu^2 and separately returns its nonnegative projection.
The n/(n-p) adjustment uses the declared mean parameter count; it does not
establish unbiasedness. A negative signed estimate and truncation describe
this moment equation, not evidence from a test that dispersion is zero.

## Outputs

The textual report includes fit and dispersion statuses, signed alpha,
projected alpha, truncation, descriptive Pearson X2, the declared parameter
count, residual degrees of freedom, and both assumption-assessment flags.
Failures stop before presenting a figure; fit, dispersion, capacity and allocation
failures print an unavailable status, while viewport refusals return without one.

The left panel shows observed count/exposure and fitted count mean/exposure,
in events per second. The right panel shows the **Poisson** Pearson residual
(y-mu)/sqrt(mu), with a zero reference line. It deliberately retains Poisson
residual scaling even when the moment estimate is positive. The panels use
shared explicit domains within each panel and checked scatter/line viewports.

The example reuses one order array and one ordinate scratch array; chart
builders copy active prefixes before the next scratch update. Labels use
padded byte literals. The Canvas allocation is checked before rendering and
freed after HTML emission. No raster file is saved. Standard output
contains the textual report followed by the HTML/SVG document. To obtain a
standalone HTML artifact, separate the report prefix from the emitted HTML;
the whole stdout stream is not claimed to be a clean standalone document.

## Scientific limits

This is not an NB likelihood regression fit, dispersion shrinkage procedure,
confidence interval, p-value, differential-expression analysis, or model
selection procedure. Means were estimated from the same observations.
Independence, the mean model, NB assumptions, and exposure adequacy remain
unassessed. The observed visual pattern and moment estimate do not certify
model adequacy. No inferential interval or uncertainty band is displayed.

The existing `tests/run-pass/viz_poisson_offsets.sio` supplies the conventions
for the Poisson rate/residual view; its historical execution evidence does
not verify this new example. The renderer's production acceptance also does
not establish this example's numerical or visual correctness.
