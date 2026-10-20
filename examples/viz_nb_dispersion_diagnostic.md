# Poisson means and NB2 moment dispersion: descriptive example

Source: `viz_nb_dispersion_diagnostic.sio`.

## Implementation status

The earlier Poisson-only residual version passed typecheck in Slurm 15747;
see the [historical receipt](../docs/plans/NB_DISPERSION_TYPECHECK_2026_10_20.md).
That result does not verify this revised dual-residual version. This change
has not been typechecked, compiled, executed, numerically verified, or
visually inspected by its author. It adds no test and does not extend a
native runtime or scientific acceptance claim.

## Data and model

Eight synthetic observations belong to two groups, A (rows 1–4) and B
(rows 5–8). Counts are [0, 1, 2, 20, 1, 3, 4, 28].
Known exposure durations in seconds are [1, 2, 4, 8] in each group.
All counts in this example are small exactly representable binary64 integers.
The existing nb2_moment_dispersion estimator converts counts to binary64
and may round integers above 2^53. The new nb2_count_diagnostic kernel
instead rejects counts above 2^53 with status 4; it does not silently round
them.

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
Failures print an unavailable status and return exit code 1. The explicit
i32 main returns zero only after completing Canvas rendering, HTML/SVG
emission and Canvas allocation release.

The left panel shows observed count/exposure and fitted count mean/exposure,
in events per second. The right panel compares **Poisson** Pearson residuals
(y-mu)/sqrt(mu) with **NB2 plug-in** Pearson residuals
(y-mu)/sqrt(mu+alpha*mu^2), with a zero reference line. Alpha here is
dispersion.alpha_nonnegative: the projected moment estimate from these same
observations, not a fixed known dispersion. Both residual series use the
same explicit viewport and axis. They are computed through the production
nb2_count_diagnostic kernel in both the bounds and plotting passes; the
example does not duplicate the residual formulas in executable code.

The right-panel NB2 scaling is descriptive. Smaller residual magnitudes
after increasing a plug-in variance do not demonstrate improved model fit,
successful distributional calibration, or a valid significance procedure.
The plotted mean remains the Poisson fitted mean; this is not an NB
likelihood regression fit.

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
