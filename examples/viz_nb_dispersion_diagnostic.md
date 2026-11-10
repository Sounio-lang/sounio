# Poisson means and NB2 moment dispersion: descriptive example

Source: `viz_nb_dispersion_diagnostic.sio`.

## Execution evidence

The revised dual-residual example was typechecked, compiled and executed in
Slurm 15783 (COMPLETED, exit 0:0), after a current-source Madaros compiler
rebuild. Four explicit typechecks passed, including this example. The native
program exited zero after completing the Canvas and SVG paths. Frozen
source commit: `4a65e767af7ce955cfde85f25484ea7f8a5f44f2`. Program ELF
SHA-256: `694a9bab24cde523bced34c36a67d5070635b601c04da575cc22ee1458fc8a13`.
See the [execution receipt](../docs/plans/NB2_DIAGNOSTICS_EXECUTION_2026_10_21.md).

The coordinator extracted the report prefix, HTML and SVG from actual native
stdout. XML parsing with xmllint exited zero. A sips-rendered 1380x620 PNG
was inspected: both panels, the shared residual axis, both residual legends,
units, caveat and minus labels were legible. The clipped QuickLook preview
was not used for that visual conclusion. The native Canvas path completed;
there was no captured-pixel comparison between Canvas and SVG.

An exact-ELF profile in Slurm 15785 (exit 0:0) recorded GNU time wall duration
0.01 s and maximum RSS 8748 KiB for one execution of this eight-row complete
program. This is a single observation of the full program, not an isolated
kernel measurement, general benchmark or comparative performance result.

These runs do not establish numerical concordance, statistical calibration,
full-corpus acceptance, CI or merge status. No tests were added or run for
this example.

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
