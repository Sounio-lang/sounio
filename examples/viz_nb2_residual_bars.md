# Supplied-parameter NB2 residual bars

Source: `viz_nb2_residual_bars.sio`.

## Scientific meaning

The example uses nine synthetic observed counts: 0, 1, 2, 3, 4, 6, 8, 12, 16.
Every observation is evaluated with supplied count mean 4 and dispersion alpha
0.25. These parameters are declared inputs; no mean, dispersion or model is
fitted.

The left panel shows checked NB2 Pearson residuals from
`stats::regression::nb2_diagnostics::nb2_count_diagnostic`. The right panel shows
checked signed saturated-deviance residuals from
`stats::regression::nb2_deviance::nb2_deviance`. Both residuals are dimensionless.

The horizontal coordinate is observation index, not count value or time.
The text report maps each observation to its count. Both panels share a
symmetric vertical domain computed in Sounio from the actual available
residuals, with padding and zero references. No precomputed residual array
or expected-value oracle is embedded.

These are descriptive transformations conditional on supplied parameters.
Independence and the NB2 count model are not assessed. Residual magnitude or
a visual comparison does not establish goodness of fit, calibration,
significance, a confidence interval or superiority of either residual.

## Status and report contract

The report prints every observation index/count and the diagnostic statuses.
Each residual payload is printed only when its own status is zero.
The squared deviance is printed only when both the residual and
`deviance_status` permit it; otherwise it is printed as unavailable.
Square availability does not control whether an available residual is plotted.
Any required residual, viewport, scene-capacity or domain failure returns exit
code 1 before SVG emission. Successful completion emits HTML/SVG and returns 0.

Residuals and available squared deviances use
`io::sci_print::println_f64_sci(value,15)`. Fifteen significant decimal digits
are report formatting, not a binary64 roundtrip, accuracy certificate or
independent numerical verification. The formatter normalizes signed zeros
to positive-zero text.

## Bar geometry and storage

Both panels use `viz_add_bar_chart` and checked `viz_bar_set_viewport`.
The X viewport spans 0 through 10 while observations occupy indices 1 through
9, providing space for the first and last bar footprints. The zero-origin
vertical intervals are recropped by the production bar geometry; the plot
does not replace the origin with its bottom edge. Explicit axes are separate
scene nodes because the checked viewport draws marks without implicit axes.

A zero residual produces a zero-height interval and no painted rectangle.
The count mapping and caption explain the observation's presence; the example
does not invent a minimum-height bar. Bar width follows the existing renderer
policy based on the first two finite input pairs and bounded pixel width.

Two 1024-element scratch arrays and one 64-byte label buffer are reused.
Builders copy the active prefix into scene storage before the next reuse.
The example emits SVG only and requests no Canvas allocation. This does not
claim the full program or imported renderer is allocation-free.
Standard output consists of a report prefix followed by one HTML document
containing one 1380x620 SVG; the prefix must be separated for standalone HTML.

## Initial preparation boundary

At initial authoring this was source preparation only. No assertions or tests were added; no
compiler, build, native execution, numerical oracle or visual inspection was
run for this new example by its author. Existing runtime/figure receipts cover
their identified sources, not this new consumer. The coordinator owns the
authoritative Slurm typecheck/native checkpoint.

Darwin `sounio_verificado` was consulted before writing the Sounio source.
Its bounded candidate returned the following invalid binding:

    let mut result = nb2_deviance(count, 4.0, 0.25);

The remote candidate check reported E040: Sounio uses `var`, not `let mut`;
`parse_failed=true`. The implementation uses an immutable `let` result and
propagates unavailable diagnostics as explicit failure, rather than returning
a misleading zero residual. This candidate check is distinct from validation
of the final repository example.

Related contracts:
[NB2 Pearson diagnostics](../stdlib/stats/regression/NB2_DIAGNOSTICS.md),
[NB2 deviance](../stdlib/stats/regression/NB2_DEVIANCE.md).

## Initial native attempt and axis budget correction

Slurm 16017 completed its compiler build and nine typechecks, but the native
consumer stopped with EXAMPLE_UNAVAILABLE: axes. The two horizontal axes
requested five intervals in a 400-pixel layout; the existing axis guard
requires at least 112 pixels per requested interval. This source revision
requests two horizontal intervals, preserving the guard, all observations,
the 0..10 viewports and every other geometry/domain choice. The axes still
express categorical observation order; ticks need not label every observation.

The corrected consumer has not yet passed a native retry. This source change
modifies no library or compiler input. A later receipt must identify the
corrected source and any attested compiler reused for its execution.

## Completed native checkpoint

Slurm16017 built the canonical compiler and passed nine typechecks, then the
initial consumer correctly stopped on an invalid horizontal-axis interval request.
After changing only those two interval arguments, Slurm16019 passed both the
corrected bars consumer and existing NB2 deviance consumer: typecheck, compilation
and native exit0. It reused the retained fresh compiler with unchanged compiler/
stdlib inputs; no compiler rebuild from the consumer-only fix is claimed.

The actual bars SVG parsed and rendered at1380x620; root and independent visual
inspection found no blocking issue. Runtime maxRSS was1584KiB. This is one measured
execution, not a general memory bound or numerical-concordance result.

[Source-bound execution receipt](../docs/plans/NB2_RESIDUAL_BARS_EXECUTION_2026_10_23.md)
contains both attempts, exact hashes, scratch-only quota fallback and remaining
acceptance limits.
