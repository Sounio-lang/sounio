# Shared logarithmic primitives: descriptive SVG example

Source: `examples/viz_logarithmic_primitives.sio`.

This example prepares an SVG/HTML visualization and a short numerical report
using the public checked logarithmic APIs. Its preparation is not a numerical
verification, accuracy certification, or independent reference calculation.
Execution and compiler identity belong to the sprint execution receipt.

## Figure

The seven positive inputs are `1e-100, 1e-50, 1e-20, 1e-12, 1e-6, 1e-2, 0.25`.
The program computes each horizontal coordinate as
`ln_checked(x).value / ln_checked(10).value`.
Both horizontal axes are labelled **log10(x), x > 0** and use the explicit
transformed-coordinate domain `[-110,10]`; they are not raw-x axes.

The left panel compares `log1p_checked(x).value / x` with the deliberately naive
`ln_checked(1+x).value / x`, using vertical domain `[0,1.2]`.
This shows the descriptive consequence of rounding `1+x` before taking its
logarithm. The naive curve is not an independent oracle.

The right panel shows `log1p_phi_checked(x).value / (x*x)` and
`-log1pmx_checked(x).value / (x*x)`, with vertical domain `[0,0.8]`.
The chosen positive inputs keep the denominator nonzero in binary64.
Lines connect these seven samples only; they are not interpolated accuracy
bounds or a continuous error survey.

The SVG canvas is `1380x620`. Plot layouts are `(180,150,400,240)` and
`(820,150,400,240)`. Legends and captions identify the transformations and
the absence of accuracy certification.

## Printed report

The program prints statuses and descriptive results for:

- `awgn_capacity_nats(1e-20)`, reporting capacity divided by SNR so that
  ordinary text formatting does not conceal the tiny positive capacity.
- Shannon, Renyi order 2 (the public `information_collision` function), and
  min entropy for `p=[0.9,0.1]`, in nats.
- KL divergence from that p to `q=[0.88,0.12]`, in nats.
- General logarithmic branches: `log1p(8)`, `log1pmx(-0.5)`, and `phi(8)`.

These calls exercise information/channel consumers. The caption also names the
shared Poisson integration; this example does not fit or execute a Poisson
model. There are no assertions, expected-value fixtures, p-values, or accuracy
acceptance thresholds. A failed checked call or scene-builder operation returns
exit status 1 with an `EXAMPLE_UNAVAILABLE` message.

## Resource contract

The figure uses one fixed-capacity scene and reuses two `[f64;1024]` scratch
arrays for its four line series. It uses one reusable `[i8;64]` label buffer;
builders retain scene-owned copies. There is no Canvas allocation or raster
buffer. Only the seven active sample entries are passed to each series.
This is a source-level resource description, not a measured RSS or speed claim.

## Status

Prepared source and documentation only. No typecheck, native execution,
numerical run, test, or rendering acceptance is claimed here. The main entry
returns `i32` and declares `IO, Mut, Div, Panic`.
