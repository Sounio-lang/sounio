# Shared logarithmic primitives: descriptive SVG example

Source: [viz_logarithmic_primitives.sio](viz_logarithmic_primitives.sio).
Execution evidence: [logarithmic sprint receipt](../docs/plans/LOGARITHMIC_EXECUTION_2026_10_21.md).

This example emits an SVG/HTML visualization and a short numerical report using
the public checked logarithmic APIs. Native execution and the inspected figure
are recorded below; they do not establish numerical concordance, an accuracy
certification, or an independent reference calculation.

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
bounds or a continuous error survey. The figure explicitly states:
**Seven samples joined by lines; no accuracy certification.**

The SVG canvas is `1380x620`. Plot layouts are `(180,150,400,240)` and
`(820,150,400,240)`. Legends and captions identify the transformations and
the limited interpretation of the plotted samples.

## Printed report

The program prints statuses and descriptive results for:

- `awgn_capacity_nats(1e-20)`, reporting capacity divided by SNR so that
  ordinary text formatting does not conceal the tiny positive capacity.
- Shannon, Renyi order 2 (the public `information_collision` function), and
  min entropy for `p=[0.9,0.1]`, in nats.
- KL divergence from that p to `q=[0.88,0.12]`, in nats.
- General logarithmic branches: `log1p(8)`, `log1pmx(-0.5)`, and `phi(8)`.

These calls exercise information/channel consumers of the shared primitives.
The library integration also includes the Poisson kernel; this example does
not fit or execute a Poisson model. There are no assertions, expected-value
fixtures, p-values, or accuracy acceptance thresholds. A failed checked call
or scene-builder operation returns exit status 1 with an
`EXAMPLE_UNAVAILABLE` message.

## Resource contract and observation

The figure uses one fixed-capacity scene and reuses two `[f64;1024]` scratch
arrays for its four line series. It uses one reusable `[i8;64]` label buffer;
builders retain scene-owned copies. There is no Canvas allocation or raster
buffer. Only the seven active sample entries are passed to each series.
These are source-level bounds, not a scaling or comparative performance result.

The final native whole-example measurement reported GNU time wall time
`0.00 s`, maximum RSS `1576 KiB`, and exit status `0`.
The timer has centisecond resolution: `0.00 s` does not mean zero runtime.
This observation covers the complete math demonstration, including its report
and SVG output; it is not an isolated kernel benchmark or a superiority claim.

## Executed status and custody

The execution receipt records:

- Job **15823**: compiler built from current sources, seven file checks, and
  two native demonstrations completed with exit status `0:0`.
- Final job **15825**: the attested compiler was reused after exact unchanged
  input-tree verification; seven file checks and both native demonstrations
  completed with exit status `0:0`. The recorded job elapsed time was
  `25 s`.
- The math report returned status `0` for AWGN, the information calls, and
  the three general-branch logarithmic calls.
- SVG XML validation exited `0`. A full `1380x620` raster preview was
  inspected, including axes, legends, and the seven-sample caption.
- The NB consumer's separate Canvas/SVG rerun passed as described in that
  receipt; it is not a numerical reference for this example.

The final source checkpoint was
`27ea4ec6c2b97892e9e92cf982b0af90e2241023`.
The main entry returns `i32` and declares `IO, Mut, Div, Panic`.

| Artifact | SHA-256 |
|---|---|
| Final Sounio example | `13f5299b56d772a7f06741eab85330c13598d765f2796c111da64782630b7e11` |
| Native math executable | `c137e7474e1ca66b483c3451764cd9dfb202a913fe574d0607d5eedd8a129669` |
| Native math stdout | `17018168d668ee34b4f3d1f48788adeeeb059690ae97167f0eaf461dc1a4e352` |

These receipts establish the stated checks, executions, and visual inspection.
They do not establish automated test acceptance, independent numerical
concordance, global ULP bounds, boundary coverage, CI acceptance, or merge.
