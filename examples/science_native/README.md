<!-- docs:meta
topic_id: repo.examples.science-native.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.examples.science-native.readme
-->

# Checked science to native figures

These four executable examples consume the production APIs reviewed in SIO-169,
170,173,174,175 and176. They do not contain replacement solvers. Each program
computes its own data, paints an actual Canvas, emits an SVG from those results,
and exports the Canvas pixels as lossless RLE comments. An independent C++23
reader checks numerical payloads before checking graphics and can export the
actual raster as PPM. No SVG rasterizer or generated illustration substitutes
for the native Canvas.

| Example | Production calculation | Figure and scientific limits |
| --- | --- | --- |
| `poisson_field.sio` | `math::poisson_multigrid::poisson_solve_into`, then `viz::dynamic_contour::prepared_contour_prepare_into` |65×65 solved nodes for a manufactured, dimensionless Dirichlet PDE. The two512×512 panels preserve equal physical x/y aspect. The first heatmap is an explicitly declared64×64 reconstruction from averages of four solved nodes, with nodal contours prepared once and borrowed by both exporters. The second is the corresponding average absolute nodal error against the known analytic solution. Neither is an uncertainty estimate. |
| `normalization.sio` | Validated `CountCSR`, then `count_csr_size_factors_into` forwarding the canonical median-ratio estimator |96features×16samples,1536 positive integer counts. Observed and normalized counts share a color domain. The count pattern is synthetic and multiplicative; this does not assess biological quality, differential expression or inferential uncertainty. |
| `survival.sio` | `stats::checked_survival::survival_checked` |Two12-observation synthetic groups with tied times and right censoring. Kaplan-Meier steps and nominal95% **pointwise asymptotic** Greenwood/log-log confidence intervals use independent availability flags. A missing CI does not erase available survival. No simultaneous-band or finite-sample coverage claim; independent censoring is unassessed. |
| `dp54_event.sio` | `ode::checked_dp54::solve_dp54_event64` and its retained dense/event owners |Decay `dy/dt=-0.5y`, decreasing terminal crossing at `y=0.4`. Only the last retained accepted segment is sampled, including its localized endpoint. Both state and the derivative of the retained quartic polynomial are shown; there is no fabricated full dense history, new RHS evaluation per query, or uncertainty interval. |

The normalization demo's generated counts never exceed768, so its conversion
to binary64 is exact. It is not a generic arbitrary-i64 plotting adapter. Size
factors are dimensionless and geometrically centered; the normalized scale is
an estimator convention rather than a physical concentration unit. All reference
features are eligible in this particular example.

The survival example retains the kernel's count/flag records and reads CI bit16
separately from survival bit1. Limits unavailable at singular transforms S=0/1
are masked, with no fabricated zero bounds. An explicit provenance index for
the first masked trailing run satisfies the ordered-line API without painting
that run or extrapolating beyond the cohort's observed follow-up.

The ODE event bracket localizes the root of the **computed interpolant**. It is
not a certified enclosure of the exact ODE event time. The solver's stopping
and event tolerances are computational criteria. Callback side effects count
calls only; they do not alter the mathematical RHS or event functional.

Viridis/inferno lookup and horizontal colorbars use the canonical SIO-176
`PaletteScale` and its declared floor-255 RGB8/floor-256 listed-bin policy. The
primary BIDS/Matplotlib assets and public-domain8×8font remain at their existing
pinned repository paths; these examples do not duplicate those assets.

## Worker replay

Run the checked-in gate on a Linux worker, from the source tree. The retained
qualification jobs use Slurm UID1000; the public gate records the actual user
and does not require a particular UID, so Linux CI runners can invoke it:

~~~sh
export MADAROS_RAW_BIN=/absolute/path/to/verified/madaros.elf
export SCIENCE_NATIVE_SEED_BIN=/absolute/path/to/verified/strict-seed.elf
export SCIENCE_NATIVE_OUTPUT=/absolute/new/output
bash scripts/stdlib/scientific_native_workflows_gate.sh
~~~

The gate sets the524288KiB stack limit, pins the public resolver to the supplied
Madaros, disables shared compiler caches, records compiler/source/ELF identities,
and refuses an existing output directory. Seed execution is optional; leaving
`SCIENCE_NATIVE_SEED_BIN` unset qualifies only the Madaros engine. The source
working directory is authoritative for both engines' imports.

Each stdout begins with its actual SVG root. The `*.actual.svg` artifact is a
byte-preserving extraction through its closing root; trailing XML comments
carry numerical records and Canvas RLE. The C++ reader requires exact raster
coverage with no gaps/overlaps before writing `*.actual-canvas.ppm`. Its output
is the decoded native Canvas, not an independently redrawn picture.

## Qualification scope

The reference checks analytic Poisson nodes and a discrete residual, exact input
counts and independent normalization factors, direct risk-set/KM/Greenwood/
log-log calculations, and analytic exponential/event/dense-derivative values.
The actual binary64 input/output bits are decoded, including signed and
subnormal values. Unavailable or nonfinite values are handled according to the
specific numerical contract rather than being globally coerced to zero.

Graphics checks include independent cell/contour/step/CI/dense geometry,
provenance and interval metadata, canonical palette pixels, document text bounds,
actual primary-font glyph pixels, and unique SVG IDs. The Poisson color oracle
explicitly excludes a two-pixel AA fringe around independently checked contour
segments; it does not claim an independent reconstruction of that blend. KM
axis-aligned step and interval pixels are exact checks. ODE checks actual full-
coverage integer endpoints; its interior AA blend is not reconstructed. Axis
rules use the same half-open pixel rectangles in Canvas and SVG.

There are dedicated changed-input controls for a solved node, SVG contour
coordinate, actual Canvas color, size factor, CI category and dense derivative.
Each mutation has a distinct input path, must change nonempty data, and must
produce its own expected diagnostic. A process failing for some unrelated
reason does not qualify a control.

A passing scoped gate is separate from GitHub CI, scientific calibration,
maintainer review and merge. Resource observations include the complete program,
renderers and RLE/SVG output; they are not isolated-kernel benchmarks or speedup
comparisons. Final receipts and actual images are recorded in
[the evidence document](../../docs/evidence/scientific-native-workflows-2026-10-25/README.md).
