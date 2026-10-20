<!-- docs:meta
topic_id: repo.docs.plans.statistics-visualization-baseline-audit
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.statistics-visualization-baseline-audit
-->

# Statistics and Scientific Visualization: Baseline Audit

## Audit identity

- Repository: remote Sounio `main`
- Commit: `f5b2c5850be9f3d3995719ffcade44a6d851658a`
- Worktree: `/workspace/sounio-sprint-stats-viz` (detached)
- Method: source, docs, and test inventory inspection only
- Builds/tests: not run

This is a source-level triage. `Observed in source` does not mean executed or independently validated.

## Confirmed findings

| Priority | Area and source | Source-level observation | Classification / next action |
|---|---|---|---|
| P1 | `stdlib/stats/descriptive.sio:186-205` | `mad` is documented as Median Absolute Deviation and constructs absolute deviations from the median, but returns their arithmetic mean. Its comment explicitly calls this a mean absolute deviation approximation. | **Semantic mismatch.** Implement median-of-absolute-deviations (and name mean absolute deviation separately if needed); add parity/edge cases before relying on the result for robust summaries or plots. |
| P1 | `stdlib/stats/inferential.sio:118-161` | `t_to_p_approx` uses a normal-tail approximation and says it is valid for df > 30; `f_to_p_approx` is described as a very rough approximation and points to the incomplete beta function as the production method. | **Documented approximation, limited suitability.** Audit all callers; implement accurate tail paths with stable special functions or return approximation/validity metadata and refuse unsupported claims. |
| P1 | `stdlib/stats/epistemic/assumptions.sio:141-152` | Shapiro-Wilk-like W calculation uses simplified weights `1/(1+i)` rather than the method's coefficient table. | **Pedagogical approximation under a production-sounding test name.** Replace with the method's valid coefficients/ranges, or rename/mark experimental and prevent it from powering normality claims. |
| P1 | `stdlib/stats/regression/linear.sio:660-678` | `dffits` is a zero-filled array; linearity confidence is a fixed `BetaConfidence`. | **Explicit placeholders.** Compute from fitted residuals/leverage/studentized residuals and implement a diagnostic with documented assumptions; otherwise remove these fields from a result that implies they were assessed. |
| P1 | `stdlib/viz/ir.sio:84-93,328-350,489-502`; `stdlib/viz/viz_canvas.sio:112-143` | Series store `vars`; the forest and uncertainty-band builders accept variances; Canvas applies `sqrt(vars)` and draws estimate ± one sigma. | **Variance is the observed storage contract, but interval meaning is underspecified.** The renderer's ±1σ is not automatically a 95% CI, prediction interval, or credible interval. Add explicit interval type/bounds and render the exact supplied meaning; avoid inferring confidence level from variance alone. |
| P1 | `stdlib/viz/ir.sio:328-350`; chart builders accept `[f64; 1024]`; series storage permits 8 slots | The add-series loop copies at most 1024 observations while `d.n` keeps the caller's `n`; multiple renderer loops cap at 1024. | **Capacity/truncation contract risk.** Validate and report over-capacity input before creating nodes; do not retain a larger logical count after silently copying only a prefix. Quantify before choosing chunking/aggregation. |
| P2 | `stdlib/viz/chart.sio:272-285` | A separate lower-level uncertainty-band renderer accepts explicit 95%, 80%, 50% lower/upper bounds and medians. | **Useful semantic primitive, distinct from the variance-based IR band.** Reconcile the two APIs around explicit bounds/interval metadata; test both renderer paths for parity. |
| P2 | `tests/run-pass/viz_chart_builders.sio:1-5,36-72` | The run-pass chart smoke builds scatter, bar, forest, and waterfall nodes from five fixed values, checks node kinds/slots and nonzero Canvas regions, then emits HTML. | **Component/render smoke, not statistical end-to-end validation.** Preserve it; add an integrated test that computes an analysis result and checks interval semantics, visible labels, scene data, and exported metadata/artifact. |
| P2 | `examples/stats/forest_plot.sio:1-28` | The example calls `stats::meta_analysis::meta_fixed` and writes a PNG, but uses five hard-coded study effects/SEs and manually draws the figure through `image::plot`; it does not use the `viz` Visual IR. | **Real vertical demonstration with fixed illustrative inputs, but a parallel rendering path.** Label its fixture provenance and use it as a migration/reference case for a generic stats-to-IR-to-export path. |
| P2 | `stdlib/graphics/lib.sio:7-15`; `stdlib/graphics/export.sio:1-8,37-40`; `stdlib/graphics/png.sio` | The graphics module overview still says PNG is a stub, but `export.sio` calls a pure-Sounio PNG encoder. | **Stale module documentation, not a missing PNG implementation.** Correct documentation after verifying the current PNG encoder/export path with its existing tests. Do not rewrite the encoder on the strength of the stale comment. |
| P2 | `stdlib/graphics/svg.sio:383-390` | One advanced epistemic hero-figure function is commented out as a placeholder; surrounding SVG primitives and other exported plotting routines exist. | **One disabled composition, not a general SVG-renderer stub.** Decide whether to implement that figure or remove the advertised promise; test current export separately. |
| P3 | `stdlib/plot/bar.sio:306-310` | An integer-format helper has placeholder wording, but its body delegates to `print(value)`. | **Likely stale comment or runtime dependency, not enough evidence that bar rendering is absent.** Verify behavior and update the comment/API status; do not classify the whole chart as a stub from this snippet alone. |
| P3 | `examples/pbpk_viz/rapamycin_anova_genotype.sio` | The example calls ANOVA and Student-t interval APIs but its cohort values and group statistics are embedded in the demo. | **Useful integration example, not an external-data validation result.** Preserve the demo, identify data as illustrative unless provenance says otherwise, and add generated/reproducible fixtures with receipts. |

## Coverage interpretation

The baseline has 41 visualization/graphics-named `tests/run-pass` files, plus 3 files under `tests/stdlib/viz` and 1 under `tests/stdlib/graphics`. These are meaningful component coverage. This inventory does not establish how many tests execute each path, nor that their expected artifacts are inspected. The focused builder smoke demonstrates node construction and non-empty Canvas regions, but not statistical correctness or equivalence between Canvas and HTML/SVG output.

The stats baseline has 163 Sounio source files and 14 tests under `tests/stdlib/stats`. This is broad API surface. It also contains a mix of exact methods, stated approximations, and explicit placeholders; classification must happen per method and caller rather than at module level.

## First implementation slice

The first implementation slice should connect one audited method to one unambiguous uncertainty representation and both existing renderers:

1. Select a method with a defensible numerical contract (for example, one-sample mean with Student-t interval after auditing the t CDF/quantile path).
2. Return estimate, sample size, interval bounds, interval kind/level, units, and provenance as explicit data.
3. Lower those bounds without reinterpretation into a Visual IR chart node.
4. Render Canvas and SVG from the same IR and include labels/metadata that name the interval and method.
5. Validate the numerical result against a standalone C++23 reference plus Sounio edge cases; validate renderer structure and deterministic identity separately.

No implementation is selected as mathematically ready until its current numerical dependencies and callers have been traced.

## Evidence still required

- Execute existing test programs under the project's remote/Slurm compiler workflow and record exact source/compiler identity.
- Trace every important `t_to_p_approx`, `f_to_p_approx`, Shapiro-like, and variance-band caller before changing compatibility behavior.
- Inspect actual exported SVG/PNG artifacts and current renderer labels/axes using deterministic fixtures.
- Measure capacities and performance before proposing a new storage representation or numeric throughput target.

## First implementation slice — completed checkpoint (2026-10-16)

The current worktree implements explicit forest interval bounds/kind/level,
atomic validation, Canvas/HTML propagation, and exact raw MAD in
stats::descriptive. It also fixes Canvas estimate-marker coordinates and keeps
full test output logs.

Final Foundry run stats-viz-forest-20261016-17, Slurm job 14665:
COMPLETED, ExitCode=0:0, three selected tests passed, zero failures/skips.
The independent C++23 reference passed for six Student-t endpoints and the raw
MAD fixtures. This is focused coverage; the wider domain and sprint remain open.

See [the execution receipt](STATISTICS_VISUALIZATION_EXECUTION_2026_10_16.md)
for exact hashes, test selection, before/after failures, resource retry, compiler
identity, numerical tolerances, SVG artifacts, and pending work.

Additional audit finding: viz_audit_hash and viz_audit_node_hash cover state and
layout summaries; viz_app_frame_hash samples three pixels. They do not digest
full data or interval semantics. The scientific receipt design must add
versioned content/data/specification digests before treating these summaries as
reproducibility evidence.
