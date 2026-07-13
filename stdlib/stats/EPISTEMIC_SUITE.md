# Epistemic Statistics Suite

## Overview

The **epistemic statistics suite** is a cohesive, pure-Sounio, dependency-light
toolkit for classical inferential statistics. It completes the stdlib
distribution family — Student's t, chi-squared and Fisher-Snedecor F, each with
a full `pdf` / `cdf` / `sf` / `quantile` surface — and adds two analyses that
clinical and pharmacometric work routinely need but that the stdlib previously
lacked: the non-parametric Wilcoxon signed-rank test and Bland-Altman
method-agreement analysis. Every routine is written in Sounio with no FFI, no
sampling dependency and no external solver: the analytic core rests only on the
special-function modules (`special::beta`, `special::gamma`, `special::igamma`,
`special::erf`). Every non-trivial routine returns an **explicit result struct**
that carries the point estimate together with a confidence interval (or, for
Wilcoxon, the test statistics, z, p-value and effect size), so callers receive
auditable numbers rather than a bare scalar.

## Modules

### `stats::student_t` — Student's t-distribution

Completes the t-distribution as a first-class object: density, cumulative,
survival, an exact two-tailed tail probability, a general quantile (inverse CDF)
for any `p`, a two-sided CI multiplier at any confidence level, and a confidence
interval for a sample mean. The analytic core is the regularized incomplete beta
and its inverse; the large-`df` path uses the normal quantile plus a
Cornish-Fisher expansion.

| Function | Signature |
|---|---|
| `t_pdf` | `pub fn t_pdf(t: f64, df: f64) -> f64 with Mut, Div, Panic` |
| `t_cdf` | `pub fn t_cdf(t: f64, df: f64) -> f64 with Mut, Div, Panic` |
| `t_sf` | `pub fn t_sf(t: f64, df: f64) -> f64 with Mut, Div, Panic` |
| `t_two_tail` | `pub fn t_two_tail(t: f64, df: f64) -> f64 with Mut, Div, Panic` |
| `t_quantile` | `pub fn t_quantile(p: f64, df: f64) -> f64 with Mut, Div, Panic` |
| `t_ci_multiplier` | `pub fn t_ci_multiplier(conf: f64, df: f64) -> f64 with Mut, Div, Panic` |
| `t_mean_ci` | `pub fn t_mean_ci(mean: f64, sd: f64, n: i64, conf: f64) -> TCI with Mut, Div, Panic` |

`t_pdf` is the density; `t_cdf` the monotone `P(T <= t)`; `t_sf = 1 - cdf`;
`t_two_tail` the two-sided `P(|T| > |t|)` a t-test needs; `t_quantile` the
inverse CDF; `t_ci_multiplier(conf, df)` the `k` with `P(-k < T < k) = conf`
(generalising the old 95%-only table in `optimize::uncertainty`).

**Returns — `struct TCI`** (from `t_mean_ci`):

| Field | Type | Meaning |
|---|---|---|
| `mean` | `f64` | sample mean (interval centre) |
| `lo` | `f64` | lower confidence bound |
| `hi` | `f64` | upper confidence bound |
| `mult` | `f64` | t-multiplier used |
| `se` | `f64` | standard error of the mean |
| `df` | `f64` | degrees of freedom (`n - 1`) |

### `stats::chi_square` — chi-squared distribution

Adds the density, survival and a general quantile (for any `p`, any `k`) on top
of the existing exact CDF (`special::igamma::chi2_cdf`), plus the classic
variance confidence interval. The quantile has no closed form: it uses the
Wilson-Hilferty cube-root normal approximation as an initial guess, then
Newton-Raphson on the exact CDF/pdf.

| Function | Signature |
|---|---|
| `chi2_pdf` | `pub fn chi2_pdf(x: f64, k: f64) -> f64 with Mut, Div, Panic` |
| `chi2_cdf` | re-exported exact form (`special::igamma::chi2_cdf`) |
| `chi2_sf` | `pub fn chi2_sf(x: f64, k: f64) -> f64 with Mut, Div, Panic` |
| `chi2_quantile` | `pub fn chi2_quantile(p: f64, k: f64) -> f64 with Mut, Div, Panic` |
| `chi2_variance_ci` | `pub fn chi2_variance_ci(s2: f64, n: i64, conf: f64) -> VarCI with Mut, Div, Panic` |

**Returns — `struct VarCI`** (from `chi2_variance_ci`):

| Field | Type | Meaning |
|---|---|---|
| `s2` | `f64` | sample variance (point estimate) |
| `lo` | `f64` | lower confidence bound on the population variance |
| `hi` | `f64` | upper confidence bound |
| `df` | `f64` | degrees of freedom (`n - 1`) |

### `stats::fisher_f` — Fisher-Snedecor F-distribution

Exposes the density, CDF, survival and a general quantile, plus the
variance-ratio confidence interval used to compare two sample variances. Built
on the regularized incomplete beta and its inverse:
`F_cdf(x) = I_{d1·x/(d1·x+d2)}(d1/2, d2/2)`.

| Function | Signature |
|---|---|
| `f_pdf` | `pub fn f_pdf(x: f64, d1: f64, d2: f64) -> f64 with Mut, Div, Panic` |
| `f_cdf` | `pub fn f_cdf(x: f64, d1: f64, d2: f64) -> f64 with Mut, Div, Panic` |
| `f_sf` | `pub fn f_sf(x: f64, d1: f64, d2: f64) -> f64 with Mut, Div, Panic` |
| `f_quantile` | `pub fn f_quantile(p: f64, d1: f64, d2: f64) -> f64 with Mut, Div, Panic` |
| `f_var_ratio_ci` | `pub fn f_var_ratio_ci(s1_2: f64, n1: i64, s2_2: f64, n2: i64, conf: f64) -> VarRatioCI with Mut, Div, Panic` |

Here `d1`/`d2` are the numerator/denominator degrees of freedom.

**Returns — `struct VarRatioCI`** (from `f_var_ratio_ci`):

| Field | Type | Meaning |
|---|---|---|
| `ratio` | `f64` | `s1²/s2²` (point estimate) |
| `lo` | `f64` | lower confidence bound on `σ1²/σ2²` |
| `hi` | `f64` | upper confidence bound |
| `d1` | `f64` | numerator dof (`n1 - 1`) |
| `d2` | `f64` | denominator dof (`n2 - 1`) |

### `stats::wilcoxon` — Wilcoxon signed-rank test

The non-parametric counterpart to the paired t-test (and the paired analogue of
the Mann-Whitney U in `stats::hypothesis`). It tests whether the median of the
paired differences — or of a single sample about a hypothesised median — is
zero, using only the ranks of the absolute differences, so it is robust to
non-normal, heavy-tailed data. Exact-zero differences are discarded
(the standard textbook convention; Pratt-style zeros are not implemented); tied
magnitudes receive average ranks with the usual tie correction to the variance.

| Function | Signature |
|---|---|
| `wilcoxon_signed_rank` | `pub fn wilcoxon_signed_rank(x: &[f64; 256], y: &[f64; 256], n: i32) -> WilcoxonResult with Mut, Div, Panic` |
| `wilcoxon_one_sample` | `pub fn wilcoxon_one_sample(x: &[f64; 256], n: i32, median0: f64) -> WilcoxonResult with Mut, Div, Panic` |

`wilcoxon_signed_rank` operates on paired differences `d_i = x_i - y_i`;
`wilcoxon_one_sample` on `d_i = x_i - median0`. Inputs are fixed-size buffers of
capacity 256 (`n <= 256`).

**Returns — `struct WilcoxonResult`:**

| Field | Type | Meaning |
|---|---|---|
| `w_plus` | `f64` | sum of ranks with positive difference (W+) |
| `w_minus` | `f64` | sum of ranks with negative difference (W-) |
| `w_stat` | `f64` | classic statistic `W = min(W+, W-)` |
| `n_eff` | `i64` | number of non-zero differences |
| `z` | `f64` | normal-approximation z (continuity-corrected, signed) |
| `p_value` | `f64` | two-tailed p-value |
| `effect_r` | `f64` | rank-biserial effect size `r = z / sqrt(n_eff)` |

The normal approximation is accurate for `n_eff` of roughly 15 or more; for very
small samples an exact permutation p-value would be tighter (not implemented
here).

### `stats::bland_altman` — Bland-Altman agreement analysis

The reusable statistics behind a Bland-Altman comparison of two measurement
methods A and B on the same subjects: bias, SD of the differences, 95% limits of
agreement, and — the part clinical papers require — the confidence intervals of
the bias and of each limit, plus a count of points outside the limits. The CIs
use the Student-t multiplier at `n - 1` degrees of freedom
(`stats::student_t::t_quantile`).

| Function | Signature |
|---|---|
| `bland_altman` | `pub fn bland_altman(a: &[f64; 256], b: &[f64; 256], n: i32, conf: f64) -> BAResult with Mut, Div, Panic` |

`a`/`b` are the paired measurements (`3 <= n <= 256`); `conf` is the confidence
level for the intervals (e.g. `0.95`). The limits of agreement themselves use
the fixed normal factor 1.96, per Bland & Altman.

**Returns — `struct BAResult`:**

| Field | Type | Meaning |
|---|---|---|
| `n` | `i64` | number of paired observations |
| `bias` | `f64` | mean difference `A - B` |
| `sd_diff` | `f64` | sample SD of the differences |
| `loa_lo` | `f64` | lower 95% limit of agreement |
| `loa_hi` | `f64` | upper 95% limit of agreement |
| `bias_ci_lo` | `f64` | CI of the bias, lower |
| `bias_ci_hi` | `f64` | CI of the bias, upper |
| `loa_lo_ci_lo` | `f64` | CI of the lower LoA, lower |
| `loa_lo_ci_hi` | `f64` | CI of the lower LoA, upper |
| `loa_hi_ci_lo` | `f64` | CI of the upper LoA, lower |
| `loa_hi_ci_hi` | `f64` | CI of the upper LoA, upper |
| `n_outside` | `i64` | points beyond `[loa_lo, loa_hi]` |
| `pct_outside` | `f64` | `100 · n_outside / n` |

### `stats::power` — statistical power & sample size

Power and required sample size for one-sample/paired and two-sample t-tests,
using the *shifted-t* approximation to the non-central t power function (Cohen
1988). `d` is Cohen's standardised effect size (mean difference / SD). Sample
sizes are found by searching up from the normal-approximation closed form to the
smallest `n` meeting the target power.

| Function | Signature |
|---|---|
| `power_ttest_onesample` | `pub fn power_ttest_onesample(d: f64, n: i64, alpha: f64, two_sided: bool) -> f64 with Mut, Div, Panic` |
| `power_ttest_twosample` | `pub fn power_ttest_twosample(d: f64, n_per_group: i64, alpha: f64, two_sided: bool) -> f64 with Mut, Div, Panic` |
| `n_for_power_onesample` | `pub fn n_for_power_onesample(d: f64, target_power: f64, alpha: f64, two_sided: bool) -> i64 with Mut, Div, Panic` |
| `n_for_power_twosample` | `pub fn n_for_power_twosample(d: f64, target_power: f64, alpha: f64, two_sided: bool) -> i64 with Mut, Div, Panic` |

Power is returned in `[0, 1]`; sample sizes are `>= 2` (two-sample returns the
size of **each** group), capped at 1 000 000. Validated against G*Power: one-sample
`d=0.5, n=34` → power ≈ 0.807; two-sample `d=0.5, n=64/group` → ≈ 0.801; the
classic `d=0.5, power=0.80` needs `n≈34` (one-sample) / `n≈64` per group. The
shifted-t approximation is tight for `n ≳ 20` and slightly over-states power at
very small `n`.

### `stats::qq_normal` — normal quantile-quantile (probability) plot

The visual normality diagnostic that pairs with the tests above: before a t-test
or ANOVA, check the data (or residuals) are plausibly normal. Computes the Q-Q plot
coordinates and the probability-plot correlation coefficient (PPCC) — near 1 for
normal data, lower for skew/heavy tails.

| Function | Signature |
|---|---|
| `qq_normal` | `pub fn qq_normal(data: &[f64; 256], n: i32, out_theoretical: &![f64; 256], out_sample: &![f64; 256]) -> QQResult with Mut, Div, Panic` |

Uses Blom's plotting position `p_i = (i - 0.375)/(n + 0.25)` and `z_i = Φ⁻¹(p_i)`.
`out_theoretical` receives the theoretical normal quantiles (plot x); `out_sample`
receives the sorted sample (plot y).

**Returns — `struct QQResult`:**

| Field | Type | Meaning |
|---|---|---|
| `n` | `i64` | sample size |
| `ppcc` | `f64` | probability-plot correlation coefficient (~1 = normal) |
| `slope` | `f64` | LS slope of sample on theoretical quantile (≈ SD) |
| `intercept` | `f64` | LS intercept (≈ mean) |
| `mean` | `f64` | sample mean |
| `sd` | `f64` | sample standard deviation (n-1 basis) |

### `stats::anova` — one-way analysis of variance

The headline group-comparison test and the first real consumer of `stats::fisher_f`.
Partitions the response variation into between- and within-groups components and
tests H0 "all group means equal" with F = MS_between / MS_within.

| Function | Signature |
|---|---|
| `anova_oneway` | `pub fn anova_oneway(data: &[f64; 256], sizes: &[i64; 16], k: i32) -> AnovaResult with Mut, Div, Panic` |

`data` holds the groups concatenated end to end; `sizes[j]` is group j's size;
`k` is the number of groups (2 ≤ k ≤ 16, total N ≤ 256).

**Returns — `struct AnovaResult`:** `k`, `n_total`, `ss_between`, `ss_within`,
`ss_total`, `df_between`, `df_within`, `ms_between`, `ms_within`, `f_stat`,
`p_value` (F survival), `eta_squared` (SS_between/SS_total). Validated against a
hand-computed table (SS 10/30, F=2.0, p=0.178, η²=0.25) and a well-separated case
(F≫1, p<0.001). The p-value uses the tail-hardened `f_sf` (accurate far tail — see
below).

### `stats::reg_bands` — linear regression with confidence & prediction bands

A minimal OLS fit that carries the quantities the bands need (residual sigma, x̄,
Sxx), plus the t-based confidence band (mean response) and prediction band (a new
observation) at each x — what a calibration / dose-response figure needs.

| Function | Signature |
|---|---|
| `reg_fit` | `pub fn reg_fit(x: &[f64; 256], y: &[f64; 256], n: i32) -> RegFit with Mut, Div, Panic` |
| `reg_predict` | `pub fn reg_predict(fit: &RegFit, x: f64) -> f64` |
| `reg_conf_band` | `pub fn reg_conf_band(fit: &RegFit, x: f64, conf: f64) -> (f64, f64) with Mut, Div, Panic` |
| `reg_pred_band` | `pub fn reg_pred_band(fit: &RegFit, x: f64, conf: f64) -> (f64, f64) with Mut, Div, Panic` |
| `reg_slope_ci` | `pub fn reg_slope_ci(fit: &RegFit, conf: f64) -> (f64, f64) with Mut, Div, Panic` |

`RegFit` fields: `n`, `slope`, `intercept`, `sigma` (residual SE), `r2`, `xbar`,
`sxx`, `df` (n-2). Band half-widths use `t·sigma·sqrt(1/n + (x-x̄)²/Sxx)`
(confidence) and `t·sigma·sqrt(1 + 1/n + (x-x̄)²/Sxx)` (prediction). Validated
against a hand-computed example (slope 0.6, sigma √0.8, slope CI [-0.30, 1.50],
conf/pred half at x̄ = 1.273 / 3.118).

### `stats::correlation` — Pearson & Spearman with inference

| Function | Signature |
|---|---|
| `pearson` | `pub fn pearson(x: &[f64; 256], y: &[f64; 256], n: i32, conf: f64) -> CorrResult with Mut, Div, Panic` |
| `spearman` | `pub fn spearman(x: &[f64; 256], y: &[f64; 256], n: i32) -> SpearmanCorr with Mut, Div, Panic` |

Pearson r with a **Fisher-z confidence interval** (`atanh(r) ± z/√(n-3)`, back-
transformed) and a t-test; Spearman rho is Pearson on the average ranks. `CorrResult`:
`n, r, t_stat, df, p_value, ci_lo, ci_hi`. `SpearmanCorr`: `n, rho, t_stat, df, p_value`.
Validated: `r=0.7746`, `t=2.121`, 95% CI `[-0.340, 0.984]`; Spearman `rho=0.738`.

### `stats::kruskal_wallis` — non-parametric one-way ANOVA

| Function | Signature |
|---|---|
| `kruskal_wallis` | `pub fn kruskal_wallis(data: &[f64; 256], sizes: &[i64; 16], k: i32) -> KWResult with Mut, Div, Panic` |

Rank-based H test (tie-corrected), `p = chi2_sf(H, k-1)`. `KWResult`: `k, n_total, h,
h_uncorrected, df, p_value, tie_factor`. Validated: no-ties `H=7.2, p=0.0273`;
tie-corrected `H=3.333` (factor 0.9143).

### `stats::tukey` — Tukey HSD post-hoc (studentized range)

| Function | Signature |
|---|---|
| `tukey_q_crit` | `pub fn tukey_q_crit(alpha: f64, k: i32, nu: f64) -> f64 with Mut, Div, Panic` |
| `tukey_hsd` | `pub fn tukey_hsd(ms_within: f64, n: i64, k: i32, dfw: f64, alpha: f64) -> f64 with Mut, Div, Panic` |
| `tukey_pair_p` | `pub fn tukey_pair_p(diff: f64, ms_within: f64, n: i64, k: i32, dfw: f64) -> f64 with Mut, Div, Panic` |
| `tukey_q_cdf` / `q_range_cdf` | studentized-range / range CDFs (numerical integration) |

Post-hoc pairwise comparisons after a significant ANOVA, controlling family-wise
error via the studentized range distribution q — computed here by Simpson
integration of the defining double integral (no closed form). Validated against
Harter's q tables: `q(0.05,3,10)=3.877`, `q(0.05,4,20)=3.958`, `q(0.05,3,∞)=3.314`
to ~1e-2. (The double integral makes this the one slow module — ~10s.)

### `stats::chi2_independence` — chi-squared test of independence

`pub fn chi2_independence(table: &[f64; 256], r: i32, c: i32) -> ChiIndep with Mut, Div, Panic`
— association in an r×c contingency table (row-major counts). `ChiIndep`: `chi2, df,
p_value, n, cramers_v, min_expected`. Validated: 2×2 `[[20,30],[30,20]]` → χ²=4, p=0.0455,
Cramér V=0.2.

### `stats::effect_size` — standardised effect sizes

| Function | Signature |
|---|---|
| `cohens_d` | `pub fn cohens_d(x: &[f64; 256], nx: i32, y: &[f64; 256], ny: i32, conf: f64) -> EffectSize with Mut, Div, Panic` |
| `eta_squared_from_f` | `pub fn eta_squared_from_f(f: f64, df1: f64, df2: f64) -> f64` |

Cohen's d + Hedges' g (bias-corrected) with SE and CI; `EffectSize`: `d, hedges_g,
se, ci_lo, ci_hi, magnitude` (0 negligible … 3 large). Validated: d=−1.2649,
g=−1.1425, 95% CI [−2.623, 0.093].

### `stats::proportion` — proportion CIs and two-proportion test

| Function | Signature |
|---|---|
| `wilson_ci` | `pub fn wilson_ci(k: i64, n: i64, conf: f64) -> (f64, f64) with Mut, Div, Panic` |
| `clopper_pearson_ci` | `pub fn clopper_pearson_ci(k: i64, n: i64, conf: f64) -> (f64, f64) with Mut, Div, Panic` |
| `two_prop_z` | `pub fn two_prop_z(k1: i64, n1: i64, k2: i64, n2: i64) -> TwoPropResult with Mut, Div, Panic` |

Wilson score and exact Clopper-Pearson intervals (the latter by bisection on the
forward `ibeta`, avoiding an `ibeta_inv` upper-tail weakness), plus the pooled
two-proportion z-test. Validated: Wilson 8/10 → [0.490, 0.943]; CP 8/10 →
[0.444, 0.975]; 30/100 vs 20/100 → z=1.633, p=0.1025.

## Importing

Import each tool directly from its module:

```sio
use stats::student_t::{TCI, t_pdf, t_cdf, t_sf, t_two_tail, t_quantile, t_ci_multiplier, t_mean_ci}
use stats::chi_square::{VarCI, chi2_pdf, chi2_cdf, chi2_sf, chi2_quantile, chi2_variance_ci}
use stats::fisher_f::{VarRatioCI, f_pdf, f_cdf, f_sf, f_quantile, f_var_ratio_ci}
use stats::wilcoxon::{WilcoxonResult, wilcoxon_signed_rank, wilcoxon_one_sample}
use stats::bland_altman::{BAResult, bland_altman}
use stats::power::{power_ttest_onesample, power_ttest_twosample, n_for_power_onesample, n_for_power_twosample}
use stats::qq_normal::{QQResult, qq_normal}
use stats::anova::{AnovaResult, anova_oneway}
use stats::reg_bands::{RegFit, reg_fit, reg_predict, reg_conf_band, reg_pred_band, reg_slope_ci}
use stats::correlation::{CorrResult, SpearmanCorr, pearson, spearman}
use stats::kruskal_wallis::{KWResult, kruskal_wallis}
use stats::tukey::{tukey_q_crit, tukey_hsd, tukey_pair_p, tukey_q_cdf, q_range_cdf}
use stats::chi2_independence::{ChiIndep, chi2_independence}
use stats::effect_size::{EffectSize, cohens_d, eta_squared_from_f}
use stats::proportion::{TwoPropResult, wilson_ci, clopper_pearson_ci, two_prop_z}
```

A worked end-to-end example that runs all five tools on one dataset lives at
`examples/stats/epistemic_suite_demo.sio`.

> **Note — no single-import facade yet.** A `mod.sio`-style `pub use` re-export
> surface (`use stats::epistemic_suite::{…}`) is what `stdlib/CONVENTIONS.md` §2
> prescribes, but the `lean_single` engine does not currently forward symbols
> through a `pub use` re-export — imported identifiers resolve only against the
> defining module. Until that compiler gap is closed, import from the five
> modules directly as above.

## Validated domain & limitations

This section records the numerical validity ranges honestly. The distribution
quantiles all ultimately depend on the special-function core, and that core has
a documented breakdown at extreme parameters — surfaced here, not hidden.

- **Student's t (`t_quantile`).** Validated for `df >= 2` (`df = 1`, the Cauchy
  limit, is out of scope). The quantile uses the **exact incomplete-beta inverse
  (`special::beta::ibeta_inv`) for `df <= 500`** and a **Cornish-Fisher expansion
  off the normal quantile for `df > 500`**; the two paths agree to `< 1e-5`
  across the crossover, and the split also routes around the extreme-parameter
  breakdown of `ibeta_inv` observed at `a >= ~1000` (`df >= ~2000`). Inline
  table cases run from `df = 2` up to `df = 100000`.

- **chi-squared (`chi2_quantile`).** Uses the **Wilson-Hilferty cube-root normal
  approximation as an initial guess, then Newton-Raphson on the exact CDF**
  (`special::igamma::chi2_cdf = P(k/2, x/2)`), so accuracy tracks that routine.
  Validated over the inline table for `k` from 1 to 10 and `p` from 0.05 to 0.95;
  at very large `k` accuracy inherits the incomplete-gamma limits below.

- **Fisher F (`f_quantile`).** A single `ibeta_inv` inversion, so it inherits
  `special::beta::ibeta_inv`'s reliable range — roughly `d1/2, d2/2` in `[1, 250]`,
  i.e. degrees of freedom up to about 500. Beyond that the underlying inverse is
  known to break.

- **Upstream special-function ceiling (`#841`).** `special::beta::ibeta` and
  `ibeta_inv` return values outside `[0, 1]` or clamp to their floor for extreme
  parameters (`ibeta` around `a ~ 5e4`; `ibeta_inv` around `a ~ 1e3` and at
  `a = 0.5`). This is tracked as **issue `#841`**. Consequently **F, chi-squared
  and t at very large degrees of freedom degrade** — the t module falls back to
  the Cornish-Fisher path to stay inside the validated region, while F and the
  raw `special::beta`-dependent paths degrade where `ibeta_inv` does. This is
  **documented, not silent**.

- **Wilcoxon.** Normal approximation with continuity correction; accurate for
  `n_eff` of roughly 15 or more. No exact permutation p-value for small samples.
  Fixed input capacity `n <= 256`.

- **Bland-Altman.** Requires `3 <= n <= 256`. Limits of agreement use the fixed
  1.96 normal factor; the CIs use the t-multiplier and therefore share the
  t-quantile validity above.

## Testing

Each module is self-validating: every source file carries inline tests against
published tables and hand-computed worked examples, and prints `ALL PASS` on
success. Run them under the lean_single engine:

```bash
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run stdlib/stats/student_t.sio
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run stdlib/stats/chi_square.sio
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run stdlib/stats/fisher_f.sio
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run stdlib/stats/wilcoxon.sio
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run stdlib/stats/bland_altman.sio
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run stdlib/stats/power.sio
```

Each command should print exactly `ALL PASS`. When running from outside the repo
root, set `SOUNIO_STDLIB_PATH=$(pwd)/stdlib` first.

## References

- Student (1908). "The probable error of a mean." *Biometrika* 6(1).
- Wilson, E. B. & Hilferty, M. M. (1931). "The distribution of chi-square."
  *PNAS* 17(12).
- Wilcoxon, F. (1945). "Individual comparisons by ranking methods."
  *Biometrics* 1(6).
- Fisher, R. A. & Cornish, E. A. (1960). "The percentile points of distributions
  having known cumulants." *Technometrics* 2(2).
- Abramowitz, M. & Stegun, I. A. (1964). *Handbook of Mathematical Functions*
  (§26.4 chi-squared, §26.6 F, §26.7 t).
- Bland, J. M. & Altman, D. G. (1986). "Statistical methods for assessing
  agreement between two methods of clinical measurement." *Lancet*
  1(8476):307-310.
- Snedecor, G. W. & Cochran, W. G. (1989). *Statistical Methods*, 8th ed.

## Licence

MIT / Apache-2.0 (same as Sounio).
