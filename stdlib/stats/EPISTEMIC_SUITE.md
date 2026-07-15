# Epistemic Statistics Suite

## Overview

The **epistemic statistics suite** is a cohesive, pure-Sounio, dependency-light
toolkit that spans the full applied-biostatistics arc — **70 modules** across
fourteen families:

- **Distributions & fitting** — Student's t, chi-squared, Fisher-Snedecor F
  (full `pdf`/`cdf`/`sf`/`quantile` surfaces), a `densities` bank, Weibull /
  negative-binomial, and method-of-moments fitting for the gamma, beta and
  lognormal.
- **Robust descriptives** — median / IQR / MAD / trimmed & Winsorized means,
  the Hodges-Lehmann estimators, and Tukey / Grubbs / modified-z outlier rules.
- **Assumption checks** — normality (Jarque-Bera, Q-Q, Kolmogorov-Smirnov,
  χ²/G goodness-of-fit), independence (Wald-Wolfowitz runs, Durbin-Watson,
  autocorrelation + Ljung-Box) and homoscedasticity (Bartlett, Levene /
  Brown-Forsythe, the variance F-test).
- **Group comparison** — one-way ANOVA, inference from summary statistics,
  standardised effect sizes, and power / sample-size planning.
- **Non-parametric tests** — sign test, Mann-Whitney, Wilcoxon signed-rank,
  Kruskal-Wallis, Mood's median, Friedman and Tukey HSD.
- **Correlation & association** — Pearson / Spearman, the Fisher-z CI,
  point-biserial and partial correlation, Kendall's τ, Goodman-Kruskal γ and
  Somers' D.
- **Regression** — OLS with bands, weighted LS, logistic and Poisson GLMs, and
  the Deming / Theil-Sen / Passing-Bablok method-comparison fits.
- **Categorical & exact** — χ² independence, Fisher's exact 2×2, McNemar,
  Cochran's Q, the Cochran-Armitage trend test and proportion CIs.
- **Agreement & reliability** — Bland-Altman, Lin's CCC + Cliff's δ, Cohen's and
  Fleiss' κ, and ICC / Cronbach's α.
- **Diagnostic accuracy** — sensitivity / specificity / likelihood ratios and
  ROC AUC.
- **Survival analysis** — Kaplan-Meier, Nelson-Aalen cumulative hazard,
  restricted mean survival time, the actuarial life table, the log-rank test and
  a parametric exponential fit.
- **Meta-analysis & epidemiology** — fixed / random-effects pooling, RR / OR /
  NNT, and person-time incidence rates.
- **Bayesian** — conjugate updates with credible sets.

Every routine is written in Sounio with no FFI, no sampling dependency and no
external solver: the analytic core rests only on the special-function modules
(`special::beta`, `special::gamma`, `special::igamma`, `special::erf`), and every
tail, quantile and p-value is built from those by hand. Every non-trivial routine
returns an **explicit result struct** carrying the point estimate together with a
confidence interval or the test statistics, z, p-value and effect size — so
callers receive auditable numbers rather than a bare scalar. Each module ships
inline tests against hand-derived textbook constants and is independently
math-reviewed by an orthogonal LLM before merge (audit trail in
`.claude/llm_offload_log.md`); the whole suite runs green under the `lean_single`
engine (`scripts/stats_epistemic_suite_selftest.sh`).

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

### `stats::densities` — direct pdf / cdf / pmf

Value (not log-density) API for the common distributions: `normal`, `exponential`,
`gamma`, `beta`, `lognormal`, `uniform` (pdf/cdf) and `poisson`, `binomial`,
`geometric` (pmf/cdf). CDFs use the special-function core (igamma/ibeta). Adds the
binomial/lognormal/geometric missing from `prob::distributions` (which exposes
log-densities). 25 cases validated (incl. pmf-sums-to-1).

### `stats::bayes_conjugate` — conjugate Bayesian updates

| Function | Signature |
|---|---|
| `beta_binomial` | `pub fn beta_binomial(a0: f64, b0: f64, k: i64, n: i64, conf: f64) -> BetaPosterior with Mut, Div, Panic` |
| `normal_normal` | `pub fn normal_normal(m0: f64, v0: f64, xbar: f64, lik_var: f64, n_obs: i64, conf: f64) -> NormalPosterior with Mut, Div, Panic` |

Exact posteriors with **credible intervals** (a probability statement about the
parameter). Beta-Binomial → `BetaPosterior` (α, β, mean, mode, variance, credible
interval via Beta quantiles); Normal-Normal (known likelihood variance) →
`NormalPosterior`. Validated: Beta(1,1)+8/10 → Beta(9,3), mean 0.75, mode 0.8;
N(0,1) prior + x̄=2 (n=4) → posterior 1.6 ± credible [0.723, 2.477].

> **Not included: survival analysis.** A clean `stats::survival` (Kaplan-Meier +
> Greenwood + log-rank) was attempted but is blocked by a `lean_single` engine
> limitation — a module with several `[128]`/`[256]` arrays live across a nested
> call produces a crashing binary. A full private implementation (with Cox PH /
> Nelson-Aalen) exists in `medical::survival`; exposing it hits the same wall.
> Tracked for the compiler lane.

A capstone example, `examples/stats/full_analysis_report.sio`, drives seven of
these modules over one dataset and prints a formatted report.

### `stats::diagnostic` — diagnostic test accuracy (2×2)

`pub fn diag_metrics(tp: i64, fp: i64, fn_: i64, tn: i64, conf: f64) -> DiagResult with Mut, Div, Panic`
— sensitivity, specificity, PPV, NPV, accuracy, likelihood ratios (LR±), Youden's J,
with Wilson CIs on sensitivity and specificity. Validated: TP/FP/FN/TN=90/20/10/80 →
sens 0.9, spec 0.8, LR+ 4.5, LR− 0.125, J 0.7.

### `stats::cohen_kappa` — chance-corrected agreement

| Function | Signature |
|---|---|
| `cohen_kappa` | `pub fn cohen_kappa(table: &[f64; 256], k: i32, conf: f64) -> KappaResult with Mut, Div, Panic` |
| `cohen_kappa_weighted` | `pub fn cohen_kappa_weighted(table: &[f64; 256], k: i32, wtype: i32, conf: f64) -> KappaResult with Mut, Div, Panic` |

κ = (p_o − p_e)/(1 − p_e) from a k×k rating table, unweighted or weighted (0=linear,
1=quadratic), with a Landis–Koch interpretation code. The SE/CI are exact for the
unweighted kappa and approximate for the weighted. Validated: 2×2 → p_o 0.7, p_e 0.5,
κ 0.4.

### `stats::roc` — ROC area under the curve

| Function | Signature |
|---|---|
| `roc_auc` | `pub fn roc_auc(score: &[f64; 256], label: &[i64; 256], n: i32, conf: f64) -> AUCResult with Mut, Div, Panic` |
| `roc_point` | `pub fn roc_point(score: &[f64; 256], label: &[i64; 256], n: i32, threshold: f64) -> (f64, f64) with Mut, Div, Panic` |

AUC = normalised Mann-Whitney U (probability a positive outscores a negative),
computed exactly by pairwise concordance; SE by Hanley & McNeil (1982), CI clipped
to [0,1]. `roc_point` gives one (TPR, FPR) operating point. Validated: AUC 0.75 on
the classic 4-point example; perfect separation → 1; a tied pair → 0.5.

### `stats::multiple_comparisons` — p-value adjustment

| Function | Signature |
|---|---|
| `bonferroni` | `pub fn bonferroni(p: &[f64; 64], m: i32, alpha: f64, out_adj: &![f64; 64]) -> i64 with Mut, Div, Panic` |
| `holm` | `pub fn holm(p: &[f64; 64], m: i32, alpha: f64, out_adj: &![f64; 64]) -> i64 with Mut, Div, Panic` |
| `bh_fdr` | `pub fn bh_fdr(p: &[f64; 64], m: i32, alpha: f64, out_adj: &![f64; 64]) -> i64 with Mut, Div, Panic` |

Adjust m raw p-values for multiple testing (writes adjusted p into `out_adj` in the
same order, returns the count rejected at `alpha`): Bonferroni and Holm control the
FWER, Benjamini-Hochberg the FDR. Validated: p=0.01..0.05 → Bonferroni/Holm reject
1, BH rejects all 5. (Library module — validation in `examples/stats/`.)

### `stats::permutation` — two-sample permutation test

`pub fn perm_test_means(x: &[f64; 256], nx: i32, y: &[f64; 256], ny: i32, iters: i32, seed: i64) -> f64 with Mut, Div, Panic`
— distribution-free test of a location difference: p = (1 + #{|Δ*|≥|Δ_obs|})/(iters+1)
over `iters` random relabellings (inline LCG, deterministic in `seed`). The mean
bootstrap CI already lives in `stats::epistemic::bootstrap`. Validated: separated
groups → p<0.05; identical → p>0.2.

> **Codegen caveat (#852):** these two are library modules with external tests
> (`examples/stats/*_test.sio`) rather than inline `main` tests — a routine with
> big local arrays, or an in-module test harness making many nested calls while
> arrays are live, triggers a silent native-codegen crash. The public functions
> are structured around it (no big callee arrays; inlined loops).

### `stats::summary_inference` — inference from summary statistics

Tests/effect sizes from published (mean, SD, n) rather than raw data — scalar, so
`#852`-safe. `t_test_summary` (Welch + Satterthwaite df), `one_sample_t_summary`,
`ci_mean_summary`, `cohens_d_summary`. Validated: Welch t≈3.30, df≈52.9, d≈0.874.

### `stats::meta_analysis` — fixed / random-effects meta-analysis

| Function | Signature |
|---|---|
| `meta_fixed` | `pub fn meta_fixed(effect: &[f64; 64], se: &[f64; 64], k: i32, conf: f64) -> MetaResult with Mut, Div, Panic` |
| `meta_random` | `pub fn meta_random(effect: &[f64; 64], se: &[f64; 64], k: i32, conf: f64) -> MetaResult with Mut, Div, Panic` |

Inverse-variance pooling with Cochran's Q, I², and DerSimonian-Laird τ² for the
random-effects model. `MetaResult`: `pooled, se, ci_lo/hi, z, p_value, q, df, i2,
tau2`. Validated: 3 studies → pooled 0.477, Q 2.69, I² 25.6%, τ² 0.0072. A
**forest plot** figure (`examples/stats/forest_plot.sio`) renders it — per-study
CI bars + the pooled diamond, pure-Sounio.

### `stats::epidemiology` — risk / odds measures (2×2)

`pub fn epi_2x2(a: i64, b: i64, c: i64, d: i64, conf: f64) -> EpiResult with Mut, Div, Panic`
— risk ratio, odds ratio and risk difference (each with a CI), plus ARR, RRR and
NNT. RR/OR CIs on the log scale, RD via the binomial SEs. Validated:
15/85/5/95 → RR 3.0, OR 3.35, RD 0.10, NNT 10.

### `stats::sample_size` — sample size for proportions & correlation

| Function | Signature |
|---|---|
| `n_two_props` | `pub fn n_two_props(p1: f64, p2: f64, alpha: f64, power: f64) -> i64 with Mut, Div, Panic` |
| `n_one_prop` | `pub fn n_one_prop(p0: f64, p1: f64, alpha: f64, power: f64) -> i64 with Mut, Div, Panic` |
| `n_correlation` | `pub fn n_correlation(r: f64, alpha: f64, power: f64) -> i64 with Mut, Div, Panic` |

Extends `stats::power` (t-tests) to the other common designs. Validated:
`p=0.5→0.3` → ~93/group; `p=0.5→0.7` one-sample → ~47; `r=0.3` → ~85.

A **box plot** figure (`examples/stats/box_plot.sio`) renders three groups with
quartile boxes, median lines, 1.5·IQR whiskers and outlier points.

### `stats::reliability` — inter-rater & internal-consistency reliability

| Function | Signature |
|---|---|
| `icc` | `pub fn icc(data: &[f64; 256], n_subj: i32, n_rater: i32) -> ICCResult with Mut, Div, Panic` |
| `cronbach_alpha` | `pub fn cronbach_alpha(data: &[f64; 256], n_subj: i32, n_item: i32) -> f64 with Mut, Div, Panic` |

ICC(2,1) (two-way random, single rater) via the ANOVA mean squares, and Cronbach's
alpha for scale internal consistency, from a subjects×raters/items matrix.
Validated: perfect agreement → ICC 1.0; a 4×3 example → α 0.930.

### `stats::concordance` — Lin's CCC & Cliff's delta

| Function | Signature |
|---|---|
| `lin_ccc` | `pub fn lin_ccc(x: &[f64; 256], y: &[f64; 256], n: i32) -> CCCResult with Mut, Div, Panic` |
| `cliff_delta` | `pub fn cliff_delta(x: &[f64; 256], nx: i32, y: &[f64; 256], ny: i32) -> DeltaResult with Mut, Div, Panic` |

Lin's concordance correlation coefficient for paired-measurement agreement about
the identity line — `CCC = 2·σxy/(σxx+σyy+(μx−μy)²)`, decomposed into precision
(Pearson ρ) and accuracy (Cb) — and Cliff's delta, a robust distribution-free
dominance effect size. Validated: precision ≈ 0.9955, CCC ≈ 0.9942 on a worked
pair; perfect identity → CCC 1.0; fully-separated samples → δ = −1 (large).

### `stats::rate_epi` — incidence-rate epidemiology (person-time)

| Function | Signature |
|---|---|
| `incidence_rate` | `pub fn incidence_rate(a: f64, pt: f64) -> RateCI with Mut, Div, Panic` |
| `rate_ratio` | `pub fn rate_ratio(a1: f64, pt1: f64, a2: f64, pt2: f64) -> RatioCI with Mut, Div, Panic` |
| `rate_difference` | `pub fn rate_difference(a1: f64, pt1: f64, a2: f64, pt2: f64) -> DiffCI with Mut, Div, Panic` |

Person-time incidence rates with the ratio (IRR) and difference measures and
their large-sample intervals: a Poisson log CI for a single rate
(`SE(ln rate)=1/√a`), a log CI for the IRR (`SE(ln IRR)=√(1/a1+1/a2)`), and a
Wald CI for the rate difference (`SE=√(a1/pt1²+a2/pt2²)`). Validated: 10 events /
500 py → rate 0.02, 95% CI [0.0108, 0.0372]; IRR 2.0, 95% CI [0.936, 4.273].

### `stats::weibull_negbin` — Weibull & negative-binomial laws

| Function | Signature |
|---|---|
| `weibull_pdf` / `weibull_cdf` / `weibull_hazard` | `pub fn weibull_*(x: f64, k: f64, l: f64) -> f64 with Mut, Div, Panic` |
| `weibull_quantile` | `pub fn weibull_quantile(p: f64, k: f64, l: f64) -> f64 with Mut, Div, Panic` |
| `weibull_median` / `weibull_mean` | `pub fn weibull_*(k: f64, l: f64) -> f64 with Mut, Div, Panic` |
| `negbin_pmf` / `negbin_cdf` | `pub fn negbin_*(j: i32, r: f64, p: f64) -> f64 with Mut, Div, Panic` |
| `negbin_mean` | `pub fn negbin_mean(r: f64, p: f64) -> f64 with Mut, Div, Panic` |

The Weibull survival law (shape `k`, scale `λ`) with pdf/cdf/quantile/median/
hazard/mean, and the negative binomial (successes `r`, prob `p`, support = failure
count) with pmf/cdf/mean via log-gamma. Validated: Weibull(2,1) → F(1)=0.6321,
median 0.8326, mean Γ(1.5)=0.8862; NegBin(3,0.5) → pmf(0)=0.125, mean 3.

### `stats::mcnemar` — McNemar's test for paired binary data

| Function | Signature |
|---|---|
| `mcnemar` | `pub fn mcnemar(a: i64, b: i64, c: i64, d: i64) -> McNemarResult with Mut, Div, Panic` |

The paired-sample analogue of the 2×2 χ² test (before/after, matched
case-control): continuity-corrected `χ² = (|b−c|−1)²/(b+c)` with its asymptotic
p, the exact two-sided sign-test p (`X ~ Bin(b+c, ½)`), and the paired odds
ratio c/b. Validated: Agresti b=6/c=16 → χ²=3.6818, OR=2.667; exact b=1/c=9 →
p=0.02148.

### `stats::friedman` — Friedman test & Kendall's W

| Function | Signature |
|---|---|
| `friedman` | `pub fn friedman(data: &[f64; 256], n: i32, k: i32) -> FriedmanResult with Mut, Div, Panic` |

Repeated-measures (matched-block) non-parametric ANOVA — the paired counterpart
to Kruskal-Wallis — on a row-major n×k subject×treatment matrix, with mid-rank
tie handling. Returns `χ²_F = 12/(n·k·(k+1))·ΣRⱼ² − 3n(k+1)` (df k−1), its
asymptotic p, and Kendall's W = χ²_F/(n(k−1)) ∈ [0,1]. Validated: perfectly
concordant 3×3 → χ²=6, W=1.0; opposing ranks → χ²=0, W=0.

### `stats::trend` — Cochran-Armitage test for trend in proportions

| Function | Signature |
|---|---|
| `cochran_armitage` | `pub fn cochran_armitage(events: &[f64; 32], totals: &[f64; 32], scores: &[f64; 32], k: i32) -> TrendResult with Mut, Div, Panic` |

The categorical dose-response analogue of a slope test: is the event proportion
linearly trending across k *ordered* groups? Returns the signed trend z, `χ² =
T²/V` (df 1), two-sided p, and the fitted slope (proportion per unit score).
Validated: doses 0–3 with proportions 0.1–0.4 → χ²=26.667, z=5.164, slope=0.10;
flat proportions → χ²=0.

### `stats::ks_test` — Kolmogorov-Smirnov tests

| Function | Signature |
|---|---|
| `ks_one_normal` | `pub fn ks_one_normal(x: &[f64; 256], n: i32, mu: f64, sigma: f64) -> KSResult with Mut, Div, Panic` |
| `ks_two` | `pub fn ks_two(x: &[f64; 256], nx: i32, y: &[f64; 256], ny: i32) -> KSResult with Mut, Div, Panic` |

The distribution-free supremum goodness-of-fit test in both forms: one-sample
against a Normal(μ,σ) reference (`D = supₓ|Fₙ(x)−Φ(x)|`) and two-sample
(`D = supₓ|F₁−F₂|`), with the Kolmogorov asymptotic p-value (Stephens 1970
finite-sample λ). Data is copied and sorted internally. Validated: {−1,0,1} vs
N(0,1) → D=0.1747; fully-separated samples → D=1; identical → D=0, p=1.

### `stats::gof` — chi-squared & G goodness-of-fit

| Function | Signature |
|---|---|
| `gof` | `pub fn gof(observed: &[f64; 64], expected: &[f64; 64], k: i32, ddof: i32) -> GofResult with Mut, Div, Panic` |

Pearson's `χ² = Σ(Oᵢ−Eᵢ)²/Eᵢ` and the likelihood-ratio `G = 2ΣOᵢln(Oᵢ/Eᵢ)`
for binned counts, both against χ²(k−1−ddof), where `ddof` counts parameters
estimated from the data. Validated: O={10,20,30,40} vs E=25 → χ²=20, G=21.288,
df=3, p≈1.7e−4; perfect fit → χ²=G=0, p=1.

### `stats::normality` — Jarque-Bera omnibus normality test

| Function | Signature |
|---|---|
| `normality` | `pub fn normality(x: &[f64; 256], n: i32) -> NormalityResult with Mut, Div, Panic` |

The moment-based (skewness + kurtosis) complement to the visual `qq_normal`
check: reports sample skewness, kurtosis (normal=3) and excess, the Jarque-Bera
statistic `JB = n/6·(S²+(K−3)²/4) ~ χ²(2)`, and its closed-form p = e^{−JB/2}.
Validated: {2,4,4,4,5,5,7,9} → S=0.6563, K=2.7813, JB=0.5902, p=0.7445.

### `stats::km` — Kaplan-Meier survival summaries

| Function | Signature |
|---|---|
| `km` | `pub fn km(time: &[f64; 256], event: &[i64; 256], n: i32, t_query: f64) -> KMResult with Mut, Div, Panic` |

The non-parametric product-limit estimator `Ŝ(t) = Π(1−dⱼ/nⱼ)`, reported as
scalar summaries — Ŝ at a queried time, median survival (first death time with
Ŝ≤½), final Ŝ, and event count. Right-censoring (event=0) reduces the risk set
without a step. Validated: times 1–5 all deaths → S(2)=0.6, median 3; with
censoring {1,0,1,0,1} → S(4)=0.5333; tied deaths handled.

### `stats::logrank` — two-group log-rank test

| Function | Signature |
|---|---|
| `logrank` | `pub fn logrank(time: &[f64; 256], event: &[i64; 256], group: &[i64; 256], n: i32) -> LogRankResult with Mut, Div, Panic` |

The Mantel-Haenszel test for a difference between two survival curves:
observed vs expected deaths with the hypergeometric variance at each distinct
death time, giving `χ²=(O₁−E₁)²/V` (df 1) and the O/E hazard ratio. Validated:
interleaved-deaths example → χ²=0.4849, E₁=3.7667, HR=0.5929; identical groups
→ χ²=0, HR=1.

### `stats::exp_survival` — parametric exponential survival fit

| Function | Signature |
|---|---|
| `exp_survival` | `pub fn exp_survival(time: &[f64; 256], event: &[i64; 256], n: i32, t_query: f64) -> ExpSurvResult with Mut, Div, Panic` |

The MLE constant-hazard model from right-censored data — the parametric
companion to Kaplan-Meier: `λ̂ = D/T` with a log-scale 95% CI `λ̂·exp(±1.96/√D)`,
plus median `ln2/λ̂`, mean `1/λ̂`, and Ŝ(t)=e^{−λ̂t}. Validated: {2,3,5⁺,6,7}
with one censored → λ̂=0.1739, median 3.986, mean 5.75, S(4)=0.4988.

### `stats::robust` — robust location & scale

| Function | Signature |
|---|---|
| `median` / `quantile` | `pub fn median(x: &[f64; 256], n: i32) -> f64 …` · `quantile(x, n, p)` (type-7) |
| `iqr` / `mad` | `pub fn iqr(x, n) -> f64` · `mad(x, n) -> f64` (σ-consistent, ×1.4826) |
| `trimmed_mean` / `winsorized_mean` | `pub fn trimmed_mean(x, n, frac) -> f64` · `winsorized_mean(x, n, frac) -> f64` |

Outlier-resistant descriptives: type-7 (R/NumPy-linear) quantiles, the median,
IQR, the σ-consistent MAD, and symmetric trimmed / Winsorized means. Validated:
{1,2,3,4,5}→median 3, IQR 2, MAD 1.4826; {1,2,3,4,1000}→median & 20%-trimmed
mean both 3 (vs raw mean 202).

### `stats::hodges_lehmann` — Hodges-Lehmann estimators

| Function | Signature |
|---|---|
| `hl_one` | `pub fn hl_one(x: &[f64; 256], n: i32) -> f64 with Mut, Div, Panic` |
| `hl_two` | `pub fn hl_two(x: &[f64; 256], nx: i32, y: &[f64; 256], ny: i32) -> f64 with Mut, Div, Panic` |

The robust location estimators that invert the Wilcoxon tests: `hl_one` is the
median of the Walsh averages `(xᵢ+xⱼ)/2` (i≤j), `hl_two` the median of the
pairwise differences `xᵢ−yⱼ`. Validated: {1,2,3,4}→2.5; {1,2,3,4,100}→3.0
(robust); two-sample shift y=x+5 → −5.

### `stats::outlier` — outlier detection

| Function | Signature |
|---|---|
| `tukey_outliers` | `pub fn tukey_outliers(x: &[f64; 256], n: i32, k: f64) -> OutlierFences with Mut, Div, Panic` |
| `grubbs` | `pub fn grubbs(x: &[f64; 256], n: i32) -> GrubbsResult with Mut, Div, Panic` |
| `modified_z` | `pub fn modified_z(x: &[f64; 256], n: i32, thresh: f64) -> ModZResult with Mut, Div, Panic` |

Three complementary rules: Tukey boxplot fences (Q1−k·IQR, Q3+k·IQR), Grubbs'
single-outlier test `G=max|xᵢ−x̄|/s` with the exact two-sided Bonferroni t
p-value, and the MAD-based Iglewicz-Hoaglin modified z-score. Validated:
{…,100}→1 Tukey outlier; {1,2,3,4,5,50}→G=2.036, p<0.01, modified-z=20.9.

### `stats::deming` — Deming regression (errors-in-variables)

| Function | Signature |
|---|---|
| `deming` | `pub fn deming(x: &[f64; 256], y: &[f64; 256], n: i32, lambda: f64) -> DemingFit with Mut, Div, Panic` |

The method-comparison regression that allows measurement error in *both*
variables (λ = error-variance ratio; λ=1 → orthogonal). Closed-form slope
`b = [(Syy−λSxx)+√((Syy−λSxx)²+4λSxy²)]/(2Sxy)`, intercept ȳ−b·x̄. Validated:
perfect line → slope 2, intercept 1; staircase → slope 0.8828 (vs OLS 0.8).

### `stats::theil_sen` — Theil-Sen robust regression

| Function | Signature |
|---|---|
| `theil_sen` | `pub fn theil_sen(x: &[f64; 256], y: &[f64; 256], n: i32) -> TSFit with Mut, Div, Panic` |

Distribution-free line fit: slope = median of pairwise slopes, intercept =
median residual — resistant to ~29% arbitrary outliers. Validated: perfect line
→ (2,1); with a wild outlier at (5,100) still → (2,1); negative slope → −3.

### `stats::passing_bablok` — Passing-Bablok regression

| Function | Signature |
|---|---|
| `passing_bablok` | `pub fn passing_bablok(x: &[f64; 256], y: &[f64; 256], n: i32) -> PBFit with Mut, Div, Panic` |

The clinical-chemistry standard for method comparison: the shifted-median slope
(pairwise slopes with the −1 exclusion and the `K` = #{slope<−1} rank offset,
per Passing & Bablok 1983 / the R `mcr` package) and the median-residual
intercept. Validated: perfect line → (2,1); −1-slope pair correctly excluded;
K-offset case → slope 2.

### `stats::fisher_exact` — Fisher's exact test (2×2)

| Function | Signature |
|---|---|
| `fisher_exact` | `pub fn fisher_exact(a: i64, b: i64, c: i64, d: i64) -> FisherResult with Mut, Div, Panic` |

The exact small-sample test of 2×2 association (valid where χ² is not): the
two-sided p sums hypergeometric probabilities of all tables with P≤P(observed);
one-sided tails and the sample odds ratio too. Validated: tea-test [[3,1],[1,3]]
→ p_two=0.4857, p_greater=0.2429, OR=9; [[9,1],[1,9]] → p<0.01, OR=81;
[[5,5],[5,5]] → p=1, OR=1.

### `stats::cochran_q` — Cochran's Q test

| Function | Signature |
|---|---|
| `cochran_q` | `pub fn cochran_q(data: &[f64; 256], n: i32, k: i32) -> CochranQResult with Mut, Div, Panic` |

McNemar extended to k≥2 matched binary conditions: `Q=(k−1)(k·ΣCⱼ²−N²)/(k·N−ΣRᵢ²)
~ χ²(k−1)` on a row-major n×k 0/1 matrix. Validated: 4×3 example → Q=2.667, df=2;
one-condition-always-positive → Q=6, p<0.05; all-agree → Q=0.

### `stats::fleiss_kappa` — Fleiss' kappa (multi-rater)

| Function | Signature |
|---|---|
| `fleiss_kappa` | `pub fn fleiss_kappa(counts: &[f64; 256], subjects: i32, k: i32, raters: i32) -> FleissResult with Mut, Div, Panic` |

Cohen's κ extended to a fixed number of raters per subject: chance-corrected
agreement `κ=(P̄−P̄ₑ)/(1−P̄ₑ)` from a row-major N×k rater-count matrix. Validated:
worked 2-subject case → P̄=0.6667, P̄ₑ=0.7222, κ=−0.2; perfect within-subject
agreement → κ=1.

### `stats::logistic` — one-predictor logistic regression

| Function | Signature |
|---|---|
| `logistic_fit` | `pub fn logistic_fit(x: &[f64; 256], y: &[f64; 256], n: i32) -> LogisticFit with Mut, Div, Panic` |
| `logistic_predict` | `pub fn logistic_predict(fit: &LogisticFit, x: f64) -> f64 with Mut, Div, Panic` |

Maximum-likelihood logistic regression `logit(pᵢ)=b₀+b₁xᵢ`, fit by
Newton-Raphson (IRLS) — the workhorse for binary dose-response. Returns the
coefficients, standard errors, slope Wald z/p and convergence diagnostics.
Validated: two-point saturated design → b₀=−0.6931, b₁=1.3863, p̂(0)=⅓, p̂(1)=⅔;
balanced null → b₁=0.

### `stats::poisson_reg` — one-predictor Poisson regression

| Function | Signature |
|---|---|
| `poisson_fit` | `pub fn poisson_fit(x: &[f64; 256], y: &[f64; 256], n: i32) -> PoissonFit with Mut, Div, Panic` |
| `poisson_predict` | `pub fn poisson_predict(fit: &PoissonFit, x: f64) -> f64 with Mut, Div, Panic` |

Maximum-likelihood Poisson (log-link) count/rate regression `log(μᵢ)=b₀+b₁xᵢ`
by Newton-Raphson; `exp(b₁)` is the rate ratio per unit x. Validated: two-point
saturated design → b₀=ln3, b₁=ln4, rate ratio 4, μ̂(0)=3, μ̂(1)=12; flat counts
→ b₁=0.

### `stats::wls` — weighted least-squares regression

| Function | Signature |
|---|---|
| `wls_fit` | `pub fn wls_fit(x: &[f64; 256], y: &[f64; 256], w: &[f64; 256], n: i32) -> WLSFit with Mut, Div, Panic` |

Straight-line regression with per-observation weights (wᵢ∝1/σᵢ²) — the fit for
heteroscedastic assay data. Closed-form coefficients, standard errors from
`σ̂²(XᵀWX)⁻¹`, and a weighted R². Validated: perfect line → (1,2), R²=1;
downweighting an outlier pulls the slope from OLS 5 to 1.229.

### `stats::runs_test` — Wald-Wolfowitz runs test

| Function | Signature |
|---|---|
| `runs_test` | `pub fn runs_test(seq: &[i64; 256], n: i32) -> RunsResult with Mut, Div, Panic` |

Tests a 0/1 sequence for randomness against serial dependence: with n₁ ones, n₂
zeros and R runs, `z=(R−μ_R)/σ_R` where `μ_R=2n₁n₂/n+1`. Too few runs →
clustering, too many → over-alternation. Validated: balanced random-like →
R=6, z=0; clustered → R=2, z=−2.683, p<0.01; alternating → R=10, z=2.683.

### `stats::durbin_watson` — Durbin-Watson statistic

| Function | Signature |
|---|---|
| `durbin_watson` | `pub fn durbin_watson(e: &[f64; 256], n: i32) -> DWResult with Mut, Div, Panic` |

First-order serial-correlation diagnostic for residuals:
`d=Σ(eᵢ−e_{i−1})²/Σeᵢ² ∈ [0,4]`, with ρ̂≈1−d/2 (d≈2 → none, d<2 → positive,
d>2 → negative). Validated: alternating residuals → d=3.333, ρ̂=−0.667;
block residuals → d=0.667, ρ̂=0.667.

### `stats::autocorr` — autocorrelation & Ljung-Box

| Function | Signature |
|---|---|
| `acf` | `pub fn acf(x: &[f64; 256], n: i32, k: i32) -> f64 with Mut, Div, Panic` |
| `ljung_box` | `pub fn ljung_box(x: &[f64; 256], n: i32, m: i32) -> LBResult with Mut, Div, Panic` |

Sample autocorrelation at lag k and the Ljung-Box portmanteau test
`Q=n(n+2)Σr_k²/(n−k) ~ χ²(m)` for white noise up to lag m. Validated:
{1,2,3,4,5}→acf(1)=0.4, acf(2)=−0.1; Ljung-Box(m=2)→Q=1.517; a monotone ramp →
acf(1)>0.6, Q significant.

### `stats::nelson_aalen` — Nelson-Aalen cumulative hazard

| Function | Signature |
|---|---|
| `nelson_aalen` | `pub fn nelson_aalen(time: &[f64; 256], event: &[i64; 256], n: i32, t_query: f64) -> NAResult with Mut, Div, Panic` |

The non-parametric cumulative hazard `Ĥ(t)=Σd_j/n_j` with variance `Σd_j/n_j²`
and implied survival `Ŝ=e^{−Ĥ}` — the KM companion, better-behaved in the tail.
Validated: times 1–5 all deaths → Ĥ(3)=0.7833, Var=0.2136, Ŝ(3)=0.4569; with
censoring → Ĥ(4)=0.5333.

### `stats::rmst` — restricted mean survival time

| Function | Signature |
|---|---|
| `rmst` | `pub fn rmst(time: &[f64; 256], event: &[i64; 256], n: i32, tau: f64) -> f64 with Mut, Div, Panic` |

The area under the KM curve up to a horizon τ — a clinically interpretable
event-free-time summary, robust to non-proportional hazards. Exact truncated
step-function integral. Validated: times 1–5 → RMST(5)=3.0, RMST(3)=2.4,
RMST(2.5)=2.1; with censoring → RMST(5)=3.667.

### `stats::life_table` — actuarial (cohort) life table

| Function | Signature |
|---|---|
| `life_table_survival` | `pub fn life_table_survival(entering: &[f64; 64], deaths: &[f64; 64], withdrawn: &[f64; 64], k: i32, q_interval: i32) -> f64 with Mut, Div, Panic` |
| `life_table_hazard` | `pub fn life_table_hazard(entering, deaths, withdrawn, k, q_interval) -> f64` |

Interval-grouped survival for count-per-interval data: with the withdrawal-
adjusted risk set `n'ᵢ=nᵢ−wᵢ/2`, `qᵢ=dᵢ/n'ᵢ` and `Ŝ=Πpᵢ` (Cutler-Ederer).
Validated: 3-interval example → S=0.9, 0.8047, 0.7231; hazard(i1)=0.1059.

### `stats::kendall_tau` — Kendall's rank correlation

| Function | Signature |
|---|---|
| `kendall_tau` | `pub fn kendall_tau(x: &[f64; 256], y: &[f64; 256], n: i32) -> KendallResult with Mut, Div, Panic` |

Concordance-based rank correlation robust to outliers and monotone non-linear
relationships: τ_a=(C−D)/(n(n−1)/2), the tie-corrected τ_b, and the no-tie null
z-test. Validated: perfect concordance → τ=1; {1,3,2,4} → τ_a=0.667, z=1.359;
ties → τ_a=0.667 but τ_b=0.8; perfect discordance → τ=−1.

### `stats::goodman_kruskal` — Goodman-Kruskal gamma

| Function | Signature |
|---|---|
| `goodman_kruskal` | `pub fn goodman_kruskal(x: &[f64; 256], y: &[f64; 256], n: i32) -> GammaResult with Mut, Div, Panic` |

Ordinal association ignoring tied pairs entirely: γ=(C−D)/(C+D) with the
pair-count Wald z. Validated: {1,3,2,4} → γ=0.667; ties (using only untied
pairs) → γ=1; perfect discordance → γ=−1.

### `stats::somers_d` — Somers' D (asymmetric)

| Function | Signature |
|---|---|
| `somers_d` | `pub fn somers_d(x: &[f64; 256], y: &[f64; 256], n: i32) -> SomersResult with Mut, Div, Panic` |

Asymmetric ordinal association: D(Y|X)=(C−D)/(C+D+Tx) penalises ties on the
predictor, so for a binary Y it equals 2·AUC−1. Validated: no ties → D=0.667;
ties → D(Y|X)=D(X|Y)=0.8; binary Y with perfect separation → D(Y|X)=1 (AUC=1).

### `stats::bartlett` — Bartlett's test for equal variance

| Function | Signature |
|---|---|
| `bartlett` | `pub fn bartlett(values: &[f64; 256], sizes: &[i32; 16], k: i32) -> BartlettResult with Mut, Div, Panic` |

Homogeneity-of-variance test across k consecutive-block groups (powerful under
normality): `χ²=[(N−k)ln(s_p²)−Σ(n_i−1)ln(s_i²)]/C ~ χ²(k−1)`. Validated:
{1..5} vs {2,4..10} → pooled 6.25, χ²=1.5868, df=1; equal-spread → χ²=0.

### `stats::levene` — Levene / Brown-Forsythe test

| Function | Signature |
|---|---|
| `levene` | `pub fn levene(values: &[f64; 256], sizes: &[i32; 16], k: i32, center: i32) -> LeveneResult with Mut, Div, Panic` |

The distribution-robust variance test: one-way ANOVA on absolute deviations from
each group's centre (`center` = 0 mean → Levene, 1 median → Brown-Forsythe).
`W ~ F(k−1, N−k)`. Validated: worked example → W=2.0571 (both centerings on
symmetric data); equal-spread → W=0.

### `stats::var_ftest` — two-sample variance F-test

| Function | Signature |
|---|---|
| `var_ftest` | `pub fn var_ftest(x: &[f64; 256], nx: i32, y: &[f64; 256], ny: i32) -> VarFResult with Mut, Div, Panic` |

The classical variance-ratio test `F=s₁²/s₂² ~ F(n₁−1,n₂−1)` with a two-sided
p-value. Validated: var 2.5 vs 10 → F=0.25, p=0.208; equal variances → F=1, p=1.

### `stats::mann_whitney` — Mann-Whitney U / rank-sum test

| Function | Signature |
|---|---|
| `mann_whitney` | `pub fn mann_whitney(x: &[f64; 256], nx: i32, y: &[f64; 256], ny: i32) -> MannWhitneyResult with Mut, Div, Panic` |

The two-sample rank-sum test (unpaired counterpart to Wilcoxon signed-rank):
`U₁=R₁−n₁(n₁+1)/2`, tie-corrected normal-approximation z. Validated: fully
separated → U=0, z=−1.964, p=0.0495; overlapping → U₁=3, p>0.5; cross-group ties
→ R₁=8.

### `stats::sign_test` — sign test

| Function | Signature |
|---|---|
| `sign_test` | `pub fn sign_test(x: &[f64; 256], y: &[f64; 256], n: i32) -> SignResult with Mut, Div, Panic` |
| `sign_test_median` | `pub fn sign_test_median(x: &[f64; 256], n: i32, m0: f64) -> SignResult with Mut, Div, Panic` |

The simplest distribution-free test — signs only, exact binomial p (zeros
dropped). Validated: 7+/2− → p=0.1797; one-sample median → p=0.625.

### `stats::mood_median` — Mood's median test

| Function | Signature |
|---|---|
| `mood_median` | `pub fn mood_median(values: &[f64; 256], sizes: &[i32; 16], k: i32) -> MoodResult with Mut, Div, Panic` |

A robust k-sample equal-medians test: χ² on the 2×k above/below-pooled-median
table. Less powerful than Kruskal-Wallis but far more outlier-resistant.
Validated: {1..5} vs {6..10} → median 5.5, χ²=10, df=1; identical groups → χ²=0.

### `stats::fit_gamma` — gamma fitting (method of moments)

| Function | Signature |
|---|---|
| `fit_gamma` | `pub fn fit_gamma(x: &[f64; 256], n: i32) -> GammaFit with Mut, Div, Panic` |

Estimates the gamma shape and scale by matching moments: `k̂=mean²/var`,
`θ̂=var/mean`, `rate=mean/var`. Validated: {2,2,6,6} → mean 4, var 5.333,
shape 3, scale 1.333, rate 0.75.

### `stats::fit_beta` — beta fitting (method of moments)

| Function | Signature |
|---|---|
| `fit_beta` | `pub fn fit_beta(x: &[f64; 256], n: i32) -> BetaFit with Mut, Div, Panic` |

Estimates the two beta shape parameters for data on (0,1): with
`c=m(1−m)/v−1`, `α̂=m·c`, `β̂=(1−m)·c`. Validated: {.25,.25,.75,.75} →
α=β=1 (uniform); skewed data → α=3.536, β=3.264.

### `stats::fit_lognormal` — lognormal fitting

| Function | Signature |
|---|---|
| `fit_lognormal` | `pub fn fit_lognormal(x: &[f64; 256], n: i32) -> LogNormalFit with Mut, Div, Panic` |

Fits a normal to the logs (sample moments): `μ̂=mean(ln x)`, `σ̂=sd(ln x)`, with
the implied `median=e^{μ̂}` and `mean=e^{μ̂+σ̂²/2}`. Natural for right-skewed
positive data (concentrations, PK exposures). Validated: data e⁰…e³ → μ=1.5,
σ=1.291, median=4.482.

### `stats::corr_ci` — Fisher-z correlation CI & test

| Function | Signature |
|---|---|
| `corr_ci` | `pub fn corr_ci(r: f64, n: i32, conf: f64) -> CorrCIResult with Mut, Div, Panic` |

Turns a Pearson r into a confidence interval and a significance test via the
Fisher z-transform: `z=atanh(r)`, `SE=1/√(n−3)`, `CI=tanh(z±z_c·SE)`, plus the
exact `t=r√(n−2)/√(1−r²)` test of ρ=0. Composes with `stats::correlation`.
Validated: r=0.8, n=20 → z=1.0986, 95% CI [0.5534, 0.9177], t=5.657.

### `stats::point_biserial` — point-biserial correlation

| Function | Signature |
|---|---|
| `point_biserial` | `pub fn point_biserial(x: &[f64; 256], y: &[f64; 256], n: i32) -> PointBiserialResult with Mut, Div, Panic` |

The correlation between a continuous variable and a 0/1 dichotomy (the effect-
size companion to the two-sample t-test), with its `t(n−2)` test. Validated:
{1,2,3,4} vs {0,0,1,1} → r=0.8944, t=2.828; equal group means → r=0.

### `stats::partial_corr` — first-order partial correlation

| Function | Signature |
|---|---|
| `partial_corr` | `pub fn partial_corr(x: &[f64; 256], y: &[f64; 256], z: &[f64; 256], n: i32) -> PartialResult with Mut, Div, Panic` |

The correlation of x and y after removing the linear effect of z (confounding
control): `r_xy·z=(r_xy−r_xz·r_yz)/√((1−r_xz²)(1−r_yz²))` with a `t(n−3)` test.
Validated: r_xy=0 but r_xz=0.6, r_yz=0.8 → partial=−1; z uncorrelated with both
→ partial=r_xy.

A **funnel plot** figure (`examples/stats/funnel_plot.sio`) accompanies the
meta-analysis (effect vs precision with the pseudo-95% funnel, for publication-bias
assessment).

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
use stats::densities::{normal_pdf, gamma_pdf, beta_pdf, lognormal_pdf, poisson_pmf, binomial_pmf, geometric_pmf}
use stats::bayes_conjugate::{BetaPosterior, NormalPosterior, beta_binomial, normal_normal}
use stats::diagnostic::{DiagResult, diag_metrics}
use stats::cohen_kappa::{KappaResult, cohen_kappa, cohen_kappa_weighted}
use stats::roc::{AUCResult, roc_auc, roc_point}
use stats::concordance::{CCCResult, DeltaResult, lin_ccc, cliff_delta}
use stats::rate_epi::{RateCI, RatioCI, DiffCI, incidence_rate, rate_ratio, rate_difference}
use stats::weibull_negbin::{weibull_pdf, weibull_cdf, weibull_quantile, weibull_median, weibull_hazard, weibull_mean, negbin_pmf, negbin_cdf, negbin_mean}
use stats::mcnemar::{McNemarResult, mcnemar}
use stats::friedman::{FriedmanResult, friedman}
use stats::trend::{TrendResult, cochran_armitage}
use stats::ks_test::{KSResult, ks_one_normal, ks_two}
use stats::gof::{GofResult, gof}
use stats::normality::{NormalityResult, normality}
use stats::km::{KMResult, km}
use stats::logrank::{LogRankResult, logrank}
use stats::exp_survival::{ExpSurvResult, exp_survival}
use stats::robust::{median, quantile, iqr, mad, trimmed_mean, winsorized_mean}
use stats::hodges_lehmann::{hl_one, hl_two}
use stats::outlier::{OutlierFences, GrubbsResult, ModZResult, tukey_outliers, grubbs, modified_z}
use stats::deming::{DemingFit, deming}
use stats::theil_sen::{TSFit, theil_sen}
use stats::passing_bablok::{PBFit, passing_bablok}
use stats::fisher_exact::{FisherResult, fisher_exact}
use stats::cochran_q::{CochranQResult, cochran_q}
use stats::fleiss_kappa::{FleissResult, fleiss_kappa}
use stats::logistic::{LogisticFit, logistic_fit, logistic_predict}
use stats::poisson_reg::{PoissonFit, poisson_fit, poisson_predict}
use stats::wls::{WLSFit, wls_fit}
use stats::runs_test::{RunsResult, runs_test}
use stats::durbin_watson::{DWResult, durbin_watson}
use stats::autocorr::{LBResult, acf, ljung_box}
use stats::nelson_aalen::{NAResult, nelson_aalen}
use stats::rmst::{rmst}
use stats::life_table::{life_table_survival, life_table_hazard}
use stats::kendall_tau::{KendallResult, kendall_tau}
use stats::goodman_kruskal::{GammaResult, goodman_kruskal}
use stats::somers_d::{SomersResult, somers_d}
use stats::bartlett::{BartlettResult, bartlett}
use stats::levene::{LeveneResult, levene}
use stats::var_ftest::{VarFResult, var_ftest}
use stats::mann_whitney::{MannWhitneyResult, mann_whitney}
use stats::sign_test::{SignResult, sign_test, sign_test_median}
use stats::mood_median::{MoodResult, mood_median}
use stats::fit_gamma::{GammaFit, fit_gamma}
use stats::fit_beta::{BetaFit, fit_beta}
use stats::fit_lognormal::{LogNormalFit, fit_lognormal}
use stats::corr_ci::{CorrCIResult, corr_ci}
use stats::point_biserial::{PointBiserialResult, point_biserial}
use stats::partial_corr::{PartialResult, partial_corr}
```

Worked end-to-end examples that compose several tools on one dataset live at
`examples/stats/epistemic_suite_demo.sio` and `examples/stats/full_analysis_report.sio`.

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
