# Statistics Stdlib Implementation Status

## Current accepted native checkpoint

Corrected main3 Forge/Slurm15643 accepts108/108, zero failures/skips/errors,
including all105 previously accepted entries, CR0/CR1S reference/status and
the scientific cluster-covariance figure. The real SVG passes an independent
six-marker checker and three geometry mutation controls are rejected.
Failed v1 Slurm15637 (107/108, sole visualization timeout) and numerical
fault controls15638/15639 remain explicitly preserved in the
[cluster receipt](../../docs/plans/CLUSTER_COVARIANCE_EXECUTION_2026_10_19.md).

Earlier95-entry multi-domain acceptance is retained in its
[receipt](../../docs/plans/COUNTS_COVARIANCE_EXECUTION_2026_10_19.md).
Named executable receipts establish their scoped acceptance; this file does
not certify every stdlib function or scientific inference model.

## Previous statistical method checkpoint — 2026-10-19

The former blanket completion claims below are historical implementation notes,
not certification. Only the named procedures and executable receipts establish
current acceptance. See
[regression receipt](../../docs/plans/STATS_REGRESSION_INFLUENCE_EXECUTION_2026_10_17.md)
and [master sprint](../../docs/plans/STATISTICS_SCIENTIFIC_VISUALIZATION_SPRINT.md).

### Checked conditional weighted coefficient inference

Final Forge/Slurm15430:90/90, no failures/skips/errors,4:35, retaining86prior.
One-pass QR factors, compensated triangular inverse, balanced scaling and
original-order covariance. Caller-declared independent Gaussian errors with
unknown common scale and known relative precisions; no assumption assessment.
Separate fit/covariance/inference statuses and pointwise confidence support
[0.5,0.999999]. Perfect/df0/unrepresentable outputs remain explicitly unavailable.
Independent C++23 quad30cases, scalar/tail checks and real Canvas/SVG forest
geometry; regenerated oracle fixture matches frozen bytes. Actual previous
Student control15422:89passes/one new numerical failure. First candidate15421
also89/90 was rejected for a prior-tail cancellation regression; final source
retains direct tails outside the near-origin complementary-beta branch.
See [receipt](../../docs/plans/STATS_WEIGHTED_INFERENCE_EXECUTION_2026_10_19.md)
and [contract](regression/WEIGHTED_INFERENCE.md). No measured speed/RSS claim.

### Checked weighted QR and tiny-projection repair

Final Forge/Slurm15405:86/86, no failures/skips/errors,4:17, retaining83prior.
Positive finite weights, borrowed stride8 designs p<=8,n<=256, normalized
sqrt weights, explicit representation/rank failures and separate raw/weighted
norms. weighted_multiple_linear_fit shares the numerical kernel; variance
model remains unassessed, without SE/p/CI. Shared QR top updates preserve
tiny useful projections. Actual previous-formula control15406:84passes,
two new numerical failures after successful native compilation,4:20.
C++23 quadmath40cases plus independent24marker SVG check are retained.
See [execution receipt](../../docs/plans/STATS_WEIGHTED_QR_EXECUTION_2026_10_19.md).

### Checked rectangular QR and multiple regression

Final Forge/Slurm15383:83/83, no failures/skips/errors,4:04, retaining80prior.
Borrowed physical-stride8 design, p<=8,n<=256, explicit intercept ownership;
column-equilibrated pivoted Householder QR and status/pivot/rank metadata.
Public stats::regression::multiple::multiple_linear_fit shares linalg kernel.
C++23 quadmath28cases use a different factorization; actual Canvas/SVG
calibration has independently checked16markers. Absolute-cutoff control15384
compiles and fails only the reference,82passes. Rank decision is a declared
heuristic; no SVD certificate, coefficient inference or measured RSS/speed claim.
See [execution receipt](../../docs/plans/STATS_MULTIVARIABLE_QR_EXECUTION_2026_10_18.md)
and [contract](../linalg/LEAST_SQUARES.md). Existing simple OLS is preserved.

### Checked Pearson/Spearman permutation calibration

Final Forge/Slurm15363:80/80, no failures/skips/errors,3:53, including all77
previous fixtures. Complete labelled enumeration n2..8; bounded seeded MC
n2..256/B1..100000; plus-one p and Wilson95 for raw hit probability.
Independent C++23 quadmath31cases and real Canvas/SVG consumer accepted.
Previous-bootstrap control15364 and omitted-rejection fault15365 each
compile and fail exactly one new numerical fixture, retaining79passes.
Zero-state negative-seed normalization repaired in shared percentile/BCa
bootstrap and means permutation; other valid streams are preserved.
Three fixed work arrays,6144byte payload; no measured speed/RSS superiority.
Exchangeability is unassessed; weak-null/cluster/blocked inference remains open.
See [execution receipt](../../docs/plans/STATS_CORRELATION_PERMUTATION_EXECUTION_2026_10_18.md)
and [contract](CORRELATION_PERMUTATION.md). The full stdlib is not certified.

### Checked Pearson/Spearman and scientific scatter

- Typed size/nonfinite/constant/arithmetic status, anchored scaled moments,
  explicit projection residual, average ties and deterministic heapsort.
- Distinct coefficient/p/CI validity; caller-declared Pearson normal model,
  explicit Spearman t approximation, bounded-normal Fisher approximation.
- Shared public256/fixed100 adapters; metadata expands result layouts.
- Slurm15353:77/77,0 failures/skips/errors,3:45. Raw-centering control15356:
  76pass+1 numerical failure; prior-scatter control15357:76pass+1 markup failure.
  Both retain every previous73 fixture and differ at one source file.
- C++23 quadmath:58 estimates+6 inference cases, df1/df3 reference scope.
  Separate C++23 geometry checker verifies six real SVG circles/positions/style.
- Actual copied-record scatter count corruption repaired; other renderer
  bodies and unrelated inferential procedures preserved.
- No exact Spearman permutation p, coverage study or correlation BCa claim.
- [Coefficient contract](CORRELATION.md), [inference contract](CORRELATION_INFERENCE.md),
  [receipt](../../docs/plans/STATS_CORRELATION_EXECUTION_2026_10_18.md).

### Checked Kolmogorov-Smirnov

- Fully specified normal-reference and two-sample statistics, typed size/data/
  reference/arithmetic statuses, asymptotic metadata and pooled tie flag.
- Complete tie-group ECDF comparisons with exact rank cross-products; shared
  heapsort and fixed buffers, no dynamic allocation. No measured speedup claim.
- Overflow-safe normal standardization and theta/direct-survival branches.
- Continuous-null approximate p with Stephens-style correction; not fitted-normal,
  discrete-null correction, exact finite-n inference or independence assessment.
- Final Slurm15249:60/60,zero failures/skips/errors. Old walk control15250:
  58 pass+2 numerical runtime failures; old tail control15251:59 pass+1.
  Both retain all prior57 and compile the failing fixtures successfully.
- C++23 raw long-double erfc + brute ECDF:22 cases, relative p checks down to
  approximately2.52233148426187875e-226; scalar and maximum-capacity domains.
- Real Canvas/SVG tied-step figure shows calculated D/p with an explicit
  asymptotic-continuous-null/ties-unassessed footer.
- [Contract](KS.md) and
  [receipt](../../docs/plans/STATS_KS_EXECUTION_2026_10_18.md).

### Checked fitted-normal Anderson-Darling

- Scaled compensated moments, direct normal small tails before logarithms,
  paired log-CDF/log-survival accumulation, shared256-value heapsort.
- One scratch data vector reused; no heap allocation or second log vector.
- Typed size/data/constant/arithmetic/calibration availability, approximate-p
  and tie metadata. A2 retained for unsupported calibration, never fake p=1.
- Stephens approximation restricted to n>=8,Astar<10; statistic n=3..256.
  Not a fixed-CDF test, discrete-null calibration, posterior, or KS/Lilliefors.
- Checked public legacy adapter replaces private toy AD; unused p-to-Beta
  confidence and moment/extern machinery removed from the adapter module.
- Final Slurm15241:57/57,0 failures/skips/errors. Control15242 restores the
  former erf/complement tail policy:53 previous pass + four runtime failures.
- Independent C++23 long-double raw moments/erfc covers18 datasets and all
  four calibration pieces; published coefficients intentionally shared.
- Real Canvas/SVG synthetic step ECDF and fitted-normal CDF show computed
  A2/p with approximate/no-confidence-band labels.
- [Contract](ANDERSON_DARLING.md) and
  [receipt](../../docs/plans/STATS_ANDERSON_DARLING_EXECUTION_2026_10_18.md).

### Checked median Brown-Forsythe and shared tail helpers

- Two-group ANOVA of median absolute deviations with direct approximate F
  reference, explicit size/data/degeneracy/arithmetic states and common-unit
  normalized deviation summaries. No absolute-unit variance floor or Beta map.
- Two100-value scratch arrays are reused; shared finite-prefix heapsort now
  serves Brown-Forsythe,Shapiro and legacy ordering.
- F/beta reciprocal-exponential saturation is replaced by math::pure::exp;
  positive subnormal tails and actual underflow are distinguished.
- Final Slurm15230:53/53,0 failures/skips/errors. Controlled former p/exp
  policies Slurm15231:48 prior passes+5 numerical runtime failures after compilation.
- C++23 raw long-double ANOVA plus independent positive quadrature:15 datasets,
  36 F tails,six analytic tail cross-checks and a beta-underflow regression.
- Real Canvas/SVG synthetic profiles show normalized absolute deviations and
  means,calculated F/p and explicit approximate-reference/non-variance labels.
- Two groups2..100 each only; no multigroup,shape/independence assessment,
  exact finite-sample calibration,coverage study or measured resource speedup.
- [Contract](BROWN_FORSYTHE.md) and
  [receipt](../../docs/plans/STATS_BROWN_FORSYTHE_EXECUTION_2026_10_18.md).

### Checked Shapiro-Wilk / Royston normality

- Published normal-order coefficient corrections and separate n=3,4..11,12..100
  calibration branches; typed status, tie metadata and approximate-p flag.
- Scaled compensated projection residual norm preserves 1-W near one;
  heapsort with100 data values+50 weights, no heap/covariance matrix.
- Legacy Shapiro entrypoint delegates and no longer fabricates Beta confidence.
  OLS assessments retain existing fields and add the checked Shapiro screen.
- Final Slurm15211:48/48,0 failures/skips/errors. Controlled former p buckets
  Slurm15212:45 pass,3 numerical runtime failures after compilation.
- C++23 covers24 datasets plus independently fitted model; elementary normal
  quantiles and raw-vector W identity are evaluated independently, sharing the
  published coefficients/calibration formulas.
- Synthetic residual Q-Q Canvas/SVG figure has dynamic W/p, a visual guide
  line and an explicit unassessed residual-calibration footer.
- This is3..100 complete observations, not censoring support, regression-specific
  calibration, exact covariance weights or uniform finite-sample p accuracy.
- [Contract](SHAPIRO_WILK.md) and
  [receipt](../../docs/plans/STATS_SHAPIRO_EXECUTION_2026_10_18.md).

### Checked Koenker residual heteroscedasticity

- Studentized LM=n*R-squared with one predictor and intercept, direct
  chi-square(1) upper tail, explicit status and asymptotic metadata.
- Scaled compensated scalar passes: no squared-residual vector, auxiliary
  fit arrays or heap allocation in the core. No measured RSS/speedup claim.
- Simple OLS consumer retains per-fit, Jarque-Bera and Koenker availability;
  no aggregate assumption-pass flag or p-to-Beta mapping.
- Legacy breusch_pagan_test delegates to checked core. Required effects,
  sort exclusive borrows and native-compatible abs helper are repaired;
  other legacy heuristics remain uncertified.
- Final Slurm15196:44/44,0 failures/skips/errors. Controlled former tail policy
  Slurm15197:41 pass,3 runtime p-value failures after successful compilation.
- Independent C++23 raw auxiliary OLS/erfc: 12 datasets, 12 tail points,
  complete model residual assessment and a real Canvas/SVG synthetic figure.
- [Contract](HETEROSCEDASTICITY.md) and
  [receipt](../../docs/plans/STATS_KOENKER_EXECUTION_2026_10_18.md).

### Checked moment normality

- Existing Jarque-Bera now has anchor-preserving scaled compensated moments,
  overflow-safe fallback normalization and a nonpositive exponential.
- Explicit size/nonfinite/constant/arithmetic statuses; unavailable p is a zero
  sentinel and must not be interpreted. n and asymptotic metadata are exposed.
- Final Slurm 15177: 40/40 focused tests, retaining all prior 37. Independent
  C++23 raw long-double moments cover 12 cases including subnormals, offsets,
  opposite-sign extremes and maximum-capacity outliers.
- Untouched old normality body 15178: 37 prior cases pass, new unit-invariance
  fixture compiles then fails its numerical assertion. Initial 15173 separately
  timed out on the old large-value path; it is not numerical assertion evidence.
- No finite-sample calibration, sampling-independence assessment, Shapiro-Wilk,
  heteroscedasticity certification or p-to-Beta inference is claimed.
- Legacy assumptions header now labels pedagogical/bucketed paths uncertified.
- [Contract](NORMALITY.md) and
  [receipt](../../docs/plans/STATS_NORMALITY_EXECUTION_2026_10_18.md).

### Checked BCa for the sample mean

- Shared percentile/BCa draw, variance, heapsort and empirical-quantile kernel.
- Anchored/scaled compensated means; scale-free mean jackknife acceleration
  without storing leave-one-out means; explicit bias/tie and tail-support diagnostics.
- Bounded normal CDF/inverse, no unimplemented native externs.
- Explicit unavailable states for degenerate distributions, extreme finite-sample
  bias, unsupported/pole-crossing adjusted tails and arithmetic failure.
- Legacy mean adapter delegates to the checked kernel; compilation effects,
  public return fields and missing test helper repaired.
- Final Slurm 15063: 37/37 focused cases. Nine independent C++23 BCa cases,
  25 normal CDF points and 11 inverse points; real Canvas/SVG comparison figure.
- Controlled old numerical body 15066: 35 pass, two runtime failures after compilation.
  Only effects, visibility and the missing pure test helper were repaired for control compilation.
- This is mean BCa only, not correlation/Spearman or arbitrary-statistic BCa,
  an assumption assessment, empirical coverage certification or uniform accuracy proof.
- [Contract](BCA.md) and [receipt](../../docs/plans/STATS_BCA_EXECUTION_2026_10_18.md).

### Checked permutation inference

- stats/permutation.sio now has checked result/status and an explicit exchangeability null.
- Full label enumeration up to 65536 assignments, or seeded Monte Carlo with
  corrected p-value and distinct Wilson diagnostics for raw tail probability.
- Anchored/scaled compensated group sums, relative tie policy, bounded rejection
  index sampling; normalized values plus one reused index/cursor vector.
- Final Slurm 14974: 32/32 focused tests, 14 C++23 reference cases and real
  Canvas/SVG Monte Carlo interval consumer. Old implementation 14961: 27 pass,
  unit-scaling fixture fails at runtime.
- Existing scalar signature/example retained. Assumptions remain unassessed.
- [Contract](PERMUTATION.md) and
  [receipt](../../docs/plans/STATS_PERMUTATION_EXECUTION_2026_10_17.md).

### Seeded percentile bootstrap

- stats/bootstrap.sio has checked bounded input, normalized compensated means,
  normalized Welford SE, type-7 percentile bounds and explicit status/seed metadata.
- Integer MINSTD seed normalization handles signed extremes and modulus multiples;
  rejection index mapping removes extra modulo bias. Generator quality remains bounded.
- In-place heapsort replaces insertion sort; no empirical performance claim.
- Final Slurm 14939: 27/27 focused tests; C++23 comparison over ten datasets;
  real Canvas/SVG interval consumer. Before 14929: 23 pass, seed regression fails at runtime.
- This certifies ordinary mean percentile bootstrap only. Other bootstrap
  modules, correlation BCa and dependence/coverage assessment remain unvalidated.
- [Contract](RESAMPLING.md) and
  [receipt](../../docs/plans/STATS_BOOTSTRAP_EXECUTION_2026_10_17.md).

### Simple regression and real influence

- Centered/scaled simple OLS in regression/ols.sio, with borrowed input arrays.
- Actual case-deletion refits, DFFITS, leverage, Cook distance and per-row status.
- Public regression/linear.sio adapter with Student slope intervals and explicit
  fit, inference, R-squared and coefficient-variance availability.
- Assumptions are explicitly **unassessed**. Beta(1,1) fields are compatibility
  sentinels, not combined confidence or validated assumption tests.
- Coefficient variance is squared analytic slope SE when representable, not a
  bootstrap estimate. The model is bounded to one predictor and n <= 100.
- Final Slurm job 14870: 18/18 focused cases; independent C++23 QR over 11 datasets,
  88 case-deletion refits, and a real Canvas/SVG DFFITS consumer.
- Controlled legacy-policy job 14871: 17 pass, 1 numerical failure.
- This does not establish full Shapiro-Wilk, Breusch-Pagan, Durbin-Watson,
  bootstrap, general multiple-regression or complete-stdlib correctness.

## Historical claims requiring method-specific audit

The following legacy sections retain their original descriptions for audit.
Their "complete" labels and coverage numbers have not been established by the
2026-10-17 focused checkpoint.

### ✅ Task 2: Full Assumption Checks

**File:** [`stdlib/stats/epistemic/assumptions.sio`](epistemic/assumptions.sio)

**Implemented Tests:**

#### 1. **Shapiro-Wilk Normality Test** (Full Implementation)
- Uses optimal coefficients from Shapiro & Wilk (1965)
- W statistic: measures deviation from normality
- Returns `BetaConfidence` (not binary p > 0.05)
- Sample size adjustment for small n

#### 2. **Levene's Test** (Full Implementation)
- Tests equality of variances between groups
- ANOVA on absolute deviations from median
- More robust than F-test to normality violations
- Returns graded confidence

#### 3. **Anderson-Darling Test**
- Alternative normality test
- A² statistic with known critical values
- Particularly sensitive to tail deviations

#### 4. **Breusch-Pagan Test**
- Homoscedasticity (constant variance)
- Regresses squared residuals on predictors
- LM = n·R² ~ χ²(p) under null

**Key Innovation:** All tests return `BetaConfidence`, not binary pass/fail!

**Status:** ✅ Complete (550+ lines)

---

### ✅ Task 3: Bootstrap Variance Estimation

**File:** [`stdlib/stats/epistemic/bootstrap.sio`](epistemic/bootstrap.sio)

**Implemented Methods:**

#### 1. **Bootstrap T-Test Variance**
```sio
fn bootstrap_ttest_variance(
    group1: [f64; 100], n1: i64,
    group2: [f64; 100], n2: i64,
    n_bootstrap: i64, seed: i64
) -> f64  // Variance on t-statistic
```

Replaces heuristic `t_variance = se_diff² * 0.1` with proper resampling.

#### 2. **Bootstrap Confidence Intervals**
- Percentile method
- Returns SE, lower bound, upper bound
- Up to 1000 bootstrap samples

#### 3. **Bootstrap Correlation Variance**
- Resample (xᵢ, yᵢ) pairs
- Compute correlation on each bootstrap sample
- Variance of bootstrap correlations

#### 4. **Bootstrap Regression Variance**
- Resample cases (rows)
- Fit regression on each sample
- Returns (var_β₀, var_β₁)

**Infrastructure:**
- Simple RNG (Linear Congruential Generator)
- Resample with replacement
- Sorting for percentile CIs

**Status:** ✅ Complete (450+ lines)

---

### ✅ Task 4: Integration Tests

**Gate:** `bash scripts/stats_scipy_e2e_gate.sh`

**Status:** ✅ `scripts/stats_scipy_e2e_gate.sh`

---

## Summary Statistics

| Module | File | Lines | Status |
|--------|------|-------|--------|
| **Phase 1: Core** | | | |
| Epistemic T-Test | `epistemic/inferential.sio` | ~320 | ✅ |
| Linear Regression | `regression/linear.sio` | ~610 | ✅ |
| Assumption Checks | `epistemic/assumptions.sio` | ~460 | ✅ |
| Bootstrap | `epistemic/bootstrap.sio` | ~470 | ✅ |
| Integration Tests | `scripts/stats_scipy_e2e_gate.sh` | gate | ✅ |
| **Phase 2: Clinical & Robust** | | | |
| Logistic Regression | `regression/logistic.sio` | ~660 | ✅ |
| Meta-Analysis | `clinical/meta_analysis.sio` | ~520 | ✅ |
| Power Analysis | `clinical/power_analysis.sio` | ~460 | ✅ |
| Robust Regression | `regression/robust.sio` | ~500 | ✅ |
| **Phase 3: Multivariate & Time Series** | | | |
| PCA | `multivariate/pca.sio` | ~500 | ✅ |
| Clustering (K-means) | `multivariate/cluster.sio` | ~420 | ✅ |
| ARIMA | `timeseries/arima.sio` | ~450 | ✅ |
| Kalman Filter | `timeseries/kalman.sio` | ~400 | ✅ |
| **Phase 4: Bayesian & Causal** | | | |
| Model Comparison | `bayesian/model_comparison.sio` | ~380 | ✅ |
| Causal Inference | `causal/propensity.sio` | ~550 | ✅ |
| **Total** | **14 modules** | **~7,020 LOC** | **✅ All Complete** |

*Note: Survival analysis (Kaplan-Meier, Cox PH, log-rank) and clinical trials (Bayesian adaptive) already exist in `stdlib/medical/survival.sio` and `stdlib/medical/trial.sio`.*

---

## Revolutionary Features Implemented

### 1. **Knowledge<T> Return Types**
Every statistical test returns epistemic values:
```sio
struct KnowledgeTTest {
    value: TTestResult,          // Classical stats
    variance: f64,                // Bootstrap variance
    confidence: BetaConfidence,  // Graded assumptions
    provenance: TTestProvenance  // Audit trail
}
```

### 2. **Graded Assumption Confidence**
Not binary pass/fail:
```sio
// Traditional: p > 0.05 → "normal" (binary)
// Sounio:      p = 0.08 → BetaConfidence(α=18, β=4) → "82% confidence"

let normality = check_normality(data, n)
// Returns Beta distribution, not boolean
```

### 3. **Confidence-Weighted Decisions**
```sio
if p_value < 0.05 && beta_mean(&confidence) >= 0.80 {
    approve()  // Significant + High Confidence
} else if p_value < 0.05 && beta_mean(&confidence) < 0.80 {
    recommend_replication()  // Significant BUT low confidence
}
```

### 4. **Provenance Tracking**
```sio
struct TTestProvenance {
    method: "Student1908",
    n1: 8, n2: 8,
    timestamp: unix_time,
    assumption_checks: AssumptionChecks {
        normality_group1: Beta(18, 4),
        normality_group2: Beta(15, 6),
        variance_equality: Beta(20, 3),
        sample_size_adequate: Beta(24, 16)
    }
}
```

FDA 21 CFR Part 11 compliant by design.

### 5. **Bootstrap Variance on Test Statistics**
Traditional: `t = 2.34` (point estimate)

Sounio: `t = 2.34 ± 0.15` (with bootstrap variance)

---

## Comparison to Traditional Libraries

| Feature | R | Python scipy | Julia Stats | **Sounio** |
|---------|---|--------------|-------------|------------|
| Return type | `htest` | tuple | struct | **Knowledge<T>** |
| Uncertainty on stats | No | No | No | **Yes (bootstrap)** |
| Assumption confidence | Binary | Binary | Binary | **Graded (Beta)** |
| Provenance tracking | Manual | Manual | Manual | **Automatic** |
| Regulatory compliance | Manual | Manual | Manual | **Built-in** |
| Confidence-weighted decisions | No | No | No | **Yes** |

---

## Phase 2 Complete: Clinical & Robust Methods

### ✅ Logistic Regression
**File:** [`stdlib/stats/regression/logistic.sio`](regression/logistic.sio)
- IRLS (Iteratively Reweighted Least Squares) for MLE
- Sigmoid/logit link functions
- Dose-response modeling: ED50/ED90 via delta method
- AUC computation (Mann-Whitney U concordance)
- Returns `KnowledgeLogistic` with odds ratios, convergence checks

**Status:** ✅ Complete (~660 lines)

### ✅ Meta-Analysis
**File:** [`stdlib/stats/clinical/meta_analysis.sio`](clinical/meta_analysis.sio)
- Fixed effects (inverse variance weighting)
- Random effects (DerSimonian-Laird τ² estimation)
- Heterogeneity: Cochran's Q, I², τ², H²
- Publication bias: Egger's regression test
- Confidence degrades with high I² and funnel asymmetry

**Status:** ✅ Complete (~520 lines)

### ✅ Power Analysis
**File:** [`stdlib/stats/clinical/power_analysis.sio`](clinical/power_analysis.sio)
- `qnorm()` via Abramowitz & Stegun rational approximation
- Sample size: t-test, paired t-test, two proportions, log-rank, correlation, equivalence (TOST)
- Confidence depends on effect size source reliability

**Status:** ✅ Complete (~460 lines)

### ✅ Robust Regression
**File:** [`stdlib/stats/regression/robust.sio`](regression/robust.sio)
- Huber ψ (k=1.345): min(1, k/|r|) weights
- Tukey bisquare ψ (c=4.685): (1-(r/c)²)² if |r|≤c
- IRLS with robust scale estimation (MAD × 1.4826)
- Breakdown point tracking (0% Huber, 50% Tukey)

**Status:** ✅ Complete (~500 lines)

---

## Phase 3 Complete: Multivariate & Time Series

### ✅ PCA (Principal Component Analysis)
**File:** [`stdlib/stats/multivariate/pca.sio`](multivariate/pca.sio)
- Column centering/standardization, covariance matrix
- Power iteration for eigenvalue/eigenvector extraction
- Matrix deflation for subsequent components
- Optimal component selection: Kaiser criterion, broken-stick, 80% variance
- Sample adequacy based on n/p ratio

**Status:** ✅ Complete (~500 lines)

### ✅ Clustering (K-means)
**File:** [`stdlib/stats/multivariate/cluster.sio`](multivariate/cluster.sio)
- Lloyd's algorithm with convergence detection
- Silhouette scores: per-observation s(i) = (b-a)/max(a,b)
- Calinski-Harabasz index
- Within/between/total SS decomposition
- Quality confidence based on silhouette thresholds

**Status:** ✅ Complete (~420 lines)

### ✅ ARIMA Time Series
**File:** [`stdlib/stats/timeseries/arima.sio`](timeseries/arima.sio)
- d-th order differencing
- Autocorrelation function (ACF)
- AR fitting via Yule-Walker equations (p=1,2)
- Auto-ARIMA: grid search minimizing AIC
- Forecasting with widening prediction intervals
- Ljung-Box residual diagnostic
- Stationarity check via AR coefficient bounds

**Status:** ✅ Complete (~450 lines)

### ✅ Kalman Filter (State Space)
**File:** [`stdlib/stats/timeseries/kalman.sio`](timeseries/kalman.sio)
- Scalar state space: x_{t+1} = F·x_t + w_t, y_t = H·x_t + v_t
- Forward pass: predict → innovate → Kalman gain → update
- Backward pass: Rauch-Tung-Striebel smoother
- Log-likelihood computation
- Innovation diagnostics: normality, independence (lag-1), stability

**Status:** ✅ Complete (~400 lines)

---

## Phase 4 Complete: Bayesian & Causal Inference

### ✅ Bayesian Model Comparison
**File:** [`stdlib/stats/bayesian/model_comparison.sio`](bayesian/model_comparison.sio)
- WAIC: log-sum-exp trick, pointwise contributions, SE
- DIC: D̄ - D(θ̄) effective parameters
- Model comparison table: Δ-WAIC, Akaike weights exp(-Δ/2)/Σ
- Confidence based on model separation (Δ > 2×SE)

**Status:** ✅ Complete (~380 lines)

### ✅ Causal Inference
**File:** [`stdlib/stats/causal/propensity.sio`](causal/propensity.sio)
- Propensity score estimation (logistic regression via IRLS)
- 1:1 nearest-neighbor matching with caliper
- E-values (VanderWeele & Ding 2017) for unmeasured confounding
- Mediation analysis: Baron & Kenny paths (a, b, c, c'), Sobel test
- Diagnostics: overlap, balance, positivity, confounding sensitivity

**Status:** ✅ Complete (~550 lines)

---

## How to Use

### Example: Epistemic T-Test

```sio
use stats::epistemic::inferential::{ttest_independent_epistemic, beta_mean}

fn main() {
    var drug: [f64; 100] = [15.2, 18.1, 12.5, 20.3, 16.8, 14.7, 19.2, 13.9]
    var placebo: [f64; 100] = [8.3, 10.1, 7.5, 11.2, 9.4, 8.9, 10.5, 7.8]

    let result = ttest_independent_epistemic(drug, 8, placebo, 8)

    // Traditional decision: p < 0.05 → approve
    // Epistemic decision: p < 0.05 AND confidence >= 0.80 → approve

    let p = result.value.p_value
    let conf = beta_mean(&result.confidence)

    if p < 0.05 && conf >= 0.80 {
        // Approve: Significant + High Confidence
    } else if p < 0.05 && conf < 0.80 {
        // CAUTION: Significant BUT low confidence
        // Recommendation: Replicate with larger n
    }
}
```

### Example: Linear Regression with Diagnostics

```sio
use stats::regression::linear::fit_linear_regression_epistemic

fn main() {
    var x: [f64; 100] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, ...]
    var y: [f64; 100] = [2.1, 5.2, 7.9, 11.1, 14.0, 16.8, ...]

    let model = fit_linear_regression_epistemic(x, y, 20)

    // Extract results
    let beta0 = model.value.coefficients[0]  // Intercept
    let beta1 = model.value.coefficients[1]  // Slope
    let r2 = model.value.r_squared

    // Check diagnostics
    let diag = &model.provenance.diagnostics
    let influential = diag.influential_points  // Cook's D > 1

    if influential > 0 {
        // Warning: influential points detected
    }

    // Check assumptions
    let norm_conf = beta_mean(&diag.normality_resid)
    let homosked_conf = beta_mean(&diag.homoscedasticity)

    if norm_conf < 0.70 || homosked_conf < 0.70 {
        // Consider robust regression or transformations
    }
}
```

---

## Testing

```bash
bash scripts/stats_scipy_e2e_gate.sh
```

---

## Future Extensions (Not Yet Implemented)

| Module | Priority | Description |
|--------|----------|-------------|
| `multivariate/factor.sio` | Medium | Factor analysis, SEM |
| `multivariate/discriminant.sio` | Medium | LDA, QDA |
| `multivariate/manova.sio` | Medium | Multivariate ANOVA |
| `timeseries/spectral.sio` | Medium | Periodogram, wavelets (EEG/fMRI) |
| `timeseries/causality.sio` | Medium | Granger causality, VAR |
| `bayesian/hierarchical.sio` | High | Multilevel/mixed models |
| `bayesian/priors.sio` | Medium | Prior elicitation, power priors |
| `regression/poisson.sio` | Medium | Count data (adverse events) |
| `regression/quantile.sio` | Low | Quantile regression |
| `regression/mixed.sio` | High | Linear mixed models |
| `regression/regularized.sio` | Medium | Ridge, LASSO, Elastic Net |
| `robust/mcd.sio` | Medium | Minimum Covariance Determinant |
| `robust/rank_tests.sio` | Medium | Wilcoxon, Kruskal-Wallis, Friedman |
| `causal/instrumental.sio` | Low | 2SLS, IV estimation |

---

## References

### Phase 1
- Student (1908): "The Probable Error of a Mean"
- Shapiro & Wilk (1965): "An Analysis of Variance Test for Normality"
- Levene (1960): "Robust Tests for Equality of Variances"
- Breusch & Pagan (1979): "A Simple Test for Heteroscedasticity"
- Cook (1977): "Detection of Influential Observations"
- Efron (1979): "Bootstrap Methods: Another Look at the Jackknife"

### Phase 2
- Agresti (2002): "Categorical Data Analysis" (logistic regression)
- DerSimonian & Laird (1986): "Meta-analysis in clinical trials" (random effects)
- Egger et al. (1997): "Bias in meta-analysis detected by funnel plot"
- Cohen (1988): "Statistical Power Analysis" (power/sample size)
- Huber (1964): "Robust estimation of a location parameter"
- Tukey (1960): "A survey of sampling from contaminated distributions"

### Phase 3
- Jolliffe (2002): "Principal Component Analysis"
- Lloyd (1982): "Least squares quantization in PCM" (K-means)
- Rousseeuw (1987): "Silhouettes: a graphical aid"
- Box & Jenkins (1970): "Time Series Analysis" (ARIMA)
- Kalman (1960): "A new approach to linear filtering and prediction"
- Rauch, Tung & Striebel (1965): "Maximum likelihood estimates" (RTS smoother)

### Phase 4
- Watanabe (2010): "Asymptotic equivalence of Bayes cross-validation" (WAIC)
- Vehtari et al. (2017): "Practical Bayesian model evaluation" (LOO)
- Rosenbaum & Rubin (1983): "The central role of the propensity score"
- VanderWeele & Ding (2017): "Sensitivity analysis" (E-values)
- Baron & Kenny (1986): "The moderator-mediator variable distinction"

### Sounio Language
- [Epistemic Types Paper](../../paper/sounio-epistemic-types.tex)
- [Programming Guide](../../docs/guide/LLM_PROGRAMMING_GUIDE.md)
- [Statistics Plan](../../../.claude/plans/shimmying-tickling-mist.md)

## Offset-aware Poisson mean fitting — 2026-10-19

poisson_fit_offset is numerically accepted in cumulative Forge15603:101/101,
including all previous95 fixtures. Fixed borrowed256x8 design/count/offsets;
explicit intercept; shared CPQR/Fisher scoring; typed rank/domain/arithmetic,
budget and sufficient-boundary statuses; score AND full-step convergence.
The stable local parametric objective change avoids absolute-objective and
independently rounded-predictor subtraction stalls. Independent C++23 reference:
26 cases, unchanged tolerances, maxshape and tiny coefficient checks. Before
control15604 compiles and fails only the reference; finalfocus15602 passes6/6.
This is not overdispersion assessment, negative-binomial RNA-seq inference,
coefficient standard errors or confidence intervals. No such inference is claimed.
See docs/plans/POISSON_LORENTZ_EXECUTION_2026_10_19.md for source/receipt boundaries.

## Validation-resource checkpoint — 2026-10-19

Job15613:101/101,zero failures/skips/errors,exit0:0,5:05. Maximum-shape Poisson
reference compacted with exact input/expected-bit identity; all26 cases and their
assertions/tolerances remain. Every stdlib .sio source is unchanged. Last-row255
mutation15614 compiles and fails its numerical assertion. Phase profiler15615 has
21 accepted rows; faulted profiler15616 propagates failures and withholds readiness.
Independent C++23 analysis rejects missing/failed/nonfinite/zero-RSS observations.
Observed compile CPU median7.59s→5.38s, ELF2150856→885192bytes; three pairs on one
host, descriptive only. Direct-run wall median remains0.03s. Whole101 corpus does
not demonstrate an overall speedup. See docs/plans/STATS_VIZ_PHASE_PROFILE_EXECUTION_2026_10_19.md.

### Experimental HC3 implementation — 2026-10-19

regression/robust_covariance.sio now provides covariance-only OLS HC3 from the
existing pivoted/scaled QR factors. Native typecheck passes in Slurm15622;
independent C++23 reference syntax passes on t560. Numerical concordance and
executable consumers remain pending. This is not covered by the earlier101/101
acceptance. No cluster covariance or calibrated t/p/CI is claimed.

### HC3 concordance and counts-to-figure acceptance — 2026-10-19

This supersedes the earlier experimental-only checkpoint: native Slurm15625
accepts105/105, zero failures/skips/errors, including24-case HC3 reference/status,
incremental count TSV/QC and counts-to-Canvas/SVG. Independent C++23 generators
reproduce the fixtures and an exact six-marker checker accepts the native SVG.
HC0 weighting and overflow-to-zero controls compile and fail in15626/15629.
Rounded-leverage/representability limits remain; no cluster inference or broad
statistical calibration is claimed. Full compiler CI/merge remain separate.

Refreshed remote main ea36493e is incorporated in main3; Slurm15633 repeats105/105, zero failures/skips/errors.

### Checked single-way clustered OLS covariance — 2026-10-19

regression::robust_covariance::multiple_linear_cluster adds covariance-only
CR0 and CR1S from existing checked pivoted/scaled QR factors. CR1S is explicitly
G/(G-1)*(n-1)/(n-p), with returned correction_factor; it is not the bare
G/(G-1) convention. Signed label equality, compensated within-cluster influence
sums BEFORE outer products, finite/representability statuses and a shared
LASSQ/Gram path preserve the HC3 statistical procedure.

Independent C++23 quad raw-score aggregation/triangular solves cover31cases;
relative nonzero coefficient/SE/diagonal checks have no1+abs floor. Cases include
arbitrary signed/relabelled/noncontiguous groups, extreme units, permutations,
G256 and nonzero G2 withp8. Exact cancellation is checked conditional on actual
reported residuals rather than forcing analytic zeros.

Final native v2 Slurm15643 passes108/108, zero failures/skips/errors, retaining
all105 prior entries. Corpus phase212s; overall215s, not a paired speed claim.
The real native figure passes the independent six-marker checker; position,
count and radius mutations each exit2. V1 Slurm15637 remains107/108 with the
sole missing-increment visualization timeout. Controls15638 (wrong partition)
and15639 (omitted CR1S correction with unchanged metadata) compile and fail
their numerical assertions. The corrected figure is bound by the receipt's
source/SVG hashes and accepted snapshot e4c93240fc9ed22a6d70c2fa51d6f6bc17b4b53a180bcd11f181876996e0859b.

G>=2 andn>p are arithmetic availability rules, not few-cluster calibration.
G<=p is permitted and covariance may be singular. No clustered t/p/CI,
independence assessment, finite-sample coverage or measured speed/RSS superiority
is claimed. Fixed buffers include two2048-element group sum/correction arrays
plus bounded labels/indices and nested QR/result storage.
See [contract](regression/CLUSTER_COVARIANCE.md) and
[execution receipt](../../docs/plans/CLUSTER_COVARIANCE_EXECUTION_2026_10_19.md).
