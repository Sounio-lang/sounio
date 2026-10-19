# Checked Shapiro-Wilk with Royston coefficients and calibration

`stats::shapiro_wilk::shapiro_wilk(&x,n)` accepts a selected prefix of
3..100 finite f64 observations, copies/sorts it, and returns W, a separately
computed 1-W, p, status, n, calibration, p_approximate and has_ties.
The input array is not mutated. Complete uncensored samples only.

## Procedure and calibration

The two most extreme normal-order weights use Royston's published polynomial
corrections; the remaining antisymmetric weights are normalized to their
remaining squared norm. Blom positions (i-3/8)/(n+1/4) use the existing bounded
normal quantile routine, whose domain covers all required positions.

W is the squared correlation with those weights, evaluated through its
projection residual norm: 1-W = sum((x-centered - projection*weight)^2) /
sum((x-centered)^2). This retains the small complement when displayed W rounds
to one. It is an independently organized implementation of the mathematical
procedure, not a claim to exact optimal covariance weights for every n.

| calibration | Statistical reference |
|---|---|
| 0 | Analytic n=3 p, using the arctangent form |
| 1 | Royston n=4..11 transformed normal approximation |
| 2 | Royston n=12..100 log-size polynomial normal approximation |

p_approximate is false only for the analytic n=3 calibration. Floating-point
arithmetic is finite precision in every branch. For n>=4 the p-value is
approximate, including when the numeric result is near one. No universal
accuracy, finite-sample exactness, power superiority or SOTA ranking is claimed.
Primary method: Royston (1995), JRSS C 44(4),547-551,
https://doi.org/10.2307/2986146.
The original procedure's broader range does not extend this bounded API.

The null reference is for an independent continuous normal sample.
has_ties reports repeated f64 observations; it does not repair quantization,
censoring, dependence or model residual correlation. A large p is not posterior
probability of normality. No BetaConfidence is constructed.

## Availability

| status | Meaning |
|---|---|
| 0 | W, 1-W and p computationally available |
| 1 | n outside 3..100 |
| 2 | Nonfinite selected input |
| 3 | Constant sample |
| 4 | Arithmetic failure |
| 5 | Calibration unavailable; W and 1-W remain available, p is zero sentinel |

Statuses 1..4 have zero numerical payload. Metadata identifies the requested
reference convention, not statistical validity. Always inspect status before
using a field. A zero sentinel is not a p-value. Status 5 avoids substituting
an arbitrary tiny p if a calibration transformation is outside its domain.
No missing values are discarded and no censoring is inferred.
No tail probability floor is imposed.

## Numerical and resource policy

Raw finite values are copied and heapsorted. Subtract the minimum before
normalizing by the finite range to preserve representable large-offset
contrasts. Opposite-sign range overflow uses scaled endpoints.
Mean, coefficient norm, projection and residual norms use compensated sums.
Direct upper incomplete gamma evaluates the normal survival without 1-CDF
cancellation in small p tails; n=3 uses the existing pure arctangent.
One 100-value data scratch array and one 50-value weight array; no heap and no
covariance matrix. Sorting is O(n log n), replacing the legacy bubble sort.
These are source-level storage/complexity properties, not measured RSS/speedup.

## Consumers and compatibility

The public by-value compatibility name
`stats::epistemic::assumptions::shapiro_wilk_test(data,n)` now delegates here
and returns ShapiroResult. Its former private Beta result, pedagogical weights
and bucketed p-values are removed from this path. Old in-file test is updated;
no external callers of the former private routine were found.
Private Levene/AD/KS and their p-to-Beta heuristics remain uncertified.

`stats::regression::assessments::assess_simple_linear` retains its existing
fit, Jarque-Bera and Koenker fields and adds shapiro. Residual buffers are reused.
On failed OLS, effective size zero makes Shapiro unavailable; on exact OLS,
constant residuals make it unavailable. Original n<=100 fit restrictions apply.
The raw-sample calibration is not validated for the correlated OLS residual
vector. This is a descriptive diagnostic, not regression-specific type-I
error calibration or an aggregate assumption-pass flag.

The synthetic n=20 Q-Q figure displays ordered actual OLS residuals against
Blom-position standard normal quantiles. A least-squares Q-Q reference line is
a visual guide, not the Shapiro coefficient vector or a confidence band.
W/p labels come from the checked procedure; the footer explicitly says residual
calibration is unassessed. Same IR feeds Canvas and SVG.

## Reference and acceptance

C++23 long-double reference uses erfc-based inverse-normal bisection,
a full antisymmetric weight vector and raw centered observations. Both direct
squared correlation and the independently accumulated projection residual
identity are checked. Royston coefficients/calibration formulas are shared
mathematics, not an independent proof of those approximations.
Standard-library atan/log/exp/erfc supply independent elementary functions.
24 datasets cover branch boundaries, unit/sign/offset invariants, opposite-sign
extremes, subnormals, ties, n=100 outlier tails and near-perfect weight profiles;
one independently fitted model checks the residual consumer.
See docs/plans/STATS_SHAPIRO_EXECUTION_2026_10_18.md for actual native receipts,
old-policy control, compiler/source identity and artifact inspection.
