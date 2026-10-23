# Checked heteroscedasticity and simple-regression residual assessment

## Procedure

`stats::heteroscedasticity::koenker_bp_simple(&residuals,&x,n)`
implements the **studentized Koenker variant** with an intercept and one
variance predictor. It is the squared correlation between x and squared
residuals, equivalently the R-squared from auxiliary least squares:
LM = n * R-squared; reference df = 1.

The p-value is Q(1/2,LM/2), the asymptotic chi-square(1) survival. The existing
upper incomplete-gamma kernel uses its continued fraction directly in the upper
tail, avoiding 1-CDF cancellation for small p. Only its a=1/2, x in [0,50]
domain is exercised by this n<=100 procedure; no general gamma certification
follows. The generated reference also checks 12 points including both sides
of the series/continued-fraction boundary and p approximately 1.52e-23.

This is not the original Gaussian Breusch-Pagan score statistic.
Reference: Roger Koenker (1981), A note on studentizing a test for
heteroscedasticity, Journal of Econometrics 17(1),107-112.
https://doi.org/10.1016/0304-4076(81)90062-2

Caller supplies residuals from the intended fitted conditional-mean model.
The standalone function cannot establish that provenance. Sampling independence,
finite moments, mean-model specification and asymptotic calibration are not
assessed. n=3 is the computational minimum, not a finite-sample validity
guarantee. A large p is not probability of homoscedasticity. No Beta posterior,
overall pass flag or automatic model acceptance is returned.

## Availability

Finite f64 arrays with 3 <= n <= 100, selected prefix only.
Read status before numerical payload.

| status | Meaning |
|---|---|
| 0 | Statistic and asymptotic reference tail available |
| 1 | Invalid n |
| 2 | Nonfinite selected input |
| 3 | Constant variance predictor / zero centered variation |
| 4 | Constant squared residuals, so auxiliary R-squared is undefined |
| 5 | Arithmetic failure |

Unavailable lm, p, R-squared and residual_scale are zero sentinels.
n echoes the request; df=1 and asymptotic=true describe the reference convention
even when status is nonzero. Do not interpret a sentinel as inference.
Perfectly explained nonconstant squared residuals have R-squared=1 and LM=n;
their p is still positive in the supported domain.

## Numerical and resource policy

The core uses scalar passes only: **no arrays, heap allocation or auxiliary OLS
result object**. x is anchor-centered before scaling when subtraction is finite;
opposite-sign extremes use maximum-absolute scaling before subtraction.
Residual magnitudes are normalized before squaring. The squared-response
contrast is factored as (a-a0)/scale * (a/scale+a0/scale), preserving adjacent
magnitudes without subtracting nearly equal squares. Mean and covariance sums
are compensated. Correlation-squared avoids 1-SSE/SST cancellation near zero.
R-squared above one by at most 64 binary64 epsilons is rounded to one; larger
excursions are unavailable. Variation already rounded away on input is lost.
Bounded scalar storage is a source-level property, not a measured stack/RSS
or speedup claim.

## Model consumer

`stats::regression::assessments::assess_simple_linear(&x,&y,n)` fits the
existing centered/scaled simple OLS once, then assesses its residuals through
checked Jarque-Bera and Koenker. The result carries the original fit and both
procedure results, with separate statuses. On fit failure the two procedures
receive effective size zero and are unavailable; inspect fit.status for cause.
An exact fit can be valid OLS with undefined residual diagnostics.
Only the integration allocates bounded residual buffers for the existing
100/256-element contracts; the Koenker core itself is scalar.

## Legacy adapter and boundaries

`stats::epistemic::assumptions::breusch_pagan_test(residuals,x,n)` keeps
by-value arrays and delegates to the checked core, returning KoenkerResult.
The former private BPTestResult/p-value buckets and Beta conversion are removed
from this path. It is now public for direct checked use. Private in-file tests
were the only old callers. Required effects and exclusive sort borrows are
declared, and unsupported native fabs calls use a pure absolute-value helper.
Other private Shapiro/Levene/Anderson-Darling heuristics remain uncertified.

## Reference and scientific figure

C++23 long-double raw squared-residual auxiliary regression computes fitted
SSE/SST independently; standard-library erfc supplies reference tails.
Twelve datasets cover signed unit changes, offsets, subnormals, extreme x,
adjacent residual magnitudes, zero correlation, n=3 and n=100.
A complete model case independently checks OLS, residual JB and Koenker.

The synthetic n=20 figure renders the same model's squared normalized residuals
and its auxiliary least-squares line. Shared domains preserve geometry across
Canvas/SVG; dynamic labels print calculated LM and p. Connected observations
are ordered by predictor, not a density or confidence band. No real dataset or
clinical/genomic conclusion is implied. See the execution receipt for the
actual accepted job, source identity, visual inspection and control result.
