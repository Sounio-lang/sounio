# Supplied-parameter NB2 saturated deviance

`nb2_deviance(count:i64,mean:f64,alpha:f64)` returns a scalar `Nb2Deviance`.
Mean and NB2 dispersion are supplied; variance convention is mean+alpha*mean².
This operation does not fit a model or estimate dispersion.

## Contract and availability

Input validation order matches the existing count diagnostic:

| status | Meaning |
|---|---|
| 0 | Signed deviance residual available; inspect deviance_status separately |
| 1 | Negative count |
| 2 | Nonfinite or nonpositive mean |
| 3 | Nonfinite or negative alpha |
| 4 | Count exceeds exact binary64 integer range 0..2^53 |
| 5 | Staged arithmetic unavailable |

`deviance_status`:0 available,1 square overflowed or rounded to zero despite a
nonzero residual,2 not computed. Unavailable numeric fields are zero placeholders;
never consume them without their status. `poisson_limit` records alpha==0 on a
successful result; negative zero also selects this branch. Count==mean returns
positive-zero residual/deviance. A nonzero contrast never returns a successful
zero residual. Status5 is conservative and does not imply the exact mathematical
result is unrepresentable. In particular the distant branch rejects a rounded-zero scaled contrast.

## Statistical quantity and arithmetic organization

For alpha>0, the per-observation quantity is

D = 2*[y*ln(y/mean) - (y+1/alpha)*ln((1+alpha*y)/(1+alpha*mean))].

At y=0 the convention y*ln(y)=0 gives D/2=ln(1+alpha*mean)/alpha.
At alpha=0 it uses the Poisson saturated-deviance formula. The signed residual is
sign(y-mean)*sqrt(D). The NB2 definition agrees with the
[official statsmodels family source](https://www.statsmodels.org/stable/_modules/statsmodels/genmod/families/family.html#NegativeBinomial).
That source defines the quantity; it does not validate this Sounio implementation.
No Python or third-party numerical implementation was executed.

The following is an algebraic derivation used to organize the implementation,
not a Lean proof or an independent floating-point verification.
For x=(y-mean)/mean, q=alpha*mean/(1+alpha*mean), the local branch |x|<=1/4 uses

D = Pearson_NB2² * 2*sum[n>=0] {(-x)^n*(1+q+...+q^n)/((n+1)*(n+2))}.

It keeps32 terms (n0..31), accumulated with compensation, and forms the residual
before squaring. q uses its limiting rounded value1 when alpha*mean overflows.
The scaled Pearson prefactor avoids constructing the variance. This description
is not a global ULP, correct-rounding or uniform relative-error claim.

For distant positive counts, with delta=y-mean, delta1=delta/(1+alpha*mean),
z=alpha*delta1, and m1=mean*(1+z), an equivalent positive decomposition is

D/2 = [y*ln(y/m1)-y+m1] + [z-ln(1+z)]/alpha.

The first bracket is a Poisson half-deviance. The second is nonnegative. m1 is
formed from positive scaled contributions, while delta1 is retained independently
rather than reconstructed by y-m1. Local component evaluation uses shared phi and
log1pmx kernels; below1e-16 their normalized leading factor1/2 is used. Far cases
use checked log1p for z>−1/2, with a checked-log difference for z<=−1/2
or positive overflow. The negative threshold avoids magnifying rounding of z
when forming its small complement 1+z; the independent logarithms are separated
from equality in this region. This is a staged numerical safeguard, not a
uniform relative-accuracy guarantee. Unexpected invalid signs are rejected, never clamped.
Component roots are combined by a scaled Euclidean norm. This preserves a finite
residual when its squared deviance cannot be represented.

The count-zero branch uses mean*log1p(t)/t symbolically, t=alpha*mean, with the
continuous ratio1 when t rounds to0. If t overflows, it evaluates log(t) through
log(alpha)+log(mean), omitting the negligible log1p(1/t) correction. Staged
binary64 rounding and underflow remain; not every representable exact result is
promised available. The scalar implementation uses no arrays or explicit heap.

### Balanced tiny-product continuation

When rounded alpha*mean is zero or subnormal, the distant positive-count branch
now forms alpha*y first and then m1=mean*(1+alpha*y)/(1+alpha*mean).
It retains delta1=(y-mean)/(1+alpha*mean) separately. This order avoids multiplying
the relative rounding error of subnormal alpha*mean by y. In this bounded region,
1+alpha*mean rounds to one; this remains staged binary64 arithmetic and is not a
correct-rounding certificate. Input count/mean/alpha validation and finite guards
remain required. Positive count<=2^53 bounds alpha*y in this branch.

If m1 itself is subnormal, its logarithm is carried independently as
ln(mean)+log1p(alpha*y)-log1p(alpha*mean). The far Poisson term consumes this
logarithm rather than ln(rounded m1), avoiding magnification of the coarse relative
rounding of a subnormal m1. The positive additive m1 remains rounded. This is an
algebraically motivated availability extension, with no global precision bound.
Other guards and independent residual/squared-deviance statuses are unchanged.
[Balanced continuation receipt](../../../docs/plans/NB2_BALANCED_EXECUTION_2026_10_22.md).

## Scientific limits and evidence

These are conditional descriptive diagnostics. There is no p-value, likelihood
optimizer, fitted-parameter correction, chi-square calibration, goodness-of-fit
acceptance, confidence interval or comparison calibrated across estimated models.
No epsilon repair converts a negative contribution into zero.

[Example](../../../examples/viz_nb2_deviance.sio) and
[execution receipt](../../../docs/plans/NB2_DEVIANCE_EXECUTION_2026_10_22.md)
track source/typecheck/native/figure evidence separately. Independent C++23/Sounio
concordance, boundary corpus, global accuracy, calibration and comparative resource
claims remain separate. No tests or independent numerical oracle were added or
run for this increment.

Balanced continuation execution: Slurm15874 completed0:0 after a fresh current-source
seed/Madaros build, five selected typechecks and real native figure execution.
The count8/mean1e-308/alpha1e-15 scenario reports residual/deviance statuses0; the
preceding frozen native report recorded status5. This establishes observed
availability in that demonstration, without independent numerical concordance,
global precision or full boundary coverage. No tests or numerical oracle were run.
