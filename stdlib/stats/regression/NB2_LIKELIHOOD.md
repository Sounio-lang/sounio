# Full scalar NB2 and Poisson log probability

stats::regression::nb2_likelihood provides:

    nb2_saturated_log_probability(count, alpha) -> Nb2Normalizer
    nb2_log_probability(count, mean, alpha) -> Nb2Likelihood

These are likelihood primitives. They do not fit coefficients or dispersion,
assess assumptions, supply p-values, or establish inferential calibration.

## Why saturated deviance alone is insufficient

For fixed alpha, minimizing saturated deviance gives the same beta objective
as maximizing NB2 likelihood. Across alpha, however, its saturated normalizer
also changes. Dispersion estimation must retain that alpha-dependent term.

The full scalar identity used here is

    log P(y | mean, alpha)
      = log P(y | mean=y, alpha) - deviance(y,mean,alpha)/2.

The count-zero saturated value is the boundary limit 0. At alpha0 the model is
Poisson. For alpha>0, r=1/alpha and the full definition is

    lgamma(r+y)-lgamma(r)-lgamma(y+1)
      + r*ln[r/(r+mean)] + y*ln[mean/(r+mean)].

The production normalizer avoids subtracting large log-gamma values.

## Saturated normalizer branches

Let C9(z) be the first nine Stirling log-gamma correction terms, z>=8:

    1/(12z) - 1/(360z³) + 1/(1260z⁵) - 1/(1680z⁷)
    + 1/(1188z⁹) - 691/(360360z¹¹) + 1/(156z¹³)
    - 3617/(122400z¹⁵) + 43867/(244188z¹⁷).

The implementation uses an inverse-power Horner polynomial. It never forms
large positive powers of z.

- Poisson, y1..7: y*ln(y)-y-sum[k1..y]ln(k).
- Poisson, y>=8: -0.5*(ln(2pi)+ln(y))-C9(y).
- NB2, r>=8: PoissonSat(y)-0.5*log1p(y/r)+C9(r+y)-C9(r).
- NB2, r<8 and y>=8:
  -0.5*(ln(r+y)+ln(y))-r+r*ln(r)-lgamma(r)+C9(r+y)-C9(y).
- NB2, r<8 and y1..7: the finite algebraic identity
  sum[k0..y-1]log1p(k/r)+y*ln(y)-ln(y!)-(r+y)*log1p(y/r).

For small positive r, lgamma(r) is evaluated through
lgamma(r+1)-ln(r), avoiding the shared gamma implementation's reflection route.
The finite-sum branch is an exact real-arithmetic identity, not a claim that
its floating-point logarithms/sum are exact.

If a positive y/r ratio overflows, its large logarithm is formed as ln(y)-ln(r).
The omitted log1p(r/y) correction is below binary64 resolution there. Shared
checked log1p owns the ordinary finite-ratio path.

## Remainder metadata is not an accuracy certificate

For a positive real argument, the first omitted Stirling term bounds the
real-arithmetic truncation error and determines its sign; see
[NIST DLMF 5.11(ii)](https://dlmf.nist.gov/5.11.ii).
The next term here is -174611/(125400*z¹⁹).

The returned remainder_estimate is only a rounded evaluation of the relevant
sum of omitted-term magnitudes. It does not include logarithm, gamma, division,
addition or rounding errors, and is not an outward-rounded numerical bound.
Its name deliberately says estimate.

If any required estimate underflows to zero, remainder_estimate_available is
false and the numeric field is an unavailable zero placeholder. That must not
be read as zero error. For branches using no Stirling terms, a zero estimate
with availability true means zero Stirling truncation, not exact total
floating-point evaluation.

The nine-term truncation and returned estimate do not establish a global ULP,
correct-rounding, uniform relative-error, profile-likelihood or calibration
guarantee.

## Independent availability

Main status: 0 log probability available, 1 negative count, 2 invalid mean,
3 invalid alpha, 4 count above 2^53, 5 arithmetic unavailable.
Mean must be finite and positive; alpha finite and nonnegative. All accepted
integer counts convert exactly.

A positive count with a nonzero alpha whose reciprocal is not finite returns
normalizer status5. It is not silently replaced by the alpha0 model.
Count0 has normalizer exactly0 without needing that reciprocal.

normalizer_status uses the saturated-normalizer statuses (0 available, 5
arithmetic for validated inputs, 2 not computed on main validation failure).
Its payload may remain available when the full log probability is unavailable.
residual_status mirrors the checked deviance primitive; read residual only
when that status is 0. deviance_status is 0 available, 1 squared-deviance
overflow/underflow, or 2 not computed. Every unavailable number is a zero
placeholder, not a valid log probability or deviance.

When squared D overflows but its signed square root is finite, the implementation
restores half-D as (abs(residual)*0.5)*abs(residual), rather than demanding that D
itself be available. This can preserve a finite log probability. A nonzero
residual whose half-D rounds away is not allowed to produce a successful zero
log probability from a zero saturated normalizer.

assumptions_assessed is always false. poisson_limit records alpha==0 on
validated evaluation; stirling_used records the selected normalizer algorithm.

## References and validation boundary

The independent C++23 reference evaluates the full log-gamma formula directly.
It includes binary128 lgammaq diagnostics where cancellation is tolerable and
400-decimal Boost.Multiprecision log-gamma for extreme r. Binary128 alone cannot
resolve lgamma(r+y)-lgamma(r) for r around 1e300 and small y. Such output is never
promoted to the canonical reference.

Independent analytic checks include count-zero Poisson and r1 geometric
likelihoods. Direct high-precision Stirling next-term checks cover z8,16,1e6;
the reference does not falsely claim it can resolve C9(1e300) by subtracting
400-digit log-gamma values of size about 1e303.

The native fixture has a loop-generated grid across counts0,1,7,8,1e6,2^53;
Poisson and r near1e-308,1,8,1e6,1e300,0.5; both sides of the r8 branch;
saturated/near-mean/tiny/huge means. It also exercises validation, inverse-alpha
unavailability and log-probability availability when squared D overflows.
The C++ reference can compare actual native scalar report values, with declared
relative tolerance and a subnormal quantum allowance.

Compilation, actual execution, scalar concordance, theoretical truncation,
floating-point accuracy and inferential calibration are separate claims.
Only the coordinator's current-source central Slurm receipts establish the
execution state. Alpha profiling remains a later integration step.
