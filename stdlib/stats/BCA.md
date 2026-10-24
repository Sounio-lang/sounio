# Checked BCa bootstrap for the sample mean

## API, availability and scope

`stats::bootstrap::bootstrap_bca_mean(&x,n,b,conf,seed)` borrows a 256-value
array, with n=2..256 and b=2..1024. Confidence must be finite and strictly
between zero and one; its computed half-tail must be at least 1e-6 for the
bounded normal inverse. Only the first n values are read.

BcaResult contains mean, SE, lo/hi, b, status, normalized seed, bias_p, z0,
acceleration, lower_p/upper_p, less_count/equal_count and tail_resolved.
All numerical payloads are unavailable zero sentinels when status is nonzero.

| Status | Meaning |
|---|---|
| 0 | Computation available under this numerical contract |
| 1 | Invalid size or resampling budget |
| 2 | Invalid confidence |
| 3 | Nonfinite input data |
| 4 | Nonfinite final estimate, interval or SE |
| 5 | Degenerate observations or bootstrap distribution |
| 6 | All finite-resample bias mass is on one side: z0 not estimable |
| 7 | Unsupported normal tail, BCa pole, or invalid adjusted probability ordering |

The routine does not silently clamp an extreme bias probability, replace an
undefined acceleration by zero, substitute a percentile interval on failure,
or certify empirical coverage. Constant data have unavailable BCa correction
(status 5); the ordinary percentile API still returns its degenerate interval.

## Shared draw/statistics kernel

Ordinary percentile and BCa now use one draw, variance, heapsort and empirical
quantile implementation. The percentile path retains its former normalized
arithmetic and RNG sequence except for the repaired invalid zero-state seed
class; all previous numerical/native fixtures are retained. Both paths reduce
the signed seed modulo 2147483647, add 2147483646 if nonpositive, then map any
remaining nonpositive result to 2147483646. Thus -2147483646 and -4294967293
reproduce positive seed2147483646 without entering an absorbing zero state.

For BCa, anchor at x[0] and normalize by the largest absolute observation.
Use (v-anchor)/scale when the subtraction is finite, otherwise
v/scale-anchor/scale. Compute the point and resample means with compensated sums.
SE is the sample SD of resample means using normalized Welford updates.
Restoration to raw units avoids intermediate overflow when a finite final
quantity is representable.

The integer MINSTD stream and rejection index mapping follow RESAMPLING.md.
The deterministic 31-bit linear generator retains its finite-period and
serial-dependence limitations; this is not a generator-quality certification.

## Bias, acceleration and empirical quantiles

Tie policy: compare each normalized bootstrap mean to the observed mean using
tolerance 64*epsilon_f64*max_abs_centered_observation.
less_count excludes that tie band; equal_count is inside it.
bias_p=(less_count+equal_count/2)/b, followed by z0=Phi_inverse(bias_p).
This documented midrank convention avoids imposing a direction on tied mass.
It differs intentionally from the former strict-less count and artificial
half-replicate clipping. Different tie/quantile conventions can differ on
discrete data without establishing an error in either convention.

For the sample mean, compute the jackknife acceleration through its centered
third/second-moment identity. Rescale deviations by their maximum magnitude
before accumulating compensated second and third powers. This avoids raw
variance cubing, absolute-unit cutoffs and an n-value jackknife buffer.
The independent C++23 reference explicitly recomputes each leave-one-out mean.

Adjusted endpoints use
Phi(z0 + (z0 + z_alpha)/(1-a*(z0+z_alpha))).
Reject nonpositive denominators or adjusted probabilities outside the bounded,
finite, ordered domain. Interpolate the sorted bootstrap means at the adjusted
probabilities with type-7 empirical quantiles, h=(b-1)*p.

tail_resolved means only that b*lower_p >= 1 and b*(1-upper_p) >= 1.
This is a minimum empirical tail-support diagnostic, not sufficient tail
precision, an effective-sample-size estimate or a coverage guarantee.
A result can be numerically available while this flag is false, as demonstrated
by the eight-replicate reference case.

## Bounded normal helpers

`stats::bca_normal` computes the central normal integral by a positive-term
series of the form discussed by Marsaglia (2004), with at most 256 terms and
finite |z|<=6. Its inverse uses 56 bracketed bisection steps on [-6,6],
accepting probabilities 1e-6..1-1e-6. Outside-domain and nonfinite values
return explicit status. No unsupported native extern erfc/fabs is called.

Native checks compare 25 CDF points against long-double erfc with absolute
tolerance 2e-14, and 11 inverse points with absolute tolerance 3e-9.
These sampled checks are not a uniform error proof over every binary64 input,
nor do they inherit a published C implementation's accuracy bound automatically.

## Resources and legacy adapter

One 1024-double distribution buffer, in-place heapsort, no heap allocation,
no per-replicate sample array and no jackknife buffer. Source work
O(n*b + b*log(b) + n), fixed distribution capacity. No measured stack/RSS or
speedup is claimed. The compatibility adapter makes one 100-to-256 data copy.

`stats::epistemic::bootstrap::bootstrap_ci_bca_mean(data,n,b,alpha,seed)`
delegates to the checked kernel with conf=1-alpha; it retains n<=100 and b<=1000.
Its three-field result has no availability field: invalid/unavailable output
is the ambiguous triple of zero sentinels. New callers must use BcaResult.
Only this mean-BCa path and necessary legacy compilation repairs are certified;
the other legacy variance/correlation/regression and percentile routines are
not validated by this checkpoint. This is not BCa for Spearman correlation.

BCa targets ordinary iid sampling for this statistic; dependence, clustering,
selection, irregular estimators and asymptotic regularity are not assessed.
Finite-resample concordance is distinct from an empirical coverage study.

## Method sources

- Marsaglia (2004), Evaluating the Normal Distribution:
  https://www.jstatsoft.org/article/view/v011i04
- Efron and Narasimhan, The automatic construction of bootstrap confidence intervals:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7958418/
