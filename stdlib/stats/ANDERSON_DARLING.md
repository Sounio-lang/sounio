# Checked fitted-normal Anderson-Darling

`anderson_darling(&[f64;256], n:i32)` estimates the normal mean and **sample**
standard deviation from the same complete sample. It is not a test against a
known fixed CDF, a two-sample AD test, or a Lilliefors/KS procedure.

## Availability

| status | Meaning |
|---|---|
| 0 | Statistic and approximate p available |
| 1 | Size outside 3..256 |
| 2 | Nonfinite active input |
| 3 | Constant active input |
| 4 | Arithmetic failure |
| 5 | A2 and adjusted A2 available; calibrated p unavailable |

Read status before payload. Status 1..4 zero numerical payload; status 5 retains
A2/A2_star and sets p=0 as an unavailable sentinel, **not a tail probability**.
Metadata n is the requested core size; p_approximate is always true.
has_ties flags exact duplicate values. Ties do not make the continuous-null
calibration valid for discrete/rounded data; no censoring/independence assessment
is performed. Large p is not a probability that normality is true.

The legacy by-value 100-element adapter is now public and returns ADResult.
It checks 3..100 before copy/narrowing and returns status 1, n=0 for invalid size.
Existing ADResult fields remain; adding metadata can require updates to external
struct literals. No other repository consumers were found by the focused search.

## Algorithm and bounds

One 256-double scratch buffer; shared in-place heapsort O(n log n), followed by
linear passes. Anchor/range normalization preserves representable small
differences at large offsets; overflowing ranges use max-absolute scaling.
Compensated mean, sum of squares, and paired log-probability sum.
No caller mutation. No dynamic allocation in the statistic.

Normal lower/upper tails use the direct upper incomplete gamma at shape 1/2;
the small tail is logged before any complement. The reverse survival sum is
reindexed to avoid a second log array. Standardizing by the same sample SD bounds
the exact standardized magnitude by (n-1)/sqrt(n), below 16 for n<=256.
Thus this use does not require extreme unbounded normal-tail extrapolation.
The complement's log may round to zero when its true magnitude is below machine
precision; the small tail remains represented and logged.

## Calibration

Astar=A2*(1+0.75/n+2.25/n^2). The four piecewise exponential approximations
documented for Stephens (1986), Table 4.9, are used only for n>=8 and Astar<10.
n=3..7 and Astar>=10 retain the statistic with status 5. No fabricated p floor
or extrapolated polynomial is returned. The approximation is neither exact
finite-sample inference nor a posterior update.

Method sources:
- [NIST AD definition and distribution-specific calibration](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35e.htm)
- [nortest author documentation of fitted-normal statistic and size domain](https://search.r-project.org/CRAN/refmans/nortest/html/ad.test.html)
- [nortest calibration domain and coefficient reference](https://github.com/cran/nortest/blob/master/R/ad.test.R)

The source was consulted as a method specification; no R program is run.
C++23 reference uses long-double raw centering and std::erfc, sorted reverse
pairing. Calibration coefficients are shared deliberately, so p concordance
does not independently validate the scientific approximation itself.
18 datasets include all four calibration pieces, sizes 3..256, ties/outliers,
large offsets, overflowing raw ranges and subnormal scale. Focused native
fixtures cover invalid sizes, NaN/Inf, constants, caller immutability and
very small p with relative bounds.

## Scientific figure

The synthetic n=20 figure shows an explicit step ECDF and a fitted-normal CDF
connected at the observed sample coordinates. Values are not a confidence band.
Mean=9.5 and sample variance=35 for the disclosed sample 0..19.
A2/p labels are obtained from the checked result, not hard-coded presentation
values. Both Canvas and SVG paths are exercised. No decimation is performed.

The former private pedagogical AD and p-to-Beta confidence conversion have been
removed from the assumptions adapter module, along with unused raw moment,
sorting and external math helpers. Other statistical procedures remain separate.
