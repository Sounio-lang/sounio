# Two-group median-centered Levene / Brown-Forsythe

`stats::brown_forsythe::brown_forsythe(&a,na,&b,nb)` computes one-way
ANOVA on the groups' absolute deviations from their own sample medians.
This is the median-centered Brown-Forsythe variant of Levene's test.
Two complete independent groups,2..100 finite f64 observations each.
No observations are discarded; the selected prefixes alone are read.

For group deviations z, the statistic is
F = (N-2)*SS_between/SS_within, with reference df1=1,df2=N-2.
The F survival is evaluated directly through the incomplete beta.
reference_approximate=true: applying this F reference is approximate for the
absolute-deviation sample. This is not an exact finite-sample test, posterior
probability of equal variances, a variance-ratio CI, or an automatic decision
to pool groups or choose a subsequent inferential model.
Sampling independence and distributional suitability are not assessed.

Method reference:
https://www.itl.nist.gov/div898/handbook/eda/section3/eda35a.htm

## Results and availability

| status | Meaning |
|---|---|
| 0 | Statistic and approximate F-reference p available |
| 1 | Either group size outside2..100 |
| 2 | Nonfinite selected observation |
| 3 | Zero within-group variation of absolute deviations |
| 4 | Arithmetic failure |

Inspect status before numerical payload. Unavailable F,p and scaled summaries
are zero sentinels, not inference. Requested n_a,n_b are echoed; df1=1,df2=0
when unavailable. df2=N-2 only accompanies an available result.

A constant raw group can be compared with a variable group if total
within-deviation variation is positive. If both groups have constant absolute
deviations, no p is fabricated, even if those constants differ.
Two observations in both groups always give zero within-deviation variation
and are unavailable. F=0,p=1 is available only with positive within variation.

## Numerical and storage policy

Two100-value scratch arrays are reused: raw observations -> normalized
coordinates -> absolute deviations. No additional deviation arrays or input
copies in the borrowed canonical API; the by-value legacy adapter preserves
its existing array call convention.

Each group is anchor-centered before range normalization. For an overflowing
opposite-sign range, maximum-absolute normalization precedes subtraction.
Median averaging takes place in bounded coordinates; adjacent middle raw
values need not have a representable raw midpoint.
Constant groups have zero deviation scale and do not let irrelevant locations
erase another group's small spread.
Both groups are then expressed in the same deviation_scale units, retaining
their relative spreads; normalizing each spread independently would be wrong.
Means and within sums of squares are compensated; between sum uses a
two-group contrast. There is no absolute-unit variance floor.
Information rounded away by the caller or finite arithmetic cannot be recovered.

mean_abs_a_scaled/mean_abs_b_scaled are **mean absolute deviations divided by
deviation_scale**, not median absolute deviation, sample variances or interval
bounds. Multiplication by deviation_scale recovers mean absolute deviation in
input units when representable. Subnormal reconstructions are quantized.

## Shared helpers and distribution repair

stats/order.sio provides finite-prefix checked in-place heapsort sort100.
Invalid size/nonfinite data returns false before mutation. Empty prefixes
succeed. Shapiro-Wilk and private legacy ordering delegate to this same helper.
This removes duplicate ordering kernels; it does not validate legacy AD/KS.

fisher_f's ff_exp and special/beta's bt_exp now delegate to math::pure::exp.
The former reciprocal positive exponential saturated at1e300, causing a
spurious1e-300-scale floor in extreme small tails. Direct shared exp preserves
representable subnormals and underflows naturally when appropriate.
This focused acceptance does not certify every beta/F inverse or parameter
domain. Regression tests cover F(1,df) with df2..198 and the selected beta case.

Two scratch arrays, shared sorting and exp are source-level resource properties,
not measured RSS, stack, wall-clock speedup or binary-size reductions.

## Compatibility and visualization

The public by-value `stats::epistemic::assumptions::levene_test(a,na,b,nb)`
delegates here and returns BrownForsytheResult. The former private Beta-return
procedure, p buckets and absolute within-variance floor are removed.
No external former private callers were found; the in-file smoke is updated.
Unvalidated AD/KS and their remaining p-to-Beta path are still separate work.

The synthetic A=5,B=6 Canvas/SVG consumer plots actual normalized absolute
deviations and their checked means in a shared domain. F/p labels are computed.
Caption says approximate F reference; footer distinguishes these profiles
from variance estimates or CIs. Lines connect sorted observation ranks as a
visual profile, not paired observations or a density. No real-domain dataset
or empirical effect conclusion is implied.

## Reference and scope

C++23 long-double raw median/deviation ANOVA is independent of the candidate
normalization. Tail probabilities use positive adaptive quadrature of a
trigonometric Student/F integral, not the candidate beta algorithm.
Six df=2 tails validate quadrature against an analytic formula.
15 datasets cover signed unit changes, independent location shifts, an
unrepresentable raw median midpoint, opposite-sign extremes, subnormals,
equal groups,one constant group,unequal sizes and2x100 groups.
36 F-tail points cover df2,4,9,10,98,198 and F through1e30, including a
positive binary64 subnormal near7.239075918e-317 and actual underflow to zero.
A direct beta underflow case also discriminates the prior reciprocal floor.
Subnormal comparisons allow two minimum binary64 steps of quantization;
normal nonzero probabilities are checked relatively.

Native identity, control policies, figure inspection and precise acceptance
are in docs/plans/STATS_BROWN_FORSYTHE_EXECUTION_2026_10_18.md.
Only two groups and median centering are implemented here. General k-group,
mean/trimmed centering and coverage/power assessment are not certified.
