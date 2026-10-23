# Median-log-ratio count size factors

Implementation: `genomics::normalization`.

## API and layout

`median_ratio_size_factors(counts:&[i64;2048],genes:i64,samples:i64)->SizeFactorResult`
uses physical row stride8, genes1..256 and samples1..8. Active cells hold
nonnegative integer raw counts. Inactive cells are never read. It validates
every active count before excluding any gene.

Result fields are status, genes, samples, genes_used, genes_excluded,
factors:[f64;8], log_factors:[f64;8], composition_assumptions_assessed:bool.
The last field is always false. Inactive factor slots are zero.

## Estimator

Any gene with a zero in any active sample is excluded. For each remaining
gene anchor at its sample0 positive count and compute mean log ratios to that
anchor across samples. Subtracting this relative mean is algebraically identical
to subtraction of the unanchored mean log count. For each sample take the
median of log(count) minus that gene's mean log count. An even median is the
midpoint of the two central LOG values. Then subtract the mean of the
sample medians and exponentiate. This explicitly chooses factors with geometric
mean one, within floating-point rounding. Normalized count means count/factor.

This is a median-log-ratio estimator with an explicit centering convention.
DESeq2's implementation also evaluates a median in log space. Its default
internal-reference path does not universally perform our final centering;
this API does not claim identical DESeq2 outputs.
Primary implementation: [DESeq2 R/core.R, estimateSizeFactorsForMatrix](https://github.com/thelovelab/DESeq2/blob/devel/R/core.R).

No pseudocount or alternative positive-count estimator is inserted when
the eligible set is empty. A single active sample has factors[0]=1 when
at least one count is positive.

## Status

| status | meaning |
|---|---|
| 0 | available |
| 1 | invalid genes or samples |
| 2 | negative active count |
| 3 | no gene positive in every active sample |
| 4 | detected arithmetic or sort failure |

Precedence: dimensions, all active negative-count validation, eligibility,
arithmetic. On every failure, requested genes/samples are preserved; all
other numerical payload including used/excluded counts is zero.

## Normalized-count evaluation

`size_factor_normalized_count(count:i64,factors:&SizeFactorResult,sample:i64)->NormalizedCountResult`
returns separate fields status:i64 and value:f64. Sample is zero-based.

| status | meaning |
|---|---|
| 0 | value=count/selected factor |
| 1 | failed result or invalid dimension/used/excluded metadata |
| 2 | sample outside active prefix |
| 3 | negative count |
| 4 | selected factor nonfinite/nonpositive, or nonfinite/underflowed positive result |

Checks follow that precedence; any error has value0. This validates the selected
factor and header, not arbitrary external mutation of every factor/log slot.
Count zero is valid and evaluates to zero with a valid factor.

## Numerical and storage contract

Uses canonical checked natural logarithm and exp, compensated log means,
and the shared deterministic stats::order::sort256. For close positive counts, subtract the integer anchor BEFORE binary64 conversion;
the signed i64 difference is safe because both operands are positive. If the
relative difference has magnitude at most0.5, evaluate log1p using32 artanh
terms; otherwise subtract canonical checked logs. This retains close-i64 log
contrasts even when both count conversions round to the same f64. The gene
relative-mean buffer uses1e300 as an exclusion sentinel: valid means have magnitude
below44. No third buffer is needed. Conversion of the anchor, far-count logarithms,
and normalized-count evaluation is still rounded, especially above2^53.
This is not an exact-integer-log certificate or a uniform relative-accuracy
promise for every contrast. Factors can round to1 while log_factors retain
nonzero tiny contrasts.
Arithmetic guards prevent publishing nonfinite or nonpositive factors.

Scratch consists of two256-element f64 arrays (gene log means and one sample's
ratios), reused across samples, plus the result. No dense floating count matrix,
sample history, or heap allocation. Time O(G*S*log G), with at most8 samples.

## Scientific interpretation and limits

The reference gene population must provide a scientifically defensible center
of relative abundance changes. Global biological shifts or a composition dominated
by directionally changing genes can bias the factors. The API does not assess
those assumptions. Comparable gene definitions and raw counts are caller inputs.
Sparse single-cell matrices can have no eligible genes.

This operation estimates relative scaling only: it does not estimate dispersion,
test differential expression, correct gene length, estimate read assignment,
supply count likelihood inference, or validate a variance model.

## Reference and execution boundary

tests/reference/genomics_size_factors_reference.cpp is an independent C++23
quad113-bit reference using raw integer logarithms, ordinary summation, and
std::sort (compile with -lquadmath). It emits the33-case run-pass reference fixture. The status fixture
separately checks dimension precedence, negative counts in excluded rows,
empty references, selected-factor mutation, and normalized-count errors.
The reference includes a3-sample even-median fault discriminator and checks
nonzero near-i64MAX log contrasts with relative tolerance on those contrasts,
without an absolute tolerance that could admit zero. C++ execution belongs on t560. Sounio compilation/execution belongs to Forge;
source preparation or C++ success alone does not establish native Sounio acceptance.
