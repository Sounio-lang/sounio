# Two-sample permutation test of the mean difference

## Checked API

`permutation_means(&x, nx, &y, ny, iters, seed, exact)` borrows two
256-value arrays. Both groups must be nonempty and nx+ny <= 256. Only active
entries are read and caller arrays remain unchanged.

The signed statistic is mean(x)-mean(y); the tail comparison is two-sided
absolute magnitude, including ties. All returned estimates require status=0.
Status 1: invalid sizes; 2: invalid Monte Carlo budget; 3: nonfinite data;
4: exact enumeration exceeds capacity; 5: nonfinite observed difference or
enumeration arithmetic failure. Failed results contain zero sentinels.
A nonrepresentable raw observed difference makes the whole result unavailable,
even if a normalized tail probability could otherwise be computed.

Result fields: status, method, observed_diff, p_value, hits, draws, seed_used,
mc_available, mc_se, mc_lo, mc_hi, exchangeability_assessed.
Method 1 is Monte Carlo; method 2 is exhaustive label enumeration.

## Exhaustive label enumeration

With exact=true, seed and iters are ignored. Enumerate each choice of nx labeled
observations from the pooled sample once, retaining multiplicity for duplicate
values. Capacity is choose(nx+ny,nx) <= 65536. A bounded multiplicative binomial
recurrence avoids factorial overflow and refuses larger requests; no silent
switch to Monte Carlo. p_value=hits/draws, without an add-one correction.
mc_available=false; mc_se/mc_lo/mc_hi are zero unavailable sentinels.

"Exact" means all admissible labelings were visited. Statistics and comparisons
still use binary64 arithmetic and the documented tie convention. It is not an
exact rational-arithmetic or unconditional finite-sample coverage claim.

## Monte Carlo

With exact=false, require 1..100000 draws; no silent cap. Reset the index vector
each draw, select the first nx labels using partial Fisher-Yates, then accumulate
the two groups independently. Labelings are sampled with replacement between
draws. Return p_value=(hits+1)/(draws+1); the raw fraction q=hits/draws is a
separate quantity. A zero-hit run does not produce a zero p-value.

MINSTD integer recurrence: state=(16807*state)%2147483647. Normalize signed seed
by remainder modulo 2147483647; nonpositive remainders add 2147483646.
If the result remains nonpositive, map it to 2147483646, repairing the negative
remainder class that previously reached an absorbing zero state. Seeds
-2147483646 and -4294967293 match positive seed2147483646; other normalized
streams are unchanged. No negation of i64 minimum. Draw state-1 and reject it above the largest multiple
of the current selection width before modulo reduction. seed_used records the
normalized initial state. This removes the extra index-reduction bias relative
to a uniform generator state; the generator itself remains a deterministic
31-bit linear sequence with serial dependence and finite period.

mc_available=true. mc_se=sqrt(q*(1-q)/draws) is a plug-in Monte Carlo diagnostic.
mc_lo/mc_hi form the nominal 95% Wilson score interval for the uncorrected
exceedance probability q under an ideal independent-uniform sampling model.
It is not an effect interval, a posterior probability of the null, or a
guaranteed-coverage interval for the deterministic PRNG. At hits=0 or hits=draws,
mc_se is zero while the Wilson interval still has positive width.
Do not make an adaptive stopping or multiple-testing guarantee from these fields.

## Numerical and resource contract

Anchor on the first observation and normalize by the largest absolute input.
Compute (v-anchor)/scale when the subtraction is finite, otherwise
v/scale-anchor/scale. Separate Kahan sums for each group avoid the loss from
subtracting a small subset from the pooled total. The observed signed difference
is transformed back to input units.

Count a tail hit when |difference*| >= |observed|*(1-64*epsilon_f64).
This relative tolerance is independent of physical units; it replaces the former
fixed raw-unit tolerance of 1e-12. It is a floating-point convention, not a
guarantee that all mathematical ties or arbitrarily close alternatives are
distinguished. Underflow, tiny contributions and extreme dynamic range remain
limited by binary64. Complete arbitrary-precision comparisons are not provided.

One normalized 256-double buffer and one 256-index vector; no heap allocation,
stored resampling distribution, factorial table or recursive enumeration.
The selected prefix also stores the enumeration cursor, eliminating a second
index buffer. Work is O(draws*(nx+ny)) and storage O(nx+ny) within fixed capacities.
These are source bounds, not a measured peak-stack/RSS or speedup claim.

## Statistical null and compatibility

The null requires exchangeability of the pooled observation labels (or a valid
randomization design yielding that assignment distribution). Equality of means
alone, unequal-variance populations, pairing, clustering, longitudinal dependence
or unequal sampling designs do not automatically justify this unstudentized test.
The routine never assesses those assumptions: exchangeability_assessed=false.

`perm_test_means` retains its scalar signature and delegates to Monte Carlo.
On invalid or unavailable input it returns 1.0 for compatibility. That scalar
cannot distinguish a valid p=1 from unavailable inference. New callers must use
the checked result. The existing separated/overlapping-group example is retained.

## Methods

- Phipson and Smyth (2010), random-permutation p-values and zero-p-value pitfalls:
  https://pubmed.ncbi.nlm.nih.gov/21044043/
- Wilson (1927), score interval for a binomial proportion:
  https://doi.org/10.1080/01621459.1927.10502953

The figure consumer shows q and Wilson bounds for B=256 and B=2048, with its
meaning and budgets in the axis title. It does not label these bounds as
uncertainty in an effect or in the truth of the null.
