# Dynamic multiple-testing adjustment

This module adjusts a declared complete family of marginal p-values. It does
not calculate p-values, assess their validity, estimate local false discovery
rates, or certify a model. The output is an **adjusted p-value**, not Storey's
estimated q-value.

## Public API

Import stats::dynamic_multiple_testing.

    dynamic_adjust_pvalues(p, available, family_size, method,
                           output, indices, stage)
    dynamic_adjust_by(p, available, family_size, output, indices, stage)

p, output and stage are NumericBuffer; available and indices are IntegerBuffer.
The second function selects BY by default.

| method | Procedure | Dependence requirement |
|---|---|---|
| 0 | Bonferroni | Arbitrary dependence, with valid marginal null p-values |
| 1 | Benjamini–Hochberg | Independence or the relevant PRDS condition |
| 2 | Benjamini–Yekutieli | Arbitrary dependence, with valid marginal null p-values |

assumptions_assessed is always false. dependence_requirement is 1 for BH and
0 otherwise; it records the method's requirement, not a successful assumption
check. Selecting BH is an explicit method choice.

These procedure meanings and the harmonic BY correction are documented in the
[official adjustment reference](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html).
Original papers:
[BH (1995)](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) and
[BY (2001)](https://doi.org/10.1214/aos/1013699998).
No R implementation is executed or used as a numerical oracle.

## Complete-family policy

family_size must equal both input lengths and the output length. Each
availability flag must be exactly 0 or 1. A flag of 1 requires finite p in [0,1].
For a flag of 0, the numeric placeholder is ignored even if it lies outside
that domain. The hypothesis remains in the family as effective p=1 and receives
output 1. It is never silently dropped, so unavailable fits cannot reduce the
multiplicity factor. The caller must define the family before interpreting
adjusted results.

## Algorithm and arithmetic

Indices are sorted by effective p, with original index breaking exact ties.
An in-place index heapsort uses O(n log n) comparisons and no internal allocation.
For sorted rank r in 1..n:

- Bonferroni: min(1,n*p).
- BH: min(1,min over later ranks of n*p_(r)/r).
- BY: the same running minimum multiplied by H_n=sum(1/r).

H_n is accumulated with compensation. Rank conversion is exact for the declared
n<=2^53 bound. The implementation computes the correction factor before
multiplying p to avoid prematurely reducing tiny probabilities. No positive
floor or epsilon repair is applied. Binary64 arithmetic is not a correct-rounding
or global relative-error certificate.

Empty families succeed without element access; harmonic_factor is 1 by identity
convention. For nonempty families harmonic_factor is H_n for BY and 1 otherwise.
The returned available_count counts flags equal to 1.

## Ownership, workspace and failure

All owners are caller allocated and remain borrowed. There are no internal
allocations. p and available contain n elements each; output contains n;
indices and stage must contain at least n. The required payload is five
8-byte arrays, or 40n bytes, excluding allocator metadata and process RSS.
No fixed 100/256/1024 family cap is introduced.

| status | Meaning |
|---|---|
| 0 | Adjustment available |
| 1 | Invalid family, shape, construction status or workspace |
| 2 | Availability flag is not 0/1 |
| 3 | Nonfinite available p-value |
| 4 | Available p-value outside [0,1] |
| 5 | Unknown method |
| 6 | Arithmetic unavailable |

Validation occurs before workspace sorting. All adjusted values are staged
before output is committed. Any failure preserves the original output; scratch
can change following successful validation. The result retains the requested
family and method; error summary values are otherwise unavailable placeholders.

## Evidence boundary

The companion C++23 reference uses exact rational arithmetic, independently
computes each candidate rank by all-pairs comparison, and takes minima by direct
enumeration rather than mirroring production heapsort/backward traversal.
It can compare actual native fixture report rows. Tests include unavailable
hypotheses, ties, zero/one p-values, method differences, subnormal probabilities,
transactional failures and a family of 513 entries.

Only the coordinator's central Slurm receipts establish current-source execution.
This module does not supply new marginal inference, a complete genomics pipeline,
calibration of arbitrary supplied p-values, or a performance superiority claim.
