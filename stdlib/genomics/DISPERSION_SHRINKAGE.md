# Empirical log-dispersion trend and conditional shrinkage

This component consumes supplied positive gene means, positive dispersion
estimates and positive **sampling variances of log dispersion**. It does not
estimate a count likelihood, sampling variances, a design matrix or RNA-seq
normalization, and is not a complete differential-expression pipeline.

## Declared working model

Let l_i=ln(estimated dispersion_i), with supplied sampling log-variance v_i.
The trend is t_i=ln(a0+a1/mean_i), with a0>0 and a1>0.
The caller supplies both positive initial parameters. The implementation fits
the trend by unweighted nonlinear least squares in log(a0),log(a1), using
dynamic pivoted QR for Gauss-Newton increments and monotone backtracking.
The positive log-sum is evaluated with max/log1p/exp, without forming
a1/mean or the sum of positive raw parameters.

The local working model is l_i=t_i+u_i+e_i, with independent zero-mean components
having variances tau² and supplied v_i. The variance correction uses the final
trend Jacobian J, not a declaration that the nonlinear model is exactly linear.
For full column rank two, h_i is the corresponding diagonal projection leverage:

    raw_tau² = [sum residual_i² - sum (1-h_i)*v_i] / (G-2).

The public residual_mean_square and sampling_correction divide their respective
sums by G. The implementation computes raw_tau² from those means to avoid
unnecessary overflow of sums. Nonpositive raw_tau² selects the explicit
nonnegative moment boundary tau²=0 and sets prior_boundary=true. There is no
positive arbitrary floor.

This correction is exact for the corresponding fixed linear working design
under its variance assumptions. Using a fitted nonlinear Jacobian is a
linearization; it is not a finite-sample unbiasedness or calibration guarantee.
All genes participate in trend and prior estimation. Outliers are identified
after those estimates; there is no undocumented robust refit or trimming.

## Conditional MAP and high-dispersion rule

For each gene, conditional on the fitted trend and moment prior variance,
the Gaussian working-model MAP log dispersion is

    mapped_i = [tau²*l_i + v_i*t_i] / (tau²+v_i)
    working_variance_i = tau²*v_i / (tau²+v_i).

Variance ratios and the smaller variance are used instead of forming reciprocal
precisions or an overflowing sum/product. At tau²=0 the trend is a point prior;
mapped_i=t_i and working_variance_i=0. A nonzero working variance rounding to
zero is arithmetic unavailable. A ratio of two positive variances rounding to zero\nalso returns status 4: its apparently vanished weight can yield a representable\ncontribution after multiplication by a nonzero log value. No such weight is\nsilently dropped. This is a conservative availability limit.

When l_i-t_i exceeds outlier_sigma*sqrt(tau²+v_i), the output is unshrunk:
mapped_i=l_i, working_variance_i=v_i, and its flag is 1. Other flags are 0.
The caller supplies outlier_sigma in [1,20]. This is a declared working rule,
not a calibrated outlier p-value. Returned variances ignore hyperparameter
estimation uncertainty. No posterior credible-interval calibration is claimed.

## Dynamic API, ownership and storage

Import genomics::dispersion_shrinkage. All NumericBuffer owners remain with
the caller. The kernel borrows them and performs no internal allocation.
There is no fixed 256-gene bound.

dispersion_shrinkage_workspace(G) describes five scratch owners:

| Buffer | Elements |
|---|---:|
| jacobian | 2G |
| response | G |
| increment | 2 |
| qr_workspace | 3G+8 |
| stage | 3G |
| Total scratch | 9G+10 |

Three input owners and three output owners each contain G doubles.
Thus requested payload for the full call is 15G+10 doubles, excluding allocator
metadata, stacks, compiler/renderer storage and process RSS.
G must be at least 3 and no larger than 2^53; shapes and arithmetic bounds are
checked before element access. Each owner has its own explicit construction
budget. The module does not silently increase those budgets.

The function dispersion_shrinkage_fit receives inputs means, estimates,
sampling_variances; positive a0_initial,a1_initial; rcond; max_iterations;
tolerance; outlier_sigma; three output buffers; and the five scratch buffers
in the table's order. Output buffers contain log MAP values, working variances,
and exact 0/1 flags represented as doubles. Exponentiation to raw dispersion is
a separate availability decision.

All lengths, construction statuses, controls and active input values are checked
before output publication. Scratch may change on failure. The three output
buffers retain their original contents unless every output row is available.

## Status and limits

| status | Meaning |
|---|---|
| 0 | Component available; calibration_assessed remains false |
| 1 | Invalid shape, construction status or insufficient scratch |
| 2 | Nonfinite input data |
| 3 | Nonpositive means, dispersion estimates or sampling variances |
| 4 | Arithmetic unavailable |
| 5 | Invalid controls or positive initialization |
| 6 | Trend/Jacobian rank unavailable |
| 7 | Iteration or backtracking limit |

Controls: max_iterations 1..100, tolerance 1e-12..1e-4, rcond 0 or (0,1).
Rank-deficient means/trends do not receive arbitrary fallback parameters.
Unrepresentable final positive a0/a1 are unavailable. Leverage overshoot is
rejected rather than clipped. Errors clear the summary numeric payload, retaining
the requested gene count.

This is a declared empirical component. It does not establish DESeq2 equivalence,
a complete empirical-Bayes analysis, calibrated posterior uncertainty, count-fit
convergence, differential-expression p-values, biological validity or inferential
superiority. Those require additional kernels and independent validation.

## Verification boundary

Source preparation is distinct from current-source typecheck/native evidence.
The coordinator runs compiler and numerical references through central Slurm.
The independent C++23 reference uses binary128 arithmetic, direct 2x2 solves,
and an inverse-Jacobian Gram leverage calculation instead of production QR and
its centered-column leverage identity. No Python/Rust oracle is used.

## Reusable active-prefix adapter

`dispersion_shrinkage_fit_prefix` takes the same inputs and controls as
`dispersion_shrinkage_fit`, with an explicit active count after the three input
buffers. Only the first active elements are read and published; allocated tails
may contain NaNs or sentinel values and remain untouched. The increment buffer
still has exactly two elements. Other input/output capacities and the queried
Jacobian/response/QR/stage scratch must cover the active count. The old facade
retains its exact-shape contract, including refusal of mismatched old buffers.
Both facades call the same private estimator engine and use the same arithmetic.

The named `dynamic_qr_least_squares_rhs_prefix` adapter reads only the matrix
layout's row count from a larger RHS buffer. Its private QR engine and all
ordinary `dynamic_qr_least_squares` exact-RHS semantics are shared. This avoids
allocating new owners when a batch pipeline compacts fewer numerically eligible
genes than its maximum family capacity; it does not change the working model,
introduce missing-data filtering inside the component, or certify accuracy.
