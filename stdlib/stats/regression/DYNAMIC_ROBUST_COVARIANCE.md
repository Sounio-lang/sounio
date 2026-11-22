# Dynamic OLS HC3 and one-way clustered covariance

`dynamic_linear_hc3` and `dynamic_linear_cluster` accept borrowed numeric owners
and a `NumericMatrix` layout; n>p>=1 is required. No intercept is implicit.
This extends the bounded HC3/cluster component without imposing n256/p8 caps.
It computes covariance and standard errors, not p values, confidence intervals,
calibrated degrees of freedom, a Wald test or finite-sample validity.
`assumptions_assessed` and `finite_sample_calibrated` remain false.

HC3 uses the checked QR point estimate and influence vectors
`(X'X)^-1 x_i * residual_i/(1-h_i)`, accumulated as a Gram matrix. Production
obtains these vectors by triangular solves, not explicit normal equations.
The residual is raw y minus the compensated raw design/coefficient dot product.
This recomputation can differ by rounding from the QR fit's normalized residual
norm. Nonfinite or lost nonzero products are conservatively unavailable.
Computed h>=1 makes HC3 unavailable; no epsilon clipping or zero-residual
exception hides a unit-leverage singularity.

Cluster method0 is CR0. Method1 is CR1S with full multiplier
`G/(G-1)*(n-1)/(n-p)`. It is not the different G/(G-1)-only correction sometimes
called CR1. Row influences are summed within exact-equality signed-i64 labels
before their Gram matrix; grouping order is found by heapsort. Labels may be
negative or noncontiguous. G>=2 is required. G<=p and a singular covariance are
allowed; no general multivariate test is returned. CR1S scales group influences
before Gram formation so an avoidable early covariance underflow is not imposed.
Cluster covariance does not divide by 1-h or require h<1.

## Work and failure

Use `dynamic_robust_workspace(n,p)` and separate beta[p], covariance[p*p], SE[p],
leverage[n], numeric work[query.numeric_elements] and integer order[n] owners.
No owner is allocated in the kernel. Query byte arithmetic is guarded and does
not represent measured RSS. Workspace is reused; cluster scores need O(np)
storage, not a history of fits. The implementation obtains QR factor offsets from the public workspace-layout
descriptor and reads strided design entries through checked scalar access.

The beta point-fit owner commits on successful QR. Covariance, SE and leverage
commit together only when covariance_status0. A covariance failure leaves all
three caller outputs byte-for-byte unchanged, including shape failures after a
successful point fit. Scratch/order contents are never public results.

Covariance statuses:0 available;1 point fit failed;2 dimensions/owners/workspace;
3 computed unit leverage (HC3) or fewer than two clusters;4 arithmetic or
representability;5 invalid method. n<=p is rejected by the workspace/entry
precondition, not represented as an available fit with unavailable residual df.
The embedded fit status remains distinct. Invalid cluster method is rejected
before fitting. Method tags in the result are0 HC3,1 CR0,2 CR1S; cluster wrapper
inputs are0 CR0 and1 CR1S. Correction factor is0 until covariance succeeds.

Products/normalizations/triangular quotients that lose a nonzero contribution
may return status4 even when a final mathematical covariance could be finite.
Positive SE with an unrepresentable diagonal is unavailable. Offdiagonal
cancellation has no relative-accuracy guarantee. A zero-response fit may have
exact zero covariance/SE when all other conditions hold. Model misspecification,
cluster independence, cluster-count adequacy and scientific design validity are
not assessed by computational availability.

## Independent reference and native target

`deep_stats_dynamic_robust_reference.cpp` uses C++23 binary128 closed-form
one-hot-design group means, leverage1/43 and raw influence sums. It forms group
scores with an independent ordered map and no production QR or grouping helper.
It covers n516/p12, HC3, CR0, CR1S, mixed column scales1e±75, original-column
permutation, row reversal, singleton-cluster HC0 identity, zero response and a padded row-strided view
with a nonzero base offset.
Native covariance/SE and coefficients are checked, with relative tolerances for
nonzero quantities and SE-product-scaled absolute tolerance for true zero
cross-covariances. The complete native marker is mandatory; prefix output is
not acceptance. Grouped offdiagonals, CR1S correction and the HC3/HC0 difference
are explicit discriminators.

```
g++ -std=c++23 -O2 tests/reference/deep_stats_dynamic_robust_reference.cpp -lquadmath -o dynamic_robust_reference
dynamic_robust_reference actual-native.stdout
```

The separate status fixture checks unit leverage, one cluster, bad method,
output shape, rank and covariance overflow/underflow, with unchanged-output
sentinels. Build, native execution and concordance are central Slurm tasks.
Prepared source is not a claim of execution, full-corpus success, calibration,
performance superiority or publication acceptance.

## Computational contrasts and independent availability

`dynamic_robust_contrast(result,contrast,beta,work)` consumes retained influence
rows from the last successful covariance call. It computes c'beta with
compensation, then the norm of the per-row/per-cluster compensated dot products
c'influence. It does not subtract entries of c'Cov*c. CR1S influences already
contain their correction; it is not applied twice. A zero contrast is valid.
No owner or row history is allocated by this operation.

`DynamicRobustContrast` status0 means estimate and standard_error are available.
Its separate variance_status0 means the square is available;1 means the SE is
available but its square overflows/underflows;2 means not computed. Status1 is
an unavailable fit/covariance;2 invalid shape/method/metadata;3 nonfinite public
coordinates;4 arithmetic, including a lost nonzero product. Failure numeric
fields are zero placeholders. There are no p values, intervals or inferred
calibration; both scientific-assessment flags remain false.

The result/work/beta association is a caller contract, not authentication.
Only use the most recent covariance_status0 with its same unmodified owners.
Any new covariance call invalidates the previous association, even if it fails.
Finite/shape checks cannot detect deliberately substituted or stale buffers.
Roundoff near cancellation in an estimate or projected influence has no relative
accuracy certificate; conservative product-loss guards are not a proof of exact
arithmetic. Whole-matrix availability remains required by this first API.

The separate contrast fixture and binary128 direct-influence oracle cover
n516/p12 HC3 and CR0 contrasts, zero contrast, bad shape, rank/covariance failure,
SE available with overflowing variance and a four-coordinate quarter-quantum
product-loss witness. The oracle requires all eight native cases and completion;
none of these computational checks certifies statistical coverage.
