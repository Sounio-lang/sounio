# Dynamic conditional Fisher covariance and contrasts for count models

`dynamic_count_fit_fisher` calls the checked dynamic fixed-alpha Poisson/NB2 fit, then computes the inverse expected coefficient information at its fitted means. The log-link design has no implicit intercept. The supplied alpha is held fixed during this computation:

I_beta = X' diag(mu_i/(1+alpha*mu_i)) X,   C = inverse(I_beta).

For alpha=0 this is the Poisson expected information. There is no multiplication by a residual mean square or estimated Gaussian noise scale, and no Student-t degrees-of-freedom correction. C is a numerical inverse-information quantity. Its existence does not prove that it equals the finite-sample sampling covariance of the estimator.

The NB2 variance convention and corresponding expected coefficient weight are described in [Love, Huber and Anders (2014), equation7](https://link.springer.com/article/10.1186/s13059-014-0550-8). This is support for the statistical parameterization, not independent verification of the implementation.

## Fit and availability contracts

Inputs and point-fit controls are those of `dynamic_count_fit`. The wrapper first fits these exact design/count/offset inputs, ensuring the coefficient and mean buffers are produced by that fit. It returns that `DynamicCountFit` as `fit`, with a separate covariance status:

| covariance_status | Meaning |
|---|---|
| 0 | Covariance matrix and standard errors available |
| 1 | Point fit failed; inspect fit.status |
| 2 | Covariance owner, workspace, shape or byte-budget invalid |
| 3 | Expected-information QR factorization failed; inspect factor_status |
| 4 | Arithmetic or conservative representability check failed |

Point coefficient/mean/predictor buffers follow the underlying fit's transactional contract. A successful point fit remains available if covariance fails. Covariance and standard-error output buffers remain unchanged unless covariance_status==0. Factor and staging owners are scratch and may be partially overwritten on any call.

Covariance output has p² entries in contiguous row-major order. Standard-error output has p entries in original design-column order. The matrix is filled symmetrically from a compensated Gram calculation; its floating-point entries are not a formal positive-semidefiniteness certificate.

Metadata `alpha_plug_in` is explicitly supplied by the caller and `alpha_fixed` is its negation. A plug-in alpha from a dispersion fit yields a covariance **conditional on that alpha**, without propagation of dispersion-estimation uncertainty. The flag does not alter the numerical calculation, and is not an automated provenance check.

`model_declared` and `independence_declared` record the caller's scientific declarations. Computational covariance can be available even when those declarations are false. `assumptions_assessed` and `finite_sample_calibrated` remain false in all results. The API produces no p-values, confidence intervals, inference-eligibility decision, cluster correction or empirical calibration.

## QR organization and representability

After the point fit, original column maxima c_j and max(W) are computed. The weighted normalized design is

B_ij = (X_ij/c_j) * sqrt(W_i)/sqrt(max(W)).

The public checked QR factorization returns B[:,permutation]=Q R diag(inner_scales[permutation]). Each column of inverse(R) is obtained by an independent back substitution. Its row j is restored to the original coefficient coordinate by division by sqrt(max(W))*inner_scale_j*c_j, using balanced mantissa/exponent products rather than overflowing products of scales. Covariance is the restored inverse-factor Gram matrix.

No X'WX matrix is formed or inverted. Every public factorization must succeed under the declared rank threshold; no rank-deficient covariance or generalized inverse is silently substituted. Nonzero coordinates lost to column or row normalization, triangular division or influence restoration return status4. Nonfinite products/sums and nonzero products lost to zero in triangular solves, influence Gram terms or normalized norm squares are rejected. A nonzero standard error whose covariance diagonal underflows or overflows also makes the matrix unavailable. These conservative guards may reject a mathematically representable final result. They are not a correctly-rounded or global relative-accuracy guarantee.

Signed off-diagonal sums and contrast estimates can suffer cancellation. Compensation reduces accumulation error, but does not establish relative accuracy near an exact zero. An exactly zero off-diagonal product is permitted, while a nonzero pair whose product rounds to zero is conservatively rejected. The independent zero-offdiagonal comparison uses a scale based on the corresponding standard errors, not an absolute floor unrelated to units.

## Computational contrasts

`dynamic_count_fisher_contrast` accepts a finite length-p contrast c and the latest successful Fisher result with its unchanged coefficient, R, inner-scale, outer-scale and permutation owners. It computes

estimate = c' beta,   SE = norm(inverse(transpose(R)) * permutation-transformed(c/scales_total)),

where scales_total includes sqrt(max(W)), inner column scaling and original column scaling. The triangular solve computes the norm directly; it does not subtract entries of c' C c. Coefficient products and their compensated sum are finite-checked; a nonzero coefficient/contrast product that rounds to zero returns arithmetic unavailability instead of a false available zero estimate. Nonzero scaling/triangular components that round to zero are conservatively unavailable.

A successful scalar result has status0 and separate variance_status:0 square available,1 standard error available but its square overflowed or underflowed,2 not computed. Status1 means unavailable Fisher result;2 invalid owner/length;3 nonfinite contrast/coefficient or malformed permutation;4 arithmetic. Failure numeric fields are zero placeholders. A zero contrast is valid and produces exact zero estimate, SE and variance.

**Factor validity is a caller contract.** Only consume factors with covariance_status==0 from the most recent wrapper call, using the same unmodified coefficient/factor/scale/permutation buffers. A new call invalidates the earlier factor/result association, including when the new call fails. The returned metadata is not an authenticated capability: the library cannot detect a caller deliberately combining an old result with changed raw buffers. Shape, finite values and permutation checks do not prove source identity. Do not consume stale buffers or a failure result.

## Caller workspace and resource limits

The implementation allocates no buffers. Use `dynamic_count_fisher_workspace(n,p)` and create distinct owners before calling. Beyond the original inputs and final coefficient/mean/predictor/covariance/SE outputs, numeric scratch totals

2np + 6n + 11p + 3p² doubles,

with p additional integer permutation entries. This includes the count-GLM workspaces, R, inverse-factor influences, covariance staging, inner and outer scales, and triangular solve. The p-entry count-step owner is reused to stage standard errors after fitting. Query payload_bytes includes all numeric/integer scratch, excludes allocator metadata and is not a measured RSS estimate. Product/sum and signed-byte-budget checks precede allocation. No fixed n256/p8 ceiling is imposed.

Covariance requires p² output and scratch; callers needing only a scalar contrast still use this first API's covariance-validity gate. Some scalar contrasts could mathematically remain representable when the full matrix is unavailable; this implementation does not claim to expose every such case.

## Independent reference and validation targets

`tests/reference/deep_stats_count_fisher_reference.cpp` is a C++23 binary128 closed-form oracle for independent group means. For a group g with n_g observations and mean mu_g, the inverse-information entry for its log mean is (1+alpha*mu_g)/(n_g*mu_g). Reparameterizing two log means into intercept and group difference gives the nonzero negative covariance explicitly. These are exact matrix identities at the declared means, not claims about exact finite-sample estimator variance.

The source fixture `stdlib_deep_stats_count_fisher.sio` includes Poisson and NB2, explicit plug-in/declaration metadata, design-unit scaling by1e±150, two512x12 one-hot cases, original-column contrast mapping, rank/count failures, covariance-shape failure with preserved outputs, invalid contrast size, the zero contrast and an identity-design half-quantum product-loss witness whose true contrast is one minimum subnormal. The oracle compares actual printed coefficient, SE, covariance and contrast values with relative tolerance for nonzero quantities and standard-error-scaled tolerance for true zero off-diagonals. It requires the final native completion marker.

Build/execution and concordance belong to central Forge/Slurm. Source preparation or typechecking alone does not establish native numerical agreement, complete boundary coverage or inferential calibration.



## Selected native checkpoint

The coordinator reported native fixture success and actual-output closed-form C++ concordance in Slurm16130 with the corrected product-loss guards. The Fisher source was SHA256 1cc46c5987f9f55814d80e9f3a2a7a14a3f9a0c352be435b7a5ff8c396855340. Selected cases cover the six numeric models, output/status contracts and the half-quantum estimate witness. The witness returns explicit arithmetic unavailability rather than an available zero.

This records computational inverse-information/contrast agreement on those cases, not exact finite-sample estimator variance or calibrated inference. It refers to the frozen dependency snapshot; later reusable NB2-slot changes require separate execution. No full-corpus, CI/merge, uniform-error or performance claim is made.


## Consumer-slot integration checkpoint

In the integrated full2 run (Slurm16139), this module's selected native fixture and its registered actual-output C++ comparisons returned0 after reusable-slot consumer integration. The count GLM, Fisher, bootstrap, dispersion-profile and scalar-likelihood rows were successful. The overall full2 gate nevertheless returned1 because the separate NB2-slot fixture and its comparison failed; a passing subset is not a passing complete gate.

The coordinator corrected that fixture contract and the focused slots_final rerun (Slurm16144) returned0 for its native slot program and independent analytic comparison. The retained million-draw PTRS reference/control also reported success with exact state/counter checks. These are compatibility and selected correctness observations, not a measured speed/RSS improvement, a universal error bound or a finite-sample calibration result. The final exact-index integration/CI gate remains separate.

Fisher continues to use the point-fit-first wrapper and unchanged QR/contrast calculations; the underlying CountGLM consumer now reuses its private deviance slot. No inference layer or new calibration claim is introduced.
