# Dynamic full-likelihood NB2 dispersion profile

`dynamic_dispersion_fit` fits the count-model coefficients at each supplied log-dispersion candidate, then searches a caller-declared finite log-alpha bracket. It consumes the full per-count `nb2_log_probability`, including its alpha-dependent saturated normalizer. It does **not** optimize deviance alone. The design has no implicit intercept, counts are integers in 0..2^53, offsets are logarithmic, and alpha is common to all observations.

This is a bounded, reproducible numerical search with explicit availability. It does not certify a global optimum, the existence of every model MLE, calibrated posterior uncertainty, a complete RNA-seq analysis or equivalence to DESeq2. The caller supplies the design, starting coefficients, bracket, tolerance and budgets.

## Criteria and parameterization

Let t=log(alpha). Method0 maximizes the mean full log likelihood. Method1 subtracts log(det(I_beta))/2n, where I_beta=X'WX and W_i=mu_i/(1+alpha*mu_i), evaluated at the fitted means. An optional prior subtracts (t-center)^2/(2*variance*n) from either criterion. It is explicitly a Gaussian working prior **on the log-alpha coordinate**, with positive caller-supplied variance. Its mode is a log-coordinate MAP; no transformation Jacobian is silently added.

The Fisher-information Cox–Reid form and a quadratic log-dispersion prior appear in [Love, Huber and Anders (2014), equations7 and9](https://link.springer.com/article/10.1186/s13059-014-0550-8). That source supports the statistical quantities, not this implementation's accuracy or software parity. This component refits coefficients at every candidate and makes its own search and availability choices.

Without a prior, an exact alpha=0 Poisson candidate is also evaluated; exact numerical ties prefer it. With the log-coordinate prior, alpha=0 is outside its support and is not evaluated. A selected Poisson candidate is labeled a boundary candidate and left-censored relative to the positive search bracket, never a certified global boundary MLE.

## Search, failures and curvature

The algorithm evaluates nine equally spaced log-alpha grid points, selects the best positive point's neighboring interval, and applies golden-section maximization until that interval's width is at most `outer_tolerance`. Every inner count fit restarts from the declared initial coefficients. A candidate that fails likelihood, convergence, rank or arithmetic checks stops the search; it is not discarded as a missing grid point. Nine grid points cannot prove unimodality or rule out a better unsampled maximum. `global_optimality_assessed`, `inference_eligible` and `assumptions_assessed` remain false.

Each candidate costs one outer evaluation, including Poisson and optional curvature candidates. The search requires at least nine evaluations with a prior or ten without it. Search-budget exhaustion leaves all three public output buffers unchanged, even if an intermediate best candidate exists. Candidate and best buffers are scratch and may change on failure.

When requested at a selected interior point, curvature uses symmetric criterion differences at h and h/2, starting h=2^-8 and reducing it to stay inside the bracket. Four additional count fits are required. Both negative second differences must be finite and positive and agree within10% of their larger value. The two `curvature_h` fields store curvatures of the **mean criterion**, not step lengths. `log_alpha_variance` is 1/(n*curvature_half_h), restored through balanced scalar products. It is a conditional local working quantity; availability does not establish frequentist coverage or posterior calibration. No standard error, p-value or confidence interval is produced.

Insufficient curvature budget, a failed curvature fit, finite-difference cancellation or disagreement leaves the selected point available and the variance unavailable with an explicit `curvature_status`. A curvature failure does not overwrite the selected point or consume its scratch output as the final fit. The variance tolerance in the independent fixture is appropriate only to its moderate grouped design and stated difference steps.

| Main status | Meaning |
|---|---|
| 0 | Selected point and public coefficient/mean/predictor outputs available |
| 1 | Invalid shape, matrix, owner, workspace size or representability budget |
| 2 | Invalid method, bracket, prior or optimizer controls |
| 3 | Unresolved candidate failure; inspect failure_stage and inner_status |
| 4 | Search arithmetic or finite-resolution failure |
| 5 | Search evaluation budget exhausted |

`profile_status` is -1 on failure,0 interior,1 Poisson candidate,2 lower range edge,3 upper range edge,4 both edges within the declared tolerance. Edge flags denote range censoring, not inferential decisions. At success, `log_alpha_available` is false for Poisson and true for a positive selected alpha. Unavailable numeric fields are zero placeholders.

`failure_stage`:1 count fit,2 full likelihood,3 Cox–Reid factorization,4 other candidate arithmetic. An inner status is meaningful together with its stage. Curvature statuses:0 available,1 not requested,2 boundary or finite resolution,3 insufficient remaining budget,4 candidate failure,5 nonpositive curvature,6 disagreement,7 arithmetic/unrepresentable variance. Boundary curvature status is2.

## Memory and numerical organization

All storage belongs to the caller. The function performs no allocation and retains no iteration history. Use `dynamic_dispersion_workspace(n,p,method)` before constructing scratch owners. It requires n>p>=1 and inherits the checked count-GLM and matrix bounds. Owners are borrowed; distinct mutable owners are required by the API.

Beyond the final outputs, total numeric scratch is 2np+10n+10p doubles for ML, and an additional p²+p doubles for Cox–Reid. Cox–Reid also needs p integer permutation entries. These totals include the candidate and best coefficient/mean/predictor buffers, count-GLM scratch and factor outputs, and exclude the caller's input and final output owners. The query checks every product/sum against the signed byte budget before allocation. Its `payload_bytes` covers all numeric and integer scratch, not allocator overhead or measured process RSS.

Full log-probability terms are nonpositive and accumulated as a scaled compensated sum before division by n; no overflowing total likelihood is required. Cox–Reid forms B_ij=(X_ij/c_j)*sqrt(w_i)/sqrt(max(w)), where c_j=max_i|X_ij|. It calls the public checked dynamic QR factorization, then restores

logdet(I_beta) = p*log(max(w)) + 2*sum_j(log|R_jj| + log(inner_scale_j) + log(c_j)).

No normal-equation matrix is formed. Column/permutation ordering does not alter this determinant sum. Nonzero coordinates lost to normalization/scaling return arithmetic unavailability. Prior penalties and variance restoration use balanced mantissa/exponent products. These guards are conservative; success is not a correctly-rounded or global relative-error certificate.

## Independent reference and execution boundary

`tests/reference/deep_stats_dispersion_profile_reference.cpp` uses C++23 binary128, analytic means in a two-group one-hot design, finite harmonic sums for integer-count digamma differences and independent score bisection. It does not reuse production count fitting, the golden-section search, QR or the Sounio likelihood implementation. It separately checks ML, Cox–Reid, log-coordinate MAP and combined criteria. Native coefficient and criterion outputs are compared against that reference; the local curvature comparison uses an explicitly looser relative tolerance because it is a finite-difference estimate.

`tests/run-pass/stdlib_deep_stats_dispersion_profile.sio` also exercises explicit Poisson selection, both search-range edges, exact search-budget exhaustion, invalid prior variance, count-conversion failure, rank failure, nonconvergence for an all-zero one-hot design and the distinct
sufficient all-zero-count boundary diagnostic with an explicit constant column. Error cases check unchanged public output buffers. This fixture is a selected contract/concordance check, not coverage of all designs, all possible boundary MLEs or every floating-point region.

Build and execution belong to the central Forge/Slurm gate. The source and fixture preparation alone are not execution evidence. No general inferential calibration is claimed.



## Selected native checkpoint

The coordinator reported all13 selected profile fixture cases and actual-output score-bisection concordance successful in Slurm16118, and reran profile native/C++ checks successfully in16130 after the CountGLM local-gain correction. The profile source was SHA256 d3e22e2c5a4b006bcedc549d5198da8d3d5d8b8a4de0c4130772021601409bfd. This covers the stated selected ML/Cox–Reid/log-MAP, boundary, budget and transactional cases; it is not a proof of unimodality, general existence, calibrated curvature or software parity.

These executions refer to their frozen dependency snapshots. Later reusable likelihood-slot changes need separate native compatibility evidence. No full-corpus, CI/merge or comparative-performance claim follows.


## Consumer-slot integration checkpoint

In the integrated full2 run (Slurm16139), this module's selected native fixture and its registered actual-output C++ comparisons returned0 after reusable-slot consumer integration. The count GLM, Fisher, bootstrap, dispersion-profile and scalar-likelihood rows were successful. The overall full2 gate nevertheless returned1 because the separate NB2-slot fixture and its comparison failed; a passing subset is not a passing complete gate.

The coordinator corrected that fixture contract and the focused slots_final rerun (Slurm16144) returned0 for its native slot program and independent analytic comparison. The retained million-draw PTRS reference/control also reported success with exact state/counter checks. These are compatibility and selected correctness observations, not a measured speed/RSS improvement, a universal error bound or a finite-sample calibration result. The final exact-index integration/CI gate remains separate.

The consumer now creates one Nb2LikelihoodSlot before the grid and reuses it for every candidate and curvature evaluation. It reads scalar log-probability/status fields without per-row rich snapshots; the public search API and numeric-buffer workspace query are unchanged.
