# Dynamic fixed-alpha count-model parametric bootstrap

`dynamic_count_lrt_bootstrap` computes a model-conditional parametric-bootstrap tail for a likelihood-ratio statistic. It fits a declared null and alternative to the observed counts, generates each replicate under the fitted **null**, and refits both models to every generated dataset. It reuses checked CountGLM, the full NB2/Poisson log PMF and the canonical checked count sampler. It creates no numeric-buffer or integer-buffer owners internally; scalar/runtime representation costs are separate.

This procedure is conditional on a supplied fixed alpha and plug-in fitted null means. It is **not an exact finite-sample test**, an automatically valid marginal p-value, a calibrated RNA-seq pipeline or an assessment of the scientific model. A positive-alpha dispersion is never silently replaced by Poisson. The current implementation does not refit dispersion.

## Models and declarations

The caller supplies two design buffers, their matrix layouts, two log-offset vectors and two initial coefficient vectors. Both models use the same n observations and count vector; p_null<=p_alternative is required. No intercept is implicit. Equal models are allowed. This dimension check cannot prove nesting.

The caller must explicitly declare nesting, the independent Poisson/NB2 working model and observation independence. Missing declarations return status11 before fitting or consuming RNG. Even when declared, `assumptions_assessed`, `global_optimality_assessed` and `finite_sample_validity_assessed` remain false. The implementation does not verify nesting or prove that each converged point is the global model MLE.

Alpha is supplied and held fixed in the observed fits, generator and every replicate fit. `alpha_refit=true` returns explicit unsupported status10. The caller explicitly supplies `alpha_plug_in`; `alpha_fixed` is its negation and records the declared origin, matching the Fisher API. The separate `alpha_refitted` result is always false. Regardless of that declared origin, this algorithm holds the supplied alpha unchanged. None of these flags verifies that alpha was known independently of these data. If a caller supplies an estimated alpha and holds it fixed, its estimation uncertainty is absent. Full dispersion refitting under both models and in every replicate is a separate future method.

## Statistic, likelihood and failure policy

For each paired fit, full per-count log probabilities, including their alpha-dependent normalizer, are accumulated as scaled compensated means. The signed difference is mean_loglik_alternative−mean_loglik_null. The statistic is 2*n times that difference. A negative result is not clipped, rounded away by an epsilon, retried or counted as zero: status6 makes the aggregate unavailable and retains the signed diagnostic. A nonnested pair may trigger this even when the caller incorrectly declares nesting.

A finite mean difference with an overflowing/unrepresentable total statistic returns arithmetic status4. Full likelihood availability is independent of point-fit convergence; unavailable full log PMFs return status9. Near an exact zero, subtracting rounded mean log likelihoods remains cancellation-sensitive. Compensation is not a certificate of relative accuracy, and a reported zero is not a mathematical equality proof. The identical-model fixture exercises a path whose two native fits and evaluations are identical.

Every failed sample, unresolved fit, invalid likelihood, negative LR or precision warning stops the aggregate. No replicate is discarded and no denominator is silently reduced. Numeric aggregate placeholders are zero on failure. Partial hits, completed-replicate counts and scratch data are diagnostics, not a successful bootstrap result.

| status | Meaning |
|---|---|
| 0 | All requested replicates resolved and aggregate available |
| 1 | Invalid owner, matrix, dimension or workspace budget |
| 2 | Invalid controls, RNG parity, count-work product or numeric options |
| 3 | Null/alternative point fit failed; inspect last fit status and failure_stage |
| 4 | Arithmetic, representability or counter-consistency failure |
| 5 | Fit or sampling work budget exhausted |
| 6 | Negative signed LR; inspect signed fields |
| 7 | Checked sampler failed; inspect sampler_status |
| 8 | Sampler resolution_limited or rate_underflow warning |
| 9 | Full log PMF unavailable; inspect likelihood_status |
| 10 | Requested alpha refitting is unsupported |
| 11 | Required scientific declarations absent |

`failure_replica` is -1 for the observed fit and otherwise a zero-based replicate index; it is reset to -1 on aggregate success. `failure_row` is the zero-based row of a failed draw, otherwise -1. Failure stages:1 null fit,2 alternative fit,3 null full likelihood,4 alternative full likelihood,5 negative LR,6 sampler failure,7 sampler precision,8 sampler counter-consistency. A fit-budget failure can occur before starting a pair with failure_stage0. Successful point fits do not by themselves authorize consuming a bootstrap p-value.

## RNG, sampling limits and budgets

The input `PcgState` is passed by value once. One reusable count cursor and one result slot serve all draws; no RNG-bearing draw result is constructed per row. The cursor advances by borrowed mutation, and its RNG is copied into the aggregate only at success/failure return checkpoints. The result always returns the actual advanced state. Precondition failures preserve it. After a partially completed draw, the returned sampler state is retained even when the aggregate fails. Seeds, pseudorandom streams and the finite53-bit uniform grid are not a proof of statistical independence or exact continuous-uniform sampling.

Work limits are explicit: maximum fit attempts, uniforms/proposals per draw and total uniforms/proposals. A proposal/uniform limit of zero is a valid exhausted budget. A null/alternative pair is not started unless both fits can be afforded; this avoids consuming samples that cannot be refitted. Actual work counts are retained, including failed attempts. Products for n*B and every workspace sum/product are checked in integer arithmetic.

The checked sampler's supported mean/shape ranges apply in addition to the fit's domain. An otherwise available fit can therefore have unavailable sampling. A count outside its exact supported range, an unavailable positive rate or another sampler error is propagated. `resolution_limited` and `rate_underflow` warnings invalidate this calibration even when a point draw has sampler status0. No rare-event precision flag is silently ignored.

## Monte Carlo output

After all B replicates succeed, hits counts replicate LR>=observed LR with the explicit native floating-point comparison, including ties. The reported tail is

p_value = (hits+1)/(B+1),

with 1<=B<=2^53−1. The plus-one form prevents a reported zero tail; it does not make a plug-in parametric bootstrap an exact test.

Separate Monte Carlo diagnostics are hits/B and sqrt((hits/B)*(1−hits/B)/B), evaluated with factored square roots. This is the estimated Bernoulli standard error for the raw tail fraction, not a confidence interval, coverage guarantee or standard error of a regression coefficient. It is zero at all-hit/no-hit endpoints and should not be interpreted as certainty. B=7 in the selected source fixture is for deterministic concordance, not an adequate generic scientific precision recommendation.

These model-conditional outputs do not prove the valid-marginal-p assumptions required by a downstream FDR procedure. Applying BH or BY does not repair unassessed model or bootstrap calibration.

## Workspace and source identity

Use `dynamic_count_bootstrap_workspace(n,p_null,p_alt)`. Two CountGLM workspaces are needed because their weighted-design owner lengths are exact n*p. Fit coefficient/mean/predictor buffers are also caller scratch; they are overwritten during refits. A separate length-n generating-mean buffer preserves the fitted observed null means across all replicates. A length-n integer buffer holds one replicate and is reused.

Total numeric scratch is

2n*(p_null+p_alt) + 17n + 9*(p_null+p_alt) doubles,

plus n integers. This includes both fit workspaces and their output scratch, generating means and replicate counts. Inputs and allocator overhead are excluded. No replicate history is retained, and no claim about measured RSS or execution time follows from these payload counts. All failure numeric outputs are gated by status/availability; scratch contents are not transactional public scientific results.

## Independent verification target

`deep_stats_count_bootstrap_reference.cpp` uses a separately implemented unsigned128 PCG/high-precision sampler route from the sibling C++23 reference source. It computes analytic group-mean refits and direct binary128 full log PMFs, independently of Sounio IRLS, QR, likelihood and bootstrap code. The selected cases cover Poisson, NB2 with alpha0.25 and identical null/alternative models. Native hits, work counts, final RNG limbs and final replicate counts must agree exactly; full LR and Monte Carlo scalar values are compared numerically. The final fixture completion marker is mandatory, preventing a prefix of successful cases from being accepted as full completion.

The source fixture also checks unsupported alpha refitting, absent declarations, exhausted fit/sampling budgets, count-conversion failure, negative nonnested LR and rare-event sampler precision. It is a selected numerical/contract gate, not a complete calibration study or theorem. Build, execution and actual-output concordance belong to the central Forge/Slurm receipt.



## Selected native checkpoint

The coordinating Slurm16130 execution reported native fixture success and actual-output C++ concordance for the three nominal seven-replicate cases and the declared status/diagnostic cases. That source checkpoint used the reusable sampling cursor, detached RNG snapshots and the corrected CountGLM local-gain line search. The bootstrap source SHA256 was 9245ed3487dd96d73e47550d8c26eb27b743e3b0427aaff834ca954e75c61692.

The independent comparison includes integer hits, proposal/uniform counts, final RNG limbs and final replicate counts, as well as scalar statistics. The earlier failure on the second Poisson replicate was preserved and resolved through the point-fit correction; the replicate was not discarded, reseeded or replaced, and B stayed seven.

This checkpoint is selected deterministic algorithm/contract concordance. It does not establish useful Monte Carlo precision for a scientific application, validity of plug-in nuisance calibration, exact finite-sample p-values, downstream FDR control, full-corpus/CI success or measured scalability. Subsequent reusable NB2-likelihood-slot plumbing requires a fresh compatibility gate and does not inherit acceptance automatically.


## Consumer-slot integration checkpoint

In the integrated full2 run (Slurm16139), this module's selected native fixture and its registered actual-output C++ comparisons returned0 after reusable-slot consumer integration. The count GLM, Fisher, bootstrap, dispersion-profile and scalar-likelihood rows were successful. The overall full2 gate nevertheless returned1 because the separate NB2-slot fixture and its comparison failed; a passing subset is not a passing complete gate.

The coordinator corrected that fixture contract and the focused slots_final rerun (Slurm16144) returned0 for its native slot program and independent analytic comparison. The retained million-draw PTRS reference/control also reported success with exact state/counter checks. These are compatibility and selected correctness observations, not a measured speed/RSS improvement, a universal error bound or a finite-sample calibration result. The final exact-index integration/CI gate remains separate.

The consumer now creates one Nb2LikelihoodSlot for both model likelihood means and all replicates, in addition to the reusable sampling cursor/result slot. It reads scalar fields without per-row rich snapshots; public arguments and numeric-buffer workspace sizes are unchanged.
