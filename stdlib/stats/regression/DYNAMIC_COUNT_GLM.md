# Dynamic fixed-alpha count GLM

The module stats::regression::dynamic_count_glm fits a log-link count regression
with caller-supplied NB2 dispersion alpha>=0. Alpha zero uses the Poisson limit.
It does not estimate dispersion, produce covariance/inference, or assess the
count-model assumptions.

## Model and supplied initialization

The supplied design and log offsets define

    eta_i = design_row_i * beta + log_offset_i
    mu_i = exp(eta_i)
    variance_i = mu_i + alpha*mu_i².

There is no implicit intercept. Every initial coefficient is supplied explicitly
in a separate buffer. Log offsets are consumed as supplied finite numbers; raw
exposures are not reconstructed.

For fixed alpha, Fisher scoring solves a weighted least-squares increment with

    working_response_i = (count_i-mu_i)/mu_i
    fisher_weight_i = mu_i/(1+alpha*mu_i).

These are point-fitting equations. No estimated residual scale substitutes for
the supplied alpha. Alpha zero follows the same public API with Poisson weights.

## API and storage

dynamic_count_workspace(rows,columns) returns checked scratch sizes.
dynamic_count_fit accepts:

1. design NumericBuffer and its NumericMatrix layout;
2. raw counts IntegerBuffer;
3. log offsets and initial coefficients NumericBuffer;
4. alpha, rcond, max_iterations, tolerance;
5. caller output buffers coefficients, means, linear_predictors;
6. caller scratch buffers weighted_design, response, step, qr_workspace, stage.

All owners remain with the caller. There are no internal allocations, no
fixed 256-observation/8-coefficient ceiling, and no value copies of large arrays.
Rows and columns must be positive, columns<=rows, and both must be <=2^53;
all workspace products and sums must fit i64. The aggregate scratch payload
byte count must also be representable under the native buffer byte-budget bound. The input matrix may use a valid
strided descriptor. The scratch weighted matrix is contiguous row-major.

| Scratch | Elements |
|---|---:|
| weighted_design | n*p |
| response | n |
| step | p |
| qr_workspace | n*p+n+4p |
| stage | 3p+4n |
| Total scratch | 2n*p+6n+8p |

This is requested payload, not RSS or allocator pages. Input/output ownership,
allocator metadata, runtime stack, compiler and executable storage are additional.
Each owner has its own explicit budget. Construction or workspace failure is
reported, not repaired by increasing a budget.

The three public output buffers are unchanged on every failure. Scratch can
change after input validation. Successful publication happens only after the
fit's score/full-step convergence and all final arithmetic checks.

## Numerical organization

The design is column-equilibrated and the working response normalized before
row weighting. Row factors use sqrt(weight)/sqrt(max_weight), avoiding the
premature underflow of weight/max_weight. Dynamic pivoted QR solves the
transformed increment. Coefficients are restored through the existing checked
least-squares scaling helper.

Nonzero values lost to zero during working-response division, column/response
normalization, row scaling, or score normalization make arithmetic unavailable.
No arbitrary floor revives a coordinate or mean. Predictors, means, products,
scores and objectives must remain finite on the declared staged path.

For alpha>0 and mu>=1, the weight is evaluated as 1/(alpha+1/mu), avoiding an
overflowing alpha*mu. For smaller means it uses mu/(1+alpha*mu).
The active means must be positive and representable.

The objective is the mean of checked per-observation saturated deviances from
nb2_deviance, retaining its count-zero, Poisson, near-mean, small-alpha and
balanced tiny-product policies. This avoids introducing a separate unstable
subtraction of NB2 likelihood terms.

This first finite-feasible contract requires every unit squared deviance to be
available. It can reject inputs for which a more elaborate scaled objective
would be representable. Dividing a positive unit deviance by n to zero is also
unavailable. The final total deviance has a separate status: overflow of
n*mean_deviance does not invalidate an otherwise available fit.

For a candidate whose relative mean changes d=(mu_trial-mu)/mu all satisfy
|d|<=1/4, backtracking compares a paired fixed-alpha log-likelihood increment
instead of subtracting two rounded absolute mean deviances. Set
W=mu/(1+alpha*mu), q=alpha*mu/(1+alpha*mu), s=((y-mu)/mu)*W. Then

    gain_i = (s*d)*A(d,q) - (W*d²)*B(d,q)
    A = integral_0^1 1/((1+d*u)*(1+q*d*u)) du
    B = integral_0^1 u/((1+d*u)*(1+q*d*u)) du.

The derivative identity is d(ell)/dd=(s-W*d)/((1+d)*(1+q*d)).
A and B use32 compensated geometric terms with S_0=1,
S_k=1+q*S_(k-1), coefficients (-d)^k*S_k/(k+1) and
(-d)^k*S_k/(k+2). For q in[0,1], S_k<=k+1 bounds either omitted
real-number coefficient tail by |d|^32/(1-|d|). This is an algebraic/truncation
argument, not a proved floating-point error certificate. The products are
balanced before rounding; q near one does not require subtracting nonlinear
terms of order y to recover a tiny conditional Fisher weight W.

The compensated mean paired gain must be nonnegative to accept a local step.
For nonlocal candidates, the previous available/nonincreasing absolute mean
deviance comparison remains. Invalid local arithmetic is explicit status4.
There is no epsilon permission for a negative paired gain, no relaxed stopping
tolerance and no increased iteration budget. The separately reported absolute
mean deviance can fluctuate by rounding ulps even when this paired criterion
improves; it is not an exact monotone floating-point trace.

Convergence requires both a column-scaled score norm (normalized by 1+sum counts) and the
next full predictor-step norm to meet tolerance. A damped accepted step is not
a convergence certificate. These are staged numerical criteria, not proof of
exact optimality or global floating-point accuracy.

## Validation and status

Counts remain i64 until validated. Negative counts and values above 2^53 are
rejected; every accepted count converts exactly to binary64.

| status | Meaning |
|---|---|
| 0 | Point fit available |
| 1 | Invalid shape, owner construction status or workspace |
| 2 | Nonfinite active design, offset or initial coefficient |
| 3 | Negative count |
| 4 | Arithmetic/scaling/objective unavailable |
| 5 | Invalid alpha or solver controls |
| 6 | Design or weighted-design rank deficiency |
| 7 | Iteration/backtracking limit |
| 8 | Sufficient all-zero-response/constant-column boundary |
| 9 | Count exceeds exact binary64 conversion range |

Size/buffer checks precede controls, then finite active inputs, negative counts,
and exact-count range. Design rank is checked before initialization evaluation.
Only a nonzero constant design column with all-zero counts is diagnosed as an
explicit boundary; this is not a general finite-MLE existence/separation test.

Controls: rcond0 selects the dynamic QR default n*binary64epsilon; otherwise
0<rcond<1. max_iterations is 1..100 and tolerance is 1e-12..1e-4.
Requested rows/columns are retained on failures; other fit payloads are
unavailable. A successful result has rank=p, df=n-p, alpha, poisson_limit,
score/step criteria and scratch payload count.

deviance_status is 0 for an available total, 1 for total overflow, and 2 when
not computed. The unavailable total is a zero placeholder. mean_deviance is a
separate finite objective on successful fits.

inference_status remains 1 and assumptions_assessed remains false. No standard
error, Wald statistic, p-value, likelihood-ratio calibration, alpha estimate or
confidence interval is claimed.

## References and evidence boundary

The unit definition is documented in [NB2_DEVIANCE.md](NB2_DEVIANCE.md).
The implementation reuses the checked dynamic QR contract rather than normal
equations. The companion C++23/binary128 reference independently uses analytic
group-intercept MLEs and verifies their count-score equations; it does not reuse
production QR, Fisher iterations or logarithmic/deviance helpers.

The native fixture exercises 512 observations and 12 coefficients, Poisson and
fixed-alpha NB2 fits, a known constant offset shift, rank/boundary failures,
exact-count validation, transactional outputs, and a finite mean objective whose
summed deviance overflows. It also contains the exact16-count bootstrap replica
that previously stalled before the score/full-step tolerance, and a supplied
alpha1e300 case where q rounds to one but W remains nonzero.
The additional C++23 direct-full-PMF reference at400 decimal digits checks the
local-integral identity/truncation, including extreme weights, independently of
the production deviance representation. Native end-to-end coefficient checks
are separate; no isolated IEEE helper-concordance claim follows from checking
the identity in high precision. Actual execution/concordance requires the coordinator's
central Slurm receipt. No host/workspace compiler execution is acceptance evidence.


## Selected native checkpoint

The coordinating Slurm job16130 reported native success for the original512x12 Poisson/NB2 fixture and the two added increment regressions, with actual-output C++ concordance. The CountGLM source at that checkpoint was SHA256 dbb97acecce18cef1b8ef71531f25dd535177185257af8647a2d4cdd4e359e22. The separate400-digit reference completed256 direct-full-PMF increment comparisons and checked the native regression coefficients through independent analytic score/full-step equations. It does not certify a global IEEE error bound.

The preserved pre-fix bootstrap replica failed alternative fitting with status7. Its exact count vector was independently reproduced by the C++ PCG route; high precision found a positive full-step likelihood gain while the old absolute-deviance line search stalled above the unchanged convergence criteria. The selected post-fix native execution resolves this witness without changing its initialization, tolerances, iteration budget or data.

This checkpoint predates subsequent reusable NB2-slot dependency/consumer plumbing. Such changes require their own fresh compatibility execution; the checkpoint is not a blanket acceptance of later source trees, a full corpus/CI result, an inferential-calibration study or a speed/RSS comparison.


## Consumer-slot integration checkpoint

In the integrated full2 run (Slurm16139), this module's selected native fixture and its registered actual-output C++ comparisons returned0 after reusable-slot consumer integration. The count GLM, Fisher, bootstrap, dispersion-profile and scalar-likelihood rows were successful. The overall full2 gate nevertheless returned1 because the separate NB2-slot fixture and its comparison failed; a passing subset is not a passing complete gate.

The coordinator corrected that fixture contract and the focused slots_final rerun (Slurm16144) returned0 for its native slot program and independent analytic comparison. The retained million-draw PTRS reference/control also reported success with exact state/counter checks. These are compatibility and selected correctness observations, not a measured speed/RSS improvement, a universal error bound or a finite-sample calibration result. The final exact-index integration/CI gate remains separate.

The consumer now creates one Nb2DevianceSlot per fit outside all evaluation/iteration loops. Row evaluations read scalar status/deviance getters; the numerical buffer workspace query and public fit signature are unchanged.

## Reusable scalar intermediates

Each fit retains one scalar dot-product slot and one balanced-product slot
outside its row, iteration and local-gain loops. Conditional weights are
scalar values with an internal failure sentinel; matrix reads after full
input validation use the checked scalar accessor. The public fit signature,
workspace query, arithmetic order and convergence controls are unchanged.

Balanced products normalize six primitive mantissa/exponent pairs. Checked
logarithms and NB2 private scratch use reusable slots through their hot call
chains. This removes those per-observation result constructions; it does not
claim the entire runtime or every public compatibility wrapper is allocation
free. Native output concordance and measured RSS remain separate evidence.
