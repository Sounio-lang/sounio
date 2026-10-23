# Offset-aware Poisson count regression

Module: `stats::regression::poisson`.

## Model and API

`poisson_fit_offset(a:&[f64;2048],counts:&[i64;256],log_offset:&[f64;256],
n:i64,p:i64,rcond:f64,max_iterations:i64,tolerance:f64)->PoissonFit`
declares Mut,Div,Panic. Physical design stride8; n1..256, p1..8, p<=n.
Only active design cells and the first n counts/offsets are read.
Supply an intercept column explicitly when wanted.

The fitted conditional mean is mu_i=exp(log_offset_i + X_i beta).
Inputs are nonnegative integer counts and finite design/log offsets.
An exposure enters through its supplied natural logarithm; exposure zero
cannot be represented by a finite offset and is outside this contract.
For genomics, a scientifically justified sample size factor can supply a
log offset; this does not establish its scientific adequacy.

PoissonFit fields: status,n,p,rank,df,iterations,coefficients[8],means[256],
linear_predictors[256],deviance,score_norm,step_norm,rcond_used,
poisson_assumptions_assessed. Linear predictors include offsets. Inactive
successful output slots are zero. df=n-p is design residual degrees of
freedom, not an inferential calibration claim. The assessment flag is always false.

## Controls and statuses

rcond0 selects existing CPQR default n*binary64epsilon; otherwise require
finite0<rcond<1. max_iterations must be1..100. Finite tolerance must be in
the inclusive interval[1e-12,1e-4].

| status | meaning |
|---|---|
| 0 | numerical score and full Fisher-step convergence |
| 1 | invalid dimensions |
| 2 | active design or log offset nonfinite |
| 3 | initial or iteration weighted design numerically rank deficient |
| 4 | detected arithmetic/representability failure |
| 5 | invalid controls |
| 6 | negative active count |
| 7 | iteration budget or safeguarded line search exhausted |
| 8 | certified sufficient all-zero-count intercept-direction boundary |

Validation precedence: dimensions, controls, all active finite design/offset
checks, all active count nonnegativity, numerical initialization/rank, then
the sufficient boundary check and iteration arithmetic.

All failed fits have zero coefficients, means, linear predictors, df,
deviance, score_norm and step_norm. Requested n/p are retained.
Status3 may retain rank and rcond_used from the failed decomposition.
Iteration rank failures may retain completed iterations. Status7 retains
attempted iterations. Other failures currently leave these diagnostics zero.
Read status before interpreting solutions; failure diagnostics are not a partial fit.

## Initialization, iteration and stopping

Initialization solves unweighted CPQR of ln(count+0.5)-log_offset against
the supplied design. The0.5 is used ONLY to initialize finite logarithms:
actual deviance, scores, weights and updates use the unmodified counts.
With no intercept, this remains a projection onto the actual supplied design;
no column is silently added and no offset is estimated.

Each Fisher-scoring iteration solves for delta using the existing bounded
column-equilibrated weighted CPQR, weights mu_i and response(count_i-mu_i)/mu_i.
It forms trial beta+alpha*delta, starting alpha1 and allowing up to32 halvings.
For movements with every |X_i*(trial_beta-beta)|<=0.5, accept a trial only when
the stable local half-deviance change is nonpositive:
sum[(current_mu-count)*d + current_mu*(exp(d)-1-d)].
Form the ACTUAL represented coefficient displacement db=trial_beta-beta,
then compute d=X_i*db with the compensated design dot product. The fixed offset
cancels. Do not substitute alpha*delta or subtract independently rounded
trial/current predictors; the latter introduces row errors independent of the
step size near stationarity. Use a compensated Taylor remainder from d^2/2
through degree32 and a compensated outer signed sum.
The criterion evaluates the local parametric mathematical objective anchored
at represented current means. Independently rounded current/trial exponentials and absolute
deviances can differ in their final bits: reported absolute deviance is not
promised bitwise monotone. For larger movements retain the absolute-deviance
nonincrease comparison; halving brings movements into the stable local branch.
Invalid trial arithmetic rejects that trial. No predictor, count or mean is clipped.

At the TOP of an iteration, require BOTH:

* score_norm<=tolerance, where score_norm is the maximum absolute score after
  scaling each design column by its maximum absolute entry and dividing the
  count residuals by1+sum(counts);
* step_norm<=tolerance, where step_norm=max_i |X_i delta|/(1+|eta_i|),
  using the FULL Fisher step and current eta including offsets.

A small accepted HALVED movement never certifies convergence. On success the
returned score/step belong to that full-step test at the returned coefficients.
An accepted update at the final allowed iteration, without a subsequent full-step
check, returns status7. Objective change alone is never a stopping criterion.
Adding a common offset and compensating an intercept does not deliberately
change the predictor-step normalization.

## Deviance and numerical boundaries

D=2 sum[y log(y/mu)-y+mu], with the y0 contribution2mu.
Near y=mu, `math::logarithmic::log1p_phi_checked` supplies the factored positive
phi evaluation for (1+d)ln(1+d)-d, avoiding subtraction of first-order terms.
This shared module owns the logarithmic helper; the former private phi numerical
body is removed. Shared-helper failure maps to existing evaluation arithmetic
status4, with the solver's existing trial-rejection/failure handling retained.
The zero-count contribution remains explicit and does not call a logarithm
outside its domain. Far from equality use checked log differences instead of
forming a possibly overflowing ratio. Deviance and score sums use compensated
summation. See [shared logarithmic contract](../../math/LOGARITHMIC.md).
Every evaluated mean must be strictly positive and finite. Subnormal positive
means are permitted only while subsequent arithmetic and the WLS transform
remain representable. Individual weighted-transform underflow follows the
existing strict WLS arithmetic-failure policy.

i64 counts are converted to binary64 for this solver. Integers above2^53 can
round, and count-minus-mean contrasts can disappear at large counts.
This block does not claim the anchored integer-contrast precision of the
separate size-factor estimator. Tiny deviances may round to zero; no universal
relative accuracy for tiny residuals or correctly rounded transcendental result
is claimed. Direct design-times-coefficient products and iteration responses
must remain representable: finite inputs alone do not guarantee a usable fit.
Large fixed offsets that an explicit intercept can absorb are distinguished
from offsets outside the representable predictor range of the supplied design.

The local objective-change comparison avoids subtracting two large nearly equal
reported deviances. It does not eliminate every floating-point limit near
stationarity: line-search or iteration exhaustion still returns status7,
without claiming convergence. Rank tolerance is a
numerical policy, not a matrix condition estimate.

## Boundary and scientific assumptions

If all counts are zero and a nonzero constant design column certifies a
direction that drives all means toward zero, return status8. Initial rank
failure has precedence. This is ONLY a sufficient detector, not a general
finite-MLE/existence or separation certificate. Other boundary cases may
exhaust the budget or encounter representability limits. All-zero counts
with a balanced no-intercept design can have a finite optimum and are not
blanket-rejected.

The likelihood model assumes conditionally independent Poisson observations
and an appropriate log-linear conditional mean, with conditional variance mu.
The function does not assess those assumptions, overdispersion, dependence,
exposure quality or design adequacy. It supplies no negative-binomial model,
dispersion estimate, standard errors, confidence intervals or significance
tests. In particular RNA-seq count overdispersion is not solved by this Poisson fit.

## Bounded implementation and reference

No heap allocation or explicit normal equations are used by the production
solver. It delegates each scoring update to existing CPQR/WLS, with their
nested fixed scratch/results, plus bounded256/8-element iteration buffers.
This is not a claim of a single-matrix total stack footprint or measured RSS.

tests/reference/poisson_reference.cpp uses independent quad113 score/Hessian
calculations and dense pivoted Gaussian elimination, rather than production
CPQR. Intercept/exposure and disjoint indicator designs have analytic reference
solutions. Other references stop with relative parameter-step below1e-14,
well below fixture comparison tolerance; full quad relative convergence is
not claimed. Compile C++23 with -lquadmath on t560.

The generated26-case stats_poisson_reference.sio covers analytic and nontrivial
designs, offsets, some/all-zero counts with finite optima, no intercept, large
integer counts, scaled columns and max256x8 dimensions. stats_poisson_status.sio
checks validation, rank, boundary, iteration budget, damping, representability
and inactive prefixes. Native Sounio acceptance is established only by Forge.

### Reference fixture compaction checkpoint

The maximum256x8 reference case now uses exact quad-rounded24-entry mean/predictor
tables and fixed-bound setup/check loops. The C++ generator bit-checks every input
and expected value against the original reference. All256 means/predictors,8
coefficients and metadata remain checked; the other25 cases and tolerances remain
unchanged. Native job15613 passes101/101. Separate phase-profile job15615 records
compilation and direct execution; see docs/plans/STATS_VIZ_PHASE_PROFILE_EXECUTION_2026_10_19.md.
This changes validation cost, not the production estimator or its scientific contract.

### Shared-logarithm integration state

This source refactor changes ownership of the local phi evaluation, preserving
the solver API, count0 case, initialization, score/full-step stopping and local
exponential objective-change safeguard. The exponential remainder used by that
safeguard is not redefined as a logarithmic remainder. Selected current-source typecheck is complete in final Slurm15825,
alongside two compiled/executed native examples. The count diagnostic completes
Poisson-based means, its in-memory Canvas path and SVG emission. Its actual
SVG/XML/full PNG preview was inspected; no Canvas pixel comparison occurred.
Frozen executable source:27ea4ec6c2b97892e9e92cf982b0af90e2241023. Compiler
reuse followed exact unchanged bin/scripts/self-hosted/stdlib verification.
See [shared-log execution receipt](../../../docs/plans/LOGARITHMIC_EXECUTION_2026_10_21.md).

No tests or independent numerical oracle were added/run. The historical Poisson
corpus receipts above do not automatically validate the changed arithmetic;
new numerical concordance, boundary/convergence coverage, full corpus and
CI/merge remain pending. No statistical calibration follows from native
demonstration success. These documentation edits postdate the frozen source.
