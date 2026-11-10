# Checked paired NB2 likelihood increments

This is an additive contrast API, not a fit, a calibration certificate or a
correct-rounding claim. Its sign convention is **second model minus first**.
A negative available contrast remains negative. No epsilon clamp is applied.

`nb2_likelihood_difference_slot()` owns two independent likelihood child slots
and one reusable balanced-product workspace. Call
`nb2_log_probability_difference_into(y,mu0,a0,mu1,a1,&!slot)`, then read
`nb2_likelihood_difference_status/value/first_status/second_status`.
`nb2_likelihood_difference_snapshot` creates a detached checkpoint record with
status, value, first_status, second_status and local_increment. The into path
does not construct a per-observation result. Slots must not be shared across
simultaneous exclusive calls.

Statuses follow the scalar primitive:0 available,1 negative count,2 invalid
mean,3 invalid alpha,4 count above2^53,5 arithmetic/unresolved contrast.
Invalid parameters are checked through the two existing scalar calls, in first
then second model order. Unavailable contrast payload is0. Individual PMF
availability is independent: a stable local contrast can be available even if
an absolute PMF was unavailable. Read its two statuses separately.

For equal alpha and relative mean displacement |d|<=1/4, the existing CountGLM
local increment is shared through `nb2_log_increment.sio`, rather than subtracting
two rounded log PMFs. It uses the same32-term compensated geometric series and
balanced conditional-weight products as the prior line search. It is a finite
binary64 approximation with explicit unavailable-product guards, not an exact
series evaluation or a uniform relative-error bound. Identical valid means and
alpha return an exact zero contrast. CountGLM consumes this same scalar helper;
its line search still uses full-step/score convergence and its existing far route.

The bounded candidate alpha increment at fixed mean is exposed separately as
`nb2_dispersion_increment_into`, with a caller-owned CheckedLog scratch and
LeastSquaresNumber output. Its finite-sum branch is limited to y<=4096,
|a1/a0-1|<=1/4 for positive a0, and representable required intermediate terms.
Poisson transitions have their own algebraic limit; reversing a transition
negates its increment. Status6 requests a checked far fallback, while status5
marks detected lost/unrepresentable arithmetic. The algebraic identity was checked against an independent finite-product full
PMF at400 decimal digits in Slurm16806,38 cases including Poisson transitions,
adjacent tiny alpha and y4096, with a rejected sign mutant. It is now integrated
into the paired route. This validates the identity on that grid, not its general
binary64 implementation error; actual native concordance remains a separate gate.

For unequal alpha the local route decomposes the total change into a mean
increment at second alpha and a dispersion increment at first mean. The latter
retains the finite-product normalizer terms; it does not cancel them as if alpha
were fixed. Required arithmetic that rounds away is unavailable.

The far/unsupported route retains both full scalar PMFs, including both
alpha-dependent saturated normalizers. It is not generally cancellation-safe
for adjacent dispersions. A rounded far difference of0 for distinct parameters
is therefore unavailable5 rather than a certified tie. Nonzero far differences
have ordinary binary64 evaluation error; successful status does not certify
their sign arbitrarily close to equality.

`nb2_log_probability_mean_difference_into(counts,means0,a0,means1,a1,n,&!slot)`
requires valid exact-length n buffers and n>=1. It scales and compensates signed
per-row contrasts before restoring the mean. It does not subtract two separately
rounded mean log-likelihoods. A nonzero term lost during normalization, or a
nonzero aggregate lost in final restoration, is unavailable5. Cancellation and
rounding still follow binary64; this is not interval arithmetic. Snapshot
local_increment means all contributing rows used a supported local route;
individual PMF statuses retain detected unavailability across the rows.

## Preserved diagnostic baseline

Slurm16763 replayed the original n16 alpha0 campaign cases3,11,13 against the
unchanged full02 library and compiler ee0f341e..., exporting raw IEEE parameters,
replicate counts and PMFs. Each CAL_RESULT matched its original failed campaign
record. Independent C++ Big400 Slurm16770 found all three *fixed native point*
LRs truly negative (approximately -3.437e-16,-5.131e-16,-3.175e-16). The old
rounded-mean calculation magnified their magnitude, but did not invent the sign.
Both groups have identical count totals in each failed replica; the alternative
coefficients are equal to each other but differ slightly from the null coefficient.
These original points remain negative regression controls. They are not replaced
by analytic refits, and paired accumulation alone must not make them available
as a valid nonnegative fitted LR.

The strict embedding/warm-start route addresses candidate optimization separately;
see DYNAMIC_COUNT_PROFILE_BOOTSTRAP.md. Focused native gate16817 completed both library checks and the paired fixture
check/compile/native/marker, followed by independent C++ concordance: all exit0.
The fixture includes61 scalar checks and three original negative aggregate
contrasts; none is clamped to zero. This finite selected grid is not a universal
relative-error certificate. Strict native regressions, resource comparison and
renewed campaign qualification remain separate gates.

An additional preserved legacy all-six-count case exposed an available negative
LR only after the paired change. Fixed-point Big400 Slurm16818 found true
LR=-1.29518154689632074287582019284227754e-21, concordant with the native paired
value, while the sum of rounded row PMF differences was0. Its legacy expectation
now requires negative-LR status6 and no random draws; it is not weakened to a
rounded tie. A separate strict all-six fit test requires the embedded candidate
comparison to remain nonnegative and stop under the declared fit budget.

Oracle refinement16820 retained all38 identity cases and promoted the three
stationary y1/mean2 tiny-alpha cases to1000-decimal full finite-product PMF
evaluation, with no absolute zero floor. The required contrasts can be around
1e-616 and have no nonzero binary64 representation; native unavailable5 is
intentional, not an available zero. The updated paired C++ comparator also
passed against the same61-scalar/three-aggregate native output and ELF from16817.

Focused integration16819 completed library/typecheck/native checks of the shared
increment consumers, legacy bootstrap/profile/CountGLM, and the strict nested
fixture. Its frozen snapshot exercises the three preserved failed-replica count
vectors as newly fitted observations, the separate all-six case and four invalid
embedding/offset/NaN/warm-size configurations. Additional overflow, underflow and
nonassociative warm-start status cases were prepared after that snapshot and
remain pending their own gate. This is a selected integration gate; B199 qualification
and resource superiority have not been established by it.

Slurm16828 additionally completed a full strict two-replicate grouped-model
bootstrap and its independent C++ comparison, including RNG/hits, last counts
and refitted dispersions. Fixed-alpha and incomplete-output controls were
rejected. This closes the B2 smoke path, not the pending B199 qualification.
