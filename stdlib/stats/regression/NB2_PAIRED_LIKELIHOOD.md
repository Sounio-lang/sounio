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

## Additive checked ranking path (candidate)

`nb2_log_probability_rank_difference_into` and
`nb2_log_probability_mean_rank_difference_into` are ranking-specific entries.
They validate the active counts and finite positive means, finite nonnegative
alphas and exact-count range before attempting the same checked local increment
primitives. A local result does not evaluate either individual full PMF:
`first_status` and `second_status` are then 2 (not computed), not0. This includes
local arithmetic unavailability; callers must inspect the main status. The
existing full-difference APIs retain their individual PMF metadata contract.

A local request outside the supported geometry (internal6) delegates to the
entire existing full-difference route, preserving both alpha-dependent
normalizers and its metadata. Arithmetic status 5 never becomes an available
tie. Invalid input also delegates to that checked route, retaining its error
precedence. The aggregate engine is shared with the full route, including its
scaled compensated sum, range checks and failure handling. No statistical
approximation or tolerance is introduced.

The traced ranking aggregate counts only PMF callsites actually executed:
zero for a local row, two for a full-fallback row, including failed prefixes.
Profile candidate ranking uses this additive entry; final bootstrap comparison
continues to use the full metadata entry. Observed-null initialization, fitting,
weighted QR and random generation are unchanged. Selected concordance and
performance acceptance for this candidate remain separate from the earlier
correctness baseline.

Focused Slurm 16912 completed checks, native execution and annotation markers for
the new ranking status/metadata fixture, the ranking and original61-scalar/three-
aggregate fixtures, the trace unit and both grouped B2 drivers. Both numerical
routes passed the same independent fixed-point C++ comparator; both B2 outputs
passed the independent profile-refit comparator. The traced B2 also retained its
bitwise off/on assertions and actual counter checks. Its scientific output rows
were byte-identical to corrected-compiler baseline16899. Recorded full-PMF row
calls were6336 versus 12224 in that baseline, with the other nine recorded work
counts unchanged. These are actual callsite counts on one fixed B2 input, not a
runtime speedup or an extrapolation to B199. The separate B31 paired resource pilot is recorded below; broader
qualification remains pending.

The selected B31 resource pilot 16915 recompiles original full02 and the strict,
replica-warmed, locally ranked candidate with the same corrected compiler
SHA92d2108ef595c5382d582aa363df6e95dab907a554f039d5f8f008073e535c21.
Its existing C++ analyzer reports154.05s versus 90.81s and maximum RSS 61568KiB
versus 59364KiB; the before/after runtime ratio is1.69639907499174099779. Dataset,
options and seed schedules were retained, and all scientific output rows,
hits30/31, plus-one tail and final RNG were byte-identical. This is one whole-
program B31 pair on one worker, combining the earlier strict/warm changes with
local ranking; it does not isolate the ranking speedup, reach the 2x target or
replace the pending B199 qualification. The untraced driver does not report
uniform/proposal/work totals, and the analyzer explicitly leaves those counts
unavailable. Nested preflight/status assertions passed before the timed pair.

## Opt-in cached ranking fallback

`Nb2CachedLikelihoodDifferenceSlot` contains two cached likelihood owners and
one private difference core. It does not wrap a legacy difference slot with
unused PMF children. The legacy slot retains two uncached likelihood owners;
neither legacy scalar likelihood slots nor legacy difference slots acquire
cache arrays. Both difference forms use the same checked local helper, domain
validator, scaled/Kahan accumulation step and mean finish. The core holds reused
scratch, result metadata and actual event counts; it adds one small managed
owner per slot factory, never per row or comparison.

The new `nb2_log_probability_rank_cached_difference_into` and aggregate/traced
forms keep local results at zero PMF calls with child statuses 2 (not computed).
Unsupported local geometry uses two complete cached PMFs, preserving both
normalizers when alphas differ. A cached normalizer hit is still a full PMF call:
the recorded row-PMF count is 2, not 0. Arithmetic unavailability, invalid-input
precedence, failed-prefix accounting and far rounded-zero rejection remain
checked. The final bootstrap comparison stays on the uncached full API.

The cached-difference payload query reports 4096 array bytes; the profile's
private-cache query reports 6144 array bytes for one objective and two ranking
owners per search. Existing likelihood workspaces, the shared core, scalar
controls, handles and other managed overhead are excluded from those payload
queries and require resource measurement. Public cached-difference reset clears
both caches and the reusable core. No QR factor, weight or fitted coefficient
is cached. The old public difference snapshot remains unchanged.

Named cache events now include the objective and cached ranking fallback. The
final uncached bootstrap comparison and sampler remain outside their scope.
Counters come from the actual first/second cached PMF flags, including failed
prefixes, with checked aggregation and trace-limit handling. The original ten
trace metric IDs and public trace snapshot do not change. Selected independent numerical, bitwise legacy-preservation and B2/RNG
results are recorded below; broader qualification remains separate.

Focused Slurm 16954 completed all thirteen selected native fixtures and their
markers. Cached/legacy/ranking 61-scalar and three-aggregate outputs passed the
independent comparator; cached/legacy 216-case outputs and exact Poisson-zero
bits passed their C++ comparators. All B2 paths passed independent concordance
and off/on/RNG assertions. Both refactored legacy output protocols were also
byte-identical to the frozen pre-core outputs from 16936, not merely within a
tolerance. The new status fixture covered invalid inputs, far rounded-zero
unavailability, failed prefixes, reset, warm hits still reporting two PMFs and
count 64 bypass.

Slurm 16965 added eight independent far full-PMF comparisons, including unequal
alphas, reversed arguments, changed means, counts 63/64 and a representable true
contrast around -5e-201 whose rounded PMFs tie: status 5 remains unavailable. The
positive native output and independent C++ 400 comparator passed. Two narrow
mutants compiled and then failed natively: using the first alpha for the second
PMF, and reporting zero PMF calls on cache hits. Neither control changes the
accepted source.

Whole-program pilot 16968 used original full02 versus the combined candidate,
both recompiled with the same 92d compiler, fixed B31/n256/alpha0/case0 dataset,
seed schedule and maxfits 64. The C++ analyzer found byte-identical scientific
rows and reported 154.90s versus 76.45s, RSS 61568KiB versus 58900KiB and a ratio of
2.02616088947024198825. This exceeds 2x only for that selected paired B31 run.
It does not establish the full B199/three-pair/median target, an isolated cache
speedup, assessed calibration or general superiority. Source, ELF, compiler
and input manifests were verified; the untraced driver leaves work counters
unavailable rather than deriving them from the fit budget.

## Conservative availability after aggregate cancellation

All full, ranking and cached-ranking aggregates share the same accumulation and
finish engine. A new guard returns status5 with cleared value0 whenever nonzero
rows were accumulated (`scale>0`) but the final binary64 scaled sum is0. The
summation formula and floating-point operation order for nonzero sums are
unchanged. Pointwise-equal row contrasts (`scale==0`) still return available0.

This policy deliberately also rejects mixed nonzero rows whose mathematical sum
is exactly0: this engine has no exact-sum certificate to distinguish that case
from loss of a tiny contribution followed by cancellation. It is conservative
availability, not an exact-summation algorithm, a sign certificate for nonzero
results, or an interval/error-bound implementation. A trace can remain available
while the numerical contrast is unavailable; it records actual calls made.

Preserved scratch Slurm16996 reproduced status0/value0 on both92d and d74
compilers for Poisson count-zero means [1e-300,2,1] versus [2e-300,1,2]. An
independent C++23 exact-binary-rational analysis of the emitted input bits found
a negative representable mean near -3.3333333333333334e-301. Reversing row order
retained that nonzero contrast; the two-row exact-cancellation control returned0.
Analysis16997 froze exact-numerator/raw-bit predicates before decimal formatting;
the initial16996 auxiliary boolean classification is superseded by that analysis.
The raw native sources, outputs and executable identities remain preserved.

The new stdlib_scale_count_aggregate_cancellation fixture checks all six traced
and untraced aggregate entries, reuse, invalid-size clearing, detached snapshot,
public cache reset, reversed nonzero rows, mixed exact cancellation and pointwise
zero. count_aggregate_cancellation_reference.cpp independently uses the analytic
Poisson count-zero identity and exact rationals reconstructed from native bits;
its --before mode requires the original false-zero witness. New-source native
acceptance remains separate from preparation until the coordinated Slurm gate.
