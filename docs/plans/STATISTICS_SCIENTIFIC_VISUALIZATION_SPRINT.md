<!-- docs:meta
topic_id: repo.docs.plans.statistics-scientific-visualization-sprint
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.statistics-scientific-visualization-sprint
-->

# Sounio standard-library sprint: statistics, scientific visualization, mathematics, genomics and physics

## Sprint mandate and continuity — refreshed 2026-10-19

The requested scope remains a major implementation sprint across statistics,
scientific visualization, mathematics, genomics and physics. These domains are
implementation tracks. Statistics and visualization determine the initial shared
contracts; the other domains are not reduced to demonstration-only work.

The packaging, profiling and compile-once bundle work is supporting infrastructure.
It has reached its bounded checkpoint. New work prioritizes numerical kernels,
scientific contracts, render semantics and domain consumers.

| Track | Implemented / previously evidenced | Current and next work |
|---|---|---|
| Statistics | Raw MAD; direct t survival; df-aware t/F helpers; public one-sample/pooled/Welch/paired tests with 95% t intervals and statuses | Stable simple OLS, real deletion DFFITS/Cook, Student slope CI and unassessed statuses accepted; seeded robust percentile bootstrap accepted; checked exhaustive/Monte Carlo permutation accepted; checked mean BCa accepted; checked JB/Koenker/Shapiro-Wilk, median Brown-Forsythe, fitted-normal AD and checked KS accepted (60/60), with repaired shared F/beta extreme tails and removed legacy assumption p-to-Beta machinery; checked Pearson/Spearman and explicit inference accepted in77/77; labelled correlation permutations and seed-state repair accepted in80/80; shared rectangular QR and multiple regression accepted in83/83; weighted fits and tiny-projection QR repair accepted in86/86; conditional weighted covariance/Student inference accepted in90/90; checked offset-aware Poisson mean fitting accepted in101/101; OLS HC3 covariance accepted in105/105 ([receipt](HC3_COUNT_TSV_ACCEPTANCE_2026_10_19.md)); CR0/CR1S covariance and synthetic coefficient/SE consumer accepted in108/108, job15643 ([receipt](CLUSTER_COVARIANCE_EXECUTION_2026_10_19.md)); next weak-null inference and empirical calibration studies |
| Mathematics | Small/large sqrt scaling; direct small-angle sine; finite predicate; analytic Cauchy quantile | Checked information measures and channel helpers accepted in73/73; checked rectangular pivoted QR accepted in83/83; shared QR tiny-projection repair accepted in86/86; balanced scale restoration and repaired Student extreme/near-origin tails accepted in90/90; next condition estimation, large-argument trig and convergence policies |
| Scientific visualization | Explicit CI/prediction/credible bounds; inferential forest consumer; shared linear axes/ticks/labels; explicit row-count read; shared line viewport/clip and legend geometry | DFFITS-linked Canvas/SVG figure, automatic nice ticks, shared line domains, compact clipped SVG and explicit log/symlog accepted; conditional weighted-coefficient forest with independent geometry checks accepted in90/90; declared-correlation/standard-uncertainty Canvas/SVG consumer accepted in95/95; offset-aware count-rate/Pearson-residual consumer accepted in101/101; counts-to-QC consumer accepted in105/105 and clustered coefficient/CR0-CR1S SE consumer accepted in108/108; shared scatter projection and checked glyph clipping accepted in111/111, job15650 ([receipt](SCATTER_PROJECTION_EXECUTION_2026_10_19.md)); next nonlinear nice ticks, remaining other-mark clipping, layout and provenance |
| Genomics | Sequence/FASTA foundations; checked streaming two-pass TPM; shared streaming k-mers with exact integer counts, checked public adapter, idempotent profiles and stable JSD accepted in the64/64 closure | Checked anchored median-log-ratio reference-panel size factors and scalar count application accepted in95/95; offset-aware Poisson mean kernel accepted in101/101, separate from negative-binomial inference; incremental count TSV/QC/256x8 adapter and counts-to-Canvas/SVG consumer accepted in105/105, repeated on refreshed main in15633 ([receipt](HC3_COUNT_TSV_ACCEPTANCE_2026_10_19.md)); long-stream resource measurements remain open; PWM/motif/hashed embedding uncertainty remains unassessed |
| Physics | Correct Compton units and trigonometry, angular uncertainty and legacy adapter | Checked isothermal entropy and explicit standard-uncertainty bars accepted in69/69; declared two-volume covariance and calibration-cancellation consumer accepted in95/95; checked Lorentz Jacobian covariance and repaired legacy adapters accepted in101/101; next units and invariant-based validation |
| Execution support | Smaller runtime package; measured build/runtime separation; compile-once bundle | Bundle phase split recorded in job 15619; reuse that receipt, extend only when a library implementation is blocked |

This is a method-level ledger, not certification of every function in a track.
A source implementation, a native acceptance receipt, remote CI, and publication
remain distinct states. The current multi-domain batch has final acceptance in job 14799:
twelve tests, zero failures/skips, including the legacy Compton adapter and
the t-test-to-Canvas/SVG consumer. The following shared-axis batch has final
acceptance in job 14832: thirteen tests, zero failures/skips/errors, with a
controlled failing-before job 14831 (twelve passes, one failure).
See SCIENTIFIC_AXES_EXECUTION_2026_10_17.md. The line clipping/legend checkpoint
has final acceptance in job 14850: fifteen tests, zero failures/skips/errors,
including degenerate-point SVG semantics and a physical Compton consumer.
Controlled before job 14851 retains thirteen passes and fails the two new consumers.
See SCIENTIFIC_CLIPPING_LEGENDS_EXECUTION_2026_10_17.md.
The regression/influence checkpoint has final acceptance in job 14870: eighteen tests, zero failures/skips/errors. Controlled legacy determinant-policy job 14871 compiles but fails the small-scale numerical assertions (seventeen passes, one failure). See STATS_REGRESSION_INFLUENCE_EXECUTION_2026_10_17.md. Automatic scientific axes and shared line-domain binding have final acceptance in job 14879: twenty tests, zero failures/skips/errors. Controlled explicit-domain-policy job 14880 retains eighteen passes and fails the two new fixtures after successful compilation. See SCIENTIFIC_AUTO_AXES_EXECUTION_2026_10_17.md. Compact clipped SVG has final acceptance in job 14889: twenty-one tests, zero failures/skips/errors. Before renderer job 14890 retains twenty passes and fails only the new markup assertion. Independent C++23 comparison confirms exact canonical geometry, style, mask and audit equality; the dense 1024-point SVG decreases from 111053 to 18050 bytes, and gzip from 7379 to 4792 bytes. See SCIENTIFIC_SVG_PATHS_EXECUTION_2026_10_17.md. Explicit log10/symlog and checked logarithm have final acceptance in job 14905: twenty-three tests, zero failures/skips/errors. Controlled identity-transform job 14906 preserves twenty-one passes and fails both new fixtures after compilation. C++23 covers 51 transforms/round trips with explicit subnormal quantization bounds, exact Compton log/log projection and byte-identical legacy SVG comparisons. See SCIENTIFIC_LOG_SYMLOG_EXECUTION_2026_10_17.md. Seeded percentile bootstrap has final acceptance in job 14939: twenty-seven tests, zero failures/skips/errors. The untouched former bootstrap in job 14929 compiles and fails the seed-degeneracy assertion while retaining twenty-three prior passes. C++23 covers ten datasets including maximum capacity, extreme finite scales and signed seed limits; the same results feed Canvas and SVG with explicit 95% percentile semantics. The same visual review also repaired SVG forest/axis RGB preservation; controlled renderer-before job 14940 retains twenty-six passes and fails the new exact-RGB assertion. See STATS_BOOTSTRAP_EXECUTION_2026_10_17.md. Checked permutation inference has final acceptance in job 14974: thirty-two tests, zero failures/skips/errors. Fourteen independent C++23 cases cover full label enumeration, 48620 assignments, 1-by-255 and 255-by-1 groups, extreme values, seeded Monte Carlo counts and Wilson bounds. Old implementation job 14961 retains twenty-seven passes and fails the raw-unit tolerance regression after compilation. A Canvas/SVG consumer displays Monte Carlo tail-probability intervals with explicit budgets and meaning; the legacy example remains accepted. One index buffer is reused as the enumeration cursor. The effect oracle uses relative accuracy against the effect itself, including on a 1e15 common offset. Main rechecked at c4f1d3ce73987de9e2694af393711180f6a57fb1; its new website/docs changes do not touch this closure. See STATS_PERMUTATION_EXECUTION_2026_10_17.md. Checked mean BCa has acceptance in job 15063: thirty-seven tests, zero failures/skips/errors, including all previous thirty-two fixtures. Nine C++23 cases independently compute explicit jackknife deletions, bias/tie counts, adjusted probabilities and endpoints; 25 CDF and 11 inverse-normal checks cover the bounded normal kernel. The mean acceleration uses a scale-free moment identity without a jackknife buffer; ordinary percentile and BCa share draw/sort/variance logic. Controlled legacy numerical body job 15066 compiles and fails both large-value consumers while retaining thirty-five passes. See STATS_BCA_EXECUTION_2026_10_18.md. Next priority is explicit assumption assessment and continued genomics/physics numerical kernels; correlation/general-statistic BCa and coverage studies remain open. The master sprint remains in progress.

## Bundle phase attribution — job 15619, 2026-10-19

Job 15619 on cpu-ops measured the existing `build_stats_viz_bundle.sh` path
separately from execution for the three accepted Poisson/Lorentz fixtures.
The frozen v3 package was copied and not modified. Archive
`806aa97f5b459ce8ada29aeadd425801e43bb4aa6127a795f0b9cbb02287cb44` checked
before and after. Manifest
`d4bb6caab77912c1972e6713cb34569038c576d6ae874048fb45a18f4be1878a`.
Pinned Madaros
`125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f`.

Each phased `souc compile` ELF matched the official bundle ELF on three
repetitions. Stdout was byte-identical and every pass marker was present.
All 51 GNU-time rows exited 0. The job itself exited 127:0 after those checks
because the worker image has no g++. Medians come from the same C++23
summarizer, compiled off the worker. Elapsed 0.00 s is below the 0.01 s clock,
not zero cost. Phase peaks are separate process maxima and must not be added.

| Fixture | Compile median KiB | Execute median KiB | Bundle-build KiB |
|---|---:|---:|---:|
| stats_poisson_reference | 1802764 | 3192 | 1802252 |
| physics_lorentz_reference | 1492492 | 3184 | 1491980 |
| viz_poisson_offsets | 3323392 | 8192 | 3323404 |

Materialize was 3680–3760 KiB. Hashing was about 3200–3280 KiB. For these
fixtures the bundle-build peak is the compile peak. These numbers do not
explain or reduce batch MaxRSS 5277668K (job 15602) or 7062324K (job 15603).
No kernel-only measurement, memory reduction, or speedup is claimed. Numerical
sources and the 101/101 acceptance are unchanged.

This measurement is distinct from the compact-reference profile in jobs
15615/15616. That profile times direct ELF execution of a smaller fixture.
Job 15619 times the existing bundle builder and its internal phases.

## Previous checkpoint — compact validation and measured phases, 2026-10-19

The compact Poisson reference retains **101/101** native acceptance in Slurm15613,
zero failures/skips/errors,exit0:0,5:05. Snapshot
a79744bdd0a0adbe29654ed3c9c3b480cf500bb967cff2f3e22e9298e2ae12f0.
All stdlib .sio production sources are byte-identical to the previous checkpoint.
The26-case reference retains every expected bit and assertion/tolerance; the other
25 cases remain byte-identical outside the replaced maximum-shape block/comment.
Last-row255 corruption15614 compiles and fails its numerical guard.

Phase profiling15615 records21 accepted observations from three paired compile/run
trials plus a scientific visualization control, with compiler/ELF/source/receipt
hashes. The faulted profile15616 propagates numerical failures and has no readiness
marker. C++23 postprocessing rejects failed/incomplete/nonfinite/zero-RSS data.

Observed median compile CPU7.59s→5.38s; executable2150856→885192bytes; compile
RSS1800204→1698828KiB. Source170556→68952bytes. Direct-run wall stays0.03s.
Three pairs on one host are descriptive; wall ranges overlap. Whole101 elapsed
5:04→5:05 and batch peak7060932K do not demonstrate whole-suite speed/memory gains.
See [phase-profile receipt](STATS_VIZ_PHASE_PROFILE_EXECUTION_2026_10_19.md).

Active worktree remains /workspace/sounio-sprint-stats-viz-main2-20261019, main base
15a1983fd8d9bb4730f2fd6ca72e50f5e0c88d0f. Work is uncommitted. Next scientific
fronts remain robust/clustered covariance and bounded count-matrix/I/O, with
negative-binomial dispersion separate; use the new phase data for execution support.

## Historical checkpoint — Poisson offsets and Lorentz covariance, 2026-10-19

Forge/Slurm15603: **101/101**, zero failures/skips/errors, exit0:0,5:04.
The previous95 fixtures remain accepted. Main base15a1983fd8d9bb4730f2fd6ca72e50f5e0c88d0f;
active worktree /workspace/sounio-sprint-stats-viz-main2-20261019.
Snapshot9bb020718429749326b7e5229f60bbaa2ee7a97a240eecb0e8181bda021bc156.

Added checked offset-aware Poisson mean fitting with CPQR/Fisher scoring,
full-step/score convergence and stable local objective-change safeguarding.
The independent C++23 reference has26 cases. Added physical Lorentz Jacobian
covariance with38 independent quad cases; repaired legacy uncertainty bodies,
imports and private public-API types. The actual two-panel scientific SVG/Canvas
consumer has24 independently checked markers. Focus15602 passes6/6; actual
before controls15604/15605 compile and each fail exactly their discriminating
numerical fixture. Numerical assertions/tolerances were preserved.

The batch reports7062324K MaxRSS including compilation/execution; phase-specific
RSS attribution and empirical speedup remain open. Remote runtime packages are
reused and verified by complete manifests; selected review artifacts stay small.
No commit, push, PR, merge or repository-wide CI is claimed.
See [execution receipt](POISSON_LORENTZ_EXECUTION_2026_10_19.md).

## Historical checkpoint — counts and correlated uncertainty on current main, 2026-10-19

Final Forge/Slurm15457: **95/95**, zero failures/skips/errors, exit0:0,4:49.
All90 previous fixtures remain accepted. Snapshot:
aed2fdcacb365c8b71c7fbfbcf7f39fc2586f8578e51190902889a9139b9109c.
The active worktree is now /workspace/sounio-sprint-stats-viz-main-20261019,
based on remote main54f73df0f2242d43d310f46d4534688e52cd8eb7. The original
f5b2c585 worktree is preserved. Port audit:259 files byte-identical,298
upstream changes with zero overlapping paths/collisions; upstream's three
stdlib changes are retained. Changes remain uncommitted; native acceptance
is distinct from repository-wide CI or merge.

Genomics now has checked median-log-ratio normalization from a supplied
reference panel, up to256 features and8 samples, with explicit geometric-
mean-one factors and scalar count application. Integer-difference anchoring
preserves tiny close-i64 log contrasts before conversion. One shared sort
and two256-element scratch buffers are reused; composition assumptions
remain unassessed. The independent quad C++23 oracle has33 cases, including
3-sample even-median discrimination and relative-only tiny-contrast checks.

Physics adds first-order isothermal entropy with a supplied initial/final
volume correlation,32 independent quad cases, correct sensitivity signs and
cancellation before norm scaling. The accepted independent API is preserved
byte for byte; rho0 delegates exactly. The actual Canvas/SVG consumer labels
standard uncertainty, units and declared correlation without implying a CI.
Its analytic checker accepts the final artifact and rejects altered position,
radius and missing-marker copies.

Main-based pre-anchor control15458:94/95,exit1:0,4:47, failing only the new
genomic reference after native compilation; one source file differs from the
candidate. Focused15450:5/5,0:17. Injected omitted-correlation15453:3/5 and
upper-median15454:4/5 fail the intended numerical consumers. Original-base
95/95 in15451 is retained separately from the final main-based acceptance.

Runtime45MB, current-main packet populated from only seven different files
plus metadata after full-manifest comparison. Selected-source custody and
all native runs retain ADR009/Forge boundaries; no measured RSS/speed claim.
See [receipt](COUNTS_COVARIANCE_EXECUTION_2026_10_19.md),
[normalization](../../stdlib/genomics/NORMALIZATION.md) and
[correlated uncertainty](../../stdlib/physics/ENTROPY.md).
Next: count-matrix I/O/offset-aware count regression, reviewed Lorentz-force
uncertainty, condition diagnostics and robust/calibration work. Statistics and
scientific rendering remain prioritized within all five domain tracks.

## Previous checkpoint — conditional weighted inference, 2026-10-19

Final Forge/Slurm15430: **90/90**, zero failures/skips/errors, 4:35.
Snapshot a457d3bb0f04bcc961d3a0df4a0f421c3f529404e465edb3a67a69c7858f358c.
All86previous fixtures are retained. Focused15429:5/5,0:26, including the
previous extreme-tail fixture alongside all four new consumers.

The public weighted_linear_inference returns covariance, SE, two-sided
Student p-values and pointwise confidence intervals only under the caller's
explicit independent-Gaussian relative-precision model. Model assessment
remains false; coefficients, covariance and inference have separate validity
statuses. Confidence support is[0.5,0.999999]. The implementation reuses the
same QR factorization, uses compensated triangular solves and balanced binary
restoration, and maps covariance back to original column order.

C++23 quadmath30cases cover scale/permutation/weight invariance,256x8,
subnormal covariance and unavailable collapsed intervals. Five independent
C++23 checkers were freshly compiled and executed; regenerated fixture bytes
match the frozen source. Actual Canvas/SVG forest intervals have checked
geometry, RGB, model labels and SE labels; three corrupted SVG copies are
rejected. Native figure XML is valid and the full preview was inspected.

Student df1 overflowing-square tails and near-origin argument rounding are
repaired. The first full candidate15421 passed89/90 and exposed cancellation
atdf100,t10; it was rejected. Complementary beta is now restricted to the
near-origin region, retaining direct tails farther out. Actual previous-tail
control15422 passes89/90 and fails only the new extreme-tail fixture after
compilation, retaining every prior fixture; its manifest differs from the
final candidate in exactly student_t.sio.

Source buffers are bounded; no measured speed/RSS superiority is claimed.
The standalone runtime package is45MB; only selected sources/evidence are
copied to the Mac. Source archive manifests match their frozen packages.
Remote main rechecked118c1baff75be12db934952307f8696dc59aba4f with no overlapping
stdlib changes from baselinef5b2c585. This remains isolated-snapshot acceptance,
not full-current-main execution, CI, commit, push, PR or merge.

See [execution receipt](STATS_WEIGHTED_INFERENCE_EXECUTION_2026_10_19.md)
and [contract](../../stdlib/stats/regression/WEIGHTED_INFERENCE.md).
Next: genomic covariate/count consumers and covariance-based physical
uncertainty propagation; robust/clustered inference, condition diagnostics
and empirical calibration studies remain open within the five-domain sprint.

## Previous checkpoint — weighted QR and calibration, 2026-10-19

Final Slurm15405: **86/86**, zero failures/skips/errors,4:17.
Snapshot2a510936c41cea08b2a698d35009a77ca049c6628451722b5d941b28d5861003.
All83prior fixtures pass. Focused15404:3/3,0:13.
Actual previous-Householder-top control15406:84passes/two new numerical
runtime failures after compilation,4:20; all83prior fixtures pass and only
one source file differs.

Checked WLS borrows explicit stride8 designs and positive finite weights.
It scales columns/response, uses sqrt(w)/sqrt(maxw), delegates pivoted QR,
restores coefficients and reports separate raw/normalizedweighted residual
norms. The variance model remains unassessed; SE/p/CI/covariance are open.
Review exposed tiny-projection cancellation in shared QR. Separate tail-dot
top updates repair it without changing rows or the critical weights.

C++23 quadmath40cases cover extreme/global-scaled weights,256x8, near
dependence, subnormals, metadata and a normalized weighted norm near1e-7
despite raw norm1e308. A real Canvas/SVG consumer compares OLS/WLS on a
synthetic calibration, with24markers checked independently.
One checker point lies on a pixel boundary: propagating the SAME numerical
reference budget admits two adjacent pixels only there, with23remaining
positions strict. Corrupted position/count/RGB copies are rejected.
Frozen Sounio is unchanged by this post-snapshot checker correction.

See [execution receipt](STATS_WEIGHTED_QR_EXECUTION_2026_10_19.md).
Next: covariance/inference under explicit weight/variance assumptions,
condition diagnostics, genomic covariate/count consumers and physical
uncertainty propagation. Five-domain scope and memory/artifact constraints
remain active; no runtime/RSS superiority or whole-stdlib certification.
Remote main118c1baf is rechecked; work remains uncommitted.

## Previous execution checkpoint — rectangular QR and multiple regression, 2026-10-18

Final Slurm15383: **83/83**, zero failures/skips/errors,4:04.
Snapshot139c6410ed92624ffd497af561fcc14a86e972eaccaa1bd5b6e7fbae73c05924.
All80prior fixtures pass. Initial adapter/figure gate15382:2/2,0:12.
Absolute-cutoff fault control15384:82pass/1reference failure after successful
compilation,4:03; it retains all80previous fixtures and changes one solver file.

New borrowed column-pivoted Householder QR handles rectangular designs
p<=8,m<=256 with explicit stride8, column equilibration, recomputed LASSQ
norms and rank/arithmetic states. A public stats::regression::multiple adapter
shares the kernel. Intercepts are explicit; no SE/p/CI or noise model is inferred.
Rank cutoff/diagonal ratio are declared heuristics, not an SVD or condition
certificate. Rank-deficient designs return no partial solution.

C++23 quadmath uses twice-projected modified Gram-Schmidt for28cases,
including256x8, ill-conditioned columns, mixed/extreme scales, subnormals,
custom rank thresholds and a coefficient restoration path whose naive
intermediate underflows. A two-sensor calibration produces actual Canvas/SVG
with observed-vs-fitted and residual panels. Independent C++checks all16
SVGmarkers, numeric coefficients, counts, units and colors. Final and focused
SVGs are identical. Job times include compilation; no kernel-speed/RSS
superiority is claimed. Remote main refreshed to118c1baf without touched-code
overlap; isolated worktree remains uncommitted, with no full-main CI claim.

See [execution receipt](STATS_MULTIVARIABLE_QR_EXECUTION_2026_10_18.md).
Next: positive finite weighted least squares, statistical covariance/inference
contract, genomic covariate/count consumers and physical uncertainty propagation.
The major five-domain sprint continues.

## Previous execution checkpoint — correlation permutations, 2026-10-18

Final Slurm15363: **80/80**, zero failures/skips/errors,3:53.
Snapshot d4a1b246fe2d1ad98f7648038e6f7749a36294d87dc69e072a32a2c27cde93d5.
All77 prior fixtures remain accepted. Control15364 restores the previous
bootstrap body:79pass,1 safe seed-regression failure,3:54.
Fault-injection control15365 omits rejection sampling:79pass,1 independent
permutation-reference failure,3:55. Both compile successfully and preserve77.

Checked Pearson/Spearman correlation permutations now support complete labelled
enumeration n<=8 and bounded seeded MC n<=256/B<=100000. Independent C++23
quadmath covers31cases, including tied multiplicity,40320permutations, forced
RNG rejection, signed seed limits and extreme values. Preparation/ranking is
shared once; two f64buffers and one reused index buffer have6144byte array
payload, excluding backend/scalar overhead. No measured speed/RSS claim.

The negative seed class that previously normalized to absorbing zero is fixed
in percentile/BCa bootstrap and means permutations as well. Other valid streams
are preserved. The new Canvas/SVG consumer distinguishes corrected MC p from
Wilson95 precision for q=hits/B and shows the exact conditional reference.

Exchangeability remains unassessed. General zero-correlation weak-null,
studentized/cluster/blocked inference and coverage studies remain open.
Pearson uses a declared relative floating tail policy; exhaustive labels do
not imply exact-real Pearson arithmetic. See
[execution receipt](STATS_CORRELATION_PERMUTATION_EXECUTION_2026_10_18.md).

Next shared kernel: checked rectangular column-pivoted Householder QR for
multivariable regression, genomic covariates and physics calibration, followed
by weighted fits, independent numerical references and scientific consumers.
The major five-domain sprint remains in progress and uncommitted.

## Previous execution checkpoint — checked correlation and scatter, 2026-10-18

Final Slurm **15353**: **77/77**, zero failures/skips/errors,3:45.
All previous73 fixtures pass. Snapshot
`a3c51c76a49708d247d38760ad1e8903c2a87062ae096a9a49020a721039034e`.
Raw-centering control15356:76pass+1 numerical runtime failure,3:45.
Prior-SVG-scatter control15357:76pass+1 markup failure,3:45.
Each control retains all73 previous fixtures and differs at one source file.

Checked Pearson and average-rank Spearman validate size/finite/constant domains,
use anchored scaled centering and compensated moments, and return explicit
projection residuals independently of rounded endpoint coefficients.
Heapsort replaces selection sorting; rank buffers and index storage are reused.
Inference validity, p/CI availability and calibration are explicit.
Pearson normal-model and Spearman t-approximation requests require caller opt-in.
Public256/fixed100 adapters share the checked kernels; result layouts expand
with metadata, so external binary/literal consumers need corresponding updates.

C++23 quadmath checks58 estimates and6 inference cases, including extreme range,
large offset/ULP spread, ties, maxcapacity and near-perfect positive residuals.
Student reference is scoped to df1/df3 closed forms; Fisher CI is approximate.
The early reference emitter's df type error was fixed without numerical changes.
Focused15352 passes all four new fixtures before the final77 run.

Actual SVG readback uncovered a renderer defect: six observations produced a
count95 comment and spurious markers through copied-record projections.
Only vh_emit_scatter was changed to typed scalar reads from the original scene.
Independent C++23 geometry rejects before and accepts after, checking six
circles, exact positions/radius/RGB. Canvas checks both axis and marker pixels.
The real scatter labels descriptive Pearson/Spearman and matched-tie overplot.

Main remains2b41c82671995278554214398b7caac01b89e524; isolated source stays
unrebased/uncommitted/unpushed. No measured speedup/RSS or full-corpus/CI claim.
Next: correlation resampling/permutation and remaining legacy consumers;
genomics I/O/count and physics covariance remain active in this same sprint.

[Execution receipt](STATS_CORRELATION_EXECUTION_2026_10_18.md),
[coefficient contract](../../stdlib/stats/CORRELATION.md),
[inference contract](../../stdlib/stats/CORRELATION_INFERENCE.md).

## Previous execution checkpoint — checked information theory, 2026-10-18

Final Slurm **15320**: **73/73**, zero failures/skips/errors,3:29, retaining
all previous69 fixtures. Snapshot
`65a557e2211528cfae1e8f77ccf9d6634c3fbd3b33c7a1aaef1d68cef3497c55`.
Support-loss control15321:70pass+3 numerical failures,3:29.
BSC-endpoint control15322:69pass+4 numerical failures,3:28.
Both controls compile the failing cases and preserve every previous69 fixture.
Each manifest differs at only its one controlled source file.

Checked Shannon/collision/min entropy, KL and Jensen-Shannon use real logarithms,
explicit normalized finite-vector domains and typed infinite KL support.
Scalar BSC/AWGN/Pinsker/reciprocal-Fisher helpers have explicit units and domains.
All nine compatibility names delegate to these checked APIs. Positive local
series, compensated sums and DBL_MIN-scaled aggregation protect near equality
and subnormal contributions. Capacity at both deterministic BSC endpoints is1.

C++23 quadmath supplies171 expected results, including one-ulp divergences
around1e-32 and seven-bin minimum-subnormal aggregates. Initial focused15316
exposed per-bin Shannon rounding and a rounded BSC complement. Both were fixed
without changing tolerance/reference; focused15318 passed all four new fixtures.
A129-point BSC figure feeds production Canvas/SVG with shared axes and bits/use.

Independent review completed. Capacity is explicitly eight categories; no
arbitrary-length, correctly-rounded or measured runtime/RSS claim is made.
Remote main remains2b41c82671995278554214398b7caac01b89e524; original isolated
baseline remains unrebased, uncommitted/unpushed. No merge/CI claim.

Next: correlation/rank inference and scientific diagnostic consumers, beginning
with guard/finite/domain/source audit, then broader resampling calibration.
Genomics/physics/math tracks remain active in the same major sprint.

[Execution receipt](MATH_INFORMATION_EXECUTION_2026_10_18.md),
[probability contract](../../stdlib/math/PROBABILITY.md),
[channel contract](../../stdlib/math/CHANNELS.md).

## Previous execution checkpoint — checked isothermal entropy, 2026-10-18

Final Slurm **15308**: **69/69**, zero failures/skips/errors,3:18.
All prior64 fixtures remain accepted. Snapshot
`bbb01e3567ec617a0e2461d4b1d4857cceaaf0bbf8690164f655b2b2faadbb1f`.
Equal-volume uncertainty-loss control15309:65pass+4 numerical failures,3:19.
Prior-SVG control15310:67pass+2 missing-metadata failures,3:18.
Both controls preserve all64 previous cases and differ at one source file only.

Checked ideal-gas entropy uses stable close-volume logs, scaled binary products,
a scaled uncertainty norm, explicit domain/arithmetic statuses and independent
first-order standard-uncertainty metadata. Equal nominal volumes retain their
nonzero volume sensitivity. The tuple adapter delegates to it. Typed vertical
+/-u bars export standard-uncertainty, level0, without a coverage interpretation.

31 independent C++23 long-double cases cover finite extremes, subnormals,
underflow policy, domain errors, unit invariance and uncertainty scaling.
A production Canvas/SVG figure uses the same outputs. Native preparation caught
and repaired dropped leading render imports, an unresolved unused units import,
legacy dynamic-global constant loads and the pinned parser's e-324 rounding to0.
The six-case focused round15307 passed before the final69-case run. Independent
source review confirms intended diffs and preserved previous bodies/imports.

Remote main advanced to2b41c82671995278554214398b7caac01b89e524;
the new compiler/docs/two GUM fixture changes do not overlap this closure.
The original isolated baseline remains unrebased; pinned-compiler acceptance
does not certify a fresh compiler build. No commit/push/PR/merge/CI claim.
Scalar-state/no-heap entropy is implemented; no measured speedup/RSS claim.

Next implementation: information-theory logarithm and domain/support contracts,
then broader statistical calibration and scientific layouts. All domain tracks
remain active in this same major sprint.

[Execution receipt](PHYSICS_ENTROPY_EXECUTION_2026_10_18.md),
[scientific contract](../../stdlib/physics/ENTROPY.md).

## Previous execution checkpoint — streaming k-mers and profiles, 2026-10-18

Final restored-source Slurm **15270** passes **64/64**, zero failures/skips/errors,
elapsed3:13, retaining every prior60 fixture. Snapshot
`2ea420b3b89b648a3aee4f98ec34008e00150526461d2934eba64d3bcba7c89d`.
Loss-of-carry control15275:62 pass+2 numerical runtime failures,3:15.
Repeated-normalization control15274:63 pass+1 numerical runtime failure,3:14.
Failing cases compile successfully; both controls preserve the earlier60 cases.

A shared fixed256-bin i64 stream processes canonical bases in O(n), preserves
windows across chunks and breaks on ambiguous bases. Bounds, sticky failures,
counter overflow and empirical-frequency availability are explicit. Public
core and4096-base profile adapters share it. Normalization is idempotent;
checked Jensen-Shannon distance handles raw/normalized data with a stable
positive near-equality formulation and canonical logs/sqrt.

C++23 independently reenumerates windows for72 unique snapshots, all12 splits
of an ambiguous sequence, and a1280-base exhaustive-word stream. Every bin and
scalarstate field is checked. A separate long-double direct-KL oracle validates
near-equal JSD; native fixtures check maximum profiles, states, overflow and
existing valid legacy examples. This is bounded kernel acceptance, not all
genomics/motif/uncertainty validation.

An independent reviewer caught an omitted legacy file suffix during an early
text composition. Root restored it, checked exact suffix preservation apart
from declared status/guard/comment edits, and reran on finalv2. Preliminary
15268/v1 is explicitly not acceptance of the restored library.

Three subagents delivered the stream/contract, numerical reference and profile
fixture/review; root integrated, repaired, gated and retained evidence.
No measured runtime/RSS improvement claimed: wider counters and adapter scratch
are disclosed. Remote main recheckeda184446537; isolated uncommitted source
retained; no commit/push/PR/merge/CI claim.

Next implementation: checked isothermal entropy and explicit standard-uncertainty
propagation, followed by information-theory logarithm/domain contracts.
Statistics, scientific rendering, mathematics and domain consumers remain the
same major sprint.

[Execution receipt](GENOMICS_KMER_EXECUTION_2026_10_18.md),
[stream contract](../../stdlib/genomics/KMER.md),
[profile/adapter contract](../../stdlib/genomics/KMER_PROFILES.md).

## Previous execution checkpoint — checked KS and parallel domain audit, 2026-10-18

Final Slurm **15249**: **60/60**, zero failures/skips/errors,3:07.
All prior57 fixtures pass. Snapshot
`5ec5b6c95e41be1fe142b89a28f91b30da7bb956db66c5ce23298246d1cd697a`.
Independent controls: old tie progression15250 passes58 and fails two numerical
fixtures after compilation; old alternating-tail policy15251 passes59 and fails
the numerical reference after compilation. Both elapsed3:03 and retain the
previous57 fixtures. Neither control is an untouched historical module.

Checked KS validates input/reference domains, groups pooled ties before ECDF
comparison, uses exact rank products, shared heapsort and bounded scratch.
Normal-reference parameters must be independently fixed. Stable theta-CDF/
direct-survival expansions replace fixed-truncation cancellation at smalllambda.
The output labels its approximate continuous-null calibration; tied-data D is
computed, but p validity for ties/discreteness/dependence remains unassessed.

Subagents delivered a22-case C++23 raw-erfc/brute-ECDF oracle, the actual Canvas/
SVG consumer and contract, independent core review, and a read-only cross-domain
audit. Root reviewed/integrated the core and fixtures, ran Forge/native acceptance,
and re-executed the C++ report/fixture comparison. No local Sounio build used.

The next implementation front is a shared checked streaming k-mer counter,
followed by isothermal entropy/uncertainty and information-theory logarithm/domain
contracts. These remain implementation work, not completed by source audit.
Performance claims remain limited to structural complexity/storage observations
until runtime/RSS measurements are available. Remote main recheckeda184446537;
isolated uncommitted baseline retained, no push/PR/merge/CI claim.

[KS receipt](STATS_KS_EXECUTION_2026_10_18.md),
[method contract](../../stdlib/stats/KS.md), and
[next domain audit](SCIENTIFIC_KERNEL_AUDIT_2026_10_18.md).

## Previous execution checkpoint — fitted-normal Anderson-Darling, 2026-10-18

Final Slurm **15241** passes **57/57**, retaining every prior53 fixture,
zero failures/skips/errors. Snapshot
`95261771888eb655be64299a258e1a55ee377b4bf06b2671ead914f0b6826169`.
Control **15242** restores only the former erf/complement tail policy:
53 previous pass + four new numerical runtime failures after compilation.
Both jobs elapsed2:51. This is focused native acceptance, not whole-library CI.

Checked AD uses scaled compensated fitted-normal moments, direct small tails,
paired logarithms and shared256-value heapsort. One data scratch vector is reused;
no second log-tail buffer or heap allocation. Approximate Stephens p is restricted
to n>=8 and Astar<10; statistic remains available outside the calibration domain
for n=3..256. Typed availability, tie metadata and a checked public legacy adapter
replace invalid p=1 and toy confidence. Unused p-to-Beta/moment/extern machinery
has been removed from the assumption adapter module.

C++23 independently evaluates18 datasets using raw long-double moments and erfc;
published p coefficients are shared, not independently proven. Real Canvas/SVG
ECDF/fitted-normal consumer has computed A2/p and explicit approximate/no-band
labels. QuickLook's square thumbnail crops the right edge; XML bounds of the
standalone SVG remain inside the viewport.

Remote main rechecked/fetched at
`a1844465373c1ae5a335de2d0a73819e5cec2230`; changes since e00 in inspected
stdlib/test/binary paths do not touch this runtime closure. The original isolated
baseline and uncommitted changes are preserved; no merge/push/PR/CI claim.
Next: checked KS reference-null/tie/size contracts and Kolmogorov tails.
Math,genomics,physics and measured runtime/memory/artifact improvements remain
implementation tracks in this same sprint.

[Execution receipt](STATS_ANDERSON_DARLING_EXECUTION_2026_10_18.md)
and [method contract](../../stdlib/stats/ANDERSON_DARLING.md).

## Previous execution checkpoint — Brown-Forsythe and shared tails, 2026-10-18

Final Slurm **15230** passes **53/53**,retaining all previous48 cases,
in **00:02:42**. Control **15231** restores former bucket p and exponential
helpers:48 prior cases pass,and all five new fixtures compile then fail their
numeric assertions. This is a policy/helper control,not an untouched old algorithm.

Two-group median-centered Brown-Forsythe now uses scaled compensated deviations,
a real F statistic/survival and explicit approximate-reference/status metadata.
The legacy adapter delegates and drops fabricated p buckets/Beta conversion.
Two buffers are reused and finite-prefix ordering is shared with Shapiro and
legacy code. F/beta exponentials reuse math::pure::exp, removing a false tail
floor and preserving positive binary64 subnormals.

Independent C++23 raw long-double ANOVA and positive quadrature cover15 datasets,
36 F tails,six analytic tail validations and a beta-underflow case. The synthetic
Canvas/SVG consumer displays deviations and their means in common units, with
computed F/p and explicit statistical meaning. SVG XML/preview inspected.
See [receipt](STATS_BROWN_FORSYTHE_EXECUTION_2026_10_18.md) and
[contract](../../stdlib/stats/BROWN_FORSYTHE.md).

Main remains e00a5d537cd0c6d01762c38fcd70cc9739062710; canonical active checkout
untouched; all changes uncommitted. No full compiler/CI/merge acceptance.
Next statistical method work is AD/KS calibration and their remaining Beta
heuristics. Shared scientific layout/nonlinear ticks/other-mark clipping,
mathematics/genomics/physics kernels and real domain I/O/consumers remain active,
as do execution/memory/artifact objectives. Storage/sort/helper consolidation
does not establish a measured speedup or whole-stdlib/SOTA certification.

## Previous execution checkpoint — Shapiro-Wilk and Q-Q figure, 2026-10-18

Final Slurm **15211** passes **48/48**, retaining all prior44 cases,
in **00:02:27**. Controlled former p buckets **15212** compile successfully:
45 pass,three numerical p assertions fail. The control restores only the former
bucket policy on the new W,not the untouched former implementation.

Royston normal-order coefficient corrections and three calibration branches
replace the old pedagogical Shapiro weights/p buckets. A compensated normalized
projection norm retains1-W; heapsort uses fixed100+50 buffers, without a heap
or covariance matrix. Computational availability, ties and approximate p are
explicit. The public legacy adapter delegates without Beta conversion, and
the existing OLS assessment adds a separate Shapiro result.

C++23 independent normal-quantile arithmetic and raw vector identities cover
24 samples and one independently fitted model; mathematical coefficients and
calibration formulas are shared published inputs, not independently proved.
The synthetic n=20 Canvas/SVG Q-Q consumer has calculated W/p, normal Blom
positions and a fitted visual guide. The figure explicitly says approximate p
and unassessed residual calibration. Domain3..100, uncensored complete finite
samples; no regression-specific type-I calibration or blanket method certainty.
See [receipt](STATS_SHAPIRO_EXECUTION_2026_10_18.md) and
[contract](../../stdlib/stats/SHAPIRO_WILK.md).

Main remains e00a5d537cd0c6d01762c38fcd70cc9739062710; worktree and canonical
active checkout boundaries are preserved. All changes remain uncommitted.
Next assumption procedure: real median-centered Levene/Brown-Forsythe,
then remaining AD/KS heuristics and their p-to-Beta contracts.
Scientific layout/nonlinear tick policies/other-mark clipping and domain I/O/
consumers remain active alongside mathematics,genomics,physics and execution/
memory/artifact optimization. No measured speedup follows from this source-level
storage/sort improvement.

## Previous execution checkpoint — Koenker and model residual figure, 2026-10-18

Final Slurm **15196** passes **44/44**, retaining all previous 40 cases.
Controlled former tail-policy Slurm **15197** has 41 passes and three numerical
p-value assertion failures after successful native compilation. Both complete
in 00:02:06. The control restores only the former bucket assignment, not the
entire legacy function; the receipt records this boundary.

The checked studentized Koenker procedure uses scaled compensated scalar passes,
LM=n*auxiliary R-squared and direct chi-square(1) survival. It stores no
squared-residual vector or auxiliary OLS arrays. A single-OLS model consumer
returns separate fit/Jarque-Bera/Koenker statuses, and the legacy BP path
delegates to the checked core. Other private assumption heuristics remain
uncertified. No posterior, small-sample calibration or measured speedup claimed.

Independent C++23 covers 12 datasets, 12 tail points and a complete model case.
The new Canvas/SVG figure displays actual squared normalized residuals and
auxiliary fit from an explicitly synthetic n=20 model, with calculated LM/p
and asymptotic method labels. Final preview and XML are inspected.
See [execution receipt](STATS_KOENKER_EXECUTION_2026_10_18.md) and
[contract](../../stdlib/stats/HETEROSCEDASTICITY.md).

Main remains e00a5d537cd0c6d01762c38fcd70cc9739062710. All changes uncommitted.
Next assumption work: genuine Shapiro-Wilk coefficients/calibration; replace
remaining Levene/AD/p-to-Beta heuristics under truthful availability contracts.
Shared scientific layout/nonlinear tick policies/other-mark clipping and real
domain I/O/consumers continue alongside math, genomics and physics kernels.
Execution, memory and artifact objectives remain active across those tracks.

## Previous execution checkpoint — checked moment normality, 2026-10-18

Final Slurm **15177** passes **40/40**, retaining every previous 37 case.
The existing Jarque-Bera now uses scale-safe compensated central moments,
typed unavailable states and explicit asymptotic metadata. Twelve independent
C++23 long-double cases cover unit changes, offsets, subnormals, opposite-sign
extremes and n=256. Untouched-body control **15178** retains 37 passes and
fails the new common-field unit-invariance assertion after successful native
compilation. Initial control **15173** separately times out on large values.
See [execution receipt](STATS_NORMALITY_EXECUTION_2026_10_18.md).

Remote main is rechecked at **e00a5d537cd0c6d01762c38fcd70cc9739062710**.
The c4f1d3ce..e00a5d53 advance changes compiler/self-hosted/CI/site/docs,
not stdlib or the pinned committed Madaros. Existing worktree and previous
receipts are preserved; this is not validation of a newly rebuilt compiler.

Next implementation: scaled Koenker/Breusch-Pagan auxiliary regression with
direct chi-square survival and explicit statuses; real regression consumer and
scientific figure. Genuine Shapiro-Wilk calibration remains a separate block.
Legacy p-to-Beta assumption heuristics remain uncertified and are labeled so.
The mathematics/genomics/physics implementation tracks and execution, memory,
artifact objectives remain active. This numerical block has fixed bounded
storage; it makes no measured performance or whole-stdlib completion claim.

### Required continuation order

1. Preserve completed multi-domain, axis and clipping/legend receipts
   (jobs 14799, 14832 and 14850).
2. Reuse the accepted linear axis, line-viewport/clip and legend path for domain
   consumers. Automatic nice ticks and snapshot axis-domain binding are accepted. Explicit log/symlog transforms and checked domains are accepted. Remaining render work includes nonlinear automatic tick policies,
   live domain tracking and other-mark clipping. Compact clipped SVG paths are accepted.
3. Replace data-independent regression diagnostics with defined statistical
   procedures and documented domains.
4. Build distribution/regression/count/physics figures from real input paths.
5. Expand domain kernels, streaming/aggregation and numerical boundary coverage
   using the same contracts, rather than creating parallel library stacks.

### Preserved execution receipts

- MAD and explicit interval forest: STATISTICS_VISUALIZATION_EXECUTION_2026_10_16.md
- Inferential tails: STATISTICS_INFERENTIAL_TAILS_EXECUTION_2026_10_16.md
- Resource plan: STATS_VIZ_RESOURCE_OPTIMIZATION_2026_10_17.md
- Direct interval construction: STATS_VIZ_DIRECT_INTERVAL_EXECUTION_2026_10_17.md
- Compile-once reuse: STATS_VIZ_BUNDLE_EXECUTION_2026_10_17.md
- Multi-domain batch: STDLIB_MULTIDOMAIN_EXECUTION_2026_10_17.md
- Shared scientific axes: SCIENTIFIC_AXES_EXECUTION_2026_10_17.md
- Line clipping and legends: SCIENTIFIC_CLIPPING_LEGENDS_EXECUTION_2026_10_17.md
- Stable regression, influence and scientific consumer: STATS_REGRESSION_INFLUENCE_EXECUTION_2026_10_17.md
- Automatic axes and shared line domains: SCIENTIFIC_AUTO_AXES_EXECUTION_2026_10_17.md
- Compact scientific SVG paths: SCIENTIFIC_SVG_PATHS_EXECUTION_2026_10_17.md
- Checked log10/symlog and scientific consumers: SCIENTIFIC_LOG_SYMLOG_EXECUTION_2026_10_17.md


## Scope and baseline

Primary focus: reliable statistical computation connected end-to-end to scientifically faithful, reproducible figures. Mathematics, genomics and physics are implementation domains as well as cross-domain consumers; statistics and visualization retain initial priority.

Baseline: remote `main`, commit `f5b2c5850be9f3d3995719ffcade44a6d851658a` (worktree prepared detached at this commit). The active workspace branch is separate and dirty; this plan does not include its changes.

The initial planning phase ran no compiler builds or tests. Subsequent execution is recorded in the linked checkpoint receipts above.

## What already exists on main

- `stdlib/viz` has a fixed-capacity Visual IR, CPU Canvas renderer, HTML/SVG export, chart builders, interaction state, replay, hashes, audit receipts, and workbench composition.
- `stdlib/graphics` already contains raster/SVG primitives and chart-oriented modules, including histograms, boxplots, error bars, heatmaps, ECDF, hexbin, and violin-related examples.
- `tests/run-pass` has 41 `viz`/`graphics` smoke programs. `tests/stdlib` has 3 tests under `viz` and 1 under `graphics`.
- `stdlib/stats` has 163 Sounio source files and 14 tests under `tests/stdlib/stats`.
- Public surfaces have constraints that need a design audit: fixed capacities in the Visual IR, a `vars` field on series data, fixed-size arrays in several plot functions, overlapping `viz`/`graphics`/`plot` entry points, and placeholder markers in `graphics/svg.sio`, `graphics/lib.sio`, and `plot/bar.sio`.
- `stats/IMPLEMENTATION_STATUS.md` marks broad areas complete while `stats/regression/linear.sio` still includes placeholder DFFITS and linearity diagnostics. Status documents must be reconciled with source and executable evidence.

The sprint should extend the existing Visual IR and renderer where they meet the contracts below. It should not start with a parallel plotting stack or treat smoke count as proof of scientific correctness.

## Product contract

A scientific figure must preserve the link:

`input data + analysis method + assumptions + uncertainty semantics -> chart specification -> deterministic renderer -> auditable artifact`

The figure and its metadata must let a reader answer: what data and rows were used, which transformation/statistic was applied, what the error bars or band mean, which units and scales are shown, and how to reproduce the rendering.

## Workstreams and priority

### P0 — Baseline audit and maturity ledger

Create a function-level inventory across `stats`, `viz`, `graphics`, and `plot`:

- classify every targeted public function as placeholder, partial, implemented-unverified, or evidence-backed;
- record consumers, capacity limits, effects, input/output contracts, assumptions, source locations, tests, and known language/backend constraints;
- trace each existing plot path from statistical result to IR node to Canvas/SVG/PNG/PPM output;
- identify duplicate APIs and select one canonical path for new scientific charts;
- verify which named smoke cases are check-only, executable, artifact-producing, or visual demos.

**Exit:** a reviewed gap matrix. Every P1 backlog item links to source and existing coverage, and names the missing evidence.

### P1 — Statistical core suitable for figures

1. Define result records for estimates, standard errors/variances, confidence intervals, tests, diagnostics, and provenance. Keep SD, SE, confidence intervals, and prediction intervals distinct in both types and names.
2. Define policies for missing/non-finite values, zero variance, ties, insufficient sample sizes, weights, and failed convergence. Invalid conditions must be observable in result status/diagnostics, never silently converted into plausible values.
3. Audit and complete descriptive statistics, distribution CDF/quantile paths, correlation/effect sizes, hypothesis tests, bootstrap/permutation intervals, regression diagnostics, multiple-testing procedures, and survival outputs in dependency order.
4. Replace known placeholders, starting with DFFITS and linearity diagnostics in `stats/regression/linear.sio`; add domain and degenerate-data handling before expanding method count.
5. Use deterministic seeding for resampling and simulation. Record method, assumptions, sample size, missing-data policy, seed where relevant, and implementation version.

**Acceptance:** published formulas/contracts; passing deterministic and edge-case Sounio coverage; independent C++23/Sounio numerical agreement; no Python, Rust, or Julia numerical reference; Lean 4 for theorem claims only. Numerical tolerances and supported ranges are documented per method.

### P2 — One scientific chart specification over the existing Visual IR

Extend the current IR with a typed scientific chart layer rather than proliferating renderer-specific plotting APIs. Keep all data/analysis meaning in Sounio; renderers consume the same validated chart representation.

The specification should cover:

- data series, grouping/facets, labels, units, categorical and continuous channels;
- mark types, scales (linear/log/symlog where justified), domains, ticks, clipping, legends, annotations, and coordinate systems;
- explicit interval semantics (SD, SE, CI, prediction interval, credible interval) and point/line/ribbon linkage;
- missing values and gaps in line series;
- deterministic ordering and style assignment;
- validation diagnostics for incompatible marks, scales, domains, units, or uncertainty data.

First audit whether `VizSeriesData.vars` means variance, standard deviation, or a general uncertainty payload. Replace ambiguous fields at the semantic layer with explicit types; preserve compatibility through a deliberate adapter if needed. Audit current fixed capacities before selecting any expansion or chunking design.

### P3 — Rendering correctness and publication outputs

Use the existing CPU Canvas and SVG renderers as the reference path. Close functional gaps and ensure both renderers represent the same chart semantics.

- Complete axes, tick placement/formatting, labels, legends, clipping, interval bands/bars, markers, and annotation handling across supported chart types.
- Complete or explicitly narrow placeholder SVG/export paths; ensure XML escaping and structural correctness for labels and metadata.
- Add accessible figure titles/descriptions and meaningful labels; retain scene/node identity in exported output for auditability.
- Provide perceptually appropriate sequential, diverging, and cyclic color maps, with accessible alternatives. Do not use rainbow palettes as the scientific default.
- Ensure export is deterministic for identical inputs/specification/version. Treat GPU rendering as an optimization only after CPU-output parity is demonstrated.
- Produce SVG and raster artifacts; include a metadata sidecar or equivalent receipt with source/data digest, chart specification, analysis method, parameters, units, missing-data policy, interval meaning, seed, and renderer version.

### P4 — Statistical figure gallery and integration

Build full vertical examples that start from data, compute results in Sounio, render, export, and inspect the artifact:

1. **Distribution:** raw points + ECDF + histogram; make bin policy and sample count explicit.
2. **Estimation:** estimate with confidence/credible interval and sample-level observations; label interval meaning.
3. **Regression:** fit, uncertainty band, residual/QQ/influence diagnostics, and failure/assumption warnings.
4. **Group comparison/meta-analysis:** effect sizes and forest plot with study-level intervals and model/provenance.
5. **Survival and time series:** censoring-aware survival curves or time-series estimates with valid uncertainty bands.
6. **Large-data view:** scatter overplotting handled by hexbin/density/aggregation, with aggregation parameters disclosed.

Then add a genomics count/variant figure and a physics uncertainty or field figure as cross-domain consumers. Those examples must use real input/output paths, not fabricated constants presented as domain analysis.

### P5 — Scalability and regression evidence

- Establish performance baselines from representative sizes before setting numeric targets. Measure analysis, layout, and raster/vector rendering separately.
- Add chunked/streaming or aggregation strategies only where measurements show fixed capacities are a real constraint; define truncation/overflow behavior explicitly.
- Add deterministic numerical tests, metamorphic/invariant tests, renderer structure tests, and integrated end-to-end tests from statistical analysis through artifact export.
- Compare SVG structure and renderer scene hashes; use fixed small raster fixtures for geometry/color regression and disclose renderer/version sensitivity.
- Keep existing component smokes. Add integrated tests that assert statistical values and inspect the corresponding chart nodes, interval semantics, labels, and exported metadata.

## Initial figure/API backlog

| Priority | Item | Main source surface | Acceptance focus |
|---|---|---|---|
| P0 | Maturity ledger and source-to-renderer trace | `stats`, `viz`, `graphics`, `plot` | Every targeted API classified with evidence gap and canonical path |
| P1 | Typed statistical result and interval contracts | `stats` result types and `viz` payloads | SD/SE/CI/prediction interval cannot be confused |
| P1 | Remove regression diagnostic placeholders | `stats/regression/linear.sio` | DFFITS and linearity outputs derived from data and edge-tested |
| P1 | Statistical distribution and resampling audit | `stats/distributions.sio`, `bootstrap*.sio`, `permutation.sio` | Stable domains, deterministic seeds, edge cases and C++23 agreement |
| P2 | Chart spec validation and uncertainty marks | `viz/chart.sio`, `viz/ir.sio` | Same typed spec lowers to Canvas and SVG without loss of meaning |
| P2 | SVG/export placeholders | `graphics/svg.sio`, `graphics/lib.sio`, `plot/bar.sio` | No advertised placeholder API; exports are deterministic and structurally valid |
| P2 | Axes, scales, ticks, labels, facets | `viz` renderer and chart layer | Correct transforms/labels including log domains and units |
| P3 | Distribution + estimate + regression gallery | `stats` + `viz` examples | End-to-end data-to-figure examples with visible uncertainty and assumptions |
| P3 | Forest plot and survival integration | `stats` + `viz` | Study/censoring semantics preserved and auditable |
| P4 | Large-data aggregation and performance | `viz`, `graphics` | Measured limits, reproducible benchmark and no silent point loss |
| P4 | Genomics/physics consumers | `genomics`, `physics`, `particle_physics` | Scientific quantities render with units, provenance, and uncertainty |

## Sequencing and checkpoints

Treat the proposed eight-week schedule as an execution envelope, not a promise independent of compiler/API constraints.

- **Checkpoint A — audit:** baseline matrix and API/render trace approved before broad implementation.
- **Checkpoint B — statistical slice:** one robust estimation method produces a typed result, uncertainty-aware figure spec, Canvas output, SVG output, and reproducibility receipt.
- **Checkpoint C — figure gallery:** distribution, regression diagnostics, and effect/forest figures use that same path.
- **Checkpoint D — hardening:** cross-domain consumers, performance evidence, docs/status corrected to match executable evidence.

Each checkpoint reports exact source commit and compiler identity. Full native/compiler validation follows the project's remote/Slurm workflow; no claim is inferred from source inspection, check-only output, smoke names, or an old status report.

## Decision rules

- Fix root causes and shared contracts before adding isolated chart/statistic functions.
- Do not rewrite `viz` wholesale; first establish whether the existing IR can carry the required semantics and identify precise gaps.
- Add APIs only when their scientific contract and consumer are clear.
- Prefer mathematically and perceptually faithful output over decorative effects.
- Keep incomplete methods visible as incomplete; do not label a model or renderer ready for production use without its evidence.

## P0 audit refinement

The source-level findings and exact classifications are recorded in [`STATISTICS_VISUALIZATION_BASELINE_AUDIT.md`](STATISTICS_VISUALIZATION_BASELINE_AUDIT.md). They refine the initial inventory: PNG encoding already exists in pure Sounio despite stale module documentation; the SVG placeholder is one disabled advanced figure; and the `plot/bar.sio` placeholder comment is in an integer-format helper whose body delegates to `print`, so it is not evidence that the whole chart is unimplemented. The audit also confirms that the Visual IR renderer interprets its series payload as variance and draws ±1σ, which must not be presented as a 95% interval without explicit bounds.

## Execution checkpoint — 2026-10-16

Completed the first statistics-to-rendering slice and corrected the approximate
MAD body. Final Slurm job 14665 passed three selected tests on the exact source
snapshot 793a279b7bbe77acb6688a8b3fcf6ee07a6315b6231b123e1e5223e4e6b95146.
C++23 independently validated the three Student-t interval fixtures and raw MAD
cases. Canvas geometry, SVG interval metadata, and legacy chart builders passed.
Changes remain uncommitted.

[Execution receipt](STATISTICS_VISUALIZATION_EXECUTION_2026_10_16.md) records
the bounded acceptance and artifacts. The next implementation front is accurate
inferential tails and their caller contracts, followed by regression diagnostics
and the scientific chart specification. Content digests, axes/ticks/labels,
publication layouts, and domain galleries remain required for full acceptance.

## Implementation continuation — HC3 and incremental counts, 2026-10-19

After the published101-test checkpoint, two experimental modules were added:
OLS HC3 covariance via existing QR factors, and incremental nonnegative-i64 count
TSV with transactional QC and a checked256x8 normalization adapter. Both pass
native typecheck in Slurm15622. The independent C++23 HC3 reference passes syntax
checking only. Numerical concordance, executable consumers, corpus inclusion and
a new scientific figure remain pending; earlier101/101 does not cover these additions.
See HC3_COUNT_TSV_IMPLEMENTATION_2026_10_19.md for source identities and boundaries.

## Continuation checkpoint — clustered OLS covariance, 2026-10-19

The HC3/incremental-count implementation-only note above is superseded by
[HC3/count acceptance](HC3_COUNT_TSV_ACCEPTANCE_2026_10_19.md):105/105 in15625
and refreshed-main repeat15633. The new bounded statistical slice is single-way
CR0/CR1S covariance using checked QR factors, signed group labels and compensated
within-group influence sums before outer products.

Independent C++23 quad covers31cluster cases, including extreme units,
partition/relabel/permutation behavior, representability and nonzeroG2,p8
maximum-shape coverage. CR1S is the explicit fullG/(G-1)*(n-1)/(n-p) convention;
no clustered t/p/CI or few-group calibration is implied. Fixed buffers and
shared Gram code are resource contracts, not a new measured RSS/speed claim.

Corrected native v2 Slurm15643 accepts108/108 with zero failures/skips/errors,
retaining all105 prior entries. The real SVG passes an independent six-marker
checker; position/count/radius mutations each exit2. Corpus phase212s and
overall215s are run observations, not a paired speed claim. V1 Slurm15637
remains preserved as107/108 with the missing-increment figure timeout.
Wrong-partition15638 and omitted-correction15639 controls compile then fail
numerically. See [cluster execution receipt](CLUSTER_COVARIANCE_EXECUTION_2026_10_19.md)
for source/snapshot/SVG identity and exact scope.

This checkpoint continues the mass implementation sprint; it does not complete
statistics, scientific visualization, mathematics, genomics or physics.
Conditioning/convergence policies, empirical inferential calibration, scientific
render semantics, count-domain consumers, physical units/invariants and measured
execution/memory/artifact work remain explicit priorities. Packaging and profiling
support those kernels and their consumers. Broader CI, merge/publication and
scientific validation remain separately reported states.

## Continuation checkpoint — shared scatter projection, 2026-10-19

Following accepted108/108 cluster covariance, the next bounded implementation
repairs a shared scientific rendering defect: constant/singleton scatter centers
differed between Canvas and SVG, and glyphs could escape the plot viewport.
Both renderers now use a finite checked projector. Scatter auto bounds use paired
finite rows; nonempty all-invalid input fails transactionally. Legacy glyph
shapes remain; the opt-in raw-linear viewport uses clipped radius3 disks/circles
with closed-center-domain classification and explicit row counters.

Independent C++23 covers64 projection cases, with byte-identical regenerated
fixtures. The real previous-renderer control15649 compiles/runs but fails its SVG
center expectation:actual(40,110),expected(140,70). Full111-entry native15650 accepts111/111 with zero failures/skips/errors,
retaining all108 previous entries. Scissor-removal control15651 compiles then
fails the selected viewport fixture0/1, zero skips.
See [scatter execution receipt](SCATTER_PROJECTION_EXECUTION_2026_10_19.md).

This continues rendering semantics within the broader statistics, mathematics,
genomics and physics sprint. Nonlinear nice ticks, other-mark clipping,
conditioning/convergence, calibration and measured resource work remain open.

Focused entries100/101/102 pass (singleton,64-case reference,viewport); the final
C++23 checker accepts the actual five-circle figure and its boundary clip
annotation. Center/clipwidth/group-removal SVG mutations each exit2. Scissor
control15651 compiles384functions then fails the viewport fixture0/1 in7s,
zero skips. Root rebuilt C++23 references warning-free with generatedfixturecmp0.
Final15650 summary passes three layers:111/111, zero failures/skips/errors.
Scientific and boundary SVG XML checks both exit0. Corpus340s and overall341s
are whole-run observations; Slurm elapsed5:42, TotalCPU11:18.137 and batch
MaxRSS7152996K describe the two-worker compile/run job, not renderer performance.
The final source snapshot370cc50246991bc39421b4530b8ec55b10b3dbd67cc2102ab710e1c055cf009e
precedes final documentation edits. Broader CI/merge remain separate states.

## Continuation checkpoint — signed bars and artifact offload, 2026-10-20

Main d7077ef1 is incorporated. Baseline Slurm15712 accepts111/111; final
Slurm15718 accepts114/114 with zero failures/skips/errors after complete HTML
recovery and a margin correction that preserves scientific minus signs.
Both renderers share checked bar rectangles, actual x positions, a true zero
baseline and whole-footprint clipping. Independent C++23 covers83 geometry/width
cases; the actual native SVG passes8-bar and15-explicit-tick checks. Previous-SVG
15713 and horizontal-intersection-removal15715 controls compile then fail their
intended expectations. See [bar execution receipt](BARS_EXECUTION_2026_10_20.md).

The authorized107-snapshot offload passed two-copy restoration and an A/B journal
audit; complete encrypted custody is on GitHub, keys held separately. Including
the earlier one-file step,1262940884 logical bytes left the hot quota. Quota stays
3TiB. Availability returned to96468992bytes, but validation snapshots reduced it
to4194304bytes: further reviewed retention is required for a massive ongoing
batch. See [offload receipt](ARTIFACT_OFFLOAD_EXECUTION_2026_10_20.md).

Statistics, mathematics, genomics and physics remain in scope. Conditioning and
convergence, inferential calibration, count/variant consumers, physical units and
invariants, and measured execution/memory/artifact work continue. This is a
bounded acceptance checkpoint; broader compiler CI and merge remain separate.

## Continuation checkpoint — count dispersion on refreshed main, 2026-10-20

The approved exact59 source offload completed with 59 recovery indices and
10,733,223,936 bytes available at the final measurement; the quota remains3TiB.
See [archival execution](SOURCE59_ARCHIVE_EXECUTION_2026_10_20.md).

A new isolated sprint branch incorporates remote main b10c046e7b and retains
all accepted increments through the signed-bar114/114 and archival checkpoints.
Production NB2 supplied-means moment dispersion is ported from9f8b8617aa with
narrow underflow corrections and explicit arithmetic/status limits. A count/offset
Poisson diagnostic consumer connects the statistics to scientific visualization.
This increment is implementation plus static review, with no new tests added or
run and no new native/figure/corpus acceptance implied.
See [implementation receipt](NB_DISPERSION_IMPLEMENTATION_2026_10_20.md).

The wider statistics, mathematics, genomics, physics and execution/memory/artifact
sprint remains active. Maximum-likelihood NB fitting, shrinkage, inferential
calibration, native figure acceptance and measured resource comparisons remain
open; the moment diagnostic does not stand in for those methods.

Current-source typecheck follow-up: Slurm15747 completed0:0 after rebuilding
the seed and modular Madaros; both NB2 module and new count-diagnostic example
returned TYPECHECK_OK. The transport label corpus_smoke represents only this
package-specific check dispatcher, not a corpus run. No tests or example
execution occurred. [Exact receipt](NB_DISPERSION_TYPECHECK_2026_10_20.md).

## Continuation checkpoint — sampled uncertainty and native SVG, 2026-10-20 UTC

Remote main085da is incorporated. The uncertainty builder now validates finite
means/X and nonnegative finite variance before mutation, includes mean±sigma in
its Y domain, and shares checked sampled-strip/center geometry between Canvas
and SVG. Direct renderers validate array prefixes before access. SVG now emits
the uncertainty marks it previously omitted. The contract remains sampled±1sigma,
with no confidence-interval or continuous-ribbon claim. Known production/demo
consumers propagate required Panic effects.

Slurm15776 completed0:0: eighteen selected files typecheck, and the native SVG
example compiled/executed with exit0. The actual complete SVG parsed as XML and
was visually inspected with legible axes, preserved minus sign and explicit
unitless synthetic caption. The compiler came from a current-source build15772,
preserved and hash-verified before scratch cleanup, then reused after proving
unchanged bin/scripts/self-hosted/stdlib input trees. Failed effect/axis-spacing
attempts are retained. [Execution receipt](SAMPLED_UNCERTAINTY_EXECUTION_2026_10_20.md).

No tests were added or run for this slice; historical114/114 is unchanged.
Numerical concordance, Canvas runtime/cross-backend comparison, calibration,
renderer latency/RSS comparisons, full corpus/CI and merge remain open.
The final checks/compile/figure job took26s; this is a whole-job observation with
compiler reuse, not a renderer benchmark. Source geometry adds no lower/upper IR
buffers and bounds Canvas traversal before loops. Shared quota remains3TiB,
with no further removals; compiler custody/prepared packets stay on t560 ext4.

Statistics, scientific rendering, mathematics, genomics and physics remain the
mass-sprint scope. Next concrete candidate: stable NB2 count diagnostics with
supplied mean/dispersion, explicit Poisson limit and representability statuses,
reusing the count/offset consumer and existing genomics input path. This candidate
has not been implemented or numerically accepted by the current checkpoint.

## Continuation checkpoint — scaled NB2 count diagnostics, 2026-10-21

The main8 branch retains main085da and all prior increments. New scalar
nb2_count_diagnostic supplies Poisson/NB2 Pearson residuals from count means and
supplied dispersion, with independent staged variance availability. Scaling
avoids first constructing the variance; alpha0 copies the Poisson residual.
Count conversion is bounded to0..2^53. The binary64/status contract explicitly
limits rounding and representability claims. No heap allocation is introduced in
the scalar kernel. Deviance and likelihood/inferential outputs remain deferred.

The synthetic count/offset consumer compares both residuals on one explicit axis,
retains observed/fitted exposure-adjusted rates, and labels the same-data moment
alpha as plug-in. It reuses its two existing1024-element scratch arrays; failed
availability paths now return1 instead of a successful exit with no figure.

Slurm15783 completed0:0 in7:08 after a current-source seed/Madaros rebuild, four
selected typechecks, native compilation and complete example execution. Canvas
rendering completed in memory and actual SVG was emitted. The complete1380x620
sips rendering of the extracted SVG was inspected with both panels, legends,
units, negative labels and caveats visible. Slurm15785 executed the same attested
ELF under GNU time:0.01s elapsed,8748KiB peak native-process RSS, exit0. This is
one eight-observation whole-program observation, not isolated kernel performance,
a scaling study or comparative superiority. Whole-job compiler/transport RSS is
recorded separately. [Execution receipt](NB2_DIAGNOSTICS_EXECUTION_2026_10_21.md).

Raw build logs/input manifests are retained losslessly compressed with original
hashes; source packets and large compiler custody remain remote. The quota stays
3TiB, without new removals. No tests were added or run. Numerical concordance,
explicit boundary runtime coverage, model calibration, Canvas pixel comparison,
full corpus/CI and merge remain open; historical114/114 is unchanged.

The larger statistics/scientific-rendering/mathematics/genomics/physics sprint
continues. Stable deviance/logarithmic primitives, genomic count consumers,
physical-unit/invariant work and measured resource comparisons remain in scope.

## Continuation checkpoint — shared checked logarithmic primitives, 2026-10-21

Main9 retains remote main085da and all prior increments. Three shared scalar
primitives now provide checked log1p, log1pmx and (1+x)log1p(x)-x with explicit
domain/availability/sign contracts and compensated factored near-zero series.
Four private bodies were removed across information theory, channels and Poisson.
The scalar helpers add no arrays or explicit heap allocation. Numerical precision
certification is a separate open requirement.

Slurm15823 rebuilt the current-source seed/Madaros and completed0:0 in7:13.
Slurm15825 completed0:0 in25s after the example caption explicitly disclosed seven
samples, reusing the exact hash-verified compiler after unchanged compiler/input
trees were proved. Both jobs ran seven typechecks and compiled/executed two real
SVG examples. Final native-process peak RSS was1576KiB for the logarithmic example
and8768KiB for NB2/Poisson. Time reporting was0.00s/0.01s respectively;0.00s is
centisecond resolution, not zero time. These whole-example observations are not
comparative kernel benchmarks. Both complete1380x620 SVG renderings were inspected.
[Execution receipt](LOGARITHMIC_EXECUTION_2026_10_21.md).

Evidence includes exact source/compiler/ELF/stdout identities, lossless compressed
logs/manifests, native receipts and complete figures. Quota remains3TiB, without
new removals. No tests were added or run. Independent numerical concordance,
boundary runtime coverage, calibration, full corpus/CI and merge remain open;
historical114/114 is unchanged.

Next concrete cross-domain migrations: sf_log_ratio in genomics normalization,
preserving exact i64 count-anchor before f64 conversion, zero exclusion,
median/centering and far log-difference; entropy_log_ratio in physics, preserving
exact equal-volume zero, scaled products, covariance and n=0 semantics. These
candidates were audited read-only and remain unimplemented. Statistics,
scientific rendering, math, genomics, physics and execution/memory/artifact costs
remain the continuing mass-sprint scope.

## Continuation checkpoint — genomics and physics share log1p, 2026-10-21

Main10 retains main085da and previous sprint increments. The two audited
cross-domain migrations are now implemented: sf_log_ratio near branch delegates
to the shared checked logarithm while preserving integer subtraction before f64
conversion; entropy_log_ratio delegates its close-volume branch while preserving
equality, far log differences, scaled products, covariance and n=0 behavior.
Two private series bodies were removed; existing public contracts/statuses remain.

The real synthetic SVG example calls median-ratio normalization and both physical
entropy uncertainty models, reusing two plotting buffers. Its panels are independent
domain demonstrations. Three eligible genes/one excluded row, categorical sample
order, seven volume ratios, supplied rho=0.5 and report-only first-order standard
uncertainties are disclosed. Full-scale/report formatting does not resolve every
tiny contrast and is not an accuracy certificate.

[Native execution and artifacts](DOMAIN_LOG1P_EXECUTION_2026_10_21.md) record frozen
source ab10a22026, fresh compiler build, exact selected targets, process resources,
figure inspection and acceptance limits. No tests were added or run. Quota stays
3TiB with no new removals; compiler custody also lives on t560 ext4. Independent
numerical concordance, boundary runtime coverage, calibration, comparative resource
studies, full corpus/CI and merge remain open.

Next bounded statistics candidate: supplied-parameter NB2 saturated deviance and
signed deviance residual per count, separately from fitting alpha or regression.
Current nb2_diagnostics exposes Pearson/variance only. A future scalar contract
must address near-mean cancellation, alpha->0 without overflowing1/alpha,
large alpha*mean, large-alpha cancellation and distinct representability of a
contribution versus its square root. Preserve zero-count, equality and exact
Poisson-limit branches. No gamma engine or renderer changes are required by the
identified scope; no inference calibration or implementation is claimed here.
Statistics, scientific rendering, math, genomics, physics and execution/memory/
artifact efficiency remain the mass-sprint scope.

## Continuation checkpoint — conditional NB2 deviance, 2026-10-22

Remote main advanced to06f927dbca5753f5d7b6f64c29eb745eff77f351; it was fetched
and merged, retaining earlier sprint work including genomic/physical shared-log
consumers. New scalar nb2_deviance supplies signed saturated-deviance residuals
and independent squared-deviance availability for supplied count mean/dispersion.
The local branch uses a compensated factored series; the distant branch uses a
positive Poisson/correction decomposition and combines component roots by scale.
No arrays or explicit heap are introduced in the kernel. Input/status guards
preserve the2^53 count bound and reject unsupported intermediate regions instead
of turning them into zero. Independent static review identified and the source
addressed subnormal amplification and missing canonical sqrt Panic effects.

The separate real SVG example compares Pearson NB2 against signed deviance at nine
counts for supplied mean4/alpha0.25, with common dimensionless residual axis and
zero reference. Supplemental parameter scenarios report availability without
assertions. This is not a fitted-model or calibrated inferential result.
[Execution receipt](NB2_DEVIANCE_EXECUTION_2026_10_22.md) records final source,
initial cancelled build, fresh corrected compiler build, selected checks/native
execution, complete figure inspection, exact hashes and per-process resources.

No tests or independent numerical oracle were added or run. Numerical concordance,
boundary corpus, global accuracy, model calibration, comparative resources, full
corpus/CI and merge remain separate. Quota remains3TiB with no new removal;
large compiler custody remains remote with verified ext4 copy. Statistics,
scientific rendering, math, genomics, physics and execution/memory/artifact costs
remain the continuing mass-sprint scope. Next numerical hardening candidate is
balanced products for the deliberately unavailable far subnormal-product region;
this requires independent concordance before widening the advertised contract.

## Continuation checkpoint — balanced tiny-product NB2 arithmetic, 2026-10-22

Main12 retains remote main06f927 and all earlier increments. The previously
unavailable distant tiny-product region now evaluates mean*(1+alpha*count) before
normalization, avoiding amplification of rounded alpha*mean by count. If the
intermediate m1 is subnormal, its logarithm is carried independently from the
original mean and checked log increments rather than taken after coarse rounding.
Other status guards and independent residual/squared-deviance availability remain.
Independent static review accepted the bounded formula/source, without a global
accuracy, correct-rounding or numerical-concordance claim.

The existing formatter now prints available residual/deviance payloads with15
significant digits and gates both payloads independently. Unavailable values are
text, not error-placeholder zeros. No points, parameters, geometry, buffers,
scene nodes or renderer changes were added. Fifteen-digit formatting is not a
binary64 roundtrip or an independent numerical reference.
[Execution receipt](NB2_BALANCED_EXECUTION_2026_10_22.md) records frozen source,
five selected checks, actual native report/figure and whole-process resources.

No tests or numerical oracle were added or run. Independent C++23/Sounio
concordance, boundary coverage, calibration, global error, comparative resources,
full corpus/CI and merge remain open. Shared quota remains3TiB with no new removal.
The statistics/scientific-rendering/math/genomics/physics/resource sprint remains
active. Cache dependency-manifest narrowing is an audited optimization candidate,
not implemented: current sprint dispatch explicitly disables cache for fresh
build provenance, and existing broad cache keys include unrelated stdlib/docs.
Any future reuse must first preserve a complete proven compiler-input manifest.

Final readback advanced main to7d67f3a944 (README-only), incorporated after the
native run. Merge input-tree comparison preserves bin/scripts/self-hosted/stdlib/
examples relative to the evidence closure; executed source remains eeeab3025c.

## Continuation checkpoint — observational compiler paths and cold-storage layout, 2026-10-22

Main13 retains mainf02db32 and earlier scientific work. A new standalone non-root
observer captures selected pathname/process/cwd syscalls, failed lookups and broad
source before/after inventories, with independent traced-command/collector states,
private raw custody and lossless archive readback. It never narrows the cache key
or authorizes reuse. Endpoint equality does not establish transient immutability
or a complete content-input closure. Open events from tree-hash utilities must be
distinguished from compiler import accesses in future analysis.

Slurm15940 completed0:0 in7:18: fresh instrumented source-tracking seed/Madaros
build, captured_endpoints_equal, empty tracer diagnostics and existing NB2 example
typecheck/compile/native execution. Native ELF/stdout identities match the preceding
NB2 balanced run. Native process RSS1052KiB/time0.00s is one whole-example observation;
traced whole-job timing is not an uninstrumented benchmark.
[Execution receipt](COMPILER_INPUT_OBSERVATION_EXECUTION_2026_10_22.md).

Existing Forge snapshot reuse was used, with a separately hashed collector/recipe
overlay and one run-local runner command delta. Compiler source remains the known
eeeab302 snapshot; collector source is3860980. This is not a fresh compiler built
from the collector Git tree. No new source archive was made. The new shared run
contains110568 apparent bytes of orchestration metadata; traces/compiler/packages
use scratch and private t560 ext4. Trace15888844 raw bytes is preserved in383405
lossless gzip bytes, with readback. Quota remains3TiB; no historical removal.

No tests or independent numerical oracle ran. Cache/build scripts and no-cache
policy remain unchanged. Next scope is process-image attribution of compiler reads
versus orchestration/hash accesses, with explicit decoding/negative-lookup gaps;
not cache activation. Generated transient seed identity and external inputs need
binding before any complete-input claim. The broader statistics, scientific
rendering, mathematics, genomics, physics and resource sprint remains active.

## Main14 — process-attributed pathname analysis (2026-10-22)

Continuation is based on remote main a59930e514, branch
`sprint/compiler-path-attribution-main-a59930-20261022`. The prior scientific
branch remains preserved at5c84f28288; merging its47 unique commits with main's
31 unique commits produces real source conflicts. New compiler/seed changes in
main require fresh native validation after that integration. Later logarithmic,
NB2, bars and uncertainty work remains queued, with the precise conflicts and
upstream fixes listed in the execution receipt below.

Delivered: C++23 bounded analyzer of the existing Slurm15940 trace. Successful exec
request epochs separate operator-listed compiler path hints from sha256sum and
other utility hints. It reconstructs598 split calls across25292 physical lines,
with zero reconstruction/decode failures for this capture. Unknown-image1210
syscall rows remain explicit. Final role rows: compiler1906, sha256sum15177,
other6106, unknown1210. These are syscall counts, not consumed dependencies or
cost measurements. There is no image inheritance or cwd/dirfd resolution.

Actual standalone analysis on t560: exit0, elapsed0.44s, maxRSS3252KiB. Private
raw tables and deterministic gzip archives have matching decompression hashes.
Only compact summaries and integrity metadata enter Git. No new compiler build,
Sounio execution, test, source snapshot, deletion or quota change in this increment.
The cache key and guards remain broad; cache_authority=false and
complete_input_closure=false.

[Execution and integration queue](COMPILER_PATH_ATTRIBUTION_EXECUTION_2026_10_22.md)
retain exact source/binary/trace identities, reviewed parser scope and open gates.

## Main15 — reconciled scientific source with current main, 2026-10-23

Source19c684a99bb5728c15b424a13fd190742efad718 merges the later scientific branch
with remote main3ef3dbaa6e and retains Main14's path-attribution tools/evidence.
The scientific integration queue in Main14 is closed at this source checkpoint:
shared logs, genomic normalization, entropy, NB moment/Poisson/NB2, signed bars,
sampled uncertainty, rendering effects and packaging selection are reconciled.
Main's explicit beta_new import, Shapiro sw_exp wrapper and Boltzmann direct loop
are preserved. Compiler/bin sources equal incorporated main; source identity is
not inferred from old compiler receipts.

Fresh Slurm15992 COMPLETED0:0: canonical modular build,23 typechecks and five
real native examples all pass. All five actual SVGs parse with xmllint and have
full-aspect PNG visual inspection, including an independent second review.
Build402.87s/maxRSS736380KiB; whole-job elapsed7:37/batchMaxRSS3588712K. These
are separate observations, not a comparative improvement claim. Canonical
compilerSHA238c7e210d82cdd8474f68c488a0aed2b22d3157f64536703f03e1d995dce734.

Attempt15990 failed before compiler execution because Forge's */data exclusion
removed stdlib/data. Exact selected-packet tar + extracted-manifest verification
and --reuse-snapshot resolved transport for15992; failure evidence is retained.
Heavy artifacts remain private ext4, source snapshot is reused for retry, quota
stays3TiB and no historical removal occurs. No test suite or numeric oracle ran.
Complete corpus, CI, numerical concordance, calibration and merge remain open.

[Completed integration receipt](SCIENTIFIC_MAIN_INTEGRATION_EXECUTION_2026_10_23.md)
contains parent/source comparisons, all execution statuses, native resources,
figure limits, custody and storage identities.
