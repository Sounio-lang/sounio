<!-- docs:meta
topic_id: repo.docs.plans.counts-covariance-execution-2026-10-19
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.counts-covariance-execution-2026-10-19
-->

# Count normalization and correlated physical uncertainty — 2026-10-19

## Source, port and acceptance boundaries

The sprint was ported to a fresh detached worktree based on the remote main
observed at54f73df0f2242d43d310f46d4534688e52cd8eb7:
/workspace/sounio-sprint-stats-viz-main-20261019.
The previous /workspace/sounio-sprint-stats-viz remains preserved atf5b2c585.
No rebase, reset, commit, push, PR or merge was performed.

Pre-port audit found259 unique authorized paths,298 upstream changed paths,
zero overlap and zero untracked collisions. All259 ported files were compared
byte for byte. The old worktree status and selected-file hashes remained
unchanged. The three upstream stdlib changes in claim_meet.sio,
nelder_mead.sio and polynomial/lib.sio retain exact new-HEAD contents.
The final ledger/receipt/README edits are later documentation, separate from
that pre-documentation port manifest and the frozen runtime snapshot.

Port manifest SHA256:
a5df3e07fe92c9a0b1bb6f7f58f47d35ebf42d38294ae775f3e84f6548614fd0.
Scoped patch SHA256:
bd4df3b24befe0e73a6022e9cabad21d8389a72df0de61c25b8402946596395a.

Committed Madaros gzip and resolvers are unchanged across these bases. The
changed legacy lean_single ELF/seed receipt remain upstream and are excluded
from the runtime package. The canonical workspace's local build is excluded.
Pinned gzip feaaf7fe181f1ee3efd30e7fe23b1709937ea7e93b52e812276fb093b6ccf52a;
decoded ELF125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f.
All native Sounio validation uses Forge/Slurm, uid/gid1000, clean environment.
Method-level95-fixture acceptance is separate from a full compiler corpus,
repository-wide CI and integration into the main branch.

## Genomic reference-panel normalization

genomics::normalization::median_ratio_size_factors accepts borrowed raw i64
counts with physical stride8,1..256 supplied features and1..8 samples. It
validates every active count before excluding any feature containing a zero.
No pseudocount or automatic sparse-data method is introduced. Empty eligible
reference populations return a typed unavailable result.

The estimator takes per-sample medians of eligible log ratios, averaging the
two middle LOG values for an even reference count. It explicitly centers
the sample log factors to mean0, so factors use geometric-mean-one scaling.
This centering convention and selected reference-panel contract are explicit;
there is no claim of default-output parity with the complete DESeq2 pipeline.
Primary algorithm source: https://raw.githubusercontent.com/thelovelab/DESeq2/devel/R/core.R.

Small log contrasts are preserved by anchoring each feature at its first
sample, subtracting positive counts in i64 before conversion, and evaluating
close log ratios with a32-term log1p series. Other ratios use canonical
ln_checked. A private out-of-domain sentinel marks exclusions; no third
feature buffer is required. The implementation reuses shared sort256 and two
256-element f64 scratch arrays across samples.

size_factor_normalized_count applies a selected scale one count at a time,
allowing application beyond the reference panel without a returned dense
matrix. Its header/index/count/factor validation precedence is documented.
Scientific composition assumptions remain unassessed. This normalization
does not estimate dispersions, fit a count likelihood or perform differential
expression tests. Scalar normalized counts are floating-point quantities;
preserving a tiny log factor does not make every large normalized integer
exactly representable.

The independent C++23 reference uses raw integers and quad113-bit logarithms,
ordinary sums and std::sort. Its33 cases cover depths, composition outliers,
zero exclusion, invalid-count precedence, permutation/global scaling, maximum
shape, i64 maxima and two close-i64 contrast cases. The small nonzero log
contrasts use relative1e-10 checks with no absolute floor that would admit0.
The dedicated even-median discriminator uses2features/3samples; independent
self-checks reject both arithmetic-ratio midpoint and upper-log median.

## Correlated physical standard uncertainty

physics::entropy::isothermal_entropy_correlated_volumes adds one declared
rho for initial/final volumes, retaining independent amount of substance.
The existing independent implementation is byte-preserved. rho0 delegates
to it exactly and changes only uncertainty_model to2. All results retain
first_order=true. Amount is in mol, entropy/standard uncertainty in J/K,
volumes share a common unit, and rho is dimensionless in[-1,1].

For positive sensitivity magnitudes cn,cf,ci, the volume covariance carries
a negative sign because the volume derivatives oppose each other:

    u^2 = cn^2 + cf^2 + ci^2 - 2*rho*cf*ci.

The implementation forms cf-ci before norm scaling, restores the cross
amplitude by balanced products, and combines cn,difference,cross with a
scaled norm. Perfect shared-volume cancellation therefore retains an
otherwise tiny amount contribution. The range of rho makes the two-volume
correlation block admissible without an unchecked general PSD backend.
The declared correlation is not estimated or assessed. GUM5.2 supplies the
first-order correlated-input law:
https://www.iso.org/sites/JCGM/GUM/JCGM100/C045315e-html/C045315e_FILES/MAIN_C045315e/05_e.html.

Statuses distinguish nonfinite1, physical domain2, uncertainty3, arithmetic4
and rho range5; errors retain model metadata with empty numeric payload.
Strict nonzero-contribution/cross-component representability failures are
explicit. Rounded binary64 sensitivities limit relative accuracy near
perfect cancellation; the API does not assert uniform relative precision
there or supply a coverage probability.

The C++23 quadmath oracle has32 cases:rho endpoints/near1, cancellation,
tiny remaining cn beside huge cancelling volume terms, unit rescaling,
swapping, subnormals, finite near-maximum norms, unavailable overflow/underflow
and precedence. Tolerance2e-11 relative plus two subnormal steps, with exact
zero checks where specified. It regenerates the submitted fixture exactly.

## Scientific Canvas/SVG consumer

The production consumer calculates five declared-rho cases at fixed1mol,
zero amount uncertainty and equal1% volume relative standard uncertainties.
It displays standard uncertainty in J/K against rho[-1,0,0.5,0.9,1], with
first-order/declared-not-estimated/no-confidence-interval labels. The shared
calibration endpoint verifies an available zero uncertainty. Canvas checks
the actual endpoint marker and axis.

An independent analytic C++23 checker verifies five circles, four segments,
RGB, radius, units and semantics. Numerical budgets are propagated before
pixel conversion. Existing scatter and clipped-line integer projection
conventions are checked separately; all resulting coordinates are unique.
The actual final SVG is XML-validated and visually inspected. A PNG preview
changes only the outer SVG width for QuickLook; native geometry stays intact.

## Execution matrix

| Stage | Job | Result | Elapsed |
|---|---:|---|---|
| Physics/figure focus |15448|3/3,exit0:0|0:11|
| Combined focus |15450|5/5,exit0:0|0:17|
| Original-base cumulative |15451|95/95,exit0:0|4:48|
| Original-base actual pre-anchor |15452|94/95,exit1:0|4:47|
| Injected omitted-correlation fault |15453|3/5,exit1:0|0:16|
| Injected upper-median fault |15454|4/5,exit1:0|0:17|
| Main-based final cumulative |15457|95/95,zero failures/skips/errors,exit0:0|4:49|
| Main-based actual pre-anchor |15458|94/95,one numerical failure,exit1:0|4:47|

The pre-anchor controls restore the captured first implementation of this
new module, before integer-difference anchoring, not an imaginary legacy
normalization API. Their one source difference is normalization.sio. All
controls compile natively before numerical assertions fail. Fault15453
changes only the correlation coefficient used in entropy and fails the
physics reference and figure; fault15454 changes only even-median selection
and fails the genomic reference. These injected faults are explicitly
distinguished from the captured actual pre-repair implementation.

Final main-based snapshot:
aed2fdcacb365c8b71c7fbfbcf7f39fc2586f8578e51190902889a9139b9109c.
Final before snapshot:fa769b1ba162e80478102ea57ef936b18e936d2243352f58c4e9adfd2e7dc4b1.
Original-base candidate:
fdcdbb7497cf8ebfc7cf565bbc6abb9a006f32601f4b68236fffcc4913ac17d4.
Original-base before:
d1dece25daab418e276d25cfac3d20b63f72b27bfebb77615ea0e0e3289d494f.
Extracted manifests are checked against frozen packages. The final before
comparison isolates normalization.sio on the same current-main base.

## Resources, custody and remaining scope

The standalone runtime is45MB. It is prepared from the new checkout, while
the t560 reuses the previous runtime and receives only seven changed files
plus package metadata. Its full manifest is identical to the freshly built
new-checkout package. Full source archives and executables stay remote;
the Mac receives selected sources, evidence, reports and figures.
No RSS/speed superiority or whole-transcriptome fitting claim is made.

Each job requests2CPUs/8192MiB; focused ceilings3minutes and cumulative7.
Elapsed times include compilation and harness work, not isolated kernels.
Three C++23 checkers are freshly built with g++ -std=c++23 -O2 -lquadmath
and executed against frozen sources. No Python/Rust/Julia numerical oracle.

Custody: Documents/Sounio-stdlib-sprint-2026-10-16/counts-covariance-2026-10-19.
SELECTED_RUNTIME_SOURCES.sha256 binds selected sources to the final snapshot;
DELIVERY.sha256 covers selected source, evidence, figure and final handoff.
The receipt/master/README updates are post-snapshot documentation.

The five-domain sprint remains active, statistics/scientific rendering
prioritized. Next work includes count-matrix I/O and offset-aware count
regression, condition/robust-inference/calibration work, and a reviewed
Lorentz-force uncertainty contract. em.sio's legacy dimensional uncertainty
sum requires a separate scalar/vector-output and correlation-model decision;
it is not used as a backend or represented as fixed by this block.
