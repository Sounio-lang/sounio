<!-- docs:meta
topic_id: repo.docs.plans.stats-multivariable-qr-execution-2026-10-18
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stats-multivariable-qr-execution-2026-10-18
-->

# Rectangular QR least squares and multivariable calibration — 2026-10-18

## Scope and source identity

This checkpoint adds a checked bounded rectangular least-squares solver,
a borrowed public multiple-regression adapter and an actual Canvas/SVG
calibration consumer. The major statistics/scientific-viz/math/genomics/physics
sprint remains active.

Worktree: /workspace/sounio-sprint-stats-viz.
Original remote-main baseline: f5b2c5850be9f3d3995719ffcade44a6d851658a.
Remote main freshly checked:118c1baff75be12db934952307f8696dc59aba4f.
Compared with baseline, changed stdlib paths are epistemic/claim_meet.sio,
optimize/nelder_mead.sio and polynomial/lib.sio; none overlaps the touched
kernel/adapter. This is acceptance of the documented isolated snapshot,
not execution of the full current-main tree or remote CI.

Pinned committed Madaros v0.80:
ELF125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f;
gzip feaaf7fe181f1ee3efd30e7fe23b1709937ea7e93b52e812276fb093b6ccf52a.
The canonical workspace's local compiler is excluded.

## Implemented library contract

linalg::least_squares::qr_least_squares borrows a fixed2048cell row-major
matrix with physical stride8 and a fixed256response. Active dimensions are
1<=p<=8, p<=m<=256; inactive cells are ignored, including nonfinite poison.
The caller supplies an intercept column if desired.

Columns are equilibrated by their active maximum absolute values; response
by its maximum absolute value (zero response uses1). Compact Householder
reflectors and column pivoting transform the matrix and response. Every
trailing column norm is recomputed with LASSQ, avoiding fragile subtractive
norm updates. Dot products and triangular sums are compensated.
No dense Q or normal equations are formed.

Numerical rank is defined by abs(Rkk)<=rcond_used*abs(R00) on the equilibrated
matrix; default rcond is m*binary64epsilon. Caller rcond must be finite and
in(0,1), or0for default. This diagonal policy is a heuristic, not an SVD
rank certificate. r_diagonal_ratio is not a condition-number estimate.
Rank-deficient inputs return explicit status3 and metadata, with no partial
solution/pseudoinverse. Nonfinite, dimension, rcond and arithmetic failures
have separate statuses and deterministic empty payloads.

The result provides original-order coefficients, fitted values and residuals,
their Euclidean norm, rank, algebraic df, pivot labels and threshold diagnostics.
Coefficient restoration tries multiplication/division orders while avoiding
prematurely subnormal intermediates. Detected nonzero coefficient underflow
or overflow is unavailable. Reported fitted values use one final binary64
multiplication, permitting ordinary zero/subnormal rounding; residuals are
b-reported_fitted in raw units. Their norm uses raw LASSQ. No epsilon zero
rule is inserted. Inputs that lose information during normalization or
ill-conditioned cancellation are not promised correctly rounded outputs.

stats::regression::multiple::multiple_linear_fit forwards borrowed inputs
to the shared kernel without extra full-matrix copies. Existing simple OLS
and Matrix8 decomposition APIs are preserved. This new API provides no
standard errors, p-values, confidence intervals, covariance, weighted fits,
regularization, missing-data rule or assumption assessment.

## Native execution and controls

Initial focused15382:2/2, exit0:0, elapsed0:12; adapter/status and Canvas/SVG.
Final15383:83/83, zero failures/skips/errors, exit0:0, elapsed4:04.
All80 previously accepted fixtures remain passing.
Absolute-cutoff control15384:82/83, one numerical runtime assertion
failure after successful compilation, zero skips/errors, exit1:0, elapsed4:03.
Only linalg_least_squares_reference fails; all80 previous fixtures pass.
Control snapshot efac0f3d582844e82f5b3609ddad4b093694116c4cd1d46563e0a1c436fc2e75.

All native execution uses Forge/Slurm, pinned compiler, uid/gid1000, CPU only.
The full candidate/control each request2CPUs,8192MiB,6minute ceiling; the
two-case initial gate used a3minute ceiling. Measured job elapsed times
include compiler/harness work and are not a kernel-performance benchmark.

Candidate snapshot SHA256:
139c6410ed92624ffd497af561fcc14a86e972eaccaa1bd5b6e7fbae73c05924.
The control changes only the solver's rank cutoff to fixed1e-15 plus the
derived manifest. This is a deliberate fault-injection policy analogous to
the private legacy QR cutoff, not an execution of that entire legacy API.
Its independent native reference fails after compilation; unrelated fixtures
remain accepted.

## Independent numerical and rendering reference

C++23 g++14.2.0/libquadmath independently solves with modified Gram-Schmidt
and a second projection pass in __float128, rather than binary64 Householder.
A separate equilibrated pivot routine evaluates the declared rank policy.
The28cases cover noisy rectangular/square systems,256x8, forced pivot,
polynomials, near dependence, tiny/huge/mixed column scales, zero response,
duplicated/zero columns, custom rank thresholds, validation precedence,
inactive poison, subnormals and coefficient restoration/representation limits.

Coefficient relative tolerances and fitted/residual tolerances in response
units are separated. No blanket absolute tolerance1 is used for tiny
nonzero coefficients. The restoration case has a first coefficient about
1.23516411460311632e-124 whose naive normalized-times-response product
rounds to zero; another scaling order is required.
Root reran the C++ oracle with exit0 and byte-identical report.
Executable SHA256:
9baed18afda9895e62ec950f9a04b97359f26ad613247270e46a196044fff290.
Report SHA256:
cc73ae7fdf269ef5461d1b23271b6c23fefcafd01677f43173c37c85cbb29d12.

The synthetic two-sensor calibration uses8observations, an explicit
intercept and2dimensionless signals; response/residual units are mV.
The independent coefficients are10,2,2.96; residual norm about0.42895221179.
Canvas checks actual axis pixels; SVG contains two8point panels, identity
and zero references, explicit domains/units, rank/df and dynamic coefficients.
No inferential interval is implied. A separate C++23 checker derives all16
marker coordinates/RGB from independent orthogonal-design projections and
checks actual SVG counts, labels and units. XML and visual inspection
supplement the native and numeric checks. Final SVG and the initial focused
SVG are byte-identical. QuickLook's square thumbnail crops the right panel;
the original1440x620SVG retains both complete panels. The thumbnail is a
preview, not a Canvas framebuffer capture.
SVG checker SHA256:
9902a7fbc55d8916dafb2916d089fd52bf800ae3830725e6a8e0b4fa7d8a3b8b.

## Resources, custody and next implementation

One2048f64 matrix workspace, one256f64 transformed response, two8f64vectors
and the bounded result arrays. Prediction rereads borrowed original inputs,
rather than keeping a second full original matrix. No heap or dense Q.
Work is O(m*p^2), with m<=256,p<=8. Backend stack/temporary overhead is
additional; no measured runtime/RSS superiority is established.

Three subagents owned kernel/contract, independent C++reference and
figure/SVGchecker. Root owned architecture decisions, adapter/status fixture,
integration, native gates/control, artifact readback and final custody.
Darwin was consulted before new Sounio. All numerical references use
C++23/Sounio per ADR009, without Python/Rust/Julia.

Custody:
/Users/demetriosagourakis/Documents/Sounio-stdlib-sprint-2026-10-16/qr-2026-10-18.
Retain selected sources, numerical/geometry reports, native logs/JUnit/receipts,
archive manifests/hashes, SVG and final contracts. No full source archives,
compiler builds or Git history are copied locally. Final plan/status/docs
are updated after the frozen runtime snapshot and tracked separately.
Sources remain uncommitted; no push, PR, merge or whole-stdlib certification.

Next implementation: positive finite weighted least squares and the
statistical covariance/inference contract, then genomic covariate/count
consumers and physical uncertainty propagation. Broader layout/provenance,
condition estimation, rank-deficient minimum-norm/SVD, empirical calibration
and execution/RSS benchmarks remain open parts of the same major sprint.

