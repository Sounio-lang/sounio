<!-- docs:meta
topic_id: repo.docs.plans.stats-weighted-inference-execution-2026-10-19
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stats-weighted-inference-execution-2026-10-19
-->

# Conditional weighted linear inference — 2026-10-19

## Scope and source identity

This block adds coefficient covariance, standard errors, two-sided Student
tests against zero and pointwise confidence intervals to the checked weighted
QR fit. The caller must explicitly declare independent Gaussian errors with
variance sigma^2/w_i and unknown common sigma. The implementation never marks
the variance model as assessed. Estimated, survey or arbitrary objective
weights do not acquire a sampling interpretation merely by entering the fit.

The work remains in /workspace/sounio-sprint-stats-viz, uncommitted at
f5b2c5850be9f3d3995719ffcade44a6d851658a. Fresh remote origin/main is
118c1baff75be12db934952307f8696dc59aba4f. Its changed stdlib paths from this
base are epistemic/claim_meet.sio, optimize/nelder_mead.sio and
polynomial/lib.sio; none overlaps the block. Acceptance applies to the
isolated snapshot, not a full-current-main build, remote CI or merge.

Pinned committed Madaros ELF:
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f.
Pinned gzip:
feaaf7fe181f1ee3efd30e7fe23b1709937ea7e93b52e812276fb093b6ccf52a.
Native execution stays in Forge/Slurm with actual uid/gid 1000.

## Implementation

The public module is stats::regression::weighted_inference. Inputs are
borrowed fixed buffers: physical design stride8, p<=8, p<=m<=256, strictly
positive finite weights and explicit intercept ownership. Coefficient,
covariance and inference availability have separate statuses. Confidence
support is [0.5,0.999999], inclusive. Invalid confidence preserves available
covariance; a perfect zero-residual fit returns zero covariance but no
Student inference. Nonrepresentable covariance, half-width or collapsed
interval bounds are reported as unavailable, preserving the point fit.

QR/WLS factor entry points share the original fit engines. Upper R is in
pivoted order, while both scale vectors are in original column order. The
covariance uses compensated triangular solves, balanced scalar restoration,
LASSQ standard errors and normalized row products. No normal equations,
explicit Q or second QR decomposition is constructed.

The six-factor scalar helper balances binary mantissas and exponents before
restoration. Final IEEE underflow is allowed at helper level; the statistical
caller rejects lost nonzero H entries and unavailable covariance diagonals.
Ordinary off-diagonal rounded zero remains allowed. This is a bounded
floating-point implementation, not a proof of correctly rounded products.

Critical values invert direct Student tails by bracketing and bisection,
with finite/range and roundtrip checks. Student df1 now uses the analytic
reciprocal Cauchy tail. Other supported df use bounded beta arguments;
the small-t branch forms the complementary argument directly. This avoids
both overflowing t*t and rounding away the near-origin tail departure.
The existing large-df normal approximation remains outside this fit's
integer df1..255 scope. Arbitrary fractional-df extreme accuracy is not
certified by these cases.

## Independent references and scientific rendering

C++23 __float128 references use raw weighted design, reorthogonalized MGS
and a small inverse R. The Student reference has its own beta continued
fraction and quantile bisection, crosschecked against analytic df1/2/3/6
forms. Thirty cases cover covariance signs/column order, design/response and
global-weight scaling, p3/p8, maximum m256, degeneracy, confidence boundaries,
malformed inputs and subnormal covariance with unavailable collapsed CIs.

The SVG checker independently computes the two-column fit and an analytic
df6 Student CDF/quantile. Numerical budgets are propagated before pixel
conversion. The six expected X positions each determine one pixel:
centers465/622 and endpoints362/569/599/646. It checks two coefficient
markers, two interval segments, four caps, RGB, labels and confidence metadata.
The real consumer also exercises Canvas and explicitly displays that the
95% Student intervals are pointwise, conditional and based on an unassessed
declared Gaussian precision model. No unavailable interval is rendered.

## Resource and claim boundaries

Factor extraction avoids a repeated decomposition when inference is requested.
Old point-fit wrappers supply bounded dummy factor arrays: QR576 bytes and
WLS640 bytes of additional source array payload, plus delegated storage.
Inference uses bounded fixed arrays; compiler frames, returned records and
temporaries add further cost. No speed or RSS improvement is claimed without
a controlled measurement. Runtime packages exclude Git history and compiler
source; the Mac receives selected sources, receipts and figure artifacts,
not full source snapshots or build trees.

This block does not implement robust/cluster covariance, prediction bands,
simultaneous intervals, uncertainty in estimated weights, SVD condition
certification or empirical coverage studies. Statistics/scientific rendering
remain the current priority within the five-domain sprint; genomics, physics
and further mathematics remain active scope. No commit, push, PR or merge
has been performed.

## Native execution and custody

| Run | Slurm job | Native result | Elapsed |
|---|---:|---|---|
| Initial extreme tail |15413|1/1, exit0:0|0:06|
| Actual previous tail, focused |15414|0/1, exit1:0|0:05|
| Initial visual |15415|1/1, exit0:0|0:12|
| Initial four-fixture focus |15420|4/4, exit0:0|0:23|
| Rejected first cumulative candidate |15421|89/90, exit1:0|4:35|
| Actual previous tail, cumulative |15422|89/90, exit1:0|4:32|
| Final focused, including previous-tail regression |15429|5/5, exit0:0|0:26|
| Final cumulative |15430|90/90, no failures/skips/errors, exit0:0|4:35|

The rejected candidate15421 exposed a cancellation regression atdf100,t10:
subtracting a nearly unit complementary beta CDF erased a small representable
tail. The final implementation uses complementary beta only when |t|<=0.5
and evaluates the direct tail farther out. The unchanged previous-tail test
passes again in15429 and15430. Initial runs do not certify the final source.

Control15422 restores the actual previous student_t.sio and differs from the
final candidate in that one source file plus its derived manifest. It retains
all86prior fixtures and fails only stats_student_extreme_tail after successful
native compilation. Every other new fixture also passes in this control.
The final cumulative run retains all86previous and allfournew fixtures.

Final source archive SHA256:
a457d3bb0f04bcc961d3a0df4a0f421c3f529404e465edb3a67a69c7858f358c.
Actual previous-tail archive:
31c43d9479312f265351394c928b5783a188572230da56da7e4d69b22beccdf8.
Rejected candidate archive:
9574949b8d977caf8ff682f6c807a2641ab5639ad48f5178e8d1cf5916a5ea41.
Extracted snapshot manifests match the frozen package manifests. The final
candidate-versus-control manifest diff contains only student_t.sio.

Final student_t.sio:
31a504c74b7ecf92dde9b9cf5efabcf23a0df021df4f13a709dde92adc1ba27c.
weighted_inference.sio:
0f4b10efeb44a48faa62513ac71ed46ba0b9380acef1be07e6f4fe74a57ec0c4.
Oracle-emitted fixture:
fb43c0b213e2dfed8783e4aa7a7922a769476aa5b45459d3659ec86d893dcc16.
Independent C++23 coefficient/covariance/Student report:
0a18a29c0a43cb7fd7effc226e048f80253c21c4530e4e3e1e1a303bba4b5c0f.

Five C++23 checkers were freshly compiled with g++ -std=c++23 -O2 and
-lquadmath against the frozen runtime sources, executed with exit0, and
the emitted numerical fixture matched the submitted fixture byte for byte.
A separate C++23 sizeof report records binary64 array payload analogues;
it is explicitly not a Sounio frame or RSS measurement.

Actual final SVG is XML-valid and passes the independent geometry checker.
The checker rejects altered cap-position, RGB and missing-marker copies.
The inspected PNG preview changes only the outer SVG width to fit QuickLook;
the native SVG geometry/viewBox and original artifact remain unchanged.

Local selected-source custody:
/Users/demetriosagourakis/Documents/Sounio-stdlib-sprint-2026-10-16/weighted-inference-2026-10-19.
SELECTED_RUNTIME_SOURCES.sha256 binds selected sources to the final executed
snapshot. DELIVERY.sha256 covers the delivered selected sources, evidence,
reports, figure and final handoff. Full source archives stay remote. The
execution receipt, master ledger and README updates are final post-snapshot
documentation; they were not part of the already frozen runtime submission.

Each job requests2CPUs/8192MiB with3-minute focused and7-minute cumulative
ceilings. Elapsed times include compilation/harness work, not isolated kernel
benchmarks. Next work remains genomic covariate/count consumers and physical
covariance-based uncertainty propagation, followed by further condition and
robust-inference work; the multi-domain sprint remains in progress.
