<!-- docs:meta
topic_id: repo.docs.plans.stats-weighted-qr-execution-2026-10-19
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stats-weighted-qr-execution-2026-10-19
-->

# Weighted QR least squares and calibration — 2026-10-19

## Accepted execution

| Run | Job | Result | Elapsed |
|---|---:|---|---|
| Initial focused |15404|3/3, exit0:0|0:13|
| Final cumulative |15405|86/86, no failures/skips/errors, exit0:0|4:17|
| Previous Householder top updates |15406|84/86, two runtime assertion failures, exit1:0|4:20|

All83previous fixtures remain passing in both cumulative runs. The before
control fails only linalg_weighted_least_squares_reference and
stats_weighted_qr_status after successful native compilation.
Its source differs in exactly least_squares.sio plus the derived manifest.
This restores the actual pre-repair formulas, rather than inventing a
different weighted dataset or reordering the rows.

Candidate source archive SHA256:
2a510936c41cea08b2a698d35009a77ca049c6628451722b5d941b28d5861003.
Before archive:
ac2d5460b0c52f216c601a45921fb8ffc26fe6fe32c75358a700c6d15a20c99d.
Both extracted archive manifests match their frozen source packages.
Selected runtime source hashes are checked separately from final documentation
and the later SVG-checker correction.

Worktree /workspace/sounio-sprint-stats-viz remains uncommitted at baseline
f5b2c5850be9f3d3995719ffcade44a6d851658a. Remote main was freshly checked at
118c1baff75be12db934952307f8696dc59aba4f. Its changed stdlib paths from baseline
are epistemic/claim_meet.sio, optimize/nelder_mead.sio and polynomial/lib.sio,
without overlap with this block. This is acceptance of the isolated snapshot,
not a full-current-main execution, remote CI, commit, push, PR or merge.

Pinned committed Madaros v0.80 ELF SHA256:
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f.
Gzip:
feaaf7fe181f1ee3efd30e7fe23b1709937ea7e93b52e812276fb093b6ccf52a.
The canonical workspace's local build is excluded. Native runs use Forge/Slurm,
uid/gid1000, CPU only; each requests2CPUs and8192MiB, with3minute focused
and6minute cumulative ceilings. Elapsed times include compilation/harness work.

## Library behavior and numerical repair

linalg::weighted_least_squares::weighted_qr_least_squares and the public
stats::regression::multiple::weighted_multiple_linear_fit borrow the design,
response and weights. Physical row stride8, active1<=p<=8,p<=m<=256;
intercepts are explicit. Only active cells are read.

Validation order: dimensions1, rcond5, all active design/response finiteness2,
then all positive finite weights6. Rank3 and arithmetic4 remain distinct.
Errors preserve requested dimensions with empty numerical payloads;
rank deficiency retains rank/pivots/threshold/diagonal ratio/weight scale.
variance_model_assessed is always false.

Original columns and response are scaled before row weighting. The row factor
is sqrt(w_i)/sqrt(maxw), avoiding underflow from forming w_i/maxw first.
The transformed system delegates to the shared checked column-pivoted QR.
Coefficients are restored with the shared scalar scaling helper. Inputs are
borrowed again for raw fitted values; residuals are b-reported_fitted.
Two LASSQ accumulators return raw residual norm and normalized weighted norm.

The normalized weighted norm uses weights divided by maxw; it is not the
unnormalized objective norm. A common rescaling of all weights preserves the
mathematical fit and normalized norm, subject to binary64 rounding and rank policy.
weight_scale records maxw. Nonzero entries disappearing during normalization
or row weighting, and nonzero raw residuals whose weighted component rounds
to zero, return explicit arithmetic status4. This is a conservative
representation limit even if a final aggregate might be representable.
Final fitted multiplication follows ordinary IEEE rounding; covariance,
SE/p/CI, frequency-weight semantics and a validated variance model are absent.

Review of extreme positive weights exposed a cancellation in shared QR:
old_top - tau*(old_top+tail_dot) can erase a tiny useful projection.
Both matrix and response top updates now separately accumulate tail_dot and use
(alpha/beta)*old_top - tau*tail_dot, using1-tau=alpha/beta.
Lower reflector updates retain the full projection.
The pre-repair source was backed up before editing.
The scalar restoration helper's arithmetic moved unchanged into
least_squares_scale.sio, with visibility/symbol changes; the numerical
Householder repair is separately identified and evidenced above.

The critical raw case A=[0,1], b=[1,2], weights=[DBL_MAX,minsub] succeeds with
coefficient2 and rank1. A second case preserves a coefficient around2e-300.
Another case has raw residual norm1e308 while its normalized weighted norm is
about1.01364851591845965e-7, checked in its own scale.

## Independent C++23 reference

g++14.2.0/libquadmath solves40cases using twice-projected modified Gram-Schmidt
in __float128 on the weighted raw design, rather than the production binary64
preparation or Householder updates. It also emits16OLS/WLS figure rows.
Coverage includes equal/global-scaled weights,256x8, noisy/square/near-dependent
designs, mixed/extreme column and response units, invalid/ignored entries,
rank/rcond/pivot metadata, subnormals and restoration/representation limits.

Coefficient forward tolerances and response-unit residual/fitted budgets are
separate. The tiny normalized weighted norm does not inherit the enormous
raw-residual absolute budget. Binary64 root-factor quantization has a declared
budget; rank fixtures are away from the cutoff.
Root re-execution exits0 and produces an identical report.

Executable SHA256:
fc6c61378ea3e55ea0b14fc89a89e8c3c2f05747759f40d800c8e0ecdbe11ef3.
Report:
a27c49011cc26e7239a7df1580205f9aaa21a7b5db74a9b308a18ee1c3d99e02.
The C++ reference includes the preserved independent LS source, retained
alongside it for reproduction.

## Scientific rendering and review

Eight synthetic mV observations compare unit weights and alternating relative
objective weights1/4. The figure shows both fits and both raw-residual sequences,
with explicit units/domains/legends and dynamic WLS coefficient labels.
Native assertions verify the strictly smaller normalized weighted objective
at the WLS solution, metadata, scene structure and actual Canvas axis pixels.
Positive objective weights are distinct from assessed inverse-variance weights;
the figure states variance_model_assessed=false and supplies no SE/p/CI.

The final1440x620SVG is byte-identical to the focused SVG and parses as XML.
Independent C++23 projections check24circles, RGB, radius, counts, unit/model
labels and formatted coefficients. The first point-only checker rejected one
pixel at the final OLS residual because the oracle lies at an integer display
boundary. The corrected sidecar propagates the SAME numerical reference budget
3e-11*||b||2 in mV through the renderer's affine map and integer conversion.
Only that point straddles: projected98 with interval approximately
[97.9999999533,98.0000000467], giving allowed Y292..293.
The other23points retain exact coordinates. X/colors/radius/counts remain strict.
This is numerical rendering tolerance, not a statistical interval.

The corrected checker accepts focused/final output and rejects copied artifacts
with Y294, a removed marker, or an altered RGB (each exits1).
Root repeats those checks. The checker update is POST-SNAPSHOT, with separate
source/executable identity; it does not change the frozen Sounio source.
Final source SHA256:
a05d14dcc355ad87914b8b173e7b3995358ed39c2a1a70f7e8aab1b6c9f1ee37.
Executable:
5a42e27de7d85b8bce105fd2303d480a64dd3c2db94773f678dc89a4419ebc7d.

For visual review, a preview copy changes only outer SVG width1440to768
while preserving the original viewBox and all geometry/text. This makes both
panels visible in QuickLook's fixed viewport. The original native SVG is
unchanged. The PNG is a preview, not a Canvas framebuffer capture.

## Resources and continuity

The WLS adapter uses one transformed matrix/response and small scale vectors,
in addition to existing QR scratch/results. No heap or explicit Q is added.
The common scaling body is shared rather than duplicated. Work remains
O(m*p^2) with bounded dimensions; backend temporaries/copies and stack overhead
are additional. No measured runtime/RSS superiority is claimed.

Three subagents owned kernel/contracts, independent reference and figure/checker.
Root owned architecture, the public adapter/status fixture, numerical review,
before/after gates, source identity, artifact readback and final custody.
Darwin was consulted before new Sounio; native execution stayed in Forge.
Numerical references use C++23/Sounio per ADR009.

Custody:
/Users/demetriosagourakis/Documents/Sounio-stdlib-sprint-2026-10-16/wls-2026-10-19.
Retain selected sources, native logs/JUnit/receipts, manifests/archive hashes,
C++ reports, checked SVG and final docs; no full source archives/builds/Git
history are copied locally. Final documentation and checker are tracked
separately from the runtime snapshot.

Next: covariance and inference with explicit variance/weight assumptions,
condition diagnostics, genomic covariate/count consumers and physical
uncertainty propagation. The broader five-domain sprint remains in progress.

