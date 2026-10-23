<!-- docs:meta
topic_id: repo.docs.plans.poisson-lorentz-execution-2026-10-19
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.poisson-lorentz-execution-2026-10-19
-->

# Offset-aware count models and Lorentz covariance — 2026-10-19

## Source and delivery boundaries

Active isolated worktree: /workspace/sounio-sprint-stats-viz-main2-20261019,
based on observed remote main15a1983fd8d9bb4730f2fd6ca72e50f5e0c88d0f.
The prior54f73-based working checkout remains preserved at its pre-port state; the accepted95-fixture runtime snapshot remains archived. A scoped port
copied274 authorized files, with six upstream changed paths and no overlap.
Old status and source hashes were checked before/after; all copied files
were byte-identical at the port checkpoint. The subsequent Poisson numerical
repair, identifier correction and documentation updates are later changes
in main2, not part of that pre-edit identity claim.

Port manifest b8660feb7cb2dc55ce69b283400d323a47231c381affceccdaf907c493a57d46.
Patch552bbac8257092f719e4889dd0d3ea2ce4a702da1721fe4d4c97e86e328b5c0e.
No reset, rebase, commit, push, PR or merge was performed.
All Sounio check/build/run evidence comes from native Forge/Slurm jobs,
using the committed Madaros gzip, clean environment, uid/gid1000.
Pinned gzip feaaf7fe181f1ee3efd30e7fe23b1709937ea7e93b52e812276fb093b6ccf52a;
decoded ELF125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f.
The changing upstream compiler source was not rebuilt in this library slice.
Native module acceptance is distinct from repository-wide CI or main integration.

## Count-model implementation

stats::regression::poisson::poisson_fit_offset accepts borrowed bounded
n<=256,p<=8 design/count/log-offset buffers, physical design stride8 and an
explicit intercept. It fits mu=exp(offset+Xbeta), without estimating offsets
or clipping means. Initialization uses CPQR of ln(y+0.5)-offset, with the0.5
used only for initialization. Fisher increments reuse checked weighted CPQR.
No production normal equations or new heap allocation are introduced.

The solver exposes dimensions, rank, degrees of freedom, iteration count,
coefficients, fitted means/predictors, deviance and convergence diagnostics.
Statuses distinguish invalid sizes/controls, nonfinite input, negative count,
numerical rank, arithmetic availability, budget exhaustion and a sufficient
all-zero/intercept boundary detector. Failed fits return no partial solution.
The boundary detector is not a general MLE-existence certificate.

Success requires BOTH the normalized score and full Fisher predictor step
at the returned parameters to meet tolerance. A small halved move and a small
objective change do not certify convergence. Near-equality deviance uses a
stable phi series; sums are compensated. Counts above2^53 can round in this
binary64 likelihood; the estimator makes no exact large-integer contrast claim.

The initial absolute-deviance line search stalled natively on deterministic
case18: status7 after100 iterations, while relaxed diagnostics converged in4.
A first direct-change implementation using independently rounded eta differences
then stalled on the max256x8 case15. Both diagnostic versions are retained.
The final criterion uses the local parametric predictor displacement from
actual trial_beta-beta, then a compensated X*displacement. For small movements,

    half change = sum[(mu-y)*d + mu*(exp(d)-1-d)].

The exponential remainder uses a compensated degree32 series for every|d|<=0.5;
larger movements retain the absolute-deviance comparison. This avoids both
subtraction of nearly equal absolute objectives and unrelated rowwise eta
rounding noise. It is anchored at represented current means, so separately
reported absolute deviance can move in its final bits. Stopping tests remain
unchanged. See stdlib/stats/regression/POISSON.md for the complete contract.

The independent C++23 quad reference uses raw score/Hessian dense pivoted
elimination plus analytic intercept/exposure/indicator designs; production
uses CPQR. Its26 cases retain exact fixture regeneration, strict tiny-coefficient
relative checks, all256 mean/predictor checks at max shape and fixed tolerances.
Large generated fixtures are split into borrowed32-row setup/check helpers to
respect the compiler IR function limit without removing coverage.

This is a Poisson mean model. Independence, exposure/design adequacy and
conditional equidispersion are unassessed; no negative-binomial dispersion,
RNA-seq differential-expression test, standard error or interval is supplied.
Official model/offset background: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/glm.html.
No R or Python numerical implementation was used.

## Physical uncertainty repair

physics::lorentz supplies checked scalar and3D force APIs. The vector API
returns F, full3x3 covariance, component standard uncertainties and the vector
root-mean-square uncertainty sqrt(trace(Cov)). It declares independent charge,
electric-field, velocity and magnetic-field components and first-order propagation.
Output covariance is assembled from the physical Jacobian, retaining shared-input
output correlations. This follows the multivariate uncertainty framework:
https://www.bipm.org/documents/20126/2071204/JCGM_102_2011_E.pdf.

The scalar proxy is signed q(E+vB), with derivatives E+vB,q,qB,qv. Shared charge
contributes once, including cancellation. The legacy3D function no longer sums
E,velocity and magnetic-field uncertainties as if they had common units.
The legacy3D scalar uncertainty inputs mean isotropic independent per-component standard errors;
the returned scalar is sqrt(trace(Cov)), not uncertainty of force magnitude.
Failures use -1 as the unavailable standard-uncertainty sentinel. Checked APIs
retain typed statuses and clear failed numerical payloads.

Private binary-scaled products/sums and scaled norms avoid several intermediate
range failures. Strict covariance-term representability is conservative;
near-cancellation relative precision and exact last-bit PSD are not guaranteed.
The independent quad oracle has26 vector and12 scalar cases, exact zero checks,
off-diagonal signs, moderate principal-minor PSD checks and extreme scales.

The formerly public3D signature exposed private Vec3 fields/constructor and a
bare units import that did not resolve. The repair exposes Vec3/vec3_new and
uses public units::lib functions. Adapter effects are explicitly Mut,Div,Panic.
The numerical-before control applies those same common import/visibility repairs
to the original numerical bodies, so its numerical failure follows compilation.
The untouched original source is retained separately. Other em uncertainty
routines retain their own limitations and are not certified by this repair.

## Native scientific rendering

The consumer uses8 synthetic counts in2 groups with unequal known exposures.
It plots observed and fitted rates and conditional Pearson residuals with
explicit units, no intervals and unassessed assumptions/overdispersion labels.
Canvas checks and actual SVG are produced by the public fit and rendering code.
The independent closed-form group reference verifies24 markers, coordinates,
colors, radius and labels, propagating mean budgets before pixel conversion.
The rates are1.2 and2.266666666666666667 events/s. SVG geometry is independently
checked; moved, missing and wrong-radius marker controls must be rejected.
A PNG is a viewer preview, not a Canvas capture. Native SVG stays unchanged.

## Acceptance

Final Forge/Slurm15603: **101/101**, zero failures/skips/errors, exit0:0,5:04.
All95 prior fixtures remain accepted. Snapshot9bb020718429749326b7e5229f60bbaa2ee7a97a240eecb0e8181bda021bc156.
Runtime manifest SHA256d4bb6caab77912c1972e6713cb34569038c576d6ae874048fb45a18f4be1878a.

| Stage | Job | Result | Elapsed |
|---|---:|---|---|
| Final focus |15602|6/6,exit0:0|0:22|
| Final cumulative |15603|101/101,exit0:0|5:04|
| Actual pre-repair Poisson |15604|5/6,only numerical reference fails after compile|0:21|
| Original Lorentz formulas with common API repair |15605|5/6,only legacy numerical fixture fails after compile|0:20|

Control manifests differ from the focused final manifest by exactly one runtime
source member each: poisson.sio or em.sio. Derived manifests and control metadata
also change. No numerical assertions or tolerances were weakened.

Focus snapshotd62060dec85eb96cbeaf1353fb98acbc0db4a6fb5b35d020e881f443336b0f95.
Poisson-before snapshot6be7f5f6860c8f7ccf925b752fe2dfcf28c2c7a401df2f8693e33574961d7378.
Lorentz-before snapshotd756126a7381c63f39aaede12b617dca3dcc3a8ad25bcc6798a96b2647fd7f7c.

Final native SVG SHA256ed1f04d79d8af8a635d58c93eac84064c36c9f010febc597c2048e55bc2d85cc;
focused and cumulative SVG bytes are identical. The final independent C++23 checker,
XML parse, visual review and all three geometry-negative controls passed.
Poisson and Lorentz generators reproduce submitted fixtures exactly. An independent
read-only source/contract review found no concrete defect in the final core.

## Resources and remaining work

Slurm reports batch MaxRSS5277668K for the focused job and7062324K for cumulative
job15603. These include compilation and execution; neither is an isolated kernel
memory measurement or evidence of a memory reduction. No speedup is inferred by
comparing suites with different membership. The production solver uses bounded
scratch and reuses existing QR; the local change needs only8 coefficient displacements.

Runtime packages remain remote. Transfers reuse the accepted previous package and
send24 changed/new source members plus manifests, then the numerical core/doc delta.
The final package was compared against a fresh worktree-generated full manifest.
The local review custody contains selected sources, reports, receipts and SVG,
not the complete compiler/runtime archive or generated executable collection.

Next work: per-phase compile/execution/RSS measurements using existing bundle
infrastructure; bounded count-matrix/I/O; robust/clustered coefficient covariance
and calibrated inference; genomic negative-binomial dispersion remains separate.
These tasks and the broader multi-domain sprint remain open.

Post-acceptance receipt/master/status-ledger edits are documentation after the
frozen runtime snapshot. Numerical production and fixture sources remain unchanged.
