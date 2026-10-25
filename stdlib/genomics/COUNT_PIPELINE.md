<!-- docs:meta
topic_id: repo.stdlib.genomics.count-pipeline
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.stdlib.genomics.count-pipeline
-->

# CSR count pipeline

`genomics::count_pipeline` composes the existing checked CSR identities,
normalization, dynamic NB2 dispersion profile, Gaussian working dispersion
component, conditional Fisher contrasts and full-family multiple testing.
`genomics::count_pipeline_report` supplies a retained row slot and a streaming
scientific TSV. This is computational inference under declared assumptions;
finite-sample calibration, RNA-seq validity and global optimization are not
certified. It is not a DESeq2 parity implementation.

## Ownership and payload budget

`count_pipeline_workspace(G,N,P,method)` queries all newly owned numeric and
integer buffer payloads before allocation. The bounds are 3<=G<=1,000,000,
2<=N<=1,000,000, 1<=P<N and P<=1,024; the canonical workspace queries impose
additional checked representability constraints. Method0 is ML, method1 is the
existing Cox-Reid adjusted criterion. Payload is the sum of element counts times
eight; it excludes transferred inputs, allocator metadata, stack and RSS.

`count_pipeline_new` consumes a validated `CountCSR`, original-row gene byte IDs
and original-column sample byte IDs, even on refusal. IDs must be unique and
have exactly the corresponding counts. An ID owner with duplicate/coalesced
original rows is refused: callers must supply an explicitly matching canonical
row identity owner. Column coalescence in CSR does not reorder rows. All
construction outcomes remain closeable with `count_pipeline_close`.

The owner retains one sample-sized current count vector, one design, one
profile/Fisher scratch set and one Gaussian-prior scratch set. It retains
O(G*P) coefficient records and O(G) family records, not per-gene fitted N-vectors.
No scratch allocation occurs inside the point, shrinkage, refit or adjustment
batch APIs. Canonical kernel result values and compiler-managed temporaries are
separate from this buffer-payload contract; measured process RSS is reported
separately. Allocation failure is owner status4; no partially allocated owner is
used for fitting.

## Stages

1. Prepare phase0 with metadata byte IDs, row-major design, supplied log exposure,
   explicit initial coefficients and contrast. An exact byte join maps metadata
   into count-column order, with missing/ambiguous IDs refused. Neither positional
   matching nor an implicit intercept is inserted. Log size factors from the same
   immutable CSR counts are added to supplied log exposure. Mode0 requires
   positive counts in every sample for a normalization gene; mode1 is the named
   positive-ratio estimator. There is no automatic mode fallback. Failed
   preparation leaves phase0 and can be explicitly retried; comparison work is
   accumulated with a checked remaining-integer budget.
2. `count_pipeline_point_batch` advances up to the requested number of original
   gene rows through the existing bounded profile algorithm. Options are stored
   once at construction. Failed genes remain in their original slots. If
   `compute_curvature=false` was supplied, point estimates remain usable but
   Gaussian-prior eligibility is unavailable. The explicitly named
   `count_pipeline_point_batch_seeded` selects the existing opt-in seeded ML and
   validated warm-start algorithm, restricted to point-only method0. It still
   evaluates candidates; a seed is never silently accepted as an estimated fit.
3. After every point row has been attempted, `count_pipeline_shrink` compacts only
   numerical interior positive-alpha points with positive available local
   log-alpha variance and positive base mean. These are **numerical training
   eligibility** criteria, not the profile kernel's inferential certification
   flag. All selected rows enter the existing Gaussian working model in original
   order. Fewer than three selected rows return prior status8; an unavailable
   trend/prior is not replaced by an alpha or variance floor. The new active
   prefix adapters ignore the remaining allocated capacity.
4. `count_pipeline_refit_batch` keeps the Gaussian closed-form mapped log-alpha
   and variance as explicit diagnostic fields. For a positive estimated prior
   variance it performs an actual penalized NB2/Cox-Reid log-alpha profile with
   `prior_center=log(a0+a1/baseMean)`. A zero prior variance is an explicit point
   prior at that trend. A high-dispersion outlier uses its unshrunk point alpha.
   The selected alpha is subsequently a plug-in input to a fresh conditional
   Fisher fit. The contrast consumes the corresponding unmodified factor and
   coefficient buffers immediately, before the next gene overwrites scratch.
   Invalid/nonrepresentable trend arithmetic is unavailable rather than floored.
5. A two-sided conditional normal Wald tail is evaluated through the canonical
   direct `GammaQ(1/2,z^2/2)` and its independently available log upper tail. A
   nonzero z whose squared half-statistic is nonfinite, zero or subnormal is
   refused. Both NB2 model and independence must have been explicitly declared;
   declaration is not assessment. Scalar tail underflow does not fabricate p=0:
   a finite log tail can remain available independently.
6. After every original row has reached a terminal refit decision, the existing
   dynamic BH and BY implementations receive the **whole original G-row family**.
   Unavailable scalar tests have effective p=1 and adjusted output1. Failures are
   never removed from the family or relabeled as successful nonsignificant tests.

`max_genes` controls batch progression, not statistical multiplicity.
`max_outer_evaluations` counts each canonical profile's objective evaluations.
The constructor requires a positive per-gene evaluation cap whose product with G
is representable in signed i64, before allocating scratch. The owner reports
the sum actually performed for point and penalized refit
profiles. Gamma-tail iterations have a separate counter and budget; those units
are never mixed. Inner iteration caps, local curvature stopping tolerances and
outer search intervals are numerical controls, not accuracy certificates.

## Availability and reports

Owner status0 means the owner and buffers were constructed, not that any fit
succeeded. Phase0 is unprepared,1 is point progression,2 is point complete,3 is
refit progression,4 is refit complete,5 is adjusted/reportable. Point and final
coefficient availability have separate accessors; coefficient getters assert
availability. All coefficient storage is initialized, including pre-fit and
failed rows. A successful fixed-alpha fit can have unavailable covariance or
p-values, so coefficient availability is not inferred from the p-value mask.

`count_pipeline_report_into` overwrites one caller-retained `CountGeneReport`.
Invalid owner/row sets status1 and preserves previous payloads; check status
before consuming them. Every valid row contains separate point, curvature,
prior-eligibility, refit, covariance, contrast and tail statuses and independent
p/log-p availability. Numeric payload0 when unavailable is not an estimate.

`count_pipeline_write_tsv` requires phase5 and emits all original gene rows.
Identifiers are hex-encoded original bytes, with no truncation or UTF-8
normalization. `NA` denotes unavailable scientific values; unavailable tests
still have explicit BH/BY1 with availability false. Display numbers use the
existing scientific printer at 15 significant digits. The retained structured
API preserves original binary64 values for plotting or lossless export; the
TSV is not advertised as binary64-round-trip serialization.

The report declares BH's valid-p/independence-or-PRDS requirement and BY's
valid-p requirement. Neither adjustment repairs uncalibrated upstream p-values.
Assumptions assessment, finite-sample calibration and global-optimality assessment
remain false throughout. Bootstrap is not implicitly applied to all genes.

## Low-level record fields

The typed report is preferred. Diagnostic numeric slots are 0 baseMean;
1 pointAlpha;2 pointLogAlpha;3 pointCriterionMean;4 pointFullLogLikelihoodMean;
5 localSamplingLogVariance;6 GaussianMappedLogAlpha;7 GaussianMappedVariance;
8 priorCenter;9 finalAlpha;10 finalLogAlpha;11 penalizedCriterionMean;
12 refitFullLogLikelihoodMean;13 contrast;14 SE;15 z;16 p;17 logP;
18 pAvailable;19 logPAvailable;20 pointLeftCensored;21 pointRightCensored;
22 curvatureH;23 curvatureHalfH. Metadata slots are 0 pointStatus;
1 pointProfileStatus;2 pointEvaluations;3 curvatureStatus;4 eligibility
(0 unselected,1 selected,2 unavailable trend representation);
5 refitStatus;6 refitProfileEvaluations;7 covarianceStatus;8 contrastStatus;
9 tailStatus;10 outlier;11 refitRoute (1 penalized profile,2 point prior,
3 unshrunk outlier). Untouched metadata starts at -1. Refit300 means unavailable
prior or ineligible row; 100+ codes identify current-row load refusal and
200+ codes a failed final fixed-alpha fit. Refit301 refuses a censored/endpoint penalized-profile result instead of producing
a conditional test from an unsupported optimum. Other nonzero codes retain the
canonical component status. Per-stage availability must be checked separately.

Validation receipts and bounded scientific qualification are recorded in
`docs/evidence/stdlib-SIO-171/`. A large point-only fixture does not qualify a
full large curvature/prior/bootstrap analysis, and a converged working model is
not a simulation-calibration study.
