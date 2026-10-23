<!-- docs:meta
topic_id: repo.docs.plans.stdlib-deep-2026-10-23
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stdlib-deep-2026-10-23
-->

# Deep stdlib sprint — implementation ledger, 2026-10-23

This ledger follows the approved four-domain sprint. It is an implementation
checkpoint, not a claim that the entire sprint, remote CI, or merge is complete.

## Contract and basis

- Remote main basis: `ce3886de895fa669ab325cb66f23fa5133ef9e90`.
- Isolated workspace: `/workspace/sounio-stdlib-deep-20261023`.
- Dynamic storage and checked APIs are canonical; bounded legacy adapters remain explicit.
- Numerical references: C++23 and Sounio. Selected structural proofs: Lean4.
- No training-quota increase or historical artifact deletion is part of this work.
- One owner per shared file: coordinator owns storage/compiler/QR/eigen/integration;
  statistics owns count models and inference; physics owns dynamic decompositions
  and ODEs; visualization owns checked rendering and plot data preparation.

## Observed native checkpoints

All execution below used the Slurm cpu-ops worker, with inputs and explicit
outputs in scratch. Endpoint hashes cover listed files, not every possible
consumed input or whole-tree immutability. Native execution, numerical
concordance, remote CI and merge are separate acceptance states.

| Checkpoint | Observed result |
|---|---|
|16045|Fresh main compiler build completed0.|
|16046|First typechecks exposed linear borrow consumption and missing ODE effects.|
|16047|Bounded checked ODE and forest/pagination native controls passed; other failures retained.|
|16049|Rebuilt named-borrow fix; the before-rejected linear-owner program compiled and ran. The old forest consumer compiled but failed at runtime before its fix.|
|16052|Lean4.33 `lake build SounioNumericBuffer` passed. Proofs concern structural arithmetic/state transitions, not generated-code or complete IEEE correctness.|
|16055|Buffers, 8192x64 QR, 20000x512 normalization, dispersion working-model component, ODE and special-function status fixtures ran. Initial FDR/axis failures were retained.|
|16059|Optimized QR and eigen controls passed; generated C++ analytic ODE, dynamic128 ODE and special-function callers compiled and ran.|
|16064|FDR actual-output rational reference, fitted-axis Canvas/SVG geometry, million-point original-index provenance, and dynamic line/interval rendering passed.|
|16067|SVD tall/wide reconstruction and Gram checks, allocation-failure rollback, normalization zero policies, and scatter/strided heatmap pixel/SVG references passed. LU/GLM effect/borrow repairs require retry.|

Measured QR runtime on the same structured8192x64 fixture changed from43.50s
to3.14s, both106496KiB peak RSS. This is one case, not a universal speed claim.
The normalization20000x512 case ran80.61s at80384KiB. Million-point reduction
ran0.52s at23552KiB. SVD reference cases, including1024x32, ran0.72s at5120KiB;
the largest case is structured, not a general dense-matrix benchmark.

## Scientific boundaries and next gates

- Fixed-alpha count fitting passed its independent512x12 actual-output reference.
  Deviance across different alpha values is NOT a dispersion likelihood.
- Full NB2 log-probability includes alpha-dependent saturated normalizers.
  Its high-precision reference and profiling remain distinct gates.
- Existing log-dispersion shrinkage is a declared Gaussian working-model component.
  It is not a complete calibrated RNA-seq pipeline or DESeq2 equivalence.
- Profile/Cox–Reid dispersion and log-coordinate MAP passed13native cases and
  independent reference checks. Conditional Fisher/contrasts also passed the
  focused gate. Cohort-refit calibration remains a separate pending layer.
- Dynamic LU/Cholesky, corrected stiff Rosenbrock, scaled plotting,
  distribution summaries and contours passed their focused gates. Integration
  and exact-tip GitHub CI remain distinct gates.
- Compiler callback changes must preserve bare/qualified private-function and
  private-function-value rejection. An intermediate prototype is not a shipping
  visibility policy.
- Packaging, exact-tip CI, independent integration review, PR queue and main merge
  remain required before declaring the integrated sprint complete.

## Verification discipline

A typecheck is not native execution. An available point fit does not establish
valid covariance, p-values, model assumptions or calibration. Displayed supplied
intervals retain their declared kind and level. Reduction never replaces the
original observations for statistical calculations.

Dangerous reads/divisions use explicit control flow. The repository does not
provide a portable effectful short-circuit contract for `&&`/`||`.

## Implementation continuation

| Checkpoint | Observed result |
|---|---|
|16074|Protected current-source compiler: two positive native regressions and seven negative controls passed; private function acquisition stayed denied.|
|16091|Factorization/512x12 count GLM concordance and scaled/distribution/contour Canvas+SVG references passed. NB2 likelihood concordance failed on extreme case58; Rosenbrock exhausted the runtime handle table, exit182.|
|16096|Updated callback effect declarations/engine annotations passed native positive plus three E175 negatives. Instrumented C++ localized the unchanged NB2 failure.|
|16098|Exact-zero LU update skip preserved factorization reference results, but did not resolve Rosenbrock handles. Initial summary typecheck failed on a private floor import; changed to guarded nonnegative rank cast.|
|16099|Histogram/summary and QR-factor fixtures compiled and ran. Initial output comparisons exposed default six-decimal printing; corrected to existing sci_print15.|
|16100|All216 NB2 high-precision actual-output comparisons passed after distant-tail log-ratio fix, including explicit case58 regression. Actual QR factor Gram identity passed. Count sampling library/status fixture passed. Summary reference required materializing a Boost expression-template lambda result; no tolerance relaxation.|
|16101|Integrated gate submitted: buffer/QR/eigen/factors/normalization/ODE/specials/visuals plus count sampling emitters and moments. Results pending; do not treat submission as success.|
|16102|Focused dispersion profile native/reference gate submitted; early module typecheck passed separately. Results pending.|

Compiler PR: https://github.com/Sounio-lang/sounio/pull/3044
Initial commit b048c49bd3ec2e94669acc27191ff4d2d21c73f7; test-contract
follow-up fd7dad09633cac9486a6c520e8e9a300740e47a3. Remote CI and main merge
remain pending. The stage2 suite failures concerned Madaros-only callback tests;
the effect ratchet also identified bare callback types. The follow-up declares
effects and the required engine. It does not excuse untested failures.

Implemented additional contracts: full NB2 scalar likelihood, dynamic
ML/Cox–Reid/log-prior dispersion profiling, fixed-alpha count fitting,
Poisson/Gamma/NB2 checked sampling, exact integer histogram counts and explicit
density normalization, type7 quantiles/Tukey summaries, exact binary64 saddle
signs and degenerate-aware contours. Preparation and acceptance remain distinct.
KDE subsequently passed its native and independent reference gates. Profile curvature is a local diagnostic; inference,
assumptions and global-optimum certificates remain unavailable.

Review found repeated minimum-subnormal quantiles were lost by naive convex
interpolation. The source now preserves equal endpoints and uses same-sign
difference interpolation, with positive/negative minimum-subnormal regressions.
This correction subsequently passed integrated native and C++ validation.

The root gate script registers each C++ oracle explicitly, distinguishes
emitter/standalone/actual-output modes, records compiler/source/executable hashes,
retains timing/stdout/status, and returns failure if any registered gate fails.
Unknown new oracles fail closed. It has been submitted for its first integrated
execution, not yet accepted as fully passing.

## Latest accepted focused gates

Foundation job16114 passed all native/C++/marker/hash checks; eigen beforecontrol
16116 failed with the old phase handling and passed with the fix. Foundation
PR3047 contains the compact receipt and reviewed code (head31a7cc3e4f).
Compiler PR3044 head62ec2985de adds required docs metadata and has auto-merge
enabled subject to GitHub checks; no main merge is claimed.

Science job16118 passed13dispersion-profile native cases and actual C++
concordance, KDE status and12generated C++ cases, and Rosenbrock original plus
additional nonautonomous/autonomous-augmented cases. Corrected Rosenbrock ran
1.28s at3072KiB RSS. The official Boost1.83 positive-d4 pure-time control fails
(exit1), corrected pure-time and adaptive controls pass, and the autonomous
augmented control passes with the correctly paired C++ include closure.

The d4sign correction is grounded in WRF/KPP RODAS4 pinned source commit
06d4240ae989cc3e50af412bb472df3d9048783c, raw SHA256
44eb88bcbd91ab2f12f37ba71d314038d6a771f1a8750f5f8b8c2b317940fa2a.
The numerical reference explicitly uses the corrected coefficient while
preserving the unmodified upstream red control; it does not disguise parity
with the known-bad nonautonomous coefficient.

Bootstrap fixed-alpha calibration and dynamic incremental count TSV subsequently
passed their focused and integrated native/reference gates described below. Full count-table IDs
and sample joins, profile-refit cohort calibration, general data validation,
finite-sample inference and dedicated CI bridge remain open. No DESeq2 parity
or universal numerical/visual performance claim is made.

## Integration checkpoint: final science source

The later checkpoints supersede the pending statuses above. Full integrated
job16139 passed the scientific kernels, references, sampling, renderers and
source/compiler integrity checks. Its only failures were the NB2 slot fixture
and corresponding concordance: the fixture demanded bit-exact analytic -1
instead of testing snapshot preservation. Directed slot diagnostics localized
that assertion. The corrected fixture stores the original scalar before slot
reuse and checks exact snapshot stability; the independent C++ oracle retains
its existing numerical tolerance for analytic -1. slots_final passed all four
status files, including corrected slots and independent PTRS reference.
The tiny-positive-alpha input now constructs a positive minimum subnormal
explicitly instead of relying on an underflowing decimal literal.

Accepted later evidence:
- job16124: copied RNG cursor ownership, exact PCG state/counters, incremental
  TSV status/stream references, selected RODAS rational Lean proof.
- job16130: all count GLM/Fisher/profile/bootstrap references and 256 Big400
  paired-likelihood increments. The preserved bootstrap replica required a
  positive likelihood gain near 1.95e-15; a compensated local paired formula
  resolves cancellation without loosening convergence tolerances or budgets.
- job16133: 26 Big400 normalization precision cases and exact selection
  partition tests. The same structured 20000x512 normalization workload fell
  from80.61s to34.79s at80384KiB RSS. These are workload measurements.
- job16139: reused NB2 slots in GLM/profile/bootstrap and one million PTRS
  Poisson(mean40) draws passed independent binary128/state/counter checks.
  The PTRS run measured14.49s and1600KiB RSS; the earlier-source paired
  control measured15.20s and1600KiB. No PTRS memory improvement is claimed.
- Separate one-million-draw controls measured uniform RSS296960->1600KiB and
  Poisson(mean0.01) RSS522752->1660KiB after reusable cursor APIs. The APIs
  preserve explicit domain, precision, budget and unavailable statuses.

Compiler PR3044 is in the protected merge queue; foundation PR3047 and scientific
2D PR3048 remain dependent until actual prerequisite merges are observed.
The final science/CI changes will receive a fresh exact-index native replay
before publication. The new CI path reuses the existing current-source witness
ELF, verifies source/ELF identities, fails closed on missing required results,
and retains compact receipts. Its operational tamper/selection controls passed;
actual GitHub execution is a separate acceptance boundary.

The fixed-alpha bootstrap does not implement profile-refit calibration or prove
finite-sample validity. Profile curvature is a local diagnostic. The Lean result
proves a selected exact rational clock relation, not the full floating-point ODE
solver. Incremental TSV does not implement sample joins or a full RNA pipeline.
No DESeq2 parity, universal performance or visual superiority is claimed.
