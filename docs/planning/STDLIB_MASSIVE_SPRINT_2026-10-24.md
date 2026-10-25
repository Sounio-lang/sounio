<!-- docs:meta
topic_id: repo.docs.planning.stdlib-massive-sprint-2026-10-24
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.planning.stdlib-massive-sprint-2026-10-24
-->

# Massive stdlib sprint: execution ledger

Approved by Demetrios Agourakis, 2026-10-24. Baseline for the first implementation
wave: remote main `a2a44df195bb8caf3fb07cbc9517275b29325e94`.
Canonical backlog: [SIO-161](https://app.notion.com/p/4033f37f937181028f25c45ef1ba6c0a).
This ledger records scope and dependencies; implementation evidence is recorded
per package. A task or accepted ADR is not evidence of executable completion.

## Execution contract

Remote worktrees with devpass branches and exact-file coordination claims; one
owner per shared file. Native compilation and numerical verification on Slurm,
UID1000, stack524288KiB. C++23/Sounio are concordance references under ADR009;
Lean4 handles proofs. No Python/Rust/Julia numerical oracle. Retained compiler
artifacts are identified by checksum and never described as current-source builds.
Freeze each source snapshot before execution and retain failed attempts separately.

Fix legacy valid outputs and preserve names/signatures where feasible; invalid or
unavailable calculations refuse explicitly. Approximation APIs are named as such.
Caller-owned reusable buffers replace arbitrary fixed-capacity increases. Report
execution time, peak RSS, retained bytes, allocation/work counts independently.
Use three paired AB/BA/AB observations for performance claims, with identical
inputs, compiler/options and core. Do not trade correctness for apparent speed.

Commit and push with lease to owned branches; PRs remain for human review.
No merge, enqueue or auto-merge to main. Preserve the shared3TiB quota and historical
artifacts. Existing qualification campaign17048 is retained, not duplicated.

## Packages and acceptance

| Task | Deliverable | Dependencies and acceptance boundary |
|---|---|---|
| SIO-162 | Checked Normal/Beta PDF, logPDF, CDF/SF, logtails and quantiles | Reuse canonical gamma/beta/math; native Beta uniform regression; direct small tails; explicit availability, work and budgets; C++ Boost/binary128 concordance. Iterative convergence is not a certified error bound. |
| SIO-163 | Checked dynamic radix2 FFT with reusable plan | Real4096 needs2049 bins; legacy fixed output refuses before writes. Caller-owned plan/work/output; cached twiddles/permutation; explicit normalization; small direct DFT, Parseval, roundtrip/convolution, DC/Nyquist; paired4096 runs. |
| SIO-164 | Checked differentiation and dynamic scratch | Representable distinct points, finite callbacks, explicit evaluation budget/count; derivative/error estimate/step; reusable gradient/Jacobian/Hessian scratch. Estimated error is not a bound. |
| SIO-165 | Classical/SR Jacobian uncertainty | Zero nominal with nonzero sensitivity; nonnegative signed-velocity uncertainty; declared covariance; scaled norms and real imports; analytic C++ references and Lorentz invariants. No model calibration claim. |
| SIO-166 | Truthful checked BDF | Linear/Newton failure reaches caller; no silent singular-zero or tolerance relaxation; actual endpoint/progress/budgets and RHS/Jacobian/factorization counters; stiff analytic/Robertson references. Preserve real dynamic Rosenbrock path. |
| SIO-167 | Genuine noncentral-t power/sample size | Depends on162; bidirectional Poisson/beta mixture about largest weight; direct tails; explicit method/work/cap statuses; power(n-1)<target<=power(n); approximation separately named. |
| SIO-168 | Exact dynamic signed ranks | Bounded DP states, declared ties/zeros/method, no silent asymptotic fallback; separate rank-biserial and z/sqrt(n); indexed O(n log n) sort and C++ enumeration. |
| SIO-169 | Dynamic KM/Greenwood/logrank | Complete curve/risk/event/censor data, event-censor tie policy, explicit CI availability; shared ordered risk representation and100k fixture. No Cox/left truncation claim. |
| SIO-170 | Genomic IDs/joins/CSR | Header and identifier custody, duplicate reject/coalesce policy, validated sparse view, no implicit normalization changes; reuse existing parser/normalizers. |
| SIO-171 | Batched genomic fitting/shrinkage/FDR/report | Depends on170; reuse real MLE/trend/prior; eligible curvature and conditional refit; failed tests retain family denominator and IDs; BH/BY; one scratch per worker,20k genes x512 samples. No DESeq2 parity claim. |
| SIO-172 | Checked phonon covariance/sensitivities | Authoritative import; Sigma'=A Sigma+Sigma A-transpose, parameter sensitivities, explicit optional Q; no clamping; decay/eigenmode/conservation/progress witnesses;64 sites. |
| SIO-173 | FEM1D and Poisson2D multigrid | Replace copied-helper tests with production imports; checked geometry/grid/SPD/capacity; small rational stiffness and manufactured refinement;257x257 O(N) workspace. No generic unstructured solver claim. |
| SIO-174 | Dynamic DP54 FSAL/dense/events | Reuse accepted-step FSAL only, quartic dense output, bracketed events and root/progress budget, actual RHS counts. Rosenbrock dense extension outside package. |
| SIO-175 | Prepared shared contour geometry | Caller-owned segments/cell-level-branch provenance, exact capacity, preserve saddle/shared-edge/mask topology and transactional failure; Canvas/SVG consume identical geometry;256-square and1024-square dual export. |
| SIO-176 | Checked color scales/canonical LUTs | Authoritative licensed256-entry tables, domain/under/over/missing policy including constant domains; raster/SVG/colorbars agree; no per-cell owner allocation. Keep legacy approximations explicit. |
| SIO-177 | Interval ABI/frontend/IR and separate e/u | ADR019; no implicit folding or point-Jacobian certified fallback; FP-state effect barriers/opaque kernels; exact host-independent literals; uncertain branch refusal. Structural contract alone is not the primitive. |
| SIO-178 | EVEX/RISC-V/RVV/PTX and Lean qualification | Depends on177; no YMM ER/FTZ/approx; one million deterministic cases per format/operation plus edges/mutants; bit identity across targets; sorry-free enclosure proofs with rounding axioms explicitly NOT ESTABLISHED against bit model. |
| SIO-179 | Qualification, resource regressions and custody | Finish existing NB2 campaign with per-cell denominators, seeds, failures, MC uncertainty and timeouts; pilots separate; immutable compiler/catalog checks before/after; no automatic assessed/calibrated labels. |

## Waves and ownership

First wave:162 distributions,163 FFT and165 physics are independent implementation
lanes; root owns177 decisions/contract and shared runner registration. Workers
supply focused gates without concurrently editing `deep_science_gate.sh`.
Next numerical wave:164 differentiation and166 BDF; statistical/genomics wave:
167–171. Physics/render wave:172–176. D34 advances alongside independent science,
with existing compiler owner coordination before shared frontend/backend edits.

Existing PR3093 (log infinity),3094 (TypeA uncertainty) and3095 (provenance trust)
remain separate review deliveries. Their focused receipts do not establish all
new package gates or whole-corpus success. Four end-to-end scientific examples
(statistics, genomics, physics and numerical trajectories) must exercise actual
imports and shared figure metadata once their dependencies are implemented.

Typed zeros SIO-87 and meet/contest SIO-88 follow ADR016/018. Censored logarithm
support retains symbolic log(LOQ) until qualified transcendental endpoints exist;
M3 consumes the original LOQ. No LOQ/2 substitution, implicit absence-to-zero,
D22 blood/plasma decision, D27 chemistry golden regeneration, general certificate
kernel or physical/clinical calibration is implied by this programme.

## Current execution state

- Accepted design: ADR019, with historical D34 alternatives retained.
- First reviewed PR deliveries:162 ([3102](https://github.com/Sounio-lang/sounio/pull/3102)),163 ([3104](https://github.com/Sounio-lang/sounio/pull/3104)),165 ([3103](https://github.com/Sounio-lang/sounio/pull/3103)); fresh-tip CI/human review remain separate.
- In progress:164 differentiation,166 BDF,167 noncentral-t/power,172 phonon transport;177 has first structural/ADR increment in [3101](https://github.com/Sounio-lang/sounio/pull/3101), with compiler integration still pending.
- Other packages: approved scope, not yet implemented by this wave.
- Final programme completion requires all package gates, integration review and
  scientific examples; no task is marked done merely because CI or a pilot passes.
