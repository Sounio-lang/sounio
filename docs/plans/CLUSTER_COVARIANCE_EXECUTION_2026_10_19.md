<!-- docs:meta
topic_id: repo.docs.plans.cluster-covariance-execution-2026-10-19
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.cluster-covariance-execution-2026-10-19
-->

# Single-way clustered OLS covariance — execution receipt

Date:2026-10-19. Active worktree:
`/workspace/sounio-sprint-stats-viz-main3-20261019`.
Source checkout HEAD observed:`45a345f2e6fac3a97beb4220d5c93f1872cfa7a6`.

> Native acceptance: corrected v2 Slurm15643 passes108/108 with zero failures,
> skips or errors. The native SVG passes the independent six-marker checker;
> three mutated SVG controls are rejected. The failed first candidate and
> intentionally faulted numerical controls remain preserved below.

The previous accepted checkpoint is105/105 in Slurm15633. The new runtime adds
three entries: cluster reference, cluster status and a scientific figure.
The ongoing multi-domain stdlib sprint remains broader than this bounded block.

## Source and scientific contract

`stats::regression::robust_covariance::multiple_linear_cluster` accepts a
borrowed256x8 physical-stride8 OLS design, response, arbitrary signed i64 cluster
labels, rcond and method. The checked OLS fit remains independently available.
Method0 is CR0; method1 is CR1S with the full factor

```text
f = G/(G-1) * (n-1)/(n-p).
```

The result exposes method, groups, correction_factor, covariance/SE and separate
fit/covariance availability. Invalid covariance clears covariance, SE and factor.
Known group count is retained after grouping. Scientific assumptions remain
unassessed. This is unweighted OLS covariance, not Poisson/WLS/M-estimator
inference. No t statistics, p values or confidence intervals are returned.

Production uses existing column-equilibrated/pivoted QR factors and compensated
triangular solves. It restores original coefficient influences and sums them
WITHIN each cluster before the Gram outer product, retaining required cross
terms. CR1S scales completed cluster vectors by sqrt(f) before covariance.
The shared LASSQ/normalized-Gram extraction preserves the HC3 numerical loop
order; no HC3 leverage correction or clamp was changed.

Both methods requireG>=2 andn>p. G<=p is allowed; covariance may be singular.
An available covariance does not certify invertibility or valid Wald inference.
The CR1S multiplier is a stated convention, not a solution to few-cluster
calibration or evidence of independence between clusters.
See [complete contract](../../stdlib/stats/regression/CLUSTER_COVARIANCE.md).

## Independent numerical reference

`tests/reference/linear_cluster_reference.cpp` uses C++23 quad arithmetic,
an independent raw-design MGS point fit, raw cluster scores X_g' e_g grouped
with std::map, and triangular solves AFTER raw-score aggregation. Production
instead groups restored observation influences, providing an independent
aggregation/factorization path. Reported residuals follow the explicit
response-minus-rounded-fitted measurand.

The31cases exercise CR0/CR1S, unequal/noncontiguous groups, arbitrary signed
and relabelled identifiers, row/column permutations, extreme response/design
units, singleton relations to HC0/HC1, zero response, cancellation, rank/input/
method/df failures, covariance underflow/overflow and256x8 designs withG256.
TheG2,p8 case has nonzero SE/covariance, preventing a maximum-shape all-zero
case from standing in for numerical coverage.

Nonzero coefficients use relative1e-9; SE/diagonal comparisons use relative
2e-7 without a1+abs floor. Off-diagonal comparisons use SE_i*SE_j as a
dimensional covariance scale. The analytically cancelling cluster-score case
uses an independent closed form conditional on actual reported residuals;
it does not overwrite numerical residuals with zeros or pretend that this
conditional comparison independently validates the point fit.

Root freshly compiled/executed the C++23 reference with GCC14.2.0,
-std=c++23 -O2 -Wall -Wextra -pedantic -lquadmath, and reproduced the generated
Sounio fixture byte-for-byte (cmp exit0). Numerical source/fixture identity is recorded
below; compiled input manifests and retained run artifacts remain authoritative.

## Native results and controls

| Job | State known at this draft | Scientific meaning |
|---|---|---|
| 15637, first108-entry candidate | 107/108 pass, zero skips; sole visualization timeout | Cluster numerical reference/status and retained HC3 pass; complete closure failed |
| 15638, partition control | Native compilation succeeds; numerical assertion fails | Replaces row_group[row] with row%groups, discarding supplied partition |
| 15639, correction control | Native compilation succeeds; numerical assertion fails | Replaces sqrt(factor) with1 while retaining nominal correction metadata |
| 15643, corrected v2 | 108/108 pass; zero failures/skips/errors | Missing figure loop increment repaired; numerical references, retained corpus and scientific figure accepted |

The two controls are intentionally injected faults, not historical production
implementations. They demonstrate sensitivity to wrong group membership and
missing numeric correction. They do not prove correctness for every design.

The first figure's missing increment is a real observed defect, not an injected
scientific control. Its timeout is retained rather than represented as a
numerical assertion failure. Only the figure was repaired for v2.

Final Slurm15643 accepts108/108 entries with zero failures/skips/errors.
The corpus phase takes212s; the overall run spans18:45:53–18:49:28 UTC,
215s. The summary records three passing layers. These elapsed observations
include compilation/harness activity and are not a paired kernel speed benchmark.
All105 previously accepted entries remain included.

Accepted native snapshot SHA256:
`e4c93240fc9ed22a6d70c2fa51d6f6bc17b4b53a180bcd11f181876996e0859b`.

Standard run artifact root:
`/var/lib/cephfs-slurm/training/sounio-foundry/runs/`.
Accepted v2 run identifier:`cluster-covariance-full-20261019-v2`.
Job IDs, per-entry logs and final summaries must be retained with the actual
run artifacts; a scheduler state alone is not an acceptance receipt.

## Scientific visualization

`viz_cluster_covariance.sio` fits a synthetic8-observation,4-cluster,2-parameter
design and constructs actual Canvas/SVG data from the returned coefficients
and CR0/CR1S standard errors. Its three series contain two points each.
Labels distinguish coefficients from standard errors, identify synthetic
provenance and explicitly state that there are no confidence intervals and
assumptions remain unassessed.

The corrected native visual entry passes in7s under the same30s limit used
for v1. The real SVG passes the independent C++23 six-marker geometry checker.
Three separate SVG mutations (position, count and radius) are each rejected
with exit2. These checks cover actual marker geometry and labels; the fixture
also checks Canvas pixels and scene structure. This is not a complete image
comparison, finite-sample calibration or a biological-data result.

Accepted SVG SHA256:
`a6d29a0119081d8cf6637b65f17731c496278926bf01d02ab149f884cde68ab0`.

Accepted native HTML SHA256:
`a880622165bc51eddc4ebdd8bfde87e683ad7ff6d624b63de1a7431feea3bde9`.

## Source identity at v2 submission

| File | SHA256 |
|---|---|
| stdlib/stats/regression/robust_covariance.sio | 7a203199fe5081069f5a2e42afcdefcca47e9a4c610eb2f8eef66624ee1c2de9 |
| stdlib/stats/regression/CLUSTER_COVARIANCE.md | 3aaf6ec0010c29a7686694921755b04409b5cc247601de35ec7cc2635a879fa0 |
| tests/reference/linear_cluster_reference.cpp | f91b335f30db5c2179fa625ff60a8ec9775c63c1874f39b28b5e39c2a63445a7 |
| tests/run-pass/stats_linear_cluster_reference.sio | 5ce2d6fd79c1a4a0c33e7ac115fff49d534531f8063195380164f7d9e1f77027 |
| tests/run-pass/stats_linear_cluster_status.sio | 2ee7528c2404dc3eb4cc0f7aa0583c47fd6bc52245332e3d6f1b6326be061c1c |
| tests/run-pass/viz_cluster_covariance.sio, corrected | 0808267b07a64498f961cad965108ecf4acc63049b4af0f2dfcd1c0eee956377 |

The corrected figure hash is distinguished from the earlier timed-out source.
Source identities above describe the compiled inputs before these final receipt
edits. Native snapshots and runtime manifests define the exact compiled inputs;
documentation finalized after execution does not alter that native evidence.

## Resources, custody and remaining scope

No heap is introduced by the covariance kernel. Fixed scratch includes group
sum and correction arrays of2048 f64 elements each, bounded256-label/row-index
arrays, R64/scales8 and small row vectors. Returned fit/result arrays and nested
QR scratch contribute additional peak storage. Shared helper reuse and bounded
array sizes do not establish measured RSS or execution-speed superiority.
Figure rendering has its own allocation/resource contract.

Scheduler accounting for15643 reports COMPLETED, exit0:0, elapsed3:35,
TotalCPU6:47.627 and batch MaxRSS7032048K. These are whole two-worker
compile/run-harness observations. They are not isolated covariance-kernel
runtime, aggregate process-tree memory, or a measured kernel peak. No resource
improvement is inferred by comparing these values with a different corpus/run.

Reference calculations/executions use C++23 under ADR009; Sounio execution uses
Forge/Slurm. No Python/Rust/Julia numerical reference was used. Darwin was
consulted before new Sounio; its invalid candidate was corrected, and compilation
feedback alone was not used as numerical acceptance.

Full compiler CI, all stdlib functions, finite-sample coverage, cluster choice,
multiway clustering, estimated weights and few-group t/p/CI remain outside this
receipt. Native108-entry acceptance and the corrected SVG checks are established
for the named snapshot only. Full compiler CI, merge/publication and broader
scientific validation remain separate states.
