<!-- docs:meta
topic_id: repo.docs.plans.domain-log1p-execution-2026-10-21
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.domain-log1p-execution-2026-10-21
-->

# Genomics/physics shared logarithms: execution receipt, 2026-10-21

## Frozen source and integration

Branch `sprint/genomics-physics-log1p-main-085da-20261021` retains remote main
`085da1da15f3c05946b66e4ddf7b60a0e29d5d40` and the earlier sprint increments.
Frozen executable source: `ab10a22026a2056ecfe8f3d22c54b48dbdd106ef`.
Later documentation/evidence commits are not relabeled as executed source.

Two private logarithmic series are removed. Genomic `sf_log_ratio` delegates only
the existing |d|<=0.5 branch to `math::logarithmic::log1p_checked`; integer
count-anchor subtraction still precedes f64 conversion. Far checked-log subtraction,
zero exclusions, median calculation and centering are preserved. Failures map to
existing status4. Physics `entropy_log_ratio` preserves exact equality zero and
far checked-log difference; its near-volume branch delegates to the same helper.
Scaled products, covariance propagation, n=0 behavior and public contracts remain.
These source changes do not establish last-bit equivalence or improved accuracy.

## Real scientific consumer

[Example contract](../../examples/viz_genomics_entropy.md).
A synthetic4-gene/3-sample matrix includes three eligible genes and one excluded
zero-containing gene. The left panel plots centered log factors in categorical
sample order. The right plots nominal ideal-gas entropy at seven volume ratios,
with n=1mol and supplied amount/volume uncertainties. Both independent and
correlated-volume APIs run; rho=0.5 is declared, not estimated. Standard
uncertainties are report-only, first order and not confidence intervals. These
are independent domain demonstrations, with no biological inference or implied
cross-domain physical relationship. Ordinary report formatting/full-scale
coordinates cannot resolve every tiny contrast; no tiny-contrast certification
is claimed. No assertions or expected numerical oracle are embedded.

Two1024-element input buffers and a64-byte label buffer are reused. Builders copy
active prefixes into scene-owned storage. SVG-only output avoids Canvas
allocation; this is not a claim that the whole program is allocation-free.

## Execution evidence

Slurm **15826 COMPLETED 0:0**, elapsed **00:07:00**, TotalCPU **06:56.392**,
batch peak RSS **2898824K**. A fresh source-tracking seed and modular Madaros were
built from the frozen input snapshot. No previous raw compiler was substituted.
The resulting raw compiler happens to match the previous build hash; this does
not change the fresh-build provenance.

Four explicit typechecks passed:

- stdlib/math/logarithmic.sio
- stdlib/genomics/normalization.sio
- stdlib/physics/entropy.sio
- examples/viz_genomics_entropy.sio

The real example compiled and executed with exit0. Completion marker:
CURRENT_SOURCE_DOMAIN_LOG1P_DEMONSTRATIONS_READY.
GNU time records elapsed/user/system **0.00/0.00/0.00s**, peak native-process RSS
**1060KiB**, exit0. Zero displayed centiseconds is reporting resolution, not zero
execution time. This is one fixed whole-example observation; compiler RSS above
is separate. No isolated-kernel or comparative speed/memory claim follows.

| Artifact | SHA-256 |
|---|---|
| Raw compiler | `0c896942018ff4be9a86d69f30e3a8aadac955f32451c445a07fbf502ae37fc3` |
| Source snapshot | `339933a8036788c51760c3e1bd99673909310f7d9f317ad86536544217f36514` |
| Raw input manifest | `8ce95ddb4464f4f655907ebe692f04f932bdaedd84ae8c6f24b74f36edae93a5` |
| Raw check/run log | `f6d7c92da095bdb977afd49eed37fd2b2be9886683b3a174967e658fd3c78ed2` |
| Example ELF | `c77edb36c84dfe0b88df85d9344291c6df7dce6f3fce588479c59c24c79a18cf` |
| Example stdout | `541b93e03ac14cb5b062fee04ff847e8518c6f9ed04c8a9e6bc1c55bbace7472` |
| Example SVG | `425b008bbb9c854e93127c703f6b1f03830069a6ec07896b1efeecd90ac932bf` |

The actual SVG parsed with xmllint --nonet --noout, exit0. Its complete1380x620
sips rendering was visually inspected: both panels, signed entropy scale,
categorical sample axis, units, legends and caveats are visible. There is no
Canvas execution/pixel comparison in this example.
[SVG](../evidence/domain-log1p-2026-10-21/viz_genomics_entropy/figure.svg),
[full preview](../evidence/domain-log1p-2026-10-21/viz_genomics_entropy/preview.png),
[raw report](../evidence/domain-log1p-2026-10-21/viz_genomics_entropy/report.txt),
[evidence inventory](../evidence/domain-log1p-2026-10-21/README.md),
[file hashes](../evidence/domain-log1p-2026-10-21/FILES.sha256).
Last storage observation:524288000bytes available within the unchanged quota.

## Acceptance boundary and custody

No tests were added or run. Independent C++23/Sounio concordance, global ULP
analysis, explicit boundary coverage, model calibration, performance comparisons,
full corpus, CI and merge remain open. Historical114/114 remains historical.
Generic Forge transport labeling such as corpus_smoke does not constitute
full-corpus acceptance.

Raw logs and input manifests are losslessly compressed with original hashes.
Compressed native ELF is preserved with its raw identity. Large compiler and
source packets remain remote; compiler custody also uses t560 ext4. Quota remains
3TiB with no new removal or quota change. File/hash checks are artifact integrity
operations, not numerical acceptance.

The next bounded statistics candidate is supplied-parameter NB2 per-count
saturated deviance and signed deviance residual. It is not implemented here.
Its contract must handle near-mean cancellation, alpha->0 without overflowing
1/alpha, large alpha*mean, and separate deviance/residual representability where
needed. Exact count/equality/Poisson branches and parameter-assumption disclosure
must remain explicit. Neither a p-value nor calibrated model comparison follows
from such a scalar diagnostic.
