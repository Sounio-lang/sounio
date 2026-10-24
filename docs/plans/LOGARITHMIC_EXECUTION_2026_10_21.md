<!-- docs:meta
topic_id: repo.docs.plans.logarithmic-execution-2026-10-21
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.logarithmic-execution-2026-10-21
-->

# Checked logarithmic primitives: execution receipt, 2026-10-21

## Source and scope

Remote-main basis: `085da1da15f3c05946b66e4ddf7b60a0e29d5d40`, retained together with the prior sprint increments on branch `sprint/logarithmic-main-085da-20261021`. Implementation commit: `5708a48a22684f242b3fa5d8d376d6389c456b68`. Final executable source: `27ea4ec6c2b97892e9e92cf982b0af90e2241023`; subsequent documentation is not relabeled as executed source.

`stdlib/math/logarithmic.sio` introduces `log1p_checked`, `log1pmx_checked` and `log1p_phi_checked` using the existing CheckedLog contract. All accept finite x > -1. Status 1 denotes finite domain failure; status 2 denotes nonfinite input or unavailable arithmetic. Quadratic results retain explicit sign guards and normalize zero to positive zero. No epsilon sign clamp is used. The scalar kernels use no arrays or explicit heap allocation.

The near-zero logarithm uses a compensated atanh series, preserving tiny arguments before halving. The quadratic forms use factored compensated degree-32 series, avoiding an early x*x. Larger arguments use checked logarithms and explicit arithmetic guards. This is an implementation description, not a global ULP or correct-rounding certification.

Four private helpers were removed across information theory, channels and Poisson. Existing failure statuses, equal/zero cases, outer compensation and specialized JSD/BSC series remain. The NB2 example exercises the changed Poisson consumer, renders Canvas in memory and emits an SVG. The new mathematical example emits two panels with seven explicitly disclosed samples, actual transformed log10 coordinates and four clipped series.

## Native execution

| Slurm job | Source | Outcome | Elapsed | Total CPU | Batch peak RSS |
|---|---|---|---|---|---|
| 15823 | 5708a48 | COMPLETED 0:0 | 00:07:13 | 07:09.365 | 3563256 K |
| 15825 | 27ea4ec | COMPLETED 0:0 | 00:00:25 | 00:24.671 | 3566940 K |

15823 freshly rebuilt the source-tracking seed and modular Madaros. 15825 reused that exact attested compiler after an empty `git diff --exit-code 5708a48 27ea4ec -- bin scripts self-hosted stdlib`. The only executable-source change was the explicit sparse-sampling caption. Compressed compiler input was hash-verified after decompression; no stale fallback was used.

Each job typechecked these seven targets: logarithmic, channels, probability, Poisson, NB2 diagnostics and both visualization examples. Each compiled and executed both native examples with exit 0. Final completion marker: `CURRENT_SOURCE_LOGARITHMIC_DEMONSTRATIONS_READY`.

| Final native child | Elapsed / user / system | Peak RSS | Exit |
|---|---|---|---|
| Mathematical logarithmic figure | 0.00 / 0.00 / 0.00 s | 1576 KiB | 0 |
| NB2/Poisson count figure | 0.01 / 0.00 / 0.00 s | 8768 KiB | 0 |

GNU time observations cover one whole fixed example each. A displayed 0.00 s is below centisecond reporting resolution; it does not mean zero execution time. Compiler/transport RSS is recorded separately above. No isolated-kernel, scaling or comparative speed/memory claim follows.

## Exact identities

| Artifact | SHA-256 |
|---|---|
| Built/reused raw compiler | `0c896942018ff4be9a86d69f30e3a8aadac955f32451c445a07fbf502ae37fc3` |
| Final source snapshot | `860899e2720636406a7b273fb478e2647aa1a2cdbf3014261c803655e5e12109` |
| Final package input manifest | `38041afebceea41049cd450888750911413bbc62214c5f2e87d2ba52b758760f` |
| Final raw check/run log | `be1b21a74e17144e44781aea59ade2abcd60300c4857f295f6b0e8b83ddf3831` |
| Mathematical ELF | `c137e7474e1ca66b483c3451764cd9dfb202a913fe574d0607d5eedd8a129669` |
| NB2 ELF | `cd514796f69585d85b6ca85a91564d0b82322c78aaaa85cbafb010bc01ce1465` |
| Mathematical stdout | `17018168d668ee34b4f3d1f48788adeeeb059690ae97167f0eaf461dc1a4e352` |
| NB2 stdout | `c4638d32173f59987360db4d33fae97f903ca71e1fe884ba44e8ac9a6709f9bc` |
| Mathematical SVG | `021dfdc10587d293386daf8f734e84665405c7b384121e64db91e8fc2fba1d56` |
| NB2 SVG | `0eddee70af81a88fb09701a8983086cb24cbc9ee651f16b9780fd0e8347a37ae` |

NB2 stdout is byte-identical to the initial run. This establishes formatted artifact identity only, not independent numerical concordance.

## Figures and custody

Both final SVGs parsed with `xmllint --nonet --noout`. Both complete 1380x620 PNG renderings produced with sips were visually inspected: panels, axes, four mathematical series, legends, negative NB2 labels, units and caveats are visible. No Canvas pixel capture or comparison is claimed.

- [Mathematical SVG](../evidence/logarithmic-2026-10-21/viz_logarithmic_primitives/figure.svg) and [preview](../evidence/logarithmic-2026-10-21/viz_logarithmic_primitives/preview.png).
- [NB2 SVG](../evidence/logarithmic-2026-10-21/viz_nb_dispersion_diagnostic/figure.svg) and [preview](../evidence/logarithmic-2026-10-21/viz_nb_dispersion_diagnostic/preview.png).
- [Evidence inventory](../evidence/logarithmic-2026-10-21/README.md) and [relative file hashes](../evidence/logarithmic-2026-10-21/FILES.sha256).

Raw build logs and input manifests are losslessly compressed with original hashes; decompression identities were checked. Native ELFs are compressed and checked against recorded raw hashes. Large source snapshots/compiler binaries remain in remote custody; source packets and evidence staging use t560 ext4. Quota remains 3 TiB; the last storage observation recorded 562036736 bytes available. No further deletion or quota change was performed.

## Acceptance limits and continuation

No tests were added or run. These are seven selected typechecks, two real native example executions and complete figure inspections. Generic transport naming such as check_corpus is not a corpus acceptance result. Independent C++23/Sounio numerical concordance, ULP analysis, explicit boundary runtime coverage, model calibration, performance comparisons, full corpus, CI and merge remain open. Historical 114/114 evidence is unchanged.

Next audited candidates are `sf_log_ratio` in genomics normalization and `entropy_log_ratio` in physics. Preserve exact i64 subtraction before conversion in genomics, zero exclusions, median/centering semantics and the far log-difference path. Preserve exact equal-volume zero, scaled products, covariance and n=0 semantics in physics. These migrations are identified, not implemented by this receipt.
