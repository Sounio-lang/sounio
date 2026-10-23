<!-- docs:meta
topic_id: repo.docs.evidence.domain-log1p-2026-10-21.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.domain-log1p-2026-10-21.readme
-->

# Shared log1p genomics/physics evidence

Slurm15826 uses frozen source ab10a22026a2056ecfe8f3d22c54b48dbdd106ef and a fresh source-tracking seed/modular compiler build. Four selected typechecks precede compilation/execution of the real synthetic example. No tests or independent numerical oracle are included. Generic Forge corpus_smoke naming is transport only.

Raw logs and manifests are losslessly compressed, with original hash sidecars. The native ELF is compressed with raw identity recorded. Compiler binary custody stays remote, also copied to t560 ext4 and verified after decompression; large compiler binaries are excluded from Git.

figure.html and figure.svg are extracted from actual stdout. SVG interior is unchanged with a trailing newline. preview.png is the full-size sips rendering. Native process measurements cover one complete fixed example, not isolated kernels or comparative benchmarks. Numerical concordance, boundary coverage, calibration, scaling, full corpus/CI and merge are separate.

The SVG passed XML parsing and its full 1380x620 PNG rendering was visually inspected. Both panels, axes, units, legends and caveats are visible. No Canvas pixel comparison was performed.
