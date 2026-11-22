<!-- docs:meta
topic_id: repo.docs.evidence.nb2-deviance-2026-10-22.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.nb2-deviance-2026-10-22.readme
-->

# Shared log1p genomics/physics evidence

Slurm15854 uses frozen source 76f30851adaace93d0dfc5ff4c0d3b8df3741973 and a fresh source-tracking seed/modular compiler build. Four selected typechecks precede compilation/execution of the real synthetic example. No tests or independent numerical oracle are included. Generic Forge corpus_smoke naming is transport only.

Raw logs and manifests are losslessly compressed, with original hash sidecars. The native ELF is compressed with raw identity recorded. Compiler binary custody stays remote, also copied to t560 ext4 and verified after decompression; large compiler binaries are excluded from Git.

figure.html and figure.svg are extracted from actual stdout. SVG interior is unchanged with a trailing newline. preview.png is the full-size sips rendering. Native process measurements cover one complete fixed example, not isolated kernels or comparative benchmarks. Numerical concordance, boundary coverage, calibration, scaling, full corpus/CI and merge are separate.

Initial job15853 was deliberately cancelled during its build before checks, after review identified missing canonical sqrt Panic effects. Its partial log and exact source identity are retained separately. Final15854 freshly rebuilt corrected source76f308 and passed all four checks plus native execution.

The final SVG passed XML parsing and its complete1040x620 sips rendering was visually inspected. No Canvas execution/pixel comparison occurred. Extraction recognizes the lowercase HTML doctype, preserving the complete HTML.
