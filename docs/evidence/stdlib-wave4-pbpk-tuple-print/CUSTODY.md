# Baseline custody

All numerical verification is C++23. These jobs characterize the unchanged raw04; they do not validate the proposed checker implementation.

- Raw04 SHA256 `a4f5e8d655754202493bbb04d8fbb15890b10e8fc861ac3dc60438452704c333`.
- Matrix19743: COMPLETED0,13s, observational capture with expected compile refusals; initial invalid nested attempts are retained.
- Corrected nested19746: COMPLETED0,6s, two valid-step negative controls and primitive-annotated positive.
- Nested scalar-only19756: COMPLETED0,4s, check/compile/native/C++23 oracle0 in both loop modes, without annotating the aggregate temporary.
- Export19762: COMPLETED0,12s, real OrangeFS mount verified, archive list audit, SHA readback and before/after manifests identical.

Durable archive:
`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/pbpk-tuple-print-export01/pbpk-tuple-print.tar.gz`

SHA256 `3267bdc32e7311b2d17afbfe2abf23d007c01e5d81a48e4142c486f38298871e`.

The archive contains the full three matrix directories, their immutable source/stdlib snapshots, exact Slurm recipes, C++ oracle sources/executables, native ELFs, full logs and source/tool/original/executable guards. It remains authoritative for the first nested attempts, which differ from the corrected committed fixture. The committed oracle includes the final scalar-only mode; earlier oracle revisions remain in the corresponding snapshots.

Original PBPK frontier remains separately archived by root:
`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/root/pbpk-compile-frontier04/pbpk-frontier.tar.gz`, SHA256 `896b7516ae6e26f0145f4216ed1d00e9b2c5086697203394a9b190533f6b763c`.

Raw04 source/seed/SDK/ELF custody:
`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/raw04-export02/raw04-sources-seed-sdk-elfs-gates-v2.tar.gz`, SHA256 `22e7f121aff116216da7f1acc35221acdcbde5d92c63b36aa60b9d89f0e43b60`.
