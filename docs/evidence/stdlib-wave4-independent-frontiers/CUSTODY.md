# Durable custody

Export Slurm19831 completed0/20seconds. It verified all three diagnostic input manifests, retained-native inputs, observability inputs and the full current source contents, then read the completed archive back with `sha256sum -c` and `tar -tzf`. The diagnostic-tree hash and full source contents were verified again afterward. Final complete path/mode/link census matched the original Git tree. No source edits or compiler execution occurred in this job.

Directory:

`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/independent-frontiers-export01`

Archive: `independent-frontiers.tar.gz`

SHA256: `3775a0c443057a38b3ad1a6e388dfff342e595e90fba9342f3415cce61f9a8fb`

The archive contains the complete diagnostic directory, including the two failed harness attempts, accepted raw04, exact current SDK stdlib, original three tests, original DP54 reference, source/content and Git census manifests, private toolchain env/wrapper references. It preserves the full-source custody reference `4a300889...40b9bcf8` described in README. The full source and private SDK dependencies remain bound by the original immutable custody rather than falsely claiming they were newly built here.

`receipts/` keeps compact job/process/result/oracle/census records, actual stdout for all four original native runs, the companion's actual numeric stdout, and the literal DP54 C++ function-body hash. The full untruncated diagnostics, compiler logs, timing, mutant outputs, failed attempts and input/ELF guards are in the archive.

Commits are additive artifacts only. The fixtures commit accidentally included an AppleDouble metadata file; the receipt commit deletes it. Apply both commits together. This file was never compiled or used by a gate.
