<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-p0-sio-93.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-p0-sio-93.readme
-->

# SIO-93: current-main logarithm boundary reconciliation

Notion task SIO-93 requests integration of PR2732. Fresh origin/main be761447e1cb21fd0f9113dccf20c9e2aed8c7a4 already contains the guard in ln_core and the original regression byte-for-byte. The guard was included in the earlier integrated scientific stdlib work (df8bbc8330). This change preserves production code and extends the regression with raw binary64 rows and an independent C++23/quadmath reference, registered in the central deep-science gate.

Slurm17062 completed0 in9seconds on the t560 CPU worker asUID1000, stack524288KiB. Compiler SHA d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6 identifies the retained source-built compiler from the previous sprint; it is not a newly rebuilt compiler from current main. Current library inputs and the source fixtures are bound by the archived INPUTS manifest, verified unchanged after execution.

The native fixture emits11rows covering1,2,0.5,largest finite, smallest normal, smallest subnormal, positive infinity, NaN, negative infinity, zero and negative one. C++ checks actual input identities, completeness, duplicate rows, NaN/infinity classes and finite/domain outputs against binary128 log references. The declared bound is4ULP; observed maximum3ULP. Domain sentinels remain legacy behavior, not standard log domain results. Logarithm accuracy for all inputs is not certified.

The C++ reference also checks guard classification on8187positive finite exponent/extremal-significand samples, not all binary64 inputs. A copied stdlib with only the guard removed compiles successfully but times out124 after3seconds on the same regression. Current-main execution and C++ concordance exit0. This is a discriminating negative control, not a claim that current main still hangs.

Reproduce on native Linux with a source-built compiler:

MADAROS_RAW_BIN=/absolute/compiler LN_BOUNDARY_OUT=/absolute/empty-output bash scripts/stdlib/math_pure_ln_boundary_gate.sh

The archive retains actual status/stdout/stderr, resources, input/compiler/executable hashes and expected-timeout control receipts. Generated ELF/compiler/reference executables and the copied control stdlib are excluded. Historical PR2732 stays open for human reconciliation; no merge, closure or automatic rebase was performed. Native replay is distinct from PR CI acceptance.
