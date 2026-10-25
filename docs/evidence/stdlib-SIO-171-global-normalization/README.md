<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-171-global-normalization.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-171-global-normalization.readme
-->

# SIO-171: global reference normalization reused by bounded workers

Based on published integration3c7c8073411a7e7254d0d9424f59f8e49eb49180 in a
separate worktree. Job17578 and its source remain immutable and do not validate
this increment. Source/references and final qualification are pending review.

CountNormalization is a sealed owner constructed only by the existing canonical
CSR median-log-ratio estimator in explicit mode0 or1. It retains factors/logs,
unique byte sample IDs and legitimate reference genes/used/excluded metadata.
It offers no arbitrary factor import or DynamicSizeFactorsSlot setter.

The existing local public pipeline API uses the same internal constructor,
workspace and preparation validation/commit helpers. The global variant omits
only norm_work and norm_anchors; factors/logs and all fit/inference buffers remain.
It performs an exact unique cohort byte-ID join, never positional matching,
local median estimation or re-centering. Log exposure is added exactly once.
The metadata and normalization joins share one declared byte-comparison budget.
A failed preparation remains at phase0, where public factor/offset accessors
cannot expose private intermediate buffers and reference metadata is unavailable.

Reports distinguish local gene rows from the global reference population;
used+excluded=reference_genes. The original local TSV schema is unchanged;
global-reference TSV adds an explicit normalization-scope header.

Resource queries describe allocated buffer payload only, excluding transferred
CSR/identity inputs, owner metadata, allocator overhead and process RSS. Building
a global owner still needs a global CSR plus temporary O(G+S) canonical scratch;
scratch closes before the retained O(S) factor/log owner returns. The owner is reused by borrowed access within one process; transport or import
across the original independent Slurm array processes is not implemented or
qualified. This is not
an out-of-core algorithm, median-of-medians approximation, universal RSS/speed
claim, global empirical-prior/FDR qualification or a new20k calibration campaign.

The proposed tiny fixture uses six heterogeneous genes, both zero policies and
normalization modes, four3+3 partitions and different sample permutations. It
compares reused factors/offsets bitwise with the same-order monolithic canonical
estimator, and checks actual outputs against independent C++23 binary128 raw
logarithms and sorted medians. Eight actual point-fit means confirm offset
consumption. Budget/phase/identity/refusal controls and a genuinely recomputed
local-factor native mutant discriminate the global reuse behavior. Six existing
pipeline fixtures in both engines remain required regression checks.

Development Slurm17579: library modules checked on modular a292; fixture failed
because a cast bound to the right modulo operand. Only fixture parentheses were
corrected. Slurm17580 checked the library modules in both engines but refused owners closed
inside fixture loops in the bootstrap checker. Moving their lifetime into small
functions preserved the algorithm; Slurm17582 then passed all source checks but
failed strict C++ compilation on misleading indentation. That formatting was
corrected without changing the reference estimator.

Slurm17584 COMPLETED0:0 in22s: both native engines and the independent C++23
reference passed16 scenarios,64 sample bindings and8 actual point-fit means.
The original unused-helper private-field control was rejected by modular a292
with E259 but accepted by seed13003 check. Its original source is retained as
unused-helper-check.sio; the exact stdout/status will remain in final custody.
This is an explicit compiler check-path limitation, not a pass for that control.
A further live-main constructed-owner field-read control was rejected with E259
by both check paths in Slurm17585; the final gate requires actual compile
refusal with E259, not a filename match, parse/type failure or check-only result.

The final expanded gate, source/toolchain closure and six-fixture regression
replay remain pending. No full final qualification is inferred from the prototype.

Tracking: [SIO-171](https://app.notion.com/p/4033f37f93718146a307f9f9616058c7).
