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
this increment. Source/references were reviewed by the coordinator before the final allocated
qualification. Publication and human/integrated review remain separate.

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

## Final allocated qualification

Slurm17586 COMPLETED0:0 in4m15s on closed source07 at
 ef4c670cf4e8f9f938c97040fa8503790b5c8df2. The submission requested1CPU,
4GiB and15min; scheduler accounting allocated2CPUs. Raw modular compiler SHA256
 a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3 and
bootstrap seed13003a8b9b1bf4bfc64ad8d13eb0795d722888fabef5bbb2913b4f7b83bb454b
were explicitly pinned. These are existing source-built compiler dependencies,
not a claim that this feature rebuilt the compiler.

Both engines passed checks of all three libraries and the fixture, native
execution and independent C++23 reference16/64/8. Both actual live-private-field
compile attempts returned exactly1 with E259 and no nonempty ELF. The unchanged
six-fixture pipeline gate passed both engines, independent references and its
four real output counterfactuals. Original unused-helper seed-check acceptance
remains documented and is not relabeled as refusal or a universal privacy proof.

The identical equality-assertion-free consumer passed against production source
and the independent oracle. An actually compiled and successfully executed
local-recompute implementation then failed that oracle with exact code2 and
message global normalization mismatch: factor. Generated consumer and complete
mutated stdlib catalogs were frozen before compilation and verified at the end;
compiler/import failures cannot satisfy this numerical mutant.

Full2117-entry workspace-to-worker source verification passed before and after
execution. Gate source catalog, generated mutation source, raw compilers and all
emitted executables passed final checks. Runtime receipts include process timing
and RSS but no paired performance improvement claim is made.

## Public entrypoints

- count_normalization_workspace and count_normalization_new create the checked
  reference owner from canonical CSR and transferred sample IDs; its factor/log
  and reference metadata accessors are read-only, and close consumes the owner.
- count_pipeline_workspace_global and count_pipeline_new_global request the
  worker variant; count_pipeline_prepare_global borrows the sealed owner and
  accepts the same explicit design/exposure/initial/contrast declarations.
- count_pipeline_normalization_scope returns -1 before preparation,0 for local
  normalization and1 for a global reference. Reference genes/used/excluded are
  unavailable before phase1. CountGeneReport has additive scope/reference fields;
  this change makes no stable external struct-ABI promise.

## Compact custody

final07-text.tar.gz is1656768bytes, SHA256
6db7ef4e979233f23c5eb6b08c476ab499aa3a8474f140b97768f6232854ebd4.
A fresh independent extraction verified408 internal hashes, all seven qualified
changed source/helper inputs, the original unused-helper fixture and acceptance
receipt,12 prior-fixture native exits, the specific successful-runtime numerical
mutant rejection and both2117-entry outer source verifications.

The packet retains changed source versions, complete input catalogs, preserved
failed development receipts, job scripts/accounting, actual output/reference
receipts and artifact hashes. Repeated full unchanged stdlib snapshots and ELF
payloads are excluded; the complete source reconstructs from the pinned base and
feature commits, and original executables remain on the worker. FILES.sha256
binds the final reviewed source and compact custody. Prepared final06 remains
unsubmitted; final07 was submitted once.

Tracking: [SIO-171](https://app.notion.com/p/4033f37f93718146a307f9f9616058c7).
