<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-171.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-171.readme
-->

# SIO-171: bounded CSR count pipeline qualification

The implementation composes canonical count normalization, genuine NB2 point
profiles, the existing declared Gaussian working log-dispersion prior, genuine
positive-prior penalized NB2 profiles, conditional plug-in Fisher contrasts and
direct tails, and full-family BH/BY. A retained report slot and streaming TSV
preserve failed gene IDs and explicit numerical availability. This is not a
DESeq2 parity or simulation-calibration claim. See
[the API and statistical contract](../../../stdlib/genomics/COUNT_PIPELINE.md).

The base is SIO-170 `7c4f9815e3a4b6ca1af94490e64e066481c28ad2` (PR3116),
composed by feature merge with SIO-169
`1d3058b45dda56c309855ba50acb0f3ca529bfc3` (PR3117), including its SIO-168/162
ancestry. Incorporated remote main is
`7ce5eddeef78c76c7e744150ed9c169c53a3d3b0`. No main merge or release is implied.
SIO-167's additive Gamma-prefix helper is not imported and is not a dependency.

## Compiler and actual native qualification

All heavy work below ran in allocated UID1000 Slurm jobs on cpu-ops, one requested
CPU, 4GiB memory, stack limit524288KiB. Slurm may report two allocated CPU slots
under the node configuration; this is not a two-thread native implementation.
The retained source-built modular compiler is
`compiler_ci_c21/madaros.elf.ready`, SHA256
`a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3`,
built from c21 source. The committed bootstrap seed is SHA256
`8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc`.
Compiler source differs from the incorporated main only in unrelated workflow
and changelog changes. Each check/compile ran from its immutable source root;
the raw seed's ignored stdlib-path environment variable cannot select a wrong
source tree here. No fresh compiler rebuild is claimed by these jobs.

Slurm17421, immutable `SIO171/source09`, completed0:0 in2m33s. The focused gate
strict-checked, compiled and executed **both** compilers for:

- G8×N12 ML case: four numerical prior-training rows, zero prior variance,
  explicit point-prior refits and one all-zero failed gene.
- G12×N16 ML case: eleven numerical training rows, positive working prior and
  actual penalized NB2 refits; the all-zero gene remains in the family.
- Three refusal chains: missing byte-ID join and successful explicit retry;
  negative join budget; constructor payload budget minus one; undeclared model
  with available fixed-alpha coefficients but unavailable test; Gamma-tail
  budget exhaustion after a successful fit; point-only mode with unavailable
  curvature/prior and full-family adjusted outputs1. Invalid seed preflight
  leaves the actual point cursor unchanged.
- Active-prefix shrinkage and QR controls, plus the original shrinkage fixture.

The final independent binary128 reference source is SHA256
`c61c6055421e99697723466dceecce4ec13f0c962893267240198c103cc182d3`.
It solves heterogeneous-offset group score equations and the dispersion profile
independently of production IRLS/QR. It checks actual profile objectives,
curvature, working prior, actual penalized refit, conditional Fisher contrasts,
erfc tails, complete-family BH/BY, complete byte IDs and TSV availability.
Nine positive-prior refits differ materially from merely substituting the
Gaussian closed-form proxy. Positive p/BH/BY checks are relative-only with
strict [0,1] bounds; zeros cannot pass as tiny positive probabilities. The
fixture's twelve binary64-versus-binary128 amplitude-to-integer conversions were
independently audited without changing the original data. Tolerances qualify
these fixtures; they are not universal error bounds or proof of unimodality.

The focused gate also independently checked the new active-prefix results and
original shrinkage outputs, and rejected four altered actual-native reports
with exact code2 and the numerical parser diagnostic. The separate reference
receipts retain eleven broader parser/scientific counterfactuals and four
zero/negative-probability controls. Missing outputs, false availability, wrong
family sizes, replacing the real refit with a Gaussian proxy, and malformed IDs
cannot pass through compiler/import failures.

Root's independent prefix qualification17393 covered G8/cap13 and G513/cap1024,
1575 values per engine, NaNs in every unused input tail, output-tail sentinels,
short/over-capacity/old-exact-shape/active-NaN atomic refusals and a direct3×2 QR
case. Original full-shape outputs were byte-identical to before17384 for each
engine. The production prefix hashes are
`1ccc7dbdfb335811e09613c97dbe5bbde72b2ce2f82a9aec027c27bd92467197`
and `0271cc4d1b0c139ba7de659a61ef862feab83ec915ed21e8750321fce2944769`.

## Counter bounds and high-outlier follow-up

Slurm17441/source15 completed0:0 in3m27s with the full dual-engine gate plus
an additional G12×N16 fixture that changes only outlier_sigma from2 to1. Two
high-dispersion rows use the unshrunk point alpha and refit route3; nine retain
the genuine positive-prior profile route. The independent reference extension
passes sigma1 to its existing Gaussian model, with no formula or tolerance
change. Its source SHA256 is
`723bd047dace3ffcf4679cf74df335d72950e95ccaaae5d6befaa9f85db8656b`.
All original modes retain their original flags and markers.

Slurm17443/source16 completed0:0 in3m28s, repeating the six-fixture gate on both
engines and checking constructor refusal before aggregate counter overflow.
The per-gene profile cap must be positive and G*cap must fit signed i64; this
is a representability bound, not an arbitrary optimizer cap. An explicit g>0
branch precedes MAX/g. A valid zero-row CSR and valid empty gene-ID owner are
refused as pipeline shape1 and closed safely. A positive family with capMAX
is refused as controls3. The pipeline source qualified in17443 is SHA256
`3c7c8594f4a22d48f02ef73c59e10fb54cf9e4f29f89ac3a4de3fd8a0ea97c64`.

During17443, an unrelated C++ pilot-verifier source was added to source16. No
selected gate input changed and that file was neither imported nor compiled by
the gate; its before/after input manifest verifies unchanged. This selected-input
qualification is retained honestly, without calling the entire source16 catalog
immutable. Final publication will use a fresh closed source catalog.

Slurm17452/source19 completed0:0 in22s, strict checking and executing both engines
with four scalar readers for original counts, sample byte IDs and the aligned
design. The fixture checks count-column order against reversed metadata order.
Its complete scientific stdout is byte-identical to the previous corresponding
engine result in17443. Source19 was closed before submission; no files were
added during this run. `reader19-text.tar.gz` retains these receipts.

## Scale status and honest limits

The full20k×512 CSR holder allocates10,506,992 bytes of new pipeline buffer
payload, excluding the transferred CSR/IDs. Two actual256-row point-only pilots
on this full holder completed: unseeded17404 native18:19.18, peak597432KiB and
8192 objective evaluations; named canonical seeded/warm-start17412 native10:59.32,
peak592572KiB and8448 evaluations. These timings include full normalization and
holder construction. Slurm17446 independently checked all actual256 rows against
binary128 analytic normalized group coefficients and exact Poisson alpha0;
maximum differences between the two printed coefficient streams were0. Its C++
calculation gives an elapsed ratio0.59982896340908677 and RSS ratio0.991865182983168
(seed/unseeded). There is no representative memory improvement. These are two
single observations, not a universal speed claim or binary64-bit equality proof.
Neither256-row result closes the20k actual-fit acceptance condition.

The scaling program uses **one shared executable** and a separately hashed
`shard_start.txt` in each worker directory. Each owner covers1000 original gene
rows with global byte IDs and512 sample columns; normalization is explicitly
per shard. No fitted row is copied to another gene. A diagnostic actual one-row
file-input run17430 reports point_done1/phase1, not a completed1000- or20k-gene
family. Shared full-shard ELF SHA256 is
`95ab65d0448827040ce935d578dac37c05e32cd20bcf7d06bde91c4bfc01c29e`;
scaling source SHA256 is
`567e87e48278061d1d7d24ccdc93f3266d05b949057444949307a42cf44a0c86`.

**Pending:** actual1000-row pilot17432, then complete20k row coverage under the
root-coordinated maximum8 concurrent jobs and20GiB host headroom. The independent
scale checker requires all20 actual reports and the exact20k ID/range union;
its separately labeled partial mode requires a complete1000-row shard. Its
synthetic parser controls17424 are preparation only and certify no native fits.
The scale workload is point-only ML on a declared repeated-within-group count
fixture. It does not qualify a full20k curvature, empirical-prior, FDR, bootstrap
or RNA-seq-calibration analysis. These pending conditions remain open.

## Custody and preserved failures

`qualification09-text.tar.gz` retains final raw checks, compiler/native output,
C++ results, process timing/RSS, source/ELF/compiler hashes, the submitted job and
counterfactual diagnostics; it excludes executables. `prefix-text.tar.gz` retains
root17392/17393/17394 and before/after compatibility receipts.
`independent-text.tar.gz` retains independent reference development, hardening,
parser controls and explicitly synthetic scale-parser preparation.
`development-text.tar.gz` preserves earlier failed attempts and their actual
inputs, including invalid fixture signatures, constructor linearity refusals,
report string-type diagnostics, transient missing-source transport refusal,
and the command-line benchmark's early refusal. A launcher sed error passed a
spurious0 compiler option in17425;17426 then established the original argument
transport refused before scientific work. The benchmark now uses qualified
file input; the earlier outcomes have not been relabeled numerical passes.

The source packets and raw receipts remain under the worker root
`/scratch/sounio-stdlib-inference-scale-20261024/SIO171`. No historical artifacts
were deleted and no shared quota was changed. `followup-text.tar.gz` retains17441/17443, the off-tree outlier witness17438,
and the actual256 numerical/metrics verifier17446 plus its exact numerical
refusal control. Text archives preserve raw compiler whitespace and scientific outputs. `FILES.sha256` covers local
custody and the relevant maintained source files.
