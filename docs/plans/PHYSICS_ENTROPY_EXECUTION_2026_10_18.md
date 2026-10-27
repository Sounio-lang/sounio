<!-- docs:meta
topic_id: repo.docs.plans.physics-entropy-execution-2026-10-18
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.physics-entropy-execution-2026-10-18
-->

# Isothermal entropy and standard-uncertainty rendering — 2026-10-18

## Scope and source

Continuation of the statistics, scientific visualization, mathematics, genomics
and physics stdlib sprint. Worktree: /workspace/sounio-sprint-stats-viz.
Original isolated baseline: f5b2c5850be9f3d3995719ffcade44a6d851658a.
Remote main first checked at a1844465373c1ae5a335de2d0a73819e5cec2230;
final recheck/fetch:2b41c82671995278554214398b7caac01b89e524.
New main commits change compiler lower.sio, KNOWN_LIMITATIONS and two GUM
fixtures, with no overlap in this entropy/render closure. The isolated source
has not been rebased onto that tip; acceptance uses the pinned compiler below.
Canonical checkout observed on fix/exp-product-mean at 9f24c028ba;
its local compiler build is excluded from acceptance. Changes are uncommitted.

## Implementation and scientific contract

- Checked physics::entropy::isothermal_entropy_independent returns nominal
  entropy change, standard uncertainty, log-volume ratio, status and model tags.
- Amount in mol; positive volumes in any common unit; entropy in J/K.
  Input uncertainties are nonnegative standard uncertainties.
- Explicit independent first-order sensitivity model, not coverage intervals.
  Equal nominal volumes have zero entropy change and may have positive uncertainty.
  Zero nominal amount with positive amount uncertainty retains its sensitivity.
- Near-equal volumes use a convergent log1p series; other volumes use shared
  checked logarithms without constructing a possibly overflowing volume ratio.
- Binary mantissa/exponent products avoid intermediate n*R and uv/V overflow.
  Subnormal reconstruction uses normal operands and one final subnormal rounding.
  A scaled norm avoids squaring large uncertainty contributions.
- Strict arithmetic status4 rejects any nonzero nominal or sensitivity contribution
  rounding entirely to zero, even if another contribution dominates the norm.
  This policy is narrower than accepting every representable aggregate norm.
- Legacy tuple API delegates to the checked kernel; unavailable is (0,-1).
  Other thermodynamic uncertainty formulas remain legacy.
- Vertical two-point raw-linear bars can carry interval kind4, standard uncertainty,
  with explicit bounds and level0. Setter validates atomically. SVG exports the
  semantic tag. Existing confidence/prediction/credible forests are unchanged.

[Contract](../../stdlib/physics/ENTROPY.md).

## Numerical evidence

Independent C++23, g++14.2 on t560, -std=c++23 -O2 -Wall -Wextra -pedantic.
31 cases use long-double log1p/log and hypot, with binary64 inputs and the same
binary64 shared SI gas-constant coefficient. Includes opposite volume extremes,
near equality, n*R overflow with finite answer, overflowing uv/V with finite
contribution, representable subnormals, strict underflow, overflow/domain statuses,
unit rescaling, sign symmetry and uncertainty scaling.

Tolerance remains relative1e-11 plus two minimum f64 steps for nonzero targets;
expected zeros and statuses are exact. Root reran the reference and byte-compared
its report and generated fixture with the candidate. No Python/Rust oracle used.

## Native acceptance

- Final candidate **15308**, COMPLETED0:0, **3:18**: **69/69 pass**,
  zero failures/skips/errors, retaining all previous64 fixtures.
- Equal-volume uncertainty-loss control **15309**, FAILED1:0, **3:19**:
 65pass +4 numerical runtime failures after successful compilation
 (reference, status, legacy and integrated figure); all previous64 pass.
- Prior-renderer control **15310**, FAILED1:0, **3:18**:
 67pass +2 missing-metadata stdout failures (generic and integrated figure);
 all previous64 pass.

| Run | Job | Source archive SHA256 |
|---|---:|---|
| stats-viz-entropy-20261018-v3 |15308|bbb01e3567ec617a0e2461d4b1d4857cceaaf0bbf8690164f655b2b2faadbb1f|
| stats-viz-entropy-control-zero-20261018-v3 |15309|459156d47dd56e43115f246fe2766d166a8bde167347c13b0ec2a1009498f970|
| stats-viz-entropy-control-svg-20261018-v3 |15310|c3caf7f9c5fd3f8b4ee38bf017f45293ae0cda2da75a4468610680a79c9b6b16|

Pinned committed Madaros v0.80 ELF SHA256:
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f.
Gzip SHA256:
feaaf7fe181f1ee3efd30e7fe23b1709937ea7e93b52e812276fb093b6ccf52a.
Forge/Slurm: GPUoff,2CPUs,8192MiB ceiling,5minute limit.
Source snapshot:9,125,220bytes compressed; no measured shrinkage claim.
C++23 executable SHA256:
44fb5a24c850d2e08e102001babe0f3aaddb17eb0fa9f521c10a27db3abb2e37.

New selected fixtures: physics_entropy_reference, physics_entropy_status,
physics_entropy_legacy, viz_standard_uncertainty, viz_isothermal_entropy.
The legacy fixture additionally imports SR, compares the adapter with the checked
kernel and invokes eight pre-existing thermodynamic examples. These smoke checks
do not certify their uncertainty formulas or every compiler resolution mode.

## Controls and preparation repairs

The numerical control retains the new checked shell but forces uncertainty to
zero when nominal volumes are equal. It is a targeted regression control, not
an untouched historical implementation. The renderer control restores the prior
accepted HTML/SVG module while retaining the new typed IR. Each final control
manifest differs from the final candidate at only its one controlled source.

Early candidate15291/v1 failed five cases. Candidate15294/v2 still contained
those preparation defects and is not acceptance. Diagnostic15299 prints failure identifiers and always finishes, so a green
harness result there is not numerical acceptance. Focused15305/v3 passed five
of six, identifying legacy examples. Legacy diagnostic15306 isolated those
examples; focused15307/v4 passed all six after repairs.

Repairs before final acceptance:
- Source extraction had dropped leading imports above the visual modules'
  comment header. Those imports were restored exactly from the accepted k-mer
  package. Final diffs retain only the new interval tag/helper and SVG metadata.
- An unused unresolved units import in thermo was removed. Existing mutation/
  division effects were declared on two legacy functions and their smoke helpers.
- Dynamic imported global R/KB initializers failed the gas-law/Boltzmann examples
  on the pinned backend. Consumers now call the same constant functions locally.
  Their formulas are otherwise preserved; heat capacity uses the same repair.
- Direct e-324 literals became zero in the pinned compiler. A constructed minimum
  subnormal passed the diagnostic. The core and generated subnormal input/reference
  literals now use normal numerators divided by 2^52; tolerances are unchanged.
  This is not a compiler literal-parser fix.

Three subagents provided the kernel/contract, C++ reference, figure and independent
source review. Root integrated, repaired, submitted native jobs and inspected the
artifacts. The larger sprint remains in progress.

## Rendering and resource boundaries

The 65-point nominal curve and five actual +/-u bars feed production Canvas/SVG.
The equal-volume label displays zero change and positive uncertainty. The fixture
checks scene counts, semantic bounds/tags and a known Canvas axis pixel. Existing
line tests cover the shared renderer; no new bar-specific Canvas pixel oracle is
claimed. Public IR fields are mutable; metadata is not tamper-proof certification.

SVG is parsed with xmllint and visually inspected through a QuickLook thumbnail.
The thumbnail crops the right edge and includes white padding; raw SVG is 960x520,
with axes/legend within bounds. The preview is not a captured Canvas framebuffer.

The entropy kernel uses scalar state, no heap or sample-sized buffers. Binary
rescaling loops are bounded by binary64 exponents. No measured execution-time,
RSS improvement, all-f64 correctly-rounded result, uncertainty coverage study,
whole-stdlib validation or SOTA superiority is claimed. Existing compact runtime
packaging is reused; only selected sources/evidence are copied into local custody.

## Custody and next work

/Users/demetriosagourakis/Documents/Sounio-stdlib-sprint-2026-10-16/entropy-2026-10-18.
Selected final sources/contracts, C++ report, native logs/metadata/JUnit/manifests,
source-archive hashes, SVG and preview are retained. The selected source manifest
and DELIVERY.sha256 document local custody. No commit/push/PR/merge or remote CI
acceptance is claimed.

Next implementation: mathematics information-theory logarithm and domain/support
contracts, then broader statistical resampling/calibration and scientific layouts.
Genomics count/I/O and physics covariance/invariant work remain in the same sprint.
